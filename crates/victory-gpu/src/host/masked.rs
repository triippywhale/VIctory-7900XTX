//! A page with more changed runs than the merge's cap leaves goes up by
//! byte masks instead of the flights being settled (`VICTORY_MASKED_PATCH=1`;
//! off by default; `alt`: twenty seconds with, twenty without, in turn).
//!
//! The title's bytes that changed under flights' shader writes are put in
//! beside those writes (`Shadow::merge_under_writes`). Runs of whole words
//! up to 256 bytes go up as pairs of one dispatch ([`super::send`]); every
//! other run is a copy command of its own, and at most
//! [`super::merge_runs`] of those are taken (`VICTORY_MERGE_RUNS`,
//! `VICTORY_SETTLE_ROOM`). Past that the merge gave up and the flights
//! were settled: the submitting thread waits for the card.
//!
//! With the switch such a page is taken all the same, by the bytes that
//! differ (not by words), and its runs are noted ([`Shadow::card_patched`]
//! takes the notes). The runs are sent as every changed run is: they are
//! in the copy's changed offsets, are noted as sent, and their pieces are
//! hashed again. Only the form differs: where the submission's sends are
//! taken ([`super::send::take`]) a changed range that holds such a run
//! goes as words of offset, mask and value ([`words`]), and one dispatch a
//! copy stores `(old & !mask) | (value & mask)` on the card's side
//! ([`MASKED`]), behind the flights in the queue. No byte but the changed
//! ones is stored, so what a flight's shader wrote beside them stays, with
//! no cap on their number.
//!
//! The note says how a range goes up, never whether: without it (no
//! pipeline, `VICTORY_SEND_OWN=0`, no memory) the same ranges are copies.
//!
//! With pieces left on the card (`crate::left`, `VICTORY_SKIP_WC_BACK`):
//! the merge has the pieces of other copies fetched before it reads title
//! memory (`crate::left::before_take`), as without the switch. A piece of
//! the copy's own stays on the card: the title's bytes there are older
//! than the card's, and they are equal to the copy's CPU side, which the
//! page is compared with. So they are in no run, and in no mask: the
//! masks hold the bytes the title wrote since, and what the card wrote
//! around them stays, by the byte where the pairs keep it by the word.
//!
//! After fxceless 3221ac4 (on by default there since 0e006f0: its run 362,
//! waiting for the card 21 -> 2 ms, the usual frame 10 to 15 ms shorter)
//! and PalmVI 3348e28d. The source kept the words of each patch from the
//! merge on; here they are made of the CPU side when the submission's
//! sends are taken, as its other sends are (`VICTORY_SEND_OWN`). It has no
//! check of its own, nor has this: `VICTORY_UPCOPY_CHECK=1`
//! ([`super::upcheck`]) compares the card's side with the CPU's.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};

/// `VICTORY_MASKED_PATCH`, read once here: 0 off (unset, or anything
/// else), 1 on (`1`), 2 in turn (`alt`). With diagnostics, 3: the switch
/// is one of `VICTORY_AB`'s.
pub(crate) fn mode() -> u8 {
    static MODE: OnceLock<u8> = OnceLock::new();
    *MODE.get_or_init(|| {
        #[cfg(feature = "diagnostics")]
        if crate::ab::bit("VICTORY_MASKED_PATCH").is_some() {
            return 3;
        }
        mode_of(std::env::var("VICTORY_MASKED_PATCH").ok().as_deref())
    })
}

fn mode_of(asked: Option<&str>) -> u8 {
    match asked.map(str::trim) {
        Some("1") => 1,
        Some("alt") => 2,
        _ => 0,
    }
}

/// Set once the pass's pipeline is made (`super::send`): until then no
/// page is taken past the cap.
pub(super) static READY: AtomicBool = AtomicBool::new(false);

/// Whether a page past the merge's cap is taken now, to go up by mask.
pub(crate) fn now() -> bool {
    if !READY.load(Ordering::Relaxed) {
        return false;
    }
    match mode() {
        0 => false,
        1 => true,
        2 => {
            static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
            static LAST: AtomicU8 = AtomicU8::new(2);
            let on = (EPOCH
                .get_or_init(std::time::Instant::now)
                .elapsed()
                .as_secs()
                / 20)
                % 2
                == 1;
            if LAST.swap(u8::from(on), Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "title bytes under flights' shader writes past the merge's cap, from now (VICTORY_MASKED_PATCH=alt): {}",
                    if on {
                        "ON, sent up by mask"
                    } else {
                        "off, the flights are settled"
                    }
                );
            }
            on
        }
        _ => {
            #[cfg(feature = "diagnostics")]
            if let Some(on) = crate::ab::pick("VICTORY_MASKED_PATCH") {
                return on;
            }
            false
        }
    }
}

/// Since the last frame line: pages taken past the cap, their changed
/// runs, and those runs' bytes.
static PATCHED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Since the last frame line: copies whose merge gave up for more changed
/// runs than the cap leaves.
static OVER: AtomicU64 = AtomicU64::new(0);

/// Since the last frame line: merges in which a page was taken past the
/// cap and no settle followed (each was a settle before), and merges that
/// still gave up over the cap (each a settle, or a wait for the flights
/// that declared the range).
static MERGES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Since the last frame line: words stored by mask, and their dispatches.
pub(super) static SENT: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// A page was taken past the cap: `runs` changed runs of `bytes` bytes.
pub(crate) fn took(runs: usize, bytes: usize) {
    PATCHED[0].fetch_add(1, Ordering::Relaxed);
    PATCHED[1].fetch_add(runs as u64, Ordering::Relaxed);
    PATCHED[2].fetch_add(bytes as u64, Ordering::Relaxed);
}

/// A copy's merge gave up over the cap.
pub(crate) fn gave_up() {
    OVER.fetch_add(1, Ordering::Relaxed);
}

/// Before a merge ([`super::stale_under_flights`]): what [`merged`] takes.
pub(super) fn marks() -> [u64; 2] {
    [
        PATCHED[0].load(Ordering::Relaxed),
        OVER.load(Ordering::Relaxed),
    ]
}

/// After the merge that began at `was`: `settles` if it leaves ranges the
/// flights are settled for.
pub(super) fn merged(was: [u64; 2], settles: bool) {
    let [patched, over] = marks();
    if patched != was[0] && !settles {
        MERGES[0].fetch_add(1, Ordering::Relaxed);
    }
    if over != was[1] {
        MERGES[1].fetch_add(1, Ordering::Relaxed);
    }
}

/// The frame line's figures of the merge's cap. Said with the switch off
/// too: the merges that gave up over the cap are what it would take.
pub(super) fn line(frames: f64) {
    let [pages, runs, bytes] =
        [0, 1, 2].map(|i| PATCHED[i].swap(0, Ordering::Relaxed) as f64 / frames);
    let [avoided, still] = [0, 1].map(|i| MERGES[i].swap(0, Ordering::Relaxed) as f64 / frames);
    let [words, dispatches] = [0, 1].map(|i| SENT[i].swap(0, Ordering::Relaxed) as f64 / frames);
    let over = OVER.swap(0, Ordering::Relaxed) as f64 / frames;
    tracing::info!(
        "pages past the merge's cap per frame (VICTORY_MASKED_PATCH {}; the cap, VICTORY_MERGE_RUNS, is {}): {pages:.1} pages patched by \
         mask, {:.3} MB in {runs:.0} changed runs, {words:.0} words stored by mask in {dispatches:.1} dispatches, {avoided:.2} settles \
         avoided; {still:.2} settles still for more changed runs than the cap leaves ({over:.1} copies gave up)",
        match (mode(), READY.load(Ordering::Relaxed)) {
            (0, _) => "off",
            (_, false) => "asked, no pipeline: off",
            (1, _) => "on",
            (2, _) => "alt",
            _ => "by VICTORY_AB",
        },
        super::merge_runs(),
        bytes / (1 << 20) as f64,
    );
}

/// One dword of a copy stored by mask: its offset in the copy (a multiple
/// of 4), the bytes it sets (a mask of whole bytes), and their values
/// (other bits 0).
pub(crate) type Word = [u32; 3];

/// A changed range up to this many bytes goes by mask if a page past the
/// cap put bytes in it ([`by_mask`]); a longer one is a copy (it holds a
/// refresh's range that touches such a run: there are few so long).
const MOST: usize = 64 << 10;

/// Whether the changed range `[a, b)` of a copy goes by mask: `patched`
/// are the runs pages past the cap put in, in order and apart. Every byte
/// of the range changed (changed ranges are joined only where they touch),
/// so all of it may be stored.
pub(super) fn by_mask(a: usize, b: usize, patched: &[(usize, usize)]) -> bool {
    if b - a > MOST {
        return false;
    }
    let i = patched.partition_point(|p| p.1 <= a);
    patched.get(i).is_some_and(|p| p.0 < b)
}

/// The words that store `bytes` at offset `at` of a copy, appended to
/// `out`: one per dword touched. Called with ranges in order: a dword two
/// of them share (the last of `out`) is one word, so that no two threads
/// of the pass store the same dword.
pub(super) fn words(at: usize, bytes: &[u8], out: &mut Vec<Word>) {
    for (k, &byte) in bytes.iter().enumerate() {
        let offset = at + k;
        let word = (offset & !3) as u32;
        let lane = ((offset & 3) * 8) as u32;
        let (mask, value) = (0xffu32 << lane, u32::from(byte) << lane);
        match out.last_mut() {
            Some(w) if w[0] == word => {
                w[1] |= mask;
                w[2] = (w[2] & !mask) | value;
            }
            _ => out.push([word, mask, value]),
        }
    }
}

/// What the pass does to a dword ([`MASKED`]'s store, for the tests).
#[cfg(test)]
fn stored(old: u32, w: Word) -> u32 {
    (old & !w[1]) | (w[2] & w[1])
}

/// The pass: `count` words of 12 bytes from byte `at` of `words`, each
/// stored into its dword of `card` under its mask. The root signature is
/// the pairs' (`super::send`), so that both are bound once.
pub(super) const MASKED: &str = r#"
#define RS "RootConstants(num32BitConstants=2, b0), SRV(t0), UAV(u0)"
cbuffer Part : register(b0) { uint at; uint count; };
ByteAddressBuffer words : register(t0);
RWByteAddressBuffer card : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= count) return;
    uint3 w = words.Load3(at + id.x * 12u);
    card.Store(w.x, (card.Load(w.x) & ~w.y) | (w.z & w.y));
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// The runs of bytes where `now` differs from `was`: (offset, bytes).
    fn changed(now: &[u8], was: &[u8]) -> Vec<(usize, usize)> {
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for i in (0..now.len()).filter(|&i| now[i] != was[i]) {
            match runs.last_mut() {
                Some(last) if last.0 + last.1 == i => last.1 += 1,
                _ => runs.push((i, 1)),
            }
        }
        runs
    }

    /// The pass stores the title's changed bytes and nothing else: every
    /// byte a "flight" wrote on the card between them stays, however many
    /// runs there are and wherever they fall in their dwords.
    /// (After fxceless 3221ac4, `the_patch_stores_only_the_changed_bytes`.)
    #[test]
    fn the_masks_store_only_the_changed_bytes() {
        let n = 4096usize;
        let at = 3 * 65536 + 4; // the page's offset in the copy (a multiple of 4)
        let old: Vec<u8> = (0..n).map(|i| (i * 7 + 1) as u8).collect();
        // The title changes every third byte and a long run.
        let mut title = old.clone();
        for i in (0..n).step_by(3) {
            title[i] ^= 0x5a;
        }
        for b in &mut title[1000..1100] {
            *b = !*b;
        }
        let runs = changed(&title, &old);
        assert!(runs.len() > 1000);
        // The card's side after the flights: the old bytes, but a flight
        // stored into bytes the title did not change.
        let mut card = old.clone();
        let mut flight = vec![false; n];
        for i in (1..n).step_by(5) {
            if title[i] == old[i] {
                card[i] = 0xee;
                flight[i] = true;
            }
        }
        // The CPU's side holds the title's bytes by then: the words are
        // made of it, range by range in order.
        let mut out: Vec<Word> = Vec::new();
        for &(r, m) in &runs {
            words(at + r, &title[r..r + m], &mut out);
        }
        // No dword twice: the pass's threads store apart.
        assert!(out.windows(2).all(|w| w[0][0] < w[1][0]));
        for w in &out {
            assert_eq!(w[0] % 4, 0);
            let k = w[0] as usize - at;
            let mut dword = [0u8; 4];
            dword.copy_from_slice(&card[k..k + 4]);
            let new = stored(u32::from_le_bytes(dword), *w).to_le_bytes();
            card[k..k + 4].copy_from_slice(&new);
        }
        for i in 0..n {
            let want = if flight[i] { 0xee } else { title[i] };
            assert_eq!(card[i], want, "byte {i}");
        }
    }

    /// Ranges that share a dword make one word of it, the later bytes
    /// beside the earlier ones.
    /// (After fxceless 3221ac4, `words_of_one_dword_fold_in_order`.)
    #[test]
    fn ranges_of_one_dword_make_one_word() {
        let bytes = [1u8, 2, 3, 4, 5, 6];
        let mut out = Vec::new();
        words(1, &bytes[1..3], &mut out);
        words(5, &bytes[5..6], &mut out);
        assert_eq!(
            out,
            vec![[0, 0x00ff_ff00, 0x0003_0200], [4, 0x0000_ff00, 0x0000_0600]]
        );
        let mut out = Vec::new();
        words(8, &[0x11], &mut out);
        words(10, &[0x33, 0x44], &mut out);
        assert_eq!(out, vec![[8, 0xffff_00ff, 0x4433_0011]]);
        assert_eq!(stored(0xaabb_ccdd, out[0]), 0x4433_cc11);
    }

    /// A changed range goes by mask where a page past the cap put bytes in
    /// it, unless it is long; the others go as before.
    #[test]
    fn a_range_goes_by_mask_where_a_patched_run_lies_in_it() {
        let patched = [(100, 103), (200, 260), (1 << 20, (1 << 20) + 8)];
        assert!(by_mask(100, 103, &patched));
        // Joined with what touched it.
        assert!(by_mask(90, 110, &patched));
        assert!(by_mask(259, 300, &patched));
        // Beside the runs, and touching one without holding a byte of it.
        assert!(!by_mask(103, 200, &patched));
        assert!(!by_mask(0, 100, &patched));
        assert!(!by_mask(260, 400, &patched));
        assert!(!by_mask(50, 60, &[]));
        // Long: a copy.
        assert!(!by_mask((1 << 20) - MOST, (1 << 20) + 8, &patched));
        assert!(by_mask((1 << 20) - MOST + 8, (1 << 20) + 8, &patched));
    }

    /// The switch's values: `1`, `alt`, anything else off.
    #[test]
    fn the_switch_is_off_unless_asked() {
        assert_eq!(mode_of(None), 0);
        assert_eq!(mode_of(Some("0")), 0);
        assert_eq!(mode_of(Some("yes")), 0);
        assert_eq!(mode_of(Some(" 1 ")), 1);
        assert_eq!(mode_of(Some("alt")), 2);
    }

    /// The pass's source compiles as the pairs' does, with the same root
    /// signature inside it (no device needed).
    /// (After fxceless 3221ac4, `the_masked_patch_compiles`.)
    #[test]
    fn the_masked_pass_compiles() {
        let code = super::super::compile_now(
            MASKED,
            windows::core::s!("main"),
            windows::core::s!("cs_5_1"),
        )
        .expect("MASKED compiles");
        assert!(code.len() > 100);
    }
}
