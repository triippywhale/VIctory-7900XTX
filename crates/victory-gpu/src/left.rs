//! What shaders wrote to card copies of write-combining title memory, left
//! on the card (`VICTORY_SKIP_WC_BACK=2`, off by default; after PalmVI
//! 06edf544f), and its audit (`VICTORY_CENSUS=1`).
//!
//! The title reserves some of its memory write-combining: memory its CPU
//! writes and does not read. What shaders write to a copy of such memory in
//! the card's memory (`crate::shadow::cards`) was fetched and merged into
//! title memory after every flight (98 MB a frame in run record-1), only to
//! be read by nobody on the CPU. With the switch, a piece the compare found
//! changed is left on the card instead, when no other copy of ours holds
//! any of its bytes ([`skip_back`]): the copy that wrote it keeps what it
//! wrote, title memory keeps what it had.
//!
//! What it must keep: title memory there is not what the card wrote, so any
//! reader of those title bytes other than the copy that holds them has the
//! piece fetched and merged first, as copy-back would have ([`before_take`],
//! `crate::host::leftq::fetch`), and that piece is never left again. The
//! readers, each of which calls [`before_take`] before it reads:
//!
//! - every refresh of a copy (`Shadow::update`: its stale pages, its taken
//!   pages, whole copies) and the title's bytes put in under shader writes
//!   (`Shadow::merge_under_writes`);
//! - a submission's steps: the title's buffer copies (source and
//!   destination), fills of upload memory, texture uploads, copies at their
//!   place, work that reads title memory, and our work that may read or
//!   write it anywhere (then every piece is fetched);
//! - texture stagings (`crate::texture`: refreshes, fingerprints, the look
//!   at whether a texture changed, any read of a texture's memory);
//! - the arenas of title memory by address (`crate::memory`).
//!
//! Those that keep title bytes and take them again only where title memory
//! was written (copies, textures, arenas) are told: a piece left is noted
//! in the write journal as a landing is (`crate::watch::landed_beside`),
//! so that their next refresh reads it, and so fetches it.
//!
//! What the copy that holds the bytes does itself: its refreshes there take
//! only the dwords where title memory differs from its CPU side, which are
//! the title's new ones, and send them up; what the card wrote around them
//! stays ([`own_left_in`]). (Where the title stores again, after the card's
//! write, the very value the CPU side held before it, that store is not
//! seen: the card's value stays, as in PalmVI.)
//!
//! Not left, ever: memory the card reads where the title has it (in place,
//! `crate::inplace`; renderer port 1's 32-bit graphics memory,
//! `crate::port`), pages an arena of `crate::memory` holds, copies with
//! stores outside their declared ranges (`Shadow::keep_stores`), and a page
//! the title shows at another address too (`VICTORY_SKIP_WC_BACK=3` is the
//! form that does those). A piece is fetched by the original copy's bytes
//! only: memory the title frees or maps anew forgets what was left there.
//!
//! `VICTORY_SKIP_WC_BACK=1` (every piece of such memory, no fetch) is not
//! in this build: it gave a black world and lost room lighting on an RTX
//! 5070 Ti in PalmVI.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use crate::shadow::Shadow;

/// The console's protection bit of write-combined memory (`xmem`'s
/// `note_protection`).
const WRITE_COMBINED: u32 = 0x400;
/// Pieces are kept by the 64 KiB of title memory they lie in (a card copy's
/// pieces lie in one each: `crate::shadow::card_piece_of`).
const PIECE: u64 = 64 << 10;

/// `VICTORY_SKIP_WC_BACK`: 2 leaves pieces on the card (see the file);
/// unset or 0: off. Other values are not modes of this build: off.
pub(crate) fn mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        let asked = std::env::var("VICTORY_SKIP_WC_BACK").unwrap_or_default();
        match asked.trim() {
            "" | "0" => 0,
            "2" => {
                tracing::info!(
                    "VICTORY_SKIP_WC_BACK=2: what shaders write to card copies of write-combining title memory stays on the card where no other copy holds it, fetched before anything else reads it"
                );
                2
            }
            other => {
                tracing::warn!("VICTORY_SKIP_WC_BACK={other} is not a mode of this build (2): off");
                0
            }
        }
    })
}

/// Whether a mode is on.
pub(crate) fn on() -> bool {
    mode() != 0
}

/// `VICTORY_CENSUS=1`: the audit (title bytes of pieces left on the card
/// read by anything but the copy that holds them, which must be none), on
/// the frame line.
pub(crate) fn census() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| on() && std::env::var_os("VICTORY_CENSUS").is_some_and(|v| v == "1"))
}

/// The title's write-combining reservations: start, bytes.
static WRITE_COMBINING: RwLock<BTreeMap<usize, usize>> = RwLock::new(BTreeMap::new());

/// The runtime reserved `[at, at + len)` for the title (`XMemVirtualAlloc`
/// at an address of its choice), with the title's own console `protect`
/// (before the runtime cleans it). Kept only with a mode on.
pub fn title_memory_reserved(at: usize, len: usize, protect: u32) {
    if protect & WRITE_COMBINED == 0 || at == 0 || len == 0 || !on() {
        return;
    }
    WRITE_COMBINING
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .insert(at, len);
}

/// Whether all of `[at, end)` is in a reservation the title asked
/// write-combining.
fn write_combining(at: u64, end: u64) -> bool {
    let map = WRITE_COMBINING.read().unwrap_or_else(|e| e.into_inner());
    map.range(..=at as usize)
        .next_back()
        .is_some_and(|(&base, &len)| (base + len) as u64 >= end)
}

/// A piece left on the card: the copy whose card side holds what was
/// written, and the offsets in it (in order, apart).
struct Left {
    writer: Arc<Shadow>,
    runs: Vec<(usize, usize)>,
    /// For a view of the title's physical pages: the section and offset of
    /// the 64 KiB page.
    page: Option<(usize, u64)>,
}

/// The pieces left, by the title address of their 64 KiB, and of those in
/// view memory by the section and offset of the page (a view of it the
/// title maps later at another address reads the same bytes); and the
/// 64 KiB pieces taken once while left, which are fetched as before from
/// then on.
struct Book {
    left: BTreeMap<u64, Left>,
    by_page: BTreeMap<(usize, u64), u64>,
    never: BTreeSet<u64>,
}

impl Book {
    /// Takes the piece at `key` out.
    fn remove(&mut self, key: u64) -> Option<Left> {
        let left = self.left.remove(&key)?;
        LEFT_N.fetch_sub(1, Ordering::SeqCst);
        if let Some(page) = left.page {
            self.by_page.remove(&page);
        }
        Some(left)
    }

    /// The pieces that hold title bytes of `[from, to)`, at those addresses
    /// or, for view memory, at another address of the same pages
    /// (`pages`: [`crate::inplace::page_keys`] of the range).
    fn touching(&self, from: u64, to: u64, pages: &[(usize, (usize, u64))]) -> Vec<u64> {
        let mut keys: Vec<u64> = self
            .left
            .range(from / PIECE * PIECE..to)
            .filter(|(_, left)| {
                let (a, b) = title_span(left);
                a < to && b > from
            })
            .map(|(&key, _)| key)
            .collect();
        if !self.by_page.is_empty() {
            keys.extend(
                pages
                    .iter()
                    .filter_map(|(_, page)| self.by_page.get(page).copied()),
            );
            keys.sort_unstable();
            keys.dedup();
        }
        keys
    }
}

static BOOK: RwLock<Book> = RwLock::new(Book {
    left: BTreeMap::new(),
    by_page: BTreeMap::new(),
    never: BTreeSet::new(),
});

/// Pieces in [`BOOK`], for the look every reader makes; and fetches under
/// way (their pieces are out of the book, their bytes not in title memory
/// yet).
static LEFT_N: AtomicU64 = AtomicU64::new(0);
static FETCHING: AtomicU64 = AtomicU64::new(0);

/// Held by every fetch, and by a refresh of a copy that holds pieces left
/// in its range: a fetch merges into the copy's CPU side, which such a
/// refresh writes.
static GATE: Mutex<()> = Mutex::new(());

/// Since the last frame line: pieces left and their bytes; pieces fetched
/// for a reader, their bytes, the fetches; pieces left again by their
/// copy's next flight (counted in the first two as well).
static STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// With [`census`], since the last frame line: title bytes of pieces left
/// read by other than the copy holding them, by a copy's refresh and by
/// anything else.
static STALE: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

thread_local! {
    /// The copy this thread refreshes ([`Taking`]): its own pieces are not
    /// stale for it.
    static TAKER: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// While alive, this thread reads title memory for copy `id` (for the audit).
pub(crate) struct Taking(u64);

impl Taking {
    pub(crate) fn new(id: u64) -> Taking {
        Taking(TAKER.replace(id))
    }
}

impl Drop for Taking {
    fn drop(&mut self) {
        TAKER.set(self.0);
    }
}

/// `runs` (in order, apart) with `[lo, hi)` added.
fn add_run(runs: &mut Vec<(usize, usize)>, lo: usize, hi: usize) {
    if lo >= hi {
        return;
    }
    runs.push((lo, hi));
    runs.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(runs.len());
    for &(a, b) in runs.iter() {
        match out.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(b),
            _ => out.push((a, b)),
        }
    }
    *runs = out;
}

/// `runs` (in order, apart) with `[lo, hi)` taken out.
fn cut_run(runs: &mut Vec<(usize, usize)>, lo: usize, hi: usize) {
    let mut out = Vec::with_capacity(runs.len() + 1);
    for &(a, b) in runs.iter() {
        if b <= lo || a >= hi {
            out.push((a, b));
            continue;
        }
        if a < lo {
            out.push((a, lo));
        }
        if b > hi {
            out.push((hi, b));
        }
    }
    *runs = out;
}

/// The title memory a piece's runs hold.
fn title_span(left: &Left) -> (u64, u64) {
    let start = left.writer.span().0;
    let first = left.runs.first().map_or(0, |r| r.0);
    let last = left.runs.last().map_or(0, |r| r.1);
    (start + first as u64, start + last as u64)
}

/// With a mode on, as the pieces a flight's shaders wrote are about to be
/// fetched: whether offsets `[lo, hi)` of `shadow`, a copy in the card's
/// memory, are left there instead. Then title memory keeps what it had and
/// the piece is noted (and told to the write journal, so that whoever keeps
/// those title bytes reads them again, and so fetches them first). If not,
/// it is fetched as before, with what was left of it earlier.
pub(crate) fn skip_back(shadow: &Arc<Shadow>, lo: usize, hi: usize) -> bool {
    if !on() || lo >= hi || !shadow.carded() {
        return false;
    }
    let (start, end) = (shadow.span().0 + lo as u64, shadow.span().0 + hi as u64);
    let key = start / PIECE * PIECE;
    if end > key + PIECE {
        return false;
    }
    if !may_leave(shadow, start, end) {
        taken_back(shadow, key, lo, hi);
        return false;
    }
    let page = crate::inplace::page_keys(key as usize, 1)
        .first()
        .filter(|p| p.0 == key as usize)
        .map(|p| p.1);
    {
        let mut book = BOOK.write().unwrap_or_else(|e| e.into_inner());
        // Taken once while left; or another copy left its bytes here, and
        // this one is a second holder after all: fetched as before.
        let other = book
            .left
            .get(&key)
            .is_some_and(|left| left.writer.id() != shadow.id());
        if other || book.never.contains(&key) {
            drop(book);
            taken_back(shadow, key, lo, hi);
            return false;
        }
        match book.left.get_mut(&key) {
            Some(left) => {
                add_run(&mut left.runs, lo, hi);
                STATS[5].fetch_add(1, Ordering::Relaxed);
            }
            None => {
                let page = page.filter(|page| !book.by_page.contains_key(page));
                if let Some(page) = page {
                    book.by_page.insert(page, key);
                }
                book.left.insert(
                    key,
                    Left {
                        writer: shadow.clone(),
                        runs: vec![(lo, hi)],
                        page,
                    },
                );
                LEFT_N.fetch_add(1, Ordering::SeqCst);
            }
        }
    }
    STATS[0].fetch_add(1, Ordering::Relaxed);
    STATS[1].fetch_add(end - start, Ordering::Relaxed);
    // After the note: whoever reads these pages from now on sees the note
    // first. Its 4 KiB pages are written as far as those who keep title
    // bytes can tell, as after a landing.
    let pages: Vec<usize> = (start as usize / 4096..(end as usize).div_ceil(4096)).collect();
    let _ = crate::watch::landed_beside(&pages, &[(start as usize, (end - start) as usize)]);
    true
}

/// Whether offsets `[start, end)` of title memory, which `writer` holds in
/// the card's memory, may stay there (see the file).
fn may_leave(writer: &Arc<Shadow>, start: u64, end: u64) -> bool {
    if !writer.plain_copy()
        || !write_combining(start, end)
        || crate::inplace::level() != 0
        || crate::port::wanted(start, end - start)
    {
        return false;
    }
    // Shown at one address only.
    if crate::inplace::pages_of((start / PIECE * PIECE) as usize).len() > 1 {
        return false;
    }
    !crate::memory::holds(start, end) && crate::shadow::holders(start, end, writer.id()).is_empty()
}

/// `[lo, hi)` of `shadow` at 64 KiB `key` is fetched as before: what of it
/// was left comes back with it.
fn taken_back(shadow: &Arc<Shadow>, key: u64, lo: usize, hi: usize) {
    if LEFT_N.load(Ordering::SeqCst) == 0 {
        return;
    }
    let mut book = BOOK.write().unwrap_or_else(|e| e.into_inner());
    let Some(left) = book
        .left
        .get_mut(&key)
        .filter(|left| left.writer.id() == shadow.id())
    else {
        return;
    };
    cut_run(&mut left.runs, lo, hi);
    if left.runs.is_empty() {
        book.remove(key);
    }
}

/// Whether anything is left or being fetched: a reader must look.
fn any() -> bool {
    LEFT_N.load(Ordering::SeqCst) != 0 || FETCHING.load(Ordering::SeqCst) != 0
}

/// Before title bytes `[from, to)` are read for copy `taker` (0: for no
/// copy): the pieces left there by any other copy are fetched, and what
/// their card wrote merged into title memory; they are not left again.
/// Waits for a fetch of another thread under way.
pub(crate) fn before_take(from: usize, to: usize, taker: u64) {
    if !any() || from >= to {
        return;
    }
    let _gate = GATE.lock().unwrap_or_else(|e| e.into_inner());
    fetch_in(from as u64, to as u64, taker);
}

/// Before our own work that may read or write title memory anywhere:
/// every piece left is fetched.
pub(crate) fn before_anything() {
    before_take(0, usize::MAX, 0);
}

/// [`before_take`]'s work, under [`GATE`].
fn fetch_in(from: u64, to: u64, taker: u64) {
    if LEFT_N.load(Ordering::SeqCst) == 0 {
        return;
    }
    // (The views first: no lock of the book is held while they are read.)
    let any_pages = !BOOK
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .by_page
        .is_empty();
    let pages = if any_pages && to - from <= 1 << 30 {
        crate::inplace::page_keys(from as usize, (to - from) as usize)
    } else {
        Vec::new()
    };
    let mut pieces: Vec<(Arc<Shadow>, usize, usize)> = Vec::new();
    {
        let mut book = BOOK.write().unwrap_or_else(|e| e.into_inner());
        let keys: Vec<u64> = book
            .touching(from, to, &pages)
            .into_iter()
            .filter(|key| {
                book.left
                    .get(key)
                    .is_some_and(|left| left.writer.id() != taker)
            })
            .collect();
        if keys.is_empty() {
            return;
        }
        FETCHING.fetch_add(1, Ordering::SeqCst);
        for key in keys {
            book.never.insert(key);
            if let Some(left) = book.remove(key) {
                pieces.extend(
                    left.runs
                        .iter()
                        .map(|&(lo, hi)| (left.writer.clone(), lo, hi)),
                );
            }
        }
    }
    STATS[2].fetch_add(pieces.len() as u64, Ordering::Relaxed);
    STATS[3].fetch_add(
        pieces.iter().map(|p| (p.2 - p.1) as u64).sum(),
        Ordering::Relaxed,
    );
    STATS[4].fetch_add(1, Ordering::Relaxed);
    match crate::host::host() {
        Some(host) => crate::host::leftq::fetch(host, &pieces),
        None => crate::host::fail(format_args!(
            "pieces left on the card and no device to fetch them (VICTORY_SKIP_WC_BACK)"
        )),
    }
    FETCHING.fetch_sub(1, Ordering::SeqCst);
}

/// Pieces whose copy nothing but this book holds any more (the title let
/// its buffer go, a window was let go): fetched now, when nothing is in the
/// air (`crate::host::sweep`), so that their bytes are not lost with the
/// copy.
pub(crate) fn sweep() {
    if LEFT_N.load(Ordering::SeqCst) == 0 {
        return;
    }
    let _gate = GATE.lock().unwrap_or_else(|e| e.into_inner());
    let orphans: Vec<u64> = {
        let book = BOOK.read().unwrap_or_else(|e| e.into_inner());
        let mut held: BTreeMap<u64, (usize, usize)> = BTreeMap::new();
        for left in book.left.values() {
            let entry = held
                .entry(left.writer.id())
                .or_insert((Arc::strong_count(&left.writer), 0));
            entry.1 += 1;
        }
        held.into_iter()
            .filter(|(_, (count, here))| count <= here)
            .map(|(id, _)| id)
            .collect()
    };
    for id in orphans {
        let span = {
            let book = BOOK.read().unwrap_or_else(|e| e.into_inner());
            book.left
                .values()
                .find(|left| left.writer.id() == id)
                .map(|left| left.writer.span())
        };
        if let Some((start, len)) = span {
            fetch_in(start, start + len, 0);
        }
    }
}

/// Title memory `[at, at + len)` is freed, unmapped or shows other pages:
/// what was left there is no longer the title's (forgotten, not fetched).
/// A length of 0: the whole reservation at `at`.
pub(crate) fn forget(at: usize, len: usize) {
    if !on() {
        return;
    }
    let len = if len != 0 {
        len
    } else {
        let mut map = WRITE_COMBINING.write().unwrap_or_else(|e| e.into_inner());
        match map.remove(&at) {
            Some(len) => len,
            None => return,
        }
    };
    let (from, to) = (at as u64, at.saturating_add(len) as u64);
    let mut book = BOOK.write().unwrap_or_else(|e| e.into_inner());
    let keys: Vec<u64> = book
        .left
        .range(from / PIECE * PIECE..to)
        .map(|(&key, _)| key)
        .collect();
    for key in keys {
        if book.remove(key).is_some() {
            FORGOTTEN.fetch_add(1, Ordering::Relaxed);
        }
    }
    let never: Vec<u64> = book
        .never
        .range(from / PIECE * PIECE..to)
        .copied()
        .collect();
    for key in never {
        book.never.remove(&key);
    }
}

/// Pieces forgotten as their memory went ([`forget`]), since the start.
static FORGOTTEN: AtomicU64 = AtomicU64::new(0);

/// Whether copy `id` holds a piece left in title memory `[from, to)`: its
/// refreshes there take only what the title changed (`Shadow::take`).
pub(crate) fn own_left_in(id: u64, from: usize, to: usize) -> bool {
    if LEFT_N.load(Ordering::SeqCst) == 0 || from >= to {
        return false;
    }
    let (from, to) = (from as u64, to as u64);
    let book = BOOK.read().unwrap_or_else(|e| e.into_inner());
    book.left.range(from / PIECE * PIECE..to).any(|(_, left)| {
        left.writer.id() == id && {
            let (a, b) = title_span(left);
            a < to && b > from
        }
    })
}

/// What a refresh of copy `id` over title memory `[from, to)` holds while
/// it runs ([`before_refresh`]).
pub(crate) struct Refreshing {
    _gate: Option<std::sync::MutexGuard<'static, ()>>,
    _taking: Taking,
}

/// Before a refresh of copy `id` takes title memory `[from, to)`: other
/// copies' pieces there are fetched; if the copy holds pieces of its own
/// there, no fetch runs until the refresh is done (it writes the CPU side
/// a fetch merges into).
pub(crate) fn before_refresh(id: u64, from: usize, to: usize) -> Refreshing {
    let taking = Taking::new(id);
    if !any() || from >= to {
        return Refreshing {
            _gate: None,
            _taking: taking,
        };
    }
    let gate = GATE.lock().unwrap_or_else(|e| e.into_inner());
    fetch_in(from as u64, to as u64, id);
    let own = own_left_in(id, from, to);
    Refreshing {
        _gate: own.then_some(gate),
        _taking: taking,
    }
}

/// For the audit ([`census`]): title memory `[at, at + len)` is read.
pub(crate) fn read(at: usize, len: usize) {
    if !census() || len == 0 || LEFT_N.load(Ordering::Relaxed) == 0 {
        return;
    }
    let (from, to) = (at as u64, at.saturating_add(len) as u64);
    let taker = TAKER.get();
    let Ok(book) = BOOK.try_read() else {
        return;
    };
    let mut stale = 0;
    for (_, left) in book.left.range(from / PIECE * PIECE..to) {
        if left.writer.id() == taker {
            continue;
        }
        let start = left.writer.span().0;
        for &(lo, hi) in &left.runs {
            let (a, b) = ((start + lo as u64).max(from), (start + hi as u64).min(to));
            stale += b.saturating_sub(a);
        }
    }
    if stale > 0 {
        STALE[usize::from(taker == 0)].fetch_add(stale, Ordering::Relaxed);
    }
}

/// For the frame line, with a mode on: per frame, what was left on the card
/// and what was fetched later; with [`census`], the audit.
pub(crate) fn log(frames: f64) {
    if !on() {
        return;
    }
    let [left, left_bytes, fetched, fetched_bytes, fetches, again] =
        STATS.each_ref().map(|s| s.swap(0, Ordering::Relaxed));
    tracing::info!(
        "copy-back left on the card (VICTORY_SKIP_WC_BACK={}): per frame {:.1} MB in {:.0} pieces ({:.0} of them left again), \
         {:.1} MB in {:.0} pieces fetched later for another reader ({:.1} fetches); {} pieces left now, {} forgotten so far \
         as their memory went",
        mode(),
        left_bytes as f64 / frames / 1e6,
        left as f64 / frames,
        again as f64 / frames,
        fetched_bytes as f64 / frames / 1e6,
        fetched as f64 / frames,
        fetches as f64 / frames,
        LEFT_N.load(Ordering::Relaxed),
        FORGOTTEN.load(Ordering::Relaxed)
    );
    if census() {
        let [refreshes, others] = STALE
            .each_ref()
            .map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames / 1e6);
        tracing::info!(
            "census audit of the copy-back left on the card: stale title bytes read per frame: {refreshes:.2} MB by refreshes of \
             other copies, {others:.2} MB by other readers (both must be 0)"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{add_run, cut_run};

    #[test]
    fn runs_join_and_cut() {
        let mut runs = Vec::new();
        add_run(&mut runs, 100, 200);
        add_run(&mut runs, 300, 400);
        add_run(&mut runs, 150, 300);
        assert_eq!(runs, vec![(100, 400)]);
        add_run(&mut runs, 500, 500);
        assert_eq!(runs, vec![(100, 400)]);
        cut_run(&mut runs, 200, 250);
        assert_eq!(runs, vec![(100, 200), (250, 400)]);
        cut_run(&mut runs, 0, 120);
        assert_eq!(runs, vec![(120, 200), (250, 400)]);
        cut_run(&mut runs, 180, 1000);
        assert_eq!(runs, vec![(120, 180)]);
        cut_run(&mut runs, 0, 1000);
        assert!(runs.is_empty());
    }
}
