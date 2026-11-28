//! `VICTORY_RING_METER=1` (off by default): what the title's CPU gets out of
//! its read-back rings.
//!
//! The engine reads what the GPU wrote through one class of its own: a ring
//! of read-back buffers (1 or 3 of them, 4 bytes to 32 KiB each), made at
//! 22 places in its code, among them the impostors', the visibility
//! buffer's and the terrain virtual texture's feedback. Each frame the GPU
//! writes one buffer of a ring and the CPU reads another through the ring's
//! reader, which answers whether there is something to read and locks the
//! buffer. On the console that buffer is shared memory. Here it is title
//! memory that holds what shaders wrote only if the layer put it there
//! (`crate::texback`, the copies back of what shaders write into buffers).
//!
//! The hooks (`victory-hle`'s `ringhook`, in the title's own code) tell
//! this module of each ring made, each read and each "written" mark. For
//! each read that was answered it looks at the bytes the title is about to
//! read: all zero, or the same as at the ring's read before. A kind of ring
//! whose reads are all zero or never change is one whose data never
//! arrives; the line names it by the place that made the ring.
//!
//! The ring's layout and the hook addresses are from Op-lite's analysis of
//! build 1.0.2.1, as kept in PastiiJay/klbf-lab (`tools/docs/ghidra-ledger.md`
//! findings F1 and F4, `crates/victory-hle/src/readback.rs`). This is a
//! meter only: it changes nothing the title or the layer does.
//!
//! shortcut: content, not provenance. A read counts as fresh when its bytes
//! changed, whoever changed them; which of the layer's copies landed there
//! and when is not followed (klbf-lab's `F1-readback-set.patch` does that).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

/// Bytes of a read looked at, at most (the rings found hold up to 32 KiB).
const LOOK_MOST: usize = 256 << 10;
/// The first rings said one by one.
const SAY_RINGS: u64 = 48;

/// Whether the meter is on (`VICTORY_RING_METER=1`).
pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_RING_METER").is_ok_and(|v| v.trim() == "1"))
}

/// Whether a hook told us anything yet (the line stays quiet until then).
static HEARD: AtomicBool = AtomicBool::new(false);

/// One ring, by its address.
struct Ring {
    site: u64,
    /// The checksum of what its last answered read held.
    last: Option<u64>,
}

/// The rings made at one place in the title's code, since the last line
/// and (`ever_*`) since the start.
#[derive(Default)]
struct Kind {
    name: &'static str,
    made: u64,
    buffers: usize,
    bytes: u64,
    reads: u64,
    answered: u64,
    zero: u64,
    same: u64,
    marks: u64,
    ever_answered: u64,
    ever_nonzero: u64,
    /// Answered reads of a ring that had one before, and those of them
    /// that held other bytes.
    ever_compared: u64,
    ever_changed: u64,
}

#[derive(Default)]
struct Meter {
    rings: HashMap<u64, Ring>,
    kinds: HashMap<u64, Kind>,
    made: u64,
}

static METER: Mutex<Option<Meter>> = Mutex::new(None);

fn with<R>(f: impl FnOnce(&mut Meter) -> R) -> R {
    let mut m = METER.lock().unwrap_or_else(|e| e.into_inner());
    f(m.get_or_insert_with(Meter::default))
}

/// A checksum that shows a change of content (FNV-1a, 64 bits).
fn sum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// Title memory read through the kernel: `None` if a page is not there.
#[cfg(windows)]
fn read(at: u64, n: usize) -> Option<Vec<u8>> {
    if at < 0x1_0000 || at >> 47 != 0 {
        return None;
    }
    let mut v = vec![0u8; n];
    crate::host::read_into(at as *const u8, &mut v).then_some(v)
}

#[cfg(not(windows))]
fn read(_: u64, _: usize) -> Option<Vec<u8>> {
    None
}

fn read_u64(at: u64) -> Option<u64> {
    read(at, 8).map(|v| u64::from_le_bytes(v.try_into().unwrap_or([0; 8])))
}

/// The buffers of the ring at `ring`: flags in the u32 at +0x54 (bit 3: one
/// buffer, else three), the buffer objects at +0, +8, +0x10; an object's
/// u64 at +8 holds its element count (low 32 bits) and stride (12 bits
/// from bit 32), its CPU address is at +0x18.
fn buffers_of(ring: u64) -> Vec<(u64, u64)> {
    let flags = read_u64(ring + 0x50).map_or(0, |v| v >> 32);
    let count = if flags & 8 != 0 { 1 } else { 3 };
    (0..count)
        .filter_map(|i| {
            let object = read_u64(ring + 8 * i).filter(|&o| o != 0)?;
            let (head, cpu) = (read_u64(object + 8)?, read_u64(object + 0x18)?);
            let bytes = (head & 0xffff_ffff) * ((head >> 32) & 0xfff);
            (cpu != 0 && bytes != 0 && bytes <= 1 << 30).then_some((cpu, bytes))
        })
        .collect()
}

/// The title made a read-back ring at `ring` (its buffers exist). `site`:
/// where its maker returns to, as an address of the image at its preferred
/// base (0: not found); `name`: what the ledger calls that place, or "".
pub fn ring_created(ring: u64, site: u64, name: &'static str) {
    if !on() || ring == 0 {
        return;
    }
    HEARD.store(true, Ordering::Relaxed);
    let buffers = buffers_of(ring);
    let n = with(|m| {
        m.made += 1;
        let k = m.kinds.entry(site).or_default();
        k.name = name;
        k.made += 1;
        k.buffers = buffers.len();
        k.bytes = buffers.first().map_or(0, |b| b.1);
        // (Rings made and dropped all run long: the oldest are let go.)
        if m.rings.len() >= 1 << 16 {
            m.rings.clear();
        }
        m.rings.insert(ring, Ring { site, last: None });
        m.made
    });
    if n <= SAY_RINGS || n.is_power_of_two() {
        let at: Vec<String> = buffers.iter().map(|b| format!("{:#x}", b.0)).collect();
        tracing::info!(
            "read-back ring {n}: {} buffers of {} bytes at [{}] (ring {ring:#x}, made for {site:#x} {name}) (VICTORY_RING_METER)",
            buffers.len(),
            buffers.first().map_or(0, |b| b.1),
            at.join(", ")
        );
    }
}

/// The ring's reader returned: `ok` if it answered, and then `record`
/// holds the CPU address to read (+0) and its bytes (u32 at +0x18).
pub fn ring_read(ring: u64, record: u64, ok: bool) {
    if !on() {
        return;
    }
    HEARD.store(true, Ordering::Relaxed);
    // What the title is about to read, looked at outside the meter's lock.
    let held = if ok && record != 0 {
        match (read_u64(record), read_u64(record + 0x18)) {
            (Some(cpu), Some(bytes)) => {
                let bytes = (bytes & 0xffff_ffff) as usize;
                (cpu != 0 && bytes != 0)
                    .then(|| read(cpu, bytes.min(LOOK_MOST)))
                    .flatten()
            }
            _ => None,
        }
    } else {
        None
    };
    let looked = held.as_ref().map(|v| (sum(v), v.iter().all(|&b| b == 0)));
    with(|m| {
        let r = m.rings.entry(ring).or_insert(Ring {
            site: 0,
            last: None,
        });
        let before = r.last;
        if let Some((s, _)) = looked {
            r.last = Some(s);
        }
        let k = m.kinds.entry(r.site).or_default();
        k.reads += 1;
        if ok {
            k.answered += 1;
            k.ever_answered += 1;
        }
        if let Some((s, zero)) = looked {
            if zero {
                k.zero += 1;
            } else {
                k.ever_nonzero += 1;
            }
            if let Some(b) = before {
                k.ever_compared += 1;
                if b == s {
                    k.same += 1;
                } else {
                    k.ever_changed += 1;
                }
            }
        }
    });
}

/// The title marked the ring as written by the GPU this frame.
pub fn ring_marked(ring: u64) {
    if !on() {
        return;
    }
    HEARD.store(true, Ordering::Relaxed);
    with(|m| {
        let site = m.rings.get(&ring).map_or(0, |r| r.site);
        m.kinds.entry(site).or_default().marks += 1;
    });
}

/// The meter's line, over `frames` frames since the one before.
pub(crate) fn line(frames: f64) {
    if !on() || !HEARD.load(Ordering::Relaxed) {
        return;
    }
    let frames = frames.max(1.0);
    let (alive, mut rows, never) = with(|m| {
        let mut never = Vec::new();
        let rows: Vec<(u64, String)> = m
            .kinds
            .iter_mut()
            .map(|(&site, k)| {
                let share = |n: u64, of: u64| if of == 0 { 0.0 } else { 100.0 * n as f64 / of as f64 };
                let looked = k.answered.max(1);
                let row = format!(
                    "{site:#x} {}{}x{} B ({} made): {:.2} reads, {:.2} answered ({:.0}% all zero, {:.0}% as the read before), {:.2} marked written",
                    if k.name.is_empty() { String::new() } else { format!("{} ", k.name) },
                    k.buffers,
                    k.bytes,
                    k.made,
                    k.reads as f64 / frames,
                    k.answered as f64 / frames,
                    share(k.zero, looked),
                    share(k.same, looked),
                    k.marks as f64 / frames,
                );
                // Read often enough to tell, and never anything in it, or
                // never anything new.
                let all_zero = k.ever_answered >= 64 && k.ever_nonzero == 0;
                let unchanged = k.ever_compared >= 64 && k.ever_changed == 0;
                if all_zero || unchanged {
                    never.push(format!(
                        "{site:#x} {} ({} reads answered since the start, {})",
                        k.name,
                        k.ever_answered,
                        if all_zero { "all zero" } else { "never other bytes than the read before" }
                    ));
                }
                let reads = k.reads;
                (k.reads, k.answered, k.zero, k.same, k.marks) = (0, 0, 0, 0, 0);
                (reads, row)
            })
            .collect();
        (m.rings.len(), rows, never)
    });
    rows.sort_by_key(|r| std::cmp::Reverse(r.0));
    let rows: Vec<String> = rows.into_iter().take(24).map(|r| r.1).collect();
    tracing::info!(
        "read-back rings (VICTORY_RING_METER), per frame by the place that made them: {}; {alive} rings known",
        rows.join("; ")
    );
    // Said at the 1st, 2nd, 4th... line that has any.
    static SAID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if !never.is_empty() && (SAID.fetch_add(1, Ordering::Relaxed) + 1).is_power_of_two() {
        tracing::warn!(
            "read-back rings whose reads never held anything new (the GPU wrote the same every time, or what it wrote does not reach title memory): {}",
            never.join("; ")
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_checksum_tells_contents_apart() {
        assert_eq!(sum(&[1, 2, 3]), sum(&[1, 2, 3]));
        assert_ne!(sum(&[1, 2, 3]), sum(&[1, 2, 4]));
        assert_ne!(sum(&[0]), sum(&[0, 0]));
    }
}
