//! `VICTORY_BVH_STORES=1` (diagnostics builds): which of the structure
//! builder's programs store by address outside the structures their dispatch
//! was given.
//!
//! Moving objects' structures turn to garbage while nothing builds over them
//! (dump bvhcheck-4: the bottom level 96 % of rays enter, refitted 102 times,
//! has 1,798 of 6,684 triangles out of range, their corners dwords of the
//! builder's own records), and the builder touches pages that are not the
//! title's memory every frame: a program stores through a pointer that is not
//! right. With the switch every builder store by address also leaves its page
//! and the dispatch's tag ([`victory_shader::emit::MEMORY_STORE_NOTES`]).
//! Once a frame the notes are read here and each is put to its dispatch:
//!
//! - **own**: within [`OWN`] after one of the dispatch's own unordered access
//!   views (its structure, scratch, results), and no other dispatch's view
//!   starts between;
//! - **past a neighbour**: after one of its own views, but another live
//!   dispatch's view starts between (a structure written over);
//! - **outside**: near none of its own views;
//! - **unknown**: the tag's dispatch is gone from the ring (more than
//!   [`RING`] builder dispatches ago).
//!
//! The first lines of the last three name the program, the page and the views,
//! and every 10 seconds a line sums them by program.

use std::collections::{BTreeMap, HashMap};

use victory_shader::emit::{MEMORY_FAULTS, MEMORY_STORE_NOTES};

use super::{Memory, PAGE};

/// Dispatches kept for their tags (the tag has 11 bits).
const RING: usize = 1 << 11;
/// A store this far past one of the dispatch's own views is its own.
const OWN: u64 = 16 << 20;
/// A view another dispatch wrote within so many frames is live.
const LIVE: u64 = 64;

pub(crate) fn on() -> bool {
    victory_shader::emit::bvh_stores()
}

struct Dispatch {
    frame: u64,
    program: u32,
    /// Its unordered access root views; the rest of its root views.
    written: Vec<u64>,
    named: Vec<u64>,
}

#[derive(Default)]
pub(crate) struct Stores {
    ring: Vec<Option<Dispatch>>,
    next: u32,
    /// Views the builder wrote: the last frame and the program.
    recent: BTreeMap<u64, (u64, u32)>,
    names: Vec<String>,
    by_name: HashMap<String, u32>,
    /// By program: own, past a neighbour, outside, unknown; and of the last
    /// three, those on pages without a copy.
    counts: HashMap<u32, [u64; 5]>,
    said: u32,
    from: Option<std::time::Instant>,
}

impl Stores {
    /// A builder dispatch's tag (set as its `USER_DISPATCH_TAG` word).
    pub(crate) fn tag(
        &mut self,
        frame: u64,
        name: &str,
        written: &[u64],
        addresses: &[u64],
    ) -> u32 {
        if self.ring.is_empty() {
            self.ring.resize_with(RING, || None);
        }
        let program = match self.by_name.get(name) {
            Some(&n) => n,
            None => {
                let n = self.names.len() as u32;
                self.names.push(name.to_string());
                self.by_name.insert(name.to_string(), n);
                n
            }
        };
        let tag = self.next % RING as u32;
        self.next = self.next.wrapping_add(1);
        let mut w: Vec<u64> = written.iter().copied().filter(|&a| a >= 1 << 32).collect();
        w.sort_unstable();
        w.dedup();
        let mut named: Vec<u64> = addresses
            .iter()
            .copied()
            .filter(|&a| a >= 1 << 32 && !w.contains(&a))
            .collect();
        named.sort_unstable();
        named.dedup();
        for &at in &w {
            self.recent.insert(at, (frame, program));
        }
        self.ring[tag as usize] = Some(Dispatch {
            frame,
            program,
            written: w,
            named,
        });
        tag
    }

    fn name(&self, program: u32) -> &str {
        self.names.get(program as usize).map_or("?", String::as_str)
    }
}

/// What a note is ([`Stores`]): 0 own, 1 past a neighbour, 2 outside, 3 unknown.
fn kind(
    d: Option<&Dispatch>,
    recent: &BTreeMap<u64, (u64, u32)>,
    page: u64,
    frame: u64,
) -> (usize, Option<u64>, Option<(u64, u32)>) {
    let Some(d) = d else {
        return (3, None, None);
    };
    let (start, end) = (page * PAGE, (page + 1) * PAGE);
    let own = d
        .written
        .iter()
        .copied()
        .filter(|&w| w < end && start < w.saturating_add(OWN))
        .max();
    let Some(own) = own else {
        return (2, None, None);
    };
    // Another live dispatch's view between its own and the page's start.
    let neighbour = recent
        .range(own + 1..start.max(own + 1))
        .find(|(at, (last, _))| !d.written.contains(at) && frame.saturating_sub(*last) <= LIVE)
        .map(|(&at, &(_, program))| (at, program));
    match neighbour {
        Some(n) => (1, Some(own), Some(n)),
        None => (0, Some(own), None),
    }
}

impl Memory {
    /// Reads the frame's store notes and puts each to its dispatch.
    pub(super) fn take_stores(&mut self) {
        if !on() {
            return;
        }
        let first = 2 * MEMORY_FAULTS as usize + 1;
        let mut notes = Vec::new();
        for i in first..first + MEMORY_STORE_NOTES as usize - 1 {
            // SAFETY: inside the notes' mapping; a note missed now comes again.
            let at = unsafe { self.faults.1.add(i) };
            // SAFETY: as above.
            let v = unsafe { at.read_volatile() };
            if v != 0 {
                notes.push(v);
                // SAFETY: as above.
                unsafe { at.write_volatile(0) };
            }
        }
        let frame = self.frame;
        for v in notes {
            let (page, tag) = (u64::from(v & 0x1F_FFFF), (v >> 21) as usize);
            let s = &self.stores;
            let d = s.ring.get(tag).and_then(Option::as_ref);
            let (k, own, neighbour) = kind(d, &s.recent, page, frame);
            let program = d.map_or(u32::MAX, |d| d.program);
            let views = (k != 0).then(|| {
                d.map_or((0, String::new(), String::new()), |d| {
                    let list = |v: &[u64]| {
                        v.iter()
                            .map(|a| format!("{a:#x}"))
                            .collect::<Vec<_>>()
                            .join(" ")
                    };
                    (
                        frame.saturating_sub(d.frame),
                        list(&d.written),
                        list(&d.named),
                    )
                })
            });
            let missing = self.copy_of(page).is_none();
            let counts = self.stores.counts.entry(program).or_default();
            counts[k] += 1;
            if k != 0 && missing {
                counts[4] += 1;
            }
            if k == 0 {
                continue;
            }
            let s = &mut self.stores;
            s.said += 1;
            if s.said <= 400 || s.said.is_multiple_of(1000) {
                let what = [
                    "own",
                    "past a neighbour's start",
                    "outside its views",
                    "of a dispatch no longer known",
                ][k];
                let (lag, written, named) = views.unwrap_or_default();
                let n = neighbour
                    .map(|(at, p)| format!("{at:#x} of {}", s.name(p)))
                    .unwrap_or_default();
                tracing::warn!(
                    n = s.said,
                    frame,
                    lag,
                    page = format_args!("{:#x}", page * PAGE),
                    own = format_args!("{:#x}", own.unwrap_or(0)),
                    neighbour = n,
                    no_copy = missing,
                    "bvh stores: {} stores by address {what}; its views written [{written}], named [{named}]",
                    s.name(program)
                );
            }
        }
        let now = std::time::Instant::now();
        let since = *self.stores.from.get_or_insert(now);
        if now.duration_since(since).as_secs_f64() >= 10.0 {
            let s = &mut self.stores;
            s.from = Some(now);
            let mut rows: Vec<(u32, [u64; 5])> = s.counts.drain().collect();
            rows.sort_by_key(|r| std::cmp::Reverse(r.1[1] + r.1[2]));
            let text: Vec<String> = rows
                .iter()
                .map(|(p, c)| format!("{}: own {}, past a neighbour {}, outside {}, unknown {}, without a copy {}", s.name(*p), c[0], c[1], c[2], c[3], c[4]))
                .collect();
            tracing::info!(
                "bvh stores (VICTORY_BVH_STORES) in 10 s, pages a frame noted, by program: {}",
                text.join("; ")
            );
            s.recent
                .retain(|_, (last, _)| frame.saturating_sub(*last) <= 4 * LIVE);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_store_is_put_to_its_dispatch() {
        let page = 0xd8_0000u64;
        let base = page * PAGE + 0x100;
        let d = Dispatch {
            frame: 10,
            program: 0,
            written: vec![base],
            named: vec![],
        };
        let mut recent = BTreeMap::new();
        recent.insert(base, (10, 0));
        assert_eq!(
            kind(Some(&d), &recent, page + 2, 10).0,
            0,
            "within its own structure"
        );
        recent.insert(base + PAGE, (9, 1));
        assert_eq!(
            kind(Some(&d), &recent, page + 2, 10).0,
            1,
            "past another dispatch's view"
        );
        assert_eq!(
            kind(Some(&d), &recent, page + 0x1000, 10).0,
            2,
            "far from its views"
        );
        assert_eq!(kind(None, &recent, page, 10).0, 3);
        recent.insert(base + PAGE, (9 - LIVE - 1, 1));
        assert_eq!(
            kind(Some(&d), &recent, page + 2, 100).0,
            0,
            "a view not written lately is no neighbour"
        );
    }
}
