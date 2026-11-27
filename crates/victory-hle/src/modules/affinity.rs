//! Thread affinity the title asks for (`VICTORY_AFFINITY`).
//!
//! The title creates its threads suspended and pins each to the console's
//! cores (`SetThreadGroupAffinity`, e.g. its five command-list recorders to
//! mask 0x1f00 = logical processors 8–12), Bink pins its workers with
//! `SetThreadAffinityMask` and sets ideal processors. On this PC those masks
//! squeeze busy threads onto a few cores (measured: recording 210 → 165 ms
//! summed a frame with the masks lifted).
//!
//! `VICTORY_AFFINITY=free` (off by default; `scripts\play.ps1 -CpuPorts` sets it):
//! calls from title code succeed, report the previous mask the title set (or
//! the thread's real one) and are remembered per thread id, so a get after a
//! set returns what the title expects, but the thread is not narrowed.
//! `VICTORY_AFFINITY=console` (the default): passed to Windows as before. Only
//! the title's imports reach these functions; the layer's own threads are
//! never touched.
//!
//! From Op-lite/victory-scarlett (`scarlett-fox`, 411a745/c45b43e), measured
//! there on a Ryzen 7 9800X3D: recording 210 -> 165 ms summed a frame, heavy
//! frames 150 -> 138 ms.
#![allow(non_snake_case)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU8, Ordering::Relaxed};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::{HleModule, hle_export};

/// `GROUP_AFFINITY`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GroupAffinity {
    pub mask: usize,
    pub group: u16,
    pub reserved: [u16; 3],
}

/// `PROCESSOR_NUMBER`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProcessorNumber {
    pub group: u16,
    pub number: u8,
    pub reserved: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Free,
    Console,
}

fn mode() -> Mode {
    static MODE: AtomicU8 = AtomicU8::new(0);
    match MODE.load(Relaxed) {
        1 => return Mode::Free,
        2 => return Mode::Console,
        _ => {}
    }
    let m = match std::env::var("VICTORY_AFFINITY").ok().as_deref().map(str::trim) {
        Some("free") => Mode::Free,
        None | Some("") | Some("console") => Mode::Console,
        Some(other) => {
            tracing::warn!(value = other, "VICTORY_AFFINITY: unknown value, using console");
            Mode::Console
        }
    };
    MODE.store(if m == Mode::Free { 1 } else { 2 }, Relaxed);
    m
}

/// What the title asked for, per thread id, and what it has asked so far.
#[derive(Debug, Default)]
pub struct Book {
    masks: HashMap<u32, GroupAffinity>,
    ideal: HashMap<u32, ProcessorNumber>,
    /// Mask → how many set calls asked for it.
    seen: BTreeMap<usize, u64>,
    ignored: u64,
    ideal_ignored: u64,
    /// Set calls counted in the last summary.
    logged: u64,
}

impl Book {
    /// Records a set the title made; returns what the title set before, if anything.
    pub fn set(&mut self, thread: u32, affinity: GroupAffinity) -> Option<GroupAffinity> {
        self.ignored += 1;
        *self.seen.entry(affinity.mask).or_default() += 1;
        self.masks.insert(thread, GroupAffinity { reserved: [0; 3], ..affinity })
    }

    /// What the title last set for `thread`.
    pub fn get(&self, thread: u32) -> Option<GroupAffinity> {
        self.masks.get(&thread).copied()
    }

    /// Records an ideal processor; returns the one the title set before.
    pub fn set_ideal(&mut self, thread: u32, ideal: ProcessorNumber) -> Option<ProcessorNumber> {
        self.ideal_ignored += 1;
        self.ideal.insert(thread, ProcessorNumber { reserved: 0, ..ideal })
    }

    /// "N set calls ignored, masks seen: 0x1f00 x5, 0xff x8" (most asked first).
    pub fn summary(&self) -> String {
        let mut seen: Vec<_> = self.seen.iter().collect();
        seen.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        let masks: Vec<String> = seen.iter().map(|(m, n)| format!("{m:#x} x{n}")).collect();
        let mut line = format!(
            "thread affinity: {} set calls ignored, masks seen: {}",
            self.ignored,
            if masks.is_empty() { "none".into() } else { masks.join(", ") }
        );
        if self.ideal_ignored > 0 {
            line += &format!("; {} ideal-processor calls ignored", self.ideal_ignored);
        }
        line
    }
}

static BOOK: Mutex<Option<Book>> = Mutex::new(None);

fn with_book<R>(f: impl FnOnce(&mut Book) -> R) -> R {
    let mut book = BOOK.lock().unwrap_or_else(|e| e.into_inner());
    f(book.get_or_insert_with(Book::default))
}

/// The first ignored call logs that masks are not applied and registers the
/// exit summary; later ones log the summary at most every 5 s while calls come.
fn note_ignored() {
    static FIRST: std::sync::Once = std::sync::Once::new();
    FIRST.call_once(|| {
        tracing::info!("thread affinity: title masks are not applied (VICTORY_AFFINITY=free)");
        // SAFETY: registering a plain function with the CRT.
        unsafe { atexit(at_exit) };
    });
    static LAST: Mutex<Option<Instant>> = Mutex::new(None);
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let now = Instant::now();
    if last.is_none_or(|t| now - t >= Duration::from_secs(5)) {
        *last = Some(now);
        drop(last);
        log_summary();
    }
}

fn log_summary() {
    if let Some(line) = with_book(|b| {
        (b.ignored + b.ideal_ignored != b.logged).then(|| {
            b.logged = b.ignored + b.ideal_ignored;
            b.summary()
        })
    }) {
        tracing::info!("{line}");
    }
}

extern "C" fn at_exit() {
    log_summary();
}

unsafe extern "C" {
    fn atexit(f: extern "C" fn()) -> i32;
}

#[link(name = "kernel32")]
unsafe extern "system" {
    #[link_name = "SetThreadGroupAffinity"]
    fn host_set_group(thread: *mut c_void, affinity: *const GroupAffinity, previous: *mut GroupAffinity) -> i32;
    #[link_name = "GetThreadGroupAffinity"]
    fn host_get_group(thread: *mut c_void, affinity: *mut GroupAffinity) -> i32;
    #[link_name = "SetThreadAffinityMask"]
    fn host_set_mask(thread: *mut c_void, mask: usize) -> usize;
    #[link_name = "SetThreadIdealProcessorEx"]
    fn host_set_ideal(thread: *mut c_void, ideal: *const ProcessorNumber, previous: *mut ProcessorNumber) -> i32;
    #[link_name = "GetThreadIdealProcessorEx"]
    fn host_get_ideal(thread: *mut c_void, ideal: *mut ProcessorNumber) -> i32;
    #[link_name = "GetThreadId"]
    fn host_thread_id(thread: *mut c_void) -> u32;
}

/// The thread's id, or `None` for a handle Windows does not know (the call
/// then goes to Windows, which fails it the usual way).
fn thread_id(thread: *mut c_void) -> Option<u32> {
    // SAFETY: a query on the title's handle.
    let id = unsafe { host_thread_id(thread) };
    (id != 0).then_some(id)
}

/// The thread's real affinity.
fn real_group(thread: *mut c_void) -> Option<GroupAffinity> {
    let mut a = GroupAffinity::default();
    // SAFETY: a query into our own struct.
    (unsafe { host_get_group(thread, &mut a) } != 0).then_some(a)
}

pub extern "system" fn SetThreadGroupAffinity(
    thread: *mut c_void,
    affinity: *const GroupAffinity,
    previous: *mut GroupAffinity,
) -> i32 {
    let id = thread_id(thread);
    let (Mode::Free, Some(id), false) = (mode(), id, affinity.is_null()) else {
        // SAFETY: the title's own arguments.
        return unsafe { host_set_group(thread, affinity, previous) };
    };
    // SAFETY: the title passes a valid GROUP_AFFINITY.
    let asked = unsafe { *affinity };
    let before = with_book(|b| b.set(id, asked)).or_else(|| real_group(thread)).unwrap_or_default();
    if !previous.is_null() {
        // SAFETY: the title passes room for one GROUP_AFFINITY.
        unsafe { *previous = before };
    }
    note_ignored();
    1
}

pub extern "system" fn GetThreadGroupAffinity(thread: *mut c_void, affinity: *mut GroupAffinity) -> i32 {
    if mode() == Mode::Free
        && !affinity.is_null()
        && let Some(set) = thread_id(thread).and_then(|id| with_book(|b| b.get(id)))
    {
        // SAFETY: the title passes room for one GROUP_AFFINITY.
        unsafe { *affinity = set };
        return 1;
    }
    // SAFETY: the title's own arguments.
    unsafe { host_get_group(thread, affinity) }
}

pub extern "system" fn SetThreadAffinityMask(thread: *mut c_void, mask: usize) -> usize {
    let (Mode::Free, Some(id), true) = (mode(), thread_id(thread), mask != 0) else {
        // SAFETY: the title's own arguments.
        return unsafe { host_set_mask(thread, mask) };
    };
    let before = with_book(|b| b.set(id, GroupAffinity { mask, ..Default::default() }))
        .or_else(|| real_group(thread))
        .map_or(0, |a| a.mask);
    note_ignored();
    // A previous mask of 0 would read as failure.
    if before == 0 { mask } else { before }
}

pub extern "system" fn SetThreadIdealProcessorEx(
    thread: *mut c_void,
    ideal: *const ProcessorNumber,
    previous: *mut ProcessorNumber,
) -> i32 {
    let (Mode::Free, Some(id), false) = (mode(), thread_id(thread), ideal.is_null()) else {
        // SAFETY: the title's own arguments.
        return unsafe { host_set_ideal(thread, ideal, previous) };
    };
    // SAFETY: the title passes a valid PROCESSOR_NUMBER.
    let asked = unsafe { *ideal };
    let before = with_book(|b| b.set_ideal(id, asked)).or_else(|| {
        let mut p = ProcessorNumber::default();
        // SAFETY: a query into our own struct.
        (unsafe { host_get_ideal(thread, &mut p) } != 0).then_some(p)
    });
    if !previous.is_null() {
        // SAFETY: the title passes room for one PROCESSOR_NUMBER.
        unsafe { *previous = before.unwrap_or_default() };
    }
    note_ignored();
    1
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "affinity",
        dlls: &[
            "kernel32.dll",
            "kernelbase.dll",
            "api-ms-win-core-processtopology-*",
            "api-ms-win-core-processthreads-*",
        ],
        exports: vec![
            hle_export!(SetThreadGroupAffinity),
            hle_export!(GetThreadGroupAffinity),
            hle_export!(SetThreadAffinityMask),
            hle_export!(SetThreadIdealProcessorEx),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mask(m: usize) -> GroupAffinity {
        GroupAffinity { mask: m, group: 0, reserved: [7; 3] }
    }

    #[test]
    fn a_get_after_a_set_returns_what_was_set() {
        let mut b = Book::default();
        assert_eq!(b.get(10), None);
        assert_eq!(b.set(10, mask(0x1f00)), None);
        assert_eq!(b.get(10), Some(GroupAffinity { mask: 0x1f00, group: 0, reserved: [0; 3] }));
        // The previous mask is the one set before.
        assert_eq!(b.set(10, mask(0xff)).map(|a| a.mask), Some(0x1f00));
        assert_eq!(b.get(10).map(|a| a.mask), Some(0xff));
        assert_eq!(b.get(11), None);
    }

    #[test]
    fn summary_counts_masks_most_asked_first() {
        let mut b = Book::default();
        for t in 0..5 {
            b.set(t, mask(0x1f00));
        }
        b.set(20, mask(0xff));
        b.set(21, mask(0x3fff));
        b.set(22, mask(0x3fff));
        assert_eq!(b.summary(), "thread affinity: 8 set calls ignored, masks seen: 0x1f00 x5, 0x3fff x2, 0xff x1");
        b.set_ideal(20, ProcessorNumber { group: 0, number: 3, reserved: 9 });
        assert_eq!(
            b.set_ideal(20, ProcessorNumber::default()),
            Some(ProcessorNumber { group: 0, number: 3, reserved: 0 })
        );
        assert!(b.summary().ends_with("; 2 ideal-processor calls ignored"), "{}", b.summary());
    }

    #[test]
    fn empty_summary() {
        assert_eq!(Book::default().summary(), "thread affinity: 0 set calls ignored, masks seen: none");
    }
}
