//! What the sweep lets go of at once, dropped on a thread of its own
//! (`VICTORY_SWEEP_ASIDE=1`; off by default), and how long the sweep takes
//! (after PalmVI 283782de7).
//!
//! The sweep runs once a second on the submitting thread, before a
//! submission starts (`super::sweep`). What it drops there itself is the
//! staging buffers of textures no list holds (`crate::texture::let_go_idle`)
//! and, with `VICTORY_BURY_LAG=0` alone, the host copies it retires while
//! nothing runs (`super::retire`): each a release of a D3D12 resource whose
//! memory the driver frees. PalmVI's profile (RTX 4090) had the sweep at
//! 31-56 ms in some frames, the card idle meanwhile. With the switch those
//! drops go to a thread that does nothing else: the same objects, released
//! a little later, off the submitting thread.
//!
//! Off unless asked for. What the sweep retires with a lag (the default,
//! `super::bury_lag`) is dropped by `super::bury` and stays there: runs ba2
//! and ba4 dropped that on a thread of its own (`VICTORY_BURY_ASIDE`, taken
//! out again) and the host's device hung in both, for a reason not known.
//! That nothing in flight names the objects handed over here is shown (a
//! staging buffer goes when its table alone holds it, under the table's
//! lock; a list that uploads from it holds it until it is reset), but so it
//! was there, and no run has had this switch on.

use super::Retired;
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{OnceLock, mpsc};

/// Whether the switch is on.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SWEEP_ASIDE").is_some_and(|v| v == "1"))
}

thread_local! {
    /// This thread is sweeping with the switch on.
    static SWEEPING: Cell<bool> = const { Cell::new(false) };
}

/// While held (made by [`sweeping`]), what this thread hands to
/// [`drop_aside`] is dropped aside.
pub(super) struct Sweeping(());

impl Drop for Sweeping {
    fn drop(&mut self) {
        SWEEPING.with(|s| s.set(false));
    }
}

/// With the switch on: drops aside on this thread until the guard goes.
pub(super) fn sweeping() -> Option<Sweeping> {
    sweeping_if(on())
}

fn sweeping_if(on: bool) -> Option<Sweeping> {
    on.then(|| {
        SWEEPING.with(|s| s.set(true));
        Sweeping(())
    })
}

/// Since the last line: objects handed to the thread, objects it dropped,
/// and the microseconds it spent dropping them.
static DROPPED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Drops `object` aside if this thread is sweeping with the switch on;
/// otherwise (or if the thread could not start, or has ended) gives it
/// back.
pub(super) fn drop_aside(object: Retired) -> Result<(), Retired> {
    if !SWEEPING.with(Cell::get) {
        return Err(object);
    }
    // The dropping thread's end of the queue; `None` if it could not start.
    static QUEUE: OnceLock<Option<mpsc::Sender<Retired>>> = OnceLock::new();
    let sender = QUEUE.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<Retired>();
        std::thread::Builder::new()
            .name("victory-sweep-aside".into())
            .spawn(move || {
                for object in rx {
                    let started = std::time::Instant::now();
                    drop(object);
                    DROPPED[1].fetch_add(1, Relaxed);
                    DROPPED[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
                }
            })
            .ok()
            .map(|_| tx)
    });
    match sender {
        Some(tx) => {
            tx.send(object).map_err(|e| e.0)?;
            DROPPED[0].fetch_add(1, Relaxed);
            Ok(())
        }
        None => Err(object),
    }
}

/// The parts of a sweep that are timed.
pub(super) const PARTS: usize = 5;

/// Since the last line: sweeps, the microseconds of the longest, and of
/// its parts (memory figures and the pipeline library, copies of windows,
/// staging buffers, host copies of textures, pieces left on the card).
static SWEPT: [AtomicU64; 2 + PARTS] = [const { AtomicU64::new(0) }; 2 + PARTS];
/// Since the start: sweeps, and the microseconds of the longest.
static EVER: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// A sweep took `parts` microseconds on the submitting thread.
pub(super) fn swept(parts: [u64; PARTS]) {
    let all: u64 = parts.iter().sum();
    SWEPT[0].fetch_add(1, Relaxed);
    EVER[0].fetch_add(1, Relaxed);
    EVER[1].fetch_max(all, Relaxed);
    // (Only the submitting thread sweeps: no other writes these.)
    if all >= SWEPT[1].load(Relaxed) {
        SWEPT[1].store(all, Relaxed);
        for (kept, part) in SWEPT[2..].iter().zip(parts) {
            kept.store(part, Relaxed);
        }
    }
}

/// For the frame-rate line: how long the sweep held the submitting thread,
/// and what was dropped aside.
pub(super) fn line() {
    let s: Vec<u64> = SWEPT.iter().map(|s| s.swap(0, Relaxed)).collect();
    if s[0] == 0 {
        return;
    }
    let ms = |i: usize| s[i] as f64 / 1e3;
    let [handed, dropped, us] = [0, 1, 2].map(|i| DROPPED[i].swap(0, Relaxed));
    tracing::info!(
        "the sweep on the submitting thread, since the last line: {}, the longest {:.1} ms ({:.1} memory figures and the pipeline \
         library, {:.1} copies of windows, {:.1} staging buffers, {:.1} host copies of textures, {:.1} pieces left on the card); \
         since the start {} sweeps, the longest {:.1} ms; what it dropped at once given to a thread of its own \
         (VICTORY_SWEEP_ASIDE={}): {handed}, {dropped} dropped there in {:.1} ms",
        s[0],
        ms(1),
        ms(2),
        ms(3),
        ms(4),
        ms(5),
        ms(6),
        EVER[0].load(Relaxed),
        EVER[1].load(Relaxed) as f64 / 1e3,
        u8::from(on()),
        us as f64 / 1e3,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// Tells the thread that dropped it.
    struct Told(Arc<Mutex<Option<std::thread::ThreadId>>>);
    impl Drop for Told {
        fn drop(&mut self) {
            *self.0.lock().unwrap() = Some(std::thread::current().id());
        }
    }

    #[test]
    fn dropped_aside_only_while_sweeping_and_each_object_once() {
        let here = std::thread::current().id();
        // Not sweeping: given back, dropped here.
        let seen = Arc::new(Mutex::new(None));
        let back = drop_aside(Retired::Any(Box::new(Told(seen.clone()))));
        assert!(back.is_err());
        drop(back);
        assert_eq!(*seen.lock().unwrap(), Some(here));
        // Sweeping: dropped by the other thread, once.
        let seen = Arc::new(Mutex::new(None));
        {
            let _sweeping = sweeping_if(true);
            assert!(drop_aside(Retired::Any(Box::new(Told(seen.clone())))).is_ok());
        }
        let started = std::time::Instant::now();
        while Arc::strong_count(&seen) > 1 && started.elapsed().as_secs() < 10 {
            std::thread::yield_now();
        }
        let by = seen.lock().unwrap().expect("dropped");
        assert_ne!(by, here);
        // The guard gone: given back again.
        assert!(drop_aside(Retired::Any(Box::new(0u8))).is_err());
        assert!(sweeping_if(false).is_none());
    }

    #[test]
    fn the_longest_sweep_keeps_its_parts() {
        // (No other test sweeps.)
        swept([1, 2, 3, 4, 5]);
        swept([10, 0, 0, 0, 0]);
        assert_eq!(SWEPT[0].load(Relaxed), 2);
        assert_eq!(SWEPT[1].load(Relaxed), 15);
        assert_eq!(SWEPT[4].load(Relaxed), 3);
        assert_eq!(EVER[1].load(Relaxed), 15);
    }
}
