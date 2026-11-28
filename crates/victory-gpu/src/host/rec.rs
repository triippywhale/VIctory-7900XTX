//! A host list's D3D12 list, with barrier calls joined
//! (`VICTORY_JOIN_BARRIERS=1`; off by default; A/B-able with
//! `VICTORY_AB=<s>,VICTORY_JOIN_BARRIERS`), and a barrier on every shader
//! write that repeats one left out (`VICTORY_UAV_BARRIER_ONCE=1`; off by
//! default; [`uav_once`]).
//!
//! The state changes the host asks for (`crate::list::ensure_state`, the
//! aliasing before a texture's first upload) and the title's own barriers
//! wait here instead of going to the list at once; they go, in the order
//! they were asked for, in one call, at the next use of the list for
//! anything else. Every use goes through [`Rec`]'s `Deref`, so nothing is
//! recorded before a barrier asked for earlier. In a replay of capture
//! replay5 joining the calls that had no work between them saved 2.4 ms of
//! 79.5 (2,490 calls joined, 1,619 of them a texture's state change after
//! its upload followed by the next texture's aliasing).

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

use windows::Win32::Graphics::Direct3D12::*;

/// Whether barriers are held now (`VICTORY_JOIN_BARRIERS=1`, or its phase
/// of `VICTORY_AB`).
pub(crate) fn on() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_JOIN_BARRIERS") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_JOIN_BARRIERS").is_some_and(|v| v == "1"))
}

/// Whether a barrier call of nothing but unordered-access barriers on every
/// resource is left out when the list's last commands were barriers that
/// held one already (`VICTORY_UAV_BARRIER_ONCE=1`, or its phase of
/// `VICTORY_AB`; off by default). After PalmVI 8953888dd.
///
/// Each of the title's barrier groups becomes one such barrier, and so does
/// a barrier of its on a buffer (`crate::list`), several in a row with no
/// work between: the first already makes every earlier shader write finish
/// before what follows, and the next has nothing more to wait for. Any
/// other command recorded on the list ([`Rec`]'s `Deref`) ends the row. Off
/// until a run shows what the card and the recording make of it.
pub(crate) fn uav_once() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_UAV_BARRIER_ONCE") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_UAV_BARRIER_ONCE").is_some_and(|v| v == "1"))
}

/// An unordered-access barrier on every resource (none named).
fn global_uav(b: &D3D12_RESOURCE_BARRIER) -> bool {
    // SAFETY: the union's member is the one `Type` says; a pointer read.
    b.Type == D3D12_RESOURCE_BARRIER_TYPE_UAV && unsafe { b.Anonymous.UAV.pResource.is_none() }
}

/// Whether `barriers` add nothing after barriers that held an
/// unordered-access barrier on every resource, with no other command since
/// (`after_global`): they are all such barriers.
fn repeats_global_uav(after_global: bool, barriers: &[D3D12_RESOURCE_BARRIER]) -> bool {
    after_global && !barriers.is_empty() && barriers.iter().all(global_uav)
}

/// Since the last frame-rate line, with [`uav_once`]: barrier calls that
/// held an unordered-access barrier on every resource, and those of them
/// left out.
pub(crate) static UAV_ONCE: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Barriers held. Their resources are borrowed: the list's states hold the
/// textures (`HostList::transition`), and the title's barriers name what
/// the title keeps alive until the list has run.
struct Held(Vec<D3D12_RESOURCE_BARRIER>);

// SAFETY: borrowed interface pointers of free-threaded objects.
unsafe impl Send for Held {}

/// Since the last frame-rate line: barrier calls asked for while held, and
/// calls made of them.
pub(crate) static STATS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

pub(crate) struct Rec {
    list: ID3D12GraphicsCommandList,
    any: AtomicBool,
    held: Mutex<Held>,
    /// The list's last commands were barriers, one of them an
    /// unordered-access barrier on every resource ([`uav_once`]).
    after_global_uav: AtomicBool,
}

impl Rec {
    pub(crate) fn new(list: ID3D12GraphicsCommandList) -> Self {
        Self { list, any: AtomicBool::new(false), held: Mutex::new(Held(Vec::new())), after_global_uav: AtomicBool::new(false) }
    }

    /// Records `barriers` now, or holds them for the next use of the list.
    ///
    /// # Safety
    /// As `ResourceBarrier`'s: an open list, live resources.
    pub(crate) unsafe fn barriers(&self, barriers: &[D3D12_RESOURCE_BARRIER]) {
        if barriers.is_empty() {
            return;
        }
        if uav_once() && barriers.iter().any(global_uav) {
            UAV_ONCE[0].fetch_add(1, Relaxed);
            if repeats_global_uav(self.after_global_uav.load(Relaxed), barriers) {
                UAV_ONCE[1].fetch_add(1, Relaxed);
                return;
            }
            self.after_global_uav.store(true, Relaxed);
        }
        if !on() {
            // (What was held while it was on goes first.)
            if self.any.load(Relaxed) {
                self.flush();
            }
            // SAFETY: the caller's.
            unsafe { self.list.ResourceBarrier(barriers) };
            return;
        }
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        // A resource whose state changes again: what is held goes first, so
        // that no call holds two changes of one resource.
        if barriers.iter().filter_map(named).any(|r| held.0.iter().filter_map(named).any(|h| h == r)) {
            // SAFETY: as in `flush`.
            unsafe { self.list.ResourceBarrier(&held.0) };
            held.0.clear();
            STATS[1].fetch_add(1, Relaxed);
        }
        // SAFETY: a plain copy of the descriptions; nothing is released
        // through them (their pointers are borrowed, `ManuallyDrop`).
        held.0.extend(barriers.iter().map(|b| unsafe { std::ptr::read(b) }));
        STATS[0].fetch_add(1, Relaxed);
        self.any.store(true, Relaxed);
    }

    /// What is held goes to the list, in one call.
    fn flush(&self) {
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        self.any.store(false, Relaxed);
        if held.0.is_empty() {
            return;
        }
        // SAFETY: the barriers were asked for on this open list, in order;
        // their resources are alive (`Held`).
        unsafe { self.list.ResourceBarrier(&held.0) };
        held.0.clear();
        STATS[1].fetch_add(1, Relaxed);
    }
}

/// The resource a state change or aliasing names (its interface pointer).
fn named(b: &D3D12_RESOURCE_BARRIER) -> Option<usize> {
    // SAFETY: the union's member is the one `Type` says; a pointer read.
    unsafe {
        let p: Option<&ID3D12Resource> = match b.Type {
            D3D12_RESOURCE_BARRIER_TYPE_TRANSITION => b.Anonymous.Transition.pResource.as_ref(),
            D3D12_RESOURCE_BARRIER_TYPE_ALIASING => b.Anonymous.Aliasing.pResourceAfter.as_ref(),
            _ => None,
        };
        p.map(|r| windows::core::Interface::as_raw(r) as usize)
    }
}

impl std::ops::Deref for Rec {
    type Target = ID3D12GraphicsCommandList;

    /// The list, with what is held recorded first.
    fn deref(&self) -> &ID3D12GraphicsCommandList {
        if self.any.load(Relaxed) {
            self.flush();
        }
        // Another command follows the barriers: a later barrier on every
        // shader write has that command's writes to wait for.
        if self.after_global_uav.load(Relaxed) {
            self.after_global_uav.store(false, Relaxed);
        }
        &self.list
    }
}

/// For the frame-rate line.
pub(crate) fn line(frames: f64) {
    let [asked, made] = [0, 1].map(|i| STATS[i].swap(0, Relaxed) as f64 / frames);
    if asked > 0.0 {
        tracing::info!(
            "barrier calls per frame (VICTORY_JOIN_BARRIERS): {asked:.0} asked for, {made:.0} made of them"
        );
    }
    let [global, left_out] = [0, 1].map(|i| UAV_ONCE[i].swap(0, Relaxed) as f64 / frames);
    if global > 0.0 {
        tracing::info!(
            "barriers on every shader write per frame (VICTORY_UAV_BARRIER_ONCE=1): {global:.0} calls held one, {left_out:.0} of them left out (they repeated one with no command between)"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (`VICTORY_UAV_BARRIER_ONCE`) Only barrier calls of nothing but
    /// unordered-access barriers on every resource, right after one, are
    /// left out: the first is kept, and so is a call with a state change.
    /// (PalmVI 8953888dd's.)
    #[test]
    fn only_global_uav_barriers_right_after_one_are_left_out() {
        let uav = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER::default()),
            },
            ..Default::default()
        };
        let transition = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER::default()),
            },
            ..Default::default()
        };
        assert!(global_uav(&uav));
        assert!(!global_uav(&transition));
        // SAFETY: descriptions that name no resource, never dropped.
        let copy = |b: &D3D12_RESOURCE_BARRIER| unsafe { std::ptr::read(b) };
        assert!(repeats_global_uav(true, &[copy(&uav)]));
        assert!(repeats_global_uav(true, &[copy(&uav), copy(&uav)]));
        assert!(!repeats_global_uav(false, &[copy(&uav)]), "the first is kept");
        assert!(!repeats_global_uav(true, &[copy(&uav), copy(&transition)]), "a state change is kept");
        assert!(!repeats_global_uav(true, &[]));
    }

    /// The title's barrier on every shader write, as `crate::list` hands it
    /// over (its own layout, no resource named), is one.
    #[test]
    fn a_zeroed_uav_barrier_names_no_resource() {
        // SAFETY: all zeros is a valid description (null pointers, type 0).
        let mut barrier: D3D12_RESOURCE_BARRIER = unsafe { std::mem::zeroed() };
        assert!(!global_uav(&barrier), "a state change of nothing is not one");
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        assert!(global_uav(&barrier));
    }
}
