//! Fence-tracked LRU garbage collection and upload/staging buffer pooling.
//!
//! Keeps VRAM and system-RAM staging footprints bounded during long sessions on
//! 8 GB VRAM GPUs and 16-32 GB RAM PCs:
//! - Reuses completed `D3D12_HEAP_TYPE_UPLOAD` (and `WRITE_BACK` shared UAV)
//!   staging buffers across frames via [`UploadPoolCore`], gated on the exact
//!   `ID3D12Fence` domain(s) that signaled their work (graphics queue vs.
//!   `VICTORY_UPLOAD_AHEAD` copy queue).
//! - Evicts stale cached texture/buffer alias allocations once
//!   `GetCompletedValue() >= completion_fence` when unused for `> 120` frames
//!   (capped at [`ROUTINE_SWEEP_CAP`] per frame so `Present` never hitches), or
//!   oldest-first without cap whenever `CurrentUsage` exceeds
//!   `VICTORY_MEM_BUDGET_MB` (defaulting to 75% of the adapter's
//!   `DXGI_QUERY_VIDEO_MEMORY_INFO` local budget) down to the 90% hysteresis
//!   target.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::{ID3D12Fence, ID3D12Resource};
use windows::core::Interface;

use crate::host::Host;

/// Frames an idle cached allocation may stay while below the VRAM watermark.
pub(crate) const IDLE_FRAMES: u64 = 120;

/// Maximum resources evicted per frame during routine (`> 120` frame) sweeps so
/// `Present` never hitches. Bypassed only when `CurrentUsage` exceeds the
/// configured VRAM budget watermark.
pub(crate) const ROUTINE_SWEEP_CAP: usize = 64;

/// Minimum bucket granularity (64 KiB, matching D3D12 buffer alignment).
const MIN_BUCKET: u64 = 64 << 10;

/// Threshold above which upload buckets step in 1 MiB multiples instead of
/// powers of two, avoiding 2x over-allocation on large staging buffers.
const POWER_OF_TWO_LIMIT: u64 = 16 << 20;

/// Default upload/staging pool ceiling on <= 8 GiB VRAM adapters.
const DEFAULT_UPLOAD_POOL_MIN: u64 = 64 << 20;
const DEFAULT_UPLOAD_POOL_MAX: u64 = 256 << 20;

/// Which D3D12 heap backing a pooled CPU-mapped buffer uses. Kept strictly
/// separate so `ALLOW_UNORDERED_ACCESS` buffers (`card_steps`) never mix with
/// `D3D12_HEAP_TYPE_UPLOAD` (`GENERIC_READ`) buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum HeapKind {
    Upload,
    SharedUav,
}

/// Abstracts an `ID3D12Fence` (or mock fence in unit tests) so [`UploadPoolCore`]
/// always queries the exact fence object that signaled a buffer's GPU work.
pub(crate) trait FenceProbe: Clone + Send {
    fn domain_id(&self) -> usize;
    fn completed_value(&self) -> u64;
}

impl FenceProbe for ID3D12Fence {
    fn domain_id(&self) -> usize {
        self.as_raw() as usize
    }

    fn completed_value(&self) -> u64 {
        // SAFETY: `ID3D12Fence::GetCompletedValue` is free-threaded on live COM fences.
        unsafe { self.GetCompletedValue() }
    }
}

/// Computes the VRAM watermark in bytes from an optional `VICTORY_MEM_BUDGET_MB`
/// override or 75% of the adapter's `DXGI_QUERY_VIDEO_MEMORY_INFO` local budget.
pub(crate) fn budget_bytes_from(
    override_mb: Option<u64>,
    dxgi_local_budget: Option<u64>,
) -> Option<u64> {
    if let Some(mb) = override_mb.filter(|&mb| mb > 0) {
        return mb.checked_shl(20);
    }
    dxgi_local_budget
        .filter(|&b| b > 0)
        .map(|b| (u128::from(b) * 75 / 100) as u64)
}

/// Hysteresis target (90% of the budget watermark) where pressure eviction stops.
pub(crate) fn hysteresis_target_bytes(budget_bytes: u64) -> u64 {
    (u128::from(budget_bytes) * 90 / 100) as u64
}

fn env_mem_budget_mb() -> Option<u64> {
    static MB: OnceLock<Option<u64>> = OnceLock::new();
    *MB.get_or_init(|| {
        std::env::var("VICTORY_MEM_BUDGET_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|&v| v > 0)
    })
}

/// Effective VRAM watermark in bytes for `video_memory` (`[local_budget, local_used, ...]`).
pub(crate) fn vram_budget_bytes(video_memory: Option<[u64; 4]>) -> Option<u64> {
    budget_bytes_from(env_mem_budget_mb(), video_memory.map(|[budget, ..]| budget))
}

/// Effective upload/staging pool budget in bytes (`VICTORY_UPLOAD_BUDGET_MB`, else
/// 1/16th of the VRAM watermark clamped to `[64 MiB, 256 MiB]`).
pub(crate) fn upload_budget_bytes(video_memory: Option<[u64; 4]>) -> u64 {
    static OVERRIDE: OnceLock<Option<u64>> = OnceLock::new();
    if let Some(bytes) = *OVERRIDE.get_or_init(|| {
        std::env::var("VICTORY_UPLOAD_BUDGET_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|&v| v > 0)
            .and_then(|mb| mb.checked_shl(20))
    }) {
        return bytes;
    }
    match vram_budget_bytes(video_memory) {
        Some(vram) => (vram / 16).clamp(DEFAULT_UPLOAD_POOL_MIN, DEFAULT_UPLOAD_POOL_MAX),
        None => 128 << 20,
    }
}

/// Rounds `requested` up to its pool size class.
pub(crate) fn bucket_size(requested: u64) -> u64 {
    let size = requested.max(MIN_BUCKET);
    if size <= POWER_OF_TWO_LIMIT {
        size.next_power_of_two()
    } else {
        size.next_multiple_of(1 << 20)
    }
}

/// One pooled staging/upload allocation and the fence gates that must complete
/// before the CPU or another queue may overwrite or free it.
struct PooledUpload<B, F: FenceProbe> {
    buffer: B,
    ptr: usize,
    size: u64,
    gates: Vec<(F, u64)>,
    last_used_frame: u64,
}

impl<B, F: FenceProbe> PooledUpload<B, F> {
    /// True only when every distinct fence domain recorded on this buffer has
    /// reached or passed its signaled value.
    fn is_completed(&self) -> bool {
        self.gates
            .iter()
            .all(|(fence, value)| fence.completed_value() >= *value)
    }

    fn matches_domain(&self, required_domain: Option<usize>) -> bool {
        match required_domain {
            Some(dom) => {
                self.gates.is_empty() || self.gates.iter().all(|(f, _)| f.domain_id() == dom)
            }
            None => true,
        }
    }
}

/// Thread-safe core of the fence-tracked upload/staging buffer pool.
pub(crate) struct UploadPoolCore<B, F: FenceProbe> {
    buckets: BTreeMap<(HeapKind, u64), Vec<PooledUpload<B, F>>>,
    total_bytes: u64,
    pub(crate) hits: u64,
    pub(crate) misses: u64,
    pub(crate) evicted: u64,
}

impl<B, F: FenceProbe> UploadPoolCore<B, F> {
    pub(crate) const fn new() -> Self {
        Self {
            buckets: BTreeMap::new(),
            total_bytes: 0,
            hits: 0,
            misses: 0,
            evicted: 0,
        }
    }

    #[cfg(test)]
    pub(crate) fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Acquires a completed buffer for `(kind, requested_size)`. If
    /// `required_domain` is `Some(id)`, only buffers whose fence gates belong to
    /// that fence domain (or unsubmitted buffers) are considered, and every gate
    /// is verified against its own [`FenceProbe::completed_value`].
    pub(crate) fn acquire(
        &mut self,
        kind: HeapKind,
        requested_size: u64,
        required_domain: Option<usize>,
        current_frame: u64,
    ) -> Option<(B, usize, u64)> {
        let size = bucket_size(requested_size);
        let list = self.buckets.get_mut(&(kind, size))?;
        // Prefer the most recently used completed entry (hot cache lines, LIFO) while
        // leaving older entries at the front to age out past `IDLE_FRAMES`.
        let idx = list
            .iter()
            .rposition(|entry| entry.matches_domain(required_domain) && entry.is_completed())?;
        let entry = list.remove(idx);
        if list.is_empty() {
            self.buckets.remove(&(kind, size));
        }
        let _ = current_frame;
        self.total_bytes = self.total_bytes.saturating_sub(entry.size);
        self.hits += 1;
        Some((entry.buffer, entry.ptr, entry.size))
    }

    pub(crate) fn note_miss(&mut self) {
        self.misses += 1;
    }

    /// Returns a buffer to the pool gated by `gates` (one `(Fence, value)` per
    /// queue that touched it). Immediately trims completed LRU buffers down to
    /// the 90% hysteresis target if `total_bytes > max_bytes`. Caller drops the
    /// returned `Vec<B>` outside the pool lock.
    pub(crate) fn release(
        &mut self,
        kind: HeapKind,
        buffer: B,
        ptr: usize,
        size: u64,
        gates: Vec<(F, u64)>,
        current_frame: u64,
        max_bytes: u64,
    ) -> Vec<B> {
        let mut deduped: Vec<(F, u64)> = Vec::with_capacity(gates.len());
        for (fence, val) in gates {
            if val == 0 {
                continue;
            }
            let dom = fence.domain_id();
            if let Some((_, existing)) = deduped.iter_mut().find(|(f, _)| f.domain_id() == dom) {
                *existing = (*existing).max(val);
            } else {
                deduped.push((fence, val));
            }
        }
        self.total_bytes = self.total_bytes.saturating_add(size);
        self.buckets
            .entry((kind, size))
            .or_default()
            .push(PooledUpload {
                buffer,
                ptr,
                size,
                gates: deduped,
                last_used_frame: current_frame,
            });
        if self.total_bytes > max_bytes {
            self.evict_to_target(hysteresis_target_bytes(max_bytes))
        } else {
            Vec::new()
        }
    }

    /// Sweeps completed buffers unused for `> idle_frames` (up to `cap`) and, if
    /// `total_bytes > max_bytes`, evicts completed buffers in strict LRU order
    /// down to the 90% hysteresis target.
    pub(crate) fn sweep(
        &mut self,
        current_frame: u64,
        idle_frames: u64,
        max_bytes: u64,
        cap: usize,
    ) -> Vec<B> {
        let mut dropped = Vec::new();
        if self.total_bytes > max_bytes {
            dropped.extend(self.evict_to_target(hysteresis_target_bytes(max_bytes)));
        }
        if cap == 0 || self.buckets.is_empty() {
            return dropped;
        }
        let mut candidates: Vec<(u64, (HeapKind, u64), usize)> = Vec::new();
        for (&key, list) in &self.buckets {
            for (idx, entry) in list.iter().enumerate() {
                if current_frame.saturating_sub(entry.last_used_frame) > idle_frames
                    && entry.is_completed()
                {
                    candidates.push((entry.last_used_frame, key, idx));
                }
            }
        }
        if candidates.is_empty() {
            return dropped;
        }
        candidates.sort_unstable_by_key(|&(frame, _, _)| frame);
        candidates.truncate(cap);
        // Remove higher indices within the same bucket first so earlier indices stay valid.
        candidates.sort_unstable_by(|a, b| a.1.cmp(&b.1).then_with(|| b.2.cmp(&a.2)));
        for (_, key, idx) in candidates {
            if let Some(list) = self.buckets.get_mut(&key)
                && idx < list.len()
            {
                let removed = list.remove(idx);
                self.total_bytes = self.total_bytes.saturating_sub(removed.size);
                self.evicted += 1;
                dropped.push(removed.buffer);
                if list.is_empty() {
                    self.buckets.remove(&key);
                }
            }
        }
        dropped
    }

    fn evict_to_target(&mut self, target_bytes: u64) -> Vec<B> {
        let mut candidates: Vec<(u64, (HeapKind, u64), usize, u64)> = Vec::new();
        for (&key, list) in &self.buckets {
            for (idx, entry) in list.iter().enumerate() {
                if entry.is_completed() {
                    candidates.push((entry.last_used_frame, key, idx, entry.size));
                }
            }
        }
        // Strict LRU: oldest `last_used_frame` first.
        candidates.sort_unstable_by_key(|&(frame, _, _, _)| frame);
        let mut to_remove: Vec<((HeapKind, u64), usize)> = Vec::new();
        let mut projected = self.total_bytes;
        for (_, key, idx, size) in candidates {
            if projected <= target_bytes {
                break;
            }
            projected = projected.saturating_sub(size);
            to_remove.push((key, idx));
        }
        to_remove.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1)));
        let mut dropped = Vec::with_capacity(to_remove.len());
        for (key, idx) in to_remove {
            if let Some(list) = self.buckets.get_mut(&key)
                && idx < list.len()
            {
                let removed = list.remove(idx);
                self.total_bytes = self.total_bytes.saturating_sub(removed.size);
                self.evicted += 1;
                dropped.push(removed.buffer);
                if list.is_empty() {
                    self.buckets.remove(&key);
                }
            }
        }
        dropped
    }
}

static UPLOAD_POOL: Mutex<UploadPoolCore<ID3D12Resource, ID3D12Fence>> =
    Mutex::new(UploadPoolCore::new());
static GC_STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

fn pool_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_UPLOAD_POOL").is_none_or(|v| v != "0"))
}

/// Acquires a persistently mapped buffer of at least `size` bytes (`(buffer, ptr, alloc_size)`),
/// reusing a fence-completed buffer from [`UPLOAD_POOL`] when available.
pub(crate) fn take_upload(
    host: &Host,
    size: u64,
    kind: HeapKind,
    required_domain: Option<usize>,
) -> Option<(ID3D12Resource, *mut u8, u64)> {
    let alloc_size = bucket_size(size);
    let frame = crate::frame::current();
    if pool_on() {
        let mut pool = UPLOAD_POOL.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((buffer, ptr, actual)) = pool.acquire(kind, alloc_size, required_domain, frame)
        {
            return Some((buffer, ptr as *mut u8, actual));
        }
        pool.note_miss();
    }
    let (buffer, ptr) = match kind {
        HeapKind::Upload => host.upload_buffer(alloc_size)?,
        HeapKind::SharedUav => host.shared_buffer(alloc_size)?,
    };
    Some((buffer, ptr, alloc_size))
}

/// Recycles a pooled buffer gated on one or more `(ID3D12Fence, value)` pairs.
pub(crate) fn give_upload_multi(
    kind: HeapKind,
    buffer: ID3D12Resource,
    ptr: *mut u8,
    alloc_size: u64,
    gates: Vec<(ID3D12Fence, u64)>,
) {
    if !pool_on() || ptr.is_null() || alloc_size == 0 {
        return;
    }
    let max_bytes = upload_budget_bytes(crate::host::host().and_then(|h| h.video_memory()));
    let frame = crate::frame::current();
    let evicted = {
        let mut pool = UPLOAD_POOL.lock().unwrap_or_else(|e| e.into_inner());
        pool.release(
            kind,
            buffer,
            ptr as usize,
            alloc_size,
            gates,
            frame,
            max_bytes,
        )
    };
    drop(evicted);
}

/// Frame-boundary GC hook invoked at the end of `Host::present`.
///
/// - Routine (`<= watermark`): evicts at most [`ROUTINE_SWEEP_CAP`] (`64`)
///   resources unused for `> 120` frames whose GPU fence has completed.
/// - Pressure (`CurrentUsage > watermark`): bypasses the 64-resource cap,
///   trims spare pools, and evicts oldest-first until reaching the 90%
///   hysteresis target.
pub(crate) fn on_present(host: &Host, frame: u64) {
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_GC").is_none_or(|v| v != "0")) {
        return;
    }
    // SAFETY: `Host::fence` is a valid free-threaded D3D12 fence.
    let completed = unsafe { host.fence.GetCompletedValue() };
    let video_mem = host.video_memory();
    let budget = vram_budget_bytes(video_mem);
    let upload_max = upload_budget_bytes(video_mem);

    crate::pool::room_with_budget(video_mem, budget);

    let over_budget_target = match (video_mem, budget) {
        (Some([_, local_used, ..]), Some(limit)) if local_used > limit => {
            let target = hysteresis_target_bytes(limit);
            Some(local_used.saturating_sub(target))
        }
        _ => None,
    };

    // 1. Sweep `UPLOAD_POOL` (drops outside the lock).
    let upload_cap = if over_budget_target.is_some() {
        usize::MAX
    } else {
        ROUTINE_SWEEP_CAP
    };
    let dropped_uploads = {
        let mut pool = UPLOAD_POOL.lock().unwrap_or_else(|e| e.into_inner());
        pool.sweep(frame, IDLE_FRAMES, upload_max, upload_cap)
    };
    let upload_evicted = dropped_uploads.len();
    drop(dropped_uploads);

    if let Some(mut needed) = over_budget_target {
        // Under VRAM pressure: trim empty 64 MiB texture/staging heaps and scratch chunks first.
        let pool_freed = crate::pool::trim_empty_blocks();
        let scratch_freed = crate::host::trim_spare_scratch(32 << 20);
        needed = needed.saturating_sub(pool_freed + scratch_freed);

        let mut freed_total = pool_freed + scratch_freed;
        let mut count_total = upload_evicted as u64;

        if needed > 0 {
            let (n_tex, b_tex) =
                crate::resource::evict_lru(completed, frame, Some(needed), usize::MAX);
            count_total += n_tex as u64;
            freed_total += b_tex;
            needed = needed.saturating_sub(b_tex);
        }
        if needed > 0 {
            let (n_sh, b_sh) = crate::shadow::evict_lru(completed, frame, Some(needed), usize::MAX);
            count_total += n_sh as u64;
            freed_total += b_sh;
        }
        let (n_st, b_st) = crate::texture::evict_lru(completed, frame, true, ROUTINE_SWEEP_CAP);
        count_total += n_st as u64;
        freed_total += b_st;

        GC_STATS[0].fetch_add(1, Ordering::Relaxed);
        GC_STATS[1].fetch_add(count_total, Ordering::Relaxed);
        GC_STATS[2].fetch_add(freed_total, Ordering::Relaxed);
        let runs = GC_STATS[0].load(Ordering::Relaxed);
        if runs == 1 || runs.is_multiple_of(60) {
            tracing::info!(
                frame,
                evicted = count_total,
                freed_mb = freed_total >> 20,
                budget_mb = budget.unwrap_or(0) >> 20,
                "gc: watermark sweep trimmed cached GPU/upload allocations to 90% hysteresis target"
            );
        }
        return;
    }

    // Routine idle sweep (> 120 frames): strictly capped at `ROUTINE_SWEEP_CAP` (64) total per frame.
    let mut left = ROUTINE_SWEEP_CAP.saturating_sub(upload_evicted);
    if left == 0 {
        return;
    }
    let (n_st, _) = crate::texture::evict_lru(completed, frame, false, left);
    left = left.saturating_sub(n_st);
    if left == 0 {
        return;
    }
    let (n_tex, b_tex) = crate::resource::evict_lru(completed, frame, None, left);
    left = left.saturating_sub(n_tex);
    if left > 0 {
        let (n_sh, b_sh) = crate::shadow::evict_lru(completed, frame, None, left);
        GC_STATS[3].fetch_add((n_tex + n_sh) as u64, Ordering::Relaxed);
        GC_STATS[2].fetch_add(b_tex + b_sh, Ordering::Relaxed);
    } else {
        GC_STATS[3].fetch_add(n_tex as u64, Ordering::Relaxed);
        GC_STATS[2].fetch_add(b_tex, Ordering::Relaxed);
    }
}

/// Orders `(id, last_used_frame, bytes, completed)` candidates for eviction:
/// - In routine mode (`over_budget_needed = None`), selects only completed items
///   with `current_frame - last_used_frame > IDLE_FRAMES`, oldest first, up to `cap`.
/// - In pressure mode (`over_budget_needed = Some(bytes_to_free)`), selects
///   completed items in strict LRU order regardless of frame age, stopping as
///   soon as accumulated bytes reach `bytes_to_free`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn select_lru_victims(
    items: &[(usize, u64, u64, bool)],
    current_frame: u64,
    over_budget_needed: Option<u64>,
    cap: usize,
) -> Vec<usize> {
    let mut eligible: Vec<(u64, usize, u64)> = items
        .iter()
        .copied()
        .filter(|&(_, last_frame, _, fence_done)| {
            fence_done
                && (over_budget_needed.is_some()
                    || current_frame.saturating_sub(last_frame) > IDLE_FRAMES)
        })
        .map(|(id, last_frame, bytes, _)| (last_frame, id, bytes))
        .collect();
    eligible.sort_unstable_by_key(|&(last_frame, id, _)| (last_frame, id));

    let mut out = Vec::new();
    match over_budget_needed {
        Some(needed) => {
            let mut freed = 0u64;
            for (_, id, bytes) in eligible {
                if freed >= needed || out.len() >= cap {
                    break;
                }
                out.push(id);
                freed = freed.saturating_add(bytes);
            }
        }
        None => {
            for (_, id, _) in eligible.into_iter().take(cap) {
                out.push(id);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[derive(Clone)]
    struct MockFence {
        domain: usize,
        completed: Arc<AtomicU64>,
    }

    impl MockFence {
        fn new(domain: usize, initial: u64) -> Self {
            Self {
                domain,
                completed: Arc::new(AtomicU64::new(initial)),
            }
        }

        fn advance_to(&self, value: u64) {
            self.completed.store(value, Ordering::SeqCst);
        }
    }

    impl FenceProbe for MockFence {
        fn domain_id(&self) -> usize {
            self.domain
        }

        fn completed_value(&self) -> u64 {
            self.completed.load(Ordering::SeqCst)
        }
    }

    /// Test 1: Fence-gated `UploadPoolCore` reuse, cross-queue fence-domain
    /// isolation (`VICTORY_UPLOAD_AHEAD` copy fence vs. graphics fence), `> 120`
    /// frame idle sweep with 64-cap, and budget-watermark hysteresis eviction.
    #[test]
    fn upload_pool_enforces_fence_domains_idle_frames_and_budget() {
        let gfx_fence = MockFence::new(1, 10);
        let copy_fence = MockFence::new(2, 3);
        let mut pool: UploadPoolCore<u32, MockFence> = UploadPoolCore::new();
        let max_bytes = 10 * MIN_BUCKET;

        // Buffer 101 was used on the async copy queue (`copy_fence` value 5, still at 3).
        // Even though `gfx_fence` is already at 10 (>= 5), `UploadPoolCore` MUST NOT
        // recycle buffer 101 until `copy_fence` reaches 5, nor hand a copy-domain
        // buffer to a strict graphics-domain query while in flight.
        let evicted = pool.release(
            HeapKind::Upload,
            101,
            0x1000,
            MIN_BUCKET,
            vec![(copy_fence.clone(), 5)],
            1,
            max_bytes,
        );
        assert!(evicted.is_empty());
        assert!(
            pool.acquire(HeapKind::Upload, MIN_BUCKET, None, 2)
                .is_none()
        );
        assert!(
            pool.acquire(HeapKind::Upload, MIN_BUCKET, Some(1), 2)
                .is_none()
        );

        // Advance `copy_fence` to 5: domain-matched and general acquisitions now succeed.
        copy_fence.advance_to(5);
        assert!(
            pool.acquire(HeapKind::Upload, MIN_BUCKET, Some(1), 2)
                .is_none()
        );
        let (buf, ptr, sz) = pool
            .acquire(HeapKind::Upload, 4096, Some(2), 2)
            .expect("copy fence completed");
        assert_eq!((buf, ptr, sz), (101, 0x1000, MIN_BUCKET));
        assert_eq!(pool.total_bytes(), 0);

        // Multi-fence safety: touched by both graphics (fence 12) and copy (fence 6).
        let _ = pool.release(
            HeapKind::Upload,
            102,
            0x2000,
            MIN_BUCKET,
            vec![(gfx_fence.clone(), 12), (copy_fence.clone(), 6)],
            5,
            max_bytes,
        );
        gfx_fence.advance_to(12);
        assert!(
            pool.acquire(HeapKind::Upload, MIN_BUCKET, None, 6)
                .is_none(),
            "copy fence still at 5"
        );
        copy_fence.advance_to(6);
        let (buf2, ..) = pool
            .acquire(HeapKind::Upload, MIN_BUCKET, None, 6)
            .expect("both fences completed");
        assert_eq!(buf2, 102);

        // Routine >120-frame sweep + 64-item cap: insert 70 completed buffers at frame 10.
        let big_cap = 200 * MIN_BUCKET;
        for id in 0..70u32 {
            let _ = pool.release(
                HeapKind::Upload,
                id,
                0x4000 + id as usize,
                MIN_BUCKET,
                vec![(gfx_fence.clone(), 12)],
                10,
                big_cap,
            );
        }
        // At frame 130 (120 frames elapsed, not > 120): none evicted.
        assert!(
            pool.sweep(130, IDLE_FRAMES, big_cap, ROUTINE_SWEEP_CAP)
                .is_empty()
        );
        // At frame 131 (121 frames elapsed): capped at 64 evictions, leaving 6.
        let swept = pool.sweep(131, IDLE_FRAMES, big_cap, ROUTINE_SWEEP_CAP);
        assert_eq!(swept.len(), ROUTINE_SWEEP_CAP);
        assert_eq!(pool.total_bytes(), 6 * MIN_BUCKET);
        let rest = pool.sweep(132, IDLE_FRAMES, big_cap, ROUTINE_SWEEP_CAP);
        assert_eq!(rest.len(), 6);
        assert_eq!(pool.total_bytes(), 0);
    }

    /// Test 2: `VICTORY_MEM_BUDGET_MB` override vs. 75% DXGI local budget
    /// default, 90% hysteresis target, and strict LRU victim ordering.
    #[test]
    fn mem_budget_default_75_percent_and_lru_hysteresis_selection() {
        let eight_gib = 8u64 << 30;
        // Default: 75% of DXGI local budget (6 GiB for an 8 GiB budget).
        assert_eq!(budget_bytes_from(None, Some(eight_gib)), Some(6 << 30));
        // Explicit `VICTORY_MEM_BUDGET_MB` overrides DXGI default.
        assert_eq!(
            budget_bytes_from(Some(4096), Some(eight_gib)),
            Some(4096 << 20)
        );
        // 90% hysteresis target on 1,000 MiB = 900 MiB.
        assert_eq!(hysteresis_target_bytes(1000 << 20), 900 << 20);

        // Candidates: (id, last_used_frame, bytes, fence_completed)
        let mb = 1u64 << 20;
        let items = vec![
            (1, 200, 100 * mb, true), // recent (50 frames ago)
            (2, 50, 120 * mb, false), // old, but GPU fence NOT completed -> must never evict!
            (3, 80, 90 * mb, true),   // idle (170 frames ago)
            (4, 40, 110 * mb, true),  // oldest completed (210 frames ago)
            (5, 190, 150 * mb, true), // recent (60 frames ago), eligible only under budget pressure
        ];

        // Routine sweep at frame 250 (> 120 frames): only 4 (frame 40) and 3 (frame 80), in LRU order.
        let routine = select_lru_victims(&items, 250, None, ROUTINE_SWEEP_CAP);
        assert_eq!(routine, vec![4, 3]);

        // Over-watermark sweep needing 250 MiB freed to reach 90% hysteresis target:
        // picks 4 (110 MiB) + 3 (90 MiB) + 5 (150 MiB) = 350 MiB >= 250 MiB, stops before 1, skips uncompleted 2!
        let pressure = select_lru_victims(&items, 250, Some(250 * mb), usize::MAX);
        assert_eq!(pressure, vec![4, 3, 5]);
    }

    /// Test 3: Live D3D12 descriptor unpinning (`evict_resource` + `make_null` +
    /// heap drop cleanup), verifying COM refcounts drop cleanly and
    /// `device.GetDeviceRemovedReason().is_ok()`.
    #[test]
    fn live_d3d12_evict_resource_nulls_bindless_slots_without_device_removal() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let Some(heap) = crate::descriptor::host::HostHeap::new(false, 8) else {
            return;
        };
        let Some((buf, _ptr)) = host.upload_buffer(65_536) else {
            return;
        };

        // Baseline: only `buf` holds a COM reference.
        assert_eq!(crate::host::references(&buf), 1);
        assert_eq!(crate::descriptor::host::slot_ref_count(&buf), 0);

        // Bind `buf` into two shader-visible descriptor slots (slot 2 and dynamic slot 9).
        // SAFETY: valid live buffer.
        let gpu = unsafe { buf.GetGPUVirtualAddress() };
        heap.constant_at(2, &buf, gpu, 256);
        heap.raw_at(9, &buf, 0, 256);
        heap.pin_for_test(9, buf.clone());

        assert_eq!(crate::descriptor::host::slot_ref_count(&buf), 2);
        assert_eq!(crate::host::references(&buf), 3);

        // Evict `buf`: both slots are nulled (including any raw twin), `RESOURCE_SLOTS`
        // lock is dropped before `kept` locks (no deadlock), and `HostHeap::kept`
        // releases its references into retirement.
        let cleared = crate::descriptor::host::evict_resource(&buf);
        assert_eq!(cleared, 2);
        assert_eq!(crate::descriptor::host::slot_ref_count(&buf), 0);
        assert!(heap.is_slot_empty_for_test(2));
        assert!(heap.is_slot_empty_for_test(9));

        // Dropping `HostHeap` deterministically purges any remaining slot entries (no UAF).
        let other = host.upload_buffer(65_536).map(|(b, _)| b).unwrap();
        // SAFETY: valid live buffer.
        let other_gpu = unsafe { other.GetGPUVirtualAddress() };
        heap.constant_at(3, &other, other_gpu, 256);
        assert_eq!(crate::descriptor::host::slot_ref_count(&other), 1);
        drop(heap);
        assert_eq!(crate::descriptor::host::slot_ref_count(&other), 0);

        // SAFETY: plain device health check.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }
}
