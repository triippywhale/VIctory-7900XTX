//! Diagnostics meter: how many nodes the title's rays visit
//! (`VICTORY_BVH_WALK=1`, diagnostics builds; after Op-lite/victory-scarlett
//! c45b43e, `host/walk.rs`). The translator makes every ray-tracing node test of
//! a compute program add to its thread's dword of a buffer of ours
//! (`victory_shader::emit::SLOT_WALK`: 1 per box node, 0x10000 per triangle).
//! After each dispatch of such a program the list copies the buffer to a
//! read-back buffer of its own and clears it from a buffer of zeros; the copy is
//! read at the list's reset (every submission of it has run). Per program and
//! frame line: threads that tested a node, node tests per such thread (mean,
//! median, p99, max) and the share at or above 1,000 and 10,000 tests; the
//! children a box test lets the ray into on average (a tight structure keeps
//! it near 1-2; loose or overlapping boxes push it to 3-4); and the threads
//! that read a node off a page the host has no copy of (it reads zeros: a
//! broken walk), overall and among those ended by the loop limit.
//!
//! What it costs: one atomic per node test, two 32 MiB copies per metered
//! dispatch, the barriers around them. What it changes: nothing the title's
//! programs read or write. Not exact: threads whose x + 4096 y fall on the same
//! slot (more than 4M threads, or rows wider than 4096) add up; dispatches of
//! two lists that run at once on two queues mix.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

use victory_shader::emit::WALK_SLOTS;

/// Two dwords a thread: node tests, then children entered and tests of
/// nodes on pages with no copy.
const BYTES: u64 = WALK_SLOTS as u64 * 8;

/// `VICTORY_BVH_WALK=1` (the translator reads the same switch).
pub(crate) fn on() -> bool {
    victory_shader::emit::bvh_walk()
}

/// The buffer shaders count in, and the buffer of zeros that clears it.
struct Buffers {
    counts: ID3D12Resource,
    zeros: ID3D12Resource,
}

// SAFETY: free-threaded objects.
unsafe impl Send for Buffers {}
// SAFETY: as above.
unsafe impl Sync for Buffers {}

fn buffer(device: &ID3D12Device, heap: D3D12_HEAP_TYPE, uav: bool, state: D3D12_RESOURCE_STATES) -> Option<ID3D12Resource> {
    let props = D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: BYTES,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: if uav { D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS } else { D3D12_RESOURCE_FLAG_NONE },
        ..Default::default()
    };
    let mut made: Option<ID3D12Resource> = None;
    // SAFETY: plain creation; committed buffers start zeroed.
    unsafe { device.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut made) }.ok()?;
    made
}

fn buffers() -> Option<&'static Buffers> {
    static MADE: OnceLock<Option<Buffers>> = OnceLock::new();
    MADE.get_or_init(|| {
        let device = &super::host()?.device;
        let made = Buffers {
            counts: buffer(device, D3D12_HEAP_TYPE_DEFAULT, true, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)?,
            zeros: buffer(device, D3D12_HEAP_TYPE_DEFAULT, false, D3D12_RESOURCE_STATE_COMMON)?,
        };
        tracing::info!("node-test meter on (VICTORY_BVH_WALK=1): a count per ray of the nodes it visits; the run is slower");
        Some(made)
    })
    .as_ref()
}

/// Where the shaders count (its GPU address), for the root UAV.
pub(crate) fn address() -> Option<u64> {
    // SAFETY: a live buffer.
    buffers().map(|b| unsafe { b.counts.GetGPUVirtualAddress() })
}

/// Read-back buffers no list holds.
static SPARE: Mutex<Vec<ID3D12Resource>> = Mutex::new(Vec::new());

/// What a list copied, to read at its reset: the program and its read-back.
pub(crate) type Taken = Mutex<Vec<(Arc<str>, ID3D12Resource)>>;

/// A dispatch of a program that counts (`name`) was recorded into `list`:
/// the counts go to a read-back buffer and are cleared.
pub(crate) fn after(taken: &Taken, list: &ID3D12GraphicsCommandList, name: &str) {
    let name: Arc<str> = Arc::from(name);
    let Some(b) = buffers() else { return };
    let back = SPARE.lock().unwrap_or_else(|e| e.into_inner()).pop().or_else(|| {
        let device = &super::host()?.device;
        buffer(device, D3D12_HEAP_TYPE_READBACK, false, D3D12_RESOURCE_STATE_COPY_DEST)
    });
    let Some(back) = back else { return };
    let transition = |before, after| D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: borrowed for the call; no reference is taken.
                pResource: unsafe { std::mem::transmute_copy(&b.counts) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    };
    // SAFETY: an open list; our buffers, in the states named.
    unsafe {
        list.ResourceBarrier(&[transition(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        list.CopyResource(&back, &b.counts);
        list.ResourceBarrier(&[transition(D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST)]);
        list.CopyResource(&b.counts, &b.zeros);
        list.ResourceBarrier(&[transition(D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
    }
    taken.lock().unwrap_or_else(|e| e.into_inner()).push((name, back));
}

/// Per program since the last line: dispatches, threads that tested a node,
/// and how many threads tested how many nodes (box tests, triangle tests).
#[derive(Default)]
struct Sums {
    dispatches: u64,
    threads: u64,
    boxes: u64,
    triangles: u64,
    /// Threads by tests (box + triangle), in buckets of powers of two.
    buckets: [u64; 24],
    most: u64,
    /// Threads with [`CUT`] tests or more.
    cut: u64,
    /// Children entered, summed over the box tests.
    entered: u64,
    /// Tests of nodes on a page with no copy (they read zeros), and the
    /// threads that made one.
    lost: u64,
    lost_threads: u64,
    /// Of the threads at [`CUT`]: those that made such a test.
    cut_lost: u64,
    /// Children entered and box tests of the threads at [`CUT`].
    cut_entered: u64,
    cut_boxes: u64,
}

impl Sums {
    /// One dispatch's read-back: [`WALK_SLOTS`] dwords of node tests (box
    /// tests low, triangle tests high), then as many of children entered
    /// (low) and tests of nodes with no copy (high).
    fn add(&mut self, counts: &[u32]) {
        let (tests, detail) = counts.split_at(WALK_SLOTS as usize);
        self.dispatches += 1;
        for (&c, &d) in tests.iter().zip(detail) {
            if c == 0 {
                continue;
            }
            let (boxes, triangles) = (u64::from(c & 0xFFFF), u64::from(c >> 16));
            let (entered, lost) = (u64::from(d & 0xFFFF), u64::from(d >> 16));
            let n = boxes + triangles;
            self.threads += 1;
            self.boxes += boxes;
            self.triangles += triangles;
            self.entered += entered;
            self.lost += lost;
            self.lost_threads += u64::from(lost > 0);
            self.buckets[(64 - n.leading_zeros() as usize).min(23)] += 1;
            self.most = self.most.max(n);
            if n >= CUT {
                self.cut += 1;
                self.cut_lost += u64::from(lost > 0);
                self.cut_entered += entered;
                self.cut_boxes += boxes;
            }
        }
    }
}

/// Tests from which a thread counts as ended by the translation's loop limit
/// (4,096 rounds in compute, a test a round at most; the walkers' longest
/// threads make 4,085-4,089, run rtwalk-B).
const CUT: u64 = 4000;

static SUMS: Mutex<Option<HashMap<Arc<str>, Sums>>> = Mutex::new(None);

/// Every submission of the list has run: what it copied is read.
pub(crate) fn reset(taken: &Taken) {
    let list = std::mem::take(&mut *taken.lock().unwrap_or_else(|e| e.into_inner()));
    for (name, back) in list {
        let mut data = std::ptr::null_mut();
        // SAFETY: a read-back buffer the finished list copied into.
        if unsafe { back.Map(0, None, Some(&mut data)) }.is_ok() {
            // SAFETY: the mapping holds 2 * WALK_SLOTS dwords.
            let counts = unsafe { std::slice::from_raw_parts(data.cast::<u32>(), 2 * WALK_SLOTS as usize) };
            let mut sums = SUMS.lock().unwrap_or_else(|e| e.into_inner());
            let s = sums.get_or_insert_with(HashMap::new).entry(name).or_default();
            s.add(counts);
            // SAFETY: mapped above.
            unsafe { back.Unmap(0, None) };
        }
        SPARE.lock().unwrap_or_else(|e| e.into_inner()).push(back);
    }
}

/// For the frame lines.
pub(crate) fn line(frames: f64) {
    if !on() {
        return;
    }
    let Some(sums) = SUMS.lock().unwrap_or_else(|e| e.into_inner()).take() else { return };
    let frames = frames.max(1.0);
    let mut sums: Vec<_> = sums.into_iter().collect();
    sums.sort_by_key(|(_, s)| std::cmp::Reverse(s.boxes + s.triangles));
    for (name, s) in sums {
        let all = s.boxes + s.triangles;
        // The bucket that holds a share of the threads: its upper bound.
        let at = |share: f64| {
            let want = (s.threads as f64 * share).ceil() as u64;
            let mut seen = 0;
            for (i, n) in s.buckets.iter().enumerate() {
                seen += n;
                if seen >= want.max(1) {
                    return if i == 0 { 0 } else { (1u64 << i) - 1 };
                }
            }
            s.most
        };
        let over = |from: u64| -> f64 {
            let i = 64 - from.leading_zeros() as usize;
            100.0 * s.buckets[i.min(23)..].iter().sum::<u64>() as f64 / s.threads.max(1) as f64
        };
        tracing::info!(
            "node tests per ray (VICTORY_BVH_WALK): {:.1} dispatches a frame, {:.0} threads testing a node per dispatch; \
             tests per such thread: mean {:.1} ({:.1} boxes, {:.1} triangles), median <= {}, p90 <= {}, p99 <= {}, max {}; \
             threads at 1024+ tests {:.2} %, at 2048+ {:.2} %, at 4000+ (ended by the loop limit, their results lost) {:.2} %, \
             at 8192+ {:.3} %; children entered per box test {:.2} (those at 4000+: {:.2}); \
             threads that read a node off a page with no copy {:.3} % ({:.2} such tests a thread; of those at 4000+: {:.1} %): {}",
            s.dispatches as f64 / frames,
            s.threads as f64 / s.dispatches.max(1) as f64,
            all as f64 / s.threads.max(1) as f64,
            s.boxes as f64 / s.threads.max(1) as f64,
            s.triangles as f64 / s.threads.max(1) as f64,
            at(0.5),
            at(0.9),
            at(0.99),
            s.most,
            over(1024),
            over(2048),
            100.0 * s.cut as f64 / s.threads.max(1) as f64,
            over(8192),
            s.entered as f64 / s.boxes.max(1) as f64,
            s.cut_entered as f64 / s.cut_boxes.max(1) as f64,
            100.0 * s.lost_threads as f64 / s.threads.max(1) as f64,
            s.lost as f64 / s.lost_threads.max(1) as f64,
            100.0 * s.cut_lost as f64 / s.cut.max(1) as f64,
            name.chars().take(200).collect::<String>()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_read_back_sums_tests_children_entered_and_pages_with_no_copy() {
        let mut counts = vec![0u32; 2 * WALK_SLOTS as usize];
        let detail = WALK_SLOTS as usize;
        // A thread with 10 box tests and 2 triangle tests, 15 children entered.
        counts[0] = 10 | 2 << 16;
        counts[detail] = 15;
        // One ended by the loop limit, 4 of its tests on a page with no copy.
        counts[7] = 4090;
        counts[detail + 7] = 8000 | 4 << 16;
        // A slot with a detail but no test (cannot happen): not a thread.
        counts[detail + 9] = 3;
        let mut s = Sums::default();
        s.add(&counts);
        assert_eq!((s.dispatches, s.threads, s.boxes, s.triangles), (1, 2, 4100, 2));
        assert_eq!((s.entered, s.lost, s.lost_threads), (8015, 4, 1));
        assert_eq!((s.cut, s.cut_lost, s.cut_entered, s.cut_boxes), (1, 1, 8000, 4090));
    }
}
