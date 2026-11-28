//! The fetch of pieces left on the card that something is about to read
//! (`crate::left::before_take`; after PalmVI 06edf544f, `left.rs` there).

use std::sync::{Arc, Mutex};

use windows::Win32::Foundation::{HANDLE, WAIT_OBJECT_0, WAIT_TIMEOUT};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::Interface;

use super::{Host, QUEUE_ORDER, fail};
use crate::shadow::Shadow;

/// A list, fence and event of this module's own, a copy queue, and the
/// memory the pieces come down to.
struct Rig {
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    direct: (ID3D12CommandAllocator, ID3D12GraphicsCommandList),
    fence: ID3D12Fence,
    event: HANDLE,
    sent: u64,
    buffer: Option<(ID3D12Resource, usize, u64)>,
}

// SAFETY: used under the lock below only.
unsafe impl Send for Rig {}

static RIG: Mutex<Option<Rig>> = Mutex::new(None);

/// A fetch that has waited for the card, for the log.
struct Waited {
    /// On the host's queue, behind every send so far; else on the copy
    /// queue of this module.
    ordered: bool,
    pieces: usize,
    bytes: u64,
    seconds: f64,
    thread: String,
    /// Of the fetch's own fence: the value reached, the value waited for.
    reached: (u64, u64),
    /// Of the host's queue: the value its fence reached, and the last
    /// submission signalled behind, if the submission lock is free.
    host: (u64, Option<u64>),
}

/// The line of a fetch that waits: the path, what is fetched, how far the
/// card is, and the locks the thread holds meanwhile (`crate::left`,
/// "Locks").
fn waited(w: &Waited) -> String {
    let path = match w.ordered {
        true => "ordered, on the host's queue behind every send so far",
        false => "on the copy queue of its own",
    };
    let submission = match w.host.1 {
        Some(sent) => format!("{sent} (the submission lock is free)"),
        None => "not known (the submission lock is held, by this thread or another)".to_owned(),
    };
    format!(
        "pieces left on the card: a fetch ({path}) of {} pieces, {} bytes, has waited {:.0} s for the card on thread {}; its fence is \
         at {} of {}, the host queue's at {}, the last submission signalled {submission}; the thread holds the gate of pieces \
         left and the fetch's rig (not the book, the order of the queue, the journal or the guards), and whatever its caller \
         held; still waiting (VICTORY_SKIP_WC_BACK)",
        w.pieces, w.bytes, w.seconds, w.thread, w.reached.0, w.reached.1, w.host.0
    )
}

/// What the card wrote in `pieces` (copy, offsets) and was left there is
/// fetched now and merged into title memory as copy-back merges
/// (`Shadow::fetched_left`). On a copy queue of its own, which does not
/// wait for the card's other work (the flights that wrote these pieces
/// have landed), unless bytes the title changed went up to one of the
/// copies and may not be there yet: then on the host's queue, behind every
/// send so far, as the relay's fetch without a copy queue (and without the
/// relay, whose sends are not noted, always so).
///
/// There is no switch to keep the fetch off the host's queue
/// (`VICTORY_SKIP_WC_BACK_ORDERED=0` was thought of, to narrow a hang
/// down): on the copy queue a fetch may run before title bytes that are
/// on their way up, and the merge then leaves every noted send out; where
/// a shader wrote over such bytes after they arrived, its write would be
/// lost to title memory for good (the piece is never left again, and the
/// card's compare knows it as unchanged). Nor does the ordered wait depend
/// on the CPU: what is ahead of it on the host's queue are lists the card
/// runs by itself (`crate::left`, "Locks").
///
/// The wait for the card is never without end and without a word: every
/// `crate::left::STEP` of it is said in the log ([`waited`]).
pub(crate) fn fetch(host: &Host, pieces: &[(Arc<Shadow>, usize, usize)]) {
    if pieces.is_empty() {
        return;
    }
    let mut rig = RIG.lock().unwrap_or_else(|e| e.into_inner());
    if rig.is_none() {
        // SAFETY: plain objects of our own.
        *rig = unsafe {
            (|| -> windows::core::Result<Rig> {
                let queue: ID3D12CommandQueue =
                    host.device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_COPY,
                        ..Default::default()
                    })?;
                let allocator: ID3D12CommandAllocator = host
                    .device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)?;
                let list: ID3D12GraphicsCommandList = host.device.CreateCommandList(
                    0,
                    D3D12_COMMAND_LIST_TYPE_COPY,
                    &allocator,
                    None,
                )?;
                list.Close()?;
                let direct_allocator: ID3D12CommandAllocator = host
                    .device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                let direct: ID3D12GraphicsCommandList = host.device.CreateCommandList(
                    0,
                    D3D12_COMMAND_LIST_TYPE_DIRECT,
                    &direct_allocator,
                    None,
                )?;
                direct.Close()?;
                let fence: ID3D12Fence = host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
                let event = CreateEventW(None, false, false, None)?;
                Ok(Rig {
                    queue,
                    allocator,
                    list,
                    direct: (direct_allocator, direct),
                    fence,
                    event,
                    sent: 0,
                    buffer: None,
                })
            })()
            .ok()
        };
    }
    let Some(r) = rig.as_mut() else {
        fail(format_args!(
            "no copy queue to fetch pieces left on the card (VICTORY_SKIP_WC_BACK)"
        ));
    };
    let total: u64 = pieces.iter().map(|p| (p.2 - p.1) as u64).sum();
    if r.buffer.as_ref().is_none_or(|b| b.2 < total) {
        let size = total.next_multiple_of(1 << 20);
        r.buffer = host
            .shared_buffer(size)
            .map(|(buffer, data)| (buffer, data as usize, size));
    }
    let Some((buffer, data, _)) = r.buffer.clone() else {
        fail(format_args!(
            "no memory to fetch pieces left on the card ({total} bytes, VICTORY_SKIP_WC_BACK)"
        ));
    };
    let ordered = !crate::shadow::relay()
        || pieces
            .iter()
            .any(|(shadow, lo, hi)| shadow.sends_noted(*lo, *hi));
    r.sent += 1;
    let (fence, event, sent) = (r.fence.clone(), r.event, r.sent);
    let (allocator, list) = match ordered {
        true => r.direct.clone(),
        false => (r.allocator.clone(), r.list.clone()),
    };
    // SAFETY: the list's last run was waited for under this lock; live
    // buffers (held by `pieces`); the fence and the event are this rig's.
    let ran = unsafe {
        (|| -> windows::core::Result<u64> {
            allocator.Reset()?;
            list.Reset(&allocator, None)?;
            let mut at = 0u64;
            for (shadow, lo, hi) in pieces {
                list.CopyBufferRegion(&buffer, at, &shadow.buffer, *lo as u64, (*hi - *lo) as u64);
                at += (*hi - *lo) as u64;
            }
            list.Close()?;
            let lists = [Some(list.cast::<ID3D12CommandList>()?)];
            let after = if ordered {
                // In the order of the queue: the sends of title bytes so
                // far are ahead of this fetch, every later one behind it.
                let order = QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner());
                host.queue.ExecuteCommandLists(&lists);
                let after = *order;
                drop(order);
                host.card_sent();
                host.queue.Signal(&fence, sent)?;
                after
            } else {
                r.queue.ExecuteCommandLists(&lists);
                r.queue.Signal(&fence, sent)?;
                // Every send still noted is left out of the merge.
                0
            };
            fence.SetEventOnCompletion(sent, event)?;
            // Never without end and without a word: a wait of
            // `crate::left::STEP` is said, each time, with what is waited
            // for and what this thread holds meanwhile.
            let began = std::time::Instant::now();
            let mut said = false;
            loop {
                match WaitForSingleObject(event, crate::left::STEP.as_millis() as u32) {
                    WAIT_OBJECT_0 => break,
                    WAIT_TIMEOUT => {}
                    _ => return Err(windows::core::Error::from_thread()),
                }
                // (A fence of a device that is gone reads as all ones.)
                if fence.GetCompletedValue() == u64::MAX {
                    host.device.GetDeviceRemovedReason()?;
                }
                said = true;
                tracing::warn!(
                    "{}",
                    waited(&Waited {
                        ordered,
                        pieces: pieces.len(),
                        bytes: total,
                        seconds: began.elapsed().as_secs_f64(),
                        thread: crate::left::thread_name(&std::thread::current()),
                        reached: (fence.GetCompletedValue(), sent),
                        host: (
                            host.fence.GetCompletedValue(),
                            host.submitted.try_lock().ok().map(|n| *n)
                        ),
                    })
                );
            }
            if said {
                tracing::warn!(
                    "pieces left on the card: the fetch that waited came after {:.1} s (VICTORY_SKIP_WC_BACK)",
                    began.elapsed().as_secs_f64()
                );
            }
            Ok(after)
        })()
    };
    let after = match ran {
        Ok(after) => after,
        Err(e) => fail(format_args!(
            "pieces left on the card could not be fetched ({e}, VICTORY_SKIP_WC_BACK)"
        )),
    };
    let mut title = Vec::new();
    let mut at = 0usize;
    for (shadow, lo, hi) in pieces {
        // SAFETY: the mapping of the fetched bytes, `total` of them; the
        // copy into it has run, and the next waits for this lock.
        let gpu = unsafe { std::slice::from_raw_parts((data + at) as *const u8, hi - lo) };
        shadow.fetched_left(*lo, gpu, after, &mut title);
        at += hi - lo;
    }
}

#[cfg(test)]
mod tests {
    use super::{Waited, waited};

    /// The line of a fetch that waits names its path, what it fetches and
    /// the locks held.
    #[test]
    fn a_fetch_that_waits_says_its_path_and_its_locks() {
        let mut w = Waited {
            ordered: true,
            pieces: 5,
            bytes: 131072,
            seconds: 4.0,
            thread: "victory-wide-2".into(),
            reached: (41, 42),
            host: (1000, None),
        };
        let line = waited(&w);
        assert!(line.contains("ordered, on the host's queue"), "{line}");
        assert!(
            line.contains("5 pieces, 131072 bytes, has waited 4 s"),
            "{line}"
        );
        assert!(line.contains("thread victory-wide-2"), "{line}");
        assert!(line.contains("at 41 of 42"), "{line}");
        assert!(line.contains("the submission lock is held"), "{line}");
        assert!(
            line.contains("holds the gate of pieces left and the fetch's rig"),
            "{line}"
        );
        w.ordered = false;
        w.host.1 = Some(1001);
        let line = waited(&w);
        assert!(line.contains("on the copy queue of its own"), "{line}");
        assert!(
            line.contains("signalled 1001 (the submission lock is free)"),
            "{line}"
        );
    }
}
