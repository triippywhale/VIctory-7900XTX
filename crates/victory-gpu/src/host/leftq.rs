//! The fetch of pieces left on the card that something is about to read
//! (`crate::left::before_take`; after PalmVI 06edf544f, `left.rs` there).

use std::sync::{Arc, Mutex};

use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
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

/// What the card wrote in `pieces` (copy, offsets) and was left there is
/// fetched now and merged into title memory as copy-back merges
/// (`Shadow::fetched_left`). On a copy queue of its own, which does not
/// wait for the card's other work (the flights that wrote these pieces
/// have landed), unless bytes the title changed went up to one of the
/// copies and may not be there yet: then on the host's queue, behind every
/// send so far, as the relay's fetch without a copy queue (and without the
/// relay, whose sends are not noted, always so).
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
            WaitForSingleObject(event, INFINITE);
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
