//! What the system's breadcrumbs (DRED, `--breadcrumbs`) say once the
//! replay's device is removed: the lists that did not finish, the operation
//! each stopped at, and where a page fault was. Without `--breadcrumbs` the
//! runtime keeps none and this says so.

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{Interface, PCWSTR};

/// Lines on why the device is gone; none while it stands.
pub(super) fn report(device: &ID3D12Device) -> Vec<String> {
    let mut out = Vec::new();
    // SAFETY: a plain query.
    if let Err(e) = unsafe { device.GetDeviceRemovedReason() } {
        out.push(format!("the device was removed: {e}"));
    } else {
        return out;
    }
    let Ok(dred) = device.cast::<ID3D12DeviceRemovedExtendedData1>() else {
        out.push("no breadcrumbs on this device".into());
        return out;
    };
    // SAFETY: the outputs are the runtime's, valid while the device lives;
    // nodes are walked as linked, histories have their counts.
    unsafe {
        match dred.GetPageFaultAllocationOutput1() {
            Ok(f) if f.PageFaultVA != 0 => {
                out.push(format!("page fault at {:#x}", f.PageFaultVA));
                for (what, head) in [
                    ("in", f.pHeadExistingAllocationNode),
                    ("in freed", f.pHeadRecentFreedAllocationNode),
                ] {
                    let mut names = Vec::new();
                    let mut node = head;
                    while let Some(n) = node.as_ref()
                        && names.len() < 12
                    {
                        names.push(format!(
                            "{} (allocation type {})",
                            wide(n.ObjectNameW),
                            n.AllocationType.0
                        ));
                        node = n.pNext;
                    }
                    if !names.is_empty() {
                        out.push(format!("  {what}: {}", names.join(", ")));
                    }
                }
            }
            Ok(_) => out.push("no page fault noted".into()),
            Err(e) => out.push(format!(
                "no page fault output ({e}): run with --breadcrumbs"
            )),
        }
        let crumbs = match dred.GetAutoBreadcrumbsOutput1() {
            Ok(c) => c,
            Err(e) => {
                out.push(format!("no breadcrumbs ({e}): run with --breadcrumbs"));
                return out;
            }
        };
        let (mut lists, mut stopped) = (0, 0);
        let mut node = crumbs.pHeadAutoBreadcrumbNode;
        while let Some(n) = node.as_ref() {
            lists += 1;
            let done = n.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
            if done < n.BreadcrumbCount && !n.pCommandHistory.is_null() {
                stopped += 1;
                if stopped <= 8 {
                    let history =
                        std::slice::from_raw_parts(n.pCommandHistory, n.BreadcrumbCount as usize);
                    let from = (done as usize).saturating_sub(3);
                    let to = (done as usize + 4).min(history.len());
                    let around: Vec<String> = (from..to)
                        .map(|i| {
                            format!(
                                "{}{}",
                                if i == done as usize { ">" } else { "" },
                                op(history[i])
                            )
                        })
                        .collect();
                    out.push(format!(
                        "list \"{}\" on queue \"{}\": {done} of {} operations done; around the stop: {}",
                        wide(n.pCommandListDebugNameW),
                        wide(n.pCommandQueueDebugNameW),
                        n.BreadcrumbCount,
                        around.join(" ")
                    ));
                }
            }
            node = n.pNext;
        }
        out.push(format!(
            "{stopped} of {lists} lists with breadcrumbs did not finish"
        ));
    }
    out
}

/// An error with what the breadcrumbs say added, when the device is gone.
pub(super) fn explain(device: &ID3D12Device, e: anyhow::Error) -> anyhow::Error {
    let lines = report(device);
    if lines.is_empty() {
        return e;
    }
    for l in &lines {
        tracing::error!("replay: {l}");
    }
    e.context(lines.join("\n"))
}

/// Names an object for the breadcrumbs.
pub(super) fn name(object: &impl Interface, text: &str) {
    if let Ok(o) = object.cast::<ID3D12Object>() {
        let w: Vec<u16> = text.encode_utf16().chain([0]).collect();
        // SAFETY: a nul-terminated string that outlives the call.
        let _ = unsafe { o.SetName(PCWSTR(w.as_ptr())) };
    }
}

fn wide(p: PCWSTR) -> String {
    if p.is_null() {
        return "unnamed".into();
    }
    // SAFETY: a nul-terminated string of the runtime's.
    unsafe { p.to_string() }.unwrap_or_default()
}

fn op(o: D3D12_AUTO_BREADCRUMB_OP) -> String {
    let name = match o {
        D3D12_AUTO_BREADCRUMB_OP_SETMARKER => "marker",
        D3D12_AUTO_BREADCRUMB_OP_BEGINEVENT => "begin",
        D3D12_AUTO_BREADCRUMB_OP_ENDEVENT => "end",
        D3D12_AUTO_BREADCRUMB_OP_DRAWINSTANCED => "draw",
        D3D12_AUTO_BREADCRUMB_OP_DRAWINDEXEDINSTANCED => "draw-indexed",
        D3D12_AUTO_BREADCRUMB_OP_EXECUTEINDIRECT => "indirect",
        D3D12_AUTO_BREADCRUMB_OP_DISPATCH => "dispatch",
        D3D12_AUTO_BREADCRUMB_OP_COPYBUFFERREGION => "copy-buffer",
        D3D12_AUTO_BREADCRUMB_OP_COPYTEXTUREREGION => "copy-texture",
        D3D12_AUTO_BREADCRUMB_OP_COPYRESOURCE => "copy-resource",
        D3D12_AUTO_BREADCRUMB_OP_COPYTILES => "copy-tiles",
        D3D12_AUTO_BREADCRUMB_OP_RESOLVESUBRESOURCE => "resolve",
        D3D12_AUTO_BREADCRUMB_OP_CLEARRENDERTARGETVIEW => "clear-rt",
        D3D12_AUTO_BREADCRUMB_OP_CLEARUNORDEREDACCESSVIEW => "clear-uav",
        D3D12_AUTO_BREADCRUMB_OP_CLEARDEPTHSTENCILVIEW => "clear-ds",
        D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER => "barrier",
        D3D12_AUTO_BREADCRUMB_OP_EXECUTEBUNDLE => "bundle",
        D3D12_AUTO_BREADCRUMB_OP_PRESENT => "present",
        D3D12_AUTO_BREADCRUMB_OP_RESOLVEQUERYDATA => "resolve-queries",
        D3D12_AUTO_BREADCRUMB_OP_BEGINSUBMISSION => "begin-submission",
        D3D12_AUTO_BREADCRUMB_OP_ENDSUBMISSION => "end-submission",
        D3D12_AUTO_BREADCRUMB_OP_ATOMICCOPYBUFFERUINT => "atomic-copy",
        D3D12_AUTO_BREADCRUMB_OP_WRITEBUFFERIMMEDIATE => "write-immediate",
        D3D12_AUTO_BREADCRUMB_OP_DISPATCHRAYS => "dispatch-rays",
        D3D12_AUTO_BREADCRUMB_OP_BUILDRAYTRACINGACCELERATIONSTRUCTURE => "build-bvh",
        D3D12_AUTO_BREADCRUMB_OP_COPYRAYTRACINGACCELERATIONSTRUCTURE => "copy-bvh",
        D3D12_AUTO_BREADCRUMB_OP_BARRIER => "barrier-enhanced",
        _ => return format!("op{}", o.0),
    };
    name.to_owned()
}
