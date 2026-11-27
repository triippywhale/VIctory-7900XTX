//! `VICTORY_RING_METER=1` (off by default): hooks in the title's own code
//! on its read-back rings, for `victory_gpu::ringmeter`. Build 1.0.2.1;
//! addresses are offsets into the image, whose preferred base is
//! 0x140000000.
//!
//! - The `call` in the ring's constructor that runs once its buffers exist
//!   (rcx = the ring): a ring is made.
//! - The ring's reader 0x14524dab0 (rcx = ring, rdx = lock record; returns
//!   a bool, by a tail jump into the engine's buffer lock when it answers):
//!   entry and return.
//! - "Mark written" 0x1453f1950 (rcx = ring): entry.
//!
//! Each site's bytes are checked before it is patched; a site that differs
//! (another build) is left alone with a warning and the meter says less.
//! Nothing here changes what the title does: each thunk saves and restores
//! every register and the flags around a call of ours
//! (`victory_loader::callhook`).
//!
//! The sites and the stack layout that names a ring's maker are Op-lite's
//! analysis (findings F1 and F4), from PastiiJay/klbf-lab
//! `crates/victory-hle/src/readback.rs`.

use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};

use victory_loader::callhook::{self, Regs};

/// The `call` (5 bytes) in the read-back ring's constructor that runs once
/// its buffers exist, with rcx = rsi = the ring (`mov rcx, rsi` before it).
const RING_SITE: usize = 0x1adc33b;
/// What it calls.
const RING_TARGET: usize = 0x1adc380;
/// The return address of the constructor's one call, in the ring wrapper
/// (0x141d40f90: eight pushes and 0x68 bytes, then the call).
const CTOR_RETURN: usize = 0x1d4107d;
/// From the hook's return slot to the constructor's return address (its
/// five pushes and 0x50 bytes, the call's own slot), and from there to the
/// wrapper's return address.
const TO_CTOR_RETURN: usize = 0x80;
const TO_WRAPPER_RETURN: usize = 0xb0;
/// Where the wrapper returns to, for the rings named so far (the others
/// are in the log by address). The names are from klbf-lab's notes
/// (`tools/docs/ghidra-perf-findings.md`, `dof-blur-analysis.md`); the wind
/// grid's and the voxelization's are by the function the place is in.
const NAMED: [(usize, &str); 12] = [
    (0xcc9067, "imposter"),
    (0x1dfee58, "visibility_buffer"),
    (0x3286358, "terrain_virtual_texture"),
    (0x57461f2, "autofocus_lens"),
    (0x5783c6c, "vfx_lightningStrike"),
    (0x5783d07, "vfx_lightningStrike"),
    (0x15ee338, "weatherparticle"),
    (0x57011a8, "puddledetect"),
    (0x5a8c7b0, "cascadeshadows_classify"),
    (0x36b933, "deferred_lighting"),
    (0x236eb5c, "wind_grid"),
    (0xcbda79, "entity_coarse_voxelization"),
];
/// The ring reader: `push r15; push r14; push rsi; push rdi; push rbp;
/// push rbx`.
const READ: usize = 0x524dab0;
const READ_PROLOGUE: [u8; 8] = [0x41, 0x57, 0x41, 0x56, 0x56, 0x57, 0x55, 0x53];
/// "Mark written": `push r14; push rsi; push rdi; push rbx`.
const MARK: usize = 0x53f1950;
const MARK_PROLOGUE: [u8; 5] = [0x41, 0x56, 0x56, 0x57, 0x53];
/// The preferred base, for addresses printed as an analyst's.
const PREFERRED: usize = 0x1_4000_0000;

static BASE: AtomicUsize = AtomicUsize::new(0);
static READ_AFTER: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// Ring reads under way on this thread: the real return address, the
    /// ring, the record.
    static READS: RefCell<Vec<(u64, u64, u64)>> = const { RefCell::new(Vec::new()) };
}

unsafe extern "C" fn ring_made(regs: *mut Regs) {
    // SAFETY: the thunk's saved registers.
    let ring = unsafe { (*regs).rcx };
    let base = BASE.load(Ordering::Relaxed);
    // The stack above the call, as the constructor and the wrapper laid it
    // out (checked by the constructor's return address).
    // SAFETY: `ret` is the call's slot on this thread's stack; the frames
    // above it are the constructor's and the wrapper's, still live.
    let slot = unsafe { std::ptr::addr_of!((*regs).ret) } as usize;
    // SAFETY: as above.
    let ctor_return = unsafe { ((slot + TO_CTOR_RETURN) as *const usize).read() };
    let site = if ctor_return == base + CTOR_RETURN {
        // SAFETY: as above.
        unsafe { ((slot + TO_CTOR_RETURN + TO_WRAPPER_RETURN) as *const usize).read() }.wrapping_sub(base)
    } else {
        0
    };
    let name = NAMED.iter().find(|n| n.0 == site).map_or("", |n| n.1);
    let site = if site == 0 { 0 } else { (site + PREFERRED) as u64 };
    let _ = std::panic::catch_unwind(|| victory_gpu::ringmeter::ring_created(ring, site, name));
}

unsafe extern "C" fn read_entered(regs: *mut Regs) {
    // SAFETY: the thunk's saved registers; `ret` is the return address,
    // which the stub takes.
    let regs = unsafe { &mut *regs };
    READS.with(|l| l.borrow_mut().push((regs.ret, regs.rcx, regs.rdx)));
    regs.ret = READ_AFTER.load(Ordering::Relaxed) as u64;
}

unsafe extern "C" fn read_returned(rax: u64, slot: *mut u64) {
    let Some((ret, ring, record)) = READS.with(|l| l.borrow_mut().pop()) else {
        // Nowhere to return to: cannot happen (each stub return was pushed).
        std::process::abort();
    };
    // SAFETY: the stub's slot for the return address.
    unsafe { *slot = ret };
    let _ = std::panic::catch_unwind(|| victory_gpu::ringmeter::ring_read(ring, record, rax & 0xff != 0));
}

unsafe extern "C" fn mark_entered(regs: *mut Regs) {
    // SAFETY: the thunk's saved registers.
    let ring = unsafe { (*regs).rcx };
    let _ = std::panic::catch_unwind(|| victory_gpu::ringmeter::ring_marked(ring));
}

/// The bytes at offset `rva` of the image at `base`.
fn bytes(base: usize, rva: usize, n: usize) -> Vec<u8> {
    // SAFETY: inside the mapped image (the sites are in its .text).
    unsafe { std::slice::from_raw_parts((base + rva) as *const u8, n) }.to_vec()
}

/// Installs the hooks if `VICTORY_RING_METER=1` asks for them.
///
/// # Safety
/// `base` is the title's mapped image of `size` bytes, and none of its code
/// runs yet.
pub unsafe fn install(base: usize, size: usize) {
    if !victory_gpu::ringmeter::on() {
        return;
    }
    if size <= MARK + 8 {
        tracing::warn!(size, "VICTORY_RING_METER: the image is too small for the hook sites (another build?): no hooks");
        return;
    }
    BASE.store(base, Ordering::Relaxed);
    // The call and the `mov rcx, rsi` before it.
    let found = bytes(base, RING_SITE - 3, 8);
    if found[..3] != [0x48, 0x89, 0xF1] {
        tracing::warn!(at = format_args!("{:#x}", PREFERRED + RING_SITE), ?found, "VICTORY_RING_METER: ring hook not installed, the bytes differ (another build?)");
        return;
    }
    // SAFETY: the caller's promise; `redirect_call` checks the call.
    match unsafe { callhook::redirect_call(base + RING_SITE, base + RING_TARGET, ring_made) } {
        Ok(()) => tracing::info!(at = format_args!("{:#x}", PREFERRED + RING_SITE), "VICTORY_RING_METER: read-back ring hook installed"),
        Err(e) => {
            tracing::warn!(error = %e, "VICTORY_RING_METER: ring hook not installed, the bytes differ (another build?)");
            return;
        }
    }
    let stub = match callhook::after_stub(read_returned) {
        Ok(stub) => stub,
        Err(e) => {
            tracing::warn!(error = %e, "VICTORY_RING_METER: ring read hook not installed: rings are listed, reads are not counted");
            return;
        }
    };
    READ_AFTER.store(stub, Ordering::Relaxed);
    // SAFETY: the caller's promise; `hook_entry` checks the prologue.
    match unsafe { callhook::hook_entry(base + READ, &READ_PROLOGUE, read_entered) } {
        Ok(()) => tracing::info!(at = format_args!("{:#x}", PREFERRED + READ), "VICTORY_RING_METER: ring read hook installed"),
        Err(e) => tracing::warn!(error = %e, found = ?bytes(base, READ, 8), "VICTORY_RING_METER: ring read hook not installed, the bytes differ (another build?): reads are not counted"),
    }
    // SAFETY: as above.
    match unsafe { callhook::hook_entry(base + MARK, &MARK_PROLOGUE, mark_entered) } {
        Ok(()) => tracing::info!(at = format_args!("{:#x}", PREFERRED + MARK), "VICTORY_RING_METER: ring mark-written hook installed"),
        Err(e) => tracing::warn!(error = %e, found = ?bytes(base, MARK, 8), "VICTORY_RING_METER: ring mark-written hook not installed, the bytes differ (another build?)"),
    }
}
