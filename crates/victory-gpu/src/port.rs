//! Renderer port #1: the title's 32-bit graphics heap as the card's own
//! memory (`VICTORY_PORT_CONSTANTS`, on by default; `=0` off; `=3`: only
//! constant buffers read in place, see [`level`]).
//!
//! What the title does (game_scarlett_master.exe, the routine at image
//! +0x1e86440, reached at its start): one `XMemVirtualAlloc` of the five
//! sizes it is given together (kind 0x42000: a reservation of 64 KiB
//! pages, flags 7: graphics memory below 4 GiB, 185 MB at 0x34310000 in
//! every run so far), split into five placeholders (`VirtualFree` with
//! `MEM_PRESERVE_PLACEHOLDER`), each turned into graphics memory of its own
//! protection (`XMemVirtualAlloc`, kind 0x20406000, flags 0x27); then ONE
//! `XMemAllocatePhysicalPages` for all their pages and five
//! `XMemMapPhysicalPages`, in order, each from the next page of that one
//! array. `xmem` makes one section of all the pages, so the whole
//! reservation is one stretch of one section: views that follow one
//! another in memory and in the section alike. The title's 174 MB buffer at
//! 0x34e10000 (where its ring of constants lies, and what the skinning
//! passes write through one view of all of it) is in it; it is never
//! tile-mapped (`CopyPageMappings` maps the GPU heap's pages, not these).
//!
//! What the layer did with it: a host copy in the card's memory, the title's
//! writes found and sent up (17 MB a frame in ~600 pieces), what shaders
//! wrote fetched back and merged, and every constant buffer read out of it
//! (~6,000 fills a frame) either waited for flights or done on the card
//! from the copy.
//!
//! What is done here instead: the stretch is mapped once more, at an
//! address of ours (a view of the same section: the same physical pages),
//! and opened as a host heap there (`OpenExistingHeapFromAddress1`); title
//! buffers and windows of title memory in it are buffers placed in that
//! heap ([`crate::shadow::Shadow`]'s in-place form): the card reads and
//! writes the title's bytes where the title wrote them. Nothing is tracked,
//! sent up, fetched back or compared for it, and no compare or hash pass
//! of ours ever runs over it (they are for copies in the card's memory:
//! one such pass over memory shared with the PC removed the device on
//! another tree). No tile mapping is ever made into this heap (that removes
//! the device).
//!
//! Why our own view: a heap opened from an address holds the pages it was
//! opened on: the title could not unmap, decommit or protect them while it
//! stands. Opened on our view, the title's own views stay free to change.
//! When the title maps other pages at such an address, what is in place
//! there is let go (`crate::inplace::title_memory_freeing`) and the region
//! forgotten; the view of ours stays (shortcut: it is never unmapped, a
//! few hundred MB of address space at most).
//!
//! What it assumes: the CPU steps of ours that read this memory at
//! submission (fills of constant buffers, the title's buffer copies done on
//! the CPU) read what the title's CPU wrote, not what a submission still in
//! the air writes there: in place nothing tells the flights' writes, so
//! nothing waits for them. The title's constants are its CPU's; how many
//! such steps there are is on the frame-rate line ([`log`]).

use std::sync::Mutex;
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::*;
use windows_sys::Win32::System::Memory as mem;

use crate::inplace::{Placed, cover, stretch};

const K64: u64 = 64 << 10;

/// The ported memory lies below this: the title's graphics memory for
/// 32-bit addresses, which only that heap is.
const LOW: u64 = 1 << 32;

/// The most bytes of one region: its pages are locked into the PC's memory
/// while the heap stands.
const MOST: u64 = 512 << 20;

/// `VICTORY_PORT_CONSTANTS`, read once here: the level of renderer port #1.
/// - 1: title buffers and windows of title memory in the title's 32-bit
///   graphics memory (view memory that is one stretch of a section) are
///   used in place, all of them.
/// - 3 (the default, and any value but 0 and 1): constants in place
///   ([`constants_only`]): no title buffer or window is put in place (they stay host
///   copies, tracked, sent up and fetched back exactly as without the
///   port); the heap is arranged and opened all the same, with one buffer
///   placed over all of it ([`Whole`]), and only constant buffers whose
///   bytes lie in it are views of the title's bytes there: a raw view at
///   their own offset on 16 bytes with renderer port 5
///   ([`constant_raw_view`]), else a constant buffer view on 256
///   ([`constant_gpu`]). No fill and no refresh for them. With 1 the
///   skinning output and every other buffer the card writes and then reads
///   lie in the PC's memory, which the card streams far slower than its
///   own; small repeated reads of constants cost about what reads of its
///   own memory do. With 3 the indirect arguments the card's culling
///   writes are host copies again, so what `VICTORY_PORT_CONSTANTS_ORDER`
///   guards does not arise.
/// - 0: off, everything as before.
pub(crate) fn level() -> u8 {
    static LEVEL: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *LEVEL.get_or_init(|| {
        let level = match std::env::var("VICTORY_PORT_CONSTANTS").ok().as_deref().map(str::trim) {
            Some("0") => 0,
            Some("1") => 1,
            _ => 3,
        };
        if level == 3 {
            tracing::info!(
                "renderer port 1 (VICTORY_PORT_CONSTANTS=3): constants in place: constant buffers in the title's 32-bit graphics heap are read \
                 where the title has them; its buffers and windows stay host copies"
            );
        }
        level
    })
}

/// Renderer port #1 on, at any level (the heap arranged by `xmem` and
/// opened here).
pub(crate) fn on() -> bool {
    level() != 0
}

/// `VICTORY_PORT_CONSTANTS=3`: constants in place, nothing else ([`level`]).
pub(crate) fn constants_only() -> bool {
    level() == 3
}

/// Whether title buffers and windows are put in place (level 1; not 3).
pub(crate) fn places() -> bool {
    on() && !constants_only()
}

/// A stretch of title memory opened as a heap through a view of ours.
struct Region {
    start: u64,
    end: u64,
    heap: ID3D12Heap,
    /// With [`constants_only`]: one buffer placed over all of it, which
    /// constant buffer views name.
    whole: Option<&'static Whole>,
}

/// With [`constants_only`]: a region's buffer placed over all of it, and
/// where. Made once per region and never let go (leaked): views made on it
/// hold nothing, and a submission in flight may still read them after the
/// region is forgotten ([`RETIRED`]); the title opens its 32-bit heap once.
struct Whole {
    start: u64,
    end: u64,
    buffer: ID3D12Resource,
    gpu: u64,
}

// SAFETY: D3D12 resources are free-threaded.
unsafe impl Send for Whole {}
// SAFETY: as above.
unsafe impl Sync for Whole {}

/// Regions forgotten ([`forget`]) that had a [`Whole`]: kept, heap and all,
/// since views made on their buffer may still be read by a submission in
/// flight. shortcut: never let go (the title remaps its 32-bit heap rarely,
/// if ever).
static RETIRED: Mutex<Vec<Region>> = Mutex::new(Vec::new());

/// With [`constants_only`]: the [`Whole`] constant buffers were last found
/// in (null: none), for the look thousands of times a frame makes without a
/// lock.
static FAST: AtomicPtr<Whole> = AtomicPtr::new(std::ptr::null_mut());

/// With [`constants_only`]: the last time ([`crate::host::now_ms`]) a region
/// was asked to be opened for a constant buffer and could not be (memory
/// below 4 GiB that is not the heap, or not yet mapped): asked again 250 ms
/// later at the soonest.
static REFUSED_AT: AtomicU64 = AtomicU64::new(0);

/// With [`constants_only`], since the last line: constant buffer views on
/// the heap in place [0] of tables, [1] root ones, [2] title slots; [3]
/// bytes of refreshes and fills not asked for; [4] constant buffers in the
/// heap not on 256 bytes (16 for raw views), bound as before; [5] of those
/// in place, raw views (renderer port 5).
static CONSTANTS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// At any level, since the last line: tables' constant buffers [0] given
/// to the card where the title has them (no fill), [1] filled at
/// submission (copied out of title memory into upload memory): what an A/B
/// of the levels reads.
static TABLES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// A table's constant buffer was given in place (`true`) or filled.
pub(crate) fn table_constant(in_place: bool) {
    TABLES[usize::from(!in_place)].fetch_add(1, Ordering::Relaxed);
}

/// How a constant buffer was bound, for [`constant_gpu`]'s count.
#[derive(Clone, Copy)]
pub(crate) enum Bound {
    Table,
    Root,
    Slot,
}

/// With [`constants_only`] (`VICTORY_PORT_CONSTANTS=3`): the buffer over
/// the in-place heap and the host GPU address in it of title memory
/// `[address, address + len)`, for a constant buffer view of `how`, if all
/// of it lies in a region and `address` is on 256 bytes; `None` leaves the
/// caller to bind it as before (a host copy and its refresh, or a fill).
/// Opens the region the first time.
///
/// What it assumes: the bytes a constant buffer view shows are the title's
/// as they are when the card reads them. The title writes its constants on
/// the CPU, into a ring it waits on its fences for; what shaders write into
/// the heap goes to its host copy and comes back into title memory once
/// that submission has run, so a view in place sees it only then (a later
/// submission, already in flight when the copy-back lands, would not:
/// constants are not written by shaders, so this is not expected to
/// matter).
pub(crate) fn constant_gpu(
    address: u64,
    len: u64,
    how: Bound,
) -> Option<(&'static ID3D12Resource, u64)> {
    constant_in_heap(address, len, how, 256)
}

/// With [`constants_only`] and renderer port 5 (`VICTORY_PORT_CBV`, whose
/// shaders read constant buffers as raw bytes): as [`constant_gpu`], for a
/// raw view of the constant buffer at its exact offset, on 16 bytes (where
/// a raw view may begin). The title's ring of constants is on 16 bytes,
/// hardly ever on 256, so only this reads it in place.
pub(crate) fn constant_raw_view(
    address: u64,
    len: u64,
    how: Bound,
) -> Option<(&'static ID3D12Resource, u64)> {
    let found = constant_in_heap(address, len, how, 16);
    if found.is_some() {
        CONSTANTS[5].fetch_add(1, Ordering::Relaxed);
    }
    found
}

/// [`constant_gpu`] and [`constant_raw_view`]: `address` on `align` bytes.
fn constant_in_heap(
    address: u64,
    len: u64,
    how: Bound,
    align: u64,
) -> Option<(&'static ID3D12Resource, u64)> {
    if !constants_only() || address == 0 {
        return None;
    }
    let end = address.checked_add(len.max(1))?;
    if end > LOW {
        return None;
    }
    let whole = in_region(address, end)?;
    if !address.is_multiple_of(align) {
        CONSTANTS[4].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    CONSTANTS[how as usize].fetch_add(1, Ordering::Relaxed);
    CONSTANTS[3].fetch_add(len, Ordering::Relaxed);
    Some((&whole.buffer, whole.gpu + (address - whole.start)))
}

/// The [`Whole`] holding `[address, end)`, from [`FAST`] or [`whole_for`].
fn in_region(address: u64, end: u64) -> Option<&'static Whole> {
    // SAFETY: FAST is null or a leaked `Whole`, never freed.
    if let Some(fast) = unsafe { FAST.load(Ordering::Acquire).as_ref() }
        && fast.start <= address
        && end <= fast.end
    {
        return Some(fast);
    }
    let whole = whole_for(address, end)?;
    FAST.store(std::ptr::from_ref(whole).cast_mut(), Ordering::Release);
    Some(whole)
}

/// [`in_region`]'s slow look: the region holding `[address, end)` with its
/// whole buffer, opened if none does (not more often than every 250 ms
/// after a refusal).
fn whole_for(address: u64, end: u64) -> Option<&'static Whole> {
    {
        let regions = REGIONS.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(r) = regions.iter().find(|r| r.start <= address && end <= r.end) {
            return r.whole;
        }
    }
    let now = crate::host::now_ms().max(1);
    let last = REFUSED_AT.load(Ordering::Relaxed);
    if last != 0 && now < last + 250 {
        return None;
    }
    let host = crate::host::host()?;
    let at = address & !(K64 - 1);
    let stop = end.next_multiple_of(K64);
    let mut regions = REGIONS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(r) = regions.iter().find(|r| r.start <= address && end <= r.end) {
        return r.whole;
    }
    match open_region(&host.device, at, stop) {
        Ok(region) => {
            STATS[0].fetch_add(1, Ordering::Relaxed);
            STATS[1].fetch_add(region.end - region.start, Ordering::Relaxed);
            let whole = region.whole;
            regions.push(region);
            bounds(&regions);
            whole
        }
        Err((stat, why)) => {
            REFUSED_AT.store(now, Ordering::Relaxed);
            let n = STATS[stat].fetch_add(1, Ordering::Relaxed) + 1;
            if n <= 8 || n.is_power_of_two() {
                tracing::warn!(
                    start = format_args!("{address:#x}"),
                    why,
                    "renderer port 1 (VICTORY_PORT_CONSTANTS=3): no region for a constant buffer, bound as before"
                );
            }
            None
        }
    }
}

// SAFETY: D3D12 heaps are free-threaded.
unsafe impl Send for Region {}

static REGIONS: Mutex<Vec<Region>> = Mutex::new(Vec::new());

/// The lowest start and highest end of every region (both 0: none), for a
/// quick look on the paths that run thousands of times a frame.
static BOUNDS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Since the start: regions opened, their bytes; title buffers and windows
/// placed; refused because the memory is not one stretch of a section,
/// not committed, refused by the host; regions forgotten because the title
/// mapped other pages there. Then, since the last line: fills read out of
/// ported memory on the CPU, their bytes, buffer copies on the CPU that
/// touch it, their bytes.
static STATS: [AtomicU64; 13] = [const { AtomicU64::new(0) }; 13];

/// Whether `[at, at + len)` may be ported: in the title's 32-bit graphics
/// memory.
pub(crate) fn wanted(at: u64, len: u64) -> bool {
    on() && at.checked_add(len).is_some_and(|end| end <= LOW && len > 0)
}

/// Title memory `[start, start + len)` in place, if [`wanted`] and the
/// memory is one stretch of a section: a buffer placed in the region's
/// heap. `None` leaves the caller to do what it did before.
pub(crate) fn place_for(start: u64, len: u64, window: bool) -> Option<Placed> {
    if !places() || !wanted(start, len) {
        return None;
    }
    let host = crate::host::host()?;
    match place_in(&host.device, start, len) {
        Ok(placed) => {
            crate::inplace::counted(window, len);
            let n = STATS[2].fetch_add(1, Ordering::Relaxed) + 1;
            if n <= 32 || n.is_power_of_two() {
                tracing::info!(
                    start = format_args!("{start:#x}"),
                    len,
                    window,
                    "renderer port 1: title memory in place, read and written by the card where the title has it"
                );
            }
            Some(placed)
        }
        Err((at, why)) => {
            let n = STATS[at].fetch_add(1, Ordering::Relaxed) + 1;
            if n <= 8 || n.is_power_of_two() {
                tracing::warn!(
                    start = format_args!("{start:#x}"),
                    len,
                    window,
                    why,
                    "renderer port 1: title memory not in place, a copy as before"
                );
            }
            None
        }
    }
}

/// [`place_for`]'s work: the region holding the pages of `[start, start +
/// len)`, opened if none does, and a buffer placed in it. `Err` with the
/// statistic to count and why.
pub(crate) fn place_in(
    device: &ID3D12Device,
    start: u64,
    len: u64,
) -> Result<Placed, (usize, String)> {
    let at = start & !(K64 - 1);
    let end = (start + len.max(1)).next_multiple_of(K64);
    let mut regions = REGIONS.lock().unwrap_or_else(|e| e.into_inner());
    let i = match regions.iter().position(|r| r.start <= at && end <= r.end) {
        Some(i) => i,
        None => {
            let region = open_region(device, at, end)?;
            regions.push(region);
            let n = regions.len() - 1;
            let r = &regions[n];
            STATS[0].fetch_add(1, Ordering::Relaxed);
            STATS[1].fetch_add(r.end - r.start, Ordering::Relaxed);
            bounds(&regions);
            n
        }
    };
    let region = &regions[i];
    let buffer = crate::inplace::place(device, &region.heap, at - region.start, end - at)
        .map_err(|e| (6, format!("no buffer placed in the region: {e}")))?;
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() } + (start - at);
    Ok(Placed {
        buffer,
        heap: region.heap.clone(),
        gpu,
        bias: start - at,
    })
}

/// Opens the stretch around `[at, end)` (64 KiB aligned) as a heap through
/// a view of ours of the same pages.
fn open_region(device: &ID3D12Device, at: u64, end: u64) -> Result<Region, (usize, String)> {
    let Some((first, bytes, section, offset)) = stretch(at as usize, (end - at) as usize) else {
        return Err((
            3,
            format!(
                "not one stretch of a section: {:?}",
                cover(at as usize, (end - at) as usize)
            ),
        ));
    };
    // The whole stretch, kept to what may be ported: the next buffer in it
    // finds the region made.
    let (mut first, mut bytes, mut offset) = (first as u64, bytes as u64, offset);
    if first < LOW && first + bytes > LOW {
        bytes = LOW.saturating_sub(first);
    }
    if bytes > MOST {
        // Only what was asked for.
        offset += at - first;
        (first, bytes) = (at, end - at);
    }
    if bytes > MOST {
        return Err((3, format!("{bytes:#x} bytes, over the most of a region")));
    }
    if first > at || first + bytes < end || offset % K64 != 0 {
        return Err((
            3,
            format!("the stretch {first:#x}+{bytes:#x} does not hold it"),
        ));
    }
    // SAFETY: the section of a view the title has; the runtime keeps its
    // handle open while a view of it is there, and the range is inside it.
    let view = unsafe {
        mem::MapViewOfFile(
            section as _,
            mem::FILE_MAP_WRITE,
            (offset >> 32) as u32,
            offset as u32,
            bytes as usize,
        )
    }
    .Value as usize;
    if view == 0 {
        return Err((
            5,
            format!(
                "no view of the stretch: {}",
                std::io::Error::last_os_error()
            ),
        ));
    }
    let unmap = || {
        // SAFETY: our own view, nothing opened on it.
        unsafe {
            mem::UnmapViewOfFile(mem::MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view as *mut _,
            })
        };
    };
    if !committed(view, bytes as usize) {
        unmap();
        return Err((4, "pages of the stretch are not committed".to_owned()));
    }
    match crate::inplace::open(device, crate::inplace::From::Part(view, bytes as usize)) {
        Ok(heap) => {
            tracing::info!(
                start = format_args!("{first:#x}"),
                bytes,
                section = format_args!("{section:#x}"),
                offset = format_args!("{offset:#x}"),
                view = format_args!("{view:#x}"),
                "renderer port 1: a stretch of the title's graphics memory opened as a host heap through a view of ours"
            );
            // shortcut: the view stays mapped for good (the heap holds its
            // pages while anything names it).
            let whole = if constants_only() {
                match crate::inplace::place(device, &heap, 0, bytes) {
                    Ok(buffer) => {
                        crate::host::name(&buffer, || {
                            format!(
                                "title memory {first:#x}+{bytes:#x} in place, for constant buffers"
                            )
                        });
                        // SAFETY: a live buffer.
                        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
                        let whole: &'static Whole = Box::leak(Box::new(Whole {
                            start: first,
                            end: first + bytes,
                            buffer,
                            gpu,
                        }));
                        Some(whole)
                    }
                    // The heap holds our view's pages; the view stays
                    // mapped (as above, never unmapped).
                    Err(e) => return Err((6, format!("no buffer placed over the region: {e}"))),
                }
            } else {
                None
            };
            Ok(Region {
                start: first,
                end: first + bytes,
                heap,
                whole,
            })
        }
        Err(e) => {
            unmap();
            Err((5, format!("refused by the host: {e}")))
        }
    }
}

/// Whether every page of `[at, at + len)` is committed.
fn committed(at: usize, len: usize) -> bool {
    let mut here = at;
    while here < at + len {
        // SAFETY: plain data filled in by the query.
        let mut info: mem::MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        if unsafe { mem::VirtualQuery(here as *const _, &mut info, size_of_val(&info)) } == 0
            || info.State != mem::MEM_COMMIT
        {
            return false;
        }
        here = (info.BaseAddress as usize).saturating_add(info.RegionSize);
    }
    true
}

fn bounds(regions: &[Region]) {
    let lo = regions.iter().map(|r| r.start).min().unwrap_or(0);
    let hi = regions.iter().map(|r| r.end).max().unwrap_or(0);
    BOUNDS[0].store(lo, Ordering::Relaxed);
    BOUNDS[1].store(hi, Ordering::Relaxed);
}

/// The title maps other pages at `[at, at + len)`: the regions over it are
/// forgotten (what is placed in them is let go by the caller).
pub(crate) fn forget(at: usize, len: usize) {
    let (at, end) = (at as u64, (at as u64).saturating_add(len as u64));
    if BOUNDS[1].load(Ordering::Relaxed) <= at || BOUNDS[0].load(Ordering::Relaxed) >= end {
        return;
    }
    let mut regions = REGIONS.lock().unwrap_or_else(|e| e.into_inner());
    let before = regions.len();
    let (gone, kept): (Vec<Region>, Vec<Region>) = std::mem::take(&mut *regions)
        .into_iter()
        .partition(|r| r.end > at && r.start < end);
    *regions = kept;
    STATS[7].fetch_add((before - regions.len()) as u64, Ordering::Relaxed);
    bounds(&regions);
    drop(regions);
    // No constant buffer view is made on them any more.
    // SAFETY: FAST is null or a leaked `Whole`, never freed.
    if let Some(fast) = unsafe { FAST.load(Ordering::Acquire).as_ref() }
        && fast.end > at
        && fast.start < end
    {
        FAST.store(std::ptr::null_mut(), Ordering::Release);
    }
    // Those with a buffer for constant buffer views are kept; the others
    // are let go as before.
    RETIRED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .extend(gone.into_iter().filter(|r| r.whole.is_some()));
}

/// Whether `[at, at + len)` touches ported memory (a quick look).
fn touches(at: usize, len: usize) -> bool {
    let hi = BOUNDS[1].load(Ordering::Relaxed);
    hi != 0
        && (at as u64) < hi
        && (at as u64).saturating_add(len as u64) > BOUNDS[0].load(Ordering::Relaxed)
}

/// A fill of upload memory read `len` bytes at `src` on the CPU.
pub(crate) fn filled(src: usize, len: usize) {
    if touches(src, len) {
        STATS[9].fetch_add(1, Ordering::Relaxed);
        STATS[10].fetch_add(len as u64, Ordering::Relaxed);
    }
}

/// A title buffer copy was done on the CPU.
pub(crate) fn copied(dst: usize, src: usize, len: usize) {
    if touches(dst, len) || touches(src, len) {
        STATS[11].fetch_add(1, Ordering::Relaxed);
        STATS[12].fetch_add(len as u64, Ordering::Relaxed);
    }
}

/// For the frame-rate line: what is in place, and what of ours read or
/// wrote it on the CPU per frame.
pub(crate) fn log(frames: f64) {
    if !on() {
        return;
    }
    let s: Vec<u64> = STATS.iter().map(|s| s.load(Ordering::Relaxed)).collect();
    let per: Vec<f64> = STATS[9..]
        .iter()
        .map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames)
        .collect();
    let tables: Vec<f64> = TABLES
        .iter()
        .map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames.max(1e-9))
        .collect();
    tracing::info!(
        "renderer port 1 (VICTORY_PORT_CONSTANTS={}) per frame: {:.0} tables' constant buffer views made in place (the title's bytes, no fill), \
         {:.0} filled at submission",
        level(),
        tables[0],
        tables[1],
    );
    if constants_only() {
        let c: Vec<f64> = CONSTANTS
            .iter()
            .map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames.max(1e-9))
            .collect();
        tracing::info!(
            "renderer port 1 constants in place (VICTORY_PORT_CONSTANTS=3) per frame: {:.0} constant buffer views read where the title has them \
             ({:.0} tables', {:.0} root, {:.0} title slots'; {:.0} of them raw views, VICTORY_PORT_CBV; {:.1} MB of refreshes and fills not asked \
             for); {:.0} in the heap not on 256 bytes (16 for raw views), bound as before",
            c[0] + c[1] + c[2],
            c[0],
            c[1],
            c[2],
            c[5],
            c[3] / (1 << 20) as f64,
            c[4],
        );
    }
    tracing::info!(
        "renderer port 1 (VICTORY_PORT_CONSTANTS={}): {} regions of the title's graphics memory in place ({} MB), {} title buffers and windows placed in them \
         (no host copy, nothing sent up or fetched back for them); per frame on the CPU: {:.0} fills read out of them ({:.1} KB), {:.1} buffer copies \
         touching them ({:.1} KB); since the start refused: {} not one stretch of a section, {} not committed, {} by the host, {} no buffer; {} regions \
         forgotten (other pages mapped there)",
        level(),
        s[0],
        s[1] >> 20,
        s[2],
        per[0],
        per[1] / 1024.0,
        per[2],
        per[3] / 1024.0,
        s[3],
        s[4],
        s[5],
        s[6],
        s[7],
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only the title's 32-bit memory is asked for.
    #[test]
    fn only_memory_below_4_gib_is_wanted_with_the_switch() {
        // The switch is read once from the environment; tests do not set
        // it, so nothing is wanted.
        if !on() {
            assert!(!wanted(0x3431_0000, 0x1000));
        }
        let low = |at: u64, len: u64| at.checked_add(len).is_some_and(|end| end <= LOW && len > 0);
        assert!(low(0x34e1_0000, 0xae0_0000));
        assert!(!low(0xffff_0000, 0x2_0000));
        assert!(!low(0xe_1c80_0000, 0x1000));
        assert!(!low(u64::MAX, 2));
    }

    /// A stretch of a reserved section, mapped as `xmem` maps physical
    /// pages (views of consecutive pages in turn, committed), is opened
    /// through a view of ours: the card reads the bytes the title's views
    /// show, at the buffer's offset, and its writes show in them; a second
    /// buffer in the stretch is placed in the same region.
    #[test]
    fn warp_a_stretch_of_a_reserved_section_is_used_in_place_through_a_view_of_ours() {
        let _turn = crate::indirect::tests::warp_turn();
        let Some(device) = crate::inplace::tests::warp() else {
            return;
        };
        if windows::core::Interface::cast::<ID3D12Device13>(&device).is_err() {
            return;
        }
        const K: usize = 64 << 10;
        // SAFETY (the whole test): the section, placeholder and views made
        // here.
        unsafe {
            let process = windows_sys::Win32::System::Threading::GetCurrentProcess();
            let section = mem::CreateFileMappingW(
                windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE,
                std::ptr::null(),
                mem::PAGE_READWRITE | mem::SEC_RESERVE,
                0,
                8 * K as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let hole = mem::VirtualAlloc2(
                process,
                std::ptr::null(),
                8 * K,
                mem::MEM_RESERVE | mem::MEM_RESERVE_PLACEHOLDER,
                mem::PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            )
            .cast::<u8>();
            assert!(!hole.is_null());
            // Two views, pages 0..3 and 3..8 of the section, in turn.
            mem::VirtualFree(
                hole.cast(),
                3 * K,
                mem::MEM_RELEASE | mem::MEM_PRESERVE_PLACEHOLDER,
            );
            let mut views = Vec::new();
            for (page, n) in [(0usize, 3usize), (3, 5)] {
                let view = mem::MapViewOfFile3(
                    section,
                    process,
                    hole.add(page * K).cast(),
                    (page * K) as u64,
                    n * K,
                    mem::MEM_REPLACE_PLACEHOLDER,
                    mem::PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                );
                assert!(!view.Value.is_null(), "{}", std::io::Error::last_os_error());
                assert!(
                    !mem::VirtualAlloc(view.Value, n * K, mem::MEM_COMMIT, mem::PAGE_READWRITE)
                        .is_null()
                );
                crate::inplace::title_memory_view(
                    view.Value as usize,
                    n * K,
                    section as usize,
                    (page * K) as u64,
                );
                views.push(view);
            }
            let base = hole as u64;
            let (start, len) = (base + K as u64 + 0x40, 5 * K as u64);
            let placed =
                place_in(&device, start, len).unwrap_or_else(|e| panic!("not placed: {e:?}"));
            assert_eq!(placed.bias, 0x40);
            let mut rig = crate::inplace::Rig::new(&device).unwrap();
            for (i, b) in std::slice::from_raw_parts_mut(start as *mut u8, len as usize)
                .iter_mut()
                .enumerate()
            {
                *b = (i * 7 + 3) as u8;
            }
            let got = rig
                .fetch(&placed.buffer, placed.bias, len as usize)
                .unwrap();
            assert!(
                got == std::slice::from_raw_parts(start as *const u8, len as usize),
                "the card reads the title's bytes, across both views"
            );
            rig.store(&placed.buffer, placed.bias + 3 * K as u64, &[0x5a; 256])
                .unwrap();
            assert!(
                std::slice::from_raw_parts((start + 3 * K as u64) as *const u8, 256)
                    .iter()
                    .all(|&b| b == 0x5a),
                "the card's write shows in the title's view"
            );
            // Another buffer in the same stretch: the same region.
            let regions = REGIONS.lock().unwrap().len();
            let second = place_in(&device, base + 6 * K as u64, 100)
                .unwrap_or_else(|e| panic!("not placed: {e:?}"));
            assert_eq!(
                REGIONS.lock().unwrap().len(),
                regions,
                "one region for the stretch"
            );
            assert_eq!(second.bias, 0);
            // The title's views may change under the heap.
            drop((placed, second));
            forget(base as usize, 8 * K);
            for view in views {
                assert!(
                    mem::UnmapViewOfFile2(process, view, mem::MEM_PRESERVE_PLACEHOLDER) != 0,
                    "the title's own view can be unmapped"
                );
            }
            crate::inplace::views_gone(base as usize, 8 * K);
            mem::VirtualFree(hole.cast(), 0, mem::MEM_RELEASE);
            mem::VirtualFree(hole.add(3 * K).cast(), 0, mem::MEM_RELEASE);
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }
}
