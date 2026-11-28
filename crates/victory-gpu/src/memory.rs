//! Title memory by address, for translated shaders that read and write
//! through raw 40-bit addresses (`victory_shader::emit::SLOT_MEMORY`: the
//! console's ray-tracing structure builder and the programs that walk its
//! structures; the structures themselves lie in plain title memory, in no
//! buffer object).
//!
//! A 64 KiB page in a title buffer that holds all of it is in that buffer's
//! copy ([`crate::shadow`]), which every other access uses. A page outside
//! buffers has a page of its own in an **arena**: memory of ours with a copy
//! on the card ([`crate::shadow::by_address`]). The streamed structures lie
//! anywhere between other data (6.6 GB of title memory in 4 MiB windows, run
//! light-1; 8 GB in windows of 256 KiB, light-2), so a page is what is kept.
//!
//! This module keeps what a shader needs to find the pages:
//!
//! - the **directory**: per 64 KiB page of title address space the number
//!   of the copy that holds the page (0: none) and the page's byte offset
//!   in it; one for all lists. An entry goes from empty to filled, and back
//!   to empty when its page is given back.
//!   The CPU writes it in memory of its own and shaders read a copy of that
//!   on the card, brought up to date as title memory is (over the bus, every
//!   step of every ray waited for it);
//! - per view heap a run of slots with a raw unordered access view of each
//!   numbered copy (slot 0 of the run: a null view, for pages without a
//!   copy);
//! - the **faults**: the page numbers shaders found empty, read back here
//!   once a frame, so the pages are there from the next frame on. A pointer a
//!   shader takes out of memory cannot be known before it runs. The programs
//!   that walk leave theirs in the first half of the notes, the builder in
//!   the second: what the builder found empty it may store in later.
//!
//! **A page that is not there when a shader touches it is a lost access**:
//! the read gives 0, the store is dropped, and the fault note only helps
//! the next frame. For the structure builder that is the same as a dropped
//! stage, so before one of its dispatches the host makes ready what it can
//! see the dispatch name:
//!
//! - 256 KiB from each root view outside buffers, and where title memory has a
//!   packed ("dehydrated") structure there, all of it: the title ships its
//!   structures packed, and the builder's rehydration stages unpack each in
//!   place once it is streamed in;
//! - 4 MiB from each unordered access view outside buffers (structures built);
//! - the page of each pair of root constants that is a title address and of
//!   each root constant that is one in 256-byte units;
//! - the first page of every structure a list's upload memory names
//!   (a top-level build's instances).
//!
//! **What comes back to title memory**: what the builder stores in title
//! buffers of at most [`BACK`] bytes it names (the sizes the title reads to
//! make a structure smaller). Structures and scratch memory stay on the card
//! (the first whole run of the builder wrote garbage over the title's
//! objects). So an arena's page is the only place a structure is whole, and
//! title memory must not go over it: of a page the title wrote only the
//! dwords that changed since they were last taken go up. A packed structure
//! goes up whole when the builder comes to it, whatever is known of it: the
//! same bytes streamed in again do not differ from those last taken, and the
//! card holds them unpacked (run light-3: a quarter of the structures
//! unpacked a second time, into garbage all over their neighbours).
//!
//! **Stores are bounded**: a structure the builder misreads makes it store
//! anywhere. Pages of title buffers take its stores only once one of its
//! dispatches was given that buffer as an unordered access view (its scratch
//! memory, its results); what else a store reaches is an arena's.
//!
//! A packed structure is gone when title memory has no header of it any more
//! (looked at over [`SWEEP`] frames); the pages it alone has are given back.
//!
//! **Pages from faults go round**: at most [`FAULT_MB`] of arena pages are
//! held because a shader found them empty. A present page leaves no note, so
//! whether such a page is still read is not known; at the limit the oldest
//! goes for the new one, and if it was still read its note comes back in the
//! next frame and it is made again from title memory (one frame of zeros
//! there; counted, see [`Memory::report`]). A page a list names, or a packed
//! structure is unpacked in, is no longer one of these: it may hold what is
//! on the card alone. A page whose title memory is gone goes at once.
//! (`VICTORY_BVH_FAULTS_RECYCLE=0`: a count for the whole run, as before;
//! see [`recycle`].)
//!
//! shortcut: a page a list named is kept for the rest of the run.
//! shortcut: pages a list does not name are looked at again in turn, over
//! [`SWEEP`] frames.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use victory_shader::emit::{MEMORY_FAULTS, MEMORY_PAGES, MEMORY_READ_ONLY};
use windows::Win32::Graphics::Direct3D12::*;

use crate::descriptor::host::HostHeap;
use crate::host::{HostList, host};
use crate::shadow::Shadow;

#[cfg(feature = "diagnostics")]
mod stores;

/// Copies that can be numbered (the run of heap slots is one longer).
const COPIES: u32 = (1 << 12) - 1;
/// A directory page.
const PAGE: u64 = 1 << 16;
/// An arena's bytes.
const ARENA: u64 = 64 << 20;
/// What is compared at a time.
const PIECE: usize = 4096;
/// Pages held, at most (3 GiB).
const MOST: usize = 3 << 14;
/// Pages of title memory with small title buffers in them, at most.
const WINDOWS: usize = 512;
/// The biggest packed structure taken for one.
const PACKED: u64 = 64 << 20;
/// For so many frames what a page holds may still be on its way.
const YOUNG: u64 = 4;
/// Free pages there are, at least, when a list is recorded.
const SPARE: usize = 1024;
/// A packed structure not named for so many frames is another one.
const AGAIN: u64 = 8;
/// From a root view; from one that is written (a structure being built).
const VIEW: u64 = 256 << 10;
const BUILT: u64 = 4 << 20;
/// The biggest title buffer whose builder results come back.
const BACK: u64 = 256 << 10;
/// Pages taken from the faults in one frame, at most: a ray in a structure
/// that is not right goes anywhere, and 2,048 a frame took all the pages
/// there are in the first seconds (run light-4).
const FAULTS: usize = 128;
/// Pages taken from faults in a whole run, the old way ([`recycle`] off).
const FAULTS_ALL: usize = 2048;
/// Arena pages held from the faults at a time, at most, in MiB. Not a count
/// for the run any more: with ray tracing on, the 2,048 pages of the run were
/// gone 7-20 seconds after the builder started (7900 XT runs, 2026-11-26),
/// and from then on up to about 1,000 pages a frame that structures point
/// into read as zeros for the rest of the run, which looks like dark and
/// flickering indirect light. What rays touch follows where the player goes,
/// so at the limit the oldest such page is given back for the new one
/// ([`Memory::give_back`]). 512 is fxceless's number (3e573d2: what its runs
/// came to hold from faults on a 12 GB card). Whether it is enough the
/// report says: pages given back and wanted again.
const FAULT_MB: usize = 512;
const FAULT_PAGES: usize = (FAULT_MB << 20) / PAGE as usize;
/// A page from a fault is not given back for so many frames after it came.
/// When more new pages are wanted than that lets through (all of
/// [`faults_most`] in under so many frames: rays that go anywhere), the new
/// ones wait, instead of the pages in use going round at [`FAULTS`] a frame.
const FAULT_YOUNG: u64 = 128;
/// A page given back and wanted again within so many frames was still read
/// (a submission lands up to two frames late).
const FAULT_SOON: u64 = 4;

/// Whether the pages taken from faults go round (`VICTORY_BVH_FAULTS_RECYCLE=0`:
/// they are kept for the rest of the run and counted for the whole run, as
/// before fxceless 3e573d2; then no more than [`faults_most`] are ever made,
/// and what is touched after that reads zeros for good).
fn recycle() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_BVH_FAULTS_RECYCLE").is_none_or(|v| v != "0"))
}

/// The number `VICTORY_BVH_FAULTS_ALL`, or else `VICTORY_BVH_FAULTS`, gives:
/// with [`recycle`], the arena pages held from faults at a time
/// ([`FAULT_PAGES`], 512 MB, unless given); without it, the pages taken from
/// faults in the whole run ([`FAULTS_ALL`] unless given). Held pages stay
/// under [`MOST`] either way.
fn faults_most() -> usize {
    static ALL: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ALL.get_or_init(|| {
        let given = std::env::var("VICTORY_BVH_FAULTS_ALL")
            .ok()
            .or_else(|| std::env::var("VICTORY_BVH_FAULTS").ok())
            .and_then(|v| v.trim().parse().ok());
        given.unwrap_or(if recycle() { FAULT_PAGES } else { FAULTS_ALL })
    })
}
/// Frames over which every page is looked at once.
const SWEEP: usize = 128;

/// With [`arenas_told`]: how many frames a page of an arena written here is
/// brought up by every list that comes to this memory, and how often (in
/// frames) the arenas are brought up whole anyway.
const TOLD_FRAMES: u64 = 8;
const TOLD_WHOLE: u64 = 64;
const TOLD_GAP: u64 = 16;

/// Whether a list brings up only the arena pages this module wrote in the
/// last [`TOLD_FRAMES`] frames, not every arena whole
/// (`VICTORY_ARENAS_TOLD=0`: whole, as before).
///
/// Only this module writes an arena (`bring`, `take`, a page newly held);
/// what shaders store there stays on the card. Brought up whole, each of
/// the 29 arenas of a ray-traced world (1.75 GB) was asked of the write
/// watch by itself, as a range of its own, once for every list that came
/// here: 290-300 kernel askings a frame, 6.9 ms of the submitting thread
/// (run rt-placed-fills-ab, 7900 XT), for some 180-680 KB that changed.
/// Every [`TOLD_WHOLE`] frames they are brought up whole all the same.
fn arenas_told() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_ARENAS_TOLD") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ARENAS_TOLD").is_none_or(|v| v != "0"))
}

/// What a draw or dispatch does with title memory by address.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Use {
    Reads,
    /// Stores by address, as the structure builder; `back`: its results in
    /// small title buffers come back.
    Builds { back: bool },
}

/// A page of title memory in an arena.
struct Held {
    /// Where in the arena's memory, and the arena's place in `copies`.
    at: usize,
    copy: usize,
    /// The write-watch asking after which it was last taken (0: never).
    taken: u64,
    /// The frame it was last looked at in, and the one it came in.
    looked: u64,
    born: u64,
    /// Which of all holdings this is (a page given back may come again).
    serial: u64,
    /// Taken from a fault and named by no list since: 1 if the programs that
    /// walk found it empty, 2 if the builder did (0: not so).
    fault: u8,
    /// Packed structures that have bytes in it, and whether it is held for
    /// them alone.
    owners: u32,
    theirs: bool,
}

/// A packed structure the builder came to.
struct Packed {
    /// The frame it last did.
    frame: u64,
    len: u64,
    /// Its first two dwords in title memory.
    header: u64,
}

struct Memory {
    /// The CPU's entries (number, offset), and the copy shaders read.
    directory: (*mut u32, Arc<Shadow>),
    faults: (ID3D12Resource, *mut u32, u64),
    /// Copy `n` is `copies[n - 1]`.
    copies: Vec<Arc<Shadow>>,
    /// Places of title buffers' copies among `copies`.
    numbers: HashMap<usize, usize>,
    /// Arenas: place in `copies`, address.
    arenas: Vec<(usize, usize)>,
    /// With [`arenas_told`]: per arena (its place in `copies`), the pages of
    /// it (offset / [`PAGE`]) this module wrote, and the last frame it did.
    told: HashMap<usize, std::collections::BTreeMap<usize, u64>>,
    /// By title page.
    pages: HashMap<u64, Held>,
    /// Their numbers and serials in the order they came, and where the sweep
    /// is.
    order: Vec<(u64, u64)>,
    sweep: usize,
    /// Windows of title memory itself ([`Memory::map`]), and their pages.
    windows: usize,
    windowed: Vec<u64>,
    /// Pages shown as title memory has them that share title buffers shaders
    /// write, with those buffers ([`Memory::shares`]).
    over: HashMap<u64, Vec<(u64, u64)>>,
    /// Packed structures the builder came to: the frame it last did.
    packed: HashMap<u64, Packed>,
    /// Their addresses in the order they came, and where the sweep is.
    packed_order: Vec<u64>,
    packed_sweep: usize,
    /// Places of pages that are free: the copy, the address in the arena.
    free: Vec<(usize, usize)>,
    /// Copies of title buffers the builder may store in.
    open: std::collections::HashSet<usize>,
    /// Pages taken from faults, in all.
    faulted: usize,
    /// Holdings so far ([`Held::serial`]).
    holdings: u64,
    /// The arena pages held from faults (the walking programs', the
    /// builder's) with their serials, in the order they came; how many of
    /// them are still so.
    fault_order: [VecDeque<(u64, u64)>; 2],
    fault_held: [usize; 2],
    /// Pages given back for others: the frame, and whether a shader was seen
    /// to want the page again within [`FAULT_SOON`] frames.
    given: HashMap<u64, (u64, bool)>,
    /// A bit per page above 4 GiB a shader found empty, in all and since the
    /// last report; how many are set.
    wanted: [Vec<u64>; 2],
    wanted_pages: [u64; 2],
    /// Whether the card has room for another arena, once asked this frame.
    room: Option<bool>,
    /// No room was found for a page from a fault this frame (not looked for
    /// again in it); the page last asked for got no place for want of room.
    full: bool,
    turned: bool,
    /// Per host view heap (by address): the first slot of its run, and how
    /// many copies have their view there.
    heaps: HashMap<usize, (u32, u32)>,
    /// The frame the faults were last read in.
    frame: u64,
    /// Pages asked for this frame that are not the title's memory.
    refused: std::collections::HashSet<u64>,
    /// Pages of title memory whose window (not the title buffer's own copy)
    /// the builder stores small results in ([`Memory::small`]), and that
    /// window's place in `copies`: where a copy of those results reads them
    /// ([`results`]).
    results: HashMap<u64, usize>,
    /// A page the current [`ready`] named found no number left for the copy
    /// that would hold it ([`Memory::number`]).
    short: bool,
    stats: Stats,
    #[cfg(feature = "diagnostics")]
    dump: dump::Dump,
    #[cfg(feature = "diagnostics")]
    trace: trace::Trace,
    #[cfg(feature = "diagnostics")]
    stores: stores::Stores,
}

/// Since the last report.
#[derive(Default)]
struct Stats {
    from: Option<(std::time::Instant, u64)>,
    uses: [u64; 2],
    faults: u64,
    unknown: u64,
    back: u64,
    /// Pages looked at, bytes that changed, packed structures sent.
    looked: u64,
    changed: u64,
    packed: u64,
    packed_bytes: u64,
    gone: u64,
    /// Pages that came for an address in constants, a written view, another
    /// view, a structure named in upload memory.
    came: [u64; 4],
    micros: u64,
    /// Runs of arena pages lists were given to bring up ([`arenas_told`]).
    told_runs: u64,
    /// The different pages faulted.
    distinct: std::collections::HashSet<u64>,
    /// Of the pages shaders found empty: made; left over the frame's limit,
    /// without room, not the title's; below 4 GiB; left by the builder.
    made: u64,
    left: [u64; 3],
    low: u64,
    builders: u64,
    /// Pages from faults given back for others, of them wanted again at
    /// once; gone with the title's memory; named by a list since.
    given: u64,
    again: u64,
    dead: u64,
    adopted: u64,
    /// Ranges copied from shader-written buffers into pages they share.
    mirrored: u64,
    /// Arrays of geometry records builder dispatches were given; pages that
    /// came for what they point to when the dispatch was recorded, and at
    /// its submission ([`Memory::land_geometry`]); ranges in title buffers
    /// whose copies were brought up to date.
    geometry: [u64; 4],
}

// SAFETY: the mappings are of memory the struct owns; all access is under
// the mutex.
unsafe impl Send for Memory {}

static MEMORY: Mutex<Option<Memory>> = Mutex::new(None);

/// What a draw binds.
pub(crate) struct Ready {
    pub(crate) directory: u64,
    /// The half of the notes this draw's shader leaves its faults in.
    pub(crate) faults: u64,
    /// The view heap slot of the null view; copy `n`'s view is `n` further.
    pub(crate) base: u32,
    /// No page the dispatch names was left without a copy for want of a
    /// number. False once the numbers have run out ([`COPIES`]): what it
    /// names then reads zeros, a dropped stage for the structure builder.
    pub(crate) whole: bool,
    /// (`VICTORY_BVH_STORES=1`) The dispatch's tag in the builder's store
    /// notes; 0 otherwise.
    pub(crate) tag: u32,
}

impl Memory {
    fn new() -> Option<Memory> {
        let host = host()?;
        let bytes = u64::from(MEMORY_PAGES) * 8;
        // Zeroed, and watched for writes as the title's memory is.
        let entries = crate::upload::reserve(bytes as usize)?;
        let directory = crate::shadow::by_address(entries as u64, bytes)?;
        crate::host::name(&directory.buffer, || "title memory by address: directory".into());
        // Twice the notes a shader knows of: the programs that walk leave
        // theirs in the first half, the builder in the second ([`ready`]).
        // After them the builder's store notes (`VICTORY_BVH_STORES=1`).
        let notes =
            MEMORY_FAULTS as usize * 8 + victory_shader::emit::MEMORY_STORE_NOTES as usize * 4;
        let (faults, data) = host.shared_buffer(notes as u64)?;
        crate::host::name(&faults, || "title memory by address: faults".into());
        // SAFETY: a fresh mapping of at least that many bytes.
        unsafe { std::ptr::write_bytes(data, 0, notes) };
        // SAFETY: a live buffer.
        let gpu = unsafe { faults.GetGPUVirtualAddress() };
        tracing::info!("title memory by address: a shader reads or writes through raw addresses");
        Some(Memory {
            directory: (entries as *mut u32, directory),
            faults: (faults, data.cast::<u32>(), gpu),
            copies: Vec::new(),
            numbers: HashMap::new(),
            arenas: Vec::new(),
            told: HashMap::new(),
            pages: HashMap::new(),
            order: Vec::new(),
            sweep: 0,
            windows: 0,
            windowed: Vec::new(),
            over: HashMap::new(),
            packed: HashMap::new(),
            packed_order: Vec::new(),
            packed_sweep: 0,
            free: Vec::new(),
            open: Default::default(),
            faulted: 0,
            holdings: 0,
            fault_order: Default::default(),
            fault_held: [0; 2],
            given: HashMap::new(),
            wanted: [vec![0; MEMORY_PAGES as usize / 64], vec![0; MEMORY_PAGES as usize / 64]],
            wanted_pages: [0; 2],
            room: None,
            full: false,
            turned: false,
            heaps: HashMap::new(),
            frame: u64::MAX,
            refused: Default::default(),
            results: HashMap::new(),
            short: false,
            stats: Stats::default(),
            #[cfg(feature = "diagnostics")]
            dump: Default::default(),
            #[cfg(feature = "diagnostics")]
            trace: Default::default(),
            #[cfg(feature = "diagnostics")]
            stores: Default::default(),
        })
    }

    /// The directory's entry of `page`.
    fn entry(&self, page: u64) -> *mut u32 {
        debug_assert!(page < u64::from(MEMORY_PAGES));
        // SAFETY: inside the directory.
        unsafe { self.directory.0.add(page as usize * 2) }
    }

    /// The place in `copies` of the copy that holds `page`, if one does.
    fn copy_of(&self, page: u64) -> Option<usize> {
        if page >= u64::from(MEMORY_PAGES) {
            return None;
        }
        // SAFETY: an entry of the directory.
        ((unsafe { self.entry(page).read() } & !MEMORY_READ_ONLY) as usize).checked_sub(1)
    }

    /// Whether `page` can be given a place: in the directory, not turned down
    /// this frame, and the title's GPU memory. (A shader's pointer, or a
    /// constant taken for an address, may be anything.)
    fn possible(&mut self, page: u64) -> bool {
        // Not the last entry: addresses past the directory read it, and it
        // stays empty.
        if page < 0x1_0000 || page >= u64::from(MEMORY_PAGES) - 1 || self.refused.contains(&page) {
            return false;
        }
        let at = (page * PAGE) as usize;
        let ours = crate::texture::committed(at as *const u8, PAGE as usize)
            && crate::watch::writes(at, at + PAGE as usize, true).is_some();
        if !ours {
            self.refused.insert(page);
            self.stats.unknown += 1;
        }
        ours
    }

    /// Gives a copy a number, if one is left.
    fn number(&mut self, shadow: Arc<Shadow>) -> Option<usize> {
        let key = Arc::as_ptr(&shadow) as usize;
        if let Some(&n) = self.numbers.get(&key) {
            return Some(n);
        }
        if self.copies.len() >= COPIES as usize {
            self.short = true;
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                let (arenas, windows) = (self.arenas.len(), self.windows);
                tracing::warn!(
                    COPIES,
                    arenas,
                    windows,
                    title_buffers = self.copies.len().saturating_sub(arenas + windows),
                    "title memory by address: no number left for another copy; its pages read zeros, and a stage of the structure builder that names one turns the builder off (numbered so far, by what they are)"
                )
            });
            return None;
        }
        // SAFETY: a live buffer.
        let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
        if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
            return None;
        }
        self.copies.push(shadow);
        let n = self.copies.len() - 1;
        self.numbers.insert(key, n);
        Some(n)
    }

    /// Gives `page` a place for a list that names it, if it has none and the
    /// memory is the title's; whether it has one then.
    fn map(&mut self, page: u64) -> bool {
        self.map_for(page, None)
    }

    /// [`Memory::map`]; `fault`: for a page a shader found empty (whether the
    /// builder did), which takes an arena's place only within
    /// [`Memory::fault_room`].
    fn map_for(&mut self, page: u64, fault: Option<bool>) -> bool {
        if self.copy_of(page).is_some() {
            if fault.is_none() {
                self.adopt(page);
            }
            return true;
        }
        if !self.possible(page) {
            return false;
        }
        let start = page * PAGE;
        if crate::shadow::in_buffer(start, PAGE).is_some() {
            let Some(shadow) = crate::shadow::find(start, PAGE) else {
                return false;
            };
            let fresh = !self.numbers.contains_key(&(Arc::as_ptr(&shadow) as usize));
            let Some(n) = self.number(shadow.clone()) else {
                return false;
            };
            // Every page the copy holds whole, not only the one asked for.
            // shortcut: a page of a title buffer inside this one that has a
            // copy of its own is given this copy too.
            if fresh {
                let (first, len) = shadow.span();
                let mut at = first.next_multiple_of(PAGE);
                while at + PAGE <= first + len && at / PAGE < u64::from(MEMORY_PAGES) - 1 {
                    let entry = self.entry(at / PAGE);
                    // SAFETY: an entry of the directory.
                    if unsafe { entry.read() } == 0 {
                        close(entry, n, at - first);
                    }
                    at += PAGE;
                }
            }
            // SAFETY: as above.
            if unsafe { self.entry(page).read() } == 0 {
                close(self.entry(page), n, start - shadow.span().0);
            }
            return true;
        }
        #[cfg(feature = "diagnostics")]
        held_over_written(start);
        // Small title buffers in it: what the builder stores in them comes
        // back, so title memory itself.
        if self.windows < WINDOWS && crate::shadow::small_buffer_in(start, PAGE, BACK) {
            if let Some(shadow) = crate::shadow::by_address(start, PAGE)
                && let Some(n) = self.number(shadow)
            {
                self.windows += 1;
                fill(self.entry(page), n, 0);
                self.windowed.push(page);
                self.shares(page);
                return true;
            }
            return false;
        }
        if fault.is_some() && (self.full || !self.fault_room()) {
            (self.full, self.turned) = (true, true);
            return false;
        }
        if self.pages.len() >= MOST {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(MOST, "title memory by address: no more pages are taken; others read zeros"));
            return false;
        }
        if self.free.is_empty() && !self.another() {
            return false;
        }
        let Some((copy, at)) = self.free.pop() else {
            return false;
        };
        self.hold(page, at, copy);
        if let Some(builds) = fault
            && let Some(held) = self.pages.get_mut(&page)
        {
            let kind = usize::from(builds);
            held.fault = kind as u8 + 1;
            self.fault_order[kind].push_back((page, held.serial));
            self.fault_held[kind] += 1;
        }
        true
    }

    /// A list names `page`, or a packed structure is unpacked in it: if it
    /// was held from a fault it is not given back any more (it may hold what
    /// is on the card alone, as any page a list names).
    fn adopt(&mut self, page: u64) {
        if self.fault_held == [0, 0] {
            return;
        }
        if let Some(held) = self.pages.get_mut(&page)
            && held.fault != 0
        {
            self.fault_held[usize::from(held.fault) - 1] -= 1;
            held.fault = 0;
            self.stats.adopted += 1;
        }
    }

    /// Whether the card has room for another arena, by the rule its copy is
    /// made by ([`crate::shadow::by_address`]: under 97 hundredths of the
    /// budget; past it the arena lies in the PC's memory and every step of
    /// every ray crosses the bus). Asked once a frame.
    fn card_room(&mut self) -> bool {
        *self.room.get_or_insert_with(|| {
            host().and_then(|h| h.video_memory()).is_none_or(|[budget, used, ..]| used + ARENA < budget / 100 * 97)
        })
    }

    /// Room for one more page from a fault: under [`faults_most`], while the
    /// card has room for the arena it may come to take; else the place of
    /// the oldest such page.
    fn fault_room(&mut self) -> bool {
        (self.fault_held[0] + self.fault_held[1] < faults_most() && self.card_room()) || self.give_back()
    }

    /// Gives back the oldest page held from a fault, unless it is younger
    /// than [`FAULT_YOUNG`] frames. Whether it is still read is not known (a
    /// present page leaves no note): if it is, its note comes back with the
    /// next frame and [`Memory::take_faults`] makes it again, first.
    fn give_back(&mut self) -> bool {
        let kind = goes_first(self.fault_held, faults_most());
        while let Some(&(page, serial)) = self.fault_order[kind].front() {
            let Some(born) = self.pages.get(&page).filter(|h| h.serial == serial && h.fault != 0).map(|h| h.born) else {
                // Named by a list since, or gone.
                self.fault_order[kind].pop_front();
                continue;
            };
            if self.frame.wrapping_sub(born) < FAULT_YOUNG {
                return false;
            }
            self.fault_order[kind].pop_front();
            self.let_go(page);
            self.given.insert(page, (self.frame, false));
            self.stats.given += 1;
            return true;
        }
        false
    }

    /// `page`, held from a fault, is given up: its place is free and shaders
    /// find it empty again.
    fn let_go(&mut self, page: u64) {
        let Some(held) = self.pages.remove(&page) else { return };
        if held.fault != 0 {
            self.fault_held[usize::from(held.fault) - 1] -= 1;
        }
        self.free.push((held.copy, held.at));
        // (Its place is another page's from here on.)
        self.over.remove(&page);
        // SAFETY: an entry of the directory, under the mutex.
        unsafe { self.entry(page).write_volatile(0) };
    }

    /// The age in frames of the oldest page held from a fault.
    fn oldest(&mut self) -> u64 {
        let mut age = 0;
        for kind in 0..2 {
            while let Some(&(page, serial)) = self.fault_order[kind].front() {
                match self.pages.get(&page) {
                    Some(h) if h.serial == serial && h.fault != 0 => {
                        age = age.max(self.frame.wrapping_sub(h.born));
                        break;
                    }
                    _ => {
                        self.fault_order[kind].pop_front();
                    }
                }
            }
        }
        age
    }

    /// Another arena's pages.
    fn another(&mut self) -> bool {
        let Some(at) = crate::upload::reserve(ARENA as usize) else {
            return false;
        };
        let Some(n) = crate::shadow::by_address(at as u64, ARENA).and_then(|s| self.number(s)) else {
            return false;
        };
        self.arenas.push((n, at));
        // The lowest first.
        self.free.extend((0..(ARENA / PAGE) as usize).rev().map(|i| (n, at + i * PAGE as usize)));
        true
    }

    /// `page` is held at `at` in the arena that is `copy`.
    fn hold(&mut self, page: u64, at: usize, copy: usize) {
        self.holdings += 1;
        let serial = self.holdings;
        self.pages.insert(
            page,
            Held { at, copy, taken: 0, looked: u64::MAX, born: self.frame, serial, fault: 0, owners: 0, theirs: false },
        );
        self.order.push((page, serial));
        fill(self.entry(page), copy, at as u64 - self.copies[copy].span().0);
        self.shares(page);
    }

    /// (`VICTORY_BVH_MIRROR=1`) Notes whether `page`, which by-address
    /// readers see as title memory has it (an arena's page, a window), lies
    /// partly in title buffers whose own copies shaders write: those bytes
    /// are on the card in the buffer's copy alone until they land in title
    /// memory, flights later. Run bvhcheck-2 found 12 such pages, among them
    /// the first page of 2 and 6 MiB buffers shaders write; moving objects'
    /// structures were built from vertices that were not there, and their
    /// boxes reached 3.4e38 (bvhcheck-1 to -3).
    fn shares(&mut self, page: u64) {
        if !mirror() {
            return;
        }
        let found = crate::shadow::written_across(page * PAGE, PAGE);
        if found.is_empty() {
            self.over.remove(&page);
        } else if self.over.insert(page, found).is_none() && self.over.len() <= 200 {
            tracing::info!(
                page = format_args!("{:#x}", page * PAGE),
                pages = self.over.len(),
                "title memory by address: a page shares title buffers shaders write; their bytes are copied into it on the card before the builder reads (VICTORY_BVH_MIRROR)"
            );
        }
    }

    /// (`VICTORY_BVH_MIRROR=1`) In `list`, before its first builder dispatch:
    /// the bytes of [`Memory::shares`]' buffers go from their own copies into
    /// the by-address copies of the pages they share.
    fn mirror(&mut self, list: &HostList) {
        let mut copies = 0;
        for (&page, buffers) in &self.over {
            let Some(k) = self.copy_of(page) else { continue };
            let to = &self.copies[k];
            // An arena's page lies at its place in the arena's memory; a
            // window is title memory itself.
            let base = self.pages.get(&page).map_or(page * PAGE, |held| held.at as u64);
            for &(start, len) in buffers {
                let Some(from) = crate::shadow::peek(start, 1).filter(|s| s.written() && !s.in_place()) else { continue };
                let (lo, hi) = (start.max(page * PAGE), (start + len).min((page + 1) * PAGE));
                if lo < hi && list.mirror_copy(&from, lo, to, base + (lo - page * PAGE), hi - lo) {
                    copies += 1;
                }
            }
        }
        self.stats.mirrored += copies;
    }

    /// With [`arenas_told`]: the page at `at` of the arena that is `copy` was
    /// written (or newly held) in this frame, so the lists recorded over the
    /// next [`TOLD_FRAMES`] frames bring it up.
    fn tell(&mut self, copy: usize, at: usize) {
        // (Noted with the switch off too: `VICTORY_AB` turns it on and off.)
        let offset = (at - self.copies[copy].span().0 as usize) / PAGE as usize;
        self.told.entry(copy).or_default().insert(offset, self.frame);
    }

    /// The runs (byte offsets) of the pages of the arena that is `copy` told
    /// in the last [`TOLD_FRAMES`] frames; older notes go.
    fn told_runs(&mut self, copy: usize) -> Vec<(u64, u64)> {
        let frame = self.frame;
        let Some(pages) = self.told.get_mut(&copy) else {
            return Vec::new();
        };
        pages.retain(|_, at| at.saturating_add(TOLD_FRAMES) >= frame);
        let mut runs: Vec<(u64, u64)> = Vec::new();
        for &page in pages.keys() {
            let (from, to) = (page as u64 * PAGE, (page as u64 + 1) * PAGE);
            match runs.last_mut() {
                // Gaps of up to [`TOLD_GAP`] pages joined: fewer, longer refreshes.
                Some(last) if from <= last.1 + TOLD_GAP * PAGE => last.1 = to,
                _ => runs.push((from, to)),
            }
        }
        runs
    }

    /// The packed structure at `address` is gone: the pages it alone had are
    /// given back.
    fn gone(&mut self, address: u64) {
        let Some(packed) = self.packed.remove(&address) else { return };
        for page in address / PAGE..=(address + packed.len - 1) / PAGE {
            let Some(held) = self.pages.get_mut(&page) else { continue };
            held.owners = held.owners.saturating_sub(1);
            if held.owners == 0 && held.theirs {
                self.free.push((held.copy, held.at));
                self.pages.remove(&page);
                // SAFETY: an entry of the directory, under the mutex.
                unsafe { self.entry(page).write_volatile(0) };
            }
        }
        self.stats.gone += 1;
    }

    /// Whether the packed structure at `address` is still in title memory.
    fn there(&self, address: u64) -> bool {
        let mut header = [0u8; 8];
        crate::texture::committed(address as *const u8, 8) && {
            crate::left::before_take(address as usize, address as usize + 8, 0);
            crate::host::copy_mapped(header.as_mut_ptr(), address as *const u8, 8, false);
            self.packed.get(&address).is_some_and(|p| p.header == u64::from_le_bytes(header))
        }
    }

    /// Looks at some of the pages and packed structures again.
    fn look_again(&mut self) {
        // (Buffers come, and are first written by shaders, after a page did.)
        if mirror() && self.frame.is_multiple_of(16) {
            for page in self.windowed.clone() {
                self.shares(page);
            }
        }
        for _ in 0..self.order.len().div_ceil(SWEEP) {
            self.sweep = (self.sweep + 1) % self.order.len();
            let (page, serial) = self.order[self.sweep];
            let fault = self.pages.get(&page).filter(|h| h.serial == serial).map(|h| h.fault);
            // The title's memory under a page from a fault is gone: so is
            // what was read there. (A question to the system: put for a
            // page every fourth time the sweep comes to it.)
            let dead = fault.is_some_and(|f| f != 0)
                && (self.frame / SWEEP as u64 + page).is_multiple_of(4)
                && !crate::texture::committed((page * PAGE) as *const u8, PAGE as usize);
            if dead {
                self.let_go(page);
                self.stats.dead += 1;
            }
            if fault.is_some() && !dead {
                self.shares(page);
                self.bring(page, page + 1);
            } else {
                self.order.swap_remove(self.sweep);
                if self.order.is_empty() {
                    break;
                }
            }
        }
        for _ in 0..self.packed_order.len().div_ceil(SWEEP) {
            self.packed_sweep = (self.packed_sweep + 1) % self.packed_order.len();
            let address = self.packed_order[self.packed_sweep];
            if !self.packed.contains_key(&address) {
                self.packed_order.swap_remove(self.packed_sweep);
                if self.packed_order.is_empty() {
                    break;
                }
            } else if !self.there(address) {
                self.gone(address);
            }
        }
    }

    /// A dispatch of the builder was given `address` to write: if it is in a
    /// title buffer's copy, stores by address reach that from here on.
    fn opens(&mut self, address: u64) {
        let Some(n) = self.map(address / PAGE).then(|| self.copy_of(address / PAGE)).flatten() else {
            return;
        };
        if self.pages.contains_key(&(address / PAGE)) || !self.open.insert(n) {
            return;
        }
        // (Its pages hold what the CPU writes beside the builder's scratch.)
        self.copies[n].keep_stores();
        let (first, len) = self.copies[n].span();
        tracing::info!(
            start = format_args!("{first:#x}"),
            len,
            "title memory by address: the structure builder may store in a title buffer"
        );
        for page in first.div_ceil(PAGE)..(first + len) / PAGE {
            if self.copy_of(page) == Some(n) {
                fill(self.entry(page), n, page * PAGE - first);
            }
        }
    }

    /// [`Memory::map`] of the pages of `[address, address + len)`, up to the
    /// first that is not the title's, and what the title wrote there since
    /// they were taken goes into the arenas.
    fn ready(&mut self, address: u64, len: u64) {
        let first = address / PAGE;
        let mut end = first;
        while end * PAGE < address + len && self.map(end) {
            end += 1;
        }
        self.bring(first, end);
    }

    /// Takes the pages `[first, end)` that are in arenas from title memory
    /// where it wrote them.
    fn bring(&mut self, first: u64, end: u64) {
        // Once a frame: the builder names the same pages list after list.
        let frame = self.frame;
        if !(first..end).any(|page| self.pages.get(&page).is_some_and(|h| h.looked != frame)) {
            return;
        }
        let started = std::time::Instant::now();
        let answer = crate::watch::writes((first * PAGE) as usize, (end * PAGE) as usize, true);
        let mut wrote_here: Vec<(usize, usize)> = Vec::new();
        for page in first..end {
            let Some(held) = self.pages.get_mut(&page) else {
                continue;
            };
            if held.looked == frame {
                continue;
            }
            held.looked = frame;
            self.stats.looked += 1;
            let wrote = answer
                .as_ref()
                .map(|(_, w)| w.binary_search_by_key(&(page as usize), |w| w.0).map_or(0, |i| w[i].1));
            let now = answer.as_ref().map_or(1, |a| a.0);
            let stale = held.taken == 0 || wrote.is_none_or(|w| w > held.taken);
            if !stale {
                continue;
            }
            let never = std::mem::replace(&mut held.taken, now.max(1)) == 0;
            let shadow = &self.copies[held.copy];
            let base = held.at - shadow.span().0 as usize;
            let mut title = [0u8; PIECE];
            let mut changed = 0;
            // Pieces of buffer copies left on the card there first
            // (`crate::left`).
            crate::left::before_take((page * PAGE) as usize, ((page + 1) * PAGE) as usize, 0);
            for piece in (0..PAGE as usize).step_by(PIECE) {
                let from = (page * PAGE) as usize + piece;
                // (Pages that are not there read as zeros.)
                crate::host::copy_mapped(title.as_mut_ptr(), from as *const u8, PIECE, false);
                // SAFETY: a page of an arena of ours, under the mutex.
                let ours = unsafe { std::slice::from_raw_parts_mut((held.at + piece) as *mut u8, PIECE) };
                changed += merge(&title, ours, never) as u64 * 4;
            }
            self.stats.changed += changed;
            if never {
                shadow.resend(base, base + PAGE as usize);
            }
            if never || changed > 0 {
                wrote_here.push((held.copy, held.at));
            }
        }
        for (copy, at) in wrote_here {
            self.tell(copy, at);
        }
        self.stats.micros += started.elapsed().as_micros() as u64;
    }

    /// The bytes of a packed structure title memory has at `address` by now,
    /// and its first two dwords.
    fn packed_at(&self, address: u64) -> Option<(u64, u64)> {
        if address & 0xFF != 0 {
            return None;
        }
        let mut header = [0u8; 8];
        crate::left::before_take(address as usize, address as usize + 8, 0);
        crate::host::copy_mapped(header.as_mut_ptr(), address as *const u8, 8, false);
        let header = u64::from_le_bytes(header);
        Some((packed_len(header)?, header))
    }

    /// At the submission of a list whose builder dispatch was given `address`
    /// outside buffers, in the order of its copies: what title memory has
    /// there now goes up. When the dispatch is recorded it may not be there
    /// yet: the title's queue waits for its streaming. (Nothing is asked of
    /// the write watch here: the submission's asking stands.)
    fn land(&mut self, address: u64) {
        let page = address / PAGE;
        if !crate::texture::committed((page * PAGE) as *const u8, PAGE as usize) {
            return;
        }
        let Some((len, header)) = self.packed_at(address) else {
            // Anything else: what came since its page was first taken.
            for page in page..=(address + VIEW - 1) / PAGE {
                if self.pages.get(&page).is_some_and(|h| h.born + YOUNG >= self.frame) {
                    self.take(page, 0, PAGE as usize, false);
                }
            }
            return;
        };
        if let Some(known) = self.packed.get_mut(&address) {
            let soon = known.frame + AGAIN >= self.frame;
            known.frame = self.frame;
            if known.header == header && soon {
                return;
            }
            if known.header != header {
                self.gone(address);
            }
        }
        let last = (address + len - 1) / PAGE;
        if !self.packed.contains_key(&address) {
            self.packed.insert(address, Packed { frame: self.frame, len, header });
            self.packed_order.push(address);
            for page in page..=last {
                let new = self.copy_of(page).is_none();
                if new && !self.place(page) {
                    continue;
                }
                if let Some(held) = self.pages.get_mut(&page) {
                    held.owners += 1;
                    held.theirs |= new;
                }
                // (Unpacked in place, on the card alone.)
                self.adopt(page);
            }
            self.stats.packed_bytes += len;
        }
        for page in page..=last {
            let from = address.max(page * PAGE) - page * PAGE;
            let to = (address + len).min((page + 1) * PAGE) - page * PAGE;
            self.take(page, from as usize, to as usize, true);
        }
        self.stats.packed += 1;
    }

    /// A place in an arena for a page of committed title memory, at a
    /// submission: only one there is already (the list brings up the arenas
    /// it knew when it was recorded).
    fn place(&mut self, page: u64) -> bool {
        if page < 0x1_0000
            || page >= u64::from(MEMORY_PAGES) - 1
            || !crate::texture::committed((page * PAGE) as *const u8, PAGE as usize)
            || crate::shadow::in_buffer(page * PAGE, PAGE).is_some()
        {
            return false;
        }
        let Some((copy, at)) = self.free.pop() else {
            return false;
        };
        self.hold(page, at, copy);
        crate::watch::wrote(self.entry(page) as usize, 8);
        self.take(page, 0, PAGE as usize, true);
        true
    }

    /// At the submission of a list whose builder dispatch was given the
    /// geometry records at `at`: the pages they point to that have no copy
    /// get one, and those that came lately are taken again (the buffers may
    /// have been streamed in since the dispatch was recorded). Read wrong
    /// once, the triangles of a packed structure are unpacked so for good.
    fn land_geometry(&mut self, at: u64) {
        for (start, len) in geometry(read_title, at) {
            for page in start / PAGE..=(start + len - 1) / PAGE {
                if self.copy_of(page).is_none() {
                    if self.place(page) {
                        self.stats.geometry[2] += 1;
                    }
                } else if self
                    .pages
                    .get(&page)
                    .is_some_and(|h| h.born + YOUNG >= self.frame)
                {
                    self.take(page, 0, PAGE as usize, false);
                }
            }
        }
    }

    /// Bytes `from..to` of `page` from title memory into its arena's page, at
    /// a submission; `whole`: sent whether they differ from those last taken
    /// or not.
    fn take(&mut self, page: u64, from: usize, to: usize, whole: bool) {
        let Some(held) = self.pages.get(&page) else { return };
        let shadow = &self.copies[held.copy];
        let base = held.at - shadow.span().0 as usize;
        let mut title = [0u8; PIECE];
        let mut at = from;
        let mut changed = false;
        // Pieces of buffer copies left on the card there first (`crate::left`).
        crate::left::before_take(page as usize * PAGE as usize + from, page as usize * PAGE as usize + to, 0);
        while at < to {
            let n = (PIECE - at % PIECE).min(to - at);
            crate::host::copy_mapped(title.as_mut_ptr(), (page as usize * PAGE as usize + at) as *const u8, n, false);
            // SAFETY: a page of an arena of ours, under the mutex.
            let ours = unsafe { std::slice::from_raw_parts_mut((held.at + at) as *mut u8, n) };
            if *ours != title[..n] {
                ours.copy_from_slice(&title[..n]);
                crate::watch::wrote(held.at + at, n);
                changed = true;
            }
            at += n;
        }
        if whole {
            shadow.resend(base + from, base + to);
        }
        if changed || whole {
            let (copy, at) = (held.copy, held.at);
            self.tell(copy, at);
        }
    }

    /// The pages shaders found empty since the last frame: those that are
    /// the title's get a place, at most [`FAULTS`] a frame, and at most
    /// [`faults_most`] arena pages are held so (with [`recycle`] off: are
    /// ever made so).
    fn take_faults(&mut self) {
        let mut raw = Vec::new();
        // (Dword 0 of each half is a sink.)
        for i in (1..2 * MEMORY_FAULTS as usize).filter(|&i| i != MEMORY_FAULTS as usize) {
            // SAFETY: inside the faults' mapping; the GPU may be writing,
            // a number missed now is written again next frame.
            let at = unsafe { self.faults.1.add(i) };
            // SAFETY: as above.
            let page = unsafe { at.read_volatile() };
            if page != 0 {
                raw.push((u64::from(page), i > MEMORY_FAULTS as usize));
                // SAFETY: as above.
                unsafe { at.write_volatile(0) };
            }
        }
        if raw.is_empty() {
            return;
        }
        let mut pages = noted(raw);
        let builders = pages.iter().filter(|p| p.1).count();
        // Below 4 GiB there is no title memory a shader reads by address: a
        // pointer made of zeros, or of something that is no pointer.
        let low = pages.partition_point(|p| p.0 < 0x1_0000);
        let frame = self.frame;
        let mut again = 0;
        for &(page, _) in &pages[low..] {
            self.stats.distinct.insert(page);
            for i in 0..2 {
                self.wanted_pages[i] += u64::from(mark(&mut self.wanted[i], page));
            }
            if let Some(given) = self.given.get_mut(&page)
                && !given.1
                && frame.wrapping_sub(given.0) <= FAULT_SOON
            {
                given.1 = true;
                again += 1;
            }
        }
        // Each is an access that read 0 or was dropped.
        static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let said = SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let say = said < 40 || said.is_multiple_of(100);
        // (Of those above 4 GiB: the others are counted, and tell nothing.)
        let some: Vec<String> = pages[low..]
            .iter()
            .step_by(pages[low..].len().div_ceil(12).max(1))
            .filter(|_| say)
            .map(|&(p, _)| format!("{:#x}", p * PAGE))
            .collect();
        // Those given back and wanted again first: they are read.
        if !self.given.is_empty() {
            let given = &self.given;
            pages.sort_by_cached_key(|p| !given.contains_key(&p.0));
        }
        let (mut made, gave) = (0, self.stats.given);
        // The old way: kept for the run, and no more than so many in it.
        let (round, most) = (recycle(), faults_most());
        // Why the others got none: the frame's limit; no room (the limit of
        // pages from faults with its oldest too young to go, or the card's
        // memory: only arena pages, a title buffer's copy takes none; the
        // old way, the run's count); not the title's memory; no place (a
        // copy that cannot be numbered, no arena).
        let mut left = [0u32; 4];
        for &(page, builds) in &pages {
            if self.copy_of(page).is_some() {
                continue;
            }
            if page < 0x1_0000 || page >= u64::from(MEMORY_PAGES) - 1 || self.refused.contains(&page) {
                left[2] += 1;
            } else if made >= FAULTS {
                left[0] += 1;
            } else if !round && self.faulted >= most {
                left[1] += 1;
            } else if self.map_for(page, round.then_some(builds)) {
                made += 1;
                self.faulted += 1;
                self.given.remove(&page);
                self.bring(page, page + 1);
            } else if self.refused.contains(&page) {
                left[2] += 1;
            } else if std::mem::take(&mut self.turned) {
                left[1] += 1;
            } else {
                left[3] += 1;
            }
        }
        let s = &mut self.stats;
        s.faults += pages.len() as u64;
        s.made += made as u64;
        for (sum, n) in s.left.iter_mut().zip(left) {
            *sum += u64::from(n);
        }
        s.low += low as u64;
        s.builders += builders as u64;
        s.again += again;
        // The first frames that leave pages without a copy for want of room:
        // all of them, and which of them hold a packed structure's first
        // bytes or lie in a title buffer (to tell walked structures from
        // stray pointers).
        static WHOLE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if left[1] > 0 && WHOLE.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 3 {
            let mut text = String::new();
            for &(page, _) in pages.iter().filter(|p| p.0 >= 0x1_0000 && self.copy_of(p.0).is_none()) {
                let at = page * PAGE;
                let committed = crate::texture::committed(at as *const u8, PAGE as usize);
                let buffer = crate::shadow::in_buffer(at, 1).is_some();
                let packed = committed && self.packed_at(at).is_some();
                text += &format!(" {at:#x}{}{}{}", if committed { "" } else { "?" }, if buffer { "b" } else { "" }, if packed { "p" } else { "" });
            }
            tracing::warn!(frame = self.frame, "title memory by address: pages left without a copy by the bounds (?: not committed, b: in a title buffer, p: a packed structure there):{text}");
        }
        if say {
            let (given_back, oldest) = (self.stats.given - gave, self.oldest());
            tracing::warn!(
                frame = self.frame,
                pages = pages.len(),
                made,
                held = self.pages.len(),
                over_the_frame = left[0],
                no_room = left[1],
                not_the_titles = left[2],
                no_place = left[3],
                below_4_gib = low,
                by_the_builder = builders,
                from_faults = self.fault_held[0] + self.fault_held[1],
                given_back,
                wanted_again = again,
                oldest,
                ?some,
                "title memory by address: shaders touched pages that had no copy (those accesses read 0 or were dropped); those that are the title's are there now"
            );
        }
    }

    /// The structures named in a list's upload memory around `address`
    /// (instances of a top-level build: a structure's address is their last
    /// 8 bytes): the first page of each.
    fn named_in_upload(&mut self, address: u64) {
        let Some((from, len)) = crate::upload::block(address as usize) else {
            return;
        };
        // SAFETY: the block is handed out and stays mapped; the title wrote
        // it before it recorded the dispatch.
        let words = unsafe { std::slice::from_raw_parts(from as *const u64, len / 8) };
        let mut seen = std::collections::HashSet::new();
        for &word in words {
            if word & 0xFF == 0 && (1 << 32..1 << 37).contains(&word) && seen.insert(word / PAGE) && self.map(word / PAGE) {
                let page = word / PAGE;
                self.bring(page, page + 1);
            }
        }
    }

    /// A small title buffer whose results come back, if its pages are
    /// windows of title memory itself (or a copy of the buffer's own).
    fn small(&mut self, list: &HostList, start: u64, len: u64) {
        let mut at = start;
        while at < start + len {
            let page = at / PAGE;
            let n = ((page + 1) * PAGE).min(start + len) - at;
            self.map(page);
            if let Some(k) = self.copy_of(page)
                && !self.pages.contains_key(&page)
            {
                self.copies[k].refresh(list, at, n);
                self.copies[k].gpu_writes(list, at, n);
                self.stats.back += 1;
                // A window of title memory around the buffer: the buffer's
                // own copy, which the title's copies read, is another one.
                let own = crate::shadow::peek(at, n).is_some_and(|s| Arc::ptr_eq(&s, &self.copies[k]));
                if !own && self.results.insert(page, k).is_none() {
                    RESULTS.store(true, std::sync::atomic::Ordering::Relaxed);
                    tracing::info!(
                        page = format_args!("{:#x}", page * PAGE),
                        "title memory by address: the structure builder's results lie in a window; copies of them read it"
                    );
                }
            }
            at += n;
        }
    }

    /// Every copy is brought up to date when `list` is submitted, after what
    /// it deferred so far (once until a copy of the title's intervenes).
    fn bring_up(&mut self, list: &HostList) {
        let (start, len) = self.directory.1.span();
        self.directory.1.refresh(list, start, len);
        // With `arenas_told`: an arena's pages this module wrote lately, and
        // all of it every `TOLD_WHOLE` frames.
        let whole = !arenas_told() || self.frame.is_multiple_of(TOLD_WHOLE);
        for n in 0..self.copies.len() {
            let shadow = self.copies[n].clone();
            let (start, len) = shadow.span();
            if whole || !self.arenas.iter().any(|a| a.0 == n) {
                shadow.refresh(list, start, len);
                continue;
            }
            let runs = self.told_runs(n);
            self.stats.told_runs += runs.len() as u64;
            for (from, to) in runs {
                shadow.refresh(list, start + from, (to - from).min(len.saturating_sub(from)));
            }
        }
    }

    /// Every ten seconds or so, what was used.
    fn report(&mut self) {
        let now = std::time::Instant::now();
        let (since, frame) = *self.stats.from.get_or_insert((now, self.frame));
        if now.duration_since(since).as_secs_f64() < 10.0 {
            return;
        }
        let frames = self.frame.wrapping_sub(frame).max(1) as f64;
        let oldest = self.oldest();
        let wanted = self.wanted_pages;
        self.wanted[1].fill(0);
        self.wanted_pages[1] = 0;
        // (What is not asked about any more.)
        let now_frame = self.frame;
        self.given.retain(|_, given| now_frame.wrapping_sub(given.0) < 2 * FAULT_YOUNG);
        for kind in 0..2 {
            if self.fault_order[kind].len() > 2 * self.fault_held[kind] + 4096 {
                let pages = &self.pages;
                self.fault_order[kind].retain(|&(p, serial)| pages.get(&p).is_some_and(|h| h.serial == serial && h.fault != 0));
            }
        }
        let s = std::mem::take(&mut self.stats);
        tracing::info!(
            "title memory by address: {} pages ({} MB) in {} arenas, {} copies in all; per frame {:.0} dispatches that read, {:.0} that build, \
             {:.1} pages that had no copy, {:.1} pages turned down, {:.1} results brought back; {:.0} pages looked at, {:.1} KB changed, \
             {:.1} packed structures ({} there), {:.2} ms; {} windows; since the last line {} KB of packed structures came and {} of them went, \
             and pages for constants {}, views written {}, others {}, upload memory {}, faults in all {}; {:.1} runs of arena pages brought up \
             (VICTORY_ARENAS_TOLD=0: arenas whole); since the last line {} different pages faulted; {:.1} ranges a frame mirrored from buffers shaders write (VICTORY_BVH_MIRROR=1); \
             from faults {} pages held ({} MB of at most {}{}: the walking programs' {}, the builder's {}), the oldest {} frames old; \
             since the last line {} made, {} given back for others ({} of them wanted again within {} frames), \
             {} gone with the title's memory, {} named by a list since; per frame {:.1} not the title's ({:.1} of them below 4 GiB), \
             {:.1} over the frame's limit, {:.1} without room, {:.1} left by the builder; \
             pages wanted above 4 GiB: {} since the last line, {} in all; geometry records given to the builder {} times, \
             pages for what they point to {} when recorded and {} at submission, ranges in buffers brought up to date {}",
            self.pages.len(),
            (self.pages.len() as u64 * PAGE) >> 20,
            self.arenas.len(),
            self.copies.len(),
            s.uses[0] as f64 / frames,
            s.uses[1] as f64 / frames,
            s.faults as f64 / frames,
            s.unknown as f64 / frames,
            s.back as f64 / frames,
            s.looked as f64 / frames,
            s.changed as f64 / 1024.0 / frames,
            s.packed as f64 / frames,
            self.packed.len(),
            s.micros as f64 / 1000.0 / frames,
            self.windows,
            s.packed_bytes >> 10,
            s.gone,
            s.came[0],
            s.came[1],
            s.came[2],
            s.came[3],
            self.faulted,
            s.told_runs as f64 / frames,
            s.distinct.len(),
            s.mirrored as f64 / frames,
            self.fault_held[0] + self.fault_held[1],
            ((self.fault_held[0] + self.fault_held[1]) as u64 * PAGE) >> 20,
            (faults_most() as u64 * PAGE) >> 20,
            if recycle() { "" } else { "; VICTORY_BVH_FAULTS_RECYCLE=0: none held so, the number is for the run" },
            self.fault_held[0],
            self.fault_held[1],
            oldest,
            s.made,
            s.given,
            s.again,
            FAULT_SOON,
            s.dead,
            s.adopted,
            s.left[2] as f64 / frames,
            s.low as f64 / frames,
            s.left[0] as f64 / frames,
            s.left[1] as f64 / frames,
            s.builders as f64 / frames,
            wanted[1],
            wanted[0],
            s.geometry[0],
            s.geometry[1],
            s.geometry[2],
            s.geometry[3]
        );
        // Where the copies shaders walk lie: a read of the PC's memory
        // crosses the bus.
        let in_pc = |shadow: &Arc<Shadow>| {
            let mut props = D3D12_HEAP_PROPERTIES::default();
            // SAFETY: a live committed buffer.
            unsafe { shadow.buffer.GetHeapProperties(Some(&mut props), None) }.is_ok()
                && props.Type != D3D12_HEAP_TYPE_DEFAULT
                && !(props.Type == D3D12_HEAP_TYPE_CUSTOM && props.MemoryPoolPreference == D3D12_MEMORY_POOL_L1)
        };
        let arena = |n: usize| self.arenas.iter().any(|a| a.0 == n);
        let (mut pc, mut pc_mb, mut pc_arenas) = (0, 0u64, 0);
        for (n, shadow) in self.copies.iter().enumerate() {
            if in_pc(shadow) {
                pc += 1;
                pc_mb += shadow.span().1 >> 20;
                pc_arenas += usize::from(arena(n));
            }
        }
        tracing::info!(
            "title memory by address: of the {} copies, {pc} in the PC's memory ({pc_mb} MB; arenas {pc_arenas} of {})",
            self.copies.len(),
            self.arenas.len()
        );
        self.stats.from = Some((now, self.frame));
    }
}

/// `VICTORY_BVH_MIRROR=1`: before a list's first builder dispatch, pages that
/// by-address readers see as title memory has them take, on the card, the
/// bytes of the title buffers in them that shaders write
/// ([`Memory::shares`]). Off unless set.
fn mirror() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_BVH_MIRROR").is_some_and(|v| v == "1"))
}

/// Whether the builder stored results in a window at all ([`results`]).
static RESULTS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether a copy of the builder's results is read out of its window
/// (`VICTORY_BVH_RESULTS_COPY=0`: out of title memory at submission, as
/// before).
fn results_copy() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_BVH_RESULTS_COPY").is_none_or(|v| v != "0"))
}

/// The copy that holds what the structure builder stored at `[src, src +
/// len)`, if that is a window of title memory around a small title buffer
/// rather than the buffer's own copy.
///
/// The builder's program of a structure's compacted size stores into a small
/// title buffer through title memory by address, so into the window of the
/// page that holds it ([`Memory::map`]). If the title then copies that buffer
/// in a list (to where its CPU reads the size), the copy was taken out of
/// title memory at submission, before the card had run the program: the size
/// of the structure asked for before ([`sizes_wait`] could not help). Memory
/// of that size taken, the copy of the new structure wrote over its
/// neighbours. A copy out of these bytes reads the window instead, on the
/// card, at its place in the list ([`crate::host::HostList::place_copy`]).
pub(crate) fn results(src: u64, len: u64) -> Option<Arc<Shadow>> {
    if len == 0 || !RESULTS.load(std::sync::atomic::Ordering::Relaxed) || !results_copy() {
        return None;
    }
    let page = src / PAGE;
    if (src + len - 1) / PAGE != page {
        return None;
    }
    let guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    let m = guard.as_ref()?;
    let &k = m.results.get(&page)?;
    let shadow = m.copies.get(k)?;
    shadow.written().then(|| shadow.clone())
}

/// Whether an arena holds a 64 KiB page of title memory `[at, end)`
/// (`crate::left`: it takes such a page again only where the title wrote).
pub(crate) fn holds(at: u64, end: u64) -> bool {
    let guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    guard.as_ref().is_some_and(|m| (at / PAGE..end.div_ceil(PAGE)).any(|page| m.pages.contains_key(&page)))
}

/// `VICTORY_BVH_TRACE=1`: a buffer copy out of the builder's results.
#[cfg(feature = "diagnostics")]
pub(crate) fn trace_results_copy(dst: usize, src: usize, len: usize, how: &str) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_BVH_TRACE").is_some_and(|v| v == "1")) {
        return;
    }
    static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if n < 400 || n.is_multiple_of(500) {
        let mut b = [0u8; 8];
        crate::host::copy_mapped(b.as_mut_ptr(), src as *const u8, 8.min(len), false);
        tracing::info!(
            n,
            frame = crate::frame::current(),
            "bvh trace: copy of the builder's results {src:#x} -> {dst:#x} ({len} bytes), {how}; title memory had {:#x}",
            u64::from_le_bytes(b)
        );
    }
}

/// A list with this key ([`HostList::used`]) asks the builder for sizes.
const SIZES: usize = usize::MAX - (9 << 33);

/// Whether a submission that asks for the sizes of structures is waited for,
/// and landed, before the title goes on (`VICTORY_BVH_SIZES_WAIT=1`; off).
///
/// Off: it does not help. In run light-15, with it, 33 of 154 copies had the
/// slot's old size still (all of them recorded one frame after the size was
/// asked for) against 53 of 230 in light-11, 48 of 275 dynamic structures
/// were good, and the title asks in every frame, so every frame waits (109 ms
/// against 90). The title reads the slot before our submission of the asking
/// list is through, or reads it for another purpose; not known.
///
/// The title asks for a new structure's compacted size (a program of one
/// thread stores it into a small buffer), reads it a frame or two later
/// without looking whether it is there, takes memory of that size and copies
/// the structure into it. With submissions that land up to two frames late it
/// read what the slot held before, some other structure's size, and the copy
/// wrote over its neighbours: of 230 such copies of run light-11, 99 were
/// asked for two frames earlier and landed in that frame, none of those with
/// the right size.
fn sizes_wait() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_BVH_SIZES_WAIT").is_some_and(|v| v == "1"))
}

/// Whether a dispatch of the builder is its program of a structure's size:
/// one thread, the structure's address and what to store (1: the compacted
/// size) as constants, a small title buffer to store into.
fn asks_size(written: &[u64], constants: &[u32], threads: Option<[u32; 3]>) -> bool {
    threads == Some([1, 1, 1])
        && matches!(*constants, [_, hi, 1..=3, ..] if (1..0x20).contains(&hi))
        && written.first().is_some_and(|&at| crate::shadow::in_buffer(at, 1).is_some_and(|b| b.1 <= BACK))
}

/// Whether one of `lists` asks for sizes.
pub(crate) fn asked_sizes<'a>(lists: impl Iterator<Item = &'a HostList>) -> bool {
    let mut lists = lists;
    let asked = sizes_wait() && lists.any(|l| l.used(SIZES));
    if asked {
        static TIMES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = TIMES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if n.is_power_of_two() || n.is_multiple_of(100) {
            tracing::info!(n, frame = crate::frame::current(), "submissions that ask for structures' sizes, waited for");
        }
    }
    asked
}

/// The notes of a frame (the page, whether the builder left it) in page
/// order, a page both left once, as the builder's.
fn noted(mut raw: Vec<(u64, bool)>) -> Vec<(u64, bool)> {
    raw.sort_unstable();
    raw.dedup_by(|later, first| {
        let same = first.0 == later.0;
        first.1 |= same && later.1;
        same
    });
    raw
}

/// Whose page from a fault is given back first, of `held` (the walking
/// programs', the builder's) with at most `most` held in all: the walking
/// programs', which read the title's own bytes and get them again whole.
/// What the builder found empty it may have stored in since, and that is on
/// the card alone: its pages go only once they are more than half of `most`,
/// or all there are.
fn goes_first(held: [usize; 2], most: usize) -> usize {
    usize::from(held[0] == 0 || held[1] > most / 2)
}

/// Sets `page`'s bit; whether it was not set.
fn mark(bits: &mut [u64], page: u64) -> bool {
    let Some(word) = bits.get_mut((page / 64) as usize) else {
        return false;
    };
    let bit = 1 << (page % 64);
    let new = *word & bit == 0;
    *word |= bit;
    new
}

/// [`fill`] of a page stores by address do not reach.
fn close(entry: *mut u32, n: usize, offset: u64) {
    fill(entry, n | MEMORY_READ_ONLY as usize, offset);
}

/// The offset first: a shader that sees the number sees it too.
fn fill(entry: *mut u32, n: usize, offset: u64) {
    // SAFETY: an entry of the directory, under the mutex.
    unsafe {
        entry.add(1).write_volatile(offset as u32);
        entry.write_volatile(n as u32 + 1);
    }
}

/// The bytes of a packed structure whose first two dwords these are: the
/// version (6) with the marks of a bottom level and of one that is packed,
/// and its size unpacked in 64-byte nodes.
fn packed_len(header: u64) -> Option<u64> {
    let (version, nodes) = (header as u32, header >> 32);
    let marks = version >> 16;
    // The header counts its nodes of 64 bytes without the last one: the
    // builder's own program of a structure's size stores one more.
    let len = (nodes + 1) * 64;
    (version & 0xFFFF == 6 && marks & 0x8100 == 0x8100 && nodes > 0 && len <= PACKED).then_some(len)
}

/// A geometry record of the builder's inputs, as LeafRehydration and the
/// builds read them from a view (`idxen`, 48 bytes apiece): the transform's
/// address, the index and vertex formats, the index and vertex counts, the
/// index and vertex buffers' addresses, the vertices' stride (its low 15
/// bits). The programs reach the buffers by those addresses alone, so the
/// views' pages being ready does not make theirs ready.
const RECORD: u64 = 48;
/// At most so many records to an array, and so many bytes of buffers.
const RECORDS: u64 = 4096;
const RECORDS_BYTES: u64 = 256 << 20;

/// The ranges of title memory the geometry records at `at` point to
/// (address, length), the records' own included; none if what lies there
/// does not read as records. `read` reads title memory, false where it has
/// none. The array ends at the first record that does not read as one.
fn geometry(read: impl Fn(u64, &mut [u8]) -> bool, at: u64) -> Vec<(u64, u64)> {
    let title = |a: u64| (1 << 32..1 << 40).contains(&a);
    let mut ranges = Vec::new();
    let mut bytes = 0;
    for n in 0..RECORDS {
        let mut r = [0u8; RECORD as usize];
        if !read(at + n * RECORD, &mut r) {
            break;
        }
        let d = |i: usize| u32::from_le_bytes(r[4 * i..4 * i + 4].try_into().unwrap());
        let q = |i: usize| u64::from(d(i)) | u64::from(d(i + 1)) << 32;
        let (transform, index_format, vertex_format) = (q(0), d(2), d(3));
        let (indices, vertices, ib, vb, stride) = (
            u64::from(d(4)),
            u64::from(d(5)),
            q(6),
            q(8),
            u64::from(d(10) & 0x7FFF),
        );
        // The index formats the program knows: 32-bit, 16-bit, its own
        // packed one (a header of offsets; at most as long as 32-bit
        // indices and a header), none.
        let index_bytes = match index_format {
            42 => indices * 4,
            57 => indices * 2,
            53 => indices * 4 + 4096,
            0 => 0,
            _ => break,
        };
        let ok = (1..132).contains(&vertex_format)
            && title(vb)
            && stride > 0
            && (1..=1 << 24).contains(&vertices)
            && indices <= 1 << 26
            && (index_format == 0 || (title(ib) && indices > 0))
            && (transform == 0 || title(transform));
        if !ok {
            break;
        }
        let these = [
            (vb, vertices * stride),
            (ib, index_bytes),
            (transform, if transform == 0 { 0 } else { 48 }),
        ];
        let len: u64 = these.iter().map(|r| r.1).sum();
        if bytes + len > RECORDS_BYTES {
            break;
        }
        bytes += len;
        ranges.extend(these.into_iter().filter(|r| r.1 > 0));
        if n == 0 {
            ranges.push((at, RECORD));
        } else if let Some(own) = ranges.iter_mut().find(|r| r.0 == at) {
            own.1 += RECORD;
        }
    }
    ranges
}

/// What title memory holds at `at`, as a shader would find it there; false
/// where it has none.
fn read_title(at: u64, out: &mut [u8]) -> bool {
    let end = at + out.len() as u64;
    if at < 1 << 32 || end > u64::from(MEMORY_PAGES) * PAGE {
        return false;
    }
    // (Committed page by page: a record may cross one.)
    let (first, last) = (at / PAGE, (end - 1) / PAGE);
    if !(first..=last).all(|p| crate::texture::committed((p * PAGE) as *const u8, PAGE as usize)) {
        return false;
    }
    crate::left::before_take(at as usize, end as usize, 0);
    crate::host::copy_mapped(out.as_mut_ptr(), at as *const u8, out.len(), false);
    true
}

/// Puts what `title` holds into `ours` (what was last taken): the dwords that
/// differ. How many.
fn merge(title: &[u8; PIECE], ours: &mut [u8], never: bool) -> usize {
    if never {
        ours.copy_from_slice(title);
        return PIECE / 4;
    }
    if ours == title {
        return 0;
    }
    let mut changed = 0;
    for (t, o) in title.chunks_exact(4).zip(ours.chunks_exact_mut(4)) {
        if t != o {
            o.copy_from_slice(t);
            changed += 1;
        }
    }
    changed
}

/// Before a draw whose shader reads title memory by address: what its
/// root arguments name is made ready (`addresses`: its root views, of them
/// `written` those it writes; `constants`: its root constants, see the
/// module), every numbered copy
/// has its view in `heap`'s run and is brought up to date when `list` is
/// submitted. None if the host cannot hold it; the draw is then dropped.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ready(
    list: &HostList,
    heap: &HostHeap,
    addresses: &[u64],
    written: &[u64],
    constants: &[u32],
    use_: Use,
    threads: Option<[u32; 3]>,
    name: Option<&str>,
) -> Option<Ready> {
    let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = Memory::new();
    }
    let m = guard.as_mut()?;
    let frame = crate::frame::current();
    if m.frame != frame {
        m.frame = frame;
        m.refused.clear();
        (m.room, m.full) = (None, false);
        m.take_faults();
        m.look_again();
        m.report();
        #[cfg(feature = "diagnostics")]
        m.trace.look(frame);
        #[cfg(feature = "diagnostics")]
        m.take_stores();
    }
    #[cfg(feature = "diagnostics")]
    if use_ != Use::Reads {
        m.trace.dispatch(frame, list, addresses, written, constants, threads);
    }
    let builds = use_ != Use::Reads;
    #[cfg(feature = "diagnostics")]
    let tag = if builds && stores::on() {
        m.stores.tag(frame, name.unwrap_or("?"), written, addresses)
    } else {
        0
    };
    #[cfg(not(feature = "diagnostics"))]
    let (tag, _) = (0, name);
    if builds && sizes_wait() && asks_size(written, constants, threads) {
        list.first_use(SIZES);
    }
    while builds && m.free.len() < SPARE && m.pages.len() + m.free.len() < MOST && m.another() {}
    // From here on: what this dispatch names. (The faults' pages and the
    // spare arena are not its own.)
    m.short = false;
    m.stats.uses[usize::from(builds)] += 1;
    // Once per list and page named: the builder dispatches hundreds of times
    // a frame with the same arguments.
    const READ: usize = usize::MAX - 0x6d;
    let named = |kind: usize, at: u64| list.first_use(usize::MAX - (kind << 26) - (at / PAGE) as usize);
    // A title address in two constants (low dword, then bits 32 to 39), or
    // in one in 256-byte units (the builder's alone). A constant that only
    // looks like one names memory nothing reads; one that is not the title's
    // memory is refused.
    let held = |m: &Memory| m.pages.len() as u64;
    let before = held(m);
    for (i, &c) in constants.iter().enumerate() {
        if let Some(&high) = constants.get(i + 1)
            && (1..=0x1F).contains(&high)
            && c >= 0x1_0000
        {
            let at = u64::from(c) | u64::from(high) << 32;
            if named(1, at) {
                m.ready(at, 1);
            }
        }
        if builds && (0x0100_0000..0x2000_0000).contains(&c) {
            let at = u64::from(c) << 8;
            if named(1, at) {
                m.ready(at, 1);
            }
        }
    }
    let mut came = [held(m) - before, 0, 0, 0];
    for &at in addresses.iter().filter(|&&a| a >= 1 << 32) {
        // (Several structures to a page.)
        let mut packed = false;
        if builds
            && list.first_use(usize::MAX - (3 << 33) - (at >> 8) as usize)
            && crate::shadow::in_buffer(at, 1).is_none()
            && m.possible(at / PAGE)
        {
            // What it needs, as far as it is known now; the rest when the
            // list is submitted ([`Memory::land`]).
            if let Some((len, _)) = m.packed_at(at) {
                packed = true;
                for page in at / PAGE..=(at + len - 1) / PAGE {
                    let new = m.copy_of(page).is_none();
                    if m.map(page)
                        && new
                        && let Some(held) = m.pages.get_mut(&page)
                    {
                        held.theirs = true;
                    }
                }
                m.bring(at / PAGE, (at + len - 1) / PAGE + 1);
            }
            list.defer_host(move || {
                if let Some(m) = MEMORY.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
                    m.land(at);
                }
            });
            m.bring_up(list);
        }
        if builds && written.contains(&at) && list.first_use(usize::MAX - (4 << 33) - (at >> 2 & 0x7FFF_FFFF) as usize) {
            if let Some((start, len)) = crate::shadow::in_buffer(at, 1)
                && len <= BACK
                && use_ == (Use::Builds { back: true })
                && list.first_use(usize::MAX - (5 << 33) - (start >> 2 & 0x7FFF_FFFF) as usize)
            {
                m.small(list, start, len);
            }
            m.opens(at);
            if !packed && crate::shadow::in_buffer(at, 1).is_none() {
                #[cfg(feature = "diagnostics")]
                {
                    let built = m.dump.built.entry(at).or_insert((frame, frame, 0));
                    (built.1, built.2) = (frame, built.2 + 1);
                }
                let before = held(m);
                m.ready(at, BUILT);
                came[1] += held(m).saturating_sub(before);
            }
        }
        if builds
            && !packed
            && !written.contains(&at)
            && list.first_use(usize::MAX - (9 << 33) - (at >> 4 & 0x7FFF_FFFF) as usize)
        {
            let ranges = geometry(read_title, at);
            if !ranges.is_empty() {
                m.stats.geometry[0] += 1;
                let before = held(m);
                for &(start, len) in &ranges {
                    m.ready(start, len);
                    // In a title buffer the directory points at the buffer's
                    // copy, which is brought up to date only by a draw or
                    // view that names it: streamed-in geometry is unpacked
                    // before anything draws with it (as a draw's vertex
                    // buffers are brought up to date).
                    if crate::shadow::in_buffer(start, len).is_some()
                        && let Some(shadow) = crate::shadow::find(start, len)
                    {
                        shadow.refresh(list, start, len);
                        m.stats.geometry[3] += 1;
                    }
                }
                m.stats.geometry[1] += held(m).saturating_sub(before);
                list.defer_host(move || {
                    if let Some(m) = MEMORY.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
                        m.land_geometry(at);
                    }
                });
                m.bring_up(list);
            }
        }
        if packed || !named(2, at) {
            continue;
        }
        let before = held(m);
        match crate::shadow::in_buffer(at, 1) {
            Some(_) => {
                m.ready(at, 1);
                m.named_in_upload(at);
                came[3] += held(m).saturating_sub(before);
            }
            None => {
                m.ready(at, if builds { VIEW } else { PAGE });
                came[2] += held(m).saturating_sub(before);
            }
        }
    }
    for (sum, n) in m.stats.came.iter_mut().zip(came) {
        *sum += n;
    }
    #[cfg(feature = "diagnostics")]
    if builds {
        m.inputs(addresses, written);
    }
    if builds && !m.over.is_empty() && list.first_use(usize::MAX - (7 << 33)) {
        m.mirror(list);
    }
    let key = std::ptr::from_ref(heap) as usize;
    let (base, made) = match m.heaps.get(&key) {
        Some(&run) => run,
        None => {
            // Held for the heap's life: not given back with a list's slots.
            let base = heap.dynamic(COPIES + 1)?;
            // Every slot of the run: a number shows no view of another kind.
            for n in 0..=COPIES {
                heap.null_raw_uav_at(base + n);
            }
            (base, 0)
        }
    };
    for (n, shadow) in m.copies.iter().enumerate().skip(made as usize) {
        let (_, len) = shadow.span();
        heap.raw_uav_at(base + 1 + n as u32, &shadow.buffer, 0, len);
    }
    m.heaps.insert(key, (base, m.copies.len() as u32));
    // Every copy by every list that has come to by-address memory, whoever
    // numbered it: an arena another list made while this one was being
    // recorded was brought up to date by that list's submission alone, which
    // may run after this one's (structures unpacked from zeros, and then
    // written over by the arena's first upload: in frames that stream much in
    // half of them, run light-8).
    let first_read = list.first_use(READ);
    if list.first_use(usize::MAX - (6 << 33) - m.copies.len()) {
        m.bring_up(list);
    }
    if first_read {
        list.keep(&m.faults.0);
        #[cfg(feature = "diagnostics")]
        m.dump(list, addresses);
    }
    #[cfg(feature = "diagnostics")]
    if use_ == Use::Reads {
        m.dump.walked(addresses);
    } else {
        m.dump.read_by_builder(addresses, written);
    }
    let directory = &m.directory.1;
    Some(Ready {
        directory: directory.gpu(directory.span().0),
        // (The shader's places are the same in either half.)
        faults: m.faults.2 + if builds { u64::from(MEMORY_FAULTS) * 4 } else { 0 },
        base,
        whole: !m.short,
        tag,
    })
}

/// `VICTORY_BVH_INPUTS=1`: where the structure builder's reads through a root
/// view into a title buffer are served from. A page a title buffer does not
/// hold whole is an arena's page or a window of title memory: what the CPU's
/// side of title memory holds, not the buffer's own copy on the card, where
/// the title's programs write (characters' skinned vertices). Run bvhcheck-1:
/// moving objects' structures were built from vertices that are not there
/// (z of 1e27 to 3.4e38, bytes of other records), and their boxes stretched
/// the top level's to infinity. Per frame line: such reads by what serves
/// them; the first buffers of each kind once.
#[cfg(feature = "diagnostics")]
#[derive(Default)]
pub(crate) struct Inputs {
    /// Reads served by the buffer's own copy; by an arena page; by a window;
    /// by nothing. And of those not by its own copy, the buffers whose copy
    /// shaders write.
    own: u64,
    arena: u64,
    window: u64,
    none: u64,
    written: u64,
    /// Pages given a place of title memory over buffers shaders write.
    pages: u64,
    said: std::collections::HashSet<(u64, u8)>,
    from: Option<std::time::Instant>,
}

#[cfg(feature = "diagnostics")]
fn inputs_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_BVH_INPUTS").is_some_and(|v| v == "1"))
}

#[cfg(feature = "diagnostics")]
static INPUTS: Mutex<Option<Inputs>> = Mutex::new(None);

/// (`VICTORY_BVH_INPUTS`) The page at `start` is given a place of title
/// memory (an arena page or a window): the title buffers in it whose own
/// copy shaders write are not what by-address readers see there. Each such
/// page said once (the first 400), and counted.
#[cfg(feature = "diagnostics")]
fn held_over_written(start: u64) {
    if !inputs_on() {
        return;
    }
    let found = crate::shadow::written_across(start, PAGE);
    if found.is_empty() {
        return;
    }
    let mut guard = INPUTS.lock().unwrap_or_else(|e| e.into_inner());
    let n = guard.get_or_insert_with(Inputs::default);
    n.pages += 1;
    if n.pages <= 400 {
        let list: Vec<String> = found.iter().take(8).map(|(s, l)| format!("{s:#x}+{l}")).collect();
        tracing::info!(
            "bvh inputs: page {start:#x} is read by address from title memory, but {} title buffers in it are written by shaders in their own copies: {}",
            found.len(),
            list.join(", ")
        );
    }
}

#[cfg(feature = "diagnostics")]
impl Memory {
    /// A builder dispatch's root views (`written`: those it stores through).
    fn inputs(&mut self, addresses: &[u64], written: &[u64]) {
        if !inputs_on() {
            return;
        }
        let mut guard = INPUTS.lock().unwrap_or_else(|e| e.into_inner());
        let n = guard.get_or_insert_with(Inputs::default);
        for &at in addresses.iter().filter(|&&a| a >= 1 << 32 && !written.contains(&a)) {
            let Some((start, len)) = crate::shadow::in_buffer(at, 1) else { continue };
            let page = at / PAGE;
            let own_copy = crate::shadow::peek(at, 1);
            let kind = match self.copy_of(page) {
                None => 3u8,
                Some(k) if own_copy.as_ref().is_some_and(|s| Arc::ptr_eq(s, &self.copies[k])) => 0,
                Some(_) if self.pages.contains_key(&page) => 1,
                Some(_) => 2,
            };
            let shaders_write = own_copy.as_ref().is_some_and(|s| s.written());
            match kind {
                0 => n.own += 1,
                1 => n.arena += 1,
                2 => n.window += 1,
                _ => n.none += 1,
            }
            if kind != 0 && shaders_write {
                n.written += 1;
            }
            if n.said.len() < 400 && n.said.insert((start, kind)) {
                let what = ["its own copy", "an arena page (title memory)", "a window of title memory", "nothing"][kind as usize];
                tracing::info!(
                    "bvh inputs: the builder reads {at:#x} in title buffer {start:#x} ({len} bytes, page {:#x} {}): from {what}; shaders write its copy: {shaders_write}",
                    page * PAGE,
                    if start <= page * PAGE && start + len >= (page + 1) * PAGE { "whole in it" } else { "shared" },
                );
            }
        }
        let now = std::time::Instant::now();
        let since = *n.from.get_or_insert(now);
        if now.duration_since(since).as_secs_f64() >= 5.0 {
            tracing::info!(
                "bvh inputs (VICTORY_BVH_INPUTS): reads of title buffers by the builder in 5 s: {} from their own copy, {} from arena pages, {} from windows, {} from nothing; of those not from their own copy, {} of buffers shaders write; pages read by address over such buffers so far {}",
                n.own,
                n.arena,
                n.window,
                n.none,
                n.written,
                n.pages
            );
            (n.own, n.arena, n.window, n.none, n.written) = (0, 0, 0, 0, 0);
            n.from = Some(now);
        }
    }
}

/// `VICTORY_BVH_TRACE=1`: the sizes the builder is asked for (its program of
/// one thread that stores a structure's compacted size) and when title memory
/// has them, and the copies to a place of that size.
#[cfg(feature = "diagnostics")]
mod trace {
    use super::*;

    #[derive(Default)]
    pub(super) struct Trace {
        /// Frame, where the size goes, the structure, what title memory had.
        asked: Vec<(u64, u64, u64, u64)>,
        /// By structure, what its size was when last seen to change.
        sizes: HashMap<u64, (u64, u64)>,
        said: u32,
    }

    fn on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("VICTORY_BVH_TRACE").is_some_and(|v| v == "1"))
    }

    fn value(at: u64) -> u64 {
        let mut b = [0u8; 8];
        crate::host::copy_mapped(b.as_mut_ptr(), at as *const u8, 8, false);
        u64::from_le_bytes(b)
    }

    impl Trace {
        fn say(&mut self, text: String) {
            self.said += 1;
            if self.said < 6000 {
                tracing::info!("bvh trace: {text}");
            }
        }

        pub(super) fn look(&mut self, frame: u64) {
            if !on() {
                return;
            }
            let mut gone = Vec::new();
            for i in 0..self.asked.len() {
                let (then, at, structure, was) = self.asked[i];
                let now = value(at);
                if now != was {
                    self.sizes.insert(structure, (now, frame));
                    gone.push(i);
                    self.say(format!("frame {frame}: size of {structure:#x} at {at:#x} asked in {then}: {was:#x} -> {now:#x}"));
                } else if frame > then + 60 {
                    gone.push(i);
                    self.say(format!("frame {frame}: size of {structure:#x} at {at:#x} asked in {then}: still {was:#x}"));
                }
            }
            for i in gone.into_iter().rev() {
                self.asked.swap_remove(i);
            }
        }

        pub(super) fn dispatch(
            &mut self,
            frame: u64,
            list: &HostList,
            addresses: &[u64],
            written: &[u64],
            constants: &[u32],
            threads: Option<[u32; 3]>,
        ) {
            if !on() {
                return;
            }
            let _ = addresses;
            let list = std::ptr::from_ref(list) as usize & 0xFFFF_FFFF;
            if threads == Some([1, 1, 1])
                && let [lo, hi, mode, ..] = *constants
                && (1..=3).contains(&mode)
                && hi < 0x20
                && let Some(&at) = written.first()
                && crate::shadow::in_buffer(at, 1).is_some_and(|b| b.1 <= BACK)
            {
                let structure = u64::from(lo) | u64::from(hi) << 32;
                let was = value(at);
                self.say(format!("frame {frame} list {list:x}: asks mode {mode} of {structure:#x} to {at:#x}, there {was:#x}"));
                if mode == 1 {
                    self.asked.push((frame, at, structure, was));
                }
            } else if threads == Some([32, 1, 1])
                && let [from, to] = *written
                && [from, to].iter().all(|&a| a >= 1 << 32 && crate::shadow::in_buffer(a, 1).is_none())
            {
                let known = self.sizes.get(&from).copied();
                self.say(format!("frame {frame} list {list:x}: copies {from:#x} to {to:#x} constants {:x?}, its size was {known:x?}", &constants[..constants.len().min(4)]));
            }
        }
    }
}

/// `VICTORY_MEMORY_DUMP=<folder>`: when a file `go` appears there, the arenas
/// as the card holds them are copied out and written (`arena-<n>.bin`), with
/// `pages.txt` (title page, arena, offset) and the structures the walking
/// programs were given: to look at the structures offline.
#[cfg(feature = "diagnostics")]
mod dump {
    use super::*;

    #[derive(Default)]
    pub(super) struct Dump {
        walked: std::collections::BTreeSet<u64>,
        /// Title buffers the builder read through root views (start, bytes).
        buffers: std::collections::BTreeSet<(u64, u64)>,
        /// Views the builder was given to write outside buffers: the first
        /// and the last frame, how often.
        pub(super) built: std::collections::BTreeMap<u64, (u64, u64, u64)>,
        /// The frame it is written at, and the copies.
        due: Option<(u64, Vec<(ID3D12Resource, usize)>)>,
        done: bool,
    }

    impl Dump {
        pub(super) fn walked(&mut self, addresses: &[u64]) {
            if !self.done && self.walked.len() < 64 {
                self.walked.extend(addresses);
            }
        }

        /// A builder dispatch's root views: the title buffers it reads.
        pub(super) fn read_by_builder(&mut self, addresses: &[u64], written: &[u64]) {
            if self.done || folder().is_none() {
                return;
            }
            for &at in addresses.iter().filter(|&&a| a >= 1 << 32 && !written.contains(&a)) {
                if let Some(span) = crate::shadow::in_buffer(at, 1) {
                    self.buffers.insert(span);
                }
            }
        }
    }

    fn folder() -> Option<&'static std::path::PathBuf> {
        static FOLDER: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
        FOLDER.get_or_init(|| std::env::var_os("VICTORY_MEMORY_DUMP").map(Into::into)).as_ref()
    }

    fn turn(buffer: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
        D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    // SAFETY: the same pointer, not counted; the barrier is
                    // used while the buffer lives.
                    pResource: unsafe { std::mem::transmute_copy(buffer) },
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: before,
                    StateAfter: after,
                }),
            },
        }
    }

    impl Memory {
        /// At a list's first use: the copies are recorded into it.
        pub(super) fn dump(&mut self, list: &HostList, _: &[u64]) {
            let Some(folder) = folder() else { return };
            if let Some((due, _)) = &self.dump.due {
                if self.frame < *due {
                    return;
                }
                let copies = self.dump.due.take().map(|d| d.1).unwrap_or_default();
                let mut text = String::new();
                for a in &self.dump.walked {
                    text += &format!("walked {a:#x}\n");
                }
                for (a, (first, last, n)) in &self.dump.built {
                    text += &format!("built {a:#x} {first} {last} {n}\n");
                }
                for (a, p) in &self.packed {
                    text += &format!("packed {a:#x} {} {}\n", p.len, p.frame);
                }
                text += &format!("frame {}\n", self.frame);
                for (page, held) in &self.pages {
                    let arena = self.arenas.iter().position(|a| a.0 == held.copy).unwrap_or(usize::MAX);
                    let at = held.at - self.arenas[arena].1;
                    text += &format!("{:#x} {arena} {at:#x}\n", page * PAGE);
                }
                let _ = std::fs::write(folder.join("pages.txt"), text);
                for (n, (_, data)) in copies.iter().enumerate() {
                    // SAFETY: a mapping of an arena's bytes; the copies ran
                    // frames ago.
                    let bytes = unsafe { std::slice::from_raw_parts(*data as *const u8, ARENA as usize) };
                    let _ = std::fs::write(folder.join(format!("arena-{n}.bin")), bytes);
                }
                tracing::info!(arenas = copies.len(), "title memory by address: the arenas are written out");
                return;
            }
            if self.dump.done || !folder.join("go").exists() {
                return;
            }
            self.dump.done = true;
            let Some(host) = host() else { return };
            let mut copies = Vec::new();
            for &(n, _) in &self.arenas {
                let buffer = &self.copies[n].buffer;
                let Some((to, data)) = host.shared_buffer(ARENA) else { break };
                // SAFETY: an open list; live buffers of those sizes. The list
                // has not used the arena yet.
                unsafe {
                    list.list.ResourceBarrier(&[turn(buffer, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                    list.list.CopyBufferRegion(&to, 0, buffer, 0, ARENA);
                    list.list.ResourceBarrier(&[turn(buffer, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON)]);
                }
                list.keep(&to);
                copies.push((to, data as usize));
            }
            self.dump.due = Some((self.frame + 12, copies));
            self.title_side(folder);
        }

        /// The CPU's side of every held page, as title memory has it while the
        /// arenas' copies are recorded (`title.bin`, pages in the order of
        /// `title-pages.txt`: page, index, the frame it came in and was last
        /// looked at, the write-watch asking it was last taken after): to
        /// tell what the card's copy of a page lacks from what title memory
        /// itself holds (run bvhcheck-2: moving objects built from vertex
        /// bytes that are not positions, in memory the builder reaches by
        /// pointer).
        fn title_side(&self, folder: &std::path::Path) {
            use std::io::Write;
            let Ok(file) = std::fs::File::create(folder.join("title.bin")) else { return };
            let mut out = std::io::BufWriter::with_capacity(1 << 22, file);
            let mut text = format!("frame {}\n", self.frame);
            let mut buf = vec![0u8; PAGE as usize];
            let mut pages: Vec<(&u64, &Held)> = self.pages.iter().collect();
            pages.sort_unstable_by_key(|p| *p.0);
            let mut n = 0;
            let mut done = std::collections::HashSet::new();
            for (page, held) in pages {
                crate::host::copy_mapped(buf.as_mut_ptr(), (page * PAGE) as *const u8, PAGE as usize, false);
                if out.write_all(&buf).is_err() {
                    return;
                }
                done.insert(*page);
                text += &format!("{:#x} {n} {} {} {}\n", page * PAGE, held.born, held.looked, held.taken);
                n += 1;
            }
            // And the title buffers the builder read through root views, as
            // title memory has them (what shaders wrote to them comes back
            // there when its copy lands): up to 1.5 GB more. Their line has
            // `buffer` for the frames and the asking.
            let mut more = 0;
            for &(start, len) in &self.dump.buffers {
                for page in start / PAGE..(start + len).div_ceil(PAGE) {
                    if more >= 24_000 || !done.insert(page) {
                        continue;
                    }
                    let at = page * PAGE;
                    if !crate::texture::committed(at as *const u8, PAGE as usize) {
                        continue;
                    }
                    crate::host::copy_mapped(buf.as_mut_ptr(), at as *const u8, PAGE as usize, false);
                    if out.write_all(&buf).is_err() {
                        return;
                    }
                    text += &format!("{at:#x} {n} buffer {start:#x} {len}\n");
                    n += 1;
                    more += 1;
                }
            }
            let _ = out.flush();
            let _ = std::fs::write(folder.join("title-pages.txt"), text);
            tracing::info!("title memory by address: the CPU's side of the held pages is written out");
        }
    }
}

/// Evicts cached descriptor-run state for `heap` when its owning `Heap` is
/// dropped so a later `Heap` allocated at the same address cannot reuse stale
/// `(base, made)` slot offsets.
pub(crate) fn forget_heap(heap: &HostHeap) {
    let key = std::ptr::from_ref(heap) as usize;
    if let Some(m) = MEMORY.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        m.heaps.remove(&key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dropping_a_view_heap_evicts_its_memory_heaps_entry() {
        use crate::descriptor::{HeapDesc, create_heap, heap};
        use crate::object::{S_OK, This, guid};
        if crate::host::host().is_none() {
            return;
        }
        let iid = guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        let desc = HeapDesc { kind: 0, count: 64, flags: 1, nodes: 1 };
        let mut h: This = std::ptr::null();
        assert_eq!(create_heap(std::ptr::null(), &desc, &iid, &mut h), S_OK);
        let host_heap = heap(h).host().expect("host view heap");
        let key = std::ptr::from_ref(host_heap) as usize;
        // A run made for it, as a draw's `ready` makes one.
        {
            let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
            if guard.is_none() {
                *guard = Memory::new();
            }
            guard.as_mut().unwrap().heaps.insert(key, (0, 0));
        }
        // Release the title heap (COM vtable slot 2): `Heap::drop` must evict `key`.
        // SAFETY: `h` is a live COM object from `create_heap`.
        let release: extern "system" fn(This) -> u32 =
            unsafe { std::mem::transmute(*(*(h as *const *const *const ())).add(2)) };
        assert_eq!(release(h), 0);
        let guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
        assert!(
            !guard.as_ref().unwrap().heaps.contains_key(&key),
            "MEMORY.heaps retained dropped HostHeap key {key:#x}"
        );
    }

    /// A geometry record as the builder reads it.
    #[allow(clippy::too_many_arguments)]
    fn record(
        transform: u64,
        index_format: u32,
        vertex_format: u32,
        indices: u32,
        vertices: u32,
        ib: u64,
        vb: u64,
        stride: u32,
    ) -> Vec<u8> {
        let mut r = Vec::new();
        r.extend(transform.to_le_bytes());
        r.extend(index_format.to_le_bytes());
        r.extend(vertex_format.to_le_bytes());
        r.extend(indices.to_le_bytes());
        r.extend(vertices.to_le_bytes());
        r.extend(ib.to_le_bytes());
        r.extend(vb.to_le_bytes());
        r.extend((stride | 0x8000).to_le_bytes());
        r.extend(0u32.to_le_bytes());
        assert_eq!(r.len() as u64, RECORD);
        r
    }

    #[test]
    fn geometry_records_name_their_buffers() {
        let at = 0x9_0000_0000;
        let mut bytes = record(
            0x9_1000_0000,
            42,
            6,
            300,
            100,
            0x9_2000_0000,
            0x9_3000_0000,
            12,
        );
        bytes.extend(record(0, 57, 16, 6, 4, 0x9_4000_0000, 0x9_5000_0000, 8));
        // Not a record: the array ends there.
        bytes.extend([0u8; RECORD as usize]);
        let read = |a: u64, out: &mut [u8]| {
            let Some(from) = a
                .checked_sub(at)
                .map(|o| o as usize)
                .filter(|&o| o + out.len() <= bytes.len())
            else {
                return false;
            };
            out.copy_from_slice(&bytes[from..from + out.len()]);
            true
        };
        let ranges = geometry(read, at);
        assert_eq!(
            ranges,
            vec![
                (0x9_3000_0000, 1200),
                (0x9_2000_0000, 1200),
                (0x9_1000_0000, 48),
                (at, 2 * RECORD),
                (0x9_5000_0000, 32),
                (0x9_4000_0000, 12),
            ]
        );
    }

    #[test]
    fn what_does_not_read_as_geometry_records_names_nothing() {
        let at = 0x9_0000_0000;
        // A vertex buffer below 4 GiB, an index format the program does not
        // know, no stride, memory that is not there.
        for bytes in [
            record(0, 42, 6, 3, 3, 0x9_2000_0000, 0x1000, 12),
            record(0, 7, 6, 3, 3, 0x9_2000_0000, 0x9_3000_0000, 12),
            record(0, 0, 6, 0, 3, 0, 0x9_3000_0000, 0x8000),
        ] {
            let read = |a: u64, out: &mut [u8]| {
                out.copy_from_slice(&bytes[(a - at) as usize..][..out.len()]);
                true
            };
            assert!(geometry(read, at).is_empty());
        }
        assert!(geometry(|_, _| false, at).is_empty());
    }

    #[test]
    fn a_piece_takes_what_changed() {
        let mut title = [7u8; PIECE];
        let mut ours = vec![0u8; PIECE];
        assert_eq!(merge(&title, &mut ours, true), PIECE / 4);
        assert_eq!(ours, title);
        assert_eq!(merge(&title, &mut ours, false), 0);
        // A dword of it.
        title[41] = 8;
        assert_eq!(merge(&title, &mut ours, false), 1);
        assert_eq!(ours, title);
    }

    #[test]
    fn a_frames_notes_come_once_a_page_and_the_builders_stay_the_builders() {
        let notes = noted(vec![(0x2_0000, false), (0x10, true), (0x1_0005, true), (0x1_0005, false), (0x1_0004, false)]);
        assert_eq!(notes, [(0x10, true), (0x1_0004, false), (0x1_0005, true), (0x2_0000, false)]);
        // Those below 4 GiB are the first of them.
        assert_eq!(notes.partition_point(|p| p.0 < 0x1_0000), 1);
    }

    #[test]
    fn the_walking_programs_pages_from_faults_go_first() {
        assert_eq!(FAULT_PAGES, 8192);
        assert_eq!(goes_first([100, 100], FAULT_PAGES), 0);
        assert_eq!(goes_first([1, FAULT_PAGES / 2], FAULT_PAGES), 0);
        // The builder's once they are more than half, or all there are.
        assert_eq!(goes_first([100, FAULT_PAGES / 2 + 1], FAULT_PAGES), 1);
        assert_eq!(goes_first([0, 3], FAULT_PAGES), 1);
    }

    #[test]
    fn a_page_wanted_is_counted_once() {
        let mut bits = vec![0u64; 4];
        assert!(mark(&mut bits, 65));
        assert!(!mark(&mut bits, 65));
        assert!(mark(&mut bits, 64));
        assert_eq!(bits, [0, 3, 0, 0]);
        // Past the directory: not counted.
        assert!(!mark(&mut bits, 4 * 64));
    }

    #[test]
    fn a_packed_structure_is_known_by_its_header() {
        let header = |version: u32, nodes: u32| u64::from(version) | u64::from(nodes) << 32;
        assert_eq!(packed_len(header(0x8102_0006, 0x9a1a)), Some(0x9a1b * 64));
        // Unpacked, a top level, another version, no size, too big.
        assert_eq!(packed_len(header(0x8002_0006, 0x9a1a)), None);
        assert_eq!(packed_len(header(0x0000_0006, 5)), None);
        assert_eq!(packed_len(header(0x8102_0007, 5)), None);
        assert_eq!(packed_len(header(0x8102_0006, 0)), None);
        assert_eq!(packed_len(header(0x8102_0006, 1 << 21)), None);
    }
}
