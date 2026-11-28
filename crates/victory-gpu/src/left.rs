//! What shaders wrote to card copies of write-combining title memory, left
//! on the card (`VICTORY_SKIP_WC_BACK`, mode 2 by default, `=0` off; after PalmVI
//! 06edf544f; `=3` also where the title shows the pages at other addresses,
//! after PalmVI 5eed56798 and d4e90ed7c), and its audit (`VICTORY_CENSUS=1`).
//!
//! The title reserves some of its memory write-combining: memory its CPU
//! writes and does not read. What shaders write to a copy of such memory in
//! the card's memory (`crate::shadow::cards`) was fetched and merged into
//! title memory after every flight (98 MB a frame in run record-1), only to
//! be read by nobody on the CPU. With the switch, a piece the compare found
//! changed is left on the card instead, when no other copy of ours holds
//! any of its bytes ([`skip_back`]): the copy that wrote it keeps what it
//! wrote, title memory keeps what it had.
//!
//! What it must keep: title memory there is not what the card wrote, so any
//! reader of those title bytes other than the copy that holds them has the
//! piece fetched and merged first, as copy-back would have ([`before_take`],
//! `crate::host::leftq::fetch`), and that piece is never left again. The
//! readers, each of which calls [`before_take`] before it reads:
//!
//! - every refresh of a copy (`Shadow::update`: its stale pages, its taken
//!   pages, whole copies) and the title's bytes put in under shader writes
//!   (`Shadow::merge_under_writes`);
//! - a submission's steps: the title's buffer copies (source and
//!   destination), fills of upload memory, texture uploads, copies at their
//!   place, work that reads title memory, and our work that may read or
//!   write it anywhere (then every piece is fetched);
//! - texture stagings (`crate::texture`: refreshes, fingerprints, the look
//!   at whether a texture changed, any read of a texture's memory);
//! - the arenas of title memory by address (`crate::memory`).
//!
//! Those that keep title bytes and take them again only where title memory
//! was written (copies, textures, arenas) are told: a piece left is noted
//! in the write journal as a landing is (`crate::watch::landed_beside`),
//! so that their next refresh reads it, and so fetches it.
//!
//! What the copy that holds the bytes does itself: its refreshes there take
//! only the dwords where title memory differs from its CPU side, which are
//! the title's new ones, and send them up; what the card wrote around them
//! stays ([`own_left_in`]). (Where the title stores again, after the card's
//! write, the very value the CPU side held before it, that store is not
//! seen: the card's value stays, as in PalmVI.)
//!
//! Not left, ever: memory the card reads where the title has it (in place,
//! `crate::inplace`; renderer port 1's 32-bit graphics memory,
//! `crate::port`), pages an arena of `crate::memory` holds, copies with
//! stores outside their declared ranges (`Shadow::keep_stores`), and, but
//! with mode 3, a page the title shows at another address too. A piece is
//! fetched by the original copy's bytes only: memory the title frees or
//! maps anew forgets what was left there (but see "stranded" below).
//!
//! # Copies on the card to the other addresses of the pages (mode 3)
//!
//! The title shows some of its physical pages at several addresses, and
//! has title buffers at more than one of them. Mode 2 fetches every piece
//! of such a page: the copies at the other addresses take the bytes from
//! title memory and send them up again. With `VICTORY_SKIP_WC_BACK=3` such
//! a piece is left too, when every copy of ours at the other addresses is
//! a plain copy in the card's memory that holds the whole piece
//! ([`judge`]): those copies are the piece's *holders*, and each is given
//! the bytes by a copy on the card, `CopyBufferRegion` from the writer's
//! card side to its own ([`Forward`], `Host::card_before`).
//!
//! PalmVI notes the piece left for each holder too and sends the copies on
//! a copy queue of their own, 64 lists in the air, each waited for by the
//! host's queue. Here a copy's refresh compares title memory with its CPU
//! side, and its compare on the card with hashes of its pieces, so more
//! must hold, and the copies go on the host's own queue, in a list ahead
//! of the next submission's: nothing waits for them, on the card or off
//! it, and there is no count of lists to run out of.
//!
//! What holds, for a piece P of writer W with holder H (W's offsets
//! `[lo, hi)`, H's the same title bytes at the other address):
//!
//! 1. **Order on the card.** The copy W to H is recorded in a list sent
//!    ahead of the first submission prepared after the landing that left P
//!    ([`forward_step`]; asked for under the book's lock, so whoever finds
//!    H a holder finds the copy asked for). It runs behind every flight
//!    sent before (the one that wrote P, and any later one that writes W
//!    again: H gets those bytes too, and that flight's landing asks for
//!    the copy again) and ahead of everything sent after.
//! 2. **Nothing older goes up after it.** Before the copy is recorded H's
//!    CPU side over P becomes W's (`Shadow::forwarded_from`): the title
//!    bytes the card's were written over. What H had still to send there
//!    is dropped. From then on H's refresh sends what W's would: the
//!    dwords where title memory differs from that reference, which are the
//!    title's stores made after W took it. A store the title made before
//!    the shader's write, and which H had not sent or not seen, is in the
//!    reference or older than it: it is never sent over the card's bytes.
//! 3. **Nothing newer is lost by it.** The copy puts W's card bytes over
//!    all of P in H. Title bytes H had taken and W not yet (the two sides
//!    differ there, and title memory is not the reference's) stay in H's
//!    CPU side and go up again behind the copy, in the same submission's
//!    sends; title bytes W took and has not sent go up to H as to W. The
//!    list of copies runs ahead of the list of sends.
//! 4. **A holder's compare does not take the bytes for its own shaders'.**
//!    The hashes of H's pieces (`Card::before`) are made again for P in
//!    the list that follows the copy. And no shader of H's writes P while
//!    H holds it: H is made a holder only with no write declared over P
//!    by a flight in the air ([`DECIDE`], held by the landing that judges
//!    and by `Shadow::writing`), and when one is declared later the piece
//!    is fetched into title memory first and never left again
//!    ([`declaring`]). Two copies' shader writes of one piece are put in
//!    order in title memory only: were H's landed there while W held P,
//!    W's refresh would take them for the title's stores and send them
//!    over newer writes of its own shaders'.
//! 5. **A holder's CPU side is not title memory there, and nobody reads it
//!    for that.** It is the reference of 2. H reads title memory over P
//!    without a fetch ([`Left::kept_by`]: that is what the mode saves),
//!    and only to compare. Every other reader, at any address of the
//!    pages, has P fetched from W first as in mode 2 (the book knows the
//!    piece by its other addresses and by its page); then the holders'
//!    pages are noted written, and they take title memory as any copy.
//! 6. **One writer a page.** A piece whose page is left through another
//!    address already is fetched, and so is one a copy of ours holds at
//!    the same address.
//!
//! Refused, each counted on the frame line by its reason ([`REFUSALS`])
//! and fetched as mode 2 fetches it: another copy at the same address; a
//! page in more than `crate::watch::MANY_VIEWS` views or with
//! `VICTORY_VIEW_GUARD=0` (a write of the title's through one view is then
//! not told to the copies at the others); another address that the title
//! did not reserve write-combining (its CPU may read there, which nothing
//! of ours sees; `VICTORY_SKIP_WC_BACK_ALIAS=1` leaves those too, as PalmVI
//! does); another address read in place or held by an arena; a holder in
//! the PC's memory, in place, with undeclared stores, in the writer's own
//! buffer, or the writer itself; a holder of only part of the piece; a
//! holder with shader writes in the air over the piece; other holders than
//! when the piece was first left; a page another copy has left.
//!
//! Dropped, not refused: a copy on the card whose piece was fetched,
//! forgotten or taken back before the next submission (title memory holds
//! the bytes then, and the holder takes them from there).
//!
//! Stranded: where the title lets go of the writer's address (unmaps it,
//! maps other pages there) and still shows the pages at another address
//! known when the piece was left, the bytes are still the title's there.
//! The piece is kept at that address with no holder ([`Book::strand`]):
//! any reader there fetches it, merged at that address, and so does the
//! writer's next refresh or declared write, before its card side there is
//! used for anything else.
//!
//! `VICTORY_CENSUS=1` counts with mode 3 also the title bytes of pieces
//! left that are read at another address of their pages by other than a
//! holder (must be 0), and checks the quick search for holders against a
//! walk of every table (misses must be 0).
//!
//! `VICTORY_SKIP_WC_BACK=1` (every piece of such memory, no fetch) is not
//! in this build: it gave a black world and lost room lighting on an RTX
//! 5070 Ti in PalmVI.
//!
//! # Locks
//!
//! The locks a fetch, a refresh and a landing take, in the order each path
//! takes them (a path never takes one to the left of one it holds):
//!
//! | path | thread | order |
//! |---|---|---|
//! | a reader, [`before_take`] | the submitting thread under the submission lock (`left_before_steps`, `merge_under_writes`, the sweep); threads of `crate::shadow::wide` (refreshes of copies and textures); whoever holds the arenas' lock (`crate::memory`) or reads a texture's memory | the caller's locks, then [`GATE`], [`BOOK`] shared, the views' (`crate::inplace`), [`BOOK`] alone, and with no lock of the book: the fetch's rig, the writer's `dirty` and `sent` notes, the order of the queue (let go before the wait), the card clock's state, **the wait for the card**, the writer's `merging` alone, its `before`, its notes again, and title memory written as a landing writes it (the guards and the journal of `crate::watch`) |
//! | a refresh, `Shadow::update` | the submitting thread and the threads of `wide` | as a reader; with pieces of its own in the range [`GATE`] stays held, then the asking of `crate::watch` (its ranges, the guards, the journal), the copy's pages, [`BOOK`] shared, the copy's `dirty` notes |
//! | a landing, [`skip_back`] | the thread that lands flights | with mode 3 [`DECIDE`], held to the note in the book; the copies' and the arenas' registers and the views' (each let go), the holders' declared ranges (each let go), [`BOOK`] alone and in it the list of copies asked for (let go), the journal. Later, per piece fetched as before: the copy's `merging` shared, its notes, title memory. Never [`GATE`] |
//! | a declared write, [`declaring`] (mode 3) | the submitting thread | the submission lock, then [`DECIDE`], [`BOOK`] shared, the copy's declared ranges; for a fetch [`DECIDE`] is let go first, then as a reader |
//! | the copies on the card, [`forward_step`] (mode 3) | the submitting thread, before the sends are taken | the submission lock, then [`GATE`], the list of copies asked for (let go), [`BOOK`] shared (let go), the writer's CPU side's lock and inside it the holder's, their notes. Then, [`GATE`] let go: the order of the queue and the card's work, as the sends |
//! | the runtime changes a mapping, the title frees memory ([`forget`]) | a thread of the title's | the mapping's lock alone (`crate::watch`), the write-combining reservations, [`BOOK`] alone, the guards. Never [`GATE`] |
//! | a write of the title's to guarded memory | a thread of the title's | the guards and the journal only: nothing of `crate::watch` calls in here but [`forget`] |
//! | a submission's sends to card copies, `Host::card_before` | the submitting thread | the order of the queue, the card's work, the copies' notes. Nothing of this file |
//!
//! Waits with locks held: the fetch waits for the card with [`GATE`] and
//! its rig held (and the caller's: the submission lock, the arenas'). On
//! the host's queue it stands behind every flight sent so far, which the
//! card runs with no help of the CPU's (the title's own queue waits are
//! kept on the CPU, `crate::queue`; `VICTORY_UP_QUEUE=1`'s wait is for a
//! signal already on its queue), so that wait ends. [`BOOK`], the journal,
//! the guards and the mapping's lock are never held across it, so the
//! title's threads (a write fault, a change of mapping) do not wait for a
//! fetch.
//!
//! What hung the process (two runs on an RX 7900 XT, seconds after the
//! world's first pieces were left): a refresh of a copy with pieces of its
//! own keeps [`GATE`] for all of `Shadow::update`, and a megabyte or more of
//! stale pages there (the pages of the pieces just left are noted written,
//! so megabytes of them are stale at the copy's next refresh) were taken on
//! several threads, `Shadow::take_all` through `wide`. The thread that
//! waits in `wide` takes whatever job is in the pool's queue: another
//! refresh that no helper has begun, which asked for the gate on the
//! thread that held it. A lock of `std` is not taken twice by one thread:
//! that thread stood for ever, its submission with it under the submission
//! lock, every other reader behind the gate, and the title behind the
//! submission. No device was lost and nothing was logged.
//!
//! What keeps it out now: with the gate held ([`gate_held`]) a refresh
//! spreads its pages over threads that run nothing but that work
//! (`crate::land::each_under`, as the title's bytes put in under shader
//! writes do for the same reason), so no job of another's runs on the
//! thread that holds the gate; and the gate knows whether the thread that
//! asks holds it ([`Gate`]): asked for again by that thread, it stops the
//! process and says so, where it stood without a word.
//!
//! What is said whenever a mode is on: a thread that has waited [`STEP`]
//! for the gate names who holds it, for what and since when; a fetch that
//! has waited so long for the card names its path, its pieces and bytes,
//! how far the fences are and the locks its thread holds
//! (`crate::host::leftq`); both go on waiting, and say so again every
//! [`STEP`]. At debug level the first [`SAY_FIRST`] pieces left are named
//! with the copy that wrote them.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock, Weak};

use crate::shadow::Shadow;

/// The console's protection bit of write-combined memory (`xmem`'s
/// `note_protection`).
const WRITE_COMBINED: u32 = 0x400;
/// Pieces are kept by the 64 KiB of title memory they lie in (a card copy's
/// pieces lie in one each: `crate::shadow::card_piece_of`).
const PIECE: u64 = 64 << 10;
/// So many of the first pieces left are said in the log, at debug level:
/// where, how long, and the copy that wrote them.
const SAY_FIRST: u64 = 64;

/// `VICTORY_SKIP_WC_BACK`: 2 (the default, and when unset) leaves pieces on
/// the card (see the file); 3: as 2, and also pieces whose pages the title
/// shows at other addresses, whose copies there get the bytes on the card
/// (see the file, "Copies on the card"; after PalmVI 5eed56798); 0: off.
/// Other values are not modes of this build: off. (One run of 3.5 minutes
/// in the world on an RX 7900 XT with 2: no stall, the census 0 throughout,
/// 77 to 93 MB a frame left on the card. 3 has not run in the game.)
/// In this crate's unit tests unset is off, so that tests of copy-back see
/// every piece fetched and merged unless they ask otherwise.
pub(crate) fn mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        let asked = std::env::var("VICTORY_SKIP_WC_BACK").unwrap_or_default();
        match asked.trim() {
            "0" => 0,
            "" if cfg!(test) => 0,
            "" | "2" => {
                tracing::info!(
                    "VICTORY_SKIP_WC_BACK=2 (the default; =0 turns it off): what shaders write to card copies of write-combining title memory stays on the card where no other copy holds it, fetched before anything else reads it"
                );
                2
            }
            "3" => {
                tracing::info!(
                    "VICTORY_SKIP_WC_BACK=3: as 2 (what shaders write to card copies of write-combining title memory stays on the card), and \
                     where the title shows the pages at other addresses too, the card copies there are given the bytes on the card; \
                     VICTORY_CENSUS=1 audits it"
                );
                3
            }
            other => {
                tracing::warn!("VICTORY_SKIP_WC_BACK={other} is not a mode of this build (2, 3): off");
                0
            }
        }
    })
}

/// Whether a mode is on.
pub(crate) fn on() -> bool {
    mode() != 0
}

/// Whether pieces whose pages the title shows at other addresses too may
/// be left, their bytes copied on the card to the copies there
/// (`VICTORY_SKIP_WC_BACK=3`). Not with `crate::host::upcheck`, which
/// compares what went up with what the card holds.
pub(crate) fn forwards() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| mode() == 3 && !crate::host::upcheck::on())
}

/// `VICTORY_SKIP_WC_BACK_ALIAS=1` (off by default; PalmVI's rule): with
/// mode 3, a piece is left also when another address of its pages is not in
/// a reservation the title asked write-combining. Off, such a piece is
/// fetched as before and counted: through an address that is not
/// write-combining the title's own CPU may read, which nothing of ours
/// sees, and title memory there would not hold what the card wrote.
fn any_alias() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SKIP_WC_BACK_ALIAS").is_some_and(|v| v == "1"))
}

/// `VICTORY_CENSUS=1`: the audit (title bytes of pieces left on the card
/// read by anything but the copy that holds them, which must be none), on
/// the frame line.
pub(crate) fn census() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| on() && std::env::var_os("VICTORY_CENSUS").is_some_and(|v| v == "1"))
}

/// The title's write-combining reservations: start, bytes.
static WRITE_COMBINING: RwLock<BTreeMap<usize, usize>> = RwLock::new(BTreeMap::new());

/// The runtime reserved `[at, at + len)` for the title (`XMemVirtualAlloc`
/// at an address of its choice), with the title's own console `protect`
/// (before the runtime cleans it). Kept only with a mode on.
pub fn title_memory_reserved(at: usize, len: usize, protect: u32) {
    if protect & WRITE_COMBINED == 0 || at == 0 || len == 0 || !on() {
        return;
    }
    WRITE_COMBINING
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .insert(at, len);
}

/// Whether all of `[at, end)` is in a reservation the title asked
/// write-combining.
fn write_combining(at: u64, end: u64) -> bool {
    let map = WRITE_COMBINING.read().unwrap_or_else(|e| e.into_inner());
    map.range(..=at as usize)
        .next_back()
        .is_some_and(|(&base, &len)| (base + len) as u64 >= end)
}

/// A piece left on the card: the copy whose card side holds what was
/// written, and the offsets in it (in order, apart).
struct Left {
    writer: Arc<Shadow>,
    /// The title address the writer's first byte is merged at when the
    /// piece is fetched: the writer's own start, but for a stranded piece.
    base: u64,
    runs: Vec<(usize, usize)>,
    /// For a view of the title's physical pages: the section and offset of
    /// the 64 KiB page.
    page: Option<(usize, u64)>,
    /// Mode 3: the other addresses of the 64 KiB page when the piece was
    /// left (none: shown at one address).
    aliases: Vec<u64>,
    /// Mode 3: the card copies at those addresses, which are given the
    /// piece's bytes on the card and keep them as the writer does.
    holders: Vec<Held>,
    /// The title let go of the piece's address while it still shows the
    /// pages at another: the piece is kept at that one (`base`), for
    /// whoever reads there, and no copy holds it for its own any more.
    stranded: bool,
}

/// A copy that holds a piece of another copy's, given it on the card
/// (mode 3): the copy (not kept alive by this), its number, and the 64 KiB
/// address at which it holds the page.
struct Held {
    copy: Weak<Shadow>,
    id: u64,
    at: u64,
}

impl Left {
    /// Whether copy `id` is given this piece's bytes on the card.
    fn holder(&self, id: u64) -> bool {
        self.holders
            .iter()
            .any(|held| held.id == id && held.copy.strong_count() > 0)
    }

    /// Whether copy `id` (0: no copy) reads title memory here without the
    /// piece fetched: it has the card's bytes itself, as the writer or by a
    /// copy on the card, and takes of title memory only what the title
    /// changed. Nobody has a stranded piece.
    fn kept_by(&self, id: u64) -> bool {
        id != 0 && !self.stranded && (self.writer.id() == id || self.holder(id))
    }

    /// The live holders, as (number, 64 KiB address), in order.
    fn held(&self) -> Vec<(u64, u64)> {
        let mut held: Vec<(u64, u64)> = self
            .holders
            .iter()
            .filter(|held| held.copy.strong_count() > 0)
            .map(|held| (held.id, held.at))
            .collect();
        held.sort_unstable();
        held
    }
}

/// The pieces left, by the title address of their 64 KiB, and of those in
/// view memory by the section and offset of the page (a view of it the
/// title maps later at another address reads the same bytes); of those
/// left with other addresses known (mode 3), by each of those; and the
/// 64 KiB pieces taken once while left, which are fetched as before from
/// then on.
struct Book {
    left: BTreeMap<u64, Left>,
    by_page: BTreeMap<(usize, u64), u64>,
    alias: BTreeMap<u64, u64>,
    never: BTreeSet<u64>,
    /// Stranded pieces in `left`.
    stranded: usize,
    /// The copies that are holders of pieces in `left`, by their number:
    /// of how many.
    holding: BTreeMap<u64, usize>,
    /// Whether its pieces are counted in [`LEFT_N`]: the book of the
    /// pieces left ([`BOOK`]), not one a test makes.
    counted: bool,
}

impl Book {
    const fn new(counted: bool) -> Book {
        Book {
            left: BTreeMap::new(),
            by_page: BTreeMap::new(),
            alias: BTreeMap::new(),
            never: BTreeSet::new(),
            stranded: 0,
            holding: BTreeMap::new(),
            counted,
        }
    }

    /// Puts a piece in at `key` (none is there).
    fn insert(&mut self, key: u64, left: Left) {
        if let Some(page) = left.page {
            self.by_page.insert(page, key);
        }
        for &alias in &left.aliases {
            self.alias.insert(alias, key);
        }
        self.stranded += usize::from(left.stranded);
        for held in &left.holders {
            *self.holding.entry(held.id).or_insert(0) += 1;
        }
        self.left.insert(key, left);
        if self.counted {
            LEFT_N.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// Takes the piece at `key` out.
    fn remove(&mut self, key: u64) -> Option<Left> {
        let left = self.left.remove(&key)?;
        if self.counted {
            LEFT_N.fetch_sub(1, Ordering::SeqCst);
        }
        if let Some(page) = left.page
            && self.by_page.get(&page) == Some(&key)
        {
            self.by_page.remove(&page);
        }
        for alias in &left.aliases {
            if self.alias.get(alias) == Some(&key) {
                self.alias.remove(alias);
            }
        }
        self.stranded -= usize::from(left.stranded);
        for held in &left.holders {
            self.held_no_longer(held.id);
        }
        Some(left)
    }

    /// The pieces that hold title bytes of `[from, to)`, at those addresses
    /// or at another address of the same pages: one known when the piece
    /// was left, or, for view memory, any (`pages`:
    /// [`crate::inplace::page_keys`] of the range).
    fn touching(&self, from: u64, to: u64, pages: &[(usize, (usize, u64))]) -> Vec<u64> {
        let mut keys: Vec<u64> = self
            .left
            .range(from / PIECE * PIECE..to)
            .filter(|(_, left)| {
                let (a, b) = title_span(left);
                a < to && b > from
            })
            .map(|(&key, _)| key)
            .collect();
        let mut more = false;
        for (&alias, &key) in self.alias.range(from / PIECE * PIECE..to) {
            if let Some(left) = self.left.get(&key) {
                let (a, b) = title_span(left);
                let (a, b) = (a - key + alias, b - key + alias);
                if a < to && b > from {
                    keys.push(key);
                    more = true;
                }
            }
        }
        if !self.by_page.is_empty() {
            keys.extend(
                pages
                    .iter()
                    .filter_map(|(_, page)| self.by_page.get(page).copied()),
            );
            more = true;
        }
        if more {
            keys.sort_unstable();
            keys.dedup();
        }
        keys
    }

    /// The pieces to fetch before `who` goes on, of those that hold title
    /// bytes of `[from, to)` ([`Book::touching`]).
    fn wanted(&self, who: For, from: u64, to: u64, pages: &[(usize, (usize, u64))]) -> Vec<u64> {
        let mut keys: Vec<u64> = match who {
            For::Reader(taker) => self
                .touching(from, to, pages)
                .into_iter()
                .filter(|key| self.left.get(key).is_some_and(|left| !left.kept_by(taker)))
                .collect(),
            For::Writing(id) => self
                .touching(from, to, pages)
                .into_iter()
                .filter(|key| self.left.get(key).is_some_and(|left| left.holder(id)))
                .collect(),
            For::Orphan(id) => self
                .left
                .iter()
                .filter(|(_, left)| left.writer.id() == id)
                .map(|(&key, _)| key)
                .collect(),
        };
        // A stranded piece is in its writer's card side at offsets the
        // writer's own address no longer speaks for: before that copy is
        // refreshed or written anywhere, it is fetched.
        let of = match who {
            For::Reader(id) | For::Writing(id) => id,
            For::Orphan(_) => 0,
        };
        if self.stranded != 0 && of != 0 {
            keys.extend(
                self.left
                    .iter()
                    .filter(|(_, left)| left.stranded && left.writer.id() == of)
                    .map(|(&key, _)| key),
            );
            keys.sort_unstable();
            keys.dedup();
        }
        keys
    }

    /// Whether the copy on the card `job` asks for still stands: its piece
    /// is left by the same copy, holds the offsets, and the copy it goes to
    /// is one of the piece's holders.
    fn stands(&self, job: &Forward) -> bool {
        self.left.get(&job.key).is_some_and(|left| {
            !left.stranded
                && left.writer.id() == job.from.id()
                && left.holder(job.to.id())
                && left
                    .runs
                    .iter()
                    .any(|&(a, b)| a <= job.lo && job.lo + job.bytes <= b)
        })
    }

    /// Copy `id` is a holder of one piece fewer.
    fn held_no_longer(&mut self, id: u64) {
        if let Some(n) = self.holding.get_mut(&id) {
            *n -= 1;
            if *n == 0 {
                self.holding.remove(&id);
            }
        }
    }

    /// Whether a declared write of copy `id` may have pieces to fetch
    /// first: it is a holder of some piece, or a piece of its own is
    /// stranded.
    fn may_hold(&self, id: u64) -> bool {
        self.holding.contains_key(&id) || self.stranded != 0
    }

    /// The title lets go of `alias`, another address (64 KiB) of a piece's
    /// page known when it was left: the piece is known there no longer,
    /// and the copies that held it there are holders no longer. Returns
    /// those that live, each with the offsets at which it held the piece.
    fn unalias(&mut self, alias: u64) -> Vec<(Arc<Shadow>, usize, usize)> {
        let mut out = Vec::new();
        let Some(key) = self.alias.remove(&alias) else {
            return out;
        };
        let Some(left) = self.left.get_mut(&key) else {
            return out;
        };
        left.aliases.retain(|&a| a != alias);
        let (base, runs) = (left.base, left.runs.clone());
        let mut gone = Vec::new();
        left.holders.retain(|held| {
            if held.at != alias {
                return true;
            }
            gone.push(held.id);
            if let Some(copy) = held.copy.upgrade() {
                let start = copy.span().0;
                for &(lo, hi) in &runs {
                    let at = base
                        .wrapping_add(lo as u64)
                        .wrapping_sub(key)
                        .wrapping_add(alias)
                        .wrapping_sub(start) as usize;
                    out.push((copy.clone(), at, at + (hi - lo)));
                }
            }
            false
        });
        for id in gone {
            self.held_no_longer(id);
        }
        out
    }

    /// The title lets go of the address of the piece at `key` and still
    /// shows its pages at `alias` (64 KiB): the piece is kept there,
    /// stranded. Whether it was.
    fn strand(&mut self, key: u64, alias: u64) -> bool {
        if self.left.contains_key(&alias) {
            return false;
        }
        let Some(mut left) = self.remove(key) else {
            return false;
        };
        left.base = left.base.wrapping_sub(key).wrapping_add(alias);
        left.aliases.retain(|&a| a != alias);
        // (No longer counted: `remove` saw to that.)
        left.holders.clear();
        left.stranded = true;
        self.insert(alias, left);
        true
    }
}

/// Whom pieces are fetched for ([`Book::wanted`]).
#[derive(Clone, Copy)]
enum For {
    /// A reader of title memory, for this copy (0: for none): every piece
    /// there but those the copy has itself.
    Reader(u64),
    /// This copy's shaders are about to write there: the pieces it holds
    /// of other copies'.
    Writing(u64),
    /// Nothing but the book holds this copy: all of its pieces, wherever.
    Orphan(u64),
}

static BOOK: RwLock<Book> = RwLock::new(Book::new(true));

/// Pieces in [`BOOK`], for the look every reader makes; and fetches under
/// way (their pieces are out of the book, their bytes not in title memory
/// yet).
static LEFT_N: AtomicU64 = AtomicU64::new(0);
static FETCHING: AtomicU64 = AtomicU64::new(0);

/// Held by every fetch, and by a refresh of a copy that holds pieces left
/// in its range: a fetch merges into the copy's CPU side, which such a
/// refresh writes.
///
/// One thread at a time, and never twice by one thread: a thread that
/// holds it and asks again is told so ([`Nested`]) and does not wait for
/// itself (see the file, "Locks").
///
/// It says who holds it: a thread that has waited [`STEP`] for it says so
/// in the log, with the holder, and goes on waiting in such steps.
struct Gate {
    held: Mutex<Option<Holder>>,
    freed: Condvar,
}

/// Who holds a [`Gate`]: the thread, what it holds it for, for which copy
/// (0: for none), and since when.
struct Holder {
    thread: std::thread::Thread,
    what: &'static str,
    copy: u64,
    since: std::time::Instant,
}

/// A wait for the gate or for the card (`crate::host::leftq`) is said in
/// the log when it has lasted so long, and again every time it has lasted
/// so long once more.
pub(crate) const STEP: std::time::Duration = std::time::Duration::from_secs(2);

/// A thread, by its name or else its number, for the log.
pub(crate) fn thread_name(thread: &std::thread::Thread) -> String {
    match thread.name() {
        Some(name) => name.to_owned(),
        None => format!("{:?}", thread.id()),
    }
}

/// The line of a thread that waits for a gate `holder` holds.
fn gate_waited(what: &str, copy: u64, waited: std::time::Duration, holder: &Holder) -> String {
    format!(
        "pieces left on the card: {what} (copy {copy}) has waited {:.0} s for the gate, which thread {} holds for {} (copy {}) since \
         {:.0} s; still waiting (VICTORY_SKIP_WC_BACK)",
        waited.as_secs_f64(),
        thread_name(&holder.thread),
        holder.what,
        holder.copy,
        holder.since.elapsed().as_secs_f64()
    )
}

/// The gate, held until dropped.
pub(crate) struct Gated<'a>(&'a Gate);

/// The thread that asks for a gate holds one.
#[derive(Debug)]
struct Nested;

thread_local! {
    /// This thread holds a [`Gate`].
    static HOLDING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

impl Gate {
    const fn new() -> Gate {
        Gate {
            held: Mutex::new(None),
            freed: Condvar::new(),
        }
    }

    /// Waits until no other thread holds the gate, and holds it, for
    /// `what` of copy `copy` (0: of none). A wait of `step` is said in the
    /// log, each time.
    fn enter(
        &self,
        what: &'static str,
        copy: u64,
        step: std::time::Duration,
    ) -> Result<Gated<'_>, Nested> {
        if HOLDING.get() {
            return Err(Nested);
        }
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        if held.is_some() {
            let began = std::time::Instant::now();
            while held.is_some() {
                let (still, waited) = self
                    .freed
                    .wait_timeout(held, step)
                    .unwrap_or_else(|e| e.into_inner());
                held = still;
                if waited.timed_out()
                    && let Some(holder) = held.as_ref()
                {
                    GATE_WAITS.fetch_add(1, Ordering::Relaxed);
                    tracing::warn!("{}", gate_waited(what, copy, began.elapsed(), holder));
                }
            }
        }
        *held = Some(Holder {
            thread: std::thread::current(),
            what,
            copy,
            since: std::time::Instant::now(),
        });
        HOLDING.set(true);
        Ok(Gated(self))
    }
}

impl Drop for Gated<'_> {
    fn drop(&mut self) {
        *self.0.held.lock().unwrap_or_else(|e| e.into_inner()) = None;
        HOLDING.set(false);
        self.0.freed.notify_one();
    }
}

static GATE: Gate = Gate::new();

/// How often a thread said that it had waited [`STEP`] for a gate.
static GATE_WAITS: AtomicU64 = AtomicU64::new(0);

/// [`GATE`], held. A thread that holds it already cannot be let in (the
/// fetch it asks for may merge into the copy whose refresh holds the gate)
/// and would wait for itself: the process stops with the reason.
fn gate(what: &'static str, copy: u64) -> Gated<'static> {
    match GATE.enter(what, copy, STEP) {
        Ok(gate) => gate,
        Err(Nested) => crate::host::fail(format_args!(
            "a thread that holds the gate of pieces left on the card asked for it again, for {what} (copy {copy}): work of \
             another's ran inside a refresh or a fetch (VICTORY_SKIP_WC_BACK)"
        )),
    }
}

/// Whether this thread holds the gate (a refresh of a copy with pieces of
/// its own, a fetch): it must run no work of another's meanwhile, which
/// may ask for the gate. `crate::shadow::wide` does, on the thread that
/// waits for its jobs.
pub(crate) fn gate_held() -> bool {
    HOLDING.get()
}

/// For tests: the gate, held as a refresh of a copy with pieces of its own
/// holds it.
#[cfg(test)]
pub(crate) fn hold_gate() -> Gated<'static> {
    gate("a test", 0)
}

/// Since the last frame line: pieces left and their bytes; pieces fetched
/// for a reader, their bytes, the fetches; pieces left again by their
/// copy's next flight (counted in the first two as well).
static STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// With [`census`], since the last frame line: title bytes of pieces left
/// read by other than a copy holding them, by a copy's refresh and by
/// anything else; and of both, those read at another address of the pages
/// than the writer's (where mode 3's holders are).
static STALE: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

thread_local! {
    /// The copy this thread refreshes ([`Taking`]): its own pieces are not
    /// stale for it.
    static TAKER: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// While alive, this thread reads title memory for copy `id` (for the audit).
pub(crate) struct Taking(u64);

impl Taking {
    pub(crate) fn new(id: u64) -> Taking {
        Taking(TAKER.replace(id))
    }
}

impl Drop for Taking {
    fn drop(&mut self) {
        TAKER.set(self.0);
    }
}

/// `runs` (in order, apart) with `[lo, hi)` added.
fn add_run(runs: &mut Vec<(usize, usize)>, lo: usize, hi: usize) {
    if lo >= hi {
        return;
    }
    runs.push((lo, hi));
    runs.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(runs.len());
    for &(a, b) in runs.iter() {
        match out.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(b),
            _ => out.push((a, b)),
        }
    }
    *runs = out;
}

/// `runs` (in order, apart) with `[lo, hi)` taken out.
fn cut_run(runs: &mut Vec<(usize, usize)>, lo: usize, hi: usize) {
    let mut out = Vec::with_capacity(runs.len() + 1);
    for &(a, b) in runs.iter() {
        if b <= lo || a >= hi {
            out.push((a, b));
            continue;
        }
        if a < lo {
            out.push((a, lo));
        }
        if b > hi {
            out.push((hi, b));
        }
    }
    *runs = out;
}

/// The title memory a piece's runs hold.
fn title_span(left: &Left) -> (u64, u64) {
    let first = left.runs.first().map_or(0, |r| r.0);
    let last = left.runs.last().map_or(0, |r| r.1);
    (
        left.base.wrapping_add(first as u64),
        left.base.wrapping_add(last as u64),
    )
}

/// A piece to fetch: the copy whose card side holds it, the offsets, and
/// the title address the copy's first byte is merged at.
pub(crate) struct Piece {
    pub(crate) copy: Arc<Shadow>,
    pub(crate) lo: usize,
    pub(crate) hi: usize,
    pub(crate) base: u64,
}

/// Mode 3: why a piece whose pages the title shows at other addresses too
/// was not left (it is fetched as before), for the frame line.
const REFUSALS: [&str; 10] = [
    "another copy at the same address",
    "the pages' other views not told of writes",
    "another address not write-combining",
    "another address read in place or by an arena",
    "a holder in the PC's memory",
    "a holder not a plain copy",
    "a holder of part of the piece",
    "a holder shaders write in the air",
    "the holders changed",
    "another copy's piece of the same pages",
];
const SAME: usize = 0;
const VIEWS: usize = 1;
const PLAIN_ALIAS: usize = 2;
const READ_THERE: usize = 3;
const IN_PC: usize = 4;
const KIND: usize = 5;
const PART: usize = 6;
const WRITTEN: usize = 7;
const CHANGED: usize = 8;
const OTHER: usize = 9;

/// Since the last frame line, mode 3: pieces not left, by [`REFUSALS`].
static REFUSED: [AtomicU64; REFUSALS.len()] = [const { AtomicU64::new(0) }; REFUSALS.len()];

/// Since the last frame line, mode 3: pieces left with copies on the card
/// asked for, their bytes, the copies asked for; the copies made, their
/// bytes, the lists they went in; copies not made because their piece was
/// fetched or forgotten first; microseconds of bringing the holders' CPU
/// sides in line; bytes sent up again behind the copies; pieces stranded.
static FORWARD_STATS: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];

/// With [`census`] and mode 3, since the last frame line: pieces whose
/// holders were checked against a walk of every table, holders the walk
/// found, and of those the ones the quick search missed.
static HOLDERS_CHECKED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Mode 3: held while a landing decides whether a piece is left with
/// copies on the card to its holders, and while a copy's shader writes are
/// declared (`Shadow::writing`, [`declaring`]): a copy is made a holder
/// only with no shader write of its own declared over the piece, and none
/// is declared over a piece it holds.
static DECIDE: Mutex<()> = Mutex::new(());

/// A copy on the card (mode 3): what the card holds at offsets `[lo, lo +
/// bytes)` of `from` goes to offsets `[at, at + bytes)` of `to`, which
/// holds the same title bytes at another address; `key` is the piece's in
/// the book.
pub(crate) struct Forward {
    pub(crate) from: Arc<Shadow>,
    pub(crate) lo: usize,
    pub(crate) to: Arc<Shadow>,
    pub(crate) at: usize,
    pub(crate) bytes: usize,
    key: u64,
}

// SAFETY: the copies are reached only through their own locks and buffers.
unsafe impl Send for Forward {}

/// The copies on the card asked for and not made yet: made ahead of the
/// next submission's lists ([`forward_step`]). Locked inside [`BOOK`]'s
/// lock by a landing, alone by the submitting thread.
static FORWARDS: Mutex<Vec<Forward>> = Mutex::new(Vec::new());

/// How many are in [`FORWARDS`].
static FORWARDS_N: AtomicU64 = AtomicU64::new(0);

/// What a landing makes of a piece.
enum Verdict {
    /// Left on the card: the other addresses of its page, and the copies
    /// there with the title address at which each holds the piece.
    Leave(Vec<u64>, Vec<(Arc<Shadow>, u64)>),
    /// Fetched as before (as mode 2 would).
    Fetch,
    /// Mode 3 would have left it but for this ([`REFUSALS`]).
    Refuse(usize),
}

/// With a mode on, as the pieces a flight's shaders wrote are about to be
/// fetched: whether offsets `[lo, hi)` of `shadow`, a copy in the card's
/// memory, are left there instead. Then title memory keeps what it had and
/// the piece is noted (and told to the write journal, so that whoever keeps
/// those title bytes reads them again, and so fetches them first); with
/// mode 3 the copies of the same pages at other addresses are noted as its
/// holders and a copy on the card to each is asked for. If not, it is
/// fetched as before, with what was left of it earlier.
pub(crate) fn skip_back(shadow: &Arc<Shadow>, lo: usize, hi: usize) -> bool {
    if !on() || lo >= hi || !shadow.carded() {
        return false;
    }
    let (start, end) = (shadow.span().0 + lo as u64, shadow.span().0 + hi as u64);
    let key = start / PIECE * PIECE;
    if end > key + PIECE {
        return false;
    }
    let deciding = forwards().then(|| DECIDE.lock().unwrap_or_else(|e| e.into_inner()));
    let (aliases, targets) = match judge(shadow, start, end) {
        Verdict::Leave(aliases, targets) => (aliases, targets),
        Verdict::Fetch => {
            taken_back(shadow, key, lo, hi);
            return false;
        }
        Verdict::Refuse(why) => {
            REFUSED[why].fetch_add(1, Ordering::Relaxed);
            taken_back(shadow, key, lo, hi);
            return false;
        }
    };
    let page = crate::inplace::page_keys(key as usize, 1)
        .first()
        .filter(|p| p.0 == key as usize)
        .map(|p| p.1);
    // Whether it is left, and new in the book; or why not.
    let noted: Result<bool, Option<usize>> = {
        let mut book = BOOK.write().unwrap_or_else(|e| e.into_inner());
        let noted = note(&mut book, shadow, key, (lo, hi), page, &aliases, &targets);
        if noted.is_ok() && !targets.is_empty() {
            // Under the book's lock: whoever finds the copy a holder finds
            // the copy on the card asked for.
            let mut forwards = FORWARDS.lock().unwrap_or_else(|e| e.into_inner());
            for (copy, at) in &targets {
                forwards.push(Forward {
                    from: shadow.clone(),
                    lo,
                    to: copy.clone(),
                    at: (at - copy.span().0) as usize,
                    bytes: hi - lo,
                    key,
                });
            }
            FORWARDS_N.store(forwards.len() as u64, Ordering::SeqCst);
        }
        noted
    };
    let new = match noted {
        Ok(new) => new,
        Err(why) => {
            if let Some(why) = why {
                REFUSED[why].fetch_add(1, Ordering::Relaxed);
            }
            taken_back(shadow, key, lo, hi);
            return false;
        }
    };
    drop(deciding);
    if !targets.is_empty() {
        FORWARD_STATS[0].fetch_add(1, Ordering::Relaxed);
        FORWARD_STATS[1].fetch_add(end - start, Ordering::Relaxed);
        FORWARD_STATS[2].fetch_add(targets.len() as u64, Ordering::Relaxed);
    }
    // At debug level: the first pieces left, with the copy that wrote them.
    static SAID: AtomicU64 = AtomicU64::new(0);
    if new && SAID.load(Ordering::Relaxed) < SAY_FIRST && tracing::enabled!(tracing::Level::DEBUG) {
        let n = SAID.fetch_add(1, Ordering::Relaxed) + 1;
        if n <= SAY_FIRST {
            let (at, len) = shadow.span();
            tracing::debug!(
                "piece left on the card, {n} of the first {SAY_FIRST}: {start:#x}+{:#x}, written by shaders to copy {} ({} \
                 {at:#x}+{len:#x}), {} other addresses of its pages, {} copies there (VICTORY_SKIP_WC_BACK)",
                end - start,
                shadow.id(),
                shadow.kind(),
                aliases.len(),
                targets.len()
            );
        }
    }
    STATS[0].fetch_add(1, Ordering::Relaxed);
    STATS[1].fetch_add(end - start, Ordering::Relaxed);
    // After the note: whoever reads these pages from now on sees the note
    // first. Its 4 KiB pages are written as far as those who keep title
    // bytes can tell, as after a landing (at every address of the pages).
    let pages: Vec<usize> = (start as usize / 4096..(end as usize).div_ceil(4096)).collect();
    let _ = crate::watch::landed_beside(&pages, &[(start as usize, (end - start) as usize)]);
    true
}

/// [`skip_back`]'s note in the book of offsets `run` of `shadow`, a piece
/// at 64 KiB `key` that may be left: whether the piece is new there. `Err`:
/// it is not left after all, and for mode 3 why ([`REFUSALS`]).
fn note(
    book: &mut Book,
    shadow: &Arc<Shadow>,
    key: u64,
    run: (usize, usize),
    page: Option<(usize, u64)>,
    aliases: &[u64],
    targets: &[(Arc<Shadow>, u64)],
) -> Result<bool, Option<usize>> {
    // Taken once while left; or another copy left its bytes here, and this
    // one is a second holder after all: fetched as before.
    let other = book
        .left
        .get(&key)
        .is_some_and(|left| left.writer.id() != shadow.id() || left.stranded);
    if other || book.never.contains(&key) {
        return Err(None);
    }
    // A piece of this copy's that the title's change of a mapping
    // stranded is in the card's side at these offsets, perhaps.
    if book.stranded != 0
        && book
            .left
            .values()
            .any(|left| left.stranded && left.writer.id() == shadow.id())
    {
        return Err(Some(CHANGED));
    }
    if !aliases.is_empty() {
        // The pages are another copy's piece already, left through
        // another of their addresses: two copies' writes of one piece are
        // put in order in title memory only.
        let Some(page) = page else {
            return Err(Some(VIEWS));
        };
        if book.by_page.get(&page).is_some_and(|&k| k != key)
            || aliases.iter().any(|a| book.left.contains_key(a))
            || book.alias.get(&key).is_some_and(|&k| k != key)
        {
            return Err(Some(OTHER));
        }
    }
    let mut holders: Vec<(u64, u64)> = targets
        .iter()
        .map(|(copy, at)| (copy.id(), at / PIECE * PIECE))
        .collect();
    holders.sort_unstable();
    match book.left.get_mut(&key) {
        Some(left) => {
            // The same copies as when it was first left: one more or one
            // fewer, and what each holds is no longer known alike.
            if left.held() != holders || left.aliases != aliases {
                return Err(Some(CHANGED));
            }
            add_run(&mut left.runs, run.0, run.1);
            STATS[5].fetch_add(1, Ordering::Relaxed);
            Ok(false)
        }
        None => {
            let page = page.filter(|page| !book.by_page.contains_key(page));
            book.insert(
                key,
                Left {
                    writer: shadow.clone(),
                    base: shadow.span().0,
                    runs: vec![run],
                    page,
                    aliases: aliases.to_vec(),
                    holders: targets
                        .iter()
                        .map(|(copy, at)| Held {
                            copy: Arc::downgrade(copy),
                            id: copy.id(),
                            at: at / PIECE * PIECE,
                        })
                        .collect(),
                    stranded: false,
                },
            );
            Ok(true)
        }
    }
}

/// What becomes of title memory `[start, end)` (in one 64 KiB), which
/// `writer` holds in the card's memory and whose shaders wrote it (see the
/// file).
fn judge(writer: &Arc<Shadow>, start: u64, end: u64) -> Verdict {
    if !writer.plain_copy()
        || !write_combining(start, end)
        || crate::inplace::level() != 0
        || crate::port::wanted(start, end - start)
        || crate::memory::holds(start, end)
    {
        return Verdict::Fetch;
    }
    let key = start / PIECE * PIECE;
    let views = crate::inplace::pages_of(key as usize);
    let same = !crate::shadow::holders(start, end, writer.id()).is_empty();
    // Shown at one address only.
    if views.len() <= 1 {
        return if same {
            Verdict::Fetch
        } else {
            Verdict::Leave(Vec::new(), Vec::new())
        };
    }
    if !forwards() {
        return Verdict::Fetch;
    }
    if same {
        return Verdict::Refuse(SAME);
    }
    // A write of the title's through one view is told to the copies at
    // the others only where the views are guarded together.
    if views.len() > crate::watch::MANY_VIEWS || !crate::watch::views_guarded() {
        return Verdict::Refuse(VIEWS);
    }
    let (within, bytes) = (start - key, end - start);
    let mut aliases: Vec<u64> = views
        .iter()
        .map(|&page| page as u64)
        .filter(|&page| page != key)
        .collect();
    aliases.sort_unstable();
    aliases.dedup();
    let mut targets: Vec<(Arc<Shadow>, u64)> = Vec::new();
    let (mine, my_end) = (writer.span().0, writer.span().0 + writer.span().1);
    for &alias in &aliases {
        let at = alias + within;
        if !any_alias() && !write_combining(at, at + bytes) {
            return Verdict::Refuse(PLAIN_ALIAS);
        }
        if crate::memory::holds(at, at + bytes) || crate::port::wanted(at, bytes) {
            return Verdict::Refuse(READ_THERE);
        }
        // The writer itself at the other address: its card side holds the
        // bytes twice, and nothing brings one to the other.
        if mine < at + bytes && my_end > at {
            return Verdict::Refuse(KIND);
        }
        let holders = crate::shadow::holders(at, at + bytes, writer.id());
        if census() {
            // The check of the holders found: every copy of every table,
            // walked whole.
            let all = crate::shadow::holders_all(at, at + bytes, writer.id());
            let missed = all
                .iter()
                .filter(|id| !holders.iter().any(|h| h.id() == **id))
                .count();
            HOLDERS_CHECKED[0].fetch_add(1, Ordering::Relaxed);
            HOLDERS_CHECKED[1].fetch_add(all.len() as u64, Ordering::Relaxed);
            HOLDERS_CHECKED[2].fetch_add(missed as u64, Ordering::Relaxed);
        }
        for holder in holders {
            if targets
                .iter()
                .any(|(copy, to)| Arc::ptr_eq(copy, &holder) && *to == at)
            {
                continue;
            }
            if !holder.carded() {
                return Verdict::Refuse(IN_PC);
            }
            if !holder.plain_copy() || holder.same_buffer(writer) {
                return Verdict::Refuse(KIND);
            }
            let (theirs, len) = holder.span();
            if theirs > at || theirs + len < at + bytes {
                return Verdict::Refuse(PART);
            }
            // Its own shaders may be writing the piece now: the copy on
            // the card would run behind them and put older bytes over
            // theirs.
            if holder.writes_pending_in(at as usize, (at + bytes) as usize) {
                return Verdict::Refuse(WRITTEN);
            }
            targets.push((holder, at));
        }
    }
    Verdict::Leave(aliases, targets)
}

/// `[lo, hi)` of `shadow` at 64 KiB `key` is fetched as before: what of it
/// was left comes back with it.
fn taken_back(shadow: &Arc<Shadow>, key: u64, lo: usize, hi: usize) {
    if LEFT_N.load(Ordering::SeqCst) == 0 {
        return;
    }
    let mut book = BOOK.write().unwrap_or_else(|e| e.into_inner());
    let Some(left) = book
        .left
        .get_mut(&key)
        .filter(|left| left.writer.id() == shadow.id() && !left.stranded)
    else {
        return;
    };
    cut_run(&mut left.runs, lo, hi);
    if left.runs.is_empty() {
        book.remove(key);
    }
}

/// Whether anything is left or being fetched: a reader must look.
fn any() -> bool {
    LEFT_N.load(Ordering::SeqCst) != 0 || FETCHING.load(Ordering::SeqCst) != 0
}

/// Before title bytes `[from, to)` are read for copy `taker` (0: for no
/// copy): the pieces left there by any other copy are fetched, and what
/// their card wrote merged into title memory; they are not left again.
/// Waits for a fetch of another thread under way.
pub(crate) fn before_take(from: usize, to: usize, taker: u64) {
    if !any() || from >= to {
        return;
    }
    let _gate = gate("a reader's fetch", taker);
    fetch_in(from as u64, to as u64, For::Reader(taker));
}

/// Before our own work that may read or write title memory anywhere:
/// every piece left is fetched.
pub(crate) fn before_anything() {
    before_take(0, usize::MAX, 0);
}

/// A piece was fetched for a copy whose shaders are about to write
/// ([`declaring`]): title memory holds it now, and the copy takes it from
/// there before it goes on.
#[derive(Debug)]
pub(crate) struct Fetched;

/// What [`declaring`] holds until the declared range is noted.
pub(crate) struct Declaring(#[allow(dead_code)] std::sync::MutexGuard<'static, ()>);

/// Mode 3, as shader writes of copy `id` over title memory `[from, to)`
/// are declared (`Shadow::writing`): while what this returns is held, no
/// landing makes the copy a holder of another copy's piece. If it is one
/// there now (or has a stranded piece of its own), those pieces are
/// fetched into title memory, and never left again: `Err`, and the caller
/// refreshes the copy and asks again.
///
/// Why: a holder's card side was given the piece by a copy on the card,
/// and a later one may come. Shader writes of the holder's own, put in
/// between, would be written over by the next; and landed into title
/// memory, they would be taken by the writer's refresh for bytes the title
/// stored, and sent up over newer writes of the writer's own shaders.
pub(crate) fn declaring(id: u64, from: usize, to: usize) -> Result<Declaring, Fetched> {
    let held = DECIDE.lock().unwrap_or_else(|e| e.into_inner());
    if LEFT_N.load(Ordering::SeqCst) == 0 || from >= to {
        return Ok(Declaring(held));
    }
    let (from, to) = (from as u64, to as u64);
    let holds = {
        let book = BOOK.read().unwrap_or_else(|e| e.into_inner());
        // (A dispatch declares thousands of ranges: most copies are no
        // holder of anything.)
        book.may_hold(id) && !book.wanted(For::Writing(id), from, to, &[]).is_empty()
    };
    if !holds {
        return Ok(Declaring(held));
    }
    // (Not held across the fetch: a landing waits for it.)
    drop(held);
    let _gate = gate("a fetch before a copy's shaders write", id);
    fetch_in(from, to, For::Writing(id));
    Err(Fetched)
}

/// The fetch of [`before_take`], [`declaring`] and [`sweep`], under
/// [`GATE`]: the pieces [`Book::wanted`] names.
fn fetch_in(from: u64, to: u64, who: For) {
    if LEFT_N.load(Ordering::SeqCst) == 0 {
        return;
    }
    // (The views first: no lock of the book is held while they are read.)
    let any_pages = !BOOK
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .by_page
        .is_empty();
    let pages = if any_pages && to - from <= 1 << 30 {
        crate::inplace::page_keys(from as usize, (to - from) as usize)
    } else {
        Vec::new()
    };
    let mut pieces: Vec<Piece> = Vec::new();
    // Of the pieces left with other addresses of their pages known: the
    // title memory each holds.
    let mut beside: Vec<(u64, u64)> = Vec::new();
    {
        let mut book = BOOK.write().unwrap_or_else(|e| e.into_inner());
        let keys = book.wanted(who, from, to, &pages);
        if keys.is_empty() {
            return;
        }
        FETCHING.fetch_add(1, Ordering::SeqCst);
        for key in keys {
            book.never.insert(key);
            if let Some(left) = book.remove(key) {
                // (At each address the pages were known by.)
                book.never.extend(left.aliases.iter().copied());
                if !left.aliases.is_empty() || left.stranded {
                    beside.push(title_span(&left));
                }
                pieces.extend(left.runs.iter().map(|&(lo, hi)| Piece {
                    copy: left.writer.clone(),
                    lo,
                    hi,
                    base: left.base,
                }));
            }
        }
    }
    STATS[2].fetch_add(pieces.len() as u64, Ordering::Relaxed);
    STATS[3].fetch_add(
        pieces.iter().map(|p| (p.hi - p.lo) as u64).sum(),
        Ordering::Relaxed,
    );
    STATS[4].fetch_add(1, Ordering::Relaxed);
    match crate::host::host() {
        Some(host) => crate::host::leftq::fetch(host, &pieces),
        None => crate::host::fail(format_args!(
            "pieces left on the card and no device to fetch them (VICTORY_SKIP_WC_BACK)"
        )),
    }
    // The copies that held these pieces by a copy on the card hold them
    // for their own no longer: what is in title memory now is theirs to
    // take, whether the merge changed a byte there or not (their CPU side
    // is the reference of bytes the card got, not what title memory was).
    for (start, end) in beside {
        let pages: Vec<usize> = (start as usize / 4096..(end as usize).div_ceil(4096)).collect();
        let _ = crate::watch::landed_beside(&pages, &[(start as usize, (end - start) as usize)]);
    }
    FETCHING.fetch_sub(1, Ordering::SeqCst);
}

/// Pieces whose copy nothing but this book holds any more (the title let
/// its buffer go, a window was let go): fetched now, when nothing is in the
/// air (`crate::host::sweep`), so that their bytes are not lost with the
/// copy.
pub(crate) fn sweep() {
    if LEFT_N.load(Ordering::SeqCst) == 0 {
        return;
    }
    let _gate = gate("the sweep's fetch", 0);
    let orphans: Vec<u64> = {
        let book = BOOK.read().unwrap_or_else(|e| e.into_inner());
        let mut held: BTreeMap<u64, (usize, usize)> = BTreeMap::new();
        for left in book.left.values() {
            let entry = held
                .entry(left.writer.id())
                .or_insert((Arc::strong_count(&left.writer), 0));
            entry.1 += 1;
        }
        held.into_iter()
            .filter(|(_, (count, here))| count <= here)
            .map(|(id, _)| id)
            .collect()
    };
    for id in orphans {
        fetch_in(0, u64::MAX, For::Orphan(id));
    }
}

/// Title memory `[at, at + len)` is freed, unmapped or shows other pages:
/// what was left there is no longer the title's (forgotten, not fetched).
/// A length of 0: the whole reservation at `at`.
///
/// But a piece left with other addresses of its pages known (mode 3),
/// while the title still shows the pages at one of them: the bytes are the
/// title's still, there. It is kept at that address, stranded: no copy
/// holds it for its own any more, so whoever reads there, its former
/// holders too, has it fetched first, and so has the copy whose card side
/// it is in before that copy is refreshed or written again.
pub(crate) fn forget(at: usize, len: usize) {
    if !on() {
        return;
    }
    let len = if len != 0 {
        len
    } else {
        let mut map = WRITE_COMBINING.write().unwrap_or_else(|e| e.into_inner());
        match map.remove(&at) {
            Some(len) => len,
            None => return,
        }
    };
    let (from, to) = (at as u64, at.saturating_add(len) as u64);
    if LEFT_N.load(Ordering::SeqCst) == 0
        && BOOK
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .never
            .is_empty()
    {
        return;
    }
    // (The views with no lock of the book held.) Of the pieces there with
    // other addresses known: the first of those outside the memory that
    // goes which still shows the piece's page.
    let known: Vec<(u64, (usize, u64), Vec<u64>)> = {
        let book = BOOK.read().unwrap_or_else(|e| e.into_inner());
        if book.alias.is_empty() {
            Vec::new()
        } else {
            book.left
                .range(from / PIECE * PIECE..to)
                .filter(|(_, left)| !left.aliases.is_empty())
                .filter_map(|(&key, left)| Some((key, left.page?, left.aliases.clone())))
                .collect()
        }
    };
    let kept: Vec<(u64, u64)> = known
        .into_iter()
        .filter_map(|(key, page, aliases)| {
            let alias = aliases.into_iter().find(|&alias| {
                (alias < from / PIECE * PIECE || alias >= to)
                    && crate::inplace::page_keys(alias as usize, 1).first()
                        == Some(&(alias as usize, page))
            })?;
            Some((key, alias))
        })
        .collect();
    let mut book = BOOK.write().unwrap_or_else(|e| e.into_inner());
    // The other addresses of pieces' pages that go: nothing of the piece
    // is there any more, and the holders there hold it no longer.
    let gone: Vec<u64> = book
        .alias
        .range(from / PIECE * PIECE..to)
        .map(|(&alias, _)| alias)
        .collect();
    let mut again = Vec::new();
    for alias in gone {
        again.extend(book.unalias(alias));
    }
    let keys: Vec<u64> = book
        .left
        .range(from / PIECE * PIECE..to)
        .map(|(&key, _)| key)
        .collect();
    for key in keys {
        let alias = kept.iter().find(|k| k.0 == key).map(|k| k.1);
        if alias.is_some_and(|alias| book.strand(key, alias)) {
            FORWARD_STATS[9].fetch_add(1, Ordering::Relaxed);
        } else if book.remove(key).is_some() {
            FORGOTTEN.fetch_add(1, Ordering::Relaxed);
        }
    }
    let never: Vec<u64> = book
        .never
        .range(from / PIECE * PIECE..to)
        .copied()
        .collect();
    for key in never {
        book.never.remove(&key);
    }
    drop(book);
    // Their card side there may hold the card's bytes of pages the address
    // no longer shows: it gets the CPU side's again, whole, and from there
    // on what the title has there, as any copy.
    for (copy, lo, hi) in again {
        copy.resend(lo, hi);
    }
}

/// Pieces forgotten as their memory went ([`forget`]), since the start.
static FORGOTTEN: AtomicU64 = AtomicU64::new(0);

/// Whether copy `id` has the card's bytes of a piece left in title memory
/// `[from, to)`, its own shaders' or, as a holder, another copy's given it
/// on the card: its refreshes there take only what the title changed
/// (`Shadow::take`). With `holding` false: only as the writer.
fn left_in(id: u64, from: usize, to: usize, holding: bool) -> bool {
    if LEFT_N.load(Ordering::SeqCst) == 0 || from >= to {
        return false;
    }
    let (from, to) = (from as u64, to as u64);
    let book = BOOK.read().unwrap_or_else(|e| e.into_inner());
    let own = book.left.range(from / PIECE * PIECE..to).any(|(_, left)| {
        left.writer.id() == id && !left.stranded && {
            let (a, b) = title_span(left);
            a < to && b > from
        }
    });
    own || (holding
        && !book.alias.is_empty()
        && book
            .alias
            .range(from / PIECE * PIECE..to)
            .any(|(&alias, &key)| {
                book.left.get(&key).is_some_and(|left| {
                    let (a, b) = title_span(left);
                    let (a, b) = (a - key + alias, b - key + alias);
                    a < to
                        && b > from
                        && !left.stranded
                        && left.holders.iter().any(|held| {
                            held.id == id && held.at == alias && held.copy.strong_count() > 0
                        })
                })
            }))
}

/// Whether copy `id` holds a piece left in title memory `[from, to)`, as
/// its writer or as a holder: its refreshes there take only what the title
/// changed (`Shadow::take`).
pub(crate) fn own_left_in(id: u64, from: usize, to: usize) -> bool {
    left_in(id, from, to, true)
}

/// What a refresh of copy `id` over title memory `[from, to)` holds while
/// it runs ([`before_refresh`]).
pub(crate) struct Refreshing {
    _gate: Option<Gated<'static>>,
    _taking: Taking,
}

/// Before a refresh of copy `id` takes title memory `[from, to)`: other
/// copies' pieces there are fetched (but those the copy holds itself, by a
/// copy on the card); if the copy holds pieces its own shaders wrote
/// there, no fetch runs until the refresh is done (it writes the CPU side
/// a fetch merges into; a fetch does not write a holder's).
pub(crate) fn before_refresh(id: u64, from: usize, to: usize) -> Refreshing {
    let taking = Taking::new(id);
    if !any() || from >= to {
        return Refreshing {
            _gate: None,
            _taking: taking,
        };
    }
    let gate = gate("a refresh", id);
    fetch_in(from as u64, to as u64, For::Reader(id));
    let own = left_in(id, from, to, false);
    Refreshing {
        _gate: own.then_some(gate),
        _taking: taking,
    }
}

/// Mode 3, on the submitting thread before a submission's sends to card
/// copies are taken (`Host::card_before`; no lock of the queue's order is
/// held yet): the copies on the card asked for since the last. Each whose
/// piece still stands in the book has its holder's CPU side brought in
/// line with the writer's (`Shadow::forwarded_from`), and is returned, for
/// the caller to record ahead of the sends and of the submission's lists.
/// Those whose piece was fetched, forgotten or taken back meanwhile are
/// dropped: title memory holds those bytes, or nobody's are they.
///
/// Under [`GATE`], which keeps fetches out (they merge into the writer's
/// CPU side, read here). It waits for no card, but for a fetch under way
/// of another thread, as any reader; that wait is said in the log like any
/// for the gate. (But where a copy stopped being a plain copy since: its
/// piece is fetched here, as a reader's, with that fetch's wait and words.)
pub(crate) fn forward_step() -> Vec<Forward> {
    if FORWARDS_N.load(Ordering::SeqCst) == 0 {
        return Vec::new();
    }
    // (Never so: the sends are taken outside every refresh. The copies
    // wait for the next submission rather than stop the process.)
    if gate_held() {
        return Vec::new();
    }
    let began = std::time::Instant::now();
    let _gate = gate("copies on the card to other addresses", 0);
    let mut jobs = {
        let mut forwards = FORWARDS.lock().unwrap_or_else(|e| e.into_inner());
        FORWARDS_N.store(0, Ordering::SeqCst);
        std::mem::take(&mut *forwards)
    };
    // (A piece left again before the next submission asks twice.)
    let same = |a: &Forward, b: &Forward| {
        Arc::ptr_eq(&a.from, &b.from)
            && Arc::ptr_eq(&a.to, &b.to)
            && (a.lo, a.at, a.bytes) == (b.lo, b.at, b.bytes)
    };
    let mut out: Vec<Forward> = Vec::with_capacity(jobs.len());
    // Pieces a copy of which is no plain copy any more.
    let mut fetch: Vec<(u64, u64)> = Vec::new();
    for job in jobs.drain(..) {
        if out.iter().any(|done| same(done, &job)) {
            continue;
        }
        // Shaders store in one of them what is not declared, since the
        // piece was left (`Shadow::keep_stores`): nothing is put over
        // such a copy's card side, and the piece is fetched, as any reader
        // has it fetched.
        if !job.to.plain_copy() || !job.from.plain_copy() {
            let at = job.from.span().0 + job.lo as u64;
            fetch.push((at, at + job.bytes as u64));
            FORWARD_STATS[6].fetch_add(1, Ordering::Relaxed);
            continue;
        }
        let stands = BOOK.read().unwrap_or_else(|e| e.into_inner()).stands(&job);
        let _taking = Taking::new(job.to.id());
        let Some(again) = stands
            .then(|| job.to.forwarded_from(&job.from, job.lo, job.at, job.bytes))
            .flatten()
        else {
            FORWARD_STATS[6].fetch_add(1, Ordering::Relaxed);
            continue;
        };
        FORWARD_STATS[3].fetch_add(1, Ordering::Relaxed);
        FORWARD_STATS[4].fetch_add(job.bytes as u64, Ordering::Relaxed);
        FORWARD_STATS[8].fetch_add(again, Ordering::Relaxed);
        out.push(job);
    }
    if !out.is_empty() {
        FORWARD_STATS[5].fetch_add(1, Ordering::Relaxed);
    }
    for (from, to) in fetch {
        fetch_in(from, to, For::Reader(0));
    }
    FORWARD_STATS[7].fetch_add(began.elapsed().as_micros() as u64, Ordering::Relaxed);
    out
}

/// For the audit ([`census`]): title memory `[at, at + len)` is read.
pub(crate) fn read(at: usize, len: usize) {
    if !census() || len == 0 || LEFT_N.load(Ordering::Relaxed) == 0 {
        return;
    }
    let (from, to) = (at as u64, at.saturating_add(len) as u64);
    let taker = TAKER.get();
    let Ok(book) = BOOK.try_read() else {
        return;
    };
    let (stale, elsewhere) = stale_in(&book, from, to, taker);
    if stale > 0 {
        STALE[usize::from(taker == 0)].fetch_add(stale, Ordering::Relaxed);
    }
    if elsewhere > 0 {
        STALE[2].fetch_add(elsewhere, Ordering::Relaxed);
    }
}

/// [`read`]'s count: the title bytes of `[from, to)` that pieces left
/// hold and that copy `taker` (0: no copy) has not itself, in all; and of
/// them those read at another address of the pages than the writer's (the
/// addresses known when the piece was left, where its holders are).
fn stale_in(book: &Book, from: u64, to: u64, taker: u64) -> (u64, u64) {
    let overlap = |left: &Left, shift: u64| -> u64 {
        left.runs
            .iter()
            .map(|&(lo, hi)| {
                let a = left
                    .base
                    .wrapping_add(lo as u64)
                    .wrapping_add(shift)
                    .max(from);
                let b = left
                    .base
                    .wrapping_add(hi as u64)
                    .wrapping_add(shift)
                    .min(to);
                b.saturating_sub(a)
            })
            .sum()
    };
    let mut stale = 0;
    for (_, left) in book.left.range(from / PIECE * PIECE..to) {
        if !left.kept_by(taker) {
            stale += overlap(left, 0);
        }
    }
    let mut elsewhere = 0;
    for (&alias, &key) in book.alias.range(from / PIECE * PIECE..to) {
        if let Some(left) = book.left.get(&key)
            && !left.kept_by(taker)
        {
            elsewhere += overlap(left, alias.wrapping_sub(key));
        }
    }
    (stale + elsewhere, elsewhere)
}

/// For the frame line, with a mode on: per frame, what was left on the card
/// and what was fetched later; with mode 3 the copies on the card and why
/// pieces were not left; with [`census`], the audit.
pub(crate) fn log(frames: f64) {
    if !on() {
        return;
    }
    let [left, left_bytes, fetched, fetched_bytes, fetches, again] =
        STATS.each_ref().map(|s| s.swap(0, Ordering::Relaxed));
    tracing::info!(
        "copy-back left on the card (VICTORY_SKIP_WC_BACK={}): per frame {:.1} MB in {:.0} pieces ({:.0} of them left again), \
         {:.1} MB in {:.0} pieces fetched later for another reader ({:.1} fetches); {} pieces left now, {} forgotten so far \
         as their memory went",
        mode(),
        left_bytes as f64 / frames / 1e6,
        left as f64 / frames,
        again as f64 / frames,
        fetched_bytes as f64 / frames / 1e6,
        fetched as f64 / frames,
        fetches as f64 / frames,
        LEFT_N.load(Ordering::Relaxed),
        FORGOTTEN.load(Ordering::Relaxed)
    );
    if forwards() {
        let stats = FORWARD_STATS
            .each_ref()
            .map(|s| s.swap(0, Ordering::Relaxed));
        let refused = REFUSED.each_ref().map(|s| s.swap(0, Ordering::Relaxed));
        tracing::info!(
            "{}",
            forwards_line(frames, &stats, FORWARDS_N.load(Ordering::Relaxed), &refused)
        );
    }
    if census() {
        let [refreshes, others, elsewhere] = STALE
            .each_ref()
            .map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames / 1e6);
        tracing::info!(
            "census audit of the copy-back left on the card: stale title bytes read per frame: {refreshes:.2} MB by refreshes of \
             other copies, {others:.2} MB by other readers (both must be 0); of them {elsewhere:.2} MB at another address of the \
             pages, where copies on the card go (must be 0)"
        );
        if forwards() {
            let [pieces, holders, missed] = HOLDERS_CHECKED
                .each_ref()
                .map(|s| s.swap(0, Ordering::Relaxed));
            tracing::info!(
                "census audit of the copies on the card (VICTORY_SKIP_WC_BACK=3): the holders at {pieces} other addresses of pieces' \
                 pages checked against a walk of every table: {holders} holders there, {missed} of them missed by the quick search \
                 (must be 0)"
            );
        }
    }
}

/// The frame line of mode 3: `stats` as [`FORWARD_STATS`], `waiting` the
/// copies asked for and not made yet, `refused` as [`REFUSED`].
fn forwards_line(
    frames: f64,
    stats: &[u64; 10],
    waiting: u64,
    refused: &[u64; REFUSALS.len()],
) -> String {
    let [
        pieces,
        bytes,
        asked,
        made,
        made_bytes,
        lists,
        dropped,
        micros,
        again,
        stranded,
    ] = *stats;
    let reasons: Vec<String> = REFUSALS
        .iter()
        .zip(refused)
        .filter(|(_, n)| **n != 0)
        .map(|(why, n)| format!("{:.1} for {why}", *n as f64 / frames))
        .collect();
    format!(
        "copies on the card to other addresses of the pages (VICTORY_SKIP_WC_BACK=3): per frame {:.1} MB forwarded in {:.0} pieces \
         ({:.0} copies asked for; {:.0} made, {:.1} MB, in {:.1} lists; {:.1} dropped as their piece was fetched first), {:.2} ms \
         bringing the holders' CPU sides in line, {:.1} KB sent up again behind the copies; {waiting} copies in the air \
         (asked for, made ahead of the next submission), {stranded} pieces stranded by a change of mapping; pieces not left \
         (fetched as before), per frame: {}",
        bytes as f64 / frames / 1e6,
        pieces as f64 / frames,
        asked as f64 / frames,
        made as f64 / frames,
        made_bytes as f64 / frames / 1e6,
        lists as f64 / frames,
        dropped as f64 / frames,
        micros as f64 / frames / 1e3,
        again as f64 / frames / 1e3,
        if reasons.is_empty() {
            "none".to_owned()
        } else {
            reasons.join(", ")
        }
    )
}

#[cfg(test)]
mod tests {
    use super::{GATE_WAITS, Gate, Holder, STEP, add_run, cut_run, gate_held, gate_waited};

    /// The cycle that hung the process, on one thread: a refresh holds the
    /// gate and a job of another's that runs on its thread asks for it. It
    /// is refused at once, not waited for; another thread waits, and is let
    /// in when the refresh is done.
    #[test]
    fn the_gate_is_not_waited_for_by_the_thread_that_holds_it() {
        let gate = Gate::new();
        assert!(!gate_held());
        let refresh = gate.enter("a refresh", 1, STEP).expect("nobody holds it");
        assert!(gate_held());
        assert!(
            gate.enter("a reader's fetch", 0, STEP).is_err(),
            "asked again by the thread that holds it"
        );
        // (Nor another gate: what the thread may not do meanwhile is the same.)
        assert!(Gate::new().enter("a refresh", 2, STEP).is_err());
        std::thread::scope(|threads| {
            let other = threads.spawn(|| {
                assert!(!gate_held());
                let got = gate.enter("a reader's fetch", 0, STEP).is_ok();
                // (Let go of when the statement above ended.)
                got && !gate_held()
            });
            std::thread::sleep(std::time::Duration::from_millis(50));
            assert!(!other.is_finished(), "let in while the gate was held");
            assert!(gate_held());
            drop(refresh);
            assert!(other.join().expect("no panic"));
        });
        assert!(!gate_held());
        // Free again, for the thread that held it too.
        let again = gate.enter("a refresh", 1, STEP).expect("let go of");
        assert!(gate_held());
        drop(again);
        assert!(!gate_held());
    }

    /// A thread that has waited a step for the gate says so, with what
    /// holds it, and goes on waiting: it is let in when the gate is free.
    #[test]
    fn a_long_wait_for_the_gate_is_said_and_goes_on() {
        use std::sync::atomic::Ordering;
        use std::time::Duration;
        let gate = Gate::new();
        let said = GATE_WAITS.load(Ordering::Relaxed);
        let held = gate.enter("a refresh", 7, STEP).expect("nobody holds it");
        std::thread::scope(|threads| {
            let other = threads.spawn(|| {
                gate.enter("a reader's fetch", 0, Duration::from_millis(10))
                    .is_ok()
            });
            std::thread::sleep(Duration::from_millis(200));
            assert!(!other.is_finished(), "let in while the gate was held");
            drop(held);
            assert!(other.join().expect("no panic"));
        });
        assert!(GATE_WAITS.load(Ordering::Relaxed) > said);
        let holder = Holder {
            thread: std::thread::current(),
            what: "a refresh",
            copy: 7,
            since: std::time::Instant::now(),
        };
        let line = gate_waited("a reader's fetch", 0, Duration::from_secs(4), &holder);
        assert!(
            line.contains("a reader's fetch (copy 0) has waited 4 s"),
            "{line}"
        );
        assert!(line.contains("holds for a refresh (copy 7)"), "{line}");
    }

    /// One thread at a time is inside, however many ask.
    #[test]
    fn the_gate_holds_one_thread_at_a_time() {
        use std::sync::atomic::{AtomicU64, Ordering};
        let gate = Gate::new();
        let (inside, passes) = (AtomicU64::new(0), AtomicU64::new(0));
        std::thread::scope(|threads| {
            for _ in 0..6 {
                threads.spawn(|| {
                    for _ in 0..500 {
                        let held = gate
                            .enter("a refresh", 3, STEP)
                            .expect("not held by this thread");
                        assert_eq!(
                            inside.fetch_add(1, Ordering::SeqCst),
                            0,
                            "two threads inside"
                        );
                        std::thread::yield_now();
                        inside.fetch_sub(1, Ordering::SeqCst);
                        passes.fetch_add(1, Ordering::Relaxed);
                        drop(held);
                    }
                });
            }
        });
        assert_eq!(passes.load(Ordering::Relaxed), 3000);
        assert!(gate.held.lock().unwrap().is_none());
    }

    #[test]
    fn runs_join_and_cut() {
        let mut runs = Vec::new();
        add_run(&mut runs, 100, 200);
        add_run(&mut runs, 300, 400);
        add_run(&mut runs, 150, 300);
        assert_eq!(runs, vec![(100, 400)]);
        add_run(&mut runs, 500, 500);
        assert_eq!(runs, vec![(100, 400)]);
        cut_run(&mut runs, 200, 250);
        assert_eq!(runs, vec![(100, 200), (250, 400)]);
        cut_run(&mut runs, 0, 120);
        assert_eq!(runs, vec![(120, 200), (250, 400)]);
        cut_run(&mut runs, 180, 1000);
        assert_eq!(runs, vec![(120, 180)]);
        cut_run(&mut runs, 0, 1000);
        assert!(runs.is_empty());
    }

    use super::{
        Book, CHANGED, For, Forward, Held, Left, OTHER, PIECE, REFUSALS, Shadow, forwards_line,
        note, stale_in, title_span,
    };
    use std::sync::{Arc, Weak};

    /// A card copy of `len` bytes of title memory at `start`, for the
    /// book's tests: no buffer behind it, and never let go.
    fn copy(start: u64, len: usize) -> Arc<Shadow> {
        let data = Box::leak(vec![0u8; len].into_boxed_slice());
        Shadow::unbacked(start, data.as_mut_ptr(), len as u64, true)
    }

    /// A piece of `writer` at 64 KiB `key`, offsets `run`, whose page the
    /// title shows at `alias` too, where `holder` is given the bytes.
    fn piece(writer: &Arc<Shadow>, run: (usize, usize), alias: u64, holder: &Arc<Shadow>) -> Left {
        Left {
            writer: writer.clone(),
            base: writer.span().0,
            runs: vec![run],
            page: Some((7, 0x30000)),
            aliases: vec![alias],
            holders: vec![Held {
                copy: Arc::downgrade(holder),
                id: holder.id(),
                at: alias,
            }],
            stranded: false,
        }
    }

    const KEY: u64 = 0x4000_0000;
    const ALIAS: u64 = 0x7000_0000;

    /// The writer and the holder read title memory over the piece without
    /// a fetch, at their own addresses; every other reader, at either
    /// address, has it fetched; a holder whose shaders are about to write
    /// has it fetched too, the writer not.
    #[test]
    fn a_holder_reads_without_a_fetch_and_every_other_reader_fetches() {
        let writer = copy(KEY - 0x1000, 0x40000);
        let holder = copy(ALIAS, 0x20000);
        let other = copy(ALIAS, 0x20000);
        let mut book = Book::new(false);
        // The piece: title bytes [KEY + 0x100, KEY + 0x900), the writer's
        // offsets 0x1100..0x1900.
        book.insert(KEY, piece(&writer, (0x1100, 0x1900), ALIAS, &holder));
        let (here, there) = ((KEY, KEY + PIECE), (ALIAS, ALIAS + PIECE));
        assert!(
            book.wanted(For::Reader(writer.id()), here.0, here.1, &[])
                .is_empty()
        );
        assert!(
            book.wanted(For::Reader(holder.id()), there.0, there.1, &[])
                .is_empty()
        );
        for reader in [0, other.id()] {
            assert_eq!(
                book.wanted(For::Reader(reader), here.0, here.1, &[]),
                vec![KEY],
                "reader {reader}"
            );
            assert_eq!(
                book.wanted(For::Reader(reader), there.0, there.1, &[]),
                vec![KEY],
                "reader {reader}"
            );
        }
        // Beside the piece's bytes at the other address: nothing.
        assert!(
            book.wanted(For::Reader(0), ALIAS + 0x900, ALIAS + PIECE, &[])
                .is_empty()
        );
        assert_eq!(
            book.wanted(For::Reader(0), ALIAS + 0x8ff, ALIAS + PIECE, &[]),
            vec![KEY]
        );
        // By the page, at an address not known when the piece was left.
        let later = [(0x9000_0000usize, (7usize, 0x30000u64))];
        assert_eq!(
            book.wanted(For::Reader(0), 0x9000_0000, 0x9001_0000, &later),
            vec![KEY]
        );
        // Declared writes.
        assert_eq!(
            book.wanted(For::Writing(holder.id()), there.0, there.1, &[]),
            vec![KEY]
        );
        assert!(
            book.wanted(For::Writing(writer.id()), here.0, here.1, &[])
                .is_empty()
        );
        assert!(
            book.wanted(For::Writing(other.id()), there.0, there.1, &[])
                .is_empty()
        );
        // The copy let go of by everything but the book.
        assert_eq!(
            book.wanted(For::Orphan(writer.id()), 0, u64::MAX, &[]),
            vec![KEY]
        );
        assert!(
            book.wanted(For::Orphan(holder.id()), 0, u64::MAX, &[])
                .is_empty()
        );
        // A holder that is gone holds nothing: a copy made anew with its
        // number would read stale bytes.
        book.left.get_mut(&KEY).unwrap().holders[0].copy = Weak::new();
        assert_eq!(
            book.wanted(For::Reader(holder.id()), there.0, there.1, &[]),
            vec![KEY]
        );
        assert!(book.may_hold(holder.id()) && !book.may_hold(writer.id()));
        assert!(book.remove(KEY).is_some());
        assert!(book.alias.is_empty() && book.by_page.is_empty() && book.left.is_empty());
        assert!(book.holding.is_empty());
    }

    /// A copy on the card asked for stands only while its piece is in the
    /// book as it was: left by the same copy, holding the offsets, with the
    /// same holder.
    #[test]
    fn a_copy_on_the_card_stands_only_while_its_piece_does() {
        let writer = copy(KEY, 0x40000);
        let holder = copy(ALIAS, 0x20000);
        let stranger = copy(0x5000_0000, 0x20000);
        let job = |from: &Arc<Shadow>, to: &Arc<Shadow>, lo: usize, bytes: usize| Forward {
            from: from.clone(),
            lo,
            to: to.clone(),
            at: lo,
            bytes,
            key: KEY,
        };
        let mut book = Book::new(false);
        assert!(
            !book.stands(&job(&writer, &holder, 0x100, 0x800)),
            "nothing is left"
        );
        book.insert(KEY, piece(&writer, (0x100, 0x900), ALIAS, &holder));
        assert!(book.stands(&job(&writer, &holder, 0x100, 0x800)));
        assert!(
            book.stands(&job(&writer, &holder, 0x200, 0x100)),
            "part of the run"
        );
        assert!(
            !book.stands(&job(&writer, &holder, 0x100, 0x801)),
            "past the run"
        );
        assert!(
            !book.stands(&job(&writer, &stranger, 0x100, 0x800)),
            "not a holder"
        );
        assert!(
            !book.stands(&job(&stranger, &holder, 0x100, 0x800)),
            "not the writer"
        );
        // Part of it was fetched as before by a later landing.
        super::cut_run(&mut book.left.get_mut(&KEY).unwrap().runs, 0x400, 0x500);
        assert!(!book.stands(&job(&writer, &holder, 0x100, 0x800)));
        assert!(book.stands(&job(&writer, &holder, 0x500, 0x400)));
        // Fetched for a reader.
        book.remove(KEY);
        assert!(!book.stands(&job(&writer, &holder, 0x500, 0x400)));
    }

    /// A piece is left again only with the holders it was first left with;
    /// one taken once is fetched from then on; and a page has one writer.
    #[test]
    fn a_piece_is_noted_with_its_holders_once() {
        let writer = copy(KEY, 0x40000);
        let holder = copy(ALIAS, 0x20000);
        let second = copy(ALIAS, 0x20000);
        let page = Some((7usize, 0x30000u64));
        let targets = vec![(holder.clone(), ALIAS + 0x100)];
        let mut book = Book::new(false);
        assert_eq!(
            note(
                &mut book,
                &writer,
                KEY,
                (0x100, 0x900),
                page,
                &[ALIAS],
                &targets
            )
            .ok(),
            Some(true)
        );
        let left = &book.left[&KEY];
        assert_eq!(left.held(), vec![(holder.id(), ALIAS)]);
        assert!(
            left.kept_by(holder.id())
                && left.kept_by(writer.id())
                && !left.kept_by(0)
                && !left.kept_by(second.id())
        );
        assert_eq!(book.alias.get(&ALIAS), Some(&KEY));
        // Left again, more of it: one piece, the runs joined.
        assert_eq!(
            note(
                &mut book,
                &writer,
                KEY,
                (0x900, 0xa00),
                page,
                &[ALIAS],
                &targets
            )
            .ok(),
            Some(false)
        );
        assert_eq!(book.left[&KEY].runs, vec![(0x100, 0xa00)]);
        // Another copy holds the other address now as well: not left.
        let more = vec![
            (holder.clone(), ALIAS + 0x100),
            (second.clone(), ALIAS + 0x100),
        ];
        assert_eq!(
            note(
                &mut book,
                &writer,
                KEY,
                (0x100, 0x900),
                page,
                &[ALIAS],
                &more
            )
            .err(),
            Some(Some(CHANGED))
        );
        assert_eq!(
            note(&mut book, &writer, KEY, (0x100, 0x900), page, &[ALIAS], &[]).err(),
            Some(Some(CHANGED))
        );
        // The holder's own shaders wrote the page at its address: the
        // page is the writer's piece, so that one is not left.
        let back = vec![(writer.clone(), KEY + 0x100)];
        assert_eq!(
            note(
                &mut book,
                &holder,
                ALIAS,
                (0x100, 0x900),
                page,
                &[KEY],
                &back
            )
            .err(),
            Some(Some(OTHER))
        );
        // Taken once: fetched from then on (no reason of mode 3's).
        book.remove(KEY);
        book.never.insert(KEY);
        assert_eq!(
            note(
                &mut book,
                &writer,
                KEY,
                (0x100, 0x900),
                page,
                &[ALIAS],
                &targets
            )
            .err(),
            Some(None)
        );
        assert!(book.left.is_empty());
    }

    /// The title lets go of the writer's address and still shows the pages
    /// at the other: the piece is kept there, for whoever reads there, its
    /// former holder too, and for the writer before it is used again.
    #[test]
    fn a_piece_whose_address_goes_is_stranded_at_the_other() {
        let writer = copy(KEY - 0x1000, 0x40000);
        let holder = copy(ALIAS, 0x20000);
        let mut book = Book::new(false);
        book.insert(KEY, piece(&writer, (0x1100, 0x1900), ALIAS, &holder));
        assert!(book.strand(KEY, ALIAS));
        assert_eq!(book.stranded, 1);
        assert!(!book.left.contains_key(&KEY));
        let left = &book.left[&ALIAS];
        assert!(left.stranded && left.holders.is_empty() && left.aliases.is_empty());
        // The same offsets of the writer, merged at the other address.
        assert_eq!(left.runs, vec![(0x1100, 0x1900)]);
        assert_eq!(title_span(left), (ALIAS + 0x100, ALIAS + 0x900));
        assert_eq!(left.base.wrapping_add(0x1100), ALIAS + 0x100);
        assert!(!left.kept_by(writer.id()) && !left.kept_by(holder.id()));
        // Nothing at the address that went; everyone at the other.
        assert!(
            book.wanted(For::Reader(0), KEY, KEY + PIECE, &[])
                .is_empty()
        );
        assert_eq!(
            book.wanted(For::Reader(holder.id()), ALIAS, ALIAS + PIECE, &[]),
            vec![ALIAS]
        );
        assert_eq!(
            book.wanted(For::Reader(0), ALIAS, ALIAS + PIECE, &[]),
            vec![ALIAS]
        );
        // The writer, wherever it is refreshed or written next.
        assert_eq!(
            book.wanted(For::Reader(writer.id()), KEY, KEY + PIECE, &[]),
            vec![ALIAS]
        );
        assert_eq!(
            book.wanted(For::Writing(writer.id()), KEY + PIECE, KEY + 2 * PIECE, &[]),
            vec![ALIAS]
        );
        assert!(
            book.wanted(For::Reader(holder.id()), KEY, KEY + PIECE, &[])
                .is_empty()
        );
        // And nothing of the writer's is left anew meanwhile.
        let page = Some((8usize, 0u64));
        assert_eq!(
            note(
                &mut book,
                &writer,
                KEY + PIECE,
                (0x11000, 0x11100),
                page,
                &[],
                &[]
            )
            .err(),
            Some(Some(CHANGED))
        );
        // No copy on the card to the former holder stands.
        let job = Forward {
            from: writer.clone(),
            lo: 0x1100,
            to: holder.clone(),
            at: 0x100,
            bytes: 0x800,
            key: KEY,
        };
        assert!(!book.stands(&job));
        book.remove(ALIAS);
        assert_eq!(book.stranded, 0);
        assert!(book.by_page.is_empty());
    }

    /// The title lets go of the other address: the copy there holds the
    /// piece no longer, no copy on the card to it stands, and the piece
    /// stays the writer's.
    #[test]
    fn an_address_of_the_pages_that_goes_takes_its_holder_with_it() {
        let writer = copy(KEY - 0x1000, 0x40000);
        let holder = copy(ALIAS - 0x2000, 0x20000);
        let mut book = Book::new(false);
        book.insert(KEY, piece(&writer, (0x1100, 0x1900), ALIAS, &holder));
        book.left.get_mut(&KEY).unwrap().runs.push((0x2000, 0x2100));
        let job = Forward {
            from: writer.clone(),
            lo: 0x1100,
            to: holder.clone(),
            at: 0x2100,
            bytes: 0x800,
            key: KEY,
        };
        assert!(book.stands(&job));
        assert!(
            book.unalias(ALIAS + PIECE).is_empty(),
            "no piece is known there"
        );
        // The offsets at which the holder held it: its card side there
        // gets its CPU side's again.
        let held = book.unalias(ALIAS);
        let offsets: Vec<(usize, usize)> = held.iter().map(|h| (h.1, h.2)).collect();
        assert_eq!(offsets, vec![(0x2100, 0x2900), (0x3000, 0x3100)]);
        assert!(held.iter().all(|h| Arc::ptr_eq(&h.0, &holder)));
        let left = &book.left[&KEY];
        assert!(left.holders.is_empty() && left.aliases.is_empty() && !left.stranded);
        assert!(book.alias.is_empty() && !book.may_hold(holder.id()));
        assert!(!book.stands(&job));
        assert!(!left.kept_by(holder.id()) && left.kept_by(writer.id()));
        // Whoever reads at the address now reads other pages: no fetch
        // for it (but by the page, should the title show it there again).
        assert!(
            book.wanted(For::Reader(0), ALIAS, ALIAS + PIECE, &[])
                .is_empty()
        );
        assert_eq!(
            book.wanted(For::Reader(0), KEY, KEY + PIECE, &[]),
            vec![KEY]
        );
        // Left again as a piece with no other address: the same piece.
        let page = Some((7usize, 0x30000u64));
        assert_eq!(
            note(&mut book, &writer, KEY, (0x1900, 0x1a00), page, &[], &[]).ok(),
            Some(false)
        );
    }

    /// The audit counts title bytes of a piece read at either address by
    /// anyone but the writer and the holders.
    #[test]
    fn the_audit_counts_reads_at_the_other_address() {
        let writer = copy(KEY, 0x40000);
        let holder = copy(ALIAS, 0x20000);
        let mut book = Book::new(false);
        book.insert(KEY, piece(&writer, (0x100, 0x900), ALIAS, &holder));
        // (In all, of them at the other address.)
        assert_eq!(stale_in(&book, ALIAS, ALIAS + PIECE, 0), (0x800, 0x800));
        assert_eq!(
            stale_in(&book, ALIAS + 0x800, ALIAS + 0x1000, 0),
            (0x100, 0x100)
        );
        assert_eq!(stale_in(&book, KEY, KEY + 0x200, 0), (0x100, 0));
        assert_eq!(stale_in(&book, ALIAS, ALIAS + PIECE, holder.id()), (0, 0));
        assert_eq!(stale_in(&book, KEY, KEY + PIECE, writer.id()), (0, 0));
        assert_eq!(
            stale_in(&book, ALIAS, ALIAS + PIECE, holder.id() + 1000),
            (0x800, 0x800)
        );
        assert_eq!(stale_in(&book, ALIAS + PIECE, ALIAS + 2 * PIECE, 0), (0, 0));
    }

    /// The frame line of mode 3 says megabytes and pieces forwarded, the
    /// copies in the air, and why pieces were not left, by reason.
    #[test]
    fn the_frame_line_says_what_was_forwarded_and_what_was_refused() {
        let stats = [
            300, 30_000_000, 360, 350, 29_000_000, 20, 10, 5_000, 40_000, 2,
        ];
        let mut refused = [0u64; REFUSALS.len()];
        refused[super::IN_PC] = 50;
        refused[super::WRITTEN] = 10;
        let line = forwards_line(10.0, &stats, 7, &refused);
        assert!(
            line.contains("per frame 3.0 MB forwarded in 30 pieces"),
            "{line}"
        );
        assert!(
            line.contains("36 copies asked for; 35 made, 2.9 MB, in 2.0 lists; 1.0 dropped"),
            "{line}"
        );
        assert!(
            line.contains("0.50 ms bringing the holders' CPU sides in line"),
            "{line}"
        );
        assert!(line.contains("7 copies in the air"), "{line}");
        assert!(line.contains("2 pieces stranded"), "{line}");
        assert!(
            line.contains(
                "5.0 for a holder in the PC's memory, 1.0 for a holder shaders write in the air"
            ),
            "{line}"
        );
        assert!(!line.contains("another copy at the same address"), "{line}");
        let line = forwards_line(10.0, &[0; 10], 0, &[0; REFUSALS.len()]);
        assert!(line.ends_with("per frame: none"), "{line}");
    }
}
