//! Host pipelines are made by workers, not by the title thread that first
//! records a draw with them. Translating a shader and the host driver's
//! compile take from milliseconds to seconds each (structurisation of the
//! biggest compute programs ~0.4 s, some driver compiles 3.4 s), and
//! the title's first world frame wants over 1,200 new pipelines: made
//! on its recording threads, that frame took more than 30 s and the
//! title's GPU watchdog ended the run (decision 0007, "Pipelines made off
//! the title's threads").
//!
//! A first use queues the pipeline and waits for it within a budget per
//! frame, so the cheap ones (menus, one-off passes such as a colour-grading
//! bake) are there at once as before; past the budget a draw or dispatch
//! whose pipeline is not ready is dropped, as an untranslated one is.
//!
//! The default budget is half a second per title frame. A pass the title
//! runs only once and whose pipeline takes longer is lost. Passes over the
//! screen (no vertex input, no depth) have workers and a longer budget of
//! their own ([`screen_budget`]): the title fills its lighting tables with
//! such draws once, at the world's first frame, and behind a world's worth of
//! pipelines they were dropped; that run had no lamps and no diffuse light to
//! its end (run textures-d2, the first after a new translation; Belobug
//! 18e57db8 found the same).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::Shader;
use super::host::{GraphicsState, HostPipeline};

/// What a host pipeline is made from, and the host pipeline once made.
pub(crate) struct Pending {
    shaders: Vec<Arc<Shader>>,
    pub(crate) graphics: Option<Box<GraphicsState>>,
    queue_state: AtomicU8,
    /// Taken by a worker from [`Queue::slow`].
    was_slow: AtomicBool,
    made: OnceLock<Option<HostPipeline>>,
    /// What identifies it across runs (shader containers and state), and
    /// whether a draw or dispatch has requested it yet.
    key: String,
    used: AtomicBool,
    /// A pass over the screen ([`GraphicsState::screen_pass`]).
    screen: bool,
    /// The title frame in which a thread first waited for it (`u64::MAX`:
    /// none yet). A pipeline is waited for in one frame only: one the
    /// worker has not made within that frame's budget (a slow or stuck
    /// driver compile) is dropped in later frames until it is made, instead
    /// of taking the whole budget again in every frame that draws with it
    /// (#1181: 500 ms waited in every world frame, felt as a freeze while
    /// the player moves). `VICTORY_PIPELINE_WAIT_EVERY_FRAME=1` waits in
    /// every frame as before.
    waited_frame: AtomicU64,
}

// SAFETY: D3D12 objects are free-threaded; the graphics state (whose input
// elements point into its own names) is only read after creation.
unsafe impl Send for Pending {}
// SAFETY: as above.
unsafe impl Sync for Pending {}

impl Pending {
    pub(crate) fn new(shaders: Vec<Arc<Shader>>, graphics: Option<Box<GraphicsState>>) -> Pending {
        let key = key(&shaders, graphics.as_deref());
        Pending {
            screen: graphics.as_ref().is_some_and(|g| g.screen_pass()),
            shaders,
            graphics,
            queue_state: AtomicU8::new(NOT_QUEUED),
            was_slow: AtomicBool::new(false),
            made: OnceLock::new(),
            key,
            used: AtomicBool::new(false),
            waited_frame: AtomicU64::new(u64::MAX),
        }
    }

    pub(super) fn key(&self) -> &str {
        &self.key
    }

    fn make(&self) {
        // Not through `OnceLock::get_or_init`: a worker the driver's compiler
        // crashed under stays inside the call for good (see [`guard_start`]),
        // and anyone else asking would wait for it forever.
        if self.made.get().is_none() {
            let made = if is_bad_key(&self.key) {
                SKIPPED_BAD.fetch_add(1, Ordering::Relaxed);
                tracing::warn!(key = self.key, shaders = ?self.shaders, "not made: the driver's compiler crashed on it before");
                None
            } else {
                let _building = crate::timeline::span("pipeline: build a host pipeline");
                let began = Instant::now();
                let made = HostPipeline::new(&self.shaders, self.graphics.as_deref());
                super::library::note_time(&self.key, began.elapsed().as_millis() as u64);
                made
            };
            let _ = self.made.set(made);
        }
        let (lock, done) = &DONE;
        let _held = lock.lock().unwrap_or_else(|e| e.into_inner());
        done.notify_all();
    }
}

/// What identifies a title pipeline across runs (the early list, the
/// pre-build's list): its shader containers and graphics state.
pub(crate) fn key(shaders: &[Arc<Shader>], graphics: Option<&GraphicsState>) -> String {
    let mut key = blake3::Hasher::new();
    for s in shaders {
        key.update(s.stage.as_bytes());
        key.update(s.hash.as_bytes());
    }
    key.update(graphics.map(|g| g.describe()).unwrap_or_default().as_bytes());
    key.finalize().to_hex()[..32].to_owned()
}

/// Pipelines not made for being on the list of those the driver's compiler
/// crashes on.
static SKIPPED_BAD: AtomicU64 = AtomicU64::new(0);

/// The keys of the pipelines the driver's compiler crashed on
/// (`warm-bad.txt` beside the pipeline library; a line is the translator
/// and driver it crashed under ([`bad_under`]), the key, then anything; `#`
/// begins a comment), loaded on first use. A line of another translator or
/// driver, or of before the lines said which, is not honoured: a pipeline
/// the compiler crashed on once was left out for good, though a fix to the
/// translation or a new driver may make it. `VICTORY_WARM_BAD=0` leaves the
/// file unread.
static BAD_KEYS: Mutex<Option<std::collections::HashSet<String>>> = Mutex::new(None);

/// The translator with its settings and the driver that this run's crashes
/// are noted under, as `victory warm` notes its own.
fn bad_under() -> &'static str {
    static FINGERPRINT: OnceLock<String> = OnceLock::new();
    FINGERPRINT.get_or_init(super::known::fingerprint)
}

/// The keys `listed` names as bad under `fingerprint`.
fn bad_listed(listed: &str, fingerprint: &str) -> std::collections::HashSet<String> {
    listed
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .filter_map(|l| {
            let mut words = l.split_whitespace();
            (words.next()? == fingerprint).then(|| words.next().map(str::to_owned))?
        })
        .collect()
}

fn bad_keys<R>(f: impl FnOnce(&mut std::collections::HashSet<String>) -> R) -> R {
    // (Before the lock: the fingerprint asks the host.)
    let fingerprint = bad_under();
    let mut keys = BAD_KEYS.lock().unwrap_or_else(|e| e.into_inner());
    let keys = keys.get_or_insert_with(|| {
        if std::env::var_os("VICTORY_WARM_BAD").is_some_and(|v| v == "0") {
            return Default::default();
        }
        let listed = super::library::folder()
            .and_then(|d| std::fs::read_to_string(d.join("warm-bad.txt")).ok())
            .unwrap_or_default();
        bad_listed(&listed, fingerprint)
    });
    f(keys)
}

fn is_bad_key(key: &str) -> bool {
    bad_keys(|keys| keys.contains(key))
}

/// Whether the driver's compiler crashes are caught
/// (`VICTORY_COMPILER_GUARD=0`: not).
fn guard_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_COMPILER_GUARD").is_none_or(|v| v != "0"))
}

/// Worker threads that can be told apart by the guard, over the run.
const MAX_SLOTS: usize = 128;
/// Per worker slot: its compile crashed (set in the handler), where.
static CRASHED: [AtomicBool; MAX_SLOTS] = [const { AtomicBool::new(false) }; MAX_SLOTS];
static FAULT_ADDR: [AtomicU64; MAX_SLOTS] = [const { AtomicU64::new(0) }; MAX_SLOTS];
static FAULT_RIP: [AtomicU64; MAX_SLOTS] = [const { AtomicU64::new(0) }; MAX_SLOTS];
static NEXT_SLOT: AtomicUsize = AtomicUsize::new(0);
/// The event the handler signals (a Win32 handle).
static CRASH_EVENT: AtomicUsize = AtomicUsize::new(0);
/// What each slot's worker is making, and whether it takes passes over the
/// screen alone. Never locked by the handler.
static ACTIVE_JOBS: Mutex<Vec<Option<(Arc<Pending>, bool)>>> = Mutex::new(Vec::new());

thread_local! {
    /// This worker's slot (`usize::MAX`: none).
    static WORKER_SLOT: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) };
    /// Whether this thread is inside the compile of a pipeline.
    static COMPILING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// A worker's compile of `p`: noted for the guard while it lasts.
struct Compile {
    slot: usize,
}

impl Compile {
    fn begin(slot: usize, p: &Arc<Pending>, screens_only: bool) -> Compile {
        if slot < MAX_SLOTS && guard_on() {
            let mut jobs = ACTIVE_JOBS.lock().unwrap_or_else(|e| e.into_inner());
            if jobs.len() < MAX_SLOTS {
                jobs.resize(MAX_SLOTS, None);
            }
            jobs[slot] = Some((p.clone(), screens_only));
            COMPILING.set(true);
        }
        Compile { slot }
    }
}

impl Drop for Compile {
    fn drop(&mut self) {
        if COMPILING.replace(false) && self.slot < MAX_SLOTS {
            let mut jobs = ACTIVE_JOBS.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(job) = jobs.get_mut(self.slot) {
                *job = None;
            }
        }
    }
}

/// Installs the handler for the driver's compiler crashing (AMD's does on an
/// RX 9070 XT, and not on the same pipelines in every run: an access of the
/// first pages in `amdxc64.dll`) and the thread that deals with what it
/// left. The handler does the least there is: it notes where, signals, and
/// leaves the worker asleep for good. No allocation, no lock, no file: the
/// crash may have been under any of them. The watcher puts the pipeline on
/// the list of those left out (`warm-bad.txt`), says it is not made, gives
/// the worker's place in the counts back and starts another worker.
fn guard_start() {
    use windows_sys::Win32::System::Diagnostics::Debug::AddVectoredExceptionHandler;
    use windows_sys::Win32::System::Threading::CreateEventW;
    static ONCE: std::sync::Once = std::sync::Once::new();
    if !guard_on() {
        return;
    }
    ONCE.call_once(|| {
        // SAFETY: an unnamed auto-reset event, no security attributes.
        let event = unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) };
        if event.is_null() {
            tracing::warn!("no event for the driver's compiler crashes: they are not caught");
            return;
        }
        CRASH_EVENT.store(event as usize, Ordering::Release);
        let spawned = std::thread::Builder::new().name("victory-pipeline-watch".into()).spawn(watch);
        if let Err(e) = spawned {
            tracing::warn!(error = %e, "no thread for the driver's compiler crashes: they are not caught");
            return;
        }
        // SAFETY: a handler that lives as long as the process.
        unsafe {
            AddVectoredExceptionHandler(1, Some(compiler_crashed));
        }
        tracing::info!("the driver's compiler crashing on a worker is caught (VICTORY_COMPILER_GUARD=0: not)");
    });
}

/// The handler: a worker inside a compile read or wrote one of the first
/// pages. See [`guard_start`].
unsafe extern "system" fn compiler_crashed(
    info: *mut windows_sys::Win32::System::Diagnostics::Debug::EXCEPTION_POINTERS,
) -> i32 {
    use windows_sys::Win32::System::Threading::{SetEvent, Sleep};
    const ACCESS_VIOLATION: i32 = 0xc000_0005_u32 as i32;
    const CONTINUE_SEARCH: i32 = 0;
    // SAFETY: the system passes a live record.
    let record = unsafe { &*(*info).ExceptionRecord };
    if record.ExceptionCode != ACCESS_VIOLATION || record.NumberParameters < 2 || record.ExceptionInformation[1] >= 0x10000 {
        return CONTINUE_SEARCH;
    }
    if !COMPILING.try_with(|c| c.get()).unwrap_or(false) {
        return CONTINUE_SEARCH;
    }
    let slot = WORKER_SLOT.try_with(|s| s.get()).unwrap_or(usize::MAX);
    if slot >= MAX_SLOTS {
        return CONTINUE_SEARCH;
    }
    let _ = COMPILING.try_with(|c| c.set(false));
    FAULT_ADDR[slot].store(record.ExceptionInformation[1] as u64, Ordering::Relaxed);
    #[cfg(target_arch = "x86_64")]
    // SAFETY: the system passes a live context.
    FAULT_RIP[slot].store(unsafe { (*(*info).ContextRecord).Rip }, Ordering::Relaxed);
    CRASHED[slot].store(true, Ordering::Release);
    let event = CRASH_EVENT.load(Ordering::Acquire);
    if event != 0 {
        // SAFETY: the event lives as long as the process.
        unsafe { SetEvent(event as windows_sys::Win32::Foundation::HANDLE) };
    }
    loop {
        // SAFETY: a plain sleep.
        unsafe { Sleep(u32::MAX - 1) };
    }
}

/// The thread that deals with the workers the compiler crashed under.
fn watch() {
    use windows_sys::Win32::System::Threading::{INFINITE, WaitForSingleObject};
    loop {
        let event = CRASH_EVENT.load(Ordering::Acquire);
        // SAFETY: the event lives as long as the process.
        unsafe { WaitForSingleObject(event as windows_sys::Win32::Foundation::HANDLE, INFINITE) };
        for slot in 0..MAX_SLOTS {
            if CRASHED[slot].swap(false, Ordering::AcqRel) {
                lost(slot);
            }
        }
    }
}

/// The worker at `slot` is gone with the pipeline it was making.
fn lost(slot: usize) {
    let job = {
        let mut jobs = ACTIVE_JOBS.lock().unwrap_or_else(|e| e.into_inner());
        jobs.get_mut(slot).and_then(Option::take)
    };
    let Some((p, screens_only)) = job else {
        return;
    };
    tracing::error!(
        key = p.key,
        shaders = ?p.shaders,
        rip = format_args!("{:#x}", FAULT_RIP[slot].load(Ordering::Relaxed)),
        address = format_args!("{:#x}", FAULT_ADDR[slot].load(Ordering::Relaxed)),
        "the driver's compiler crashed making a pipeline: it is left out, the worker is replaced"
    );
    let known = bad_keys(|keys| !keys.insert(p.key.clone()));
    if !known && let Some(dir) = super::library::folder() {
        use std::io::Write;
        let file = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("warm-bad.txt"));
        if let Ok(mut f) = file {
            let _ = writeln!(f, "{} {} ended the compiler in a run: {:?}", bad_under(), p.key, p.shaders);
        }
    }
    let _ = p.made.set(None);
    if p.was_slow.swap(false, Ordering::Relaxed) {
        let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
        queue.slow_now = queue.slow_now.saturating_sub(1);
        AVAILABLE.notify_all();
    }
    {
        let (lock, done) = &DONE;
        let _held = lock.lock().unwrap_or_else(|e| e.into_inner());
        done.notify_all();
    }
    if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
        drained();
    }
    if !spawn_worker(screens_only) {
        tracing::warn!("no worker to take the place of the one the compiler crashed under");
    }
}

/// Why a pipeline has no host pipeline (diagnostics): its shaders that do
/// not translate, else "refused" or "being made"; `None` if it has one.
#[cfg(feature = "diagnostics")]
pub(crate) fn missing(p: &Pending) -> Option<String> {
    match p.made.get() {
        Some(Some(_)) => None,
        None => Some("being made".into()),
        Some(None) => {
            let failed: Vec<String> = p
                .shaders
                .iter()
                .filter(|s| matches!(s.translated.get(), Some(None)))
                .map(|s| format!("{} {}", s.stage, s.hash))
                .collect();
            Some(if failed.is_empty() {
                format!("refused {:?}", p.shaders)
            } else {
                failed.join(" + ")
            })
        }
    }
}

/// The hashes of the shaders `p` is made of.
pub(crate) fn hashes(p: &Pending) -> String {
    let hashes: Vec<&str> = p.shaders.iter().map(|s| s.hash.as_str()).collect();
    hashes.join("+")
}

/// Signalled whenever a worker has made a pipeline.
static DONE: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());

/// Waiting allowed per title frame, over all threads.
fn wait_budget() -> Duration {
    static BUDGET: OnceLock<Duration> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let millis = std::env::var("VICTORY_PIPELINE_WAIT_MS")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(500);
        tracing::info!(millis, "host pipeline wait budget per frame");
        Duration::from_millis(millis)
    })
}

/// As [`wait_budget`], for passes over the screen (`VICTORY_SCREEN_WAIT_MS`,
/// 10 s; 0: like the others). Below the title's 30 s watchdog.
fn screen_budget() -> Duration {
    static BUDGET: OnceLock<Duration> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let millis = std::env::var("VICTORY_SCREEN_WAIT_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(10_000);
        Duration::from_millis(millis)
    })
}

/// Whether passes over the screen go before the others.
fn screens_first() -> bool {
    !screen_budget().is_zero()
}

/// Pipelines queued and not made yet.
static WAITING: AtomicU64 = AtomicU64::new(0);

/// Pipelines queued and not made yet (for the frame line).
pub(crate) fn waiting() -> u64 {
    WAITING.load(Ordering::Relaxed)
}

/// The host pipeline if it is made (or is made within the frame's budget);
/// queues it on first use.
pub(crate) fn ready(p: &Arc<Pending>) -> Option<&HostPipeline> {
    // A still-pending pipeline was requested by the title too. Prewarm it
    // on the next launch, even if this run never got to use it.
    if !p.used.load(Ordering::Relaxed) && !p.used.swap(true, Ordering::Relaxed) {
        super::library::note_used(&p.key);
    }
    if p.made.get().is_none() {
        start(p, REQUESTED);
        wait(p, crate::frame::current());
    }
    p.made.get()?.as_ref()
}

/// Without waiting: `Some(true)` if `p` has a host pipeline, `Some(false)`
/// if it was refused, `None` while it is being made (asked for now, ahead
/// of everything no draw or dispatch asked for).
pub(crate) fn poll(p: &Arc<Pending>) -> Option<bool> {
    if p.made.get().is_none() {
        start(p, REQUESTED);
    }
    p.made.get().map(Option::is_some)
}

/// At the title's creation of a pipeline: starts it at once if an earlier
/// run drew with it (during loading, so the world's first frames find it).
pub(crate) fn early(p: &Arc<Pending>) {
    if !p.shaders.is_empty() && super::library::wanted_early(&p.key) {
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        start(p, EARLY);
    }
}

/// Whether the frame's budget is one for all threads together: a thread
/// that waits takes what is left of it and gives back what it did not
/// use. Counted only when a wait had ended, the budget was one per thread:
/// each of the title's five recording threads began its first wait of a
/// frame with nothing counted yet and waited the whole half second, 2.5 s
/// of waiting in every frame in which pipelines were being made (runs
/// 34-49: "2504 ms waiting for pipelines" in each such frame).
/// On by default; `VICTORY_PIPELINE_WAIT_SHARED=0` counts as before.
/// Whether a pipeline is waited for in every frame that draws with it (the
/// old behaviour), not only in the first one that had budget for it.
fn wait_every_frame() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_WAIT_EVERY_FRAME").is_some_and(|v| v == "1"))
}

/// Claims frame `frame` as the one in which `p` is waited for; whether
/// `frame` is (or already was) that frame.
fn claim_wait(p: &Pending, frame: u64) -> bool {
    if wait_every_frame() {
        return true;
    }
    match p.waited_frame.compare_exchange(u64::MAX, frame, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => true,
        Err(first) => first == frame,
    }
}

fn shared_budget() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_WAIT_SHARED").is_none_or(|v| v != "0"))
}

/// Waits for `p` while frame `frame` has budget left.
fn wait(p: &Pending, frame: u64) {
    let screen = p.screen && screens_first();
    let budget = if screen { screen_budget() } else { wait_budget() };
    if budget.is_zero() {
        return;
    }
    // The frame the budget belongs to, and what was waited of it (or is
    // taken by a thread waiting now); of the passes over the screen too.
    static SPENT: [Mutex<(u64, Duration)>; 2] = [const { Mutex::new((u64::MAX, Duration::ZERO)) }; 2];
    let spent_of = &SPENT[usize::from(screen)];
    let shared = shared_budget();
    let left = {
        let mut spent = spent_of.lock().unwrap_or_else(|e| e.into_inner());
        if spent.0 != frame {
            *spent = (frame, Duration::ZERO);
        }
        let left = budget.saturating_sub(spent.1);
        // Claimed only with time left, so a pipeline first bound when the
        // frame's budget is spent keeps its one wait for a later frame.
        // Passes over the screen keep waiting in every frame (their own
        // budget, `VICTORY_SCREEN_WAIT_MS`): dropping one shows nothing.
        if left.is_zero() || (!screen && !claim_wait(p, frame)) {
            return;
        }
        if shared {
            spent.1 += left;
        }
        left
    };
    let waiting = crate::timeline::span("pipeline: first use waits for a host pipeline");
    let started = Instant::now();
    let (lock, done) = &DONE;
    let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
    while p.made.get().is_none() {
        let rest = left.saturating_sub(started.elapsed());
        if rest.is_zero() {
            break;
        }
        held = done.wait_timeout(held, rest).unwrap_or_else(|e| e.into_inner()).0;
    }
    drop(held);
    let waited = started.elapsed().min(left);
    waiting.end();
    if screen && waited > Duration::from_millis(500) {
        tracing::info!(
            ms = waited.as_millis() as u64,
            made = p.made.get().is_some(),
            shaders = ?p.shaders,
            "a pass over the screen was waited for (VICTORY_SCREEN_WAIT_MS=0: dropped past half a second a frame)"
        );
    }
    {
        let mut spent = spent_of.lock().unwrap_or_else(|e| e.into_inner());
        // A frame that began meanwhile has its own budget.
        if spent.0 == frame {
            if shared {
                spent.1 = spent.1.saturating_sub(left - waited);
            } else {
                spent.1 += waited;
            }
        }
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(
        crate::copystat::C::PipelineWaitUs,
        started.elapsed().as_micros() as u64,
    );
}

const NOT_QUEUED: u8 = 0;
const EARLY: u8 = 1;
const REQUESTED: u8 = 2;
const MAKING: u8 = 3;

/// How many workers may make pipelines at once that are known to take long
/// and that no draw has asked for (asked for too, with [`low_memory`]): a
/// quarter of them ([`slow_limit_of`]). On a fresh cache (a
/// change of the translator or the driver) the early list begins with
/// programs that take tens of seconds each: all 16 workers were in them
/// through the whole of the title's loading, 97 pipelines made in 20 s, and
/// the first minute and a half of the world ran at 0.5 to 1.6 s a frame,
/// dropping up to 2,900 draws a frame (run shaders-1).
/// `VICTORY_PIPELINE_SLOW=0`: in the order the title creates them, as before.
fn slow_limit() -> usize {
    static LIMIT: OnceLock<usize> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        if std::env::var_os("VICTORY_PIPELINE_SLOW").is_some_and(|v| v == "0") {
            usize::MAX
        } else {
            slow_limit_of(configured_workers())
        }
    })
}

/// A quarter of `workers`, at least one: of the workers there are
/// ([`configured_workers`], `VICTORY_PIPELINE_WORKERS` included), not of the
/// default for the processors (with 4 workers asked on 32 processors, 4 of
/// them could be in long ones at once).
fn slow_limit_of(workers: usize) -> usize {
    (workers / 4).max(1)
}

/// Whether pipelines known to take long count against [`slow_limit`] when a
/// draw or a pass over the screen asks for them too, not only when they were
/// started early (`VICTORY_PIPELINE_LOW_MEMORY=1`; off by default). Those
/// keep their place ahead of the early ones and take the next free slot,
/// and quick ones go on beside them while they wait; a pipeline asked for
/// can then wait behind a long one that is being made. For runs short of
/// memory: each long driver compile holds hundreds of MB while it runs, and
/// unbounded, first uses could start one on every worker (issue #56). Known
/// is all there is to go on: a pipeline the library holds is counted too.
fn low_memory() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_PIPELINE_LOW_MEMORY").is_some_and(|v| v == "1");
        if on {
            tracing::info!(slow_limit = slow_limit(), "pipelines known to take long are limited at first use too");
        }
        on
    })
}

struct Queue {
    /// Requested passes over the screen: before all others, and the only ones
    /// [`SCREEN_WORKERS`] take.
    screen: VecDeque<Arc<Pending>>,
    early: VecDeque<Arc<Pending>>,
    /// Those of the early ones known to take long, and how many of them are
    /// being made ([`slow_limit`]).
    slow: VecDeque<Arc<Pending>>,
    slow_now: usize,
    requested: VecDeque<Arc<Pending>>,
    /// Requested ones known to take long, passes over the screen and others
    /// ([`low_memory`]; else they are in `screen` and `requested`): counted
    /// in `slow_now`, and before the early ones known to take long.
    slow_screen: VecDeque<Arc<Pending>>,
    slow_requested: VecDeque<Arc<Pending>>,
}

impl Queue {
    const fn new() -> Self {
        Self {
            screen: VecDeque::new(),
            early: VecDeque::new(),
            slow: VecDeque::new(),
            slow_now: 0,
            requested: VecDeque::new(),
            slow_screen: VecDeque::new(),
            slow_requested: VecDeque::new(),
        }
    }

    /// Queues an unrequested pipeline or promotes one queued at creation.
    /// Returns whether it is new to the queue, or `None` if nothing changed.
    fn push(&mut self, p: &Arc<Pending>, priority: u8) -> Option<bool> {
        let slow = slow_limit() != usize::MAX && super::library::slow(&p.key);
        self.push_in(p, priority, slow, low_memory())
    }

    /// As [`Queue::push`], with `low_memory` off.
    #[cfg(test)]
    fn push_as(&mut self, p: &Arc<Pending>, priority: u8, slow: bool) -> Option<bool> {
        self.push_in(p, priority, slow, false)
    }

    /// As [`Queue::push`]: `slow`, whether `p` is known to take long;
    /// `low_memory`, whether it is limited when requested too ([`low_memory`]).
    fn push_in(&mut self, p: &Arc<Pending>, priority: u8, slow: bool, low_memory: bool) -> Option<bool> {
        let prev = p.queue_state.load(Ordering::Relaxed);
        if prev >= priority {
            return None;
        }
        p.queue_state.store(priority, Ordering::Relaxed);
        let limited = slow && (priority != REQUESTED || low_memory);
        let list = match (priority == REQUESTED, p.screen && screens_first(), limited) {
            (true, true, false) => &mut self.screen,
            (true, true, true) => &mut self.slow_screen,
            (true, false, false) => &mut self.requested,
            (true, false, true) => &mut self.slow_requested,
            (false, _, true) => &mut self.slow,
            (false, _, false) => &mut self.early,
        };
        list.push_back(p.clone());
        Some(prev == NOT_QUEUED)
    }

    /// The next pipeline to make: passes over the screen, then requested
    /// ones, then early ones; of each, one known to take long only while
    /// fewer than `slow_limit` such are being made. Workers that make passes
    /// over the screen alone (`screens_only`) take nothing else.
    fn pop(&mut self, screens_only: bool, slow_limit: usize) -> Option<Arc<Pending>> {
        // An entry can have been promoted and claimed already: an early one
        // is then in a requested list too.
        let unclaimed = |list: &mut VecDeque<Arc<Pending>>, state: u8| {
            std::iter::from_fn(|| list.pop_front()).find(|p| p.queue_state.load(Ordering::Relaxed) == state)
        };
        let free = self.slow_now < slow_limit;
        let mut slow = false;
        let mut p = self.screen.pop_front();
        if p.is_none() && free {
            p = unclaimed(&mut self.slow_screen, REQUESTED);
            slow = p.is_some();
        }
        if p.is_none() && !screens_only {
            p = self.requested.pop_front();
            if p.is_none() && free {
                p = unclaimed(&mut self.slow_requested, REQUESTED);
                slow = p.is_some();
            }
            if p.is_none() {
                p = unclaimed(&mut self.early, EARLY);
            }
            if p.is_none() && free {
                p = unclaimed(&mut self.slow, EARLY);
                slow = p.is_some();
            }
        }
        let p = p?;
        if slow {
            p.was_slow.store(true, Ordering::Relaxed);
            self.slow_now += 1;
        }
        p.queue_state.store(MAKING, Ordering::Relaxed);
        Some(p)
    }
}

static QUEUE: Mutex<Queue> = Mutex::new(Queue::new());
static AVAILABLE: Condvar = Condvar::new();

/// Start or promote work for a draw without waiting behind the entire early
/// list. Queue changes and worker claims happen under the same lock.
fn start(p: &Arc<Pending>, priority: u8) {
    if p.queue_state.load(Ordering::Relaxed) >= priority {
        return;
    }
    if !workers() {
        if p.queue_state.swap(MAKING, Ordering::Relaxed) == NOT_QUEUED {
            p.make();
        }
        return;
    }
    let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(new) = queue.push(p, priority) {
        if new {
            PEAK.fetch_max(WAITING.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
        }
        // (One woken could be a worker that takes passes over the screen
        // alone: it would leave the pipeline to nobody.)
        AVAILABLE.notify_all();
    }
}

/// Workers that make passes over the screen alone, besides the others.
const SCREEN_WORKERS: usize = 2;

fn next(screens_only: bool) -> Arc<Pending> {
    let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        if let Some(p) = queue.pop(screens_only, slow_limit()) {
            return p;
        }
        queue = AVAILABLE.wait(queue).unwrap_or_else(|e| e.into_inner());
    }
}

/// How many workers make pipelines: `asked` (`VICTORY_PIPELINE_WORKERS`, 1 to
/// 64), else two thirds of the logical processors, 2 to 16. They run below the
/// title's own threads, and a cold cache queues hundreds at once.
fn worker_count(asked: Option<usize>, processors: usize) -> usize {
    match asked {
        Some(n) => n.clamp(1, 64),
        None => (processors * 2 / 3).clamp(2, 16),
    }
}

/// [`worker_count`] of this machine and `VICTORY_PIPELINE_WORKERS`.
fn configured_workers() -> usize {
    worker_count(
        std::env::var("VICTORY_PIPELINE_WORKERS")
            .ok()
            .and_then(|v| v.trim().parse().ok()),
        std::thread::available_parallelism().map_or(2, |n| n.get()),
    )
}

/// Starts the workers ([`worker_count`]) on first use.
fn workers() -> bool {
    static RUNNING: OnceLock<bool> = OnceLock::new();
    *RUNNING.get_or_init(|| {
        let mut running = false;
        let workers = configured_workers();
        tracing::info!(workers, "pipeline workers");
        let screens = if screens_first() { SCREEN_WORKERS } else { 0 };
        guard_start();
        for i in 0..workers + screens {
            if spawn_worker(i >= workers) {
                running = true;
            }
        }
        running
    })
}

/// Starts one worker: one for passes over the screen alone, or one of the
/// others.
fn spawn_worker(screens_only: bool) -> bool {
    let slot = NEXT_SLOT.fetch_add(1, Ordering::Relaxed);
    let spawned = std::thread::Builder::new()
        .name(format!("victory-pipelines-{slot}"))
        .spawn(move || {
            WORKER_SLOT.set(slot);
            // Below the title's own threads (loading, recording),
            // and the streaming one, which is below normal itself
            // (`VICTORY_PIPELINE_LOWEST=0`: level with it, as before).
            let lowest = !screens_only && std::env::var_os("VICTORY_PIPELINE_LOWEST").is_none_or(|v| v != "0");
            // SAFETY: a plain call on our own thread.
            unsafe {
                use windows::Win32::System::Threading::*;
                let priority = if lowest { THREAD_PRIORITY_LOWEST } else { THREAD_PRIORITY_BELOW_NORMAL };
                let _ = SetThreadPriority(GetCurrentThread(), priority);
            }
            loop {
                let p = next(screens_only);
                {
                    // (If the compiler crashes in here this thread never
                    // returns: `lost` does what follows for it.)
                    let _compile = Compile::begin(slot, &p, screens_only);
                    p.make();
                }
                if p.was_slow.load(Ordering::Relaxed) {
                    let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
                    queue.slow_now -= 1;
                    AVAILABLE.notify_all();
                }
                if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
                    drained();
                } else {
                    super::library::save(false);
                }
            }
        });
    if let Err(e) = spawned {
        tracing::warn!(error = %e, "no pipeline worker");
        false
    } else {
        true
    }
}

/// The most pipelines waiting at once since the workers last ran out.
static PEAK: AtomicU64 = AtomicU64::new(0);

/// The workers ran out of work: logged after a batch of 10 or more, and
/// the cache saved (at most every 20 s).
fn drained() {
    use super::library::{EARLY, LOADED, MADE};
    if PEAK.swap(0, Ordering::Relaxed) >= 10 {
        tracing::info!(
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipelines all made"
        );
    }
    super::library::save(false);
}

/// The queue with [`low_memory`] on (issue #56), driven as the workers
/// drive it: no threads, no host device, no title data.
#[cfg(test)]
mod low_memory_tests {
    use super::*;

    fn job(screen: bool) -> Arc<Pending> {
        let mut p = Pending::new(Vec::new(), None);
        p.screen = screen;
        Arc::new(p)
    }

    /// What a worker does once it has made `p`.
    fn finish(queue: &mut Queue, p: &Pending) {
        if p.was_slow.load(Ordering::Relaxed) {
            queue.slow_now -= 1;
        }
    }

    fn pops(queue: &mut Queue, screens_only: bool, limit: usize, p: &Arc<Pending>) -> bool {
        queue.pop(screens_only, limit).is_some_and(|got| Arc::ptr_eq(&got, p))
    }

    /// One slot, held by an early long one: a long first use and a long pass
    /// over the screen wait, quick ones go on (one of them queued after the
    /// long first use, which heads its queue); then the slot goes to the
    /// pass over the screen, then to the first use, each once.
    #[test]
    fn requested_long_ones_wait_for_a_slot_while_quick_ones_go_on() {
        let (held, first_use, screen) = (job(false), job(false), job(true));
        let (quick_use, quick_early) = (job(false), job(false));
        let mut queue = Queue::new();
        assert_eq!(queue.push_in(&held, EARLY, true, true), Some(true));
        assert!(pops(&mut queue, false, 1, &held));
        assert_eq!(queue.slow_now, 1);

        assert_eq!(queue.push_in(&first_use, REQUESTED, true, true), Some(true));
        assert_eq!(queue.push_in(&screen, REQUESTED, true, true), Some(true));
        assert_eq!(queue.push_in(&quick_use, REQUESTED, false, true), Some(true));
        assert_eq!(queue.push_in(&quick_early, EARLY, false, true), Some(true));

        assert!(pops(&mut queue, false, 1, &quick_use));
        assert!(pops(&mut queue, false, 1, &quick_early));
        assert!(queue.pop(false, 1).is_none());
        assert!(queue.pop(true, 1).is_none());
        for quick in [&quick_use, &quick_early] {
            assert!(!quick.was_slow.load(Ordering::Relaxed));
            finish(&mut queue, quick);
        }
        assert_eq!(queue.slow_now, 1, "quick ones hold no slot");

        finish(&mut queue, &held);
        assert!(pops(&mut queue, false, 1, &screen));
        assert!(screen.was_slow.load(Ordering::Relaxed));
        assert!(queue.pop(false, 1).is_none());
        finish(&mut queue, &screen);
        assert!(pops(&mut queue, false, 1, &first_use));
        assert!(first_use.was_slow.load(Ordering::Relaxed));
        assert!(queue.pop(false, 1).is_none());
        finish(&mut queue, &first_use);
        assert_eq!(queue.slow_now, 0);
        assert!(queue.pop(false, 1).is_none());
    }

    /// A long one queued early and then drawn with is in two lists; the
    /// freed slot goes to it as a first use, ahead of an earlier early
    /// one, and it is made once.
    #[test]
    fn a_promoted_long_one_goes_before_the_early_ones_and_is_made_once() {
        let (held, early, promoted) = (job(false), job(false), job(false));
        let mut queue = Queue::new();
        assert_eq!(queue.push_in(&held, EARLY, true, true), Some(true));
        assert!(pops(&mut queue, false, 1, &held));
        assert_eq!(queue.push_in(&early, EARLY, true, true), Some(true));
        assert_eq!(queue.push_in(&promoted, EARLY, true, true), Some(true));
        assert_eq!(queue.push_in(&promoted, REQUESTED, true, true), Some(false));
        assert_eq!(queue.push_in(&promoted, REQUESTED, true, true), None);
        assert!(queue.pop(false, 1).is_none());

        finish(&mut queue, &held);
        assert!(pops(&mut queue, false, 1, &promoted));
        finish(&mut queue, &promoted);
        assert!(pops(&mut queue, false, 1, &early));
        finish(&mut queue, &early);
        // The stale early entry of the promoted one is not made again.
        assert!(queue.pop(false, 1).is_none());
        assert_eq!(queue.slow_now, 0);
    }

    /// Workers that make passes over the screen alone take a long one too
    /// once a slot is free, and still nothing else.
    #[test]
    fn screen_workers_wait_for_a_slot_too() {
        let (held, screen, world) = (job(false), job(true), job(false));
        let mut queue = Queue::new();
        assert_eq!(queue.push_in(&held, REQUESTED, true, true), Some(true));
        assert!(pops(&mut queue, false, 1, &held));
        assert_eq!(queue.push_in(&screen, REQUESTED, true, true), Some(true));
        assert_eq!(queue.push_in(&world, REQUESTED, true, true), Some(true));
        assert!(queue.pop(true, 1).is_none());
        finish(&mut queue, &held);
        assert!(pops(&mut queue, true, 1, &screen));
        finish(&mut queue, &screen);
        assert!(queue.pop(true, 1).is_none());
        assert!(pops(&mut queue, false, 1, &world));
    }

    /// A cache hit: a pipeline made already is not queued at all, and one
    /// not known to take long (the library loads it in milliseconds) holds
    /// no slot. A miss known to take long waits for one.
    #[test]
    fn cache_hits_go_on_while_a_miss_waits() {
        let made = job(false);
        assert!(made.made.set(None).is_ok());
        assert!(ready(&made).is_none());
        assert_eq!(made.queue_state.load(Ordering::Relaxed), NOT_QUEUED);

        let (held, hit, miss) = (job(false), job(false), job(false));
        let mut queue = Queue::new();
        assert_eq!(queue.push_in(&held, REQUESTED, true, true), Some(true));
        assert!(pops(&mut queue, false, 1, &held));
        assert_eq!(queue.push_in(&miss, REQUESTED, true, true), Some(true));
        assert_eq!(queue.push_in(&hit, REQUESTED, false, true), Some(true));
        assert!(pops(&mut queue, false, 1, &hit));
        assert!(!hit.was_slow.load(Ordering::Relaxed));
        assert!(queue.pop(false, 1).is_none());
        finish(&mut queue, &held);
        assert!(pops(&mut queue, false, 1, &miss));
    }

    /// Without a pipeline library (none in tests: no host device) nothing
    /// is loaded, and a pipeline that took long in this run is known to: a
    /// later request for the same key is limited.
    #[test]
    fn without_a_library_a_pipeline_that_took_long_is_limited() {
        let held = job(false);
        // A key of its own: every pipeline of no shaders has the same one,
        // and the others' tests must not find it known to take long.
        let mut again = Pending::new(Vec::new(), None);
        again.key = "low-memory-test-took-long".to_owned();
        let again = Arc::new(again);
        super::super::library::note_time(&again.key, super::super::library::SLOW_MS);
        assert!(super::super::library::slow(&again.key));
        assert!(!super::super::library::slow(&held.key));
        let mut queue = Queue::new();
        assert_eq!(queue.push_in(&held, REQUESTED, true, true), Some(true));
        assert!(pops(&mut queue, false, 1, &held));
        let slow = super::super::library::slow(&again.key);
        assert_eq!(queue.push_in(&again, REQUESTED, slow, true), Some(true));
        assert!(queue.pop(false, 1).is_none());
        finish(&mut queue, &held);
        assert!(pops(&mut queue, false, 1, &again));
    }

    /// The limit is a quarter of the workers there are, the switch's count
    /// included, and that many long first uses are made at once.
    #[test]
    fn the_limit_follows_the_worker_count_asked_for() {
        assert_eq!(slow_limit_of(worker_count(None, 24)), 4);
        assert_eq!(slow_limit_of(worker_count(Some(4), 32)), 1);
        assert_eq!(slow_limit_of(worker_count(Some(1), 32)), 1);
        assert_eq!(slow_limit_of(worker_count(Some(20), 4)), 5);
        assert_eq!(slow_limit_of(worker_count(Some(200), 4)), 16);

        let limit = slow_limit_of(worker_count(Some(8), 32));
        assert_eq!(limit, 2);
        let jobs: Vec<_> = (0..3).map(|_| job(false)).collect();
        let mut queue = Queue::new();
        for p in &jobs {
            assert_eq!(queue.push_in(p, REQUESTED, true, true), Some(true));
        }
        assert!(pops(&mut queue, false, limit, &jobs[0]));
        assert!(pops(&mut queue, false, limit, &jobs[1]));
        assert!(queue.pop(false, limit).is_none());
        finish(&mut queue, &jobs[0]);
        assert!(pops(&mut queue, false, limit, &jobs[2]));
    }

    /// With the switch off, a requested one known to take long is made at
    /// once, as before: over the screen or not, slot held or not.
    #[test]
    fn off_requested_long_ones_take_no_slot() {
        let (held, screen, world) = (job(false), job(true), job(false));
        let mut queue = Queue::new();
        assert_eq!(queue.push_in(&held, EARLY, true, false), Some(true));
        assert!(pops(&mut queue, false, 1, &held));
        assert_eq!(queue.push_in(&screen, REQUESTED, true, false), Some(true));
        assert_eq!(queue.push_in(&world, REQUESTED, true, false), Some(true));
        assert!(pops(&mut queue, true, 1, &screen));
        assert!(pops(&mut queue, false, 1, &world));
        assert!(!screen.was_slow.load(Ordering::Relaxed));
        assert!(!world.was_slow.load(Ordering::Relaxed));
        assert_eq!(queue.slow_now, 1);
        assert!(queue.slow_screen.is_empty() && queue.slow_requested.is_empty());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pipeline the compiler crashed on is left out only under the
    /// translator and driver it crashed under; lines of another, and lines
    /// of before they said which (a key first), are tried again.
    #[test]
    fn a_bad_pipeline_is_tried_again_by_another_translator_or_driver() {
        let listed = "1111 aaaa ended the compiler in a run: [compute x]\n\
                      2222 bbbb exit code: 101\n\
                      cccc ended the compiler in a run: [compute z]\n\
                      \n\
                      # 1111 eeee\n\
                      1111 dddd\n\
                      1111\n";
        let keys = |of| bad_listed(listed, of);
        assert_eq!(keys("1111"), std::collections::HashSet::from(["aaaa".to_owned(), "dddd".to_owned()]));
        assert_eq!(keys("2222"), std::collections::HashSet::from(["bbbb".to_owned()]));
        assert!(keys("3333").is_empty());
    }

    #[test]
    fn worker_count_follows_the_processors_and_the_switch() {
        assert_eq!(worker_count(None, 1), 2);
        assert_eq!(worker_count(None, 24), 16);
        assert_eq!(worker_count(None, 64), 16);
        assert_eq!(worker_count(Some(0), 24), 1);
        assert_eq!(worker_count(Some(20), 24), 20);
    }

    /// A first use queues the pipeline and, within the budget, gets the
    /// worker's answer (here: no host pipeline for no shaders).
    #[test]
    fn first_use_is_made_by_a_worker() {
        let p = Arc::new(Pending::new(Vec::new(), None));
        assert!(ready(&p).is_none());
        assert_ne!(p.queue_state.load(Ordering::Relaxed), NOT_QUEUED);
        let started = Instant::now();
        while p.made.get().is_none() && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert!(matches!(p.made.get(), Some(None)));
    }

    /// A pipeline the compiler crashed on is not made again: no compile.
    #[test]
    fn a_pipeline_on_the_bad_list_is_not_compiled() {
        let mut p = Pending::new(Vec::new(), None);
        p.key = "bad-pipeline-test".to_owned();
        bad_keys(|keys| keys.insert(p.key.clone()));
        let before = SKIPPED_BAD.load(Ordering::Relaxed);
        p.make();
        assert!(matches!(p.made.get(), Some(None)));
        assert!(SKIPPED_BAD.load(Ordering::Relaxed) > before);
    }

    #[test]
    fn pending_pipeline_is_recorded_for_the_next_launch() {
        let mut p = Pending::new(Vec::new(), None);
        p.key = "pending-pipeline-test".to_owned();
        // Simulate a worker that has taken the job but cannot finish it yet.
        p.queue_state.store(MAKING, Ordering::Relaxed);
        let p = Arc::new(p);

        assert!(!super::super::library::wanted_early(&p.key));
        assert!(ready(&p).is_none());
        assert!(p.made.get().is_none());
        assert!(super::super::library::wanted_early(&p.key));
    }

    #[test]
    fn draws_promote_early_pipelines_without_claiming_them_twice() {
        let earlier = Arc::new(Pending::new(Vec::new(), None));
        let promoted = Arc::new(Pending::new(Vec::new(), None));
        let requested = Arc::new(Pending::new(Vec::new(), None));
        let mut queue = Queue::new();

        assert_eq!(queue.push(&earlier, EARLY), Some(true));
        assert_eq!(queue.push(&promoted, EARLY), Some(true));
        assert_eq!(queue.push(&requested, REQUESTED), Some(true));
        assert_eq!(queue.push(&promoted, REQUESTED), Some(false));
        assert_eq!(queue.push(&promoted, REQUESTED), None);

        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &requested));
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &promoted));
        assert_eq!(queue.push(&promoted, REQUESTED), None);
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &earlier));
        assert!(queue.pop(false, 1).is_none());
    }

    #[test]
    fn pipelines_known_to_take_long_leave_workers_for_the_others() {
        let new = || Arc::new(Pending::new(Vec::new(), None));
        let (long, longer, asked, quick) = (new(), new(), new(), new());
        let mut queue = Queue::new();
        for p in [&long, &longer, &asked] {
            assert_eq!(queue.push_as(p, EARLY, true), Some(true));
        }
        assert_eq!(queue.push_as(&quick, EARLY, false), Some(true));
        // The quick one first, then one long one at a time; one a draw asks
        // for waits for nobody.
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &quick));
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &long));
        assert!(queue.pop(false, 1).is_none());
        assert_eq!(queue.push_as(&asked, REQUESTED, true), Some(false));
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &asked));
        assert!(!asked.was_slow.load(Ordering::Relaxed));
        assert!(queue.pop(false, 1).is_none());
        queue.slow_now -= 1;
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &longer));
        assert!(queue.pop(false, 1).is_none());
    }

    /// Passes over the screen go before all others, and workers of their own
    /// take nothing else.
    #[test]
    fn passes_over_the_screen_go_first() {
        let of = |screen| {
            let mut p = Pending::new(Vec::new(), None);
            p.screen = screen;
            Arc::new(p)
        };
        let (world, screen) = (of(false), of(true));
        let mut queue = Queue::new();
        assert_eq!(queue.push(&world, REQUESTED), Some(true));
        assert_eq!(queue.push(&screen, REQUESTED), Some(true));
        assert!(Arc::ptr_eq(&queue.pop(true, 1).unwrap(), &screen));
        assert!(queue.pop(true, 1).is_none());
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &world));
    }

    /// A pipeline still being made that was waited for in an earlier frame
    /// is not waited for again (#1181).
    #[test]
    fn a_pipeline_is_waited_for_in_one_frame_only() {
        let p = Pending::new(Vec::new(), None);
        // Claimed by a worker that has not finished.
        p.queue_state.store(MAKING, Ordering::Relaxed);
        let frame = crate::frame::current();
        // Another frame than this one, and never `u64::MAX` (none yet): with
        // the frame counter still at 0, `0 - 1` would be that.
        p.waited_frame.store(frame.wrapping_add(1), Ordering::Relaxed);
        let p = Arc::new(p);
        let started = Instant::now();
        assert!(ready(&p).is_none());
        assert!(started.elapsed() < Duration::from_millis(50), "waited {:?}", started.elapsed());
    }

    /// The claim: the first frame keeps it, the same frame may wait again
    /// (several draws in one frame), a later frame may not.
    #[test]
    fn the_first_frame_claims_the_wait() {
        let p = Pending::new(Vec::new(), None);
        assert!(claim_wait(&p, 7));
        assert!(claim_wait(&p, 7));
        assert!(!claim_wait(&p, 8));
        assert_eq!(p.waited_frame.load(Ordering::Relaxed), 7);
    }
}
