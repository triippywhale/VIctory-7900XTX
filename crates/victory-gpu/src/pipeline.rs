//! Pipeline states. Both descriptions start as on PC: the root signature,
//! then the shader stages (five for graphics, one for compute); streams
//! carry the rest of the graphics state as subobjects of the standard
//! types.
//!
//! Host pipelines are made on a pipeline's first draw, not at creation:
//! The title creates some 60,000 pipelines in its first 20
//! seconds and draws with a few dozen (decision 0007, step 3). Shader
//! containers are shared by content hash and translated once, on first
//! use (decision 0003).
//!
//! shortcut: only vertex plus pixel pipelines from streams and compute
//! pipelines get host pipelines; mesh, tessellation and geometry stages,
//! and the plain graphics description, keep their containers only. The 32-byte
//! packet after the header stays zero: titles only copy it inline, which
//! our null command-list limits keep from running.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::HashMap;
use std::mem::offset_of;
use std::sync::{Arc, Mutex, OnceLock};

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, PIPELINE, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

const PIPELINE_IIDS: [Iid; 4] = [
    guid(
        0x765a30f3,
        0xf624,
        0x4c6f,
        [0xa8, 0x28, 0xac, 0xe9, 0x48, 0x62, 0x24, 0x45],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
pub(crate) struct Bytecode {
    code: *const u8,
    size: usize,
}

#[repr(C)]
pub(crate) struct GraphicsHead {
    root: This,
    /// Vertex, pixel, domain, hull, geometry.
    stages: [Bytecode; 5],
}

#[repr(C)]
pub(crate) struct ComputeHead {
    root: This,
    stage: Bytecode,
}

/// A shader container a title passed, shared by every pipeline with the
/// same bytes.
pub(crate) struct Shader {
    pub(crate) stage: &'static str,
    /// The first 8 bytes of its BLAKE3, as logged.
    pub(crate) hash: String,
    /// The program's name, where the container has one (`CS_...`).
    pub(crate) name: Option<String>,
    container: Vec<u8>,
    /// Its part in the stand-in for the indirect light.
    #[cfg(windows)]
    pub(crate) ambient: Option<crate::ambient::Role>,
    /// Its part in the light probe (`crate::lightprobe`).
    #[cfg(windows)]
    pub(crate) probe: Option<crate::lightprobe::Role>,
    #[cfg(windows)]
    translated: OnceLock<Option<Translated>>,
}

impl std::fmt::Debug for Shader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.stage, self.hash)?;
        match &self.name {
            Some(name) => write!(f, " {name}"),
            None => Ok(()),
        }
    }
}

/// Every container seen, by full content hash.
static SHADERS: Mutex<Option<HashMap<[u8; 32], Arc<Shader>>>> = Mutex::new(None);

/// For the memory lines: containers kept and their bytes, translations
/// kept and the bytes of their host code. None of them is ever let go.
static KEPT: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// What the shader table and the pipeline library keep in memory:
/// containers and their bytes, translations and their bytes, the bytes of
/// the library's file (kept whole while the library lives) and what its
/// last save wrote.
#[cfg(windows)]
#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
pub(crate) fn kept() -> [u64; 7] {
    use std::sync::atomic::Ordering::Relaxed;
    [
        KEPT[0].load(Relaxed),
        KEPT[1].load(Relaxed),
        KEPT[2].load(Relaxed),
        KEPT[3].load(Relaxed),
        library::BLOB.load(Relaxed),
        library::SAVED.load(Relaxed),
        library::TRIMMED_BYTES.load(Relaxed),
    ]
}

/// The first program name in a container (`CS_`, `PS_`, `VS_`... and a
/// word), as the title's containers carry them, to tell passes apart in the
/// logs.
fn program_name(bytes: &[u8]) -> Option<String> {
    let word = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    (1..bytes.len().saturating_sub(6)).find_map(|i| {
        let at = &bytes[i..];
        let starts = b"CVPMAHDG".contains(&at[0])
            && at[1] == b'S'
            && at[2] == b'_'
            && at[3].is_ascii_alphabetic()
            && !word(bytes[i - 1]);
        if !starts {
            return None;
        }
        let len = at.iter().take(120).take_while(|&&b| word(b)).count();
        (len >= 8).then(|| String::from_utf8_lossy(&at[..len]).into_owned())
    })
}

/// The shared container for `bytes`.
fn shader(stage: &'static str, bytes: &[u8]) -> Arc<Shader> {
    let full = blake3::hash(bytes);
    let mut shaders = SHADERS.lock().unwrap_or_else(|e| e.into_inner());
    shaders
        .get_or_insert_with(HashMap::new)
        .entry(*full.as_bytes())
        .or_insert_with(|| {
            KEPT[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            KEPT[1].fetch_add(bytes.len() as u64, std::sync::atomic::Ordering::Relaxed);
            Arc::new(Shader::new(stage, bytes))
        })
        .clone()
}

impl Shader {
    /// A container of its own, not in the shared table (the pre-build's:
    /// it must not keep every container of the title).
    fn new(stage: &'static str, bytes: &[u8]) -> Shader {
        Shader {
            stage,
            hash: blake3::hash(bytes).to_hex()[..16].to_owned(),
            name: program_name(bytes),
            container: bytes.to_vec(),
            #[cfg(windows)]
            ambient: (stage == "compute").then(|| crate::ambient::role(bytes)).flatten(),
            #[cfg(windows)]
            probe: crate::lightprobe::on().then(|| crate::lightprobe::role(bytes)).flatten(),
            #[cfg(windows)]
            translated: OnceLock::new(),
        }
    }
}

/// A translated shader and the console resources its code reads.
#[cfg(windows)]
pub(crate) struct Translated {
    pub(crate) host: victory_shader::emit::HostShader,
    /// A compute program's threads in a group.
    pub(crate) threads: Option<[u32; 3]>,
    pub(crate) used: Vec<victory_shader::UsedResource>,
    pub(crate) tables: crate::draw::TablePlan,
    /// Whether the shader's code can store to buffer memory ([`stores`]);
    /// true when its code is not known.
    pub(crate) stores: bool,
    /// Whether it can store through raw addresses ([`stores_by_address`]).
    pub(crate) stores_by_address: bool,
    /// Whether it stores through a view it picks from the heap by an index,
    /// once asked (`crate::picked`).
    pub(crate) picks: std::sync::atomic::AtomicU8,
}

#[cfg(windows)]
impl Shader {
    /// The translation, made on first use; a shader that does not
    /// translate is logged once.
    pub(crate) fn translated(&self) -> Option<&Translated> {
        self.translated
            .get_or_init(|| {
                let done = {
                    let _translation = crate::timeline::span("pipeline: translate a shader");
                    translate(&self.container)
                };
                // `VICTORY_SHADER_SAVE=<dir>`: each shader as the title gave it (`.cdxil`) and as
                // translated (`.dxil`), by the hash crash reports name it with
                if let Some(dir) = std::env::var_os("VICTORY_SHADER_SAVE") {
                    let dir = std::path::PathBuf::from(dir);
                    let _ = std::fs::create_dir_all(&dir);
                    let stem = format!("{}.{}", self.hash, self.stage);
                    let _ = std::fs::write(dir.join(format!("{stem}.cdxil")), &self.container);
                    if let Ok(t) = &done {
                        let _ = std::fs::write(dir.join(format!("{stem}.dxil")), &t.host.dxil);
                    }
                }
                if let Ok(t) = &done {
                    KEPT[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    KEPT[3].fetch_add(t.host.dxil.len() as u64, std::sync::atomic::Ordering::Relaxed);
                }
                match &done {
                    Ok(t) => tracing::info!(
                        stage = self.stage,
                        hash = %self.hash,
                        user_data = t.host.user_data.len(),
                        samplers = t.host.samplers.len(),
                        "shader translated"
                    ),
                    Err(e) => tracing::warn!(
                        stage = self.stage,
                        hash = %self.hash,
                        error = %e,
                        "shader does not translate; its pipelines' draws are dropped"
                    ),
                }
                done.ok()
            })
            .as_ref()
    }
}

#[cfg(windows)]
impl Shader {
    /// The console programs of the container; `None` if they are not known.
    pub(crate) fn programs(&self) -> Option<Vec<victory_shader::NativeProgram>> {
        use victory_shader::{Container, FourCC};
        let c = Container::parse(&self.container).ok()?;
        let part = c.part(FourCC::DXIL).or(c.part(FourCC::ACCC))?;
        victory_shader::native_programs(part.data).ok()
    }
}

#[cfg(windows)]
fn translate(container: &[u8]) -> Result<Translated, String> {
    use victory_shader::{Container, FourCC, HwStage};
    // The translation depends on it (a view index past a capped heap).
    crate::descriptor::host::probe_view_limit();
    let c = Container::parse(container).map_err(|e| e.to_string())?;
    let programs = match c.part(FourCC::DXIL).or(c.part(FourCC::ACCC)) {
        Some(part) => victory_shader::native_programs(part.data).map_err(|e| e.to_string())?,
        None => Vec::new(),
    };
    let host = crate::shader_cache::host_shader(container)?;
    let stores = programs.is_empty() || programs.iter().any(stores);
    let threads = programs.iter().find_map(|p| match &p.io {
        victory_shader::native::StageIo::Cs(cs) if cs.threads.iter().all(|&n| n > 0) => Some(cs.threads),
        _ => None,
    });
    let stores_by_address = programs.is_empty() || programs.iter().any(stores_by_address);
    let used = programs
        .into_iter()
        .find(|p| {
            matches!(
                p.stage,
                HwStage::Vs | HwStage::Ps | HwStage::Ms | HwStage::Cs | HwStage::Gs
            )
        })
        .map(|p| p.used)
        .unwrap_or_default();
    let tables = crate::draw::TablePlan::new(&used, &host.root_indexed_views);
    Ok(Translated {
        host,
        threads,
        used,
        tables,
        stores,
        stores_by_address,
        picks: Default::default(),
    })
}

/// The compute pipelines of the console's ray-tracing structure builder
/// (`XboxBvhBuilder.dll`; its programs are the ones with a spilled
/// user-data table) run **all or none**. Its stages feed each other
/// through memory: a stage that runs on what a dropped stage never wrote
/// computes its store offsets from garbage. The structures lie in plain
/// title memory, between the title's own objects, and a root view has no
/// size, so those stores land on the neighbours (run 98, with 14 of 35
/// stages translated: the title died within a second of the first builder
/// submission, on pointers that had become bounding-box bit patterns).
///
/// So a builder dispatch runs only while every builder pipeline the title
/// has created has a host pipeline; the first builder dispatch asks the
/// workers for them all (ahead of the rest) and the builder does not run,
/// none of its stages, until none is still being made (a stage "still being
/// made" would be a dropped stage). It no longer makes them on the spot:
/// that held the title's recording thread for minutes (PalmVIDev #255). While one cannot be made the builder
/// is off; it is looked at again when the title creates another. A stage
/// that could not be bound stops the stages after it for the rest of that
/// frame and the next ([`builder_rests`]); it turned the builder off for the
/// rest of the run at first.
///
/// shortcut: and only with `VICTORY_BVH_BUILDER=1`, until a run has shown
/// the whole builder and the host side of title memory by address
/// (`memory.rs`) working on a card; nothing reads the structures before
/// the programs that walk them translate.
#[cfg(windows)]
struct Builder {
    pipelines: Vec<Arc<make::Pending>>,
    /// Every one of `pipelines` has a host pipeline.
    whole: bool,
    /// One could not be made, of so many (0: none).
    off: usize,
    /// No stage runs before this frame.
    rests: u64,
    /// It waited for pipelines being made since it last ran.
    waited: bool,
}

#[cfg(windows)]
static BUILDER: Mutex<Builder> = Mutex::new(Builder {
    pipelines: Vec::new(),
    whole: false,
    off: 0,
    rests: 0,
    waited: false,
});

/// Whether `shaders` are a pipeline of the structure builder.
#[cfg(windows)]
fn of_builder(shaders: &[Arc<Shader>]) -> bool {
    use victory_shader::{Container, FourCC};
    let [shader] = shaders else { return false };
    if shader.stage != "compute" {
        return false;
    }
    let Ok(c) = Container::parse(&shader.container) else {
        return false;
    };
    c.part(FourCC::DXIL)
        .and_then(|part| victory_shader::native_programs(part.data).ok())
        .is_some_and(|programs| programs.iter().any(victory_shader::spill::is_spilled))
}

/// `VICTORY_BVH_QUARANTINE=1`: what the structure builder's stages store
/// stays in the host's copies and is not copied back into title memory. A
/// check for a first run of the whole builder: a stage that stores where
/// it should not can then not reach the title's own objects (run 98), and
/// the programs that walk the structures read the host's copies anyway.
/// shortcut: not a way to run: the title also reads results back (sizes,
/// counters) and gets none, and a copy's pages are overwritten from title
/// memory when the title writes them.
#[cfg(windows)]
pub(crate) fn builder_quarantined(this: This) -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    is_builder(this) && *ON.get_or_init(|| std::env::var_os("VICTORY_BVH_QUARANTINE").is_some_and(|v| v == "1"))
}

/// Whether the title's pipeline `this` is one of the structure builder's.
#[cfg(windows)]
pub(crate) fn is_builder(this: This) -> bool {
    !this.is_null() && pipeline(this).builder
}

/// A stage of the builder was dropped after all: none runs for the rest of
/// this frame and the next (what it fed is not there).
#[cfg(windows)]
pub(crate) fn builder_rests(why: &'static str) {
    let mut b = BUILDER.lock().unwrap_or_else(|e| e.into_inner());
    b.rests = crate::frame::current() + 2;
    static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 20 {
        tracing::warn!(why, "the ray-tracing structure builder rests for this frame and the next: its stages run all or none");
    }
}

/// At a dispatch of a builder pipeline: whether the builder runs (see
/// [`Builder`]).
#[cfg(windows)]
pub(crate) fn builder_runs() -> bool {
    static ALLOWED: OnceLock<bool> = OnceLock::new();
    if !*ALLOWED.get_or_init(|| std::env::var_os("VICTORY_BVH_BUILDER").is_some_and(|v| v == "1")) {
        return false;
    }
    let mut b = BUILDER.lock().unwrap_or_else(|e| e.into_inner());
    if b.off == b.pipelines.len() || crate::frame::current() < b.rests {
        return false;
    }
    if b.whole {
        return true;
    }
    // Asked for and not waited for (PalmVIDev #255): making them all here,
    // on the title's recording thread under this lock, froze a world run's
    // frame for 5 minutes (53 programs, seconds of driver compile each). The
    // workers make them ahead of the rest; the builder stays off, all of it,
    // until none is being made, and starts at the next frame so that no
    // frame runs its later stages without its earlier ones.
    let states: Vec<Option<bool>> = b.pipelines.iter().map(make::poll).collect();
    if builder_waits(&states) {
        b.waited = true;
        static WAITING: std::sync::Once = std::sync::Once::new();
        WAITING.call_once(|| {
            tracing::info!(
                pipelines = states.len(),
                making = states.iter().filter(|s| s.is_none()).count(),
                "the ray-tracing structure builder waits for its programs' host pipelines (made by the workers, ahead of the rest)"
            );
        });
        return false;
    }
    let missing: Vec<String> = b
        .pipelines
        .iter()
        .zip(&states)
        .filter(|(_, s)| **s != Some(true))
        .map(|(p, _)| make::hashes(p))
        .collect();
    if !missing.is_empty() {
        b.off = b.pipelines.len();
        tracing::warn!(
            missing = missing.len(),
            of = b.pipelines.len(),
            which = ?missing,
            "the ray-tracing structure builder stays off: not all of its programs have a host pipeline, and its stages run all or none"
        );
        return false;
    }
    b.off = 0;
    tracing::info!(
        pipelines = b.pipelines.len(),
        "the ray-tracing structure builder runs: all of its programs have host pipelines"
    );
    b.whole = true;
    if std::mem::take(&mut b.waited) {
        b.rests = crate::frame::current() + 1;
        return false;
    }
    true
}

/// Whether the builder waits for its pipelines ([`make::poll`] of each):
/// some are still being made. Refused ones do not count: those decide
/// whether it runs at all.
#[cfg_attr(not(windows), allow(dead_code))]
fn builder_waits(states: &[Option<bool>]) -> bool {
    states.iter().any(Option::is_none)
}

/// Whether a console program can store to buffer memory: it has a buffer,
/// typed-buffer, scalar-memory, flat or global store or atomic, or code
/// that is not all known (an instruction of those kinds outside our table,
/// an indirect jump, code that does not decode). Stores to textures
/// (image instructions) and to the program's own scratch memory do not
/// count: neither reaches a buffer's host copy. After upstream #413,
/// which leaves scalar-memory stores out and counts image stores.
#[cfg(windows)]
fn stores(program: &victory_shader::NativeProgram) -> bool {
    use victory_shader::gfx10::{Encoding, decode};
    let Ok(code) = decode(&program.code) else {
        return true;
    };
    code.iter().any(|i| {
        let name = i.name();
        match i.encoding {
            Encoding::Mubuf
            | Encoding::Mtbuf
            | Encoding::Smem
            | Encoding::Flat
            | Encoding::Global => name.is_none_or(|n| {
                (n.contains("store") || n.contains("atomic")) && !n.contains("scratch")
            }),
            Encoding::Sop1 => name.is_none_or(|n| matches!(n, "s_setpc_b64" | "s_swappc_b64")),
            _ => false,
        }
    })
}

/// Whether a console program can store to memory through raw addresses
/// (`crate::memory`): a flat or global store or atomic, or code that does
/// not decode. The programs that only walk ray-tracing structures do not,
/// whatever they store through views.
#[cfg(windows)]
fn stores_by_address(program: &victory_shader::NativeProgram) -> bool {
    use victory_shader::gfx10::{Encoding, decode};
    let Ok(code) = decode(&program.code) else {
        return true;
    };
    code.iter().any(|i| {
        matches!(i.encoding, Encoding::Flat | Encoding::Global)
            && i.name()
                .is_none_or(|n| (n.contains("store") || n.contains("atomic")) && !n.contains("scratch"))
    })
}

#[repr(C)]
struct Pipeline {
    object: Object,
    packet: [u32; 8],
    /// Each used stage's container.
    shaders: Vec<Arc<Shader>>,
    /// Whether a list has bound it yet (its shaders are logged then).
    bound: std::sync::atomic::AtomicBool,
    /// What its host pipeline is made from, and the host pipeline once made
    /// (by a worker: the title may release the pipeline meanwhile).
    #[cfg(windows)]
    host: Arc<make::Pending>,
    /// One of the ray-tracing structure builder's ([`Builder`]).
    #[cfg(windows)]
    builder: bool,
}
const _: () = assert!(offset_of!(Pipeline, packet) == 0x10);

/// Stage names, in description order.
const GRAPHICS_STAGES: [&str; 5] = ["vertex", "pixel", "domain", "hull", "geometry"];

fn pipeline<'a>(this: This) -> &'a Pipeline {
    // SAFETY: the title passes one of our pipelines.
    unsafe { &*this.cast::<Pipeline>() }
}

/// List method: logs each pipeline's shaders the first time a list binds
/// it, so a run lists the shaders its frames need first.
pub(crate) fn bound(this: This) {
    if this.is_null() {
        return;
    }
    let p = pipeline(this);
    if p.bound.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let frame = crate::frame::current();
    for shader in &p.shaders {
        tracing::info!(pipeline = ?this, frame, stage = shader.stage, size = shader.container.len(), hash = %shader.hash, "first bind");
    }
}

/// The host pipeline of a title pipeline; `None` if one of its shaders does
/// not translate, the host refuses it, or it is still being made (its first
/// use starts that, see [`make`]).
#[cfg(windows)]
pub(crate) fn host_pipeline<'a>(this: This) -> Option<&'a host::HostPipeline> {
    if this.is_null() {
        return None;
    }
    make::ready(&pipeline(this).host)
}

/// Host pipelines queued and not made yet.
#[cfg(windows)]
pub(crate) fn making() -> u64 {
    make::waiting()
}

/// Draws and dispatches dropped since the last [`take_drops`], by reason
/// (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
static DROPS: Mutex<Option<HashMap<String, u64>>> = Mutex::new(None);

/// Shader hashes as printed in the pipeline diagnostics.
#[cfg(all(windows, feature = "diagnostics"))]
fn parse_skip_hashes(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|hash| !hash.is_empty())
        .filter_map(|hash| {
            if hash.len() == 16 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                Some(hash.to_ascii_lowercase())
            } else {
                tracing::warn!(
                    hash,
                    "VICTORY_SKIP_PIPELINES expects 16-digit shader hashes"
                );
                None
            }
        })
        .collect()
}

#[cfg(all(windows, feature = "diagnostics"))]
fn skipped_shader(p: &Pipeline) -> Option<&str> {
    static HASHES: OnceLock<Vec<String>> = OnceLock::new();
    let hashes = HASHES.get_or_init(|| {
        let value = std::env::var("VICTORY_SKIP_PIPELINES").unwrap_or_default();
        parse_skip_hashes(&value)
    });
    if hashes.is_empty() {
        return None;
    }
    p.shaders
        .iter()
        .find(|shader| hashes.contains(&shader.hash))
        .map(|shader| shader.hash.as_str())
}

#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn is_skipped(this: This) -> bool {
    !this.is_null() && skipped_shader(pipeline(this)).is_some()
}

/// Counts a draw or dispatch of `this` intentionally skipped or dropped
/// for want of a host pipeline (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn note_drop(this: This) {
    let why = if this.is_null() {
        Some("no pipeline set".to_owned())
    } else {
        let p = pipeline(this);
        skipped_shader(p)
            .map(|hash| format!("skipped shader {hash}"))
            .or_else(|| make::missing(&p.host))
    };
    if let Some(why) = why {
        note_why(why);
    }
}

/// Counts a draw or dispatch dropped for `why` (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn note_why(why: String) {
    crate::capture::op(0, || format!("DROPPED {why}"));
    let mut drops = DROPS.lock().unwrap_or_else(|e| e.into_inner());
    *drops.get_or_insert_with(HashMap::new).entry(why).or_default() += 1;
}

/// The drops counted since the last call, most first (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn take_drops() -> Vec<(String, u64)> {
    let drops = DROPS.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut drops: Vec<(String, u64)> = drops.into_iter().flatten().collect();
    // Frame reports only show the first 40 reasons. Keep ties ordered so
    // a HashMap iteration order change cannot change which reasons appear.
    drops.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    drops
}

/// The stage's code, if it has some.
fn code_of(code: &Bytecode) -> Option<*const [u8]> {
    if code.code.is_null() || code.size == 0 {
        return None;
    }
    Some(std::ptr::slice_from_raw_parts(code.code, code.size))
}

/// The stage's container, if it has one.
fn copy(stage: &'static str, code: &Bytecode) -> Option<Arc<Shader>> {
    // SAFETY: the title passes `size` bytes of shader code.
    code_of(code).map(|b| shader(stage, unsafe { &*b }))
}

fn hand_out(
    shaders: Vec<Arc<Shader>>,
    #[cfg(windows)] graphics: Option<Box<host::GraphicsState>>,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    tracing::trace!(?shaders, "pipeline state");
    #[cfg(windows)]
    let host = Arc::new(make::Pending::new(shaders.clone(), graphics));
    #[cfg(windows)]
    known::note(host.key(), &shaders, host.graphics.as_deref());
    #[cfg(windows)]
    make::early(&host);
    #[cfg(windows)]
    let builder = of_builder(&shaders);
    #[cfg(windows)]
    if builder {
        // Another stage: the group is looked at again at its next dispatch.
        let mut b = BUILDER.lock().unwrap_or_else(|e| e.into_inner());
        b.pipelines.push(host.clone());
        b.whole = false;
    }
    let pipeline = Box::new(Pipeline {
        object: Object::new(&FUNCTIONS),
        packet: [0; 8],
        #[cfg(windows)]
        host,
        #[cfg(windows)]
        builder,
        shaders,
        bound: std::sync::atomic::AtomicBool::new(false),
    });
    object::hand_out(PIPELINE, pipeline, iid, &PIPELINE_IIDS, out)
}

/// Device method.
pub(crate) extern "system" fn create_graphics(
    _: This,
    desc: *const GraphicsHead,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    let stages = GRAPHICS_STAGES.iter().zip(&desc.stages);
    let shaders = stages.filter_map(|(s, c)| copy(s, c)).collect();
    hand_out(
        shaders,
        #[cfg(windows)]
        None,
        iid,
        out,
    )
}

/// Device method.
pub(crate) extern "system" fn create_compute(
    _: This,
    desc: *const ComputeHead,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    hand_out(
        copy("compute", &desc.stage).into_iter().collect(),
        #[cfg(windows)]
        None,
        iid,
        out,
    )
}

/// Device method 0x1f0: the plain compute description plus console
/// extensions `(desc, extension count, extensions, iid, out)`.
/// shortcut: the extensions (console shader settings) are ignored, as is
/// the rest of the description.
pub(crate) extern "system" fn create_compute_x(
    this: This,
    desc: *const ComputeHead,
    _extensions: u32,
    _extended: *const u8,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    create_compute(this, desc, iid, out)
}

/// What device method 0x218 is given: a compute pipeline already compiled.
/// Each part is an address and a length in title memory.
#[repr(C)]
pub(crate) struct PrecompiledCompute {
    root: This,
    /// Thread-group size and the two hardware program words (0x1c bytes).
    settings: Bytecode,
    /// A 32-bit hash and a 3 (0x14 bytes; not read).
    id: Bytecode,
    /// GFX10 code, in GPU memory, padded to 64 bytes.
    code: Bytecode,
    /// The descriptors the code reads.
    bindings: Bytecode,
    /// Two pointers, a 64-bit hash and a zero (not read).
    rest: [u64; 4],
}

/// The largest part of a precompiled pipeline taken (the code of the ones
/// seen reaches 0x1f9c0 bytes).
const PRECOMPILED_PART_LIMIT: usize = 4 << 20;

/// The bytes of a precompiled pipeline's part, copied out of title memory;
/// `None` if it is empty, implausibly long or not all readable (the parts
/// lie where the title put them, the code in GPU memory).
fn precompiled_part(part: &Bytecode) -> Option<Vec<u8>> {
    if part.code.is_null() || part.size == 0 || part.size > PRECOMPILED_PART_LIMIT {
        return None;
    }
    let mut bytes = vec![0u8; part.size];
    #[cfg(windows)]
    if !crate::host::read_into(part.code, &mut bytes) {
        return None;
    }
    #[cfg(not(windows))]
    // SAFETY: (tests) the caller passes `size` readable bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(part.code, bytes.as_mut_ptr(), part.size)
    };
    Some(bytes)
}

/// Whether the precompiled-pipeline methods make pipelines
/// (`VICTORY_PRECOMPILED=1`; off by default).
///
/// Device methods 0x1c8 and 0x218 are the title's cache of pipelines it has
/// already compiled. When one fails, the title makes the same pipeline from
/// its shader containers, through the stream or the compute method
/// (PalmVIDev d2bd60ea; our runs agree: every shader a frame binds has a
/// host pipeline while both methods fail). So failing costs no pipeline,
/// and the switch is for measuring what answering from the cache changes.
pub(crate) fn precompiled_wanted() -> bool {
    static WANTED: OnceLock<bool> = OnceLock::new();
    *WANTED.get_or_init(|| std::env::var_os("VICTORY_PRECOMPILED").is_some_and(|v| v == "1"))
}

/// Device method 0x218: a compute pipeline from code the title holds
/// already compiled, `(desc, iid, out)`. Without [`precompiled_wanted`] it
/// fails with `out` null, and the title makes the pipeline from its
/// container instead.
pub(crate) extern "system" fn create_precompiled_compute(
    _: This,
    desc: *const PrecompiledCompute,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    if precompiled_wanted() {
        return precompiled_compute(desc, iid, out);
    }
    static SAID: std::sync::Once = std::sync::Once::new();
    SAID.call_once(|| {
        tracing::info!(
            "precompiled compute pipelines (device method 0x218) are refused: the title makes them from its shader containers (VICTORY_PRECOMPILED=1 makes them from the cache)"
        );
    });
    if !out.is_null() {
        // SAFETY: the title's out pointer.
        unsafe { *out = std::ptr::null() };
    }
    crate::object::E_NOTIMPL
}

/// [`create_precompiled_compute`] when asked for. The program is written
/// into a container as the console compiler stores its own
/// (`victory_shader::precompiled`), and from there on the pipeline is any
/// compute pipeline: translated on first use, cached by its bytes.
/// A description that does not read fails with `out` null, and the title
/// falls back to its container.
fn precompiled_compute(desc: *const PrecompiledCompute, iid: *const Iid, out: *mut This) -> HRESULT {
    use std::sync::atomic::{AtomicU32, Ordering};
    static CALLS: AtomicU32 = AtomicU32::new(0);
    static REFUSED: AtomicU32 = AtomicU32::new(0);
    let calls = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
    if !out.is_null() {
        // SAFETY: the title's out pointer.
        unsafe { *out = std::ptr::null() };
    }
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    let parts = [&desc.settings, &desc.code, &desc.bindings].map(precompiled_part);
    let container = match &parts {
        [Some(settings), Some(code), Some(bindings)] => {
            victory_shader::precompiled::compute_container(settings, code, bindings).map_err(|e| e.to_string())
        }
        _ => Err("a part of the description is missing or not readable".to_owned()),
    };
    match container {
        Ok(container) => {
            if calls <= 4 || calls.is_power_of_two() {
                tracing::info!(calls, code = desc.code.size, "precompiled compute pipeline (device method 0x218)");
            }
            hand_out(
                vec![shader("compute", &container)],
                #[cfg(windows)]
                None,
                iid,
                out,
            )
        }
        Err(why) => {
            let refused = REFUSED.fetch_add(1, Ordering::Relaxed) + 1;
            if refused <= 4 || refused.is_power_of_two() {
                tracing::warn!(
                    calls,
                    refused,
                    why,
                    settings = desc.settings.size,
                    code = desc.code.size,
                    bindings = desc.bindings.size,
                    "precompiled compute pipeline (device method 0x218) not read: the title makes it from its container"
                );
            }
            E_INVALIDARG
        }
    }
}

/// A pipeline described as a stream: its size and the subobjects, each a
/// type then its data, padded to 8 bytes.
#[repr(C)]
pub(crate) struct StreamDesc {
    size: usize,
    stream: *const u8,
}

/// A subobject type's size (type included), and the stage it is if it is a
/// shader. The console numbers amplification and mesh shaders 23 and 24
/// (PC: 24 and 25); its own types from 25 on are not sized yet.
fn subobject(kind: u32) -> Option<(usize, Option<&'static str>)> {
    let stage = |name| Some((24, Some(name)));
    match kind {
        1 => stage("vertex"),
        2 => stage("pixel"),
        3 => stage("domain"),
        4 => stage("hull"),
        5 => stage("geometry"),
        6 => stage("compute"),
        23 => stage("amplification"),
        24 => stage("mesh"),
        0 => Some((16, None)),
        12 | 19 => Some((24, None)),
        7 | 15 => Some((40, None)),
        8 => Some((336, None)),
        10 => Some((48, None)),
        11 => Some((56, None)),
        17 => Some((16, None)),
        21 => Some((64, None)),
        22 => Some((32, None)),
        9 | 13 | 14 | 16 | 18 | 20 => Some((8, None)),
        _ => None,
    }
}

/// Device method: a pipeline from a stream; the shaders are kept, and the
/// graphics state for the host pipeline.
pub(crate) extern "system" fn create_stream(
    _: This,
    desc: *const StreamDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    if desc.stream.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: the title passes `size` bytes of stream, whose pointers point
    // at live shader code and input elements.
    let stream = unsafe { std::slice::from_raw_parts(desc.stream, desc.size) };
    #[cfg_attr(not(windows), allow(unused_variables))]
    let (shaders, graphics) = read_stream(stream, copy);
    hand_out(
        shaders,
        #[cfg(windows)]
        Some(Box::new(graphics)),
        iid,
        out,
    )
}

/// The shaders of a creation stream (each made by `shader` from its stage
/// and bytecode) and the graphics state it describes: what a run keeps of
/// a title's stream, and the pre-build of the pack's (`Prebuild`), by the
/// same code.
#[cfg(windows)]
type StreamState = host::GraphicsState;
#[cfg(not(windows))]
type StreamState = ();

fn read_stream(
    stream: &[u8],
    shader: impl Fn(&'static str, &Bytecode) -> Option<Arc<Shader>>,
) -> (Vec<Arc<Shader>>, StreamState) {
    let mut shaders = Vec::new();
    #[cfg_attr(not(windows), allow(unused_mut))]
    let mut graphics = StreamState::default();
    let mut at = 0;
    while at + 8 <= stream.len() {
        let kind = u32::from_le_bytes(stream[at..at + 4].try_into().expect("4 bytes"));
        let Some((size, stage)) = subobject(kind) else {
            tracing::warn!(
                kind,
                at,
                "pipeline subobject of unknown size; the rest is ignored"
            );
            break;
        };
        if let Some(stage) = stage {
            // SAFETY: a shader subobject has its bytecode at +8 (a stream
            // of ours need not be aligned).
            let code = unsafe { stream.as_ptr().add(at + 8).cast::<Bytecode>().read_unaligned() };
            shaders.extend(shader(stage, &code));
        }
        #[cfg(windows)]
        // SAFETY: a whole subobject of this kind lies at `at`.
        unsafe {
            graphics.read(kind, stream.as_ptr().add(at))
        };
        at += size;
    }
    (shaders, graphics)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &PIPELINE_IIDS).then_some(this);
    object::answer(PIPELINE, iid, found, out)
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Pipeline>(this)
}

/// IUnknown 3, object 4, device child 1, pipeline 3.
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<PIPELINE, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    Table(f)
};

#[cfg(windows)]
pub(crate) mod host;
#[cfg(windows)]
pub mod known;
#[cfg(windows)]
mod library;
#[cfg(windows)]
mod make;

/// Lets the pipeline library's mapped file go from memory when no pipeline
/// has loaded for a while (`library::trim`); called by the host's sweep.
#[cfg(windows)]
pub(crate) fn trim_library() {
    library::trim();
}

/// The pipeline library's part of the report when the host device is
/// removed (`library::removal_note`).
#[cfg(windows)]
pub(crate) fn library_removal_note() -> String {
    library::removal_note()
}

/// The folder the pipeline library is kept in, if a cache folder is set.
#[cfg(windows)]
pub(crate) fn cache_folder() -> Option<std::path::PathBuf> {
    library::folder()
}

/// Keeps compiled host pipelines, and the list of title pipelines to start
/// early, in `dir` (a folder of the cache) across runs.
#[cfg(windows)]
pub fn use_pipeline_cache(dir: std::path::PathBuf) {
    library::use_pipeline_cache(dir);
}

/// Saves the pipeline cache now if anything was added (at the end of a run).
#[cfg(windows)]
pub fn save_pipeline_cache() {
    library::save(true);
    known::flush();
}

/// The pre-build (`victory translate --cache --pipelines`): a title
/// pipeline made ahead of a run into the pipeline library
/// (`use_pipeline_cache`), by the code a run makes it with
/// ([`host::HostPipeline::new`], whose library name is the hash of the
/// host description: the translated code, our root signature, the state),
/// so the run loads it under the same name instead of making it.
#[cfg(windows)]
pub struct Prebuild {
    shaders: Vec<Arc<Shader>>,
    graphics: Option<Box<host::GraphicsState>>,
}

// SAFETY: as `make::Pending`: the graphics state's input elements point
// into its own names, and it is only read.
#[cfg(windows)]
unsafe impl Send for Prebuild {}

#[cfg(windows)]
impl Prebuild {
    /// The pipeline a creation stream describes, as the title passes it to
    /// the device (`create_stream`).
    ///
    /// # Safety
    /// The stream's pointers point at live shader code and input elements.
    pub unsafe fn graphics(stream: &[u8]) -> Prebuild {
        // Shaders of their own: a pre-build must not keep every container.
        let own = |stage, code: &Bytecode| {
            // SAFETY: the caller's pointers.
            code_of(code).map(|b| Arc::new(Shader::new(stage, unsafe { &*b })))
        };
        let (shaders, graphics) = read_stream(stream, own);
        Prebuild {
            shaders,
            graphics: Some(Box::new(graphics)),
        }
    }

    /// The compute pipeline of a container (`create_compute`).
    pub fn compute(container: &[u8]) -> Prebuild {
        Prebuild {
            shaders: vec![Arc::new(Shader::new("compute", container))],
            graphics: None,
        }
    }

    /// What a run lists the pipeline by (its early list).
    pub fn key(&self) -> String {
        make::key(&self.shaders, self.graphics.as_deref())
    }

    /// Its shaders and state, as a run describes them.
    pub fn describe(&self) -> String {
        let state = self.graphics.as_ref().map(|g| g.describe()).unwrap_or_default();
        format!("{:?}\t{state}", self.shaders)
    }

    /// Makes the host pipeline, or finds it in the library; why not if it
    /// gets none.
    pub fn make(&self) -> Result<(), String> {
        let key = self.key();
        // Without the run's log lines per shader and pipeline: tens of
        // thousands of them would bury the pre-build's own report.
        let quiet = tracing::Dispatch::none();
        let made = tracing::dispatcher::with_default(&quiet, || {
            host::HostPipeline::new(&self.shaders, self.graphics.as_deref())
        });
        if made.is_some() {
            library::note_prebuilt(key);
        }
        library::save(false);
        match made {
            Some(_) => Ok(()),
            None => Err(match self.shaders.iter().find(|s| s.translated().is_none()) {
                Some(s) => format!("{} shader does not translate", s.stage),
                None => "the host refuses it".into(),
            }),
        }
    }
}

/// Readies a pre-build: the host device (whose wave size the translator
/// takes), the clamp of its view heaps and their heap map (as a run sets
/// them before its first translation), the library, the early list (so a
/// save keeps it) and the pre-build's own list. Whether there is a device.
#[cfg(windows)]
pub fn start_prebuild() -> bool {
    crate::descriptor::host::probe_view_limit();
    if crate::host::host().is_none() {
        return false;
    }
    library::start_prebuild();
    true
}

/// Host pipelines made, and loaded from the library, since the start.
#[cfg(windows)]
pub fn pipeline_counts() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (library::MADE.load(Relaxed), library::LOADED.load(Relaxed))
}

/// Whether a pre-build has put the title pipeline `key` in the library as
/// it is now (a second pre-build skips it without translating or loading
/// anything).
#[cfg(windows)]
pub fn prebuilt(key: &str) -> bool {
    library::prebuilt(key)
}

/// Whether a run drew or dispatched with the title pipeline `key` (the
/// early list): those are made first.
#[cfg(windows)]
pub fn wanted_early(key: &str) -> bool {
    library::wanted_early(key)
}

#[cfg(test)]
mod tests {
    /// The builder waits (does not run, and is not turned off) while any of
    /// its pipelines is still being made; made and refused ones decide.
    #[test]
    fn the_builder_waits_for_pipelines_being_made() {
        assert!(super::builder_waits(&[Some(true), None, Some(true)]));
        assert!(super::builder_waits(&[None, Some(false)]));
        assert!(!super::builder_waits(&[Some(true), Some(true)]));
        assert!(!super::builder_waits(&[Some(true), Some(false)]));
        assert!(!super::builder_waits(&[]));
    }

    #[cfg(all(windows, feature = "diagnostics"))]
    #[test]
    fn skip_hashes_accepts_full_hashes_only() {
        assert!(super::parse_skip_hashes(" , ").is_empty());
        assert_eq!(
            super::parse_skip_hashes(" 0123456789ABCDEF, 0123456789abcde, xyz, 0011223344556677,"),
            ["0123456789abcdef", "0011223344556677"]
        );
    }

    use super::*;
    use crate::object::{S_OK, slot};

    fn shaders(p: This) -> Vec<(&'static str, Vec<u8>)> {
        pipeline(p)
            .shaders
            .iter()
            .map(|s| (s.stage, s.container.clone()))
            .collect()
    }

    #[test]
    fn pipelines_keep_their_shaders() {
        let (vs, ps) = ([1u8, 2, 3], [4u8; 5]);
        let none = || Bytecode {
            code: std::ptr::null(),
            size: 0,
        };
        let desc = GraphicsHead {
            root: std::ptr::null(),
            stages: [
                Bytecode {
                    code: vs.as_ptr(),
                    size: vs.len(),
                },
                Bytecode {
                    code: ps.as_ptr(),
                    size: ps.len(),
                },
                none(),
                none(),
                none(),
            ],
        };
        let mut p = std::ptr::null();
        assert_eq!(
            create_graphics(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut p),
            S_OK
        );
        assert_eq!(
            shaders(p),
            [("vertex", vs.to_vec()), ("pixel", ps.to_vec())]
        );
        // A second pipeline with the same pixel shader shares its container.
        let mut q = std::ptr::null();
        create_graphics(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut q);
        assert!(Arc::ptr_eq(
            &pipeline(p).shaders[1],
            &pipeline(q).shaders[1]
        ));
        assert_eq!(slot::<extern "system" fn(This) -> u32>(p, 0x10)(p), 0);
    }

    /// Device method 0x218 with the first call a run logged (2026-11-25):
    /// the pipeline holds one compute shader, a container the translator
    /// reads the program from, with its 8x8 thread group.
    #[test]
    fn a_precompiled_compute_pipeline_keeps_its_program() {
        fn hex(s: &str) -> Vec<u8> {
            (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect()
        }
        let settings = hex("00cf05400201096c00000000080008000100000043032c408a0b0000");
        // `s_mov_b32 s8, s4; s_mov_b32 s9, 0; s_load_dwordx8 s[16:23],
        // s[8:9]; s_endpgm`.
        let code = hex("0403881b800389be040308f4000000fa000081bf");
        let mut bindings = Vec::new();
        for v in [0x403u32, 0x44, 0x60, 0, 0, 0x0001_0440] {
            bindings.extend_from_slice(&v.to_le_bytes());
        }
        // A texture at dword 0 of the table in user-data dword 4.
        for v in [0x41u32, 0x300, 0, 0x10a, 4, 0x408, u32::MAX, 0] {
            bindings.extend_from_slice(&v.to_le_bytes());
        }
        bindings.extend_from_slice(&0x43u32.to_le_bytes());
        bindings.extend_from_slice(&[0; 28]);
        let part = |b: &[u8]| Bytecode { code: b.as_ptr(), size: b.len() };
        let desc = PrecompiledCompute {
            root: std::ptr::null(),
            settings: part(&settings),
            id: Bytecode { code: std::ptr::null(), size: 0 },
            code: part(&code),
            bindings: part(&bindings),
            rest: [0; 4],
        };
        let mut p = std::ptr::null();
        let hr = precompiled_compute(&desc, &PIPELINE_IIDS[0], &mut p);
        assert_eq!(hr, S_OK);
        let [shader] = pipeline(p).shaders.as_slice() else { panic!("one shader") };
        assert_eq!(shader.stage, "compute");
        let c = victory_shader::Container::parse(&shader.container).unwrap();
        let programs = victory_shader::native_programs(c.part(victory_shader::FourCC::DXIL).unwrap().data).unwrap();
        assert_eq!(programs[0].code, code);
        assert_eq!(programs[0].layout[0].user_data, 4);
        // A description whose binding table does not read fails, `out` null.
        let desc = PrecompiledCompute { bindings: part(&settings), ..desc };
        let mut q = std::ptr::dangling();
        let hr = precompiled_compute(&desc, &PIPELINE_IIDS[0], &mut q);
        assert_eq!(hr, E_INVALIDARG);
        assert!(q.is_null());
        // Not asked for (the default): refused, so that the title makes the
        // pipeline from its container.
        let mut q = std::ptr::dangling();
        let hr = create_precompiled_compute(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut q);
        assert_eq!(hr, crate::object::E_NOTIMPL);
        assert!(q.is_null());
    }

    /// The stream layout the title builds (title+0x86f090): root
    /// signature, five shader slots (the first a mesh shader), blend, sample
    /// mask, rasterizer, depth-stencil 1, sample description, input layout,
    /// topology, render target and depth formats: 0x2b0 bytes.
    #[test]
    fn streams_yield_their_shaders() {
        let (ms, ps) = ([9u8; 7], [4u8; 5]);
        let mut stream = vec![0u8; 0x2b0];
        let mut put = |at: usize, v: u64| stream[at..at + 8].copy_from_slice(&v.to_le_bytes());
        for (at, kind) in [
            (0x10, 24),
            (0x28, 2),
            (0x40, 3),
            (0x58, 4),
            (0x70, 5),
            (0x88, 8),
            (0x1d8, 9),
            (0x1e0, 10),
            (0x210, 21),
            (0x250, 17),
            (0x260, 12),
            (0x278, 14),
            (0x280, 15),
            (0x2a8, 16),
        ] {
            put(at, kind);
        }
        put(0x18, ms.as_ptr() as u64);
        put(0x20, ms.len() as u64);
        put(0x30, ps.as_ptr() as u64);
        put(0x38, ps.len() as u64);
        // Topology type 3 (triangles), one render target of format 28.
        put(0x278, 14 | 3 << 32);
        put(0x280, 15 | 28 << 32);
        put(0x2a0, 1 << 32);
        let desc = StreamDesc {
            size: stream.len(),
            stream: stream.as_ptr(),
        };
        let mut p = std::ptr::null();
        let hr = create_stream(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut p);
        assert_eq!(hr, S_OK);
        assert_eq!(shaders(p), [("mesh", ms.to_vec()), ("pixel", ps.to_vec())]);
        #[cfg(windows)]
        {
            let g = pipeline(p).host.graphics.as_deref().unwrap();
            assert_eq!(g.topology.0, 3);
            assert_eq!(
                (g.targets.NumRenderTargets, g.targets.RTFormats[0].0),
                (1, 28)
            );
        }
    }
}
