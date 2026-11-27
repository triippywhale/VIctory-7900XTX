//! Our own helper shaders, ready before their first use.
//!
//! The layer has passes of its own beside the title's pipelines: the
//! indirect-argument clamps (`crate::indirect`), the changed words sent to
//! the card (`host/send.rs`), the card compare, the stand-in for the
//! indirect light (`crate::ambient`), the two-plane composer, the depth-plane
//! pass, the count that keeps the card awake. Each is written as HLSL in its
//! module, compiled by the system's compiler and made into a host pipeline
//! on first use: on whichever thread came to need it first, a title's
//! recording thread or the submission thread, and again in every run (the
//! title's pipelines have the shader cache and the pipeline library; these
//! had neither).
//!
//! Two things here:
//!
//! 1. [`read`] and [`write`]: the compiled code kept on disk beside the
//!    pipeline library (`<cache>/pipelines/helpers`), named by a hash of the
//!    source with what was defined before it, the entry and the target. A
//!    changed source has another name; a file that does not hold what its
//!    first 32 bytes say (the BLAKE3 of the rest) is compiled again and
//!    written over. `VICTORY_HELPER_CODE=0`: compiled in every run, as
//!    before.
//! 2. [`ahead`]: once the title has created its device and the host device
//!    exists, a thread of ours makes the pipelines of the passes this run's
//!    settings use, while the title still loads. A thread that needs one
//!    meanwhile waits for that one as it would have made it itself.
//!    `VICTORY_HELPERS_AHEAD=0`: each on first use, as before.

use std::path::{Path, PathBuf};

/// The form of the files, part of their names.
const FORM: &[u8] = b"victory helper code 1";

/// The name compiled code is kept under: `source` (with its defines)
/// compiled from `entry` for `target`.
pub(crate) fn key(source: &str, entry: &[u8], target: &[u8]) -> String {
    let mut hash = blake3::Hasher::new();
    for part in [FORM, source.as_bytes(), entry, target] {
        hash.update(&(part.len() as u64).to_le_bytes());
        hash.update(part);
    }
    hash.finalize().to_hex().to_string()
}

fn file(dir: &Path, key: &str) -> PathBuf {
    dir.join(format!("{key}.bin"))
}

/// The code kept under `key`, if the file holds all of it.
pub(crate) fn read(dir: &Path, key: &str) -> Option<Vec<u8>> {
    let bytes = std::fs::read(file(dir, key)).ok()?;
    let (sum, code) = bytes.split_first_chunk::<32>()?;
    (!code.is_empty() && blake3::hash(code).as_bytes() == sum).then(|| code.to_vec())
}

/// Keeps `code` under `key`: written beside and renamed, so that a reader
/// never sees a part (two threads, or a pre-build beside a run, may write
/// the same one). Not kept if it cannot be written.
pub(crate) fn write(dir: &Path, key: &str, code: &[u8]) {
    static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let beside = dir.join(format!("{key}.{}-{n}.tmp", std::process::id()));
    let bytes = [&blake3::hash(code).as_bytes()[..], code].concat();
    let written = std::fs::create_dir_all(dir)
        .and_then(|()| std::fs::write(&beside, bytes))
        .and_then(|()| std::fs::rename(&beside, file(dir, key)));
    if let Err(e) = written {
        let _ = std::fs::remove_file(&beside);
        tracing::debug!(error = %e, key, "a helper shader's code is not kept");
    }
}

/// Where the code is kept; `None` with no cache folder or with
/// `VICTORY_HELPER_CODE=0`.
#[cfg(windows)]
pub(crate) fn folder() -> Option<PathBuf> {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_HELPER_CODE").is_none_or(|v| v != "0")) {
        return None;
    }
    Some(crate::pipeline::cache_folder()?.join("helpers"))
}

/// Compiles found on disk and made, since the start.
#[cfg(windows)]
pub(crate) static COUNTS: [std::sync::atomic::AtomicU32; 2] =
    [const { std::sync::atomic::AtomicU32::new(0) }; 2];

/// The title has created its device: the passes are made ahead as soon as
/// there is a host device ([`start`]).
#[cfg(windows)]
static WANTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// At the title's device creation.
#[cfg(windows)]
pub(crate) fn ahead() {
    WANTED.store(true, std::sync::atomic::Ordering::Relaxed);
    if crate::host::host_made() {
        start();
    }
}

/// At the host device's creation, and at the title's ([`ahead`]): whichever
/// comes last starts the thread, once. A pre-build, the offline commands and
/// the tests create no title device, and make nothing ahead.
#[cfg(windows)]
pub(crate) fn start() {
    use std::sync::atomic::Ordering::Relaxed;
    static ONCE: std::sync::Once = std::sync::Once::new();
    if cfg!(test) || !WANTED.load(Relaxed) {
        return;
    }
    ONCE.call_once(|| {
        if std::env::var_os("VICTORY_HELPERS_AHEAD").is_some_and(|v| v == "0") {
            return;
        }
        let work = || {
            let began = std::time::Instant::now();
            // (Waits here while the host device is still being made.)
            let Some(host) = crate::host::host() else {
                return;
            };
            crate::indirect::ahead();
            host.send_ahead();
            if crate::shadow::cards() {
                host.card_ready();
            }
            crate::ambient::ahead();
            // Their pipelines need what only their first use knows (the
            // presented format, a depth format): the code alone.
            crate::compose::ahead();
            crate::host::planes_ahead();
            tracing::info!(
                ms = began.elapsed().as_millis() as u64,
                found = COUNTS[0].load(Relaxed),
                compiled = COUNTS[1].load(Relaxed),
                "our own shaders' passes made ahead of their first use"
            );
        };
        if let Err(e) = std::thread::Builder::new().name("victory-helpers".into()).spawn(work) {
            tracing::debug!(error = %e, "no thread to make our own passes ahead: each on first use");
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_is_kept_by_source_entry_and_target() {
        let dir = tempfile::tempdir().unwrap();
        let k = key("void main() {}", b"main", b"cs_5_1");
        assert_eq!(read(dir.path(), &k), None);
        write(dir.path(), &k, &[1, 2, 3]);
        assert_eq!(read(dir.path(), &k), Some(vec![1, 2, 3]));
        // Another source, entry or target is another file.
        let others = [
            key("void main() { }", b"main", b"cs_5_1"),
            key("void main() {}", b"mean", b"cs_5_1"),
            key("void main() {}", b"main", b"ps_5_1"),
            // (Lengths are part of the name: no part runs into the next.)
            key("void main() {}m", b"ain", b"cs_5_1"),
        ];
        for other in &others {
            assert_ne!(other, &k);
            assert_eq!(read(dir.path(), other), None);
        }
        // Nothing left beside it.
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn a_file_that_is_not_whole_is_not_served() {
        let dir = tempfile::tempdir().unwrap();
        let k = key("s", b"e", b"t");
        write(dir.path(), &k, &[7; 100]);
        let path = file(dir.path(), &k);
        let whole = std::fs::read(&path).unwrap();
        // Cut short, a byte changed, the sum alone, nothing.
        let mut changed = whole.clone();
        changed[60] ^= 1;
        for bad in [&whole[..90], &changed[..], &whole[..32], &[][..]] {
            std::fs::write(&path, bad).unwrap();
            assert_eq!(read(dir.path(), &k), None);
        }
        // Written over, it is served again.
        write(dir.path(), &k, &[7; 100]);
        assert_eq!(read(dir.path(), &k), Some(vec![7; 100]));
    }

    #[test]
    fn a_folder_that_cannot_be_made_keeps_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let taken = dir.path().join("helpers");
        std::fs::write(&taken, b"a file where the folder would be").unwrap();
        write(&taken, "k", &[1]);
        assert_eq!(read(&taken, "k"), None);
    }
}
