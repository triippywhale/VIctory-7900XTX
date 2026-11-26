//! The table of everything we implement.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::LazyLock;

use crate::modules;

/// One function we implement, under the name the title imports (`#n` for an
/// ordinal).
#[derive(Debug, Clone, Copy)]
pub struct HleExport {
    pub name: &'static str,
    pub address: usize,
}

/// A group of implementations and the DLL names they answer for.
#[derive(Debug)]
pub struct HleModule {
    /// Human-readable name, e.g. "kernel".
    pub name: &'static str,
    /// Lower-case DLL names served. A trailing `*` matches a prefix, which
    /// covers versioned API sets (`api-ms-win-core-libraryloader-l1-2-0.dll`).
    pub dlls: &'static [&'static str],
    pub exports: Vec<HleExport>,
}

impl HleModule {
    pub fn serves(&self, dll: &str) -> bool {
        let dll = dll.to_ascii_lowercase();
        self.dlls
            .iter()
            .any(|pattern| match pattern.strip_suffix('*') {
                Some(prefix) => dll.starts_with(prefix),
                None => dll == *pattern,
            })
    }
}

/// Declares an export whose Rust name equals the imported name.
#[macro_export]
macro_rules! hle_export {
    ($f:ident) => {
        $crate::HleExport {
            name: stringify!($f),
            address: $f as *const () as usize,
        }
    };
}

static REGISTRY: LazyLock<Vec<HleModule>> = LazyLock::new(modules::all);

/// All modules we implement.
pub fn registry() -> &'static [HleModule] {
    &REGISTRY
}

/// FNV-1a: the keys are short export names built into the binary, so the
/// default hasher's resistance to chosen keys buys nothing here, and FNV
/// hashes them in a fraction of its time.
#[derive(Clone, Copy)]
struct Fnv(u64);

impl Default for Fnv {
    fn default() -> Self {
        Fnv(0xCBF2_9CE4_8422_2325)
    }
}

impl Hasher for Fnv {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x100_0000_01B3);
        }
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

type Map<V> = HashMap<&'static str, V, BuildHasherDefault<Fnv>>;

/// Whether `pattern` (lower case, a trailing `*` matching a prefix) names
/// `dll` in any case, as [`HleModule::serves`] decides it.
fn matches(pattern: &str, dll: &str) -> bool {
    match pattern.strip_suffix('*') {
        Some(prefix) => dll
            .as_bytes()
            .get(..prefix.len())
            .is_some_and(|start| start.eq_ignore_ascii_case(prefix.as_bytes())),
        None => dll.eq_ignore_ascii_case(pattern),
    }
}

/// The registry's exports by name, for [`lookup`].
struct Index {
    modules: &'static [HleModule],
    /// Export name to (module, address) in registry order: the first export
    /// of the name in each module that has one, as a scan finds it.
    exports: Map<Vec<(usize, usize)>>,
}

impl Index {
    fn new(modules: &'static [HleModule]) -> Index {
        let mut exports = Map::<Vec<(usize, usize)>>::default();
        for (i, m) in modules.iter().enumerate() {
            for e in &m.exports {
                let owners = exports.entry(e.name).or_default();
                if owners.last().is_none_or(|&(m, _)| m != i) {
                    owners.push((i, e.address));
                }
            }
        }
        Index { modules, exports }
    }

    /// The address of `name` in the first module, in registry order, that
    /// serves `dll` and exports it. Only the modules exporting `name` (one,
    /// nearly always) test their DLL patterns.
    fn find(&self, dll: &str, name: &str) -> Option<usize> {
        self.exports
            .get(name)?
            .iter()
            .find(|&&(m, _)| self.modules[m].dlls.iter().any(|p| matches(p, dll)))
            .map(|&(_, address)| address)
    }
}

static INDEX: LazyLock<Index> = LazyLock::new(|| Index::new(registry()));

/// The address we bind `dll!name` to (`#n` for an ordinal), if a module
/// serving `dll` implements it. Called once per title import at load and on
/// every `GetProcAddress` of a host module.
pub(crate) fn lookup(dll: &str, name: &str) -> Option<usize> {
    INDEX.find(dll, name)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`lookup`] as it was: walking every module and export.
    fn scan(modules: &[HleModule], dll: &str, name: &str) -> Option<usize> {
        modules
            .iter()
            .filter(|m| m.serves(dll))
            .flat_map(|m| m.exports.iter())
            .find(|e| e.name == name)
            .map(|e| e.address)
    }

    #[test]
    fn export_names_are_unique_per_dll_pattern() {
        for module in registry() {
            let mut names: Vec<_> = module.exports.iter().map(|e| e.name).collect();
            names.sort_unstable();
            let before = names.len();
            names.dedup();
            assert_eq!(
                before,
                names.len(),
                "duplicate export in module {}",
                module.name
            );
        }
    }

    /// DLL names that reach every pattern of `modules`: exact names as
    /// written and upper-cased, prefixes with a version suffix, and names no
    /// module serves.
    fn dll_names(modules: &[HleModule]) -> Vec<String> {
        let mut names = vec![
            "nothing.dll".to_owned(),
            "api-ms-win-core-unserved-l1-1-0.dll".to_owned(),
            String::new(),
            "x".repeat(200),
        ];
        for m in modules {
            for pattern in m.dlls {
                let name = match pattern.strip_suffix('*') {
                    Some(prefix) => format!("{prefix}l1-2-0.dll"),
                    None => (*pattern).to_owned(),
                };
                names.push(name.to_ascii_uppercase());
                names.push(name);
            }
        }
        names
    }

    fn export_names(modules: &[HleModule]) -> Vec<&'static str> {
        let mut names: Vec<_> = modules
            .iter()
            .flat_map(|m| m.exports.iter().map(|e| e.name))
            .collect();
        names.extend(["NotExported", "#999", ""]);
        names
    }

    fn agrees_with_scan(modules: &'static [HleModule]) -> usize {
        let index = Index::new(modules);
        let mut checked = 0;
        for dll in dll_names(modules) {
            for name in export_names(modules) {
                assert_eq!(
                    index.find(&dll, name),
                    scan(modules, &dll, name),
                    "{dll}!{name}"
                );
                checked += 1;
            }
        }
        checked
    }

    #[test]
    fn index_agrees_with_a_scan_on_the_registry() {
        assert!(agrees_with_scan(registry()) > 1000);
        let ours = lookup("XGameRuntime.dll", "XGameRuntimeInitialize");
        assert!(ours.is_some());
        assert_eq!(
            ours,
            scan(registry(), "XGameRuntime.dll", "XGameRuntimeInitialize")
        );
    }

    fn export(name: &'static str, address: usize) -> HleExport {
        HleExport { name, address }
    }

    /// Overlapping exact and prefix patterns, a name two modules export and
    /// a name one module exports twice.
    fn overlapping() -> &'static [HleModule] {
        Box::leak(Box::new([
            HleModule {
                name: "files",
                dlls: &["kernel32.dll", "api-ms-win-core-file-*"],
                exports: vec![
                    export("CreateFileW", 1),
                    export("ReadFile", 2),
                    export("ReadFile", 3),
                ],
            },
            HleModule {
                name: "kernel",
                dlls: &["kernel32.dll", "api-ms-win-core-*"],
                exports: vec![export("CreateFileW", 4), export("Sleep", 5)],
            },
            HleModule {
                name: "empty",
                dlls: &[],
                exports: vec![export("Sleep", 6)],
            },
            HleModule {
                name: "late",
                dlls: &["api-ms-win-core-file-l1-2-0.dll"],
                exports: vec![export("GetFileType", 7), export("Sleep", 8)],
            },
        ]))
    }

    #[test]
    fn the_first_serving_module_and_first_export_win() {
        let modules = overlapping();
        assert!(agrees_with_scan(modules) > 0);
        let index = Index::new(modules);
        let find = |dll: &str, name| index.find(dll, name);
        assert_eq!(find("KERNEL32.DLL", "CreateFileW"), Some(1));
        assert_eq!(find("kernel32.dll", "ReadFile"), Some(2));
        assert_eq!(find("kernel32.dll", "Sleep"), Some(5));
        assert_eq!(find("api-ms-win-core-file-l1-2-0.dll", "Sleep"), Some(5));
        assert_eq!(
            find("api-ms-win-core-file-l1-2-0.dll", "GetFileType"),
            Some(7)
        );
        assert_eq!(find("api-ms-win-core-file-l1-1-0.dll", "GetFileType"), None);
        assert_eq!(
            find("api-ms-win-core-synch-l1-1-0.dll", "CreateFileW"),
            Some(4)
        );
        assert_eq!(find("api-ms-win-core-", "Sleep"), Some(5));
        assert_eq!(find("kernel32", "Sleep"), None);
    }

    #[test]
    fn patterns_match_in_any_case_as_serves_does() {
        let patterns = ["kernel32.dll", "api-ms-win-core-file-*", "", "*"];
        let dlls = [
            "KERNEL32.DLL",
            "kernel32.dll",
            "kernel32",
            "kernel32.dll2",
            "Api-MS-Win-Core-File-L1-2-0.dll",
            "api-ms-win-core-file-",
            "api-ms-win-core-fil",
            "KÉRNEL32.DLL",
            "",
        ];
        for pattern in patterns {
            let m = HleModule {
                name: "t",
                dlls: Box::leak(Box::new([pattern])),
                exports: vec![],
            };
            for dll in dlls {
                assert_eq!(matches(pattern, dll), m.serves(dll), "{pattern} {dll}");
            }
        }
    }

    #[test]
    fn registry_patterns_are_lower_case() {
        for m in registry() {
            for pattern in m.dlls {
                assert_eq!(*pattern, pattern.to_ascii_lowercase(), "module {}", m.name);
            }
        }
    }

    #[test]
    fn many_modules_exporting_one_name_keep_registry_order() {
        let many: &'static [HleModule] = Box::leak(
            (0..300)
                .map(|i| HleModule {
                    name: "m",
                    dlls: if i % 2 == 0 {
                        &["even.dll"]
                    } else {
                        &["odd.dll", "many.dll"]
                    },
                    exports: vec![export("F", i)],
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
        let index = Index::new(many);
        for (dll, first) in [
            ("EVEN.dll", Some(0)),
            ("odd.DLL", Some(1)),
            ("many.dll", Some(1)),
            ("none.dll", None),
        ] {
            assert_eq!(index.find(dll, "F"), first);
            assert_eq!(scan(many, dll, "F"), first);
        }
    }

    /// Import binding throughput: every registry export looked up under
    /// each DLL name that serves it, as a title's import table does, plus
    /// a miss per DLL that falls through to the host. Run with
    /// `cargo test --release -p victory-hle registry_lookup_throughput -- --ignored --nocapture`.
    #[test]
    #[ignore = "benchmark"]
    fn registry_lookup_throughput() {
        let modules = registry();
        let mut imports = Vec::new();
        for m in modules {
            for pattern in m.dlls {
                let dll = match pattern.strip_suffix('*') {
                    Some(prefix) => format!("{}L1-1-0.dll", prefix.to_ascii_uppercase()),
                    None => pattern.to_ascii_uppercase(),
                };
                for e in &m.exports {
                    imports.push((dll.clone(), e.name.to_owned()));
                }
                imports.push((dll.clone(), "HostOnlyFunction".to_owned()));
            }
        }
        let _ = lookup("warm.dll", "up");
        let rounds = 200;
        let time = |f: &dyn Fn(&str, &str) -> Option<usize>| {
            let mut best = f64::MAX;
            let mut sum = 0usize;
            for _ in 0..5 {
                let started = std::time::Instant::now();
                for _ in 0..rounds {
                    for (dll, name) in &imports {
                        let found = std::hint::black_box(f(dll, name));
                        sum = sum.wrapping_add(found.unwrap_or(1));
                    }
                }
                let ns = started.elapsed().as_nanos() as f64 / (rounds * imports.len()) as f64;
                best = best.min(ns);
            }
            (best, sum)
        };
        let (before, a) = time(&|dll, name| scan(modules, dll, name));
        let (after, b) = time(&|dll, name| lookup(dll, name));
        assert_eq!(a, b);
        println!(
            "{} modules, {} imports: scan {before:.0} ns, index {after:.0} ns per import ({:.1}x)",
            modules.len(),
            imports.len(),
            before / after
        );
    }

    #[test]
    fn api_set_prefix_matching() {
        let m = HleModule {
            name: "t",
            dlls: &["api-ms-win-core-libraryloader-*"],
            exports: vec![],
        };
        assert!(m.serves("API-MS-WIN-CORE-LIBRARYLOADER-L1-2-0.dll"));
        assert!(!m.serves("api-ms-win-core-file-l1-1-0.dll"));
    }
}
