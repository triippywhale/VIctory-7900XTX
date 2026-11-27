//! What makes the title's rays test many nodes, from a memory dump
//! (`VICTORY_MEMORY_DUMP=<folder>`, then a file `go` there):
//!
//! ```text
//! cargo run --release -p victory-shader --example bvh_dump -- <folder> [--all]
//! ```
//!
//! Finds the structures' headers in the pages the dump holds and checks
//! every top level (and, with `--all`, every bottom level too) with
//! `victory_shader::bvh_check`: broken pointers (cycles, nodes reached
//! twice, nodes outside the structure, nodes the dump does not have), loose
//! and escaping boxes, and the surface estimate of the node tests a ray
//! through the root makes. Then the bottom levels that cost the top levels
//! the most.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use victory_shader::bvh_check::{self, Header, Memory, PAGE, Report};

/// The dump: arenas, and where each title page is in them.
struct Dump {
    arenas: Vec<Vec<u8>>,
    pages: HashMap<u64, (usize, usize)>,
    walked: BTreeSet<u64>,
    built: Vec<u64>,
    packed: Vec<(u64, u64)>,
}

impl Memory for Dump {
    fn page(&self, number: u64) -> Option<&[u8]> {
        let &(arena, at) = self.pages.get(&number)?;
        self.arenas.get(arena)?.get(at..at + PAGE as usize)
    }
}

fn number(s: &str) -> Option<u64> {
    match s.strip_prefix("0x") {
        Some(h) => u64::from_str_radix(h, 16).ok(),
        None => s.parse().ok(),
    }
}

fn load(folder: &Path) -> std::io::Result<Dump> {
    let text = std::fs::read_to_string(folder.join("pages.txt"))?;
    let mut d = Dump {
        arenas: Vec::new(),
        pages: HashMap::new(),
        walked: BTreeSet::new(),
        built: Vec::new(),
        packed: Vec::new(),
    };
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f.as_slice() {
            ["walked", a] => d.walked.extend(number(a)),
            ["built", a, ..] => d.built.extend(number(a)),
            ["packed", a, len, ..] => d.packed.extend(number(a).zip(number(len))),
            ["frame", _] => {}
            [page, arena, at] => {
                if let (Some(page), Some(arena), Some(at)) =
                    (number(page), number(arena), number(at))
                {
                    d.pages.insert(page / PAGE, (arena as usize, at as usize));
                }
            }
            _ => {}
        }
    }
    let arenas = d
        .pages
        .values()
        .map(|p| p.0)
        .filter(|&a| a != usize::MAX)
        .max()
        .map_or(0, |a| a + 1);
    for n in 0..arenas {
        d.arenas
            .push(std::fs::read(folder.join(format!("arena-{n}.bin"))).unwrap_or_default());
    }
    Ok(d)
}

/// Headers on 256 bytes in the pages held that look like a structure's.
fn headers(d: &Dump) -> Vec<Header> {
    let mut pages: Vec<u64> = d.pages.keys().copied().collect();
    pages.sort_unstable();
    let mut out = Vec::new();
    for page in pages {
        for k in 0..PAGE / 256 {
            let address = page * PAGE + 256 * k;
            let Some(h) = Header::read(d, address) else {
                continue;
            };
            let plausible = h.marks & !0x8106_0000 == 0
                && h.bytes > 0
                && h.bytes < 1 << 28
                && matches!(h.root & 7, 4 | 5)
                && h.bounds.iter().all(|x| x.is_finite())
                && h.bounds[0] <= h.bounds[3]
                && h.bounds[1] <= h.bounds[4]
                && h.bounds[2] <= h.bounds[5];
            if plausible {
                out.push(h);
            }
        }
    }
    out
}

fn line(what: &str, address: u64, h: &Header, r: &Report) {
    println!(
        "{what} {address:#x} marks {:#x} {} bytes: {} box nodes, {} triangles, {} instances, depth {}; \
         broken: {} cycles, {} reached twice, {} outside, {} not in the dump, {} odd; \
         loose: mean {:.2}, most {:.1}, {} over 2, {} escaping; \
         children entered per box test {:.2}; expected node tests {:.1} (own {:.1}); \
         {} huge boxes; header bounds {:?}, root's children span {:?}",
        h.marks >> 16,
        h.bytes,
        r.boxes,
        r.triangles,
        r.instances,
        r.depth,
        r.cycles,
        r.shared,
        r.outside,
        r.missing,
        r.odd,
        r.loose_mean(),
        r.loose_most,
        r.loose_over_2,
        r.escapes,
        r.children_entered(),
        r.expected,
        r.expected_own,
        r.huge,
        h.bounds,
        r.root,
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(folder) = args.iter().find(|a| !a.starts_with("--")) else {
        eprintln!("usage: bvh_dump <folder written by VICTORY_MEMORY_DUMP> [--all]");
        std::process::exit(2);
    };
    let all = args.iter().any(|a| a == "--all");
    let d = match load(Path::new(folder)) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("{folder}: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "{} pages in {} arenas; {} addresses the walking programs were given, {} the builder wrote, {} packed structures",
        d.pages.len(),
        d.arenas.len(),
        d.walked.len(),
        d.built.len(),
        d.packed.len()
    );
    let found = headers(&d);
    let (tops, bottoms): (Vec<&Header>, Vec<&Header>) = found.iter().partition(|h| !h.bottom());
    println!(
        "{} top levels and {} bottom levels found",
        tops.len(),
        bottoms.len()
    );
    let mut checked: HashMap<u64, Option<Report>> = HashMap::new();
    // Bottom levels by what they cost the top levels: (summed chance x
    // estimate, instances, estimate, static).
    let mut cost: HashMap<u64, (f64, u64, f64, bool)> = HashMap::new();
    for h in &tops {
        let mut instanced = Vec::new();
        let Some(r) = bvh_check::check(&d, h.address, &mut checked, &mut instanced) else {
            continue;
        };
        let walked = if d.walked.contains(&h.address) {
            " (walked)"
        } else {
            ""
        };
        line(&format!("top{walked}"), h.address, h, &r);
        // The instances whose boxes a ray through the root enters most often:
        // one in nearly every ray's way is tested by nearly every ray.
        let mut by_chance: Vec<&bvh_check::Instanced> = instanced.iter().collect();
        by_chance.sort_by(|a, b| b.chance.total_cmp(&a.chance));
        let wide = |share: f64| instanced.iter().filter(|i| i.chance >= share).count();
        println!(
            "  instances entered by a ray through the root: {} at 50 %+, {} at 10 %+, {} at 1 %+ of {}; {} with huge boxes; sum of chances {:.2}",
            wide(0.5),
            wide(0.1),
            wide(0.01),
            instanced.len(),
            instanced
                .iter()
                .filter(|i| bvh_check::huge(&i.world))
                .count(),
            instanced.iter().map(|i| i.chance).sum::<f64>(),
        );
        for i in by_chance.iter().take(12) {
            println!(
                "    chance {:.3}, bottom {:#x} ({}, {:.1} tests a ray that enters it), box {:?}",
                i.chance,
                i.address,
                if i.static_root { "static" } else { "dynamic" },
                i.expected,
                i.world
            );
        }
        for i in instanced {
            let c = cost
                .entry(i.address)
                .or_insert((0.0, 0, i.expected, i.static_root));
            c.0 += i.chance * i.expected;
            c.1 += 1;
        }
    }
    let mut cost: Vec<_> = cost.into_iter().collect();
    cost.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
    println!("bottom levels by what they add to the top levels' estimates:");
    for (address, (added, n, expected, static_root)) in cost.iter().take(40) {
        let kind = if *static_root { "static" } else { "dynamic" };
        print!("  adds {added:.1} ({n} instances, {kind}, {expected:.1} a ray that enters it): ");
        match (
            Header::read(&d, *address),
            checked.get(address).cloned().flatten(),
        ) {
            (Some(h), Some(r)) => line("bottom", *address, &h, &r),
            _ => println!("{address:#x} not readable"),
        }
    }
    if all {
        let mut sum = Report::default();
        let mut n = 0u64;
        for h in &bottoms {
            let r = match checked.get(&h.address) {
                Some(r) => r.clone(),
                None => bvh_check::check(&d, h.address, &mut HashMap::new(), &mut Vec::new()),
            };
            let Some(r) = r else { continue };
            line("bottom", h.address, h, &r);
            n += 1;
            sum.cycles += r.cycles;
            sum.shared += r.shared;
            sum.outside += r.outside;
            sum.missing += r.missing;
            sum.loose_sum += r.loose_sum;
            sum.loose_count += r.loose_count;
            sum.loose_over_2 += r.loose_over_2;
            sum.escapes += r.escapes;
        }
        println!(
            "all {n} bottom levels: {} cycles, {} reached twice, {} outside, {} not in the dump; loose mean {:.2}, {} over 2, {} escaping",
            sum.cycles,
            sum.shared,
            sum.outside,
            sum.missing,
            sum.loose_mean(),
            sum.loose_over_2,
            sum.escapes
        );
    }
}
