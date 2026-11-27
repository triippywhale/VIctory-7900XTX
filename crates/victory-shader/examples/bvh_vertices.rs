//! Where a bottom level's triangles came from, in a memory dump
//! (`VICTORY_MEMORY_DUMP`):
//!
//! ```text
//! cargo run --release -p victory-shader --example bvh_vertices -- <folder> <bottom level address>
//! ```
//!
//! Collects the corners of the structure's triangles, then looks through
//! every page of the dump for the sane ones as three floats in a row (x y z)
//! and as three halves in a row: the builder's input. For each place found it
//! prints the stride between neighbouring corners and what lies there for the
//! corners that are not sane, so it tells whether the input already held them
//! (the program that wrote the vertices) or the builder made them (how it
//! read them).

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use victory_shader::bvh;
use victory_shader::bvh_check::{Header, Memory, PAGE, huge, node_address};

struct Dump {
    arenas: Vec<Vec<u8>>,
    pages: HashMap<u64, (usize, usize)>,
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
    let mut pages = HashMap::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if let [page, arena, at] = f.as_slice()
            && let (Some(page), Some(arena), Some(at)) = (number(page), number(arena), number(at))
        {
            pages.insert(page / PAGE, (arena as usize, at as usize));
        }
    }
    let n = pages
        .values()
        .map(|p| p.0)
        .filter(|&a| a != usize::MAX)
        .max()
        .map_or(0, |a| a + 1);
    let arenas = (0..n)
        .map(|k| std::fs::read(folder.join(format!("arena-{k}.bin"))).unwrap_or_default())
        .collect();
    Ok(Dump { arenas, pages })
}

/// Every triangle corner of the structure at `address`.
fn corners(d: &Dump, address: u64) -> Vec<[f32; 3]> {
    let Some(h) = Header::read(d, address) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut work = vec![h.root];
    let mut seen = std::collections::HashSet::new();
    while let Some(p) = work.pop() {
        let at = node_address(address, p);
        if !seen.insert((at, p & 7)) || seen.len() > 1 << 22 {
            continue;
        }
        let Some(first) = d.dwords::<16>(at) else {
            continue;
        };
        let mut words = [0u32; 28];
        words[..16].copy_from_slice(&first);
        match p & 7 {
            4 | 5 => {
                for c in bvh::children(&words) {
                    if c != bvh::NONE {
                        work.push(c);
                    }
                }
            }
            k @ 0..=3 => out.extend(bvh::triangle_vertices(&words, k)),
            _ => {}
        }
    }
    out
}

fn half(x: f32) -> u16 {
    // Round to nearest: enough to find a half that widened to `x`.
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    if e <= 0 || e >= 31 {
        return sign;
    }
    sign | (e as u16) << 10 | ((b >> 13) & 0x3ff) as u16
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(folder), Some(address)) = (args.first(), args.get(1).and_then(|a| number(a))) else {
        eprintln!("usage: bvh_vertices <dump folder> <bottom level address>");
        std::process::exit(2);
    };
    let d = match load(Path::new(folder)) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("{folder}: {e}");
            std::process::exit(1);
        }
    };
    let all = corners(&d, address);
    // The structure's own triangle nodes hold the corners too: not looked in.
    let own = Header::read(&d, address).map_or(0..0, |h| address..address + h.bytes);
    let box_of = |v: &[f32; 3]| [v[0], v[1], v[2], v[0], v[1], v[2]];
    let (bad, good): (Vec<[f32; 3]>, Vec<[f32; 3]>) = all.iter().partition(|v| huge(&box_of(v)));
    println!(
        "{address:#x}: {} corners, {} not sane",
        all.len(),
        bad.len()
    );
    for v in bad.iter().take(12) {
        println!(
            "  not sane: {:?} = {:08x} {:08x} {:08x}",
            v,
            v[0].to_bits(),
            v[1].to_bits(),
            v[2].to_bits()
        );
    }
    // The sane corners by their x y z dwords, and by their halves.
    let mut want32: HashMap<[u32; 3], [f32; 3]> = HashMap::new();
    let mut want16: HashMap<[u16; 3], [f32; 3]> = HashMap::new();
    for v in good.iter().take(4096) {
        want32.insert(v.map(f32::to_bits), *v);
        want16.insert(v.map(half), *v);
    }
    // Places in the dump, by the page they are on.
    let mut found32: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    let mut found16: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    let mut pages: Vec<u64> = d.pages.keys().copied().collect();
    pages.sort_unstable();
    for page in pages {
        let Some(bytes) = d.page(page) else { continue };
        let dw = |k: usize| u32::from_le_bytes(bytes[k..k + 4].try_into().unwrap());
        let hw = |k: usize| u16::from_le_bytes(bytes[k..k + 2].try_into().unwrap());
        for k in (0..bytes.len() - 12).step_by(4) {
            if want32.contains_key(&[dw(k), dw(k + 4), dw(k + 8)]) {
                let at = page * PAGE + k as u64;
                if !own.contains(&at) {
                    found32.entry(page).or_default().push(at);
                }
            }
        }
        for k in (0..bytes.len() - 6).step_by(2) {
            let h = [hw(k), hw(k + 2), hw(k + 4)];
            if h != [0, 0, 0] && want16.contains_key(&h) {
                let at = page * PAGE + k as u64;
                if !own.contains(&at) {
                    found16.entry(page).or_default().push(at);
                }
            }
        }
    }
    for (what, found, size) in [("floats", &found32, 4u64), ("halves", &found16, 2u64)] {
        let total: usize = found.values().map(Vec::len).sum();
        println!(
            "sane corners found as three {what}: {total} on {} pages",
            found.len()
        );
        for (page, at) in found.iter().take(16) {
            // The most common gap between neighbours: the vertex stride.
            let mut gaps: HashMap<u64, usize> = HashMap::new();
            for w in at.windows(2) {
                *gaps.entry(w[1] - w[0]).or_default() += 1;
            }
            let stride = gaps.iter().max_by_key(|g| g.1).map_or(0, |g| *g.0);
            println!(
                "  page {:#x}: {} found, first {:#x}, stride {stride}",
                page * PAGE,
                at.len(),
                at[0]
            );
            // What the stride holds after the first find, eight vertices.
            if stride > 0 {
                for n in 0..8u64 {
                    let a = at[0] + n * stride;
                    let w: Option<[u32; 4]> = d.dwords(a & !3);
                    if let Some(w) = w {
                        let as_f: Vec<f32> = w.iter().map(|x| f32::from_bits(*x)).collect();
                        let _ = size;
                        println!(
                            "    {a:#x}: {:08x} {:08x} {:08x} {:08x} = {as_f:?}",
                            w[0], w[1], w[2], w[3]
                        );
                    }
                }
            }
        }
    }
    // Where the not-sane corners' dwords are, if anywhere.
    let mut hits = 0;
    for v in bad.iter().take(64) {
        let key = v.map(f32::to_bits);
        for (&page, &(_, _)) in &d.pages {
            let Some(bytes) = d.page(page) else { continue };
            for k in (0..bytes.len() - 12).step_by(4) {
                let w = |o: usize| u32::from_le_bytes(bytes[k + o..k + o + 4].try_into().unwrap());
                if [w(0), w(4), w(8)] == key && !own.contains(&(page * PAGE + k as u64)) {
                    if hits < 16 {
                        println!(
                            "not-sane corner {:?} as three floats at {:#x}",
                            v,
                            page * PAGE + k as u64
                        );
                    }
                    hits += 1;
                }
            }
        }
    }
    println!(
        "not-sane corners found as three floats: {hits} (of the first {})",
        bad.len().min(64)
    );
}
