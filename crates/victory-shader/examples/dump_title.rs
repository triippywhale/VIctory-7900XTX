//! The card's copy of title memory against the CPU's side of it, in a memory
//! dump (`VICTORY_MEMORY_DUMP`: `arena-*.bin` with `pages.txt`, and
//! `title.bin` with `title-pages.txt`):
//!
//! ```text
//! cargo run --release -p victory-shader --example dump_title -- <folder> <address> [dwords]
//! ```
//!
//! Prints `dwords` (64) dwords from `address` as the card holds them and as
//! title memory holds them, marking those that differ, and how many dwords of
//! the page differ; and the page's line in `title-pages.txt` (the frame it
//! came in, was last looked at, and the write-watch asking it was last taken
//! after).

use std::collections::HashMap;
use std::path::Path;

const PAGE: u64 = 1 << 16;

fn number(s: &str) -> Option<u64> {
    match s.strip_prefix("0x") {
        Some(h) => u64::from_str_radix(h, 16).ok(),
        None => s.parse().ok(),
    }
}

fn dword(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(folder), Some(address)) = (args.first(), args.get(1).and_then(|a| number(a))) else {
        eprintln!("usage: dump_title <dump folder> <address> [dwords]");
        std::process::exit(2);
    };
    let count = args.get(2).and_then(|a| number(a)).unwrap_or(64) as usize;
    let folder = Path::new(folder);
    let page = address / PAGE * PAGE;
    // The card's page.
    let card = std::fs::read_to_string(folder.join("pages.txt"))
        .ok()
        .and_then(|text| {
            text.lines().find_map(|line| {
                let f: Vec<&str> = line.split_whitespace().collect();
                match f.as_slice() {
                    [p, arena, at] if number(p) == Some(page) => {
                        let (arena, at) = (number(arena)?, number(at)?);
                        let bytes =
                            std::fs::read(folder.join(format!("arena-{arena}.bin"))).ok()?;
                        bytes
                            .get(at as usize..(at + PAGE) as usize)
                            .map(<[u8]>::to_vec)
                    }
                    _ => None,
                }
            })
        });
    // The CPU's side.
    let mut line = None;
    let title = std::fs::read_to_string(folder.join("title-pages.txt"))
        .ok()
        .and_then(|text| {
            let found: HashMap<u64, (usize, String)> = text
                .lines()
                .filter_map(|l| {
                    let f: Vec<&str> = l.split_whitespace().collect();
                    Some((
                        number(f.first()?)?,
                        (number(f.get(1)?)? as usize, l.to_string()),
                    ))
                })
                .collect();
            let (n, l) = found.get(&page)?.clone();
            line = Some(l);
            use std::io::{Read, Seek, SeekFrom};
            let mut file = std::fs::File::open(folder.join("title.bin")).ok()?;
            file.seek(SeekFrom::Start(n as u64 * PAGE)).ok()?;
            let mut bytes = vec![0u8; PAGE as usize];
            file.read_exact(&mut bytes).ok()?;
            Some(bytes)
        });
    println!(
        "page {page:#x}: card {}, title memory {}; title-pages.txt: {}",
        if card.is_some() {
            "held"
        } else {
            "not in the dump"
        },
        if title.is_some() {
            "held"
        } else {
            "not in the dump"
        },
        line.as_deref().unwrap_or("-")
    );
    if let (Some(c), Some(t)) = (&card, &title) {
        let differ = (0..PAGE as usize)
            .step_by(4)
            .filter(|&k| dword(c, k) != dword(t, k))
            .count();
        println!("dwords of the page that differ: {differ} of {}", PAGE / 4);
    }
    let start = (address - page) as usize & !3;
    for k in 0..count {
        let at = start + 4 * k;
        if at + 4 > PAGE as usize {
            break;
        }
        let c = card.as_ref().map(|b| dword(b, at));
        let t = title.as_ref().map(|b| dword(b, at));
        let show = |x: Option<u32>| x.map_or("--------".to_string(), |x| format!("{x:08x}"));
        let mark = if c != t { " *" } else { "" };
        let f = |x: Option<u32>| x.map_or(String::new(), |x| format!("{:e}", f32::from_bits(x)));
        println!(
            "{:#x}: card {} ({}) title {} ({}){mark}",
            page + at as u64,
            show(c),
            f(c),
            show(t),
            f(t)
        );
    }
}
