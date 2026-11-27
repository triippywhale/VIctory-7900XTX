//! Offline checks of the title's ray-tracing structures, as a memory dump
//! holds them (`VICTORY_MEMORY_DUMP`, read by the example `bvh_dump`): what
//! makes a ray test many nodes.
//!
//! For a structure (from its 64-byte header, see `docs/fork-ray-tracing.md`)
//! the nodes reachable from its root are walked once, and counted:
//!
//! - **broken**: a pointer back to a node on the way to it (a cycle: a walk
//!   goes round until the loop limit), a node reached twice by different
//!   ways (a walk tests it twice), a node outside the structure's own size,
//!   a node on a page the dump does not have;
//! - **loose**: for each child box node, the box its parent keeps for it
//!   against the union of the boxes it keeps for its own children (1 when
//!   tight; a refit that does not shrink, or a builder that writes wrong
//!   bounds, makes it larger), and boxes of a node that stick out of the box
//!   its parent keeps for it (a ray can miss what is there);
//! - **expected tests**: the surface-area estimate of the node tests a ray
//!   that enters the root makes when it goes on through everything it
//!   enters (no closest-hit shortening): each child adds its tests weighed
//!   by the share of the parent's box surface its own box has. For the top
//!   level, an instance adds the estimate of its bottom level.
//!
//! The estimate is not what the title's rays do (they stop at the closest
//! hit, and they are not random), but it moves with what costs them, and it
//! says which structures do.

use std::collections::{HashMap, HashSet};

use crate::bvh::{self, NONE};

/// Bytes of a node.
const NODE: u64 = 64;
/// Bytes of a page in a dump.
pub const PAGE: u64 = 1 << 16;
/// A header's version (dword 0, low half).
const VERSION: u32 = 6;
/// A header's mark of a bottom level (dword 0).
pub const BOTTOM: u32 = 0x8000_0000;
/// An instance root pointer's mark of a static structure.
const STATIC_ROOT: u32 = 1 << 30;

/// Title memory as far as a dump holds it.
pub trait Memory {
    /// The page `number` (its address / [`PAGE`]): [`PAGE`] bytes.
    fn page(&self, number: u64) -> Option<&[u8]>;

    /// `N` dwords from `address` (on 4 bytes), if every byte is there.
    fn dwords<const N: usize>(&self, address: u64) -> Option<[u32; N]> {
        let mut out = [0u32; N];
        for (k, w) in out.iter_mut().enumerate() {
            let a = address + 4 * k as u64;
            let page = self.page(a / PAGE)?;
            let at = (a % PAGE) as usize;
            *w = u32::from_le_bytes(page.get(at..at + 4)?.try_into().ok()?);
        }
        Some(out)
    }
}

impl Memory for HashMap<u64, Vec<u8>> {
    fn page(&self, number: u64) -> Option<&[u8]> {
        self.get(&number).map(Vec::as_slice)
    }
}

/// A structure's header.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Header {
    pub address: u64,
    /// Dword 0's high half: 0x8000 bottom level, 0x4 compacted, 0x2
    /// rehydrated, 0x100 still packed.
    pub marks: u32,
    /// Bytes from the header the structure says it takes.
    pub bytes: u64,
    pub elements: u32,
    pub bounds: [f32; 6],
    pub root: u32,
}

impl Header {
    pub fn read(memory: &impl Memory, address: u64) -> Option<Self> {
        let w: [u32; 16] = memory.dwords(address)?;
        (w[0] & 0xffff == VERSION).then(|| Self {
            address,
            marks: w[0] & 0xffff_0000,
            bytes: u64::from(w[1]) * NODE,
            elements: w[3],
            bounds: std::array::from_fn(|k| f32::from_bits(w[4 + k])),
            root: w[12],
        })
    }

    pub fn bottom(&self) -> bool {
        self.marks & BOTTOM != 0
    }
}

/// The byte address of the node a pointer of the structure at `base` names
/// (the title's 64-bit form adds `base >> 3` to it; `base` is on 256 bytes).
pub fn node_address(base: u64, pointer: u32) -> u64 {
    base + (u64::from(pointer) >> 3) * NODE
}

/// An axis-aligned box: minimum x y z, maximum x y z.
pub type Aabb = [f32; 6];

/// Half the surface of a box (0 for an empty or inverted one).
pub fn area(b: &Aabb) -> f64 {
    let d: [f64; 3] = std::array::from_fn(|a| f64::from(b[3 + a] - b[a]).max(0.0));
    if d.iter().any(|x| !x.is_finite()) {
        return f64::INFINITY;
    }
    d[0] * d[1] + d[1] * d[2] + d[2] * d[0]
}

fn valid(b: &Aabb) -> bool {
    b[0] <= b[3] && b[1] <= b[4] && b[2] <= b[5]
}

fn union(a: Option<Aabb>, b: &Aabb) -> Aabb {
    match a {
        None => *b,
        Some(a) => std::array::from_fn(|k| {
            if k < 3 {
                a[k].min(b[k])
            } else {
                a[k].max(b[k])
            }
        }),
    }
}

/// How far `inner` sticks out of `outer`, as a share of `outer`'s largest
/// extent (0 inside).
fn outside(inner: &Aabb, outer: &Aabb) -> f64 {
    let size = (0..3)
        .map(|a| f64::from(outer[3 + a] - outer[a]))
        .fold(0.0, f64::max)
        .max(1e-6);
    (0..3)
        .map(|a| {
            f64::from(outer[a] - inner[a])
                .max(f64::from(inner[3 + a] - outer[3 + a]))
                .max(0.0)
        })
        .fold(0.0, f64::max)
        / size
}

/// A box node's children and their boxes (the valid ones), or `None` when
/// the node is not there.
fn box_node(memory: &impl Memory, address: u64, kind: u32) -> Option<Vec<(u32, Aabb)>> {
    let mut words = [0u32; 28];
    let first: [u32; 16] = memory.dwords(address)?;
    words[..16].copy_from_slice(&first);
    let boxes = if kind == 5 {
        let second: [u32; 12] = memory.dwords(address + NODE)?;
        words[16..].copy_from_slice(&second);
        bvh::boxes32(&words)
    } else {
        bvh::boxes16(&words)
    };
    Some(
        bvh::children(&words)
            .into_iter()
            .zip(boxes)
            .filter(|(c, b)| *c != NONE && valid(b))
            .collect(),
    )
}

/// What a walk of one structure found.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Report {
    pub boxes: u64,
    pub triangles: u64,
    pub instances: u64,
    pub depth: u32,
    /// Pointers back to a node on the way to them.
    pub cycles: u64,
    /// Nodes reached a second time by another way.
    pub shared: u64,
    /// Nodes outside the structure's size.
    pub outside: u64,
    /// Nodes (or instances' structures) the dump does not hold.
    pub missing: u64,
    /// Pointers of kind 7, or instances in a bottom level.
    pub odd: u64,
    /// Children's boxes against the union of their own children: sum, the
    /// most, and how many of them are over 2.
    pub loose_sum: f64,
    pub loose_count: u64,
    pub loose_most: f64,
    pub loose_over_2: u64,
    /// Boxes that stick out of their parent's by more than 1 % of it.
    pub escapes: u64,
    /// Children entered per box node, weighed by the chance a ray gets there
    /// (the surface estimate of [`Report::children_entered`]).
    pub entered: f64,
    pub entered_weight: f64,
    /// The surface estimate of the node tests a ray through the root
    /// makes: in this structure, and with the bottom levels of its
    /// instances.
    pub expected_own: f64,
    pub expected: f64,
}

impl Report {
    pub fn loose_mean(&self) -> f64 {
        self.loose_sum / self.loose_count.max(1) as f64
    }

    /// The children a box test lets a ray into, on average over the tests
    /// the estimate counts (the meter's `children entered per box test`).
    pub fn children_entered(&self) -> f64 {
        self.entered / self.entered_weight.max(1e-12)
    }
}

/// A bottom level an instance names, as the top level's walk found it.
#[derive(Debug, Clone, PartialEq)]
pub struct Instanced {
    pub address: u64,
    pub static_root: bool,
    /// The chance a ray through the top level's root enters the instance's
    /// box, and the bottom level's estimate.
    pub chance: f64,
    pub expected: f64,
}

/// Walks the structure whose header is at `address`. For a top level, its
/// instances' bottom levels are walked as well (once each, their reports in
/// `bottoms`), and the instances found are listed.
pub fn check(
    memory: &impl Memory,
    address: u64,
    bottoms: &mut HashMap<u64, Option<Report>>,
    instanced: &mut Vec<Instanced>,
) -> Option<Report> {
    let header = Header::read(memory, address)?;
    let mut r = Report::default();
    let root_kind = header.root & 7;
    let root = node_address(address, header.root);
    let entry = if header.bounds.iter().all(|x| x.is_finite()) && valid(&header.bounds) {
        header.bounds
    } else {
        [f32::MIN, f32::MIN, f32::MIN, f32::MAX, f32::MAX, f32::MAX]
    };
    // Iterative: (node address, kind, the box it was entered by, the chance
    // a ray through the root gets here, depth), and leaving a node's path.
    let mut on_path: HashSet<u64> = HashSet::new();
    let mut seen: HashSet<u64> = HashSet::new();
    enum Step {
        Enter(u64, u32, Aabb, f64, u32),
        Leave(u64),
    }
    let mut work = vec![Step::Enter(root, root_kind, entry, 1.0, 1)];
    while let Some(step) = work.pop() {
        let (at, kind, by, chance, depth) = match step {
            Step::Leave(at) => {
                on_path.remove(&at);
                continue;
            }
            Step::Enter(a, k, b, c, d) => (a, k, b, c, d),
        };
        if on_path.contains(&at) {
            r.cycles += 1;
            continue;
        }
        // Four triangles share a node: each is its own test.
        let key = if kind < 4 {
            at ^ u64::from(kind) << 60
        } else {
            at
        };
        if !seen.insert(key) {
            r.shared += 1;
            continue;
        }
        r.depth = r.depth.max(depth);
        if at < address || at >= address + header.bytes.max(NODE) {
            r.outside += 1;
        }
        // This node's test.
        let tests = chance;
        match kind {
            4 | 5 => {
                let Some(children) = box_node(memory, at, kind) else {
                    r.missing += 1;
                    continue;
                };
                r.boxes += 1;
                r.expected_own += tests;
                r.expected += tests;
                let by_area = area(&by);
                let mut tight: Option<Aabb> = None;
                let mut entered = 0.0;
                for (_, b) in &children {
                    tight = Some(union(tight, b));
                    if outside(b, &by) > 0.01 && by_area.is_finite() {
                        r.escapes += 1;
                    }
                    entered += if by_area > 0.0 && by_area.is_finite() {
                        (area(b) / by_area).min(1.0)
                    } else {
                        1.0
                    };
                }
                r.entered += chance * entered;
                r.entered_weight += chance;
                if let Some(t) = tight {
                    let (outer, inner) = (area(&by), area(&t));
                    if depth > 1 && outer.is_finite() && inner > 0.0 {
                        let loose = outer / inner;
                        r.loose_sum += loose;
                        r.loose_count += 1;
                        r.loose_most = r.loose_most.max(loose);
                        r.loose_over_2 += u64::from(loose > 2.0);
                    }
                }
                on_path.insert(at);
                work.push(Step::Leave(at));
                for (pointer, b) in children.into_iter().rev() {
                    let share = if by_area > 0.0 && by_area.is_finite() {
                        (area(&b) / by_area).min(1.0)
                    } else {
                        1.0
                    };
                    work.push(Step::Enter(
                        node_address(address, pointer),
                        pointer & 7,
                        b,
                        chance * share,
                        depth + 1,
                    ));
                }
            }
            0..=3 => {
                r.triangles += 1;
                r.expected_own += tests;
                r.expected += tests;
            }
            6 => {
                r.instances += 1;
                if header.bottom() {
                    r.odd += 1;
                    continue;
                }
                match instance(memory, at) {
                    Some((blas, static_root)) => {
                        let report = bottoms
                            .entry(blas)
                            .or_insert_with(|| {
                                check(memory, blas, &mut HashMap::new(), &mut Vec::new())
                            })
                            .clone();
                        match report {
                            Some(b) => {
                                r.expected += chance * b.expected;
                                instanced.push(Instanced {
                                    address: blas,
                                    static_root,
                                    chance,
                                    expected: b.expected,
                                });
                            }
                            None => r.missing += 1,
                        }
                    }
                    None => r.missing += 1,
                }
            }
            _ => r.odd += 1,
        }
    }
    Some(r)
}

/// An instance's bottom level and whether its root pointer marks it static:
/// the dword after the matrix (12-15) that names, in 256-byte units, a
/// bottom level's header, and the one before it as the root pointer.
pub fn instance(memory: &impl Memory, address: u64) -> Option<(u64, bool)> {
    let w: [u32; 16] = memory.dwords(address)?;
    (12..16).find_map(|k| {
        let blas = u64::from(w[k]) << 8;
        let h = Header::read(memory, blas).filter(|h| h.bottom())?;
        Some((h.address, k > 12 && w[k - 1] & STATIC_ROOT != 0))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Title memory built by the tests: structures at 256-byte boundaries.
    #[derive(Default)]
    struct Built(HashMap<u64, Vec<u8>>);

    impl Built {
        fn put(&mut self, address: u64, words: &[u32]) {
            for (k, w) in words.iter().enumerate() {
                let a = address + 4 * k as u64;
                let page = self
                    .0
                    .entry(a / PAGE)
                    .or_insert_with(|| vec![0; PAGE as usize]);
                let at = (a % PAGE) as usize;
                page[at..at + 4].copy_from_slice(&w.to_le_bytes());
            }
        }

        fn header(&mut self, address: u64, bottom: bool, nodes: u32, bounds: Aabb, root: u32) {
            let mut w = [0u32; 16];
            w[0] = VERSION | if bottom { BOTTOM } else { 0 };
            w[1] = nodes;
            for k in 0..6 {
                w[4 + k] = bounds[k].to_bits();
            }
            w[12] = root;
            self.put(address, &w);
        }

        /// A 32-bit box node at node `n` of the structure at `base`.
        fn box32(&mut self, base: u64, n: u32, children: [(u32, Aabb); 4]) {
            let words = bvh::tests::box32(children.map(|c| c.0), children.map(|c| c.1));
            self.put(base + u64::from(n) * NODE, &words);
        }
    }

    const EMPTY: (u32, Aabb) = (NONE, [1.0, 1.0, 1.0, -1.0, -1.0, -1.0]);

    /// Pointer to node `n` of kind `k`.
    fn at(n: u32, k: u32) -> u32 {
        n << 3 | k
    }

    fn cube(lo: f32, hi: f32) -> Aabb {
        [lo, lo, lo, hi, hi, hi]
    }

    #[test]
    fn a_tight_tree_has_no_findings_and_a_small_estimate() {
        let mut m = Built::default();
        let base = 0x10_0000;
        m.header(base, true, 8, cube(0.0, 2.0), at(1, 5));
        // Root: two halves, each with two triangles.
        m.box32(
            base,
            1,
            [
                (at(3, 5), [0.0, 0.0, 0.0, 1.0, 2.0, 2.0]),
                (at(5, 5), [1.0, 0.0, 0.0, 2.0, 2.0, 2.0]),
                EMPTY,
                EMPTY,
            ],
        );
        m.box32(
            base,
            3,
            [
                (at(7, 0), [0.0, 0.0, 0.0, 1.0, 1.0, 2.0]),
                (at(7, 1), [0.0, 1.0, 0.0, 1.0, 2.0, 2.0]),
                EMPTY,
                EMPTY,
            ],
        );
        m.box32(
            base,
            5,
            [
                (at(7, 2), [1.0, 0.0, 0.0, 2.0, 1.0, 2.0]),
                (at(7, 3), [1.0, 1.0, 0.0, 2.0, 2.0, 2.0]),
                EMPTY,
                EMPTY,
            ],
        );
        let r = check(&m.0, base, &mut HashMap::new(), &mut Vec::new()).unwrap();
        assert_eq!((r.boxes, r.triangles, r.depth), (3, 4, 3));
        assert_eq!(
            (r.cycles, r.shared, r.outside, r.missing, r.escapes),
            (0, 0, 0, 0, 0)
        );
        assert!((r.loose_mean() - 1.0).abs() < 1e-9, "{}", r.loose_mean());
        // The root's test, each half's by its share of the root's surface,
        // each triangle's by its own.
        let half = area(&[0.0, 0.0, 0.0, 1.0, 2.0, 2.0]) / area(&cube(0.0, 2.0));
        let quarter = area(&[0.0, 0.0, 0.0, 1.0, 1.0, 2.0]) / area(&cube(0.0, 2.0));
        let want = 1.0 + 2.0 * half + 4.0 * quarter;
        assert!((r.expected - want).abs() < 1e-9, "{} {want}", r.expected);
    }

    #[test]
    fn a_pointer_back_up_the_tree_is_a_cycle_and_outside_pointers_are_counted() {
        let mut m = Built::default();
        let base = 0x20_0000;
        m.header(base, true, 6, cube(0.0, 1.0), at(1, 5));
        m.box32(
            base,
            1,
            [
                (at(3, 5), cube(0.0, 1.0)),
                (at(100, 5), cube(0.0, 0.5)),
                EMPTY,
                EMPTY,
            ],
        );
        // Node 3 points back at the root.
        m.box32(base, 3, [(at(1, 5), cube(0.0, 1.0)), EMPTY, EMPTY, EMPTY]);
        let r = check(&m.0, base, &mut HashMap::new(), &mut Vec::new()).unwrap();
        assert_eq!(r.cycles, 1);
        // Node 100 lies past the structure's 6 nodes, and its page is there
        // but holds zeros: a box node with four children at pointer 0 (kind
        // 0, triangles of node 0) and empty boxes.
        assert_eq!(r.outside, 1);
    }

    #[test]
    fn a_loose_box_and_an_escaping_child_are_found() {
        let mut m = Built::default();
        let base = 0x30_0000;
        m.header(base, true, 8, cube(0.0, 4.0), at(1, 5));
        // The root keeps a box of 4 for node 3, whose children span 1.
        m.box32(base, 1, [(at(3, 5), cube(0.0, 4.0)), EMPTY, EMPTY, EMPTY]);
        // ... and one of them sticks out of that box.
        m.box32(
            base,
            3,
            [
                (at(5, 0), cube(0.0, 1.0)),
                (at(5, 1), cube(0.5, 4.5)),
                EMPTY,
                EMPTY,
            ],
        );
        let r = check(&m.0, base, &mut HashMap::new(), &mut Vec::new()).unwrap();
        assert_eq!(r.escapes, 1);
        assert_eq!(r.loose_count, 1);
        let loose = area(&cube(0.0, 4.0)) / area(&cube(0.0, 4.5));
        assert!((r.loose_most - loose).abs() < 1e-9);

        let mut m = Built::default();
        m.header(base, true, 8, cube(0.0, 4.0), at(1, 5));
        m.box32(base, 1, [(at(3, 5), cube(0.0, 4.0)), EMPTY, EMPTY, EMPTY]);
        m.box32(
            base,
            3,
            [
                (at(5, 0), cube(0.0, 1.0)),
                (at(5, 1), cube(0.0, 1.0)),
                EMPTY,
                EMPTY,
            ],
        );
        let r = check(&m.0, base, &mut HashMap::new(), &mut Vec::new()).unwrap();
        assert_eq!((r.escapes, r.loose_over_2), (0, 1));
        assert!((r.loose_most - 16.0).abs() < 1e-9);
    }

    #[test]
    fn a_node_reached_twice_and_a_node_not_in_the_dump_are_counted() {
        let mut m = Built::default();
        let base = 0x40_0000;
        m.header(base, true, 2000, cube(0.0, 1.0), at(1, 5));
        // Node 3 twice; node 1500 on a page the dump does not have.
        m.box32(
            base,
            1,
            [
                (at(3, 5), cube(0.0, 1.0)),
                (at(3, 5), cube(0.0, 1.0)),
                (at(1500, 5), cube(0.0, 1.0)),
                EMPTY,
            ],
        );
        m.box32(base, 3, [(at(5, 0), cube(0.0, 1.0)), EMPTY, EMPTY, EMPTY]);
        let r = check(&m.0, base, &mut HashMap::new(), &mut Vec::new()).unwrap();
        assert_eq!((r.shared, r.missing, r.cycles), (1, 1, 0));
    }

    #[test]
    fn a_top_level_adds_its_instances_bottom_levels_by_their_chance() {
        let mut m = Built::default();
        let (top, bottom) = (0x50_0000u64, 0x60_0000u64);
        // The bottom level: a root over two triangles that fill it.
        m.header(bottom, true, 4, cube(0.0, 1.0), at(1, 5));
        m.box32(
            bottom,
            1,
            [
                (at(3, 0), cube(0.0, 1.0)),
                (at(3, 1), cube(0.0, 1.0)),
                EMPTY,
                EMPTY,
            ],
        );
        // The top level: one instance in a box with a quarter of the root's
        // surface.
        m.header(top, false, 4, cube(0.0, 2.0), at(1, 5));
        m.box32(top, 1, [(at(3, 6), cube(0.0, 1.0)), EMPTY, EMPTY, EMPTY]);
        let mut inst = [0u32; 16];
        inst[0] = 1f32.to_bits();
        inst[5] = 1f32.to_bits();
        inst[10] = 1f32.to_bits();
        inst[14] = at(1, 5) | STATIC_ROOT;
        inst[15] = (bottom >> 8) as u32;
        m.put(top + 3 * NODE, &inst);
        let (mut bottoms, mut instanced) = (HashMap::new(), Vec::new());
        let r = check(&m.0, top, &mut bottoms, &mut instanced).unwrap();
        assert_eq!((r.boxes, r.instances, r.missing), (1, 1, 0));
        assert_eq!(instanced.len(), 1);
        assert_eq!(instanced[0].address, bottom);
        assert!(instanced[0].static_root);
        // Bottom level: 1 + 2 triangles; entered with a quarter's chance.
        assert!((instanced[0].expected - 3.0).abs() < 1e-9);
        assert!((instanced[0].chance - 0.25).abs() < 1e-9);
        assert!((r.expected_own - 1.0).abs() < 1e-9);
        assert!((r.expected - 1.75).abs() < 1e-9);
        assert!(bottoms[&bottom].is_some());
    }
}
