//! I/O raising of geometry programs ([`crate::io`]): what a GS program reads
//! from its input vertices, what it writes for the vertices it makes, and
//! what its copy shader exports of them.
//!
//! The console runs a geometry shader as the GS half of a legacy (not NGG)
//! merged ES+GS stage. Each lane takes one input primitive; the ES half has
//! left every vertex's outputs in the ES-GS ring in LDS, dword `4 * row +
//! channel` of a vertex's item (rows of the GS's input signature), and the
//! lane finds its vertices' items at the dword offsets in the 16-bit fields
//! of v0, v1 and v4 (vertex 0 in v0 bits 0-15, vertex 1 in bits 16-31...).
//! The lane writes the vertices it makes into its item of the GS-VS ring in
//! LDS (vertex k at dword `k * vertex_dwords`), the item at `lane * item`
//! bytes from a base: the wave's ring offset in s2 plus a dword the code
//! loads from the driver's own table ([`crate::resource::SYSTEM`]) times
//! 256. `s_sendmsg` GS EMIT counts a vertex, CUT ends a strip, GS_DONE ends
//! the wave's output. The copy shader then runs once per emitted vertex: it
//! reads the vertex from the ring at the dword offset in its v0 (plus the
//! same base) and exports it as a vertex shader does.
//!
//! The host's geometry stage has none of those rings: it reads input
//! vertices with `loadInput` and emits vertices with `storeOutput`. Every
//! LDS address of the program is evaluated here from probe values of its
//! inputs, so that each read names an input vertex and a dword of its item,
//! and each write a dword of the GS-VS item; an address that is not such a
//! sum is refused. The ring addresses themselves are then dead: the inputs
//! that make them (the vertex offsets, s2, the table's dword) may flow into
//! nothing else ([`tainted_use`]).

use std::collections::{HashMap, HashSet};

use crate::ShaderError;
use crate::gfx10::{Fields, NULL, Operand};
use crate::io::{Copied, Geometry, Message, Target};
use crate::ir::{self, EXEC, Inst, Program, Reg, Value, operand_reads};
use crate::native::{GeometryIo, HwStage, NativeProgram};
use crate::resource::{Site, inline};

/// `s_sendmsg` message ids and GS operations.
const MSG_GS: u16 = 2;
const MSG_GS_DONE: u16 = 3;
const GS_OP_CUT: u16 = 1;
const GS_OP_EMIT: u16 = 2;
const GS_OP_EMIT_CUT: u16 = 3;
/// Input vertices a primitive has at most (triangles with adjacency).
const MAX_INPUT_VERTICES: usize = 6;
/// How far an address is followed back through the code.
const DEPTH: u32 = 64;

/// Values of the registers a program starts with, for one evaluation.
#[derive(Debug, Clone, Copy)]
struct Probe {
    /// Each input vertex's offset in the ES-GS ring, in dwords.
    vertices: [u32; MAX_INPUT_VERTICES],
    /// The lane (below 32: `v_mbcnt` is read as the lane plus its operand).
    lane: u32,
    /// The dword of the driver's table.
    base: u32,
    /// s2 of a GS program: the wave's offset in the GS-VS ring, in bytes.
    wave: u32,
    /// v0 of a copy shader: the vertex's offset in the GS-VS ring, in dwords.
    vertex: u32,
}

/// A table address the probes give user data 0 and 1 (bits 32-39).
const TABLE: (u32, u32) = (0x1234_5000, 0x42);

const PROBES: [Probe; 2] = [
    Probe {
        vertices: [0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000],
        lane: 0,
        base: 0,
        wave: 0,
        vertex: 0,
    },
    Probe {
        vertices: [0x2400, 0x0C00, 0x3800, 0x1400, 0x4C00, 0x5400],
        lane: 5,
        base: 3,
        wave: 0x40,
        vertex: 0x120,
    },
];

/// The zero probe, and the same with one input changed: the address a
/// GS-VS ring write (or a copy shader's read) must change with by a fixed
/// amount.
fn zero() -> Probe {
    Probe {
        vertices: PROBES[0].vertices,
        lane: 0,
        base: 0,
        wave: 0,
        vertex: 0,
    }
}

/// Evaluates a program's SSA values from probe values of its inputs.
struct Eval<'a> {
    p: &'a Program,
    stage: HwStage,
    defs: HashMap<Value, Site>,
    inputs: HashMap<Value, Reg>,
    /// The one byte of the driver's table the program loads; `None`
    /// while it is being found (any offset is then that dword).
    table_offset: Option<i32>,
}

impl<'a> Eval<'a> {
    fn new(p: &'a Program, stage: HwStage) -> Self {
        let mut defs = HashMap::new();
        for (b, block) in p.blocks.iter().enumerate() {
            for (i, inst) in block.insts.iter().enumerate() {
                for w in &inst.writes {
                    defs.insert(w.1, (b, i));
                }
            }
        }
        let inputs = p.inputs.iter().map(|&(r, v)| (v, r)).collect();
        let mut e = Eval {
            p,
            stage,
            defs,
            inputs,
            table_offset: None,
        };
        e.table_offset = e.table_offsets().first().copied();
        e
    }

    /// The distinct bytes of the driver's table the program loads.
    fn table_offsets(&self) -> Vec<i32> {
        let mut out = Vec::new();
        for inst in self.p.blocks.iter().flat_map(|b| &b.insts) {
            if let Fields::Smem { base, offset, .. } = &inst.fields
                && let Some(&(_, v)) = inst.writes.first()
                && self.table(inst, base, *offset, v, &zero(), DEPTH).is_some()
                && !out.contains(offset)
            {
                out.push(*offset);
            }
        }
        out
    }

    fn inst(&self, s: Site) -> &'a Inst {
        &self.p.blocks[s.0].insts[s.1]
    }

    /// What `v` holds in lane `probe.lane` with the probe's inputs, if the
    /// code that computes it is integer arithmetic this follows.
    fn value(&self, v: Value, probe: &Probe, depth: u32) -> Option<u32> {
        if depth == 0 {
            return None;
        }
        if let Some(&reg) = self.inputs.get(&v) {
            return self.input(reg, probe);
        }
        let site = *self.defs.get(&v)?;
        let inst = self.inst(site);
        let n = inst.name();
        let src = |k: usize| -> Option<u32> {
            match &inst.fields {
                Fields::Valu(f) => {
                    let mut x = self.operand(inst, f.src.get(k)?, probe, depth - 1)?;
                    if let Some(s) = f.sdwa {
                        x = select(x, *s.src_sel.get(k)?, s.sext >> k & 1 != 0)?;
                    }
                    Some(x)
                }
                Fields::Sop { src, .. } => self.operand(inst, src.get(k)?, probe, depth - 1),
                _ => None,
            }
        };
        match &inst.fields {
            Fields::Valu(f) => {
                if f.neg != 0 || f.abs != 0 || f.clamp || f.omod != 0 || f.dpp.is_some() {
                    return None;
                }
                if f.sdwa.is_some_and(|s| s.dst_sel != 6) {
                    return None;
                }
                // The first dword written: the result (not a carry).
                if inst.writes.first()?.1 != v {
                    return None;
                }
            }
            Fields::Sop { .. } => {
                if !matches!(inst.writes.first(), Some(&(Reg::S(_), w)) if w == v) {
                    return None;
                }
            }
            Fields::Smem { base, offset, .. } => {
                return self.table(inst, base, *offset, v, probe, depth);
            }
            _ => return None,
        }
        let s24 = |x: u32| x & 0xFF_FFFF;
        Some(match n {
            "v_mov_b32" | "s_mov_b32" | "v_readfirstlane_b32" => src(0)?,
            "s_movk_i32" => match &inst.fields {
                Fields::Sop { imm: Some(k), .. } => *k as i16 as i32 as u32,
                _ => return None,
            },
            "v_lshlrev_b32" => src(1)? << (src(0)? & 31),
            "v_lshrrev_b32" => src(1)? >> (src(0)? & 31),
            "s_lshl_b32" => src(0)? << (src(1)? & 31),
            "s_lshr_b32" => src(0)? >> (src(1)? & 31),
            "v_and_b32" | "s_and_b32" => src(0)? & src(1)?,
            "v_or_b32" | "s_or_b32" => src(0)? | src(1)?,
            "v_xor_b32" | "s_xor_b32" => src(0)? ^ src(1)?,
            "v_add_nc_u32" | "v_add_u32" | "v_add_co_u32" | "s_add_u32" | "s_add_i32" => {
                src(0)?.wrapping_add(src(1)?)
            }
            "v_sub_nc_u32" | "v_sub_co_u32" | "s_sub_u32" | "s_sub_i32" => {
                src(0)?.wrapping_sub(src(1)?)
            }
            "v_subrev_nc_u32" | "v_subrev_co_u32" => src(1)?.wrapping_sub(src(0)?),
            "v_mul_u32_u24" => s24(src(0)?).wrapping_mul(s24(src(1)?)),
            "v_mul_lo_u32" | "s_mul_i32" => src(0)?.wrapping_mul(src(1)?),
            "v_mad_u32_u24" => s24(src(0)?)
                .wrapping_mul(s24(src(1)?))
                .wrapping_add(src(2)?),
            "v_lshl_add_u32" => (src(0)? << (src(1)? & 31)).wrapping_add(src(2)?),
            "v_add_lshl_u32" => src(0)?.wrapping_add(src(1)?) << (src(2)? & 31),
            "v_lshl_or_b32" => src(0)? << (src(1)? & 31) | src(2)?,
            "v_and_or_b32" => src(0)? & src(1)? | src(2)?,
            "v_or3_b32" => src(0)? | src(1)? | src(2)?,
            "v_add3_u32" => src(0)?.wrapping_add(src(1)?).wrapping_add(src(2)?),
            "v_bfe_u32" => {
                let (x, o, w) = (src(0)?, src(1)? & 31, src(2)? & 31);
                if w == 0 {
                    0
                } else {
                    x >> o & (u32::MAX >> (32 - w))
                }
            }
            "s_bfe_u32" => {
                let (x, f) = (src(0)?, src(1)?);
                let (o, w) = (f & 31, f >> 16 & 0x7F);
                if w == 0 {
                    0
                } else {
                    x >> o & (u32::MAX >> (32 - w.min(32)))
                }
            }
            "s_lshl1_add_u32" => (src(0)? << 1).wrapping_add(src(1)?),
            "s_lshl2_add_u32" => (src(0)? << 2).wrapping_add(src(1)?),
            "s_lshl3_add_u32" => (src(0)? << 3).wrapping_add(src(1)?),
            "s_lshl4_add_u32" => (src(0)? << 4).wrapping_add(src(1)?),
            // With every lane counted: the lane's index plus the operand.
            "v_mbcnt_lo_u32_b32" if src(0)? == u32::MAX => probe.lane.wrapping_add(src(1)?),
            "v_mbcnt_hi_u32_b32" if src(0)? == u32::MAX => src(1)?,
            _ => return None,
        })
    }

    fn input(&self, reg: Reg, probe: &Probe) -> Option<u32> {
        let v = &probe.vertices;
        let user = |k: u8| Reg::S(self.stage.user_sgpr(k));
        match (self.stage, reg) {
            (_, r) if r == user(0) => Some(TABLE.0),
            (_, r) if r == user(1) => Some(TABLE.1),
            (HwStage::Gs, Reg::V(0)) => Some(v[0] | v[1] << 16),
            (HwStage::Gs, Reg::V(1)) => Some(v[2] | v[3] << 16),
            (HwStage::Gs, Reg::V(4)) => Some(v[4] | v[5] << 16),
            (HwStage::Gs, Reg::S(2)) => Some(probe.wave),
            (HwStage::Vs, Reg::V(0)) => Some(probe.vertex),
            _ => None,
        }
    }

    /// A load of one dword from the driver's table (its address in user
    /// data 0 and the low byte of user data 1): the probe's dword.
    fn table(
        &self,
        inst: &Inst,
        base: &Operand,
        offset: i32,
        v: Value,
        probe: &Probe,
        depth: u32,
    ) -> Option<u32> {
        let Operand::Scalar { code, .. } = *base else {
            return None;
        };
        if inst.name() != "s_load_dword" || inst.writes.first()?.1 != v {
            return None;
        }
        if let Fields::Smem { soffset, .. } = &inst.fields
            && !matches!(soffset, Operand::Scalar { code: NULL, .. })
        {
            return None;
        }
        let lo = self.value(inst.read(Reg::S(code))?, probe, depth - 1)?;
        let hi = self.value(inst.read(Reg::S(code + 1))?, probe, depth - 1)?;
        let known = self.table_offset.is_none_or(|t| t == offset);
        (lo == TABLE.0 && hi & 0xFF == TABLE.1 && known).then_some(probe.base)
    }

    fn operand(&self, inst: &Inst, op: &Operand, probe: &Probe, depth: u32) -> Option<u32> {
        match *op {
            Operand::Scalar { code: NULL, .. } => Some(0),
            Operand::Scalar { code, .. } => self.value(inst.read(Reg::S(code))?, probe, depth),
            Operand::Vector { reg, .. } => self.value(inst.read(Reg::V(reg))?, probe, depth),
            Operand::Const(c) => inline(c),
            Operand::Literal(x) => Some(x),
            Operand::Attr { .. } => None,
        }
    }

    /// The byte address each dword of an LDS access at `site` reads or
    /// writes, from the probe's inputs.
    fn lds(&self, site: Site, probe: &Probe) -> Option<Vec<u32>> {
        let inst = self.inst(site);
        let Fields::Ds {
            addr: Some(Operand::Vector { reg, .. }),
            offset0,
            offset1,
            gds: false,
            ..
        } = &inst.fields
        else {
            return None;
        };
        let a = self.value(inst.read(Reg::V(*reg))?, probe, DEPTH)?;
        let mut out = Vec::new();
        for (off, dwords) in ds_elements(inst.name(), *offset0, *offset1)? {
            for k in 0..dwords {
                out.push(a.wrapping_add(off).wrapping_add(4 * k));
            }
        }
        Some(out)
    }
}

/// An SDWA source select: a byte or word of `x`, zero- or sign-extended.
fn select(x: u32, sel: u8, sext: bool) -> Option<u32> {
    let (shift, bits) = match sel {
        0..=3 => (8 * u32::from(sel), 8),
        4 | 5 => (16 * u32::from(sel - 4), 16),
        6 => return Some(x),
        _ => return None,
    };
    let v = x >> shift & ((1 << bits) - 1);
    Some(if sext && v >> (bits - 1) != 0 {
        v | !((1 << bits) - 1)
    } else {
        v
    })
}

/// The (byte offset, dwords) of each element a 32-bit LDS read or write
/// moves, as `lower` takes them; `None` for other LDS operations.
fn ds_elements(name: &str, offset0: u8, offset1: u8) -> Option<Vec<(u32, u32)>> {
    let rest = name
        .strip_prefix("ds_read")
        .or(name.strip_prefix("ds_write"))?;
    let width = |w: &str| match w {
        "b32" => Some(1),
        "b64" => Some(2),
        "b96" => Some(3),
        "b128" => Some(4),
        _ => None,
    };
    if let Some(pair) = rest.strip_prefix('2') {
        let (st64, w) = match pair.strip_prefix("st64_") {
            Some(w) => (true, w),
            None => (false, pair.strip_prefix('_')?),
        };
        let w = width(w)?;
        let scale = 4 * w * if st64 { 64 } else { 1 };
        return Some(vec![
            (u32::from(offset0) * scale, w),
            (u32::from(offset1) * scale, w),
        ]);
    }
    let w = width(rest.strip_prefix('_')?)?;
    Some(vec![(u32::from(offset0) | u32::from(offset1) << 8, w)])
}

/// Raises a GS program's ring accesses, messages and copy shader.
pub(crate) fn raise(
    p: &Program,
    native: &NativeProgram,
    gs: &GeometryIo,
) -> Result<Geometry, ShaderError> {
    let unsupported = |offset, what| ShaderError::Unsupported { offset, what };
    let e = Eval::new(p, HwStage::Gs);
    let copy = copy_shader(native, gs)?;
    // One dword of the table, the same in both programs: the ring's base.
    let both = e.table_offset.zip(copy.table_offset);
    if e.table_offsets().len() > 1 || both.is_some_and(|(a, b)| a != b) {
        return Err(unsupported(
            0,
            "a geometry program that reads more of the driver's table",
        ));
    }
    let item = gs.max_vertices * gs.vertex_dwords;
    let mut out = Geometry {
        max_vertices: gs.max_vertices,
        vertex_dwords: gs.vertex_dwords,
        copies: copy.exports,
        ..Geometry::default()
    };
    for (b, block) in p.blocks.iter().enumerate() {
        for (i, inst) in block.insts.iter().enumerate() {
            let site = (b, i);
            let at = inst.machine.offset;
            let name = inst.name();
            match &inst.fields {
                Fields::Ds { .. } if name.starts_with("ds_read") => {
                    let a = PROBES.map(|probe| e.lds(site, &probe));
                    let [Some(a0), Some(a1)] = a else {
                        return Err(unsupported(
                            at,
                            "an LDS read of a geometry program not from its input vertices",
                        ));
                    };
                    let mut dwords = Vec::new();
                    for (x0, x1) in a0.into_iter().zip(a1) {
                        let found = (0..MAX_INPUT_VERTICES).find_map(|k| {
                            let c = x0.wrapping_sub(4 * PROBES[0].vertices[k]);
                            let same = x1.wrapping_sub(4 * PROBES[1].vertices[k]) == c;
                            (same && c < 0x1000 && c % 4 == 0).then_some((k as u8, c / 4))
                        });
                        dwords.push(found.ok_or(unsupported(
                            at,
                            "an LDS read of a geometry program not from its input vertices",
                        ))?);
                    }
                    out.reads.push((site, dwords));
                }
                Fields::Ds { data, .. } if name.starts_with("ds_write") && !data.is_empty() => {
                    let what = "an LDS write of a geometry program not into its GS-VS ring item";
                    let at_probe = |f: &dyn Fn(&mut Probe)| {
                        let mut probe = zero();
                        f(&mut probe);
                        e.lds(site, &probe).ok_or(unsupported(at, what))
                    };
                    let z = at_probe(&|_| {})?;
                    // The next lane's item, the base, the wave's offset: each
                    // moves the whole access by its share; the input vertices
                    // do not move it.
                    let moved = [
                        (at_probe(&|q| q.lane = 1)?, 4 * item),
                        (at_probe(&|q| q.base = 1)?, copy.base_bytes),
                        (at_probe(&|q| q.wave = 0x40)?, 0x40),
                        (at_probe(&|q| q.vertices = PROBES[1].vertices)?, 0),
                    ];
                    if moved
                        .iter()
                        .any(|(m, by)| m.iter().zip(&z).any(|(a, b)| a.wrapping_sub(*b) != *by))
                    {
                        return Err(unsupported(at, what));
                    }
                    let dwords = z
                        .iter()
                        .map(|&x| (x % 4 == 0 && x / 4 < item).then_some(x / 4))
                        .collect::<Option<Vec<u32>>>()
                        .ok_or(unsupported(at, what))?;
                    out.writes.push((site, dwords));
                }
                Fields::Ds { .. } => {
                    return Err(unsupported(at, "an LDS operation of a geometry program"));
                }
                Fields::Sopp { imm } if name == "s_sendmsg" || name == "s_sendmsghalt" => {
                    let (id, op, stream) = (imm & 0xF, imm >> 4 & 3, imm >> 8 & 3);
                    let m = match (id, op) {
                        (MSG_GS, GS_OP_EMIT) => Message::Emit,
                        (MSG_GS, GS_OP_CUT) => Message::Cut,
                        (MSG_GS, GS_OP_EMIT_CUT) => Message::EmitCut,
                        (MSG_GS_DONE, _) => Message::Done,
                        _ => return Err(unsupported(at, "a message of a geometry program")),
                    };
                    if stream != 0 && m != Message::Done {
                        return Err(unsupported(
                            at,
                            "a geometry program's output stream other than 0",
                        ));
                    }
                    out.messages.push((site, m));
                }
                _ => {}
            }
        }
    }
    // The translation keeps the ring and the vertices emitted in values as
    // it goes (`lower`): each access and message must run once, on the way
    // to every end of the program.
    let blocks = out
        .writes
        .iter()
        .map(|w| w.0)
        .chain(out.messages.iter().map(|m| m.0));
    if let Some(s) = blocks.map(|s| s.0).find(|&b| !once_on_every_path(p, b)) {
        let at = p.blocks[s].offset;
        return Err(unsupported(
            at,
            "a geometry program that emits in a branch or loop",
        ));
    }
    if let Some((s, what)) = tainted_use(p, &e, &out) {
        let at = p.blocks[s.0].insts[s.1].machine.offset;
        return Err(unsupported(at, what));
    }
    Ok(out)
}

/// Whether block `b` dominates every end of the program and lies on no
/// cycle.
fn once_on_every_path(p: &Program, b: usize) -> bool {
    let order = ir::reverse_postorder(&p.blocks);
    let mut index = vec![usize::MAX; p.blocks.len()];
    for (n, &k) in order.iter().enumerate() {
        index[k] = n;
    }
    let idom = ir::idoms(&p.blocks, &order, &index);
    let dominates = |mut d: usize| loop {
        if d == b {
            break true;
        }
        match idom.get(d) {
            Some(&up) if up != d && up != usize::MAX => d = up,
            _ => break false,
        }
    };
    let every_end = (p.blocks.iter().enumerate())
        .filter(|(_, k)| matches!(k.exit, ir::Exit::End | ir::Exit::Indirect))
        .all(|(e, k)| k.exit == ir::Exit::End && dominates(e));
    // On a cycle: reached again from its successors.
    let mut seen = vec![false; p.blocks.len()];
    let mut work = p.blocks[b].exit.successors();
    while let Some(k) = work.pop() {
        if k == b {
            return false;
        }
        if !std::mem::replace(&mut seen[k], true) {
            work.extend(p.blocks[k].exit.successors());
        }
    }
    every_end
}

/// Where a value made of the ring inputs (the vertex offsets in v0, v1 and
/// v4, the wave's offset in s2, the driver table's dword) reaches anything
/// but integer arithmetic and the addresses of raised LDS accesses, which
/// the translation reads 0 for. Values flow through the operands an
/// instruction reads (`operand_reads`), and through the old value a write
/// keeps outside the entry block.
///
/// shortcut: in the entry block a write's old value (kept in lanes not in
/// EXEC) is not followed; every lane the program starts with stays in EXEC
/// there in the programs seen.
fn tainted_use(p: &Program, e: &Eval, g: &Geometry) -> Option<(Site, &'static str)> {
    let mut tainted: HashSet<Value> = p
        .inputs
        .iter()
        .filter(|(r, _)| matches!(r, Reg::V(0 | 1 | 4) | Reg::S(2)))
        .map(|&(_, v)| v)
        .collect();
    let raised: HashSet<Site> = g
        .reads
        .iter()
        .map(|r| r.0)
        .chain(g.writes.iter().map(|w| w.0))
        .collect();
    loop {
        let before = tainted.len();
        for (b, block) in p.blocks.iter().enumerate() {
            for phi in &block.phis {
                if phi.args.iter().any(|a| tainted.contains(a)) {
                    tainted.insert(phi.value);
                }
            }
            for (i, inst) in block.insts.iter().enumerate() {
                let site = (b, i);
                let mut reads: Vec<Value> = operand_reads(p, inst)
                    .iter()
                    .filter_map(|&r| inst.read(r))
                    .collect();
                if b != 0 {
                    reads.extend(inst.reads.iter().map(|r| r.1));
                }
                let table = matches!(inst.fields, Fields::Smem { .. })
                    && inst
                        .writes
                        .first()
                        .is_some_and(|w| e.value(w.1, &zero(), DEPTH).is_some());
                if !table && !reads.iter().any(|v| tainted.contains(v)) {
                    continue;
                }
                match &inst.fields {
                    Fields::Sop { .. } | Fields::Valu(_) => {
                        if inst
                            .writes
                            .iter()
                            .any(|w| matches!(w.0, Reg::S(EXEC) | Reg::S(127)))
                        {
                            return Some((
                                site,
                                "a lane mask made of a geometry program's ring offsets",
                            ));
                        }
                        tainted.extend(inst.writes.iter().map(|w| w.1));
                    }
                    Fields::Smem { .. } if table => tainted.extend(inst.writes.iter().map(|w| w.1)),
                    Fields::Ds {
                        addr: Some(Operand::Vector { reg, .. }),
                        ..
                    } if raised.contains(&site) => {
                        let address = inst.read(Reg::V(*reg));
                        let other = operand_reads(p, inst)
                            .iter()
                            .filter(|&&r| r != Reg::V(*reg))
                            .filter_map(|&r| inst.read(r))
                            .any(|v| tainted.contains(&v) && Some(v) != address);
                        if other {
                            return Some((
                                site,
                                "a geometry program's ring offset written as data",
                            ));
                        }
                    }
                    _ => return Some((site, "a geometry program's ring offset used as data")),
                }
            }
        }
        if tainted.len() == before {
            return None;
        }
    }
}

/// What a copy shader takes from the ring: each export, and how the
/// driver table's dword moves its addresses (bytes per unit).
struct Copy {
    exports: Vec<(Target, [Option<Copied>; 4])>,
    base_bytes: u32,
    table_offset: Option<i32>,
}

/// Raises the copy shader: each channel it exports is a dword of the
/// emitted vertex (in the GS-VS ring) or a constant.
fn copy_shader(native: &NativeProgram, gs: &GeometryIo) -> Result<Copy, ShaderError> {
    let unsupported = |what| ShaderError::Unsupported { offset: 0, what };
    let p = ir::build(&gs.copy, native.wave64, 256)?;
    let e = Eval::new(&p, HwStage::Vs);
    let mut exports: Vec<(Target, [Option<Copied>; 4])> = Vec::new();
    let mut base_bytes = None;
    for block in &p.blocks {
        for inst in &block.insts {
            let Fields::Exp {
                target, src, compr, ..
            } = &inst.fields
            else {
                if matches!(
                    inst.fields,
                    Fields::Buffer { .. } | Fields::Flat { .. } | Fields::Image { .. }
                ) || matches!(inst.fields, Fields::Ds { .. })
                    && !inst.name().starts_with("ds_read")
                {
                    return Err(unsupported(
                        "a copy shader with memory accesses besides its ring reads",
                    ));
                }
                continue;
            };
            if *compr {
                return Err(unsupported("a compressed copy shader export"));
            }
            let target = match *target {
                12 => Target::Position(0),
                t @ 13..=16 => Target::Position(t - 12),
                t @ 32..=63 => {
                    let o = gs
                        .outputs
                        .iter()
                        .find(|o| o.param == t - 32)
                        .ok_or(unsupported(
                            "a copy shader export the stage data does not list",
                        ))?;
                    if o.fp16 || o.pack2 {
                        return Err(unsupported("a 16-bit geometry output"));
                    }
                    Target::Param {
                        usage: o.usage,
                        row: o.row,
                    }
                }
                _ => return Err(unsupported("a copy shader export target")),
            };
            if exports.iter().any(|x| x.0 == target) {
                return Err(unsupported("a copy shader that exports a target twice"));
            }
            let mut values = [None; 4];
            for (ch, s) in src.iter().enumerate() {
                let Some(Operand::Vector { reg, .. }) = *s else {
                    continue;
                };
                let v = inst
                    .read(Reg::V(reg))
                    .ok_or(unsupported("a copy shader export"))?;
                values[ch] = Some(copied(&e, v, &mut base_bytes, gs)?);
            }
            exports.push((target, values));
        }
    }
    if !exports.iter().any(|x| x.0 == Target::Position(0)) {
        return Err(unsupported("a copy shader without a position"));
    }
    // An unused base moves nothing (the GS's addresses then may not move).
    if e.table_offsets().len() > 1 {
        return Err(unsupported(
            "a copy shader that reads more of the driver's table",
        ));
    }
    Ok(Copy {
        exports,
        base_bytes: base_bytes.unwrap_or(0),
        table_offset: e.table_offset,
    })
}

/// A copy shader's exported value: a dword of the vertex or a constant.
fn copied(
    e: &Eval,
    v: Value,
    base_bytes: &mut Option<u32>,
    gs: &GeometryIo,
) -> Result<Copied, ShaderError> {
    let unsupported = |what| ShaderError::Unsupported { offset: 0, what };
    let mut v = v;
    // Through plain copies.
    while let Some(&s) = e.defs.get(&v) {
        let inst = e.inst(s);
        match &inst.fields {
            Fields::Valu(f)
                if inst.name() == "v_mov_b32"
                    && f.sdwa.is_none()
                    && f.dpp.is_none()
                    && f.neg == 0
                    && f.abs == 0
                    && matches!(f.src.first(), Some(Operand::Vector { .. })) =>
            {
                let Some(&Operand::Vector { reg, .. }) = f.src.first() else {
                    unreachable!()
                };
                v = inst
                    .read(Reg::V(reg))
                    .ok_or(unsupported("a copy shader export"))?;
            }
            _ => break,
        }
    }
    let Some(&site) = e.defs.get(&v) else {
        return Err(unsupported("a copy shader export of an input"));
    };
    let inst = e.inst(site);
    if !inst.name().starts_with("ds_read") {
        let [a, b] = PROBES.map(|probe| e.value(v, &probe, DEPTH));
        return match (a, b) {
            (Some(a), Some(b)) if a == b => Ok(Copied::Const(a)),
            _ => Err(unsupported(
                "a copy shader export the translator does not follow",
            )),
        };
    }
    let n = inst
        .writes
        .iter()
        .position(|w| w.1 == v)
        .ok_or(unsupported("a copy shader export"))?;
    let what = "a copy shader read not from its vertex";
    let at = |f: &dyn Fn(&mut Probe)| {
        let mut probe = zero();
        f(&mut probe);
        e.lds(site, &probe)
            .and_then(|a| a.get(n).copied())
            .ok_or(unsupported(what))
    };
    let z = at(&|_| {})?;
    let vertex = at(&|q| q.vertex = 0x100)?;
    let base = at(&|q| q.base = 1)?.wrapping_sub(z);
    if vertex.wrapping_sub(z) != 0x400 || *base_bytes.get_or_insert(base) != base {
        return Err(unsupported(what));
    }
    if z % 4 != 0 || z / 4 >= gs.vertex_dwords {
        return Err(unsupported(what));
    }
    Ok(Copied::Ring(z / 4))
}
