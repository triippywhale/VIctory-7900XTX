//! Geometry programs (the GS half of a legacy merged ES+GS stage, with the
//! copy shader of its record) as host geometry shaders: synthetic programs
//! (assembled by hand for gfx1030, wave64) that pass each input triangle
//! through and give its vertices a colour.

use super::HostShader;
#[cfg(windows)]
use super::bitcode::{Attr, Term};
use super::container::{self, SigElement};
#[cfg(windows)]
use super::shader::Shader;
#[cfg(windows)]
use super::validator;
use crate::dxbc::{Container, FourCC};
use crate::native::{GeometryIo, HwStage, NativeProgram, StageIo, Usage, VsOutput};
use crate::{ShaderError, io, resource, spill, structure};

/// GS-VS ring dwords per vertex.
const VERTEX: u32 = 8;
/// The colour each emitted vertex gets (dwords 4-6 of its ring vertex; the
/// copy shader adds alpha 1).
const COLOUR: [f32; 4] = [0.25, 0.5, 1.0, 1.0];

/// The geometry program: the GS lane count made into EXEC, the three input
/// vertices' positions (row 0) read from the ES-GS ring at the offsets in
/// v0 and v1, the lane's GS-VS ring item at `lane * 96` plus s2 plus the
/// driver table's dword (at 0xb8 of the table in user data 0-1) times 256;
/// each vertex written (position, colour) and emitted; a cut, then done.
fn gs_words() -> Vec<u32> {
    let mut w = vec![
        0x9380_FF03,
        0x0008_0008, // s_bfe_u32 s0, s3, 0x80008 (GS lanes)
        0x92FE_8000, // s_bfm_b64 exec, s0, 0
        0xF400_0104,
        0xFA00_00B8, // s_load_dword s4, s[8:9], 0xb8 (driver table)
        0x3414_0082, // v_lshlrev_b32 v10, 2, v0
        0x3614_14FF,
        0x0003_FFFC, // v_and_b32 v10, 0x3fffc, v10 (vertex 0)
        0x2C16_0090, // v_lshrrev_b32 v11, 16, v0
        0x3416_1682, // v_lshlrev_b32 v11, 2, v11 (vertex 1)
        0x3418_0282, // v_lshlrev_b32 v12, 2, v1
        0x3618_18FF,
        0x0003_FFFC, // v_and_b32 v12, 0x3fffc, v12 (vertex 2)
        0xD8DC_0100,
        0x1400_000A, // ds_read2_b32 v[20:21], v10 offset1:1
        0xD8DC_0302,
        0x1600_000A, // ds_read2_b32 v[22:23], v10 offset0:2 offset1:3
        0xD8DC_0100,
        0x1800_000B, // ds_read2_b32 v[24:25], v11 offset1:1
        0xD8DC_0302,
        0x1A00_000B, // ds_read2_b32 v[26:27], v11 offset0:2 offset1:3
        0xD8DC_0100,
        0x1C00_000C, // ds_read2_b32 v[28:29], v12 offset1:1
        0xD8DC_0302,
        0x1E00_000C, // ds_read2_b32 v[30:31], v12 offset0:2 offset1:3
        0xD765_000D,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v13, -1, 0
        0xD766_000D,
        0x0002_1AC1, // v_mbcnt_hi_u32_b32 v13, -1, v13
        0x161A_1AFF,
        0x0000_0060, // v_mul_u32_u24 v13, 96, v13 (the lane's item)
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x8F04_8804, // s_lshl_b32 s4, s4, 8
        0x8004_0204, // s_add_u32 s4, s4, s2
        0x4A1A_1A04, // v_add_nc_u32 v13, s4, v13
        0x7E20_02FF,
        0x3E80_0000, // v_mov_b32 v16, 0.25
        0x7E1C_02F0, // v_mov_b32 v14, 0.5
        0x7E1E_02F2, // v_mov_b32 v15, 1.0
    ];
    // Vertex k at dword 8k: position (4 dwords from its two reads), colour.
    for (k, regs) in [(0u32, 20u32), (1, 24), (2, 28)] {
        let d = 8 * k;
        let write2 =
            |d0: u32, a: u32, b: u32| [0xD838_0000 | (d0 + 1) << 8 | d0, b << 16 | a << 8 | 13];
        w.extend(write2(d, regs, regs + 1));
        w.extend(write2(d + 2, regs + 2, regs + 3));
        w.extend(write2(d + 4, 16, 14));
        w.extend([0xD834_0000 | (d + 6) * 4, 15 << 8 | 13]); // ds_write_b32 v13, v15
        w.push(0xBF90_0022); // s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)
    }
    w.extend([
        0xBF90_0012, // s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 0)
        0xBF8C_0000, // s_waitcnt 0
        0xBF90_0003, // s_sendmsg sendmsg(MSG_GS_DONE, GS_OP_NOP)
        0xBF81_0000, // s_endpgm
    ]);
    w
}

/// The copy shader: the vertex at the ring offset in v0 (plus the table's
/// dword times 256), exported as the position and parameter 0.
const COPY: &[u32] = &[
    0xF400_0100,
    0xFA00_00B8, // s_load_dword s4, s[0:1], 0xb8
    0x3402_0082, // v_lshlrev_b32 v1, 2, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0x8F04_8804, // s_lshl_b32 s4, s4, 8
    0x4A02_0204, // v_add_nc_u32 v1, s4, v1
    0xD8DC_0100,
    0x0200_0001, // ds_read2_b32 v[2:3], v1 offset1:1
    0xD8DC_0302,
    0x0400_0001, // ds_read2_b32 v[4:5], v1 offset0:2 offset1:3
    0xD8DC_0504,
    0x0600_0001, // ds_read2_b32 v[6:7], v1 offset0:4 offset1:5
    0xD8D8_0018,
    0x0800_0001, // ds_read_b32 v8, v1 offset:24
    0x7E12_02F2, // v_mov_b32 v9, 1.0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xF800_08CF,
    0x0504_0302, // exp pos0 v2, v3, v4, v5 done
    0xF800_020F,
    0x0908_0706, // exp param0 v6, v7, v8, v9
    0xBF81_0000, // s_endpgm
];

fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn element(name: &str, index: u32, kind: u8, row: u32) -> SigElement {
    SigElement {
        name: name.into(),
        index,
        kind,
        comp: 9,
        interp: 0,
        row: Some(row),
        col: 0,
        cols: 4,
        usage: 0xF,
        half: false,
    }
}

fn native() -> NativeProgram {
    NativeProgram {
        stage: HwStage::Gs,
        wave64: true,
        vgprs: 32,
        layout: Vec::new(),
        used: Vec::new(),
        io: StageIo::Gs(GeometryIo {
            outputs: vec![VsOutput {
                usage: Usage::Generic,
                row: 1,
                param: 0,
                fp16: false,
                pack2: false,
            }],
            max_vertices: 3,
            vertex_dwords: VERTEX,
            copy: bytes(COPY),
        }),
        code: bytes(&gs_words()),
    }
}

/// `PSV0` of a geometry shader: triangles in, triangle strips out, stream
/// 0, three vertices (no signature elements listed).
fn psv0() -> Vec<u8> {
    let mut info = [0u8; 52];
    info[0..4].copy_from_slice(&3u32.to_le_bytes());
    info[4..8].copy_from_slice(&5u32.to_le_bytes());
    info[8..12].copy_from_slice(&1u32.to_le_bytes());
    info[24] = 2;
    info[26..28].copy_from_slice(&3u16.to_le_bytes());
    let mut out = 52u32.to_le_bytes().to_vec();
    out.extend(info);
    out.extend(0u32.to_le_bytes());
    out
}

fn container() -> Vec<u8> {
    let isg1 = container::signature(&[element("SV_Position", 0, 3, 0)], true);
    let osg1 = container::signature(
        &[
            element("SV_Position", 0, 3, 0),
            element("TEXCOORD", 0, 0, 1),
        ],
        false,
    );
    crate::dxbc::build(&[
        (FourCC::ISG1, &isg1),
        (FourCC::OSG1, &osg1),
        (FourCC::PSV0, &psv0()),
    ])
}

fn geometry_shader(native: &NativeProgram) -> Result<HostShader, ShaderError> {
    let bytes = container();
    let c = Container::parse(&bytes)?;
    let p = spill::build(native)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, native)?;
    super::emit(&c, native, &p, &st, &io, &res)
}

/// What the program reads and writes: each read a vertex's position,
/// each write a dword of the item, three emits and a cut; the copy
/// shader's exports are ring dwords and alpha 1. The driver table's load
/// traces to the table, and the GS lane count is patched to all lanes.
#[test]
fn ring_accesses_are_raised() {
    let n = native();
    assert_ne!(n.patched_code(), n.code, "the GS lane count is patched");
    let p = spill::build(&n).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &n.layout, n.stage, Some(&st)).unwrap();
    assert!(res.scalar.iter().any(|s| s.1.param == resource::SYSTEM));
    let g = io::raise(&p, &n).unwrap().geometry.unwrap();
    let reads: Vec<_> = g.reads.iter().flat_map(|r| r.1.clone()).collect();
    let want: Vec<(u8, u32)> = (0..3).flat_map(|k| (0..4).map(move |d| (k, d))).collect();
    assert_eq!(reads, want);
    let mut written: Vec<u32> = g.writes.iter().flat_map(|w| w.1.clone()).collect();
    written.sort_unstable();
    let want: Vec<u32> = (0..3)
        .flat_map(|k| (0..7).map(move |d| 8 * k + d))
        .collect();
    assert_eq!(written, want);
    let messages: Vec<_> = g.messages.iter().map(|m| m.1).collect();
    use io::Message::*;
    assert_eq!(messages, [Emit, Emit, Emit, Cut, Done]);
    use io::Copied::*;
    assert_eq!(
        g.copies,
        [
            (
                io::Target::Position(0),
                [Some(Ring(0)), Some(Ring(1)), Some(Ring(2)), Some(Ring(3))]
            ),
            (
                io::Target::Param {
                    usage: Usage::Generic,
                    row: 1
                },
                [
                    Some(Ring(4)),
                    Some(Ring(5)),
                    Some(Ring(6)),
                    Some(Const(1.0f32.to_bits()))
                ]
            ),
        ]
    );
}

/// A ring offset used as anything but an address, a GS-VS ring write at
/// an address that does not move with the lane's item, or an emit in a
/// branch is refused.
#[test]
fn what_the_rings_cannot_mean_is_refused() {
    let refused = |words: Vec<u32>| {
        let mut n = native();
        n.code = bytes(&words);
        let p = spill::build(&n).unwrap();
        io::raise(&p, &n).unwrap_err()
    };
    // The vertex offset (v10) written as data.
    let mut w = gs_words();
    let at = w.iter().position(|&x| x == 0xD834_0018).unwrap();
    w[at + 1] = 10 << 8 | 13;
    assert!(
        matches!(refused(w), ShaderError::Unsupported { what, .. } if what.contains("ring offset"))
    );
    // The item without the lane's share (no v_mul_u32_u24).
    let mut w = gs_words();
    let at = w.iter().position(|&x| x == 0x161A_1AFF).unwrap();
    w[at] = 0x7E1A_0280; // v_mov_b32 v13, 0
    w[at + 1] = 0xBF80_0000; // s_nop 0
    assert!(
        matches!(refused(w), ShaderError::Unsupported { what, .. } if what.contains("GS-VS ring"))
    );
    // A second stream.
    let mut w = gs_words();
    let at = w.iter().position(|&x| x == 0xBF90_0022).unwrap();
    w[at] = 0xBF90_0122;
    assert!(matches!(refused(w), ShaderError::Unsupported { what, .. } if what.contains("stream")));
}

/// The host shader is a geometry shader of triangles in and strips out,
/// with the container's signatures, and the validator, where there is one,
/// accepts it.
#[test]
fn the_geometry_program_is_a_geometry_shader() {
    let gs = geometry_shader(&native()).unwrap();
    assert_eq!(gs.stage, HwStage::Gs);
    // The table's dword is the driver's, not user data the title binds.
    assert!(gs.user_data.is_empty());
    let c = Container::parse(&gs.dxil).unwrap();
    // Program version: geometry (2), shader model 6.6.
    assert_eq!(c.part(FourCC::DXIL).unwrap().data[..4], [0x66, 0, 2, 0]);
    let psv = c.part(FourCC::PSV0).unwrap().data;
    assert_eq!(psv[4..17], [3, 0, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 1]);
    assert_eq!(psv[28..32], [2, 0, 3, 0]);
    #[cfg(windows)]
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut gs.dxil.clone()).unwrap();
    }
}

/// POSITION to SV_Position.
#[cfg(windows)]
fn vertex_shader() -> HostShader {
    let mut s = Shader::new(false);
    let b = s.entry;
    let undef = s.m.undef(s.t.i32);
    for col in 0..4u8 {
        let args = [s.m.i32(0), s.m.i32(0), s.m.i8(col), undef];
        let v = s.op(b, "loadInput.f32", Attr::ReadNone, s.t.f32, 4, &args);
        let out = [s.m.i32(0), s.m.i32(0), s.m.i8(col), v];
        s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &out);
    }
    s.inputs.push(element("POSITION", 0, 0, 0));
    s.outputs.push(SigElement {
        interp: 4,
        ..element("SV_Position", 0, 3, 0)
    });
    s.m.terminate(b, Term::Ret);
    host_shader(HwStage::Vs, s)
}

/// TEXCOORD0 (row 1) to target 0.
#[cfg(windows)]
fn pixel_shader() -> HostShader {
    let mut s = Shader::new(true);
    let b = s.entry;
    let undef = s.m.undef(s.t.i32);
    for col in 0..4u8 {
        let args = [s.m.i32(1), s.m.i32(0), s.m.i8(col), undef];
        let v = s.op(b, "loadInput.f32", Attr::ReadNone, s.t.f32, 4, &args);
        let out = [s.m.i32(0), s.m.i32(0), s.m.i8(col), v];
        s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &out);
    }
    s.inputs.push(SigElement {
        interp: 4,
        usage: 0,
        ..element("SV_Position", 0, 3, 0)
    });
    s.inputs.push(SigElement {
        interp: 2,
        ..element("TEXCOORD", 0, 0, 1)
    });
    s.outputs.push(element("SV_Target", 0, 16, 0));
    s.m.terminate(b, Term::Ret);
    host_shader(HwStage::Ps, s)
}

#[cfg(windows)]
fn host_shader(stage: HwStage, s: Shader) -> HostShader {
    HostShader {
        stage,
        dxil: s.finish(),
        user_data: Vec::new(),
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
        root_uavs: Vec::new(),
        root_indexed_views: Vec::new(),
    }
}

/// On the host device: a triangle over the whole target, drawn through the
/// translated geometry shader, comes out whole and in the colour the
/// program gives its vertices.
#[test]
#[cfg(windows)]
fn the_geometry_shader_draws_what_the_program_emits() {
    use super::host::{Draw, Host};
    use windows::Win32::Graphics::Direct3D12::*;
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 device; skipped");
        return;
    };
    let (vs, gs, ps) = (
        vertex_shader(),
        geometry_shader(&native()).unwrap(),
        pixel_shader(),
    );
    let root = host
        .root_signature_for(&[
            (&vs, D3D12_SHADER_VISIBILITY_VERTEX),
            (&gs, D3D12_SHADER_VISIBILITY_GEOMETRY),
            (&ps, D3D12_SHADER_VISIBILITY_PIXEL),
        ])
        .unwrap();
    let pso = host
        .pipeline_with_gs(&root, &vs.dxil, Some(&gs.dxil), &ps.dxil)
        .unwrap();
    let vertices = [
        -1.0, -1.0, 0.5, 1.0, 3.0, -1.0, 0.5, 1.0, -1.0, 3.0, 0.5, 1.0,
    ];
    let d = Draw {
        roots: Vec::new(),
        textures: &[],
        samplers: &[],
        vertices: &vertices,
        stride: 16,
        size: (8, 8),
        mesh_groups: 0,
    };
    let texels = host.draw(&root, &pso, &d).unwrap();
    assert_eq!(texels.len(), 64);
    for (i, t) in texels.iter().enumerate() {
        assert_eq!(*t, COLOUR, "texel {i}");
    }
}
