//! Compute waves can contain threads from different rows of a group.

use windows::Win32::Graphics::Direct3D12::{
    D3D12_FEATURE_D3D12_OPTIONS1, D3D12_FEATURE_DATA_D3D12_OPTIONS1,
};

use super::bitcode::{Attr, Bin, Term, V};
use super::host::{Dispatch, Host, Root, View};
use super::lower::{self, Lowered};
use super::shader::Shader;
use super::{HostShader, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const SHAPES: [[u32; 3]; 8] = [
    [64, 1, 1],
    [8, 4, 1],
    [32, 2, 1],
    [32, 8, 1],
    [24, 8, 1],
    [12, 12, 1],
    [10, 10, 1],
    [8, 3, 3],
];

/// Add 1 + the console lane to the output at the thread's flat ID. Positive
/// increments detect both missing IDs and overlapping writes. The lane count
/// distinguishes a bijection that assigns the wrong wave position.
fn counter_program([x, y, z]: [u32; 3], lds_bytes: u32, wave64: bool) -> NativeProgram {
    let mut code = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0A02,
        0xFA00_0000, // s_load_dwordx4 s[40:43], s[4:5], 0
        0xD765_0005,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v5, -1, 0
    ];
    if wave64 {
        code.extend([0xD766_0005, 0x0002_0AC1]); // v_mbcnt_hi_u32_b32 v5, -1, v5
    }
    code.extend([
        0x0B << 25 | 3 << 17 | 2 << 9 | (128 + y), // v_mul_u32_u24 v3, y_size, v2
        0x4A06_0701,                               // v_add_nc_u32 v3, v1, v3
        0x0B << 25 | 3 << 17 | 3 << 9 | (128 + x), // v_mul_u32_u24 v3, x_size, v3
        0x4A06_0700,                               // v_add_nc_u32 v3, v0, v3
        0x3408_0682,                               // v_lshlrev_b32 v4, 2, v3
        0x4A0A_0A81,                               // v_add_nc_u32 v5, 1, v5
        0xBF8C_C07F,                               // s_waitcnt lgkmcnt(0)
    ]);
    if lds_bytes != 0 {
        code.extend([0xD834_0000, 0x0000_0504]); // ds_write_b32 v4, v5
    }
    code.extend([
        0xE0C8_1000,
        0x800A_0504, // buffer_atomic_add v5, v4, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ]);
    NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1 | 2 << 11 | (lds_bytes / 512) << 15,
            threads: [x, y, z],
        }),
        code: code.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

fn lower_for_wave32(native: &NativeProgram) -> Lowered {
    let p = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, native).unwrap();
    let previous = lower::TEST_HOST_WAVE.replace(Some(32));
    let lowered = lower::lower(&lower::Parts {
        native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    });
    lower::TEST_HOST_WAVE.set(previous);
    lowered.unwrap()
}

fn wave_options(host: &Host) -> D3D12_FEATURE_DATA_D3D12_OPTIONS1 {
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
    // SAFETY: options has the layout and size required by OPTIONS1.
    unsafe {
        host.device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS1,
            (&raw mut options).cast(),
            size_of_val(&options) as u32,
        )
    }
    .unwrap();
    assert!(options.WaveOps.as_bool());
    options
}

fn run_counters(host: &Host, cs: &HostShader, count: u32, wave: u32, name: &str, shape: [u32; 3]) {
    let (root, pso) = host
        .compute_pipeline(cs)
        .unwrap_or_else(|e| panic!("{name} {shape:?}: {e}"));
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(
                    0,
                    Root::Constants(vec![0; cs.user_data_dwords().max(1) as usize]),
                )],
                views: &[View::RawUav(count.next_multiple_of(wave) + wave)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("{name} {shape:?}: {e}"));
    for (i, &word) in out[0].iter().enumerate() {
        let i = i as u32;
        let expected = if i < count { i % wave + 1 } else { 0 };
        assert_eq!(word, expected, "{name} {shape:?}, flat ID {i}");
    }
}

#[test]
fn wave_ticket_translation_respects_the_group_shape_and_lds_budget() {
    let mut validator =
        validator::Validator::load(validator::Validator::default_path().as_deref()).ok();
    for (shape, bytes, split, dwords) in [
        ([64, 1, 1], 0, false, 0),
        ([8, 4, 1], 0, false, 0),
        ([32, 2, 1], 0, false, 1),
        ([10, 10, 1], 0, false, 1),
        ([8, 3, 3], 512, false, 129),
        ([32, 2, 1], 32 * 1024, false, 8192),
        ([10, 10, 1], 0, true, 5),
    ] {
        let mut lowered = lower_for_wave32(&counter_program(shape, bytes, split));
        assert_eq!(lowered.wave_size, Some(32), "{shape:?}");
        assert_eq!(lowered.lds_dwords, dwords, "{shape:?}");
        if let Some(v) = &mut validator {
            v.validate(&mut lowered.dxil)
                .unwrap_or_else(|e| panic!("{shape:?}, {split}: {e}"));
        }
    }
    if validator.is_some() {
        eprintln!("DXIL validator accepted all wave ticket cases");
    }
}

#[test]
fn wave_ticket_counters_cover_every_id_on_wave32_hardware() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if std::env::var_os("VICTORY_WARP").is_some_and(|v| v == "1") {
        eprintln!("WARP was selected; hardware dispatches skipped");
        return;
    }
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; hardware dispatches skipped");
        return;
    };
    let options = wave_options(&host);
    if options.WaveLaneCountMax != 32 || options.WaveLaneCountMin != 32 {
        eprintln!("the device does not use fixed 32-lane waves; hardware dispatches skipped");
        return;
    }
    for (shape, wave64) in SHAPES.into_iter().map(|s| (s, false)).chain([
        ([40, 1, 1], true),
        ([10, 10, 1], true),
        ([8, 3, 3], true),
    ]) {
        let lowered = lower_for_wave32(&counter_program(shape, 0, wave64));
        let shader = HostShader {
            stage: HwStage::Cs,
            dxil: lowered.dxil,
            user_data: lowered.user_data,
            samplers: lowered.samplers,
            static_samplers: Vec::new(),
            root_cbvs: lowered.root_cbvs,
            root_srvs: lowered.root_srvs,
            root_uavs: lowered.root_uavs,
            root_indexed_views: Vec::new(),
        };
        run_counters(
            &host,
            &shader,
            shape.into_iter().product(),
            if wave64 { 64 } else { 32 },
            "hardware",
            shape,
        );
    }
    eprintln!("all 11 translated group shapes wrote each console ID once on 32-lane hardware");
}

#[test]
fn wave_ticket_prologue_runs_full_and_partial_waves_on_warp() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(true) else {
        eprintln!("no SM 6.6 WARP device; software dispatches skipped");
        return;
    };
    let options = wave_options(&host);
    let wave = options.WaveLaneCountMin;
    assert_eq!(
        wave, options.WaveLaneCountMax,
        "WARP should use a fixed wave size"
    );
    for shape in [[wave, 2, 1], [wave / 2 + 1, 3, 1], [wave / 2, 3, 2]] {
        let mut s = Shader::new(false);
        s.threads = Some(shape);
        s.lds_dwords = 1;
        let flat = lower::wave_ticket_prologue(&mut s, 0, wave);
        let e = s.entry;
        let zero = s.m.i32(0);
        let one = s.m.i32(1);
        let four = s.m.i32(4);
        let width = s.m.i32(wave);
        let byte = s.m.bin(e, Bin::Mul, flat, four);
        let lane = s.m.bin(e, Bin::URem, flat, width);
        let count = s.m.bin(e, Bin::Add, lane, one);
        let handle = s.heap_handle_at(e, zero, false, false, (11 | 0x1000, 0), true);
        let undef: V = s.m.undef(s.t.i32);
        s.op(
            e,
            "atomicBinOp.i32",
            Attr::NoUnwind,
            s.t.i32,
            78,
            &[handle, zero, byte, undef, undef, count],
        );
        s.m.terminate(e, Term::Ret);
        let cs = HostShader {
            stage: HwStage::Cs,
            dxil: s.finish(),
            user_data: Vec::new(),
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
            root_uavs: Vec::new(),
            root_indexed_views: Vec::new(),
        };
        run_counters(&host, &cs, shape.into_iter().product(), wave, "WARP", shape);
    }
    eprintln!("all 3 ticket dispatches wrote each ID once on {wave}-lane WARP");
}
