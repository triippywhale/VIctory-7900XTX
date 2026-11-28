//! Console LDS accesses out of bounds read 0 and drop writes (RDNA2), with
//! `VICTORY_LDS_BOUNDS=1`, on the host GPU: tests by mathcx from
//! uiliannn/VItoriaBr 40beb26 (PalmVI 5cac9c03), translated here with the
//! switch on.

use super::host::{Dispatch, Host, Root, Stride, View};
use super::{HostShader, emit, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{Container, ShaderError, io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

fn ds(op: u16, offset: u16, addr: u8, data0: u8, data1: u8, dst: u8) -> [u32; 2] {
    [
        0xD800_0000 | (u32::from(op) << 18) | u32::from(offset),
        (u32::from(dst) << 24)
            | (u32::from(data1) << 16)
            | (u32::from(data0) << 8)
            | u32::from(addr),
    ]
}

fn v_mov_imm(dst: u8, lit: u32) -> [u32; 2] {
    [0x7E00_02FF | (u32::from(dst) << 17), lit]
}

fn buf_load_dword(dst: u8, vaddr: u8, srsrc_base: u8, offset: u16) -> [u32; 2] {
    [
        0xE030_1000 | u32::from(offset & 0xFFF),
        (0x80 << 24) | (u32::from(srsrc_base / 4) << 16) | (u32::from(dst) << 8) | u32::from(vaddr),
    ]
}

fn buf_store_dword(vdata: u8, vaddr: u8, srsrc_base: u8, offset: u16) -> [u32; 2] {
    [
        0xE070_1000 | u32::from(offset & 0xFFF),
        (0x80 << 24)
            | (u32::from(srsrc_base / 4) << 16)
            | (u32::from(vdata) << 8)
            | u32::from(vaddr),
    ]
}

/// A compute program with `lds_blocks` 512-byte granules of LDS, a raw
/// buffer SRV in s[8:11] and a raw UAV in s[12:15], translated with the
/// bounds on and validated.
fn compile_lds_cs(
    words: &[u32],
    wave64: bool,
    vgprs: u16,
    lds_blocks: u32,
    threads: [u32; 3],
) -> Result<HostShader, ShaderError> {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 4,
                dwords: 4,
                user_data: 0,
                slot: 0,
            },
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 8,
                lower_bound: 0,
                stride: 4,
                dwords: 4,
                user_data: 0,
                slot: 0,
            },
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: (lds_blocks << 15) | 2,
            threads,
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let bytes = crate::dxbc::build(&[]);
    let c = Container::parse(&bytes).unwrap();
    let p = ir::build(&native.patched_code(), native.wave64, native.vgprs)?;
    let res = resource::track(&p, &native.layout, native.stage, None)?;
    let io = io::raise(&p, &native)?;
    let st = structure::structure(&p)?;
    let was = lower::TEST_LDS_BOUNDS.replace(Some(true));
    let shader = emit(&c, &native, &p, &st, &io, &res);
    lower::TEST_LDS_BOUNDS.set(was);
    let shader = shader?;
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone()).unwrap();
    }
    Ok(shader)
}

/// Dynamic out-of-bounds LDS reads, writes, sub-dword accesses, and
/// integer/float atomics read 0 and drop writes without clobbering `lds[0]`.
#[test]
fn lds_out_of_bounds_reads_zero_and_drops_writes_on_gpu() {
    let mut w = vec![
        0xBE81_0380, // s_mov_b32 s1, 0
        0xF408_0200,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[0:1], 0x0   (SRV)
        0xF408_0300,
        0xFA00_0020, // s_load_dwordx4 s[12:15], s[0:1], 0x20 (UAV)
    ];
    // The sequence runs in lane 0 only, inside a group of 64 threads in
    // 32-lane waves (no wave size required, so WARP runs it too).
    let mut body = Vec::new();
    body.extend(v_mov_imm(1, 0));
    // Byte offsets loaded from the SRV into v2..v8, so no bounds check folds:
    // v2 = 0, v3 = 508 (the last dword of 512 bytes of LDS), v4 = 512 (the
    // first dword outside), v5 = 0xFFFF_FFF0 (far outside, wraps), v6 = 1
    // (byte 1 of lds[0]), v7 = 509 (byte 1 of lds[127]), v8 = 602 (a half
    // outside).
    for (idx, reg) in (2u8..=8).enumerate() {
        body.extend(buf_load_dword(reg, 1, 8, (idx * 4) as u16));
    }

    // lds[0] = 0x1122_3344 and lds[127] = 0xAABB_CCDD.
    body.extend(v_mov_imm(9, 0x1122_3344));
    body.extend(ds(0x0d, 0, 2, 9, 0, 0)); // ds_write_b32 v2, v9
    body.extend(v_mov_imm(9, 0xAABB_CCDD));
    body.extend(ds(0x0d, 0, 3, 9, 0, 0)); // ds_write_b32 v3, v9

    // 1. ds_write_b32 at 512: dropped, lds[0] untouched.
    body.extend(v_mov_imm(9, 0xDEAD_BEEF));
    body.extend(ds(0x0d, 0, 4, 9, 0, 0));

    // 2. ds_write2_b32 at base 508 (offset0 = 0: 508, inside; offset1 = 1:
    // 512, outside): lds[127] becomes 0xCAFE_BABE, the second dword is
    // dropped without touching lds[0].
    body.extend(v_mov_imm(9, 0xCAFE_BABE));
    body.extend(v_mov_imm(10, 0xBAD0_0000));
    body.extend(ds(0x0e, 0x0100, 3, 9, 10, 0));

    // 3. Sub-dword writes outside (`ds_write_b8` at 512 and
    // `ds_write_b16_d16_hi` at 0xFFFF_FFF0): dropped.
    body.extend(v_mov_imm(9, 0x0000_00FF));
    body.extend(ds(0x1e, 0, 4, 9, 0, 0));
    body.extend(v_mov_imm(9, 0x9999_0000));
    body.extend(ds(0xa1, 0, 5, 9, 0, 0));

    // 4. Integer and float `_rtn` atomics outside return 0 into registers
    // that held something else, and write nothing.
    body.extend(v_mov_imm(11, 0x7777_7777));
    body.extend(v_mov_imm(9, 0x100));
    body.extend(ds(0x20, 4, 4, 9, 0, 11)); // ds_add_rtn_u32 v11, v4, v9 offset:4 (516)
    body.extend(v_mov_imm(12, 0x8888_8888));
    body.extend(v_mov_imm(9, 99.0f32.to_bits()));
    body.extend(ds(0x33, 8, 4, 9, 0, 12)); // ds_max_rtn_f32 v12, v4, v9 offset:8 (520)

    // 5. ds_read2_b32 at 508 (inside) and 512 (outside) into v[13:14].
    body.extend(ds(0x37, 0x0100, 3, 0, 0, 13));

    // 6. ds_read_b32 at 512: v15 = 0; at 0: v16 = 0x1122_3344.
    body.extend(ds(0x36, 0, 4, 0, 0, 15));
    body.extend(ds(0x36, 0, 2, 0, 0, 16));

    // 7. Sub-dword reads: ds_read_u8 at byte 1: 0x33 (v17); ds_read_u8
    // outside: 0 (v18); ds_read_i8 at byte 509 (0xBA = -70): 0xFFFF_FFBA
    // (v19); ds_read_u16_d16_hi outside into v20 = 0x5555_6666: 0x0000_6666
    // (the high half zero, the low half kept).
    body.extend(ds(0x3a, 0, 6, 0, 0, 17));
    body.extend(ds(0x3a, 0, 4, 0, 0, 18));
    body.extend(ds(0x39, 0, 7, 0, 0, 19));
    body.extend(v_mov_imm(20, 0x5555_6666));
    body.extend(ds(0xa7, 0, 8, 0, 0, 20));

    for (slot, reg) in (11u8..=20).enumerate() {
        body.extend(buf_store_dword(reg, 1, 12, (slot * 4) as u16));
    }

    w.extend([
        0x7D84_0080,                       // v_cmp_eq_u32 vcc_lo, 0, v0
        0xBE82_3C6A,                       // s_and_saveexec_b32 s2, vcc_lo
        0xBF88_0000 | (body.len() as u32), // s_cbranch_execz -> restore
    ]);
    w.extend(body);
    w.push(0x887E_027E); // s_or_b32 exec_lo, exec_lo, s2
    w.push(S_ENDPGM);

    let cs = compile_lds_cs(&w, false, 24, 1, [64, 1, 1]).unwrap();
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let addrs = [0u32, 508, 512, 0xFFFF_FFF0, 1, 509, 602];
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            continue;
        };
        let (root, pso) = host.compute_pipeline(&cs).unwrap();
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(vec![0]))],
                    views: &[
                        View::Buffer {
                            data: &addrs,
                            stride: Stride::Raw,
                        },
                        View::RawUav(64),
                    ],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap();
        assert_eq!(
            &out[0][..10],
            &[
                0,             // v11: ds_add_rtn_u32 outside returns 0
                0,             // v12: ds_max_rtn_f32 outside returns 0
                0xCAFE_BABE,   // v13: ds_read2_b32's first dword, inside (lds[127])
                0,             // v14: its second dword, outside (512)
                0,             // v15: ds_read_b32 outside (512)
                0x1122_3344,   // v16: lds[0], untouched by the writes and atomics outside
                0x33,          // v17: ds_read_u8 at byte 1
                0,             // v18: ds_read_u8 outside, at byte 512
                -70i32 as u32, // v19: ds_read_i8 at byte 509 (0xBA), sign-extended
                0x0000_6666,   // v20: ds_read_u16_d16_hi outside: high half 0, low half kept
            ],
            "backend: {label}"
        );
    }
}

/// Under a divergent `EXEC` mask, a lane that does not run keeps its
/// destination register, a running lane outside the LDS gets 0 without
/// touching `lds[0]`, and a running lane inside updates `lds[0]` and gets
/// the old value.
#[test]
fn lds_atomic_under_exec_mask_preserves_inactive_lane_and_zeros_oob_active_lane() {
    let mut w = vec![
        0xBE81_0380, // s_mov_b32 s1, 0
        0xF408_0200,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[0:1], 0x0   (SRV)
        0xF408_0300,
        0xFA00_0020, // s_load_dwordx4 s[12:15], s[0:1], 0x20 (UAV)
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0 (the lane's byte offset: v0 * 4)
    ];
    // lds[0] = 0x50, then every lane meets at a barrier.
    w.extend(v_mov_imm(3, 0));
    w.extend(v_mov_imm(4, 0x50));
    w.extend(ds(0x0d, 0, 3, 4, 0, 0)); // ds_write_b32 v3, v4
    w.push(0xBF8A_0000); // s_barrier

    // Each lane's LDS byte address from SRV[v0] into v5, and
    // v2 = 0x1234_0000 + v0 before lane 0 is masked off.
    w.extend(buf_load_dword(5, 1, 8, 0));
    w.extend(v_mov_imm(2, 0x1234_0000));
    w.push(0x6804_0002); // v_add_nc_u32 v2, v2, v0
    w.extend(v_mov_imm(6, 0x10));

    // Lane 0 is masked off (`vcc_lo = (0 == v0)`; `exec_lo = ~vcc_lo &
    // exec_lo`); `ds_add_rtn_u32 v2, v5, v6` runs on lanes 1..63 (lane 2
    // inside, at 0; every other one outside, at 512).
    w.extend([
        0x7D84_0080, // v_cmp_eq_u32 vcc_lo, 0, v0
        0xBE82_446A, // s_andn1_saveexec_b32 s2, vcc_lo
        0xBF88_0002, // s_cbranch_execz 2
    ]);
    w.extend(ds(0x20, 0, 5, 6, 0, 2)); // ds_add_rtn_u32 v2, v5, v6
    w.push(0x887E_027E); // s_or_b32 exec_lo, exec_lo, s2
    w.push(0xBF8A_0000); // s_barrier

    // Each lane's `v2` to UAV[v0], and `lds[0]` to UAV[64].
    w.extend(buf_store_dword(2, 1, 12, 0));
    w.extend(ds(0x36, 0, 3, 0, 0, 7)); // ds_read_b32 v7, v3
    w.extend(v_mov_imm(8, 64 * 4));
    w.extend(buf_store_dword(7, 8, 12, 0));
    w.push(S_ENDPGM);

    let cs = compile_lds_cs(&w, false, 10, 1, [64, 1, 1]).unwrap();
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut addrs = [512u32; 64];
    addrs[0] = 0; // lane 0 is masked off
    addrs[2] = 0; // lane 2 runs, inside
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            continue;
        };
        let (root, pso) = host.compute_pipeline(&cs).unwrap();
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(vec![0]))],
                    views: &[
                        View::Buffer {
                            data: &addrs,
                            stride: Stride::Raw,
                        },
                        View::RawUav(260),
                    ],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap();
        assert_eq!(
            out[0][0], 0x1234_0000,
            "{label}: the lane that does not run keeps v2"
        );
        assert_eq!(out[0][1], 0, "{label}: lane 1, running and outside, gets 0");
        assert_eq!(
            out[0][2], 0x50,
            "{label}: lane 2, running and inside, gets the old lds[0]"
        );
        for lane in 3..64 {
            assert_eq!(
                out[0][lane], 0,
                "{label}: lane {lane}, running and outside, gets 0"
            );
        }
        assert_eq!(
            out[0][64], 0x60,
            "{label}: lds[0] added to once, the 62 atomics outside dropped"
        );
    }
}
