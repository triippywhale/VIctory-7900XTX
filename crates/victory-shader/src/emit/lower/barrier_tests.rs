//! Barrier flags of the automatic group sync (upstream #109): kept out of
//! `lower.rs`, which the repository's file-size policy caps.

use super::{Parts, lower};
use crate::native::{ComputeIo, HwStage, NativeProgram, StageIo};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

#[test]
fn automatic_barriers_keep_fences_and_only_drop_unsafe_sync() {
    for writes in [false, true] {
        let fence = 8 | if writes { 2 } else { 0 };
        for (host, console, threads, divergent, waits) in [
            (32, 64, Some(64), true, false),
            (32, 64, Some(64), false, true),
            (32, 64, Some(128), true, true),
            (32, 64, None, true, true),
            (64, 64, Some(64), true, true),
            (32, 32, Some(32), true, true),
            (0, 64, Some(64), true, true),
        ] {
            assert_eq!(
                super::barrier_mode(3, host, console, threads, divergent, writes),
                fence | u32::from(waits),
            );
        }
        assert_eq!(super::barrier_mode(0, 32, 64, Some(64), false, writes), fence);
        assert_eq!(super::barrier_mode(1, 64, 64, Some(64), true, writes), fence);
        assert_eq!(super::barrier_mode(1, 32, 64, Some(64), false, writes), fence | 1);
        assert_eq!(super::barrier_mode(2, 32, 64, Some(64), true, writes), fence | 1);
    }
}

#[test]
fn automatic_barrier_flags_reach_the_emitted_shader() {
    let divergent = [
        0x7D88_0090, // v_cmp_gt_u32 vcc, 16, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0001, // s_cbranch_execz past the barrier
        0xBF8A_0000, // s_barrier
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
    ];
    let mut divergent32 = divergent;
    divergent32[1] = 0xBE8C_3C6A; // s_and_saveexec_b32 s12, vcc_lo
    divergent32[4] = 0xBEFE_030C; // s_mov_b32 exec_lo, s12
    for (body, wave64, threads, host, expected) in [
        (&divergent[..], true, 64, 32, 8),
        (&divergent[..], true, 128, 32, 9),
        (&divergent[..], true, 64, 64, 9),
        (&divergent32[..], false, 64, 32, 9),
        (&[0xBF8A_0000][..], true, 64, 32, 9),
    ] {
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64,
            vgprs: 16,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo { rsrc2: 0, threads: [threads, 1, 1] }),
            code: body.iter().copied().chain([S_ENDPGM])
                .flat_map(u32::to_le_bytes).collect(),
        };
        let p = ir::build(&native.code, wave64, native.vgprs).expect("IR");
        let st = structure::structure(&p).expect("structure");
        let res = resource::track(&p, &[], native.stage, Some(&st)).expect("resources");
        let io = io::raise(&p, &native).expect("IO");
        let old_wave = super::TEST_HOST_WAVE.replace(Some(host));
        let old_mode = super::super::TEST_GROUP_SYNC.replace(Some(3));
        let shader = lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
        });
        super::TEST_HOST_WAVE.set(old_wave);
        super::super::TEST_GROUP_SYNC.set(old_mode);
        let shader = shader.unwrap_or_else(|e| panic!("wave64={wave64}, host={host}: {e:?}"));
        assert_eq!(shader.barrier_modes, vec![expected]);
    }
}
