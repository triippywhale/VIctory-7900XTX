//! A compute program as device method 0x218 passes one: settings, code and
//! binding table laid out as logged (the module docs), the code a short
//! stand-in written here.

use super::*;

fn hex(s: &str) -> Vec<u8> {
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
        .collect()
}

/// An 8x8 group with COMPUTE_PGM_RSRC2 0xb8a.
const SETTINGS: &str = "00cf05400201096c00000000080008000100000043032c408a0b0000";

/// `s_mov_b32 s8, s4; s_mov_b32 s9, 0; s_load_dwordx8 s[16:23], s[8:9];
/// s_endpgm`: the table's address in user-data dword 4.
const CODE: &str = "0403881b800389be040308f4000000fa000081bf";

/// A compute table: a texture at dword 0 of the table in user-data dword 4,
/// then a hash entry and the end.
fn table() -> Vec<u8> {
    let mut t = Vec::new();
    for v in [KIND_COMPUTE, 0x44, 0x60, 0, 0, 0x0001_0440] {
        t.extend_from_slice(&v.to_le_bytes());
    }
    for v in [TAG_USED, 0x300, 0, 0x10a, 4, 0x408, u32::MAX, 0] {
        t.extend_from_slice(&v.to_le_bytes());
    }
    t.extend_from_slice(&TAG_HASH.to_le_bytes());
    t.extend_from_slice(&[0; 28]);
    t
}

fn program() -> crate::NativeProgram {
    let c = compute_container(&hex(SETTINGS), &hex(CODE), &table()).unwrap();
    let c = crate::Container::parse(&c).unwrap();
    let mut programs = crate::native_programs(c.part(FourCC::DXIL).unwrap().data).unwrap();
    assert_eq!(programs.len(), 1);
    programs.remove(0)
}

/// The settings and table read as the code uses them.
#[test]
fn a_compute_program_reads_as_its_code_uses_it() {
    use crate::native::{ResourceType, RootType};
    let p = program();
    assert_eq!(p.stage, crate::HwStage::Cs);
    assert_eq!(p.code, hex(CODE));
    assert_eq!(p.layout.len(), 1);
    let table = p.layout[0];
    assert_eq!((table.root, table.user_data), (RootType::Table, 4));
    assert!(
        p.used
            .iter()
            .any(|u| u.kind == ResourceType::Srv && u.root == RootType::Table)
    );
}

/// A table whose stated length runs past its part is refused.
#[test]
fn a_table_longer_than_its_part_is_refused() {
    let mut t = table();
    t[4..8].copy_from_slice(&0x48u32.to_le_bytes());
    assert!(compute_container(&hex(SETTINGS), &hex(CODE), &t).is_err());
}
