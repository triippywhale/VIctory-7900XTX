//! The flags of a title barrier's DXIL `barrier` (upstream #109), out of
//! `lower.rs`, which the repository's file-size policy caps.

/// The title's barrier retains its memory fences even when group sync
/// would wait for a host wave that never visits it. Explicit overrides
/// keep their old meanings; automatic mode is a targeted compatibility
/// workaround for narrow-wave hosts and is restricted to single-console-wave
/// groups (`group_threads <= console_wave`), so multi-console-wave groups
/// never lose cross-wave synchronization.
pub(super) fn barrier_mode(
    setting: u8,
    host_wave: u32,
    console_wave: u32,
    group_threads: Option<u32>,
    divergent: bool,
    writes_uavs: bool,
) -> u32 {
    let single_console_wave = group_threads.is_some_and(|t| t <= console_wave);
    let waits = match setting {
        0 => false,
        1 => !divergent,
        2 => true,
        _ => !(host_wave != 0 && host_wave < console_wave && single_console_wave && divergent),
    };
    8 | if writes_uavs { 2 } else { 0 } | u32::from(waits)
}
