//! Small hooks in the title's code: a `call rel32` redirected to a thunk
//! (call-site redirect), or a function entry turned into a jump to one
//! (entry hook). Same mechanism as the static TLS redirect (`crate::tls`):
//! generated code on a page within `rel32` reach of the image, the title's
//! bytes rewritten while no title code runs, after checking them.
//!
//! The thunk saves every volatile register, the flags and xmm0-5, calls a
//! routine of ours with a pointer to the saved general registers
//! ([`Regs`]), restores everything, runs the bytes it replaced (an entry
//! hook's prologue; nothing for a call site) and jumps on to the original
//! code: the title sees the same registers, flags and stack as without it.
//! The routine runs on a 16-byte aligned stack with its shadow space.
//!
//! A routine may replace [`Regs::ret`] (the return address of the call
//! being made) with an [`after_stub`]: the function then returns there,
//! the stub calls a second routine with the function's `rax` and a slot to
//! put the real return address in, and returns there with `rax` and xmm0
//! as the function left them.

use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Result, bail};
use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, VirtualProtect};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

/// The general registers as the thunk saved them, lowest address first;
/// `ret` is the return address on the stack (writable: see the module).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Regs {
    pub r11: u64,
    pub r10: u64,
    pub r9: u64,
    pub r8: u64,
    pub rdx: u64,
    pub rcx: u64,
    pub rax: u64,
    pub flags: u64,
    pub ret: u64,
}

/// Called before the hooked code, with the saved registers.
pub type Before = unsafe extern "C" fn(*mut Regs);
/// Called when a function whose return an [`after_stub`] took returns:
/// its `rax`, and the slot to write the real return address into.
pub type After = unsafe extern "C" fn(u64, *mut u64);

/// Whether `code` (5 bytes at address `site`) is `call target`.
pub fn is_call_to(code: &[u8], site: usize, target: usize) -> bool {
    code.len() >= 5
        && code[0] == 0xE8
        && (site as i64 + 5 + i64::from(i32::from_le_bytes([code[1], code[2], code[3], code[4]]))) as usize == target
}

/// The thunk: saves, calls `before`, restores, runs `replaced`, jumps to
/// `resume`. Position independent.
fn before_thunk(before: usize, replaced: &[u8], resume: usize) -> Vec<u8> {
    let mut c = vec![0x9C, 0x50, 0x51, 0x52, 0x41, 0x50, 0x41, 0x51, 0x41, 0x52, 0x41, 0x53];
    c.extend([0x48, 0x81, 0xEC, 0x88, 0, 0, 0]); // sub rsp, 0x88
    for x in 0..6u8 {
        c.extend([0xF3, 0x0F, 0x7F, 0x44 | x << 3, 0x24, 0x20 + 0x10 * x]); // movdqu [rsp+..], xmmN
    }
    c.extend([0x48, 0x8D, 0x8C, 0x24, 0x88, 0, 0, 0]); // lea rcx, [rsp+0x88]: the saved registers
    c.extend([0x48, 0xB8]);
    c.extend((before as u64).to_le_bytes()); // mov rax, before
    c.extend([0xFF, 0xD0]); // call rax
    for x in 0..6u8 {
        c.extend([0xF3, 0x0F, 0x6F, 0x44 | x << 3, 0x24, 0x20 + 0x10 * x]); // movdqu xmmN, [rsp+..]
    }
    c.extend([0x48, 0x81, 0xC4, 0x88, 0, 0, 0]); // add rsp, 0x88
    c.extend([0x41, 0x5B, 0x41, 0x5A, 0x41, 0x59, 0x41, 0x58, 0x5A, 0x59, 0x58, 0x9D]);
    c.extend(replaced);
    c.extend([0xFF, 0x25, 0, 0, 0, 0]); // jmp [rip]
    c.extend((resume as u64).to_le_bytes());
    c
}

/// The stub a function returns to when a [`Before`] routine took its
/// return ([`after_stub`]).
fn after_code(after: usize) -> Vec<u8> {
    let mut c = vec![0x48, 0x83, 0xEC, 0x08, 0x50, 0x48, 0x83, 0xEC, 0x30]; // sub rsp,8; push rax; sub rsp,0x30
    c.extend([0xF3, 0x0F, 0x7F, 0x44, 0x24, 0x20]); // movdqu [rsp+0x20], xmm0
    c.extend([0x48, 0x89, 0xC1, 0x48, 0x8D, 0x54, 0x24, 0x38]); // mov rcx, rax; lea rdx, [rsp+0x38]
    c.extend([0x48, 0xB8]);
    c.extend((after as u64).to_le_bytes());
    c.extend([0xFF, 0xD0]); // call rax
    c.extend([0xF3, 0x0F, 0x6F, 0x44, 0x24, 0x20]); // movdqu xmm0, [rsp+0x20]
    c.extend([0x48, 0x83, 0xC4, 0x30, 0x58, 0xC3]); // add rsp,0x30; pop rax; ret
    c
}

/// Our code on a fresh page near `near_to`, made executable.
fn place(near_to: usize, code: &[u8]) -> Result<usize> {
    if code.len() > 4096 {
        bail!("hook thunk too large");
    }
    let page = crate::tls::near(near_to, 0)?;
    // SAFETY: a fresh writable page of ours, 4096 bytes.
    unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), page, code.len()) };
    let mut old = 0;
    // SAFETY: our page.
    if unsafe { VirtualProtect(page as _, 4096, PAGE_EXECUTE_READ, &mut old) } == 0 {
        bail!("VirtualProtect(hook thunk) failed: {}", std::io::Error::last_os_error());
    }
    Ok(page as usize)
}

/// Writes `bytes` over the code at `at`.
///
/// # Safety
/// `at..at + bytes.len()` is mapped code no thread is running.
unsafe fn write_code(at: usize, bytes: &[u8]) -> Result<()> {
    let mut old = 0;
    // SAFETY: the caller's range.
    if unsafe { VirtualProtect(at as _, bytes.len(), PAGE_EXECUTE_READWRITE, &mut old) } == 0 {
        bail!("VirtualProtect(code) failed: {}", std::io::Error::last_os_error());
    }
    // SAFETY: writable now; see the caller's promise.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), at as *mut u8, bytes.len());
        VirtualProtect(at as _, bytes.len(), old, &mut old);
        FlushInstructionCache(GetCurrentProcess(), at as _, bytes.len());
    }
    Ok(())
}

fn rel32(from_end: usize, to: usize) -> Result<[u8; 4]> {
    let rel = to as i64 - from_end as i64;
    match i32::try_from(rel) {
        Ok(r) => Ok(r.to_le_bytes()),
        Err(_) => bail!("hook thunk out of rel32 reach"),
    }
}

/// Redirects the `call target` at `site` to a thunk that runs `before`,
/// then `target`. Refuses (and changes nothing) unless the 5 bytes there
/// are that call.
///
/// # Safety
/// `site` is mapped title code that no thread runs yet.
pub unsafe fn redirect_call(site: usize, target: usize, before: Before) -> Result<()> {
    // SAFETY: the caller's mapped code.
    let found = unsafe { std::slice::from_raw_parts(site as *const u8, 5) };
    if !is_call_to(found, site, target) {
        bail!("bytes at {site:#x} are {found:02x?}, not a call to {target:#x}");
    }
    let thunk = place(site, &before_thunk(before as usize, &[], target))?;
    let mut call = vec![0xE8];
    call.extend(rel32(site + 5, thunk)?);
    // SAFETY: the caller's promise.
    unsafe { write_code(site, &call) }
}

/// Turns the function entry at `entry`, whose first bytes must be
/// `prologue` (whole instructions, position independent, at least 5
/// bytes), into a jump to a thunk that runs `before`, then the prologue,
/// then the function from `entry + prologue.len()`. Refuses (and changes
/// nothing) if the bytes differ.
///
/// # Safety
/// `entry` is mapped title code that no thread runs yet, and no code jumps
/// into the middle of the prologue.
pub unsafe fn hook_entry(entry: usize, prologue: &[u8], before: Before) -> Result<()> {
    if prologue.len() < 5 {
        bail!("a prologue of {} bytes has no room for a jump", prologue.len());
    }
    // SAFETY: the caller's mapped code.
    let found = unsafe { std::slice::from_raw_parts(entry as *const u8, prologue.len()) };
    if found != prologue {
        bail!("bytes at {entry:#x} are {found:02x?}, expected {prologue:02x?}");
    }
    let thunk = place(entry, &before_thunk(before as usize, prologue, entry + prologue.len()))?;
    let mut jump = vec![0xE9];
    jump.extend(rel32(entry + 5, thunk)?);
    jump.resize(prologue.len(), 0x90);
    // SAFETY: the caller's promise.
    unsafe { write_code(entry, &jump) }
}

/// Writes `new` over the code at `at` if the bytes there are `expected`
/// (refuses, changing nothing, otherwise; the error names what is there).
///
/// # Safety
/// `at` is mapped title code that no thread runs yet.
pub unsafe fn patch_bytes(at: usize, expected: &[u8], new: &[u8]) -> Result<()> {
    // SAFETY: the caller's mapped code.
    let found = unsafe { std::slice::from_raw_parts(at as *const u8, expected.len()) };
    if found != expected {
        bail!("bytes at {at:#x} are {found:02x?}, expected {expected:02x?}");
    }
    // SAFETY: the caller's promise.
    unsafe { write_code(at, new) }
}

/// The address of a stub that calls `after` (see the module); one per
/// routine, made once.
pub fn after_stub(after: After) -> Result<usize> {
    static MADE: std::sync::Mutex<Vec<(usize, usize)>> = std::sync::Mutex::new(Vec::new());
    static NEAR: AtomicUsize = AtomicUsize::new(0);
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&(_, stub)) = made.iter().find(|m| m.0 == after as usize) {
        return Ok(stub);
    }
    // Anywhere does: the stub is reached by a return, not a rel32.
    let near_to = match NEAR.load(Ordering::Relaxed) {
        0 => after as usize,
        n => n,
    };
    let stub = place(near_to, &after_code(after as usize))?;
    NEAR.store(stub, Ordering::Relaxed);
    made.push((after as usize, stub));
    Ok(stub)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static SEEN: Mutex<Vec<Regs>> = Mutex::new(Vec::new());
    // The two hook tests share SEEN: run one at a time (tests run in parallel; a busy machine made one clear the other's calls).
    static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());
    static RETURNED: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());
    static STUB: AtomicUsize = AtomicUsize::new(0);

    extern "C" fn target(a: u64, b: u64, c: u64, d: u64) -> u64 {
        a * 1000 + b * 100 + c * 10 + d
    }

    unsafe extern "C" fn record(regs: *mut Regs) {
        // SAFETY: the thunk's saved registers.
        let r = unsafe { *regs };
        SEEN.lock().unwrap().push(r);
    }

    thread_local! {
        static RETURNS: std::cell::RefCell<Vec<u64>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    unsafe extern "C" fn take_return(regs: *mut Regs) {
        // SAFETY: the thunk's saved registers.
        let r = unsafe { &mut *regs };
        SEEN.lock().unwrap().push(*r);
        RETURNS.with(|s| s.borrow_mut().push(r.ret));
        r.ret = STUB.load(Ordering::Relaxed) as u64;
    }

    unsafe extern "C" fn returned(rax: u64, slot: *mut u64) {
        let ret = RETURNS.with(|s| s.borrow_mut().pop()).expect("a taken return");
        RETURNED.lock().unwrap().push((rax, ret));
        // SAFETY: the stub's slot.
        unsafe { *slot = ret };
    }

    /// Writable code of ours near `to`: `bytes` at its start.
    fn code_page(to: usize, bytes: &[u8]) -> usize {
        let page = crate::tls::near(to, 0).unwrap();
        // SAFETY: our fresh page.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), page, bytes.len()) };
        let mut old = 0;
        // SAFETY: our page.
        unsafe { VirtualProtect(page as _, 4096, PAGE_EXECUTE_READ, &mut old) };
        page as usize
    }

    #[test]
    fn a_call_is_recognised_by_its_bytes() {
        let site = 0x1_4000_1000usize;
        let target = 0x1_4000_1040usize;
        let mut code = vec![0xE8];
        code.extend(((target - site - 5) as i32).to_le_bytes());
        assert!(is_call_to(&code, site, target));
        assert!(!is_call_to(&code, site, target + 1));
        code[0] = 0xE9;
        assert!(!is_call_to(&code, site, target));
        assert!(!is_call_to(&code[..4], site, target));
    }

    #[test]
    fn a_redirected_call_runs_the_routine_then_the_target_with_the_same_registers() {
        let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let t = target as *const () as usize;
        // sub rsp,0x28; call target; add rsp,0x28; ret
        let mut code = vec![0x48, 0x83, 0xEC, 0x28, 0xE8, 0, 0, 0, 0, 0x48, 0x83, 0xC4, 0x28, 0xC3];
        let probe = crate::tls::near(t, 0).unwrap() as usize;
        code[5..9].copy_from_slice(&rel32(probe + 9, t).unwrap());
        // SAFETY: our fresh page.
        unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), probe as *mut u8, code.len()) };
        let mut old = 0;
        // SAFETY: our page.
        unsafe { VirtualProtect(probe as _, 4096, PAGE_EXECUTE_READ, &mut old) };
        let site = probe + 4;
        let f: extern "C" fn(u64, u64, u64, u64) -> u64 = unsafe { std::mem::transmute(probe) };
        // Wrong target: refused, nothing changed.
        // SAFETY: our page, not running.
        assert!(unsafe { redirect_call(site, t + 1, record) }.is_err());
        // SAFETY: our page, not running.
        unsafe { redirect_call(site, t, record) }.unwrap();
        SEEN.lock().unwrap().clear();
        assert_eq!(f(1, 2, 3, 4), 1234);
        assert_eq!(f(5, 6, 7, 8), 5678);
        let seen = SEEN.lock().unwrap().clone();
        assert_eq!(seen.len(), 2);
        assert_eq!((seen[0].rcx, seen[0].rdx, seen[0].r8, seen[0].r9), (1, 2, 3, 4));
        assert_eq!(seen[0].ret as usize, site + 5);
        assert_eq!((seen[1].rcx, seen[1].r9), (5, 8));
    }

    #[test]
    fn an_entry_hook_sees_the_arguments_and_the_return() {
        let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        // push rsi; push rdi; push rbx; sub rsp,0x20 | lea rax,[rcx+rdx] |
        // add rsp,0x20; pop rbx; pop rdi; pop rsi; ret
        let prologue = [0x56, 0x57, 0x53, 0x48, 0x83, 0xEC, 0x20];
        let mut code = prologue.to_vec();
        code.extend([0x48, 0x8D, 0x04, 0x11, 0x48, 0x83, 0xC4, 0x20, 0x5B, 0x5F, 0x5E, 0xC3]);
        let entry = code_page(record as *const () as usize, &code);
        STUB.store(after_stub(returned).unwrap(), Ordering::Relaxed);
        // SAFETY: our page, not running.
        assert!(unsafe { hook_entry(entry, &[0x55, 0x57, 0x53, 0x48, 0x83, 0xEC, 0x20], take_return) }.is_err());
        // SAFETY: as above: refused, so the bytes are as they were.
        assert_eq!(unsafe { std::slice::from_raw_parts(entry as *const u8, 7) }, &prologue);
        // SAFETY: our page, not running.
        unsafe { hook_entry(entry, &prologue, take_return) }.unwrap();
        let f: extern "C" fn(u64, u64) -> u64 = unsafe { std::mem::transmute(entry) };
        SEEN.lock().unwrap().clear();
        RETURNED.lock().unwrap().clear();
        assert_eq!(f(40, 2), 42);
        assert_eq!(f(7, 8), 15);
        let seen = SEEN.lock().unwrap().clone();
        assert_eq!((seen[0].rcx, seen[0].rdx), (40, 2));
        let back = RETURNED.lock().unwrap().clone();
        assert_eq!(back.iter().map(|b| b.0).collect::<Vec<_>>(), [42, 15]);
        assert_eq!(back[0].1, seen[0].ret);
    }
}
