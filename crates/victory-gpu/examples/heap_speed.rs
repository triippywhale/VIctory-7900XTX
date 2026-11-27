//! How fast the card reads memory shared with the PC, by how the memory came
//! to the card. Renderer port 1 hands the title's own pages to the driver
//! (`OpenExistingHeapFromAddress1` on a view of a section, `inplace.rs`).
//! xcl-dev/xcl measured on NVIDIA (RTX 5070 Ti, f2248f06) that GPU loads from
//! such imported pages ran 8-24x slower than from a heap the driver made
//! itself (`CUSTOM`, write-back, pool L0), at the same CPU caching. This
//! times the same on this PC's card, without the title:
//!
//! - `imported view`: a view of a pagefile section, opened as port 1 opens
//!   the title's 32-bit graphics heap;
//! - `imported private`: committed private memory (`OpenExistingHeapFromAddress`);
//! - `driver custom L0`: a committed buffer in a `CUSTOM` write-back L0 heap;
//! - `card memory`: a `DEFAULT` heap buffer, for reference.
//!
//! Two passes each, timed on the card with timestamps, best of five:
//! dependent loads (each lane follows a random chain through the buffer, one
//! load waiting for the last: latency), and a streaming read of the whole
//! buffer (bandwidth, as constants are read).
//!
//! `cargo run --release -p victory-gpu --example heap_speed [MB]` (64 by
//! default). `VICTORY_D3D12_ADAPTER` picks the card as for a run.

#[cfg(not(windows))]
fn main() {
    eprintln!("heap_speed runs on Windows only");
}

#[cfg(windows)]
fn main() {
    let mb: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(64)
        .clamp(16, 1024);
    if let Err(e) = bench::run(mb << 20) {
        eprintln!("heap_speed: {e}");
        std::process::exit(1);
    }
}

#[cfg(windows)]
mod bench {
    use std::ffi::c_void;

    use windows::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
    use windows::core::{Interface, PCSTR, s};
    use windows_sys::Win32::System::Memory as mem;

    /// Bytes between the nodes of a chain: one per cache line and more.
    const NODE: usize = 256;
    /// Lanes that follow chains at once, and the loads each makes.
    const CHASERS: u32 = 4096;
    const STEPS: u32 = 4096;
    /// Lanes of the streaming read.
    const STREAMERS: u32 = 64 * 1024;

    const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), UAV(u0), UAV(u1)"
cbuffer K : register(b0) { uint nodes; uint steps; uint bytes; uint lanes; };
RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void chase(uint t : SV_DispatchThreadID) {
    uint p = ((t * 7919u) % nodes) * 256u;
    [loop] for (uint i = 0; i < steps; i++) p = src.Load(p);
    dst.Store(t * 4, p);
}
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void stream(uint t : SV_DispatchThreadID) {
    uint4 sum = 0;
    [loop] for (uint at = t * 16u; at < bytes; at += lanes * 16u) sum += src.Load4(at);
    dst.Store(t * 4, sum.x ^ sum.y ^ sum.z ^ sum.w);
}
"#;

    type R<T> = Result<T, String>;

    fn e(what: &str) -> impl FnOnce(windows::core::Error) -> String + '_ {
        move |err| format!("{what}: {err}")
    }

    struct Gpu {
        device: ID3D12Device,
        queue: ID3D12CommandQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        done: u64,
        event: HANDLE,
        root: ID3D12RootSignature,
        chase: ID3D12PipelineState,
        stream: ID3D12PipelineState,
        stamps: ID3D12QueryHeap,
        stamps_back: ID3D12Resource,
        out: ID3D12Resource,
        ticks: u64,
    }

    fn buffer_desc(size: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
        D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: flags,
            ..Default::default()
        }
    }

    fn compile(entry: PCSTR) -> R<Vec<u8>> {
        let mut code: Option<ID3DBlob> = None;
        let mut errors: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call.
        let done = unsafe {
            D3DCompile(
                SOURCE.as_ptr().cast(),
                SOURCE.len(),
                s!("heap_speed"),
                None,
                None,
                entry,
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                Some(&mut errors),
            )
        };
        if let Err(err) = done {
            let text = errors
                // SAFETY: the blob's own bytes.
                .map(|b| unsafe {
                    String::from_utf8_lossy(std::slice::from_raw_parts(
                        b.GetBufferPointer().cast::<u8>(),
                        b.GetBufferSize(),
                    ))
                    .into_owned()
                })
                .unwrap_or_default();
            return Err(format!("compile: {err} {text}"));
        }
        let code = code.ok_or("compile: no code")?;
        // SAFETY: the blob's own bytes.
        Ok(unsafe {
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
                .to_vec()
        })
    }

    impl Gpu {
        fn new() -> R<Gpu> {
            let device =
                victory_shader::emit::host::create_preferred_device().map_err(e("device"))?;
            // SAFETY: plain creation calls on a live device.
            unsafe {
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .map_err(e("queue"))?;
                let allocator: ID3D12CommandAllocator = device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .map_err(e("allocator"))?;
                let list: ID3D12GraphicsCommandList = device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .map_err(e("list"))?;
                list.Close().map_err(e("close"))?;
                let fence: ID3D12Fence = device
                    .CreateFence(0, D3D12_FENCE_FLAG_NONE)
                    .map_err(e("fence"))?;
                let event = CreateEventW(None, false, false, None).map_err(e("event"))?;
                let chase_code = compile(s!("chase"))?;
                let stream_code = compile(s!("stream"))?;
                let root: ID3D12RootSignature = device
                    .CreateRootSignature(0, &chase_code)
                    .map_err(e("root"))?;
                let pso = |code: &[u8]| -> R<ID3D12PipelineState> {
                    device
                        .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                            pRootSignature: std::mem::transmute_copy(&root),
                            CS: D3D12_SHADER_BYTECODE {
                                pShaderBytecode: code.as_ptr().cast(),
                                BytecodeLength: code.len(),
                            },
                            ..Default::default()
                        })
                        .map_err(e("pipeline"))
                };
                let (chase, stream) = (pso(&chase_code)?, pso(&stream_code)?);
                let mut stamps: Option<ID3D12QueryHeap> = None;
                device
                    .CreateQueryHeap(
                        &D3D12_QUERY_HEAP_DESC {
                            Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                            Count: 2,
                            NodeMask: 0,
                        },
                        &mut stamps,
                    )
                    .map_err(e("query heap"))?;
                let ticks = queue
                    .GetTimestampFrequency()
                    .map_err(e("timestamp frequency"))?;
                let mut gpu = Gpu {
                    stamps_back: committed(
                        &device,
                        D3D12_HEAP_TYPE_READBACK,
                        16,
                        D3D12_RESOURCE_FLAG_NONE,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                    )?,
                    out: committed(
                        &device,
                        D3D12_HEAP_TYPE_DEFAULT,
                        u64::from(STREAMERS.max(CHASERS)) * 4,
                        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COMMON,
                    )?,
                    device,
                    queue,
                    allocator,
                    list,
                    fence,
                    done: 0,
                    event,
                    root,
                    chase,
                    stream,
                    stamps: stamps.ok_or("query heap")?,
                    ticks,
                };
                gpu.done = 0;
                Ok(gpu)
            }
        }

        fn run(&mut self, record: impl FnOnce(&ID3D12GraphicsCommandList)) -> R<()> {
            // SAFETY: our own list and queue, one run at a time.
            unsafe {
                self.allocator.Reset().map_err(e("reset"))?;
                self.list.Reset(&self.allocator, None).map_err(e("reset"))?;
                record(&self.list);
                self.list.Close().map_err(e("close"))?;
                self.queue
                    .ExecuteCommandLists(&[Some(self.list.cast().map_err(e("cast"))?)]);
                self.done += 1;
                self.queue
                    .Signal(&self.fence, self.done)
                    .map_err(e("signal"))?;
                self.fence
                    .SetEventOnCompletion(self.done, self.event)
                    .map_err(e("event"))?;
                if WaitForSingleObject(self.event, 60_000).0 != 0 {
                    return Err("the card did not finish in a minute".into());
                }
                self.device
                    .GetDeviceRemovedReason()
                    .map_err(e("device removed"))
            }
        }

        /// Card milliseconds of one dispatch of `pso` over `src`.
        fn time(
            &mut self,
            pso: &ID3D12PipelineState,
            src: u64,
            groups: u32,
            k: [u32; 4],
        ) -> R<f64> {
            let (root, out, stamps, back) = (
                self.root.clone(),
                self.out.clone(),
                self.stamps.clone(),
                self.stamps_back.clone(),
            );
            let pso = pso.clone();
            // SAFETY: an open list of ours; the views name live buffers.
            self.run(|list| unsafe {
                list.SetComputeRootSignature(&root);
                list.SetPipelineState(&pso);
                list.SetComputeRoot32BitConstants(0, 4, k.as_ptr().cast(), 0);
                list.SetComputeRootUnorderedAccessView(1, src);
                list.SetComputeRootUnorderedAccessView(2, out.GetGPUVirtualAddress());
                list.EndQuery(&stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0);
                list.Dispatch(groups, 1, 1);
                list.EndQuery(&stamps, D3D12_QUERY_TYPE_TIMESTAMP, 1);
                list.ResolveQueryData(&stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &back, 0);
            })?;
            let mut data = std::ptr::null_mut();
            // SAFETY: the readback buffer holds two u64 until unmapped.
            let [a, b] = unsafe {
                self.stamps_back
                    .Map(0, None, Some(&mut data))
                    .map_err(e("map"))?;
                let t = *data.cast::<[u64; 2]>();
                self.stamps_back.Unmap(0, None);
                t
            };
            Ok(b.saturating_sub(a) as f64 * 1000.0 / self.ticks as f64)
        }

        /// Best of five of both passes over `src` of `bytes`.
        fn measure(&mut self, src: u64, bytes: usize) -> R<(f64, f64)> {
            let nodes = (bytes / NODE) as u32;
            let (chase, stream) = (self.chase.clone(), self.stream.clone());
            let (mut lat, mut bw) = (f64::MAX, f64::MAX);
            for _ in 0..5 {
                lat = lat.min(self.time(
                    &chase,
                    src,
                    CHASERS / 64,
                    [nodes, STEPS, bytes as u32, CHASERS],
                )?);
                bw = bw.min(self.time(
                    &stream,
                    src,
                    STREAMERS / 64,
                    [nodes, STEPS, bytes as u32, STREAMERS],
                )?);
            }
            Ok((lat, bw))
        }
    }

    fn committed(
        device: &ID3D12Device,
        kind: D3D12_HEAP_TYPE,
        size: u64,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> R<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: kind,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation.
        unsafe {
            device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &buffer_desc(size, flags),
                state,
                None,
                &mut buffer,
            )
        }
        .map_err(e("buffer"))?;
        buffer.ok_or_else(|| "buffer".into())
    }

    /// One random cycle through all nodes (Sattolo), each node holding the
    /// byte offset of the next.
    fn chains(bytes: usize) -> Vec<u8> {
        let nodes = bytes / NODE;
        let mut order: Vec<u32> = (0..nodes as u32).collect();
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        for i in (1..nodes).rev() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            order.swap(i, (x % i as u64) as usize);
        }
        let mut out = vec![0u8; bytes];
        for (n, &next) in order.iter().enumerate() {
            out[n * NODE..n * NODE + 4].copy_from_slice(&(next * NODE as u32).to_le_bytes());
        }
        out
    }

    /// A buffer placed at the start of `heap`.
    fn placed(device: &ID3D12Device, heap: &ID3D12Heap, bytes: usize) -> R<ID3D12Resource> {
        let mut buffer: Option<ID3D12Resource> = None;
        let flags =
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;
        // SAFETY: plain creation in a live heap at least `bytes` long.
        unsafe {
            device.CreatePlacedResource(
                heap,
                0,
                &buffer_desc(bytes as u64, flags),
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut buffer,
            )
        }
        .map_err(e("placed buffer"))?;
        buffer.ok_or_else(|| "placed buffer".into())
    }

    fn line(name: &str, bytes: usize, r: R<(f64, f64)>, card: Option<(f64, f64)>) {
        match r {
            Ok((lat, bw)) => {
                let ns = lat * 1e6 / f64::from(STEPS);
                let gbs = bytes as f64 / (bw / 1000.0) / 1e9;
                let rel = card.map_or(String::new(), |(cl, cb)| {
                    format!(
                        "  ({:.1}x card latency, {:.1}x card read time)",
                        lat / cl,
                        bw / cb
                    )
                });
                println!(
                    "{name:<18} dependent load {ns:>7.0} ns   streaming read {bw:>7.2} ms = {gbs:>6.1} GB/s{rel}"
                );
            }
            Err(err) => println!("{name:<18} not measured: {err}"),
        }
    }

    pub fn run(bytes: usize) -> R<()> {
        let mut gpu = Gpu::new()?;
        let pattern = chains(bytes);
        println!(
            "heap_speed: {} MB, {CHASERS} lanes x {STEPS} dependent loads, {STREAMERS} lanes streaming; card ms are the best of five",
            bytes >> 20
        );

        // Card memory, for reference: filled by a copy from an upload buffer.
        let card = (|| -> R<(f64, f64)> {
            let up = committed(
                &gpu.device,
                D3D12_HEAP_TYPE_UPLOAD,
                bytes as u64,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let vram = committed(
                &gpu.device,
                D3D12_HEAP_TYPE_DEFAULT,
                bytes as u64,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            )?;
            let mut data = std::ptr::null_mut();
            // SAFETY: the upload buffer's mapping holds `bytes`.
            unsafe {
                up.Map(0, None, Some(&mut data)).map_err(e("map"))?;
                std::ptr::copy_nonoverlapping(pattern.as_ptr(), data.cast::<u8>(), bytes);
                up.Unmap(0, None);
            }
            // SAFETY: two live buffers of `bytes`.
            gpu.run(|list| unsafe { list.CopyBufferRegion(&vram, 0, &up, 0, bytes as u64) })?;
            // SAFETY: a live buffer.
            gpu.measure(unsafe { vram.GetGPUVirtualAddress() }, bytes)
        })();
        let reference = card.as_ref().ok().copied();
        line("card memory", bytes, card, None);

        // A heap the driver makes in the PC's memory, write-back, pool L0.
        let custom = (|| -> R<(f64, f64)> {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            // SAFETY: plain creation.
            unsafe {
                gpu.device.CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &buffer_desc(bytes as u64, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
            }
            .map_err(e("custom buffer"))?;
            let buffer = buffer.ok_or("custom buffer")?;
            let mut data = std::ptr::null_mut();
            // SAFETY: the mapping holds `bytes`; it stays mapped while used.
            unsafe {
                buffer.Map(0, None, Some(&mut data)).map_err(e("map"))?;
                std::ptr::copy_nonoverlapping(pattern.as_ptr(), data.cast::<u8>(), bytes);
            }
            // SAFETY: a live buffer.
            gpu.measure(unsafe { buffer.GetGPUVirtualAddress() }, bytes)
        })();
        line("driver custom L0", bytes, custom, reference);

        // Committed private memory handed to the driver.
        let private = (|| -> R<(f64, f64)> {
            // SAFETY: our own allocation, released below.
            let at = unsafe {
                mem::VirtualAlloc(
                    std::ptr::null(),
                    bytes,
                    mem::MEM_RESERVE | mem::MEM_COMMIT,
                    mem::PAGE_READWRITE,
                )
            };
            if at.is_null() {
                return Err("VirtualAlloc".into());
            }
            // SAFETY: `bytes` of our own committed memory.
            unsafe { std::ptr::copy_nonoverlapping(pattern.as_ptr(), at.cast::<u8>(), bytes) };
            let result = (|| {
                let device3: ID3D12Device3 = gpu.device.cast().map_err(e("ID3D12Device3"))?;
                // SAFETY: the base of a whole committed allocation.
                let heap: ID3D12Heap =
                    unsafe { device3.OpenExistingHeapFromAddress(at as *const c_void) }
                        .map_err(e("open"))?;
                let buffer = placed(&gpu.device, &heap, bytes)?;
                // SAFETY: a live buffer.
                let r = gpu.measure(unsafe { buffer.GetGPUVirtualAddress() }, bytes);
                drop((buffer, heap));
                r
            })();
            // SAFETY: our own allocation; the heap on it is gone.
            unsafe { mem::VirtualFree(at, 0, mem::MEM_RELEASE) };
            result
        })();
        line("imported private", bytes, private, reference);

        // A view of a section handed to the driver, as port 1 opens the title's heap.
        let view = (|| -> R<(f64, f64)> {
            // SAFETY: a pagefile-backed section of our own and a view of it, both let go below.
            let section = unsafe {
                mem::CreateFileMappingW(
                    INVALID_HANDLE_VALUE.0 as _,
                    std::ptr::null(),
                    mem::PAGE_READWRITE | mem::SEC_COMMIT,
                    (bytes as u64 >> 32) as u32,
                    bytes as u32,
                    std::ptr::null(),
                )
            };
            if section.is_null() {
                return Err("CreateFileMappingW".into());
            }
            // SAFETY: the whole section, read and write.
            let at = unsafe { mem::MapViewOfFile(section, mem::FILE_MAP_ALL_ACCESS, 0, 0, bytes) };
            if at.Value.is_null() {
                // SAFETY: our own handle.
                unsafe { CloseHandle(HANDLE(section)).ok() };
                return Err("MapViewOfFile".into());
            }
            // SAFETY: `bytes` of the committed view.
            unsafe {
                std::ptr::copy_nonoverlapping(pattern.as_ptr(), at.Value.cast::<u8>(), bytes)
            };
            let result = (|| {
                let device13: ID3D12Device13 = gpu.device.cast().map_err(e("ID3D12Device13"))?;
                // SAFETY: a committed view of `bytes`.
                let heap: ID3D12Heap = unsafe {
                    device13.OpenExistingHeapFromAddress1(at.Value as *const c_void, bytes)
                }
                .map_err(e("open"))?;
                let buffer = placed(&gpu.device, &heap, bytes)?;
                // SAFETY: a live buffer.
                let r = gpu.measure(unsafe { buffer.GetGPUVirtualAddress() }, bytes);
                drop((buffer, heap));
                r
            })();
            // SAFETY: our own view and handle; the heap on them is gone.
            unsafe {
                mem::UnmapViewOfFile(at);
                CloseHandle(HANDLE(section)).ok();
            }
            result
        })();
        line("imported view", bytes, view, reference);
        println!(
            "(imported view is how renderer port 1 opens the title's 32-bit graphics heap today)"
        );
        Ok(())
    }
}
