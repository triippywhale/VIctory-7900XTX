# What xcl-dev/xcl has for running the title on PC (2026-11-27)

[xcl-dev/xcl](https://github.com/xcl-dev/xcl) is another Xbox Series → Windows
layer in Rust (GPL-3.0-or-later, as this fork): the title's x64 code runs
natively, console libraries are its own, RDNA2 code is translated to DXIL
(directly, or through HLSL and DXC). Read at `main` 6089122a, `amd-dev`
7c795f5e (302 commits past main), `nvidia-dev` and `shader-prep`.

Its README puts a newer retail title at about 5 fps in the world on an RTX
5090 (the NVIDIA branch), menus at 30 fps, "not playable yet". This fork runs
the title at 9-10 fps with ray tracing on a 7900 XTX, so xcl is not ahead on
the title. What it has that we lacked is a set of AMD driver findings from
its RX 9070 / 6800 / 6700 XT contributors.

## Taken (this change)

| From xcl | Here |
|---|---|
| ea088a58: `image_get_resinfo` at a level past the last gives 0 sizes; AMD's driver answers with the sizes of the level it clamped to | The size query compares the level with the view's level count; past it, every size reads 0 as on the console. Translations change |
| 5b3f0a4b: `amdxc64` folds `Bfrev(Countbits(K))` of a constant K wrong (0x80000001 for K = 2) | The optimiser's FOLD pass computes `Countbits` and `Bfrev` of 32-bit constants itself, so the driver never sees the pair. Test `bit_counts_and_reversals_of_constants_fold` |
| 0574950e: an AMD driver gives a D3D12 device to nine processes at once; a tenth fails with 0x887A0005 | `victory warm` starts at most 8 worker processes on an AMD card by default (it started two thirds of the logical processors: 16 on a 24-thread CPU). `--jobs` is taken as given |

`TRANSLATOR_VERSION` 5 → 7 for the first two (pull request #41 takes 6):
cached translations are made again (`victory shaders`).

## Already here (checked, nothing to take)

- Pixel-shader helper lanes in `v_readfirstlane` / waterfall loops (xcl
  6ec402b1, f44e17b7): `Lower::first_lane` and the helper-lane rule.
- `s_barrier` fencing UAV memory where the program writes it (85a9480d).
- `[earlydepthstencil]` from `DB_SHADER_CONTROL` (324193c0).
- Thread-dimension dispatches, flag 0x20 (32f09674 part 2): `draw.rs`.
- Ray node tests on missing pages and 128-byte nodes across a page (9f02a3b2):
  `intersect.rs` reads such a page as zeros and gives no hit, and notes the
  page for the host.
- i64 `Bfrev` / `Countbits` bugs (dc2d8f1a, a5af72bb): our translation only
  emits the 32-bit forms.
- Compute lanes in 2-D / 3-D groups (46a1c3f4): xcl used the host thread ids
  as they came; ours relabels each thread from its wave's first flat id, which
  holds for any order inside an aligned run of ids (the 2 x 2 quad order xcl
  measured keeps runs aligned for the group shapes it names). The split form
  takes ids from an LDS counter already (`wave_base`).
- Pipelines made ahead and a pipeline library kept between runs
  (`xcl pso-precompile`, the `.plib` work): `victory warm`, `translate --cache
  --pipelines`, `pipeline/library.rs`.

## Not taken, and why

- Direct DXIL (2026-11-24-direct-dxil.md): ours has always written DXIL
  itself.
- AMD `GetCachedBlob` stubs under 4 KiB (1dcfde8c, 806eee33): we keep
  pipelines in `ID3D12PipelineLibrary1`, not as cached blobs.
- Serialising every `Create*PipelineState` on AMD (32f09674 part 1): xcl made
  it opt-in again (b1248844); our compiler-crash guard and per-pipeline
  workers cover crashes.
- Compute loop budgets kept to one counter for AMD's LLPC (bfb40311): we have
  no per-region loop budgets in compute.
- MSAA `textureLoad` before `getDimensions` (425b8692): a shape only xcl's
  lowering made.
- Their memory, sparse-tile and mirror work (most of `amd-dev`): built
  around xcl's own memory model (UMA windows, sparse mirrors), which this
  fork does differently (renderer port 1 reads the title's memory in place).
- BCPack lookup tables (558d10a5): `docs/README-fxceless.md` measured loading
  as limited by writes into title memory, not by unpacking.

None of this was run in the game here.
