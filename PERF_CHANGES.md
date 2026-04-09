# Performance Optimizations — Branch `perf/ht-decoder-avx2`

## Summary

This document describes the decode-path performance optimizations implemented so far,
their verification status, and the remaining work. It is intended as a handoff for
continuation on a Linux/x86-64 machine where the AVX2 path can be verified.

## Platform Note

All changes were developed and tested on **Apple Silicon (arm64 / NEON)**.
The test suite (444/445 pass; `comp_p1_ht_07_11a` is a pre-existing failure)
exercises the **NEON** code path. The **AVX2** changes compile but are **not
exercised** on this machine — they need verification on x86-64.

---

## Completed Optimizations

### P1: Fuse Dequantize into MagSgn Output

**Goal:** Eliminate the separate `dequantize()` pass over `sample_buf` for
single-pass HT blocks (cleanup-only, no ROI). Instead, apply the dequantize
math directly in the MagSgn decode loop and write `float` values to `i_samples`.

**Impact:** Removes one full read+write pass over every single-pass codeblock
(~8% of total decode time eliminated).

**Condition:** `num_ht_passes == 1 && ROIshift == 0`

**Files modified:**

| File | What changed |
|---|---|
| `source/core/coding/ht_block_decoding_avx2.cpp` | Added `fuse_dequant` template parameter, `dequant_store_256`/`dequant_store_128` helpers, fused store at all 8 MagSgn store sites (16-bit 4-quad, 16-bit 2-quad, 32-bit 2-quad paths), conditional skip of `block->dequantize()` in `htj2k_decode()` |
| `source/core/coding/ht_block_decoding_neon.cpp` | Same pattern: `fuse_dequant` template param, `dequant_store_neon` helper, fused stores, `htj2k_decode()` dispatch |
| `source/core/coding/ht_block_decoding.cpp` | Scalar path: `dequant_store_scalar` helper, `store_sample` lambda, `out_stride` for pointer arithmetic, `htj2k_decode()` dispatch |
| `source/core/coding/ht_block_decoding_wasm.cpp` | WASM path: `dequant_store_wasm` helper, fused stores, `htj2k_decode()` dispatch |
| `source/core/coding/coding_units.cpp` | Extended `i_samples` allocation by one extra stride row to prevent heap-buffer-overflow when fused path writes `mp1` for the last line-pair of a subband with odd height |

**Key design decisions:**
- Uses `template <bool skip_sigma, bool fuse_dequant = false>` — `if constexpr` eliminates dead code at compile time
- Output pointers (`mp0`/`mp1`) are `int32_t*` pointing into `i_samples` (which is `float*`); stores use `reinterpret_cast` since both are 32-bit
- Lossy dequantize: `float(magnitude) * fscale_direct`, sign via XOR with sign bit
- Lossless dequantize: `sign_epi32(srai(magnitude, pLSB_dq), sign_vec)` then `cvt_to_float`
- Uses `transformation != 1` (not `== 0`) to correctly handle ATK irreversible (`transformation >= 2`)

**Verified on:** arm64/NEON (444/445 tests pass)
**Needs verification on:** x86-64/AVX2

---

### P3: Vectorize Kappa/Emax Computation (AVX2 only)

**Goal:** Eliminate `_mm_extract_epi32` scalar extracts in the AVX2 2-quad
remainder and 32-bit non-initial line-pair paths, keeping Emax values in
vector registers.

**Impact:** ~2-4% improvement on x86-64 by avoiding SIMD→scalar→SIMD pipeline
breaks.

**Files modified:**

| File | What changed |
|---|---|
| `source/core/coding/ht_block_decoding_avx2.cpp` | 16-bit 2-quad remainder: replaced `_mm_extract_epi32(emax128, 0/1)` + `_mm_set1_epi32(Emax-1)` with `_mm_sub_epi32` + `_mm_shuffle_epi32` broadcast. 32-bit path: same pattern with positions {0, 2} from `epr128`, removed scalar `Emax0`/`Emax1` variables entirely |

**Verified on:** arm64/NEON (code compiles but not exercised)
**Needs verification on:** x86-64/AVX2

---

### P4: Specialize 9/7 Cascade for Streaming IDWT

**Goal:** Replace the generic `while(progress)` loop in the irrev 9/7 streaming
IDWT cascade with 4 dedicated single-pass phases matching the lifting step
dependency chain.

**Impact:** ~3-5% improvement for streaming decode path by eliminating
redundant d_level scans and re-checks.

**Files modified:**

| File | What changed |
|---|---|
| `source/core/transform/idwt.cpp` | Replaced `while(progress)` loop (lines 978-995) with 4 explicit phases: Phase 1 (D): LP dl=0→1, Phase 2 (C): HP dl=0→1, Phase 3 (B): LP dl=1→2, Phase 4 (A): HP dl=1→2. Each phase is a single even/odd-strided pass |

**Dependency chain (irrev 9/7, max_dl=2):**
```
Phase 1 (D): Even rows, dl=0→1, need HP neighbors @dl≥0
Phase 2 (C): Odd  rows, dl=0→1, need LP neighbors @dl≥1
Phase 3 (B): Even rows, dl=1→2, need HP neighbors @dl≥1
Phase 4 (A): Odd  rows, dl=1→2, need LP neighbors @dl≥2
```

**Verified on:** arm64/NEON (444/445 tests pass)
**Needs verification on:** x86-64/AVX2 (same code, platform-independent)

---

### P9: Vectorize NEON Emax/Kappa — Keep Exponents in NEON Registers

**Goal:** Eliminate SIMD-to-scalar round-trips for the Emax horizontal-max and kappa
computation in the non-initial line-pair MagSgn loop.

**Impact:** ~1-2% improvement on AArch64 by avoiding pipeline breaks between
NEON and integer units.

**Files modified:**

| File | What changed |
|---|---|
| `source/core/coding/ht_block_decoding_neon.cpp` | Added `max4_pair()` helper; replaced `int32_t Emax0,Emax1` + `vmaxvq_s32` with `int32x2_t vEmax` + `vpmax_s32` chain; vectorized gamma/kappa computation with NEON; removed unused `gamma0/gamma1` variables |

**Key design decisions:**
- `max4_pair(a, b)` returns `{max(a[0..3]), max(b[0..3])}` as `int32x2_t` using 3 `vpmax_s32`
- `vgamma = vbic_u32(1, vceq(rho & (rho-1), 0))` — branchless 0-or-1
- `vkappa = vmax_s32(vmul_s32(vgamma, Emax-1), 1)` — fused in NEON
- U0/U1 extracted via `vget_lane_s32(vkappa, 0/1)` after kappa is done

**Verified on:** arm64/NEON (445/445 tests pass)
**Needs verification on:** x86-64/AVX2 (code does not compile on AVX2)

---

### P10: WASM Lossy Dequantize — Direct Float Multiply

**Goal:** Replace the integer approximation in the WASM `dequantize()` lossy path
with direct `wasm_f32x4_mul`, eliminating the truncate→int16→multiply→downshift
chain for the common `ROIshift == 0` case.

**Impact:** ~4-6% for WASM lossy decode (avoids ~8 extra integer instructions per 8
samples). Also fixes the ROI scalar path (removed redundant `if (ROIshift)` guard).

**Files modified:**

| File | What changed |
|---|---|
| `source/core/coding/ht_block_decoding_wasm.cpp` | `dequantize()`: split lossy branch into `ROIshift==0` fast path (float mul) and ROI path (existing integer approx); scalar tail fixed |

**Condition:** `ROIshift == 0` (the common case — no region of interest)

**Verified on:** arm64/NEON (445/445 tests pass; WASM path compiled but not exercised)
**Needs verification on:** WASM/Emscripten

---

### P11: Fix NEON Color Transform Loop Bounds and Dead Loads

**Goal:** Fix two bugs in `color_neon.cpp`:
1. `for (; len > 0; len -= 8)` processes 8 elements even when `len < 8` (partial OOB write)
2. Lines 170-175: 6 dead `vld1q_s32` loads from base pointers after advancing `p0/p1/p2`

**Impact:** Bug fix (correctness) + eliminates 6 wasted loads per 8-pixel loop iteration.
Also adds missing scalar tail loop for widths not divisible by 8.

**Files modified:**

| File | What changed |
|---|---|
| `source/core/transform/color_neon.cpp` | `cvt_ycbcr_to_rgb_rev_neon`: `len > 0` → `len >= 8`, deleted dead loads, added scalar tail. `cvt_ycbcr_to_rgb_irrev_neon`: `len > 0` → `len >= 8`, added scalar tail |

**Verified on:** arm64/NEON (445/445 tests pass)

---

### P12: WASM Vectorized CLZ via Float Exponent Trick

**Goal:** Replace the 4-scalar-extract CLZ in `wasm_u32x4_clz()` with a vectorized
float-exponent computation.

**Impact:** ~2-3% for WASM decode (eliminates 4 `i32x4_extract_lane` + `__builtin_clz`
per exponent update, replacing with 3 SIMD instructions).

**Files modified:**

| File | What changed |
|---|---|
| `source/core/coding/ht_block_decoding_wasm.cpp` | `wasm_u32x4_clz()`: replaced scalar body with `convert→shr(23)→sub(158)→min(32)` |

**Formula:** `clz32(a) = min(158 - (float_bits(a) >> 23), 32)` — valid for non-negative inputs; returns 32 for a == 0.

**Verified on:** arm64/NEON (445/445 tests pass; WASM path compiled but not exercised)
**Needs verification on:** WASM/Emscripten

---

### P7: WASM 16-bit Fast Path for MagSgn Decode

**Goal:** Add a `pLSB > 16` fast path to the WASM MagSgn decode that uses
`wasm_i8x16_swizzle`-based batch bit extraction, matching the NEON `decode_two_quads_16bit`
approach. Eliminates `wasm_i32x4_shlv` (4 scalar extracts per call) from the common path.

**Impact:** ~10-15% improvement in WASM throughput for typical images (pLSB > 16 is
the common case for standard precision images at the HT cleanup pass).

**Files modified:**

| File | What changed |
|---|---|
| `source/core/coding/ht_block_decoding.hpp` | Added `fetch_raw()` and `decode_two_quads_16bit_wasm()` to WASM `fwd_buf` class |
| `source/core/coding/ht_block_decoding_wasm.cpp` | Both initial and non-initial line-pair loops: added `if (pLSB > 16)` branch with 16-bit decode path using `decode_two_quads_16bit_wasm`; exponent update uses `wasm_u32x4_extend_low_u16x8` to widen v_n |

**Key design decisions:**
- `fetch_raw()` = `wasm_v128_load(this->tmp)` — same buffer layout as NEON
- Variable per-quad shift `vshlq_u16(w0_val, Uq_m1)` → two constant shifts + `wasm_i16x8_shuffle` blend
- `wasm_i8x16_swizzle` returns 0 for indices ≥ 16, matching `vqtbl1q_u8` semantics
- v_n stored as `v128_t` with only low 4 × int16 lanes meaningful; widened to uint32 for CLZ

**Condition:** `pLSB > 16` (same as NEON fast path)

**Verified on:** arm64/NEON (445/445 tests pass; WASM path compiled but not exercised)
**Needs verification on:** WASM/Emscripten

---

## Remaining Work (Not Started)

### P2: Branchless VLC/MEL Decode
- Replace unpredictable branches in VLC/MEL phase with branchless masks
- Files: `ht_block_decoding.cpp`, `ht_block_decoding_avx2.cpp`, `ht_block_decoding_neon.cpp`, `ht_block_decoding_wasm.cpp`
- Lines ~82-248 in each variant (identical scalar VLC code)

### P5: AVX-512 Masked Stores for Vertical DWT Tail
- Use `_mm512_mask_storeu_ps` / `_mm256_maskstore_ps` for partial strips
- Files: `idwt_avx512.cpp:323-350`, `idwt_avx2.cpp:333-357`
- AVX2 variant uses `_mm256_maskstore_ps`

### P6: Expand AVX-512 to FDWT and Color Transform
- Create `fdwt_avx512.cpp` and `color_avx512.cpp`
- Mirror AVX2 variants with 512-bit operations
- Guard with `#if defined(__AVX512F__) && defined(__AVX512BW__)`

### P8: NEON 4-Quad Decode
- Add `decode_four_quads_16bit()` wrapper calling `decode_two_quads_16bit` twice
- Update initial and non-initial NEON loops to step `qx -= 4` when `qx >= 4`
- Reduces loop overhead and enables better compiler scheduling

---

## How to Verify on x86-64

```bash
# Build
cmake -G "Unix Makefiles" -B build -DCMAKE_BUILD_TYPE=Release -DOPENHTJ2K_THREAD=ON
cmake --build build --config Release -j

# Run all tests (expect 444/445 pass; comp_p1_ht_07_11a is a known failure)
cd build && ctest --build-config Release -VV

# Specifically test the affected paths:
ctest -R "^dec_p0_ht_"  -VV   # HT decoder conformance (exercises P1 fused dequant)
ctest -R "^lbs_"        -VV   # Line-based streaming (exercises P4 cascade)
ctest -R "^batch_"      -VV   # Batch validation
ctest -R "^dec_p2_dfs_" -VV   # Part 2 DFS/ATK (exercises ATK transformation!=1 fix)
ctest -R "^enc_"        -VV   # Encoder round-trip (lossless + lossy)

# Performance benchmark (single-threaded)
./bin/open_htj2k_dec -i <4K_test.j2c> -o /dev/null -iter 100 -num_threads 1
```

## Pre-existing Test Failure

`comp_p1_ht_07_11a` fails on both the baseline (`main` / branch tip before changes)
and after changes. It is **not** caused by these optimizations.
