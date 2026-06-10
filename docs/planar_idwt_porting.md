# Porting the planar horizontal IDWT to AVX2 / AVX-512 / WASM

Status as of PR #406 (June 2026): the streaming horizontal IDWT lifts directly
from the planar LP/HP subband rows **on NEON only** (9/7 float and int32 5/3).
All other platforms — AVX2, AVX-512, WASM, scalar — go through a bit-identical
fallback that interleaves into the ring slot and runs the historical in-place
kernels, at unchanged cost. This document is the map for converting those
platforms. It assumes no access to the NEON machine; everything needed is in
the tree.

Measured upside on NEON (8K 12-bit, M3 Max, single thread, 30-iter mean):
lossy 9/7 **−5.5%**, lossless 5/3 i32 **−2.5%** end-to-end decode. On AVX2 the
upside is plausibly larger (see §5.1): the current in-place kernels waste half
their lanes on pass-through elements, which the planar formulation does not.

---

## 1. Architecture after PR #406

```
idwt_level_src_fn (coding_units.cpp)          ← selects lp_ptr/hp_ptr, zero scan
        │
        ▼
idwt_1d_row_from_planar (idwt.cpp)            ← THE dispatcher; the only place
        │                                        that decides planar vs fallback
        ├─ planar kernel        (NEON today)  ← lp/hp planes → natural row in
        │                                        the ring slot; no interleave
        └─ fallback: interleave_row_planes()  ← LP at u0%2, HP at 1-u0%2, then
           + idwt_1d_row_inplace[_range|_i32]    in-place PSE fill + lifting
```

Key property that keeps the port contained: **ring-buffer rows hold
natural-domain rows**. The interleaved format only ever existed between the
source callback and the horizontal kernel. Vertical lifting, the cascade,
zero-row tracking, PSE row copies, `pull_row_ref`, and the whole batch path
(`idwt_2d_sr_fixed`) are untouched and must stay untouched.

The port therefore adds, per platform:

1. Two kernels in `idwt_avx2.cpp` / `idwt_wasm.cpp` (and later `idwt_avx512.cpp`):
   - `idwt_1d_filtr_irrev97_planar_<isa>(sprec_t *out, const sprec_t *lp, const sprec_t *hp, int32_t u0, int32_t u1)`
   - `idwt_1d_filtr_rev53_planar_i32_<isa>(int32_t *out, const int32_t *lp, const int32_t *hp, int32_t u0, int32_t u1)`
2. Declarations in the matching `#elif` block of `dwt.hpp`.
3. An `#elif` branch in the planar fast path of `idwt_1d_row_from_planar`
   (idwt.cpp), which today reads `#if defined(OPENHTJ2K_ENABLE_ARM_NEON)`.
   Note `OPENHTJ2K_ENABLE_AVX2` is also defined on AVX-512 builds, so an AVX2
   planar kernel automatically serves AVX-512 hosts until a dedicated
   `_avx512` variant exists (the in-place dispatch tables prefer AVX-512; the
   planar dispatch can start with AVX2 only).

Nothing else changes. `coding_units.cpp` already calls the dispatcher.

## 2. Dispatcher contract (do not relax without re-deriving §3)

The planar fast path runs only when **all** of these hold; everything else
falls back (and must keep falling back):

| Guard | Why |
|---|---|
| `(u0 & 1) == 0` | keeps `E[j] = lp[j]`, `O[j] = hp[j]` index-aligned; odd u0 shifts the LP plane by one and is rare (odd tile/precinct origins) |
| `N = u1/2 - u0/2 >= 12` | the kernels' SIMD warmup loads blocks j=0..7 unconditionally; short rows go through the fallback's scalar-friendly path |
| 9/7 float: `col_lo <= u0 && col_hi >= u1` | narrow JPIP column ranges use the scalar sub-range lifter on the interleaved row; not worth porting |
| 5/3: `use_i32 == true` | the float-5/3 streaming path is cold (lossless decode is i32); i32 ignores the column range by design, matching `idwt_1d_row_inplace_i32` |

Buffer contract the dispatcher guarantees to the kernel:

- `out` is a ring-slot (or `horz_out_buf`) data pointer: **8 writable floats
  before index 0** (`IDWT_RING_PSE_LEFT`) and **≥ 32 after index width-1**
  (`SIMD_PADDING` in `slot_stride`). The kernels write `out[-2]`, `out[-1]`
  (the "final E[-1]/O[-1]" the in-place kernel also produced) and up to
  `out[2N+2]` past the data end — all within that slack.
- `lp` has `ceil(u1/2) - u0/2` valid samples (= N or N+1), `hp` has exactly N.
  **There are no PSE margins on the planes** — never load past those counts
  (see §3 loop bounds). Plane sources are codeblock band rows, the child
  state's ring row, or the zeroed `lp_tmp`; over-reading band rows is not
  covered by any allocation guarantee at these widths.
- `out` never aliases `lp`/`hp`.

## 3. The math (identical on every platform)

### 3.1 Recurrences

With `E[j] = lp[j]`, `O[j] = hp[j]`, `N = u1/2 - u0/2` (u0 even — guard above):

9/7 synthesis, fused single pass (same scheme as the in-place fused kernel):

```
S1[j] = E[j]  - fD*(O[j-1]  + O[j])     j ∈ [-1, N+1]
S2[j] = O[j]  - fC*(S1[j]   + S1[j+1])  j ∈ [-1, N]
S3[j] = S1[j] - fB*(S2[j-1] + S2[j])    j ∈ [ 0, N]
S4[j] = S2[j] - fA*(S3[j]   + S3[j+1])  j ∈ [ 0, N-1]
out:  E[-1]=S1[-1], O[-1]=S2[-1], E[j]=S3[j], O[j]=S4[j], O[N]=S2[N], E[N+1]=S1[N+1]
```

int32 5/3 synthesis:

```
S1[j] = E[j] - ((O[j-1] + O[j] + 2) >> 2)   j ∈ [0, N]
S2[j] = O[j] + ((S1[j]  + S1[j+1]) >> 1)    j ∈ [0, N)
out:  E[j]=S1[j] (→ out[2j]), O[j]=S2[j] (→ out[2j+1])
```

### 3.2 Boundary taps — mirror within the plane

Whole-sample symmetric extension preserves parity: positions `u0-k` and
`u0+k` are both even or both odd, so **extension never crosses planes**. Any
raw `E[j]`/`O[j]` outside the valid range maps to an in-range plane index via
`PSEo` on the absolute position:

```cpp
auto E = [&](int32_t j) -> float { return lp[PSEo(u0 + 2 * j,     u0, u1) >> 1]; };
auto O = [&](int32_t j) -> float { return hp[PSEo(u0 + 2 * j + 1, u0, u1) >> 1]; };
```

These return *exactly* the values the in-place kernels found in their
PSE-filled margins (that is the bit-exactness argument for the boundaries; do
not hand-derive per-parity mirror formulas, just use `PSEo`). They are used
only in the scalar warmup/drain — a dozen taps per row.

### 3.3 Loop bounds (the one place planar differs from in-place)

The in-place fused NEON kernel ran its SIMD loop to `4n+3 <= N+1` because the
interleaved row had PSE-filled margins to read. The planes do not, so the
planar SIMD loop stops at **`4n+3 <= N-1`** (NEON, 4-lane; for 8-lane AVX2 the
analogous bound keeps every plane load ≤ index N-1) and the scalar drain
covers the remaining j with the mirrored accessors. Same for the 5/3 i32 main
loop: `j + 4 <= N - 1`, because the `s1_next` lookahead reads `lp[j+4]` /
`hp[j+4]`.

### 3.4 Bit-exactness mandate

`lbs_*` tests enforce batch == line-based bit-exact, and the dispatcher's
fallback must equal the planar path. So a planar kernel must reproduce **the
same single-rounded operation sequence per element as the same platform's
in-place kernel**:

- 9/7: one fused multiply-add per stage. NEON uses `vfmsq_f32` / `std::fmaf`;
  the existing AVX2 in-place kernel uses `_mm256_fnmadd_ps` — also
  single-rounded, and `fnmadd(sum, coeff, x) == fmaf(-coeff, sum, x)`, so the
  NEON pipeline math transfers as-is. Keep `std::fmaf` in the scalar
  warmup/drain on every platform. The neighbor sum is always
  `(left + right)` in that order, then the FMA.
- 5/3 i32: integer adds/shifts — exact regardless of structure.
- 5/3 float (if ever ported): `floorf((a+b+2)*0.25f)` / `floorf((a+b)*0.5f)`
  with `_mm256_floor_ps` / `wasm_f32x4_floor`; mul+floor, **not** FMA.

Verify against the conformance suite *and* a whole-image `cmp` (§6) — the
tests alone don't cover 8K-scale row widths.

## 4. The NEON reference implementations

Read these first; they are the template (`source/core/transform/idwt_neon.cpp`,
bottom of the file):

- `idwt_1d_filtr_irrev97_planar_neon` — warmup (scalar j=-1 taps + S1 of
  blocks 0/1 + S2/S3 of block 0), steady state (a depth-2 software pipeline:
  iteration n loads E/O block n with one `vld1q` per plane and stores finished
  natural-row block n-2 with one `vst2q`), drain (scalar, ≤ 12 elements,
  mirror accessors). The cross-element shifts (`O[j-1]`, `S1[j+1]`, `S2[j-1]`,
  `S3[j+1]`) are `vextq_f32` register shuffles between the previous and
  current block's vectors.
- `idwt_1d_filtr_rev53_planar_i32_neon` — single loop, no pipeline depth: the
  one cross-block lane `S1[j+4]` is computed scalar from raw memory
  (`s1_next`) and `vextq`-ed in. Scalar tail of ≤ 5 elements.

Output-side note: the `vst2q` (interleave-on-store) is **irreducible** — the
natural-domain row *is* alternating E/O. The win is on the input side (one
plain load per plane instead of deinterleaving loads) plus the deleted
interleave pass in the caller.

## 5. Platform notes

### 5.1 AVX2 (do this one first)

The current in-place AVX2 kernels (`idwt_avx2.cpp`) do **not** deinterleave:
each pass loads overlapping unaligned vectors over the interleaved row and
updates only every other element, using a `_mm256_slli_epi64(…, 32)` trick to
zero the pass-through lanes. Net effect: 4 useful results per 8-lane op, four
passes over the row, plus the caller's interleave pass. A planar kernel gets 8
useful results per op and one pass — which is why the expected gain here
exceeds NEON's.

Porting the NEON pipeline to 8 lanes, the only non-mechanical part is the
cross-element shift of *computed* vectors (raw `O[j-1]` can simply be an
unaligned reload from `hp + 4n - 1` in steady state, where j-1 ≥ 0 — cheaper
than shuffling on x86). For the pipeline-internal values use the standard
two-instruction idiom (no AVX-512 needed):

```cpp
// y = [a1..a7, b0]   (shift left by one element, carry-in from next vector b)
__m256 t = _mm256_permute2f128_ps(a, b, 0x21);          // [a4..a7, b0..b3]
__m256 y = _mm256_castsi256_ps(_mm256_alignr_epi8(
             _mm256_castps_si256(t), _mm256_castps_si256(a), 4));

// z = [p7, a0..a6]   (shift right by one element, carry-in from previous p)
__m256 t = _mm256_permute2f128_ps(p, a, 0x21);          // [p4..p7, a0..a3]
__m256 z = _mm256_castsi256_ps(_mm256_alignr_epi8(
             _mm256_castps_si256(a), _mm256_castps_si256(t), 12));
```

Interleaved store: `unpacklo/hi_ps` + `permute2f128` (the inverse of the
pattern in `interleave_row_planes`, idwt.cpp).

A lower-effort fallback if the full pipeline proves fiddly: keep the four-pass
in-place structure but fuse the interleave into pass 1 (read `lp`/`hp`, write
the S1-updated interleaved row). That removes only the caller's interleave
pass (~half the win) but is a small, low-risk diff. Measure before settling.

Warmup/drain: reuse the NEON scalar code verbatim (`std::fmaf`, mirror
accessors) — it is platform-independent. With 8-lane blocks the drain spans up
to ~20 elements; size the `s1t/s2t/s3t` staging arrays accordingly (NEON uses
16 for 4-lane blocks; 8-lane needs 32 — re-derive from the loop-exit bound the
way the NEON drain comment does, don't guess).

### 5.2 AVX-512

`idwt_avx512.cpp` exists for IDWT only. Don't start here: ship the AVX2
planar kernels first (they run on AVX-512 hosts via `OPENHTJ2K_ENABLE_AVX2`),
then evaluate whether 16-lane + `vpermt2ps` (single-instruction cross-lane
shifts) pays for a dedicated variant. Keep the dispatch precedence consistent
with the in-place tables (AVX-512 before AVX2) when adding it.

### 5.3 WASM

`idwt_wasm.cpp` currently emulates `vld2q/vst2q` with paired loads +
`wasm_i32x4_shuffle` (helpers at the top of the file). The planar port is a
near line-for-line transcription of the NEON kernels at the same 4-lane width:

- `vextq_f32(a, b, k)` → `wasm_i32x4_shuffle(a, b, k, k+1, k+2, k+3)`
- **Rounding differs from NEON/AVX2 — this is the one place the NEON code
  cannot be transcribed blindly.** The in-place WASM 9/7 kernel uses
  `wasm_f32x4_sub(x, wasm_f32x4_mul(sum, coeff))` — separately-rounded
  mul+sub, not a fused op (SIMD128 has none). The planar kernel must match:
  mul+sub in vectors, and plain `x - coeff*(a+b)` float expressions in the
  scalar warmup/drain — NOT `std::fmaf`. (Plain scalar expressions are safe
  on WASM: there is no scalar FMA instruction for clang to contract into, so
  they compile to f32.mul + f32.sub, matching the vector lanes.) The 5/3 i32
  kernel has no such concern.
- store: `wasm_i32x4_shuffle` zip pair + two `wasm_v128_store` (same as the
  existing `vst2q` emulation).

Build/test: `subprojects/` (emcmake) where checked out; minimum bar without a
browser harness is `emcc -fsyntax-only -msimd128 -DOPENHTJ2K_ENABLE_WASM_SIMD`
on the touched files plus the native test suite (the WASM guards compile to
nothing natively, so native ctest only proves you didn't break the others).

## 6. Verification protocol (per platform, no shortcuts)

1. **ctest**: full Release suite (407 tests at time of writing) + Debug.
2. **Whole-image bit-exactness** vs `main` at 8K scale, lossless and lossy:
   ```
   open_htj2k_enc -i u01_Books_8k_12bit.ppm -o 8k_lossless.j2c Creversible=yes
   open_htj2k_enc -i u01_Books_8k_12bit.ppm -o 8k_lossy.j2c    Creversible=no Qfactor=85
   <baseline_bin>/open_htj2k_dec -i 8k_<x>.j2c -o base.ppm -num_threads 1
   <new_bin>/open_htj2k_dec      -i 8k_<x>.j2c -o new.ppm  -num_threads 1
   cmp base.ppm new.ppm     # must be identical, both streams
   ```
3. **A/B benchmark**: snapshotted `bin/` dirs (baseline = a worktree of main,
   built once), interleaved runs, `-iter 30 -num_threads 1`, ≥ 3 repetitions
   per config, both streams. Don't trust a single pair of runs.
4. **lb_compare** on at least one stream (`lb_compare <codestream>`) — directly
   exercises batch-vs-streaming equality, which is exactly the fallback-vs-
   planar boundary.

## 7. Encoder mirror (separate work, not this port)

`fdwt_level_sink_fn` (coding_units.cpp) still deinterleaves the FDWT
horizontal output before quantizing (NEON `vld2q` + fused quantize). The
symmetric change — FDWT horizontal kernels writing LP/HP planes with plain
stores, sink quantizing planes directly — is unstarted on every platform.
There the *input*-side `ld2` of the natural row is the irreducible half.
