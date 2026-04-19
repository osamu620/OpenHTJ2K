# JPIP Large-Image Viewer Plan (revised)

Interactive pan+zoom viewer for gigapixel images via JPIP.

## Test image

`heic2501a.j2c` — 42,208 x 9,870 (416 MP), 270 MB, 103,179 precincts.

## V1 status (completed)

The viewer works but has two problems at high zoom:

1. **Blurry** — the decoded texture is limited to 4096 pixels
   (maxDim cap) regardless of zoom.  At 50% zoom on a 42K image,
   the viewport shows ~21K canvas pixels but the texture has only
   4096 — a 5:1 downsampling that smears all detail.

2. **Slow** — `jpip_end_frame` always runs the IDWT on the full
   canvas at reduce_NL resolution.  At reduce=0, that's 42208×9870
   → ~3000ms in WASM.  The server correctly sends only viewport
   precincts (roff/rsiz), but the decoder ignores this and
   processes every row.

Both problems have the same root cause: **the decoder cannot output
a spatial sub-region of the canvas.**  It always decodes the full
canvas and downsamples to the output buffer.

## The fix: viewport-region decode

The viewer needs a WASM API that:

1. Takes a viewport rectangle (ox, oy, w, h) in canvas coordinates
2. Decodes ONLY the precincts that contribute to that rectangle
3. Outputs a viewport-sized (w × h) RGBA buffer
4. Runs the IDWT only on the rows within the rectangle

### Why this is hard

The line-based IDWT is sequential: row N depends on rows N-1 and
N+1 via the lifting filter.  You cannot jump to row oy without
processing rows 0 through oy-1.

However, with the precinct filter (Phase 4A), rows outside the
viewport region have absent precincts → zero subband data → the
IDWT zero-skip optimization makes those rows nearly free (~10% of
full cost).  The remaining cost is the structural overhead of
iterating rows 0 to oy-1 even though they produce zeros.

### Approach: early-start + early-exit

```
Row 0          ← zero-skip (absent precincts, ~free)
...            ← zero-skip
Row oy         ← START writing to output buffer
...            ← full IDWT (viewport data)
Row oy+h       ← STOP (early exit, skip all remaining rows)
...
Row H          ← never reached
```

**Early exit** (rows after oy+h): skip entirely.  `idwt_2d_state_free`
is safe with partial pulls — verified in the Phase 4B analysis.  Saves
~(H - oy - h) / H of the total cost.

**Zero-skip** (rows before oy): Phase 4A's `adv_step` already skips
lifting when both neighbor rows are zero.  Cost is ~10% per row
(counter increments + zero checks, no SIMD arithmetic).

**Combined savings** for a centered viewport on 42K image:

| Region | Rows | Cost per row | Total |
|---|---|---|---|
| Before viewport (zero-skip) | ~4000 | ~10% | 400 |
| Viewport | ~1000 | 100% | 1000 |
| After viewport (early exit) | ~5000 | 0% | 0 |
| **Total** | 10000 | | **1400** (vs 10000 full) |

**Speedup: ~7x** (from ~3000ms to ~420ms in WASM for reduce=0).

With reduce=1 (half resolution): canvas is 21104×4935.  The same
approach gives ~210ms → acceptable for interactive browsing.

### Implementation plan

#### V2-1 · `jpip_end_frame_region` WASM API

New function in `jpip_wrapper.cpp`:

```c
int jpip_end_frame_region(void *handle, uint8_t *rgb_out,
                          int out_w, int out_h,
                          int region_x, int region_y,
                          int region_w, int region_h);
```

- Sets `reduce_NL` from `ctx->reduce_NL` (as now)
- Sets precinct filter: only precincts overlapping
  `[region_x, region_y, region_w, region_h]` at the reduced level
- Calls `invoke_line_based_stream` with a row callback that:
  - Skips output for rows < region_y
  - Writes rows [region_y, region_y+region_h) to rgb_out
  - After row region_y+region_h: no more writes (but rows still
    iterate until early-exit is added)
- Output buffer is `out_w × out_h` pixels (viewport-sized)

This gives the **correct output** (no blur) without changing the
decoder core.  The decode is still slow (iterates all rows) but
the output quality is perfect.

#### V2-2 · Early-exit row limit (re-implement Phase 4B)

Add `row_limit` parameter to `decode_line_based_stream`:

```cpp
void decode_line_based_stream(
    j2k_main_header &hdr, uint8_t reduce_NL,
    const std::function<...> &cb,
    uint32_t row_limit = UINT32_MAX);
```

When `row_limit < H`, the strip loop and Phase 1 pulls stop at
`row_limit`.  The IDWT state is freed via `finalize_line_decode`
which handles partial pulls safely.

`jpip_end_frame_region` sets `row_limit = region_y + region_h`.

**This is the same change we reverted earlier**, but now it's used
inside `jpip_end_frame_region` — the output buffer contains the
correct viewport data (no black regions) because the row callback
only writes rows within the viewport.

#### V2-3 · Wire into viewer JavaScript

Update `jpip_viewer.html` to call `jpip_end_frame_region` instead
of `jpip_end_frame`:

```javascript
const rc = M._jpip_end_frame_region(ctx, rgbPtr, vpW, vpH,
    Math.round(panX), Math.round(panY),
    Math.round(vpW / zoom), Math.round(vpH / zoom));
```

Remove the maxDim cap — the output is always viewport-sized.
Remove the reduce-level-dependent fetch mode — always fetch the
viewport region via roff/rsiz.

#### V2-4 · Precinct-to-region filter

The precinct filter for the viewport region: given (region_x,
region_y, region_w, region_h) in canvas coordinates, determine
which precincts at the active resolution level overlap.

This is already solved by `resolve_view_window` — it maps a
ViewWindow (fsiz/roff/rsiz) to a set of precinct keys.  The
`jpip_end_frame_region` can build a ViewWindow from the region
parameters and call `resolve_view_window` on the client-side
CodestreamIndex.

Alternatively, just use the server's response: the server already
sends only viewport precincts via roff/rsiz.  The client doesn't
need its own filter — absent precincts naturally produce zeros,
and the IDWT zero-skip handles them.

### Expected result

| Zoom | Reduce | Decode (before) | Decode (after) | Quality |
|---|---|---|---|---|
| 5% | 4 | 50ms | 50ms (unchanged) | Good |
| 10% | 3 | 100ms | 80ms | Good |
| 25% | 2 | 200ms | 60ms | Sharp |
| 50% | 0 | 3000ms | 420ms | **Sharp** (was blurry) |
| 100% | 0 | 3000ms | 250ms | **Pixel-perfect** |

### Risk

The IDWT zero-skip for rows before the viewport depends on ALL
precincts outside the viewport being absent.  If the server sends
extra precincts (e.g., overlapping the viewport boundary), some
pre-viewport rows may have data, reducing the zero-skip benefit.
This is a minor performance issue, not a correctness issue.

## V3 · Smooth transitions and caching (future)

- Cache decoded tiles by (reduce, region) key
- On pan: display cached tile immediately, refetch in background
- On zoom: show cached low-res scaled up while high-res loads
- Progressive: coarse response first, refine with follow-up

## V4 · Native viewer (future)

- GLFW-based pan+zoom with the same region-decode API
- Native decode is ~5-10x faster than WASM → sub-50ms at any zoom
