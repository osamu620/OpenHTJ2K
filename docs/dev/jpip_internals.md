# JPIP internals

How OpenHTJ2K's JPIP (ISO/IEC 15444-9) implementation works inside: for people
changing it or building on its library and WASM interfaces. Using the server,
the demos and the benchmark is covered in [jpip.md](../jpip.md).

## Core architecture

```
source/core/jpip/
  precinct_index.{hpp,cpp}       — (t, c, r, p_rc) → JPIP sequence number s, identifier I
  view_window.{hpp,cpp}          — §C.4 view-window → precinct set
  vbas.{hpp,cpp}                 — VBAS codec (§A.2.1)
  jpp_message.{hpp,cpp}          — JPP message headers (§A.2, Tables A.1, A.2)
  data_bin_emitter.{hpp,cpp}     — header + precinct data-bin emission
  packet_locator.{hpp,cpp}       — per-precinct byte ranges in the codestream
  codestream_walker.{hpp,cpp}    — one-time codestream layout pass
  jpp_parser.{hpp,cpp}           — wire stream → DataBinSet (one-shot
                                    `parse_jpp_stream` + resumable
                                    `StreamingJppParser` for chunked
                                    clients; they share a single
                                    try_decode_one_message helper)
  codestream_assembler.{hpp,cpp} — DataBinSet → sparse codestream
  cache_model.{hpp,cpp}          — §C.9 client cache model
  jpip_request.{hpp,cpp}         — query parser
  jpip_response.{hpp,cpp}        — HTTP framer; chunked + Content-Length
                                    writers + parsers
  jpip_client.{hpp,cpp}          — HTTP/1.1 client with incremental
                                    chunked-transfer state machine
                                    (fetch_streaming + progress callback)
  tcp_socket.{hpp,cpp}           — cross-platform TCP wrapper
  h3_server.{hpp,cpp}            — MsQuic + nghttp3 HTTP/3 server
  h3_client.{hpp,cpp}            — MsQuic + nghttp3 HTTP/3 client
```

### Data-bin classes

- 0 — precinct (JPP-stream)
- 1 — extended precinct (JPP-stream, has Aux)
- 2 — tile header
- 4 — tile (JPT-stream only; not emitted by this server)
- 5 — extended tile (JPT-stream only, has Aux; not emitted by this server)
- 6 — main header
- 8 — metadata

The End-of-Response (EOR) message is deliberately not a class. Per
§D.3 it sits outside Annex A and uses a special 0x00 identifier byte
rather than a Bin-ID VBAS; see the dedicated EOR handling below.

### JPIP sequence number / identifier

Per §A.3.2.1:

```
s = Σ_{r' < r} npw[r'] · nph[r']  +  p_rc
I = t + (c + s · num_components) · num_tiles
```

`I` is the in-class identifier used in every precinct data-bin
message header. `CodestreamIndex::I(t, c, r, p_rc)` computes it;
the client reassembler decomposes `I` back to `(t, c, r, p_rc)` by
inverting the formula and linear-scanning `s_offset[]`.

### Zero-skip IDWT

Absent precincts appear in the sparse codestream as one-byte empty
packet headers (`0x00`). Rows that entirely derive from absent
precincts carry zero subband data, and `idwt_2d_state::adv_step()`
short-circuits the lifting pass when both its neighbour rows are
zero. The cost drops to ~10 % of a populated row (the counter
increments and zero checks stay).

### Viewport-region decode

`jpip_end_frame_region(handle, rgb, w, h, x, y, rw, rh)`:

1. **Precinct filter** — the server already sends only the precincts
   that overlap the region; absent rows naturally zero-skip.
2. **Row limit** — `set_row_limit(ry1)` tells the decoder to stop
   iterating after the last row of the region (`ry1 = ceil((y+rh) /
   2^reduce)`). Rows before `ry0` still iterate but zero-skip.
3. **Column range** — `set_col_range(rx0, rx1)` makes the horizontal
   1D IDWT process only the columns inside the region. Samples
   outside the range aren't stored or lifted.

Result: a centred viewport on a 42K × 10K image decodes in ~400 ms
in WASM at `reduce=0` instead of ~3000 ms for the full canvas.

### Cache-model field (§C.9)

`CacheModel` tracks the data-bins a client holds — completely, or up
to a byte offset. Its `format()` method emits the §C.9 model field
body used as `&model=` — for example `Hm,Ht0,Ht1-5,M0,P7:987`. Range
compression (`Ht1-5` instead of `Ht1,Ht2,Ht3,Ht4,Ht5`) keeps the
query short on multi-tile images; partial holdings carry the §C.9.2
`:bytes` qualifier and are never range-compressed.

| Class | Prefix | Example |
|---|---|---|
| Main header (6) | `Hm` | `Hm` (id always 0) |
| Tile header (2) | `Ht` | `Ht0`, `Ht1-5` |
| Precinct (0) | `P` | `P7`, `P7:987` (first 987 bytes) |
| Metadata (8) | `M` | `M0` |

`parse()`/`apply()` accept the legacy `Hp` precinct prefix, the
`:bytes` partial qualifier (the server resumes delivery from that
offset), and `-`-prefixed subtractive statements (`-P5` — the client
discarded the bin; the server forgets it was sent). The same model
statements work in stateless requests and as session-mode updates.

A statement is a few bytes and can name any range of bins, and a
model takes memory for every bin it holds, so a server reads the
field within the limits of the image it serves: `parse()` and
`apply()` take a `CacheModel::Limits` (the precinct identifiers
below `CodestreamIndex::precinct_id_bound()`, the tile headers of
its tiles, the metadata-bins it has, and the number of bins of the
image), and `ChannelManager::apply_model()` passes it on.
Statements about other bins are ignored. Without limits a model
stops growing at 2^22 bins, and one call does at most 2^22 bin
operations.

The browser demos track only headers — precincts are intentionally
excluded so the foveation demo's periphery decays when the gaze
moves.

### WASM C API

Defined in `web/src/jpip_wrapper.cpp`, exported via
`-sEXPORTED_FUNCTIONS` in `web/CMakeLists.txt`:

| Function | Purpose |
|---|---|
| `jpip_create_context(jpp, len)` | parse initial JPP-stream, build `CodestreamIndex` |
| `jpip_get_canvas_{width,height}` | canvas dimensions |
| `jpip_get_num_components`, `jpip_get_total_precincts` | asset metadata |
| `jpip_set_reduce(n)` | discard N DWT levels |
| `jpip_get_response_buffer(ctx, min_size)` | grow-only staging pointer; avoid per-frame `_malloc` |
| `jpip_add_response(ctx, buf, len)` | one-shot: parse a complete JPP-stream buffer and merge into the `DataBinSet` |
| `jpip_feed_stream_begin(ctx)` | progressive path: reset the per-context streaming parser before the first chunk of a new response (v0.17.0) |
| `jpip_feed_stream(ctx, buf, len)` | feed the next HTTP chunk of a chunked response; incomplete messages are buffered internally until the next call supplies the rest |
| `jpip_feed_stream_end(ctx)` | finalize the response; returns 0 if the stream ended at a clean JPP message boundary, nonzero if a partial message was still pending |
| `jpip_track_precincts_in_cache(ctx, enabled)` | opt-in LRU precinct cache (v0.18.0) — when on, received precincts are tracked in the client cache model so the server skips redelivery on subsequent pans |
| `jpip_set_precinct_cache_budget(ctx, lo, hi)` | LRU cache budget in bytes (64-bit value split lo/hi for Emscripten i32 calling convention); default 64 MB |
| `jpip_get_precinct_cache_count(ctx)` | diagnostic: number of precinct bins currently in the LRU |
| `jpip_get_cache_model(ctx)` | C-string for `&model=` advertisement |
| `jpip_reset_cache(ctx)` | soft reset — drops precincts, keeps headers; also clears the LRU when precinct tracking is on |
| `jpip_end_frame(ctx, rgb, w, h)` | full-canvas decode |
| `jpip_end_frame_region(ctx, rgb, w, h, x, y, rw, rh)` | viewport-region decode |
| `jpip_destroy_context(ctx)` | release |

The browser demos use the `jpip_feed_stream*` trio (not
`jpip_add_response`) so they can feed HTTP chunks into the decoder
as `response.body.getReader()` yields them; `jpip_add_response` is
preserved for one-shot callers that have the full body in hand.
