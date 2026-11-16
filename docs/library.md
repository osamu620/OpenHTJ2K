# Using the library

The `openhtj2k` library decodes and encodes JPEG 2000 Part 1 and HTJ2K from
C++11 or later. The API is the classes `open_htj2k::openhtj2k_decoder` and
`open_htj2k::openhtj2k_encoder`, declared in `decoder.hpp` and `encoder.hpp`.

## Linking

After `cmake --install` (see [building.md](building.md#installing-and-using-the-library)):

```cmake
find_package(openhtj2k CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE openhtj2k::openhtj2k)
```

or, without CMake, `c++ my_app.cpp $(pkg-config --cflags --libs open_htj2k)`.

## Decoding

```cpp
#include <open_htj2k/interface/decoder.hpp>
#include <algorithm>
#include <vector>

// Decodes a codestream, JPH or JP2 file into one plane per component.
std::vector<std::vector<int32_t>> decode(const char *path, std::vector<uint32_t> &width,
                                         std::vector<uint32_t> &height, std::vector<uint8_t> &depth) {
  open_htj2k::openhtj2k_decoder dec(path, 0, 0);  // full resolution, all cores
  dec.parse();
  std::vector<bool> is_signed;
  std::vector<std::vector<int32_t>> planes;
  dec.invoke_line_based_stream(
      [&](uint32_t y, int32_t *const *rows, uint16_t nc) {
        if (planes.empty())
          for (uint16_t c = 0; c < nc; ++c) planes.emplace_back(size_t(width[c]) * height[c]);
        for (uint16_t c = 0; c < nc; ++c) {
          const uint32_t r = dec.get_stream_component_row(c, y);  // UINT32_MAX: none here
          if (r != UINT32_MAX)
            std::copy(rows[c], rows[c] + width[c], planes[c].begin() + size_t(r) * width[c]);
        }
      },
      width, height, depth, is_signed);
  return planes;
}
```

- The constructor takes a file name, or a buffer and its size. Its other
  arguments are the number of resolution levels to drop (1 gives half the
  width and height) and the number of threads (see [Threads](#threads)).
- To decode one codestream after another with the same decoder (video, for
  example), call `dec.init(buf, size, reduce_NL)` and then `parse()` again.
  It keeps the decoder's thread count. Passing 1 as a fourth argument would
  decode on the calling thread only.
- A YCbCr image coded with the colour transform comes out as RGB.
- `get_stream_component_row()` handles subsampled components (4:2:2, 4:2:0).
  Without subsampling, `rows[c]` is simply row `y` of component `c`.

Other ways to get the samples:

| Call | You get | For |
|---|---|---|
| `invoke_line_based_stream()` | one row at a time, through a callback (above) | most uses; needs memory for a few rows only |
| `invoke_line_based()` | whole planes that the decoder allocates; free each with `delete[]` | simple whole-image access |
| `invoke_line_based_direct()` | 8- or 16-bit samples written into your own plane buffers (`PlanarOutputDesc`) | display and video pipelines |
| `enable_single_tile_reuse(true)`, then `invoke_line_based_stream_reuse()` | as the stream call, keeping the decoder's state from frame to frame | a sequence of single-tile frames (video) |

## Encoding

```cpp
#include <open_htj2k/interface/encoder.hpp>
#include <vector>

// Encodes three 8-bit planes (R, G, B) of w x h samples as HTJ2K at Qfactor 90.
std::vector<uint8_t> encode_rgb8(std::vector<std::vector<int32_t>> &rgb, uint32_t w, uint32_t h) {
  open_htj2k::siz_params siz{};
  siz.Xsiz = siz.XTsiz = w;  // one tile covering the image
  siz.Ysiz = siz.YTsiz = h;
  siz.Csiz = 3;
  siz.Ssiz = {7, 7, 7};  // bit depth - 1, plus 0x80 for signed samples
  siz.XRsiz = siz.YRsiz = {1, 1, 1};
  open_htj2k::cod_params cod{};
  cod.blkwidth = cod.blkheight = 4;  // 64 x 64 code-blocks: log2(64) - 2
  cod.is_max_precincts = true;
  cod.number_of_layers = 1;
  cod.use_color_trafo = 1;     // RGB to YCbCr
  cod.dwt_levels = 5;
  cod.codeblock_style = 0x40;  // HT block coder; 0 for the Part 1 block coder
  cod.transformation = 0;      // 0: 9/7 wavelet (lossy); 1: 5/3 wavelet (lossless)
  open_htj2k::qcd_params qcd{};
  qcd.number_of_guardbits = 1;
  qcd.base_step = 1.0 / 256;   // the step size when there is no Qfactor
  std::vector<int32_t *> planes = {rgb[0].data(), rgb[1].data(), rgb[2].data()};
  std::vector<uint8_t> out;
  open_htj2k::openhtj2k_encoder enc("", planes, siz, cod, qcd, 90 /* Qfactor; 0xFF: none */,
                                    false /* not JPH */, 0 /* RGB */, 0 /* all cores */);
  enc.set_output_buffer(out);  // without it, the encoder writes the file named in place of ""
  enc.invoke_line_based();
  return out;
}
```

- **Lossless:** `cod.transformation = 1` and Qfactor `0xFF`.
- **Part 1** (readable by any JPEG 2000 decoder): `cod.codeblock_style = 0`.
- **File format:** `set_file_format(open_htj2k::file_format::jph)` for HTJ2K,
  `file_format::jp2` for Part 1.
- **Quality settings:** `set_qfactor_weights(1)` gives flat weights for
  scientific and other non-visual data. `set_qfactor_scale(1)` scales the
  Qfactor with the sample precision, for data of more than 8 bits. See
  [qfactor.md](qfactor.md).
- **Target size** (experimental): `set_target_size(bytes)`; see
  [rate control](cli_encoder.md#rate-control).
- **Rows from a callback** instead of whole planes: pass an empty `planes`
  vector and call `invoke_line_based_stream()` with a function that fills
  `rows[c][x]` for row `y`.

The comments in `encoder.hpp` describe every field and setting.

## Threads

The last constructor argument is the number of threads. 1 works on the
calling thread only. 0 uses one thread per hardware thread, and `n` uses `n`.
All decoders and encoders in a process share one worker pool. The first
instance that needs it creates it with its own thread count.

You can use several instances from several threads at once, but each instance
from only one thread at a time.

## Errors and warnings

- **Errors are exceptions** derived from `std::exception`, and `what()`
  describes the problem.
- **The library never prints anything.** Warnings and informational messages
  go to a handler you install, and without one they are discarded:

  ```cpp
  open_htj2k::set_message_handler(
      [](open_htj2k::message_level level, const char *text, void *) {
        std::fprintf(stderr, "%s: %s\n",
                     level == open_htj2k::message_level::warning ? "warning" : "info", text);
      },
      nullptr);
  ```

  The handler can be called from the library's worker threads, several at
  once, so it must be thread-safe.
- **A truncated codestream** (a cut-off file or download) still decodes from
  the data it holds. A warning reports the missing data, and `is_truncated()`
  returns true after the decode.
