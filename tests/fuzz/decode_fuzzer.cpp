// Fuzz target: the public decoder API (decoder.hpp) on an untrusted codestream or JPH/JP2
// file.
//
// Input = the file, then 8 option bytes (fuzz_common.hpp says why they come last):
//   o0 & 3   entry point: 0 invoke_line_based, 1 invoke_line_based_stream,
//            2 invoke_line_based_stream_reuse, 3 invoke_line_based_direct
//            (2 and 3 with the single-tile reuse enabled)
//   o1       reduce_NL: 0, or (o1 & 7) when bit 6 is set, or (o1 & 0x3f) when bit 7 is set
//   o2       bit0 4 threads, bit1 init_borrow, bit2 skip_mct, bit3 row range,
//            bit4 column range, bit5 precinct filter, bit6 packet observer, bit7 row limit
//   o3 & 3   a second frame on the same decoder: 0 none, 1 the same bytes, 2 one byte
//            changed, 3 cut short; after 2 and 3 the first frame is decoded once more
//   o4 o5    where the change or the cut is (scaled to the length)
//   o6       xor value of the change
//   o7       seed of the row and column ranges and of the shape of the direct planes
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <vector>
#include "decoder.hpp"
#include "fuzz_common.hpp"

using namespace open_htj2k;

namespace {
const size_t kOptions = 8;

struct Options {
  uint8_t entry, reduce, flags, second, seed, xorv;
  uint16_t pos;
};

// One frame through the chosen entry point.  Throws what the library throws.
void decode_frame(openhtj2k_decoder &dec, const uint8_t *cs, size_t len, const Options &o) {
  const uint32_t threads = (o.flags & 1) ? 4 : 1;
  bool col_window        = false;  // a column window is set (o.flags & 0x10)
  uint32_t col_lo        = 0;
  std::unique_ptr<uint8_t[]> padded;
  if (o.flags & 2) {
    padded.reset(new uint8_t[len + 16]);
    std::memcpy(padded.get(), cs, len);
    std::memset(padded.get() + len, 0, 16);
    dec.init_borrow(padded.get(), len, o.reduce, threads);
  } else {
    dec.init(cs, len, o.reduce, threads);
  }
  dec.parse();
  const uint16_t nc = dec.get_num_component();
  uint64_t samples  = 0;
  for (uint16_t i = 0; i < nc; ++i) {
    samples += static_cast<uint64_t>(dec.get_component_width(i)) * dec.get_component_height(i);
  }
  fuzz::use(static_cast<uint64_t>(dec.get_minumum_DWT_levels()) + dec.get_max_safe_reduce_NL()
            + dec.get_colorspace() + dec.get_mct());
  if (nc == 0 || nc > fuzz::kMaxComponents || samples > fuzz::kMaxSamples) return;

  dec.set_skip_mct((o.flags & 4) != 0);
  const uint32_t rows = dec.get_stream_row_count();
  if (o.flags & 8) {
    const uint32_t lo = rows ? (o.seed * 7u) % (rows + 1) : 0;
    dec.set_row_range(lo, lo + 1 + (o.seed >> 2));
  } else {
    dec.set_row_range(0, UINT32_MAX);
  }
  if (o.flags & 0x80) dec.set_row_limit(1u + o.seed);
  if (o.flags & 0x10) {
    const uint32_t w  = dec.get_component_width(0);
    const uint32_t lo = w ? (o.seed * 13u) % (w + 1) : 0;
    dec.set_col_range(lo, lo + 1 + (o.seed >> 1));
    col_window = true;
    col_lo     = lo;
  } else {
    dec.set_col_range(0, UINT32_MAX);
  }
  if (o.flags & 0x20) {
    const uint8_t s = o.seed;
    dec.set_precinct_filter([s](uint16_t t, uint16_t c, uint8_t r, uint32_t p) {
      return ((t * 31u + c * 17u + r * 7u + p + s) % 3u) != 0;
    });
  } else {
    dec.set_precinct_filter(std::function<bool(uint16_t, uint16_t, uint8_t, uint32_t)>());
  }
  if (o.flags & 0x40) {
    dec.set_packet_observer([](uint16_t, uint16_t, uint8_t, uint32_t, uint16_t, uint64_t offset,
                               uint64_t length) { fuzz::use(offset + length); });
  } else {
    dec.set_packet_observer(
        std::function<void(uint16_t, uint16_t, uint8_t, uint32_t, uint16_t, uint64_t, uint64_t)>());
  }

  std::vector<uint32_t> w, h;
  std::vector<uint8_t> depth;
  std::vector<bool> sgn;
  switch (o.entry) {
    case 0: {
      std::vector<int32_t *> buf;
      struct Free {
        std::vector<int32_t *> &b;
        ~Free() {
          for (int32_t *p : b) delete[] p;
        }
      } guard{buf};
      dec.invoke_line_based(buf, w, h, depth, sgn);
      uint64_t acc = 0;
      for (size_t i = 0; i < buf.size() && i < w.size(); ++i) {
        const size_t n = static_cast<size_t>(w[i]) * h[i];
        if (buf[i] == nullptr) continue;
        for (size_t k = 0; k < n; k += 1 + n / 4096) acc += static_cast<uint32_t>(buf[i][k]);
        if (n) acc += static_cast<uint32_t>(buf[i][n - 1]);
      }
      fuzz::use(acc);
      break;
    }
    case 1:
    case 2: {
      uint64_t acc = 0;
      auto on_row  = [&](uint32_t y, int32_t *const *row, uint16_t n) {
        for (size_t i = 0; i < n && i < w.size(); ++i) {
          if (row[i] == nullptr || w[i] == 0) continue;
          // Outside a column window the samples are indeterminate (set_col_range()): read
          // one column inside it, scaled to this component's width.
          uint32_t x0 = 0, x1 = w[i] - 1;
          if (col_window) {
            const uint64_t xs = static_cast<uint64_t>(col_lo) * w[i] / std::max(w[0], 1u);
            x0                = static_cast<uint32_t>(std::min<uint64_t>(xs, w[i] - 1));
            x1 = x0;
          }
          acc += static_cast<uint32_t>(row[i][x0]) + static_cast<uint32_t>(row[i][x1]) + y;
          acc += dec.get_stream_component_row(static_cast<uint16_t>(i), y);
        }
      };
      if (o.entry == 2) {
        dec.invoke_line_based_stream_reuse(on_row, w, h, depth, sgn);
      } else {
        dec.invoke_line_based_stream(on_row, w, h, depth, sgn);
      }
      fuzz::use(acc);
      break;
    }
    default: {
      // Planes with a shape of their own (a client with fixed buffers): at most the component's.
      std::vector<std::vector<uint16_t>> p16(nc);
      std::vector<std::vector<uint8_t>> p8(nc);
      std::vector<PlanarOutputDesc> descs(nc);
      for (uint16_t i = 0; i < nc; ++i) {
        uint32_t cw     = dec.get_component_width(i);
        uint32_t ch     = dec.get_component_height(i);
        const int d     = std::min<int>(std::max<int>(dec.get_component_depth(i), 1), 16);
        const bool is16 = ((o.seed >> (i & 3)) & 1) != 0;
        if (o.seed & 0x80) {  // smaller than the component
          cw = std::min<uint32_t>(cw, 1u + (o.seed & 0x3fu));
          ch = std::min<uint32_t>(ch, 1u + ((o.seed * 5u) & 0x3fu));
        }
        PlanarOutputDesc &pd = descs[i];
        pd.stride            = cw;
        pd.width             = cw;
        pd.height            = ch;
        pd.yr                = 1;
        pd.dc                = 1 << (d - 1);
        pd.maxval            = is16 ? (1 << d) - 1 : 255;
        pd.minval            = 0;
        pd.is_16bit          = is16;
        pd.depth_shift       = is16 ? 0 : std::max(0, d - 8);
        const size_t n       = static_cast<size_t>(cw) * ch;
        if (is16) {
          p16[i].assign(n, 0);
          pd.base = p16[i].data();
        } else {
          p8[i].assign(n, 0);
          pd.base = p8[i].data();
        }
      }
      dec.invoke_line_based_direct(descs.data(), nc, w, h, depth, sgn);
      break;
    }
  }
  fuzz::use(dec.is_truncated());
}
}  // namespace

extern "C" int LLVMFuzzerInitialize(int *, char ***) {
  set_message_handler(fuzz::quiet, nullptr);
  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size < kOptions + 2) return 0;
  const size_t len   = size - kOptions;
  const uint8_t *opt = data + len;
  Options o;
  o.entry  = opt[0] & 3;
  o.reduce = static_cast<uint8_t>((opt[1] & 0x80) ? (opt[1] & 0x3f) : ((opt[1] & 0x40) ? (opt[1] & 7) : 0));
  o.flags  = opt[2];
  o.second = opt[3] & 3;
  o.pos    = static_cast<uint16_t>((opt[4] << 8) | opt[5]);
  o.xorv   = opt[6];
  o.seed   = opt[7];
  if (fuzz::too_many_tiles(data, len)) return 0;

  openhtj2k_decoder dec;
  if (o.entry >= 2) dec.enable_single_tile_reuse(true);
  try {
    decode_frame(dec, data, len, o);
  } catch (std::exception &) {
  }
  if (o.second == 0) return 0;
  std::vector<uint8_t> second(data, data + len);
  if (o.second == 2) {
    second[(static_cast<size_t>(o.pos) * len) >> 16] ^= static_cast<uint8_t>(o.xorv | 1);
  } else if (o.second == 3) {
    second.resize(std::max<size_t>(2, (static_cast<size_t>(o.pos) * len) >> 16));
  }
  try {
    decode_frame(dec, second.data(), second.size(), o);
  } catch (std::exception &) {
  }
  if (o.second >= 2) {
    try {
      decode_frame(dec, data, len, o);
    } catch (std::exception &) {
    }
  }
  return 0;
}
