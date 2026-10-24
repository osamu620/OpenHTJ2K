// Built by tests/install_consumer/run.cmake against an *installed* OpenHTJ2K:
// only the installed headers, the installed library and the flags from
// `pkg-config --cflags --libs open_htj2k` are available, as for any external
// project.  It is compiled as C++11.
//
// Decodes the given codestream, losslessly re-encodes its first component into
// memory, decodes that back and checks the samples round-trip exactly.
#include <cstdint>
#include <cstdio>
#include <exception>
#include <vector>
#include "decoder.hpp"
#include "encoder.hpp"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("usage: %s <input.j2k>\n", argv[0]);
    return 2;
  }
  try {
    std::vector<int32_t *> buf;
    std::vector<uint32_t> w, h;
    std::vector<uint8_t> depth;
    std::vector<bool> is_signed;
    open_htj2k::openhtj2k_decoder dec(argv[1], 0, 1);
    dec.parse();
    dec.invoke_line_based(buf, w, h, depth, is_signed);
    if (buf.empty()) return 1;

    open_htj2k::siz_params siz;
    siz.Rsiz   = 0;
    siz.Xsiz   = w[0];
    siz.Ysiz   = h[0];
    siz.XOsiz  = 0;
    siz.YOsiz  = 0;
    siz.XTsiz  = w[0];
    siz.YTsiz  = h[0];
    siz.XTOsiz = 0;
    siz.YTOsiz = 0;
    siz.Csiz   = 1;
    siz.Ssiz.push_back(static_cast<uint8_t>((depth[0] - 1) | (is_signed[0] ? 0x80 : 0)));
    siz.XRsiz.push_back(1);
    siz.YRsiz.push_back(1);
    open_htj2k::cod_params cod;
    cod.blkwidth          = 4;  // 64x64 code-blocks (log2 - 2)
    cod.blkheight         = 4;
    cod.is_max_precincts  = true;
    cod.use_SOP           = false;
    cod.use_EPH           = false;
    cod.progression_order = 0;
    cod.number_of_layers  = 1;
    cod.use_color_trafo   = 0;
    cod.dwt_levels        = 3;
    cod.codeblock_style   = 0x040;  // HT
    cod.transformation    = 1;      // reversible 5/3
    open_htj2k::qcd_params qcd;
    qcd.is_derived          = false;
    qcd.number_of_guardbits = 1;
    qcd.base_step           = 1.0 / static_cast<double>(1 << depth[0]);

    std::vector<int32_t *> no_input;  // rows come from the callback below
    std::vector<uint8_t> j2c;
    {
      open_htj2k::openhtj2k_encoder enc("", no_input, siz, cod, qcd, 0xFF, false, 0, 1);
      enc.set_output_buffer(j2c);
      const int32_t *src  = buf[0];
      const uint32_t cols = w[0];
      enc.invoke_line_based_stream([src, cols](uint32_t y, int32_t **rows, uint16_t) {
        for (uint32_t x = 0; x < cols; ++x) rows[0][x] = src[static_cast<size_t>(y) * cols + x];
      });
    }

    std::vector<int32_t *> buf2;
    std::vector<uint32_t> w2, h2;
    std::vector<uint8_t> depth2;
    std::vector<bool> is_signed2;
    open_htj2k::openhtj2k_decoder dec2(j2c.data(), j2c.size(), 0, 1);
    dec2.parse();
    dec2.invoke_line_based(buf2, w2, h2, depth2, is_signed2);
    bool same = buf2.size() == 1 && w2[0] == w[0] && h2[0] == h[0];
    for (size_t i = 0; same && i < static_cast<size_t>(w[0]) * h[0]; ++i) same = buf[0][i] == buf2[0][i];

    printf("install_consumer: %ux%u, %u-bit, %zu-byte lossless re-encode: %s\n", w[0], h[0],
           static_cast<unsigned>(depth[0]), j2c.size(), same ? "round-trip exact" : "MISMATCH");
    for (auto p : buf) delete[] p;
    for (auto p : buf2) delete[] p;
    return same ? 0 : 1;
  } catch (std::exception &e) {
    printf("install_consumer: ERROR: %s\n", e.what());
    return 1;
  }
}
