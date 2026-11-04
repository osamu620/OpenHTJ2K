// Copyright (c) 2019 - 2026, Osamu Watanabe
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
//    modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
//    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// qfactor_guideline_check: checks the quantization steps of a Qfactor codestream
// against "Controlling JPEG 2000 image quality using a single parameter (Qfactor)
// v2.0" (ISO/IEC JTC 1/SC 29/WG 1 N101553), computed here from the guideline alone:
// its 9/7 synthesis filters, Tables 2-4 and the BT.601 YCbCr-to-RGB matrix. It
// shares no code with the encoder (visual_weighting.hpp), so it catches a wrong
// table in either place.
//
//   qfactor_guideline_check <codestream.j2c> <Qfactor>
//
// The codestream must come from OpenHTJ2K's default (legacy) Qfactor weighting on
// gray, gray + alpha, YCbCr (4:4:4 through the ICT, or pre-transformed 4:2:0 /
// 4:2:2) or YCbCr + alpha input. Each signalled step must be within 1/2048 of the
// guideline's (one mantissa step, which covers rounding and the encoder's 4-digit
// colour gains); a step the codestream cannot represent (exponent < 0) must be
// clamped to the coarsest one, (0, 2047). Exit status 0 = all steps match.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

struct Header {
  uint16_t Csiz = 0;
  std::vector<int> depth, xr, yr;
  int levels = -1;
  std::vector<uint16_t> qcd;
  std::vector<std::vector<uint16_t>> qcc;  // per component, empty if absent
};

bool read_file(const char *path, std::vector<uint8_t> &out) {
  FILE *f = std::fopen(path, "rb");
  if (!f) return false;
  uint8_t buf[65536];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) out.insert(out.end(), buf, buf + n);
  std::fclose(f);
  return true;
}

uint16_t be16(const uint8_t *p) { return static_cast<uint16_t>((p[0] << 8) | p[1]); }

std::vector<uint16_t> step_words(const uint8_t *p, size_t n) {
  std::vector<uint16_t> w;
  for (size_t i = 0; i + 1 < n; i += 2) w.push_back(be16(p + i));
  return w;
}

bool parse(const std::vector<uint8_t> &b, Header &h) {
  if (b.size() < 4 || be16(&b[0]) != 0xFF4F) return false;
  size_t pos = 2;
  while (pos + 4 <= b.size()) {
    const uint16_t marker = be16(&b[pos]);
    if (marker == 0xFF90) break;  // first SOT: end of the main header
    const uint16_t L = be16(&b[pos + 2]);
    if (L < 2 || pos + 2 + L > b.size()) return false;
    const uint8_t *seg = &b[pos + 4];
    const size_t n     = L - 2u;
    if (marker == 0xFF51 && n >= 36) {  // SIZ
      h.Csiz = be16(seg + 34);
      if (n < 36u + 3u * h.Csiz) return false;
      for (uint16_t c = 0; c < h.Csiz; ++c) {
        h.depth.push_back((seg[36 + 3 * c] & 0x7F) + 1);
        h.xr.push_back(seg[37 + 3 * c]);
        h.yr.push_back(seg[38 + 3 * c]);
      }
      h.qcc.resize(h.Csiz);
    } else if (marker == 0xFF52 && n >= 6) {  // COD: Scod, SGcod (4 bytes), levels
      h.levels = seg[5];
    } else if (marker == 0xFF5C && n >= 1) {   // QCD: Sqcd, SPqcd
      if ((seg[0] & 0x1F) != 2) return false;  // Qfactor writes expounded steps
      h.qcd = step_words(seg + 1, n - 1);
    } else if (marker == 0xFF5D && h.Csiz > 0) {  // QCC: Cqcc (1 or 2 bytes), Sqcc, SPqcc
      const size_t ci     = (h.Csiz < 257) ? 1 : 2;
      const uint16_t comp = (ci == 1) ? seg[0] : be16(seg);
      if (n < ci + 1 || comp >= h.Csiz || (seg[ci] & 0x1F) != 2) return false;
      h.qcc[comp] = step_words(seg + ci + 1, n - ci - 1);
    }
    pos += 2u + L;
  }
  return h.Csiz > 0 && h.levels >= 0 && !h.qcd.empty();
}

// Squared norms of the level-n 9/7 synthesis basis vectors (WG1N101553, "Computing G_b").
void basis_norms(int N, std::vector<double> &nl, std::vector<double> &nh) {
  const double gL[7] = {-0.091271763114250, -0.057543526228500, 0.591271763114250, 1.115087052457000,
                        0.591271763114250,  -0.057543526228500, -0.091271763114250};
  const double gH[9] = {0.026748757410811,  0.016864118442875, -0.078223266528990,
                        -0.266864118442875, 0.602949018236360, -0.266864118442875,
                        -0.078223266528990, 0.016864118442875, 0.026748757410811};
  std::vector<double> L(gL, gL + 7), H(gH, gH + 9);
  for (int n = 1; n <= N; ++n) {
    if (n > 1) {  // g^(n) = g_L^(1) o (g^(n-1) upsampled by 2)
      std::vector<double> l2(7 + 2 * L.size() - 2, 0.0), h2(7 + 2 * H.size() - 2, 0.0);
      for (size_t i = 0; i < 7; ++i) {
        for (size_t j = 0; j < L.size(); ++j) l2[i + 2 * j] += gL[i] * L[j];
        for (size_t j = 0; j < H.size(); ++j) h2[i + 2 * j] += gL[i] * H[j];
      }
      L.swap(l2);
      H.swap(h2);
    }
    double sl = 0, sh = 0;
    for (size_t i = 0; i < L.size(); ++i) sl += L[i] * L[i];
    for (size_t i = 0; i < H.size(); ++i) sh += H[i] * H[i];
    nl.push_back(sl);
    nh.push_back(sh);
  }
}

// sqrt(W_b) of WG1N101553 Tables 2-4, per level 1..5: {HH, LH, HL}, plus the
// weight of the "..." (levels past 5) and LLN columns.
struct WeightTable {
  double band[5][3];
  double tail;
};
const WeightTable kY444 = {
    {{0.0901, 0.2758, 0.2758}, {0.7018, 0.8378, 0.8378}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}}, 1.0};
const WeightTable kCb444 = {{{0.0263, 0.0863, 0.0863},
                             {0.1362, 0.2564, 0.2564},
                             {0.3346, 0.4691, 0.4691},
                             {0.5444, 0.6523, 0.6523},
                             {0.7078, 0.7797, 0.7797}},
                            1.0};
const WeightTable kCr444 = {{{0.0773, 0.1835, 0.1835},
                             {0.2598, 0.4130, 0.4130},
                             {0.5040, 0.6464, 0.6464},
                             {0.7220, 0.8254, 0.8254},
                             {0.8769, 0.9424, 0.9424}},
                            1.0};
const WeightTable kCb420 = {{{0.2724, 0.5128, 0.5128},
                             {0.6692, 0.9382, 0.9382},
                             {1.0888, 1.3046, 1.3046},
                             {1.4156, 1.5594, 1.5594},
                             {2.0, 2.0, 2.0}},
                            2.0};
const WeightTable kCr420 = {{{0.5196, 0.826, 0.826},
                             {1.008, 1.2928, 1.2928},
                             {1.444, 1.6508, 1.6508},
                             {1.7538, 1.8848, 1.8848},
                             {2.0, 2.0, 2.0}},
                            2.0};
const WeightTable kCb422 = {{{0.1220, 0.1220, 0.3626},
                             {0.3626, 0.3626, 0.6634},
                             {0.6634, 0.6634, 0.9225},
                             {0.9225, 0.9225, 1.1027},
                             {1.1027, 1.1027, 1.4142}},
                            1.4142};
const WeightTable kCr422 = {{{0.2595, 0.2595, 0.5841},
                             {0.5841, 0.5841, 0.9141},
                             {0.9141, 0.9141, 1.1673},
                             {1.1673, 1.1673, 1.3328},
                             {1.3328, 1.3328, 1.4142}},
                            1.4142};

// M_Q, the weight exponent and alpha_Q (WG1N101553, "Computing Delta_ref", "Computing W_b").
double M_of(int q) { return (q < 50) ? 50.0 / q : 2.0 * (1.0 - q / 100.0); }
double weight_power(int q) {
  if (q <= 65) return 1.0;
  if (q >= 97) return 0.0;
  return (std::log(M_of(q)) - std::log(M_of(97))) / (std::log(M_of(65)) - std::log(M_of(97)));
}
double alpha_of(int q) {
  if (q <= 65) return 0.04;
  if (q >= 97) return 0.10;
  return 0.10 * std::pow(0.04 / 0.10, weight_power(q));
}

struct Step {
  int eps;
  int mu;
};

// The guideline's step for every band of one component, in QCD/QCC order (LL, then
// HL, LH, HH from the coarsest level), as (epsilon, mu) of Equation E-3.
std::vector<Step> guideline_steps(int q, int RI, int N, const WeightTable &t, double Mc) {
  std::vector<double> nl, nh;
  basis_norms(N, nl, nh);
  const double d_ref = std::ldexp(alpha_of(q) * M_of(q), RI) + 1.0 / std::sqrt(2.0);
  const double p     = weight_power(q);
  struct Band {
    double Gb, sqrtW;
    int gain;
  };
  std::vector<Band> bands;
  const size_t top = static_cast<size_t>(N) - 1;
  bands.push_back({nl[top] * nl[top], t.tail, 0});
  for (size_t n = static_cast<size_t>(N); n >= 1; --n) {
    const double hh = (n <= 5) ? t.band[n - 1][0] : t.tail;
    const double lh = (n <= 5) ? t.band[n - 1][1] : t.tail;
    const double hl = (n <= 5) ? t.band[n - 1][2] : t.tail;
    bands.push_back({nl[n - 1] * nh[n - 1], hl, 1});
    bands.push_back({nl[n - 1] * nh[n - 1], lh, 1});
    bands.push_back({nh[n - 1] * nh[n - 1], hh, 2});
  }
  std::vector<Step> out;
  for (size_t i = 0; i < bands.size(); ++i) {
    const double Wb = std::pow(bands[i].sqrtW * bands[i].sqrtW, p);
    const double d  = d_ref / std::sqrt(Wb * bands[i].Gb * Mc);
    // Delta_b = 2^(R_b - eps) (1 + mu / 2^11), R_b = RI + log2(band gain).
    const double x = std::ldexp(d, -(RI + bands[i].gain));
    int e          = 0;
    while (std::ldexp(x, e) < 1.0 && e < 64) ++e;
    int mu = static_cast<int>(std::floor((std::ldexp(x, e) - 1.0) * 2048.0 + 0.5));
    if (mu >= 2048) {
      mu = 0;
      --e;
    }
    out.push_back({e, mu});
  }
  return out;
}

double step_value(int eps, int mu) { return std::ldexp(1.0 + mu / 2048.0, -eps); }

}  // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <codestream.j2c> <Qfactor>\n", argv[0]);
    return EXIT_FAILURE;
  }
  const int q = std::atoi(argv[2]);
  std::vector<uint8_t> bytes;
  Header h;
  if (q < 1 || q > 100 || !read_file(argv[1], bytes) || !parse(bytes, h) || h.levels < 1) {
    std::fprintf(stderr, "ERROR: cannot read a Qfactor codestream from %s (Qfactor %s)\n", argv[1],
                 argv[2]);
    return EXIT_FAILURE;
  }

  // Colour layout: components 1 and 2 are Cb and Cr when there are three or more.
  enum { YCC444, YCC420, YCC422 } fmt = YCC444;
  const bool colour                   = h.Csiz >= 3;
  if (colour) {
    if (h.xr[1] != h.xr[2] || h.yr[1] != h.yr[2] || h.xr[0] != 1 || h.yr[0] != 1) {
      std::fprintf(stderr, "ERROR: unsupported sub-sampling\n");
      return EXIT_FAILURE;
    }
    if (h.xr[1] == 2 && h.yr[1] == 2) {
      fmt = YCC420;
    } else if (h.xr[1] == 2 && h.yr[1] == 1) {
      fmt = YCC422;
    } else if (h.xr[1] != 1 || h.yr[1] != 1) {
      std::fprintf(stderr, "ERROR: unsupported sub-sampling %dx%d\n", h.xr[1], h.yr[1]);
      return EXIT_FAILURE;
    }
  }
  // G_c = squared norm of column c of the YCbCr-to-RGB matrix; M_c = G_c / G_Y.
  const double G_Y  = 3.0;
  const double G_Cb = 0.344136 * 0.344136 + 1.772 * 1.772;
  const double G_Cr = 1.402 * 1.402 + 0.714136 * 0.714136;

  int checked = 0, failed = 0;
  for (uint16_t c = 0; c < h.Csiz; ++c) {
    const bool chroma    = colour && (c == 1 || c == 2);
    const WeightTable *t = &kY444;  // luma, gray, and an alpha channel
    double Mc            = 1.0;
    const char *role     = (c == 0) ? "Y" : "A";
    if (chroma) {
      const bool cb = (c == 1);
      role          = cb ? "Cb" : "Cr";
      Mc            = (cb ? G_Cb : G_Cr) / G_Y;
      if (fmt == YCC420) {
        t = cb ? &kCb420 : &kCr420;
      } else if (fmt == YCC422) {
        t = cb ? &kCb422 : &kCr422;
      } else {
        t = cb ? &kCb444 : &kCr444;
      }
    }
    const std::vector<uint16_t> &words = (c > 0 && !h.qcc[c].empty()) ? h.qcc[c] : h.qcd;
    const std::vector<Step> ref        = guideline_steps(q, h.depth[c], h.levels, *t, Mc);
    if (words.size() != ref.size()) {
      std::printf("FAIL component %u: %zu step words, %zu bands\n", c, words.size(), ref.size());
      ++failed;
      continue;
    }
    for (size_t i = 0; i < ref.size(); ++i) {
      const int eps = words[i] >> 11, mu = words[i] & 0x7FF;
      bool ok;
      if (ref[i].eps < 0) {
        ok = (eps == 0 && mu == 2047);  // coarser than the codestream can signal
      } else {
        const double r = step_value(eps, mu) / step_value(ref[i].eps, ref[i].mu);
        ok             = std::fabs(r - 1.0) <= 1.0 / 2048.0 + 1e-12;
      }
      ++checked;
      if (!ok) {
        ++failed;
        std::printf("FAIL component %u (%s) band %zu: codestream (%d, %d), guideline (%d, %d)\n", c, role,
                    i, eps, mu, ref[i].eps, ref[i].mu);
      }
    }
  }
  const char *fmt_name = (fmt == YCC420) ? "4:2:0" : (fmt == YCC422) ? "4:2:2" : "4:4:4";
  std::printf("qfactor_guideline_check: %s, Q=%d, %d levels, %u components: %d of %d steps differ\n",
              fmt_name, q, h.levels, h.Csiz, failed, checked);
  return failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
