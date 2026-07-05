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

// visual_weight_check: dump and self-check harness for the analytic per-subband
// visual (CSF) weighting in source/core/codestream/visual_weighting.hpp.
//
//   visual_weight_check check
//     Verifies the sub-sampling-aware frequency-mapping invariants and exits
//     non-zero on the first violation (used by ctest):
//       1. (1,1) golden pin  -- analytic luma and Cb/Cr 4:4:4 weights match
//          the values produced before the per-axis (sx, sy) generalization
//          to 1e-6 (regression guard for the "4:4:4 output never changes"
//          contract; tolerance only absorbs cross-toolchain libm/FMA ULPs).
//       2. Isotropic shift   -- a (2, 2) sub-sampled component's weights at
//          level l equal the (1, 1) weights at level l+1, bit-for-bit, for the
//          chroma CSF, the reuse-luma CSF, and the generic/luma path. This is
//          the classic "chroma is weighted like luma one level down" rule,
//          falling out of the frequency mapping with no manual level offset.
//       3. 4:2:2 anisotropy  -- with (sx, sy) = (2, 1), HL (horizontal detail,
//          the sub-sampled axis) always outweighs LH. No scalar per-subband
//          table can represent this; it is the signature that the per-axis
//          mapping is live.
//       4. Default role      -- without a ctype hint an unlabelled (no-MCT)
//          component takes the luminance CSF, the historical behavior.
//
//   visual_weight_check dump [--csf mannos|daly] [--ppd F] [--zoom F]
//                            [--levels N] [--ctype Y|Cb|Cr|generic]
//                            [--sx N] [--sy N] [--reuse-luma]
//                            [--legacy-format 444|420|422]
//     Prints the derived per-level [HH, LH, HL] weight table for one component
//     configuration (one command per T.802 Annex B comparison).

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include "visual_weighting.hpp"

namespace {

using namespace open_htj2k;

int failures = 0;

void expect(bool cond, const char *what) {
  if (!cond) {
    printf("FAIL: %s\n", what);
    ++failures;
  }
}

// Weight vectors for one role/sub-sampling configuration. Cb/Cr go through
// chroma_visual_weights with a no-MCT ctype hint (the sub-sampled-chroma
// configuration); Y/generic go through the luminance path.
std::vector<double> weights_for(uint8_t levels, const visual_weighting_params &vp, component_type role,
                                int sx, int sy) {
  if (role == component_type::Cb || role == component_type::Cr) {
    const int comp_index = (role == component_type::Cb) ? 1 : 2;
    return chroma_visual_weights(levels, vp, comp_index, 0, color_transform::none, role, sx, sy);
  }
  return luma_visual_weights(levels, vp, sx, sy);
}

// --- check 1: (1,1) golden pin ---------------------------------------------
// Captured from the pre-generalization implementation (mannos_sakrison model,
// ref_ppd = 72, zoom = 1, 5 levels). 17 significant digits round-trip doubles
// exactly, so == is a bit-identity comparison.
void check_golden_pin() {
  const double golden_luma[15] = {0.096817675716936266,
                                  0.30684673263740037,
                                  0.30684673263740037,
                                  0.60532817339313194,
                                  0.864324646011562,
                                  0.864324646011562,
                                  0.99043615641172789,
                                  1.0,
                                  1.0,
                                  1.0,
                                  1.0,
                                  1.0,
                                  1.0,
                                  1.0,
                                  1.0};
  const double golden_cb[15]   = {0.034960121305188455, 0.081551235507758238, 0.081551235507758238,
                                  0.15359475448367257,  0.24653448385473559,  0.24653448385473559,
                                  0.35113522529678459,  0.4573791611974069,   0.4573791611974069,
                                  0.55729121401399506,  0.64597480708719213,  0.64597480708719213,
                                  0.72135733443385142,  0.78339023685855358,  0.78339023685855358};
  const double golden_cr[15]   = {0.071702417473406138, 0.16019272426776643, 0.16019272426776643,
                                  0.28006527167126605,  0.41292177620501364, 0.41292177620501364,
                                  0.54080888621381895,  0.65234354609406364, 0.65234354609406364,
                                  0.74313662078190756,  0.813575745821562,   0.813575745821562,
                                  0.86642359141464398,  0.90515943597529802, 0.90515943597529802};

  visual_weighting_params vp;
  vp.model = csf_model::mannos_sakrison;

  // NOT a bitwise compare: the reference values were captured on one
  // toolchain, and the analytic weights go through libm exp/pow and
  // compiler-contracted (FMA) float expressions, which legitimately differ by
  // ULPs across compilers (clang vs gcc vs MSVC) and platforms. 1e-6 is far
  // below any real formula change but far above toolchain noise. Same-binary
  // bit-identity of the (sx, sy) generalization at (1, 1) holds by
  // construction (identical expressions); cross-run byte-identity of the
  // emitted markers is guarded by the qfest_* round-trip tests.
  auto near_golden = [](double a, double b) { return std::fabs(a - b) <= 1e-6; };

  const std::vector<double> luma = luma_visual_weights(5, vp);
  const std::vector<double> cb   = chroma_visual_weights(5, vp, 1, 0, color_transform::ict);
  const std::vector<double> cr   = chroma_visual_weights(5, vp, 2, 0, color_transform::ict);
  expect(luma.size() == 15 && cb.size() == 15 && cr.size() == 15, "golden pin: 15 weights per table");
  for (size_t i = 0; i < 15 && failures == 0; ++i) {
    expect(near_golden(luma[i], golden_luma[i]), "golden pin: analytic luma weight drifted at (1,1)");
    expect(near_golden(cb[i], golden_cb[i]), "golden pin: analytic Cb 4:4:4 weight drifted");
    expect(near_golden(cr[i], golden_cr[i]), "golden pin: analytic Cr 4:4:4 weight drifted");
  }
}

// --- check 2: isotropic one-level shift -------------------------------------
// (2,2) sub-sampling divides every subband's angular frequency by 2, which is
// identically a one-decomposition-level shift, so the weight tables must match
// bit-for-bit -- proof that the mapping subsumes the manual level offset.
void check_isotropic_shift() {
  const csf_model models[2]         = {csf_model::mannos_sakrison, csf_model::daly};
  const component_type roles[3]     = {component_type::Cb, component_type::Cr, component_type::generic};
  const bool reuse_luma_variants[2] = {false, true};
  for (int m = 0; m < 2; ++m) {
    for (int r = 0; r < 3; ++r) {
      for (int v = 0; v < 2; ++v) {
        visual_weighting_params vp;
        vp.model                      = models[m];
        vp.chroma_reuse_luma_csf      = reuse_luma_variants[v];
        const std::vector<double> w22 = weights_for(5, vp, roles[r], 2, 2);
        const std::vector<double> w11 = weights_for(6, vp, roles[r], 1, 1);
        expect(w22.size() == 15 && w11.size() == 18, "isotropic shift: table sizes");
        for (size_t i = 0; i < w22.size(); ++i) {
          expect(w22[i] == w11[i + 3],
                 "isotropic shift: (2,2) level l != (1,1) level l+1 (must be bit-identical)");
          if (failures) return;
        }
      }
    }
  }
}

// --- check 3: 4:2:2 anisotropy ----------------------------------------------
// Horizontal-only sub-sampling halves only the horizontal frequency, so HL
// (horizontal detail) must outweigh LH. The chroma CSF is strictly decreasing
// (strict >); the luminance CSF is flat below its peak, so the reuse-luma and
// generic variants assert >= everywhere and strict > at the finest level.
void check_422_anisotropy() {
  const csf_model models[2] = {csf_model::mannos_sakrison, csf_model::daly};
  for (int m = 0; m < 2; ++m) {
    visual_weighting_params vp;
    vp.model = models[m];
    for (int r = 1; r <= 3; ++r) {  // 1 = Y-like generic handled below; use Cb, Cr, generic
      const component_type role   = (r == 1)   ? component_type::Cb
                                    : (r == 2) ? component_type::Cr
                                               : component_type::generic;
      const bool strict_all       = (role == component_type::Cb || role == component_type::Cr);
      const std::vector<double> w = weights_for(5, vp, role, 2, 1);
      for (size_t lvl = 0; lvl < 5; ++lvl) {
        const double w_lh = w[3 * lvl + 1];
        const double w_hl = w[3 * lvl + 2];
        if (strict_all) {
          expect(w_hl > w_lh, "4:2:2 anisotropy: chroma-CSF w_HL not strictly > w_LH");
        } else {
          expect(w_hl >= w_lh, "4:2:2 anisotropy: luminance-CSF w_HL < w_LH");
          if (lvl == 0) expect(w_hl > w_lh, "4:2:2 anisotropy: no split at the finest level");
        }
      }
    }
  }
}

// --- check 4: default role --------------------------------------------------
// Unhinted no-MCT components keep the historical luminance-CSF treatment.
void check_default_role() {
  visual_weighting_params vp;
  vp.model                           = csf_model::mannos_sakrison;
  const std::vector<double> unhinted = chroma_visual_weights(5, vp, 1, 0, color_transform::none);
  const std::vector<double> luma     = luma_visual_weights(5, vp);
  expect(unhinted == luma, "default role: unhinted no-MCT component must take the luminance weights");
}

int run_checks() {
  check_golden_pin();
  check_isotropic_shift();
  check_422_anisotropy();
  check_default_role();
  if (failures == 0) {
    printf("visual_weight_check: all invariants hold\n");
    return EXIT_SUCCESS;
  }
  printf("visual_weight_check: %d failure(s)\n", failures);
  return EXIT_FAILURE;
}

int run_dump(int argc, char **argv) {
  visual_weighting_params vp;
  vp.model            = csf_model::mannos_sakrison;
  component_type role = component_type::Y;
  int sx = 1, sy = 1;
  int levels        = 5;
  int legacy_format = -1;
  for (int i = 2; i < argc; ++i) {
    if (std::strcmp(argv[i], "--csf") == 0 && i + 1 < argc) {
      const char *m = argv[++i];
      if (std::strcmp(m, "mannos") == 0) {
        vp.model = csf_model::mannos_sakrison;
      } else if (std::strcmp(m, "daly") == 0) {
        vp.model = csf_model::daly;
      } else {
        fprintf(stderr, "ERROR: unknown --csf '%s' (use mannos|daly)\n", m);
        return EXIT_FAILURE;
      }
    } else if (std::strcmp(argv[i], "--ppd") == 0 && i + 1 < argc) {
      vp.ref_ppd = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--zoom") == 0 && i + 1 < argc) {
      vp.zoom = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--levels") == 0 && i + 1 < argc) {
      levels = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--sx") == 0 && i + 1 < argc) {
      sx = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--sy") == 0 && i + 1 < argc) {
      sy = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--ctype") == 0 && i + 1 < argc) {
      const char *t = argv[++i];
      if (std::strcmp(t, "Y") == 0) {
        role = component_type::Y;
      } else if (std::strcmp(t, "Cb") == 0) {
        role = component_type::Cb;
      } else if (std::strcmp(t, "Cr") == 0) {
        role = component_type::Cr;
      } else if (std::strcmp(t, "generic") == 0) {
        role = component_type::generic;
      } else {
        fprintf(stderr, "ERROR: unknown --ctype '%s' (use Y|Cb|Cr|generic)\n", t);
        return EXIT_FAILURE;
      }
    } else if (std::strcmp(argv[i], "--reuse-luma") == 0) {
      vp.chroma_reuse_luma_csf = true;
    } else if (std::strcmp(argv[i], "--legacy-format") == 0 && i + 1 < argc) {
      const char *f = argv[++i];
      legacy_format = (std::strcmp(f, "420") == 0) ? 1 : (std::strcmp(f, "422") == 0) ? 2 : 0;
    } else {
      fprintf(stderr, "ERROR: unrecognized argument '%s'\n", argv[i]);
      return EXIT_FAILURE;
    }
  }
  if (levels < 1 || levels > 32 || sx < 1 || sy < 1) {
    fprintf(stderr, "ERROR: --levels must be 1..32 and --sx/--sy >= 1\n");
    return EXIT_FAILURE;
  }

  std::vector<double> w;
  if (legacy_format >= 0) {
    // Historical T.802-derived table row (15 entries = 5 levels), for
    // side-by-side comparison with the analytic output.
    const int comp_index = (role == component_type::Cr) ? 2 : 1;
    w                    = legacy_chroma_row(comp_index, legacy_format);
    levels               = 5;
    printf("# legacy table, comp=%s, format=%s\n", (comp_index == 1) ? "Cb" : "Cr",
           (legacy_format == 1)   ? "4:2:0"
           : (legacy_format == 2) ? "4:2:2"
                                  : "4:4:4");
  } else {
    w = weights_for(static_cast<uint8_t>(levels), vp, role, sx, sy);
    printf("# csf=%s ppd=%g zoom=%g ctype=%s sx=%d sy=%d%s\n",
           (vp.model == csf_model::daly) ? "daly" : "mannos", vp.ref_ppd, vp.zoom,
           (role == component_type::Y)    ? "Y"
           : (role == component_type::Cb) ? "Cb"
           : (role == component_type::Cr) ? "Cr"
                                          : "generic",
           sx, sy, vp.chroma_reuse_luma_csf ? " reuse-luma" : "");
  }
  printf("# level        HH            LH            HL   (finest first; sqrt-domain weights)\n");
  for (size_t lvl = 0; lvl < static_cast<size_t>(levels) && 3 * lvl + 2 < w.size(); ++lvl) {
    printf("%7d  %.10f  %.10f  %.10f\n", static_cast<int>(lvl) + 1, w[3 * lvl], w[3 * lvl + 1],
           w[3 * lvl + 2]);
  }
  return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc >= 2 && std::strcmp(argv[1], "check") == 0) {
    return run_checks();
  }
  if (argc >= 2 && std::strcmp(argv[1], "dump") == 0) {
    return run_dump(argc, argv);
  }
  fprintf(stderr,
          "Usage: %s check\n"
          "       %s dump [--csf mannos|daly] [--ppd F] [--zoom F] [--levels N]\n"
          "               [--ctype Y|Cb|Cr|generic] [--sx N] [--sy N] [--reuse-luma]\n"
          "               [--legacy-format 444|420|422]\n",
          argv[0], argv[0]);
  return EXIT_FAILURE;
}
