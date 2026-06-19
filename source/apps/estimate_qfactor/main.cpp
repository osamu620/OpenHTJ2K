// Estimate the Qfactor used by an OpenHTJ2K-style encoder from a J2C codestream.
//
// Qfactor is not signaled in the bitstream; it only reshapes QCD/QCC step
// sizes via the formula in source/core/codestream/j2kmarkers.cpp. This tool
// parses the main header, recomputes the predicted (epsilon, mantissa) pairs
// for each candidate Q in [1..100], and reports the best-fit Q together with
// a residual. Large residual => the file was likely not produced by a Qfactor
// pipeline (or used different weighting tables).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

#include "visual_weighting.hpp"  // shared encoder weighting (single source of truth)

namespace {

constexpr uint16_t SOC = 0xFF4F;
constexpr uint16_t SIZ = 0xFF51;
constexpr uint16_t COD = 0xFF52;
constexpr uint16_t QCD = 0xFF5C;
constexpr uint16_t QCC = 0xFF5D;
constexpr uint16_t SOT = 0xFF90;

constexpr uint8_t YCC444 = 0;
constexpr uint8_t YCC420 = 1;
constexpr uint8_t YCC422 = 2;

struct Band {
  uint8_t epsilon;   // 0..31
  uint16_t mantissa; // 0..2047
};

struct QuantMarker {
  uint16_t component_index; // 0xFFFF for QCD, otherwise QCC's Cqcc
  uint8_t Sq;               // raw Sqcd/Sqcc
  uint8_t style;            // Sq & 0x1F (0=lossless, 1=derived, 2=expounded)
  uint8_t guardbits;        // Sq >> 5
  std::vector<Band> bands;  // signaling order: LL_N, HL_N, LH_N, HH_N, HL_{N-1}, ...
  std::vector<uint8_t> raw; // verbatim segment bytes (marker..end), for --dump-quant
};

struct Header {
  // SIZ
  uint16_t Csiz = 0;
  std::vector<uint8_t> Ssiz;     // bit-depth byte (top bit = signed)
  std::vector<uint8_t> XRsiz;
  std::vector<uint8_t> YRsiz;
  // COD
  uint8_t dwt_levels = 5;
  uint8_t transformation = 0;    // 0 = 9/7 (irreversible), 1 = 5/3 (reversible)
  bool use_color_trafo = false;  // SGcod MCT byte != 0
  // Quantization
  std::vector<QuantMarker> qmarkers;
};

uint16_t rd_u16(const uint8_t* p) { return static_cast<uint16_t>((p[0] << 8) | p[1]); }

uint8_t infer_chroma_format(const Header& h) {
  if (h.Csiz != 3) return YCC444;
  if (h.XRsiz[1] == 2 && h.XRsiz[2] == 2) {
    if (h.YRsiz[1] == 2 && h.YRsiz[2] == 2) return YCC420;
    if (h.YRsiz[1] == 1 && h.YRsiz[2] == 1) return YCC422;
  }
  return YCC444;
}

bool parse_main_header(const std::vector<uint8_t>& buf, Header& out) {
  if (buf.size() < 2 || rd_u16(buf.data()) != SOC) {
    fprintf(stderr, "ERROR: input does not start with SOC marker (0xFF4F)\n");
    return false;
  }
  size_t p = 2;
  bool got_siz = false, got_cod = false, got_qcd = false;
  while (p + 4 <= buf.size()) {
    uint16_t marker = rd_u16(&buf[p]);
    if (marker == SOT) break; // end of main header
    if ((marker & 0xFF00) != 0xFF00) {
      fprintf(stderr, "ERROR: invalid marker 0x%04X at offset %zu\n", marker, p);
      return false;
    }
    p += 2;
    if (p + 2 > buf.size()) return false;
    uint16_t Lmar = rd_u16(&buf[p]);
    if (Lmar < 2 || p + Lmar > buf.size()) {
      fprintf(stderr, "ERROR: marker 0x%04X has bad length %u\n", marker, Lmar);
      return false;
    }
    const uint8_t* body = &buf[p + 2];
    size_t body_len = static_cast<size_t>(Lmar) - 2;

    switch (marker) {
      case SIZ: {
        if (body_len < 36) return false;
        // body[0..1] = Rsiz, [2..5] = Xsiz, ..., [34..35] = Csiz
        out.Csiz = rd_u16(&body[34]);
        if (body_len < 36u + static_cast<size_t>(out.Csiz) * 3u) return false;
        for (uint16_t c = 0; c < out.Csiz; ++c) {
          const uint8_t* sc = &body[36 + 3 * c];
          out.Ssiz.push_back(sc[0]);
          out.XRsiz.push_back(sc[1]);
          out.YRsiz.push_back(sc[2]);
        }
        got_siz = true;
        break;
      }
      case COD: {
        if (body_len < 1 + 4 + 5) return false;
        uint8_t Scod = body[0];
        // SGcod (4 bytes): progression, layers (2), MCT (1)
        out.use_color_trafo = body[1 + 3] != 0;
        // SPcod: NLevels, cblkW, cblkH, style, transformation
        out.dwt_levels = body[1 + 4 + 0];
        out.transformation = body[1 + 4 + 4];
        (void)Scod;
        got_cod = true;
        break;
      }
      case QCD:
      case QCC: {
        QuantMarker qm{};
        // Verbatim segment bytes (marker + Lmar + payload). The marker began at
        // p-2 (we advanced p past it above); the whole segment is 2 + Lmar bytes.
        const uint8_t* seg = buf.data() + (p - 2);
        qm.raw.assign(seg, seg + 2u + Lmar);
        size_t off = 0;
        if (marker == QCC) {
          if (out.Csiz < 257) {
            qm.component_index = body[0];
            off = 1;
          } else {
            qm.component_index = rd_u16(&body[0]);
            off = 2;
          }
        } else {
          qm.component_index = 0xFFFF;
        }
        if (off + 1 > body_len) return false;
        qm.Sq = body[off++];
        qm.style = static_cast<uint8_t>(qm.Sq & 0x1F);
        qm.guardbits = static_cast<uint8_t>(qm.Sq >> 5);

        size_t band_count = 0;
        size_t bytes_per_band = 0;
        if (qm.style == 0) {           // no-quant (lossless): 1 byte per band, ε in upper 5 bits
          band_count = (body_len - off);
          bytes_per_band = 1;
        } else if (qm.style == 1) {    // scalar-derived: one (ε, μ) for LL only
          band_count = 1;
          bytes_per_band = 2;
        } else if (qm.style == 2) {    // scalar-expounded: 2 bytes per band
          band_count = (body_len - off) / 2;
          bytes_per_band = 2;
        } else {
          fprintf(stderr, "ERROR: unknown quant style %u\n", qm.style);
          return false;
        }
        for (size_t b = 0; b < band_count; ++b) {
          if (off + bytes_per_band > body_len) return false;
          if (bytes_per_band == 1) {
            qm.bands.push_back({static_cast<uint8_t>(body[off] >> 3), 0});
          } else {
            uint16_t v = rd_u16(&body[off]);
            qm.bands.push_back({static_cast<uint8_t>(v >> 11),
                                static_cast<uint16_t>(v & 0x7FF)});
          }
          off += bytes_per_band;
        }
        out.qmarkers.push_back(std::move(qm));
        if (qm.component_index == 0xFFFF) got_qcd = true;
        break;
      }
      default:
        break;
    }
    p += Lmar;
  }
  if (!got_siz || !got_cod || !got_qcd) {
    fprintf(stderr, "ERROR: missing SIZ/COD/QCD in main header\n");
    return false;
  }
  return true;
}

// Forward step-size predictor — mirrors QCD_marker / QCC_marker exactly.
// Returns predicted (epsilon, mantissa) per band in QCD signaling order
// (LL_N, HL_N, LH_N, HH_N, HL_{N-1}, ..., HL_1, LH_1, HH_1).
std::vector<Band> predict_bands(uint8_t qfactor, uint8_t dwt_levels, uint8_t RI,
                                uint8_t Cqcc, uint8_t chroma_format,
                                const open_htj2k::visual_weighting_params& vp, bool mct_on) {
  const std::vector<double> D97SL = {-0.091271763114250, -0.057543526228500, 0.591271763114250,
                                     1.115087052457000,  0.5912717631142500, -0.05754352622850,
                                     -0.091271763114250};
  const std::vector<double> D97SH = {0.053497514821622,  0.033728236885750,
                                     -0.156446533057980, -0.533728236885750,
                                     1.205898036472720,  -0.533728236885750,
                                     -0.156446533057980, 0.033728236885750,
                                     0.053497514821622};

  // Visual weights from the shared encoder header (single source of truth), so the
  // inversion tracks whatever model produced the file. Component 0 is luma (QCD);
  // components 1/2 are chroma (QCC) under a luma/chroma transform.
  const open_htj2k::color_transform ct = open_htj2k::resolve_color_transform(vp, mct_on);
  const std::vector<double> weights =
      (Cqcc == 0) ? open_htj2k::luma_visual_weights(dwt_levels, vp)
                  : open_htj2k::chroma_visual_weights(dwt_levels, vp, Cqcc, chroma_format, ct);

  // Build wmse in the encoder's accumulation order: HH_1, LH_1, HL_1, ..., LL_N.
  const size_t num_bands = static_cast<size_t>(3 * dwt_levels + 1);
  std::vector<double> wmse;
  wmse.reserve(num_bands);
  std::vector<double> outL(D97SL), outH(D97SH);
  double gain_low = 0, gain_high = 0;
  if (dwt_levels == 0) {
    wmse.push_back(1.0);
  } else {
    for (uint8_t lvl = 0; lvl < dwt_levels; ++lvl) {
      gain_low = 0;
      gain_high = 0;
      for (double e : outL) gain_low += e * e;
      for (double e : outH) gain_high += e * e;
      wmse.push_back(gain_high * gain_high); // HH
      wmse.push_back(gain_low * gain_high);  // LH
      wmse.push_back(gain_high * gain_low);  // HL
      auto upsample = [](const std::vector<double>& v) {
        std::vector<double> r;
        r.reserve(v.size() * 2);
        for (double x : v) {
          r.push_back(x);
          r.push_back(0.0);
        }
        return r;
      };
      std::vector<double> L2 = upsample(outL);
      std::vector<double> H2 = upsample(outH);
      std::vector<double> tmpL(D97SL.size() + L2.size() - 1, 0.0);
      for (size_t i = 0; i < D97SL.size(); ++i)
        for (size_t j = 0; j < L2.size(); ++j) tmpL[i + j] += D97SL[i] * L2[j];
      std::vector<double> tmpH(D97SL.size() + H2.size() - 1, 0.0);
      for (size_t i = 0; i < D97SL.size(); ++i)
        for (size_t j = 0; j < H2.size(); ++j) tmpH[i + j] += D97SL[i] * H2[j];
      outL = tmpL;
      outH = tmpH;
    }
    wmse.push_back(gain_low * gain_low); // LL_N
  }

  // Q-dependent scalars from the shared encoder header (same q_to_delta() the
  // encoder uses), so the inversion matches the encoder's step exactly.
  const open_htj2k::q_scaling qs = open_htj2k::q_to_delta(qfactor, RI);
  const double qpower    = qs.qfactor_power;
  const double delta_ref = qs.delta_Q * open_htj2k::color_gain(ct, 0);
  const double G_c       = open_htj2k::color_gain(ct, Cqcc);

  std::vector<Band> out(num_bands);
  for (size_t i = 0; i < num_bands; ++i) {
    // LL band (last entry, always weight 1.0) and any extra low-freq bands beyond the 5-level table
    double w_b = (i == num_bands - 1 || i >= weights.size()) ? 1.0 : std::pow(weights[i], qpower);
    double fval = delta_ref / (std::sqrt(wmse[i]) * w_b * G_c);
    int32_t exponent = 0;
    while (fval < 1.0) {
      fval *= 2.0;
      exponent++;
    }
    int32_t mantissa =
        static_cast<int32_t>(std::floor((fval - 1.0) * static_cast<double>(1 << 11) + 0.5));
    if (mantissa >= (1 << 11)) {
      mantissa = 0;
      exponent--;
    }
    if (exponent > 31) {
      exponent = 31;
      mantissa = 0;
    }
    if (exponent < 0) {
      exponent = 0;
      mantissa = (1 << 11) - 1;
    }
    out[num_bands - i - 1] = {static_cast<uint8_t>(exponent), static_cast<uint16_t>(mantissa)};
  }
  return out;
}

double step_value(const Band& b) {
  return std::ldexp(1.0 + static_cast<double>(b.mantissa) / 2048.0, -static_cast<int>(b.epsilon));
}

struct ScoreResult {
  uint8_t best_q;
  double best_residual; // sum of (log2 step_obs - log2 step_pred)^2 over bands
  double median_per_band;
};

ScoreResult find_best_q(const std::vector<Band>& observed, uint8_t dwt_levels, uint8_t RI,
                        uint8_t Cqcc, uint8_t chroma_format,
                        const open_htj2k::visual_weighting_params& vp, bool mct_on) {
  ScoreResult best{0, std::numeric_limits<double>::infinity(), 0.0};
  for (int q = 1; q <= 100; ++q) {
    auto pred = predict_bands(static_cast<uint8_t>(q), dwt_levels, RI, Cqcc, chroma_format, vp, mct_on);
    if (pred.size() != observed.size()) continue;
    double sumsq = 0;
    for (size_t i = 0; i < pred.size(); ++i) {
      double d = std::log2(step_value(observed[i])) - std::log2(step_value(pred[i]));
      sumsq += d * d;
    }
    if (sumsq < best.best_residual) {
      best.best_residual = sumsq;
      best.best_q = static_cast<uint8_t>(q);
    }
  }
  best.median_per_band =
      observed.empty() ? 0.0 : std::sqrt(best.best_residual / static_cast<double>(observed.size()));
  return best;
}

void print_band_table(const std::vector<Band>& obs, const std::vector<Band>& pred) {
  size_t n = std::min(obs.size(), pred.size());
  printf("    band  observed (eps, mu)   predicted (eps, mu)   step_obs/step_pred\n");
  for (size_t i = 0; i < n; ++i) {
    double ratio = step_value(obs[i]) / step_value(pred[i]);
    printf("    %3zu   (%2u, %4u)            (%2u, %4u)             %.4f\n",
           i, obs[i].epsilon, obs[i].mantissa, pred[i].epsilon, pred[i].mantissa, ratio);
  }
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr,
            "Usage: %s <codestream.j2c> [--verbose] [--csf legacy|mannos|daly] [--ppd F] [--zoom F]\n"
            "  Estimates the OpenHTJ2K Qfactor [0..100] used at encode time by inverting the\n"
            "  QCD/QCC step-size formula. The visual-weighting model is NOT signaled in the\n"
            "  codestream, so for an analytic (EXPERIMENTAL) encode pass the same --csf/--ppd/\n"
            "  --zoom used at encode time; a low residual confirms the assumption was right.\n"
            "  --expect-q N / --max-residual F: exit non-zero if violated (for scripts/CI).\n"
            "  --dump-quant FILE: write the verbatim QCD/QCC marker bytes to FILE and exit.\n"
            "    These bytes are a function of Qfactor + bit-depth + the visual-weighting\n"
            "    tables only (never of sample data), so they are identical on every platform\n"
            "    -- unlike the entropy-coded payload -- which makes them a portable golden\n"
            "    anchor for the \"default Qfactor output is bit-identical\" contract.\n",
            argv[0]);
    return 1;
  }
  bool verbose = false;
  int expect_q = -1;           // >= 0 enables an exit-code check on the recovered Q
  double max_residual = -1.0;  // >= 0 enables an exit-code check on the per-band residual
  const char* dump_quant = nullptr;        // != null: dump verbatim QCD/QCC bytes and exit
  open_htj2k::visual_weighting_params vp;  // default: legacy table (bit-identical inversion)
  for (int i = 2; i < argc; ++i) {
    if (std::strcmp(argv[i], "--verbose") == 0) {
      verbose = true;
    } else if (std::strcmp(argv[i], "--csf") == 0 && i + 1 < argc) {
      const char* m = argv[++i];
      if (std::strcmp(m, "legacy") == 0) {
        vp.model = open_htj2k::csf_model::legacy_table;
      } else if (std::strcmp(m, "mannos") == 0) {
        vp.model = open_htj2k::csf_model::mannos_sakrison;
      } else if (std::strcmp(m, "daly") == 0) {
        vp.model = open_htj2k::csf_model::daly;
      } else {
        fprintf(stderr, "ERROR: unknown --csf '%s' (use legacy|mannos|daly)\n", m);
        return 1;
      }
    } else if (std::strcmp(argv[i], "--ppd") == 0 && i + 1 < argc) {
      vp.ref_ppd = std::atof(argv[++i]);
      if (vp.ref_ppd <= 0.0) {
        fprintf(stderr, "ERROR: --ppd must be > 0\n");
        return 1;
      }
    } else if (std::strcmp(argv[i], "--zoom") == 0 && i + 1 < argc) {
      vp.zoom = std::atof(argv[++i]);
      if (vp.zoom <= 0.0) {
        fprintf(stderr, "ERROR: --zoom must be > 0\n");
        return 1;
      }
    } else if (std::strcmp(argv[i], "--expect-q") == 0 && i + 1 < argc) {
      expect_q = std::atoi(argv[++i]);
      if (expect_q < 0 || expect_q > 100) {
        fprintf(stderr, "ERROR: --expect-q must be in [0, 100]\n");
        return 1;
      }
    } else if (std::strcmp(argv[i], "--max-residual") == 0 && i + 1 < argc) {
      max_residual = std::atof(argv[++i]);
      if (max_residual < 0.0) {
        fprintf(stderr, "ERROR: --max-residual must be >= 0\n");
        return 1;
      }
    } else if (std::strcmp(argv[i], "--dump-quant") == 0 && i + 1 < argc) {
      dump_quant = argv[++i];
    } else {
      // Reject unknown flags (and flags missing their value) so a typo in a
      // CI/script invocation fails loudly instead of silently checking nothing.
      fprintf(stderr, "ERROR: unrecognized argument '%s'\n", argv[i]);
      return 1;
    }
  }

  std::ifstream in(argv[1], std::ios::binary);
  if (!in) {
    fprintf(stderr, "ERROR: cannot open '%s'\n", argv[1]);
    return 1;
  }
  std::vector<uint8_t> buf((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());

  Header h;
  if (!parse_main_header(buf, h)) return 1;

  // --dump-quant: emit the verbatim QCD/QCC marker segment bytes (a portable,
  // architecture-independent golden anchor) and exit before any Q inversion.
  if (dump_quant) {
    std::ofstream df(dump_quant, std::ios::binary | std::ios::trunc);
    if (!df) {
      fprintf(stderr, "ERROR: cannot write '%s'\n", dump_quant);
      return 1;
    }
    df << "# OpenHTJ2K Qfactor QCD/QCC golden dump -- verbatim marker bytes.\n"
          "# These derive only from Qfactor + bit-depth + visual_weighting.hpp\n"
          "# tables, never from sample data, so they are byte-identical on every\n"
          "# platform (unlike the entropy-coded payload). A change here means the\n"
          "# default Qfactor quantization output drifted.\n";
    for (const auto& qm : h.qmarkers) {
      char label[24];
      if (qm.component_index == 0xFFFF) {
        std::snprintf(label, sizeof label, "QCD     :");
      } else {
        std::snprintf(label, sizeof label, "QCC c=%u :", qm.component_index);
      }
      df << label;
      char hb[4];
      for (uint8_t b : qm.raw) {
        std::snprintf(hb, sizeof hb, " %02x", b);
        df << hb;
      }
      df << "\n";
    }
    return df.good() ? 0 : 1;
  }

  uint8_t chroma_format = infer_chroma_format(h);
  const char* cf_name = (chroma_format == YCC444) ? "4:4:4"
                        : (chroma_format == YCC420) ? "4:2:0" : "4:2:2";

  printf("File:        %s\n", argv[1]);
  printf("Components:  %u (chroma format %s)\n", h.Csiz, cf_name);
  printf("DWT levels:  %u\n", h.dwt_levels);
  printf("Transform:   %s\n", h.transformation == 1 ? "5/3 (reversible)" : "9/7 (irreversible)");
  printf("MCT:         %s\n", h.use_color_trafo ? "ON (RGB->YCbCr)" : "OFF");
  {
    const char* csf_name = (vp.model == open_htj2k::csf_model::legacy_table)      ? "legacy table"
                           : (vp.model == open_htj2k::csf_model::mannos_sakrison) ? "Mannos-Sakrison"
                                                                                  : "Daly";
    printf("Weighting:   %s", csf_name);
    if (vp.model != open_htj2k::csf_model::legacy_table) {
      printf(" (ref_ppd=%.1f, zoom=%.2f)", vp.ref_ppd, vp.zoom);
    }
    printf("\n");
  }

  if (h.transformation == 1) {
    printf("\nVerdict: lossless (5/3) — Qfactor does not apply.\n");
    if (expect_q >= 0 || max_residual >= 0.0) {
      printf("CHECK SKIP: no Qfactor to evaluate (lossless input)\n");
      return 3;  // check requested but not evaluable
    }
    return 0;
  }

  // Pair each quantization marker with its component / weighting table.
  // QCD is a fallback for any component without an explicit QCC.
  const QuantMarker* qcd = nullptr;
  for (const auto& qm : h.qmarkers) {
    if (qm.component_index == 0xFFFF) {
      qcd = &qm;
      break;
    }
  }
  if (!qcd) {
    fprintf(stderr, "ERROR: missing QCD marker\n");
    return 1;
  }
  if (qcd->style == 1) {
    printf("\nVerdict: QCD uses scalar-derived signaling (Sqcd & 0x1F = 1).\n");
    printf("         OpenHTJ2K's encoder forces expounded signaling whenever Qfactor\n");
    printf("         is enabled, so this stream was NOT produced with a Qfactor.\n");
    if (expect_q >= 0 || max_residual >= 0.0) {
      printf("CHECK SKIP: not a Qfactor stream (scalar-derived quantization)\n");
      return 3;  // check requested but not evaluable
    }
    return 0;
  }

  printf("\nGuard bits:  %u\n", qcd->guardbits);
  printf("Qstyle:      %u (%s)\n", qcd->style,
         qcd->style == 0 ? "no-quant" : qcd->style == 2 ? "scalar-expounded" : "scalar-derived");

  // Score per component.
  std::vector<std::pair<uint16_t, ScoreResult>> per_component;
  for (uint16_t c = 0; c < h.Csiz; ++c) {
    const QuantMarker* qm = qcd;
    for (const auto& m : h.qmarkers) {
      if (m.component_index == c) {
        qm = &m;
        break;
      }
    }
    uint8_t RI = static_cast<uint8_t>((h.Ssiz[c] & 0x7F) + 1);
    auto score = find_best_q(qm->bands, h.dwt_levels, RI,
                             static_cast<uint8_t>(c), chroma_format, vp, h.use_color_trafo);
    per_component.emplace_back(c, score);
    printf("\nComponent %u  (RI=%u, %s)\n", c, RI,
           qm->component_index == 0xFFFF ? "QCD" : "QCC");
    printf("  best Q     : %u\n", score.best_q);
    printf("  residual   : %.4f (sum log2 step^2)\n", score.best_residual);
    printf("  per-band   : %.4f log2 (~ factor %.4fx)\n",
           score.median_per_band, std::pow(2.0, score.median_per_band));
    if (verbose) {
      auto pred = predict_bands(score.best_q, h.dwt_levels, RI,
                                static_cast<uint8_t>(c), chroma_format, vp, h.use_color_trafo);
      print_band_table(qm->bands, pred);
    }
  }

  // Aggregate verdict.
  // Each ε,μ has ~1/2048 mantissa precision -> log2 step error floor ≈ 1/2048
  // per band. Treat residuals below ~0.05 per-band as a Qfactor match.
  double max_per_band = 0;
  for (auto& kv : per_component) max_per_band = std::max(max_per_band, kv.second.median_per_band);

  printf("\nSummary:\n");
  if (max_per_band < 0.05) {
    printf("  Qfactor estimate: %u  (consistent across components, residual %.4f log2)\n",
           per_component[0].second.best_q, max_per_band);
    printf("  This stream is a strong match for OpenHTJ2K's Qfactor pipeline.\n");
  } else if (max_per_band < 0.25) {
    printf("  Qfactor estimate: ~%u  (loose fit, per-band residual %.4f log2)\n",
           per_component[0].second.best_q, max_per_band);
    printf("  Likely Qfactor-encoded, but with a different chroma format or weighting.\n");
  } else {
    printf("  No Qfactor match (per-band residual %.4f log2).\n", max_per_band);
    printf("  This stream was probably encoded with an explicit base step size,\n");
    printf("  a different encoder, or a non-Qfactor rate-control scheme.\n");
  }

  // Scriptable / CI check mode: non-zero exit if any component violates the
  // expected Q or the per-band residual ceiling. Enabled by --expect-q / --max-residual.
  if (expect_q >= 0 || max_residual >= 0.0) {
    bool ok = true;
    for (const auto& kv : per_component) {
      if (expect_q >= 0 && kv.second.best_q != expect_q) ok = false;
      if (max_residual >= 0.0 && kv.second.median_per_band > max_residual) ok = false;
    }
    printf("\nCHECK %s:", ok ? "PASS" : "FAIL");
    if (expect_q >= 0) printf(" expect-q=%d", expect_q);
    if (max_residual >= 0.0) printf(" max-residual=%.4f", max_residual);
    for (const auto& kv : per_component)
      printf("  [c%u Q=%u r=%.4f]", kv.first, kv.second.best_q, kv.second.median_per_band);
    printf("\n");
    if (!ok) return 2;
  }
  return 0;
}
