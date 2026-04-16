// jpip_index_check: Phase-1 ctest harness for the JPIP precinct index.
// Usage: jpip_index_check <input.j2c> [--total N] [--per-res t,c=r0,r1,...]
//
// --total N             assert total_precincts() == N
// --per-res t,c=r0,r1,…  assert tile_component(t,c).npw[r]·nph[r] == r_i for each r
// --print               dump the index to stdout
//
// Exit 0 on all assertions passing, 1 otherwise.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "precinct_index.hpp"
#include "view_window.hpp"

static std::vector<uint8_t> read_file(const char *path) {
  FILE *f = std::fopen(path, "rb");
  if (!f) {
    std::fprintf(stderr, "ERROR: cannot open %s\n", path);
    return {};
  }
  std::fseek(f, 0, SEEK_END);
  std::size_t sz = static_cast<std::size_t>(std::ftell(f));
  std::fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> buf(sz);
  std::size_t rd = std::fread(buf.data(), 1, sz, f);
  std::fclose(f);
  if (rd != sz) {
    std::fprintf(stderr, "ERROR: partial read of %s\n", path);
    buf.clear();
  }
  return buf;
}

static void print_summary(const open_htj2k::jpip::CodestreamIndex &idx) {
  std::printf("tiles=%ux%u  components=%u  progression=%u  total_precincts=%llu\n",
              idx.num_tiles_x(), idx.num_tiles_y(), idx.num_components(),
              idx.progression_order(),
              static_cast<unsigned long long>(idx.total_precincts()));
  for (uint32_t t = 0; t < idx.num_tiles(); ++t) {
    for (uint16_t c = 0; c < idx.num_components(); ++c) {
      const auto &info = idx.tile_component(static_cast<uint16_t>(t), c);
      std::printf("  t=%u c=%u NL=%u total=%u  per-res:",
                  t, c, info.NL, info.total);
      for (uint8_t r = 0; r <= info.NL; ++r) {
        std::printf(" r%u=%ux%u(%u)", r, info.npw[r], info.nph[r],
                    info.npw[r] * info.nph[r]);
      }
      std::printf("\n");
    }
  }
}

// Parse "t,c=r0,r1,r2,..." into the four numbers and a vector of expected counts.
static bool parse_per_res_spec(const char *spec, uint16_t &t, uint16_t &c,
                               std::vector<uint32_t> &expected) {
  // Find '='
  const char *eq = std::strchr(spec, '=');
  if (!eq) return false;
  // Parse t,c
  uint32_t tt = 0, cc = 0;
  if (std::sscanf(spec, "%u,%u", &tt, &cc) != 2) return false;
  t = static_cast<uint16_t>(tt);
  c = static_cast<uint16_t>(cc);
  // Parse comma-separated expected counts after '='
  expected.clear();
  const char *p = eq + 1;
  while (*p) {
    char *end = nullptr;
    unsigned long v = std::strtoul(p, &end, 10);
    if (end == p) return false;
    expected.push_back(static_cast<uint32_t>(v));
    p = end;
    while (*p == ',') ++p;
  }
  return !expected.empty();
}

// Parse --vw arg of shape
//   "fx,fy,ox,oy,sx,sy[,round][,comps=c1:c2:…]=N"
// where round ∈ {down,up,closest} (default down), N is the expected
// precinct count, and comps is an optional colon-separated list.  Returns
// true on success.
static bool parse_vw_spec(const char *spec, open_htj2k::jpip::ViewWindow &vw,
                          uint64_t &expected_count) {
  // Split on the trailing '='
  const char *eq = std::strrchr(spec, '=');
  if (!eq) return false;
  expected_count = std::strtoull(eq + 1, nullptr, 10);

  // Walk the prefix split on commas.
  std::string prefix(spec, eq);
  std::size_t pos = 0;
  auto next = [&](std::string &out) -> bool {
    std::size_t comma = prefix.find(',', pos);
    if (comma == std::string::npos) {
      out = prefix.substr(pos);
      pos = prefix.size();
    } else {
      out = prefix.substr(pos, comma - pos);
      pos = comma + 1;
    }
    return !out.empty();
  };
  std::string field;
  if (!next(field)) return false; vw.fx = static_cast<uint32_t>(std::stoul(field));
  if (!next(field)) return false; vw.fy = static_cast<uint32_t>(std::stoul(field));
  if (!next(field)) return false; vw.ox = static_cast<uint32_t>(std::stoul(field));
  if (!next(field)) return false; vw.oy = static_cast<uint32_t>(std::stoul(field));
  if (!next(field)) return false; vw.sx = static_cast<uint32_t>(std::stoul(field));
  if (!next(field)) return false; vw.sy = static_cast<uint32_t>(std::stoul(field));
  while (next(field)) {
    if (field == "down")    vw.round = open_htj2k::jpip::ViewWindow::Round::Down;
    else if (field == "up") vw.round = open_htj2k::jpip::ViewWindow::Round::Up;
    else if (field == "closest") vw.round = open_htj2k::jpip::ViewWindow::Round::Closest;
    else if (field.rfind("comps=", 0) == 0) {
      std::string list = field.substr(6);
      std::size_t lp = 0;
      while (lp < list.size()) {
        std::size_t colon = list.find(':', lp);
        std::string item = list.substr(lp, colon == std::string::npos ? std::string::npos : colon - lp);
        if (!item.empty()) vw.comps.push_back(static_cast<uint16_t>(std::stoul(item)));
        if (colon == std::string::npos) break;
        lp = colon + 1;
      }
    } else {
      return false;
    }
  }
  return true;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "Usage: jpip_index_check <input.j2c> [--total N] "
                 "[--per-res t,c=r0,...] [--vw fx,fy,ox,oy,sx,sy[,round]=N] "
                 "[--print]\n");
    return 1;
  }
  std::string infile     = argv[1];
  long long expected_tot = -1;
  bool do_print          = false;
  std::vector<std::string> per_res_specs;
  std::vector<std::string> vw_specs;

  for (int i = 2; i < argc; ++i) {
    if (std::strcmp(argv[i], "--total") == 0 && i + 1 < argc) {
      expected_tot = std::atoll(argv[++i]);
    } else if (std::strcmp(argv[i], "--per-res") == 0 && i + 1 < argc) {
      per_res_specs.emplace_back(argv[++i]);
    } else if (std::strcmp(argv[i], "--vw") == 0 && i + 1 < argc) {
      vw_specs.emplace_back(argv[++i]);
    } else if (std::strcmp(argv[i], "--print") == 0) {
      do_print = true;
    } else {
      std::fprintf(stderr, "ERROR: unknown arg %s\n", argv[i]);
      return 1;
    }
  }

  auto bytes = read_file(infile.c_str());
  if (bytes.empty()) return 1;

  std::unique_ptr<open_htj2k::jpip::CodestreamIndex> idx;
  try {
    idx = open_htj2k::jpip::CodestreamIndex::build(bytes.data(), bytes.size());
  } catch (std::exception &e) {
    std::fprintf(stderr, "ERROR build: %s\n", e.what());
    return 1;
  }

  if (do_print) print_summary(*idx);

  int failed = 0;
  if (expected_tot >= 0) {
    auto got = idx->total_precincts();
    if (got != static_cast<uint64_t>(expected_tot)) {
      std::fprintf(stderr, "FAIL --total: expected %lld got %llu\n",
                   expected_tot, static_cast<unsigned long long>(got));
      ++failed;
    } else {
      std::printf("OK --total %lld\n", expected_tot);
    }
  }

  for (const auto &spec : per_res_specs) {
    uint16_t t = 0, c = 0;
    std::vector<uint32_t> expected;
    if (!parse_per_res_spec(spec.c_str(), t, c, expected)) {
      std::fprintf(stderr, "FAIL --per-res: cannot parse '%s'\n", spec.c_str());
      ++failed;
      continue;
    }
    const auto &info = idx->tile_component(t, c);
    if (expected.size() != static_cast<std::size_t>(info.NL) + 1u) {
      std::fprintf(stderr,
                   "FAIL --per-res t=%u c=%u: expected %zu resolutions, got NL+1=%u\n",
                   t, c, expected.size(), info.NL + 1u);
      ++failed;
      continue;
    }
    bool spec_ok = true;
    for (uint8_t r = 0; r <= info.NL; ++r) {
      const uint32_t got = info.npw[r] * info.nph[r];
      if (got != expected[r]) {
        std::fprintf(stderr,
                     "FAIL --per-res t=%u c=%u r=%u: expected %u got %u (npw=%u, nph=%u)\n",
                     t, c, r, expected[r], got, info.npw[r], info.nph[r]);
        spec_ok = false;
        ++failed;
      }
    }
    if (spec_ok) std::printf("OK --per-res %s\n", spec.c_str());
  }

  for (const auto &spec : vw_specs) {
    open_htj2k::jpip::ViewWindow vw;
    uint64_t expected = 0;
    if (!parse_vw_spec(spec.c_str(), vw, expected)) {
      std::fprintf(stderr, "FAIL --vw: cannot parse '%s'\n", spec.c_str());
      ++failed;
      continue;
    }
    auto got = open_htj2k::jpip::resolve_view_window(*idx, vw);
    if (got.size() != expected) {
      std::fprintf(stderr,
                   "FAIL --vw '%s': expected %llu precincts, got %zu\n",
                   spec.c_str(), static_cast<unsigned long long>(expected),
                   got.size());
      ++failed;
    } else {
      std::printf("OK --vw %s\n", spec.c_str());
    }
  }

  return failed == 0 ? 0 : 1;
}
