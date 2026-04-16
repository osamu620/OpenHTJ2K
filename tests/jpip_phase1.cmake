# JPIP Phase 1, commit 1 — precinct-index ctests.
#
# Validates CodestreamIndex against two NASA Blue Marble derivatives that
# live in build/bin/ rather than conformance_data/, since the source PPM
# is too large to commit.  Both tests are skipped (status PASSED via
# WILL_FAIL FALSE + manual existence check) when the asset is absent.

set(_JPIP_BIN_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

# ── Asset 1: full-resolution NASA Blue Marble (LRCP, default precincts) ──
# Each (t, c) → NL+1 = 6 resolutions, each holding exactly one precinct
# because PPx=PPy=15 (max) covers the whole resolution.  3 components ×
# 6 resolutions = 18 precincts total.
if (EXISTS "${_JPIP_BIN_DIR}/land_shallow_topo_21600.j2c")
    add_test(NAME jpip_idx_land21600_total
             COMMAND jpip_index_check ${_JPIP_BIN_DIR}/land_shallow_topo_21600.j2c
                     --total 18)
    add_test(NAME jpip_idx_land21600_per_res
             COMMAND jpip_index_check ${_JPIP_BIN_DIR}/land_shallow_topo_21600.j2c
                     --per-res 0,0=1,1,1,1,1,1
                     --per-res 0,1=1,1,1,1,1,1
                     --per-res 0,2=1,1,1,1,1,1)
endif()

# ── Asset 2: 1920×1920 foveation demo (PCRL, 64×64 precincts) ──
# Per-component breakdown for canvas 1920×1920, NL=5, PPx=PPy=64:
#   r=0 (LL,    60×60): npw=1  nph=1  →   1
#   r=1 (      120×120): npw=2  nph=2  →   4
#   r=2 (      240×240): npw=4  nph=4  →  16
#   r=3 (      480×480): npw=8  nph=8  →  64
#   r=4 (      960×960): npw=15 nph=15 → 225
#   r=5 (full 1920×1920): npw=30 nph=30 → 900
# Per component total = 1210; × 3 components = 3630.
if (EXISTS "${_JPIP_BIN_DIR}/land_shallow_topo_1920_fov.j2c")
    add_test(NAME jpip_idx_land1920_total
             COMMAND jpip_index_check ${_JPIP_BIN_DIR}/land_shallow_topo_1920_fov.j2c
                     --total 3630)
    add_test(NAME jpip_idx_land1920_per_res
             COMMAND jpip_index_check ${_JPIP_BIN_DIR}/land_shallow_topo_1920_fov.j2c
                     --per-res 0,0=1,4,16,64,225,900
                     --per-res 0,1=1,4,16,64,225,900
                     --per-res 0,2=1,4,16,64,225,900)
endif()
