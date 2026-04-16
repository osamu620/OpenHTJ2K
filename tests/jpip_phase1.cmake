# JPIP Phase 1, commit 1 — precinct-index ctests.
#
# Validates CodestreamIndex against two NASA Blue Marble derivatives that
# live in build/bin/ rather than conformance_data/, since the source PPM
# is too large to commit.  Both tests are skipped (status PASSED via
# WILL_FAIL FALSE + manual existence check) when the asset is absent.

set(_JPIP_BIN_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

# ── Decoder precinct-filter sanity (§M.4.1 partial-decode plumbing) ──
# Exercises the public openhtj2k_decoder::set_precinct_filter hook against a
# Part-1 and a Part-15 conformance stream.  The assets live under
# conformance_data/ so these tests always run.
#
#   --identity : keep-every-precinct filter → byte-identical to no-filter.
#   --empty N  : drop-every-precinct filter → uniform field with peak-absolute
#                spread ≤ N per component.  9/7 lossy + ICT needs N=4 to
#                absorb inverse-colour-transform rounding of a (0,0,0) input.
add_test(NAME jpip_pd_identity_p0_04
         COMMAND jpip_partial_decode_check ${CONFORMANCE_DATA_DIR}/p0_04.j2k --identity)
add_test(NAME jpip_pd_empty_p0_04
         COMMAND jpip_partial_decode_check ${CONFORMANCE_DATA_DIR}/p0_04.j2k --empty 4)
add_test(NAME jpip_pd_identity_ht_01
         COMMAND jpip_partial_decode_check ${CONFORMANCE_DATA_DIR}/ds0_ht_01_b11.j2k --identity)
add_test(NAME jpip_pd_empty_ht_01
         COMMAND jpip_partial_decode_check ${CONFORMANCE_DATA_DIR}/ds0_ht_01_b11.j2k --empty 4)


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

    # ── view-window resolver (§C.4 + §M.4.1) ──
    # Full-image at full resolution → every precinct.
    add_test(NAME jpip_vw_land1920_full_res
             COMMAND jpip_index_check ${_JPIP_BIN_DIR}/land_shallow_topo_1920_fov.j2c
                     --vw 1920,1920,0,0,1920,1920=3630)
    # Empty region (sx=sy=0) is treated as whole-image, per §C.4.4.
    add_test(NAME jpip_vw_land1920_empty_region
             COMMAND jpip_index_check ${_JPIP_BIN_DIR}/land_shallow_topo_1920_fov.j2c
                     --vw 1920,1920,0,0,0,0=3630)
    # Half-res whole image — r*=1 drops r=5 (900/component).  310·3 = 930.
    add_test(NAME jpip_vw_land1920_half_res
             COMMAND jpip_index_check ${_JPIP_BIN_DIR}/land_shallow_topo_1920_fov.j2c
                     --vw 960,960,0,0,960,960=930)
    # Round-down at fx=800 → r*=2 (480).  Kept r=0..3: 1+4+16+64=85, ×3 = 255.
    add_test(NAME jpip_vw_land1920_round_down
             COMMAND jpip_index_check ${_JPIP_BIN_DIR}/land_shallow_topo_1920_fov.j2c
                     --vw 800,800,0,0,800,800,down=255)
    # Round-up at fx=800 → r*=1 (960).  Region sx=800 on fx'=960 grid maps to
    # (0,0)..(1600,1600) on canvas — NOT the whole canvas.  717 precincts.
    add_test(NAME jpip_vw_land1920_round_up
             COMMAND jpip_index_check ${_JPIP_BIN_DIR}/land_shallow_topo_1920_fov.j2c
                     --vw 800,800,0,0,800,800,up=717)
    # 200×200 centred RoI at full res — foveation probe.  Per resolution at
    # r=0..5 with DWT over-fetch margin 8: 1, 4, 4, 4, 9, 16 per component
    # → 38·3 = 114.
    add_test(NAME jpip_vw_land1920_centre_roi
             COMMAND jpip_index_check ${_JPIP_BIN_DIR}/land_shallow_topo_1920_fov.j2c
                     --vw 1920,1920,860,860,200,200=114)
    # Same RoI, Y component only (comps=0).  Per-component total 38.
    add_test(NAME jpip_vw_land1920_centre_roi_yonly
             COMMAND jpip_index_check ${_JPIP_BIN_DIR}/land_shallow_topo_1920_fov.j2c
                     --vw "1920,1920,860,860,200,200,comps=0=38")
    # Corner 960×960 quadrant at full res — per component 225+64+16+4+4+1 = 314,
    # ×3 = 942 (includes DWT over-fetch at coarser resolutions).
    add_test(NAME jpip_vw_land1920_corner_quadrant
             COMMAND jpip_index_check ${_JPIP_BIN_DIR}/land_shallow_topo_1920_fov.j2c
                     --vw 1920,1920,0,0,960,960=942)

    # Partial-decode sanity on the 9/7 foveation asset (PAE ≤ 2 since Cycc=on).
    add_test(NAME jpip_pd_identity_land1920_fov
             COMMAND jpip_partial_decode_check
                     ${_JPIP_BIN_DIR}/land_shallow_topo_1920_fov.j2c --identity)
    add_test(NAME jpip_pd_empty_land1920_fov
             COMMAND jpip_partial_decode_check
                     ${_JPIP_BIN_DIR}/land_shallow_topo_1920_fov.j2c --empty 2)
endif()
