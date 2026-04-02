# Part 2 decoder conformance tests: DFS (0xFF72) and ATK (0xFF79)
# Each codestream is tested with both the batch path (-batch) and the default
# line-based streaming path (invoke_line_based_stream).

# ── ATK_DFS_IRV: irrev97 (9/7) + DFS (L1-L2 BIDIR, L3-L5 HORZ) ─────────────
# 4K YCbCr 4:2:2 12-bit lossy
# Note: line-based streaming does not yet support non-BIDIR DFS levels; use -batch.

# Batch path
add_test(NAME dec_p2_dfs_irv COMMAND open_htj2k_dec -batch -i ${CONFORMANCE_DATA_DIR}/ATK_DFS_IRV.j2c -o p2_dfs_irv.pgx)

# Thresholds: PAE≤2, PSNR≥70 dB
add_test(NAME comp_p2_dfs_irv_c0 COMMAND imgcmp p2_dfs_irv_00.pgx ${CONFORMANCE_DATA_DIR}/references/p2_dfs_irv-0.pgx 2 70)
set_tests_properties(comp_p2_dfs_irv_c0 PROPERTIES DEPENDS dec_p2_dfs_irv)
add_test(NAME comp_p2_dfs_irv_c1 COMMAND imgcmp p2_dfs_irv_01.pgx ${CONFORMANCE_DATA_DIR}/references/p2_dfs_irv-1.pgx 2 70)
set_tests_properties(comp_p2_dfs_irv_c1 PROPERTIES DEPENDS dec_p2_dfs_irv)
add_test(NAME comp_p2_dfs_irv_c2 COMMAND imgcmp p2_dfs_irv_02.pgx ${CONFORMANCE_DATA_DIR}/references/p2_dfs_irv-2.pgx 2 70)
set_tests_properties(comp_p2_dfs_irv_c2 PROPERTIES DEPENDS dec_p2_dfs_irv)

# Line-based streaming path (invoke_line_based_stream)
add_test(NAME dec_p2_dfs_irv_lb COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ATK_DFS_IRV.j2c -o p2_dfs_irv_lb.pgx)

add_test(NAME comp_p2_dfs_irv_lb_c0 COMMAND imgcmp p2_dfs_irv_lb_00.pgx ${CONFORMANCE_DATA_DIR}/references/p2_dfs_irv-0.pgx 2 70)
set_tests_properties(comp_p2_dfs_irv_lb_c0 PROPERTIES DEPENDS dec_p2_dfs_irv_lb)
add_test(NAME comp_p2_dfs_irv_lb_c1 COMMAND imgcmp p2_dfs_irv_lb_01.pgx ${CONFORMANCE_DATA_DIR}/references/p2_dfs_irv-1.pgx 2 70)
set_tests_properties(comp_p2_dfs_irv_lb_c1 PROPERTIES DEPENDS dec_p2_dfs_irv_lb)
add_test(NAME comp_p2_dfs_irv_lb_c2 COMMAND imgcmp p2_dfs_irv_lb_02.pgx ${CONFORMANCE_DATA_DIR}/references/p2_dfs_irv-2.pgx 2 70)
set_tests_properties(comp_p2_dfs_irv_lb_c2 PROPERTIES DEPENDS dec_p2_dfs_irv_lb)

# ── ATK_DFS_REV: ATK irrev53 (idx=2, Katk=1.0) + same DFS ───────────────────
# 4K YCbCr 4:2:2 12-bit lossy
# Note: line-based streaming does not yet support non-BIDIR DFS levels; use -batch.

# Batch path
add_test(NAME dec_p2_dfs_atk COMMAND open_htj2k_dec -batch -i ${CONFORMANCE_DATA_DIR}/ATK_DFS_REV.j2c -o p2_dfs_atk.pgx)

# Thresholds: PAE≤256, PSNR≥65 dB
add_test(NAME comp_p2_dfs_atk_c0 COMMAND imgcmp p2_dfs_atk_00.pgx ${CONFORMANCE_DATA_DIR}/references/p2_dfs_atk-0.pgx 256 65)
set_tests_properties(comp_p2_dfs_atk_c0 PROPERTIES DEPENDS dec_p2_dfs_atk)
add_test(NAME comp_p2_dfs_atk_c1 COMMAND imgcmp p2_dfs_atk_01.pgx ${CONFORMANCE_DATA_DIR}/references/p2_dfs_atk-1.pgx 256 65)
set_tests_properties(comp_p2_dfs_atk_c1 PROPERTIES DEPENDS dec_p2_dfs_atk)
add_test(NAME comp_p2_dfs_atk_c2 COMMAND imgcmp p2_dfs_atk_02.pgx ${CONFORMANCE_DATA_DIR}/references/p2_dfs_atk-2.pgx 256 65)
set_tests_properties(comp_p2_dfs_atk_c2 PROPERTIES DEPENDS dec_p2_dfs_atk)

# Line-based streaming path (invoke_line_based_stream)
add_test(NAME dec_p2_dfs_atk_lb COMMAND open_htj2k_dec -i ${CONFORMANCE_DATA_DIR}/ATK_DFS_REV.j2c -o p2_dfs_atk_lb.pgx)

add_test(NAME comp_p2_dfs_atk_lb_c0 COMMAND imgcmp p2_dfs_atk_lb_00.pgx ${CONFORMANCE_DATA_DIR}/references/p2_dfs_atk-0.pgx 256 65)
set_tests_properties(comp_p2_dfs_atk_lb_c0 PROPERTIES DEPENDS dec_p2_dfs_atk_lb)
add_test(NAME comp_p2_dfs_atk_lb_c1 COMMAND imgcmp p2_dfs_atk_lb_01.pgx ${CONFORMANCE_DATA_DIR}/references/p2_dfs_atk-1.pgx 256 65)
set_tests_properties(comp_p2_dfs_atk_lb_c1 PROPERTIES DEPENDS dec_p2_dfs_atk_lb)
add_test(NAME comp_p2_dfs_atk_lb_c2 COMMAND imgcmp p2_dfs_atk_lb_02.pgx ${CONFORMANCE_DATA_DIR}/references/p2_dfs_atk-2.pgx 256 65)
set_tests_properties(comp_p2_dfs_atk_lb_c2 PROPERTIES DEPENDS dec_p2_dfs_atk_lb)
