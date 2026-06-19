cmake_policy(SET CMP0076 NEW)
target_sources(estimate_qfactor
    PRIVATE
    main.cpp
)
# Share the encoder's visual-weighting header (header-only, self-contained) so the
# Qfactor inversion always matches the encoder's analytic CSF / color machinery.
target_include_directories(estimate_qfactor PRIVATE ${CMAKE_SOURCE_DIR}/source/core/codestream)
