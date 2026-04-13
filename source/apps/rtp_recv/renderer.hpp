// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Licensed under the same BSD 3-Clause terms as the rest of OpenHTJ2K.

#pragma once

// Compile-time renderer dispatch: MetalRenderer on macOS, GlRenderer elsewhere.
// All pipeline code uses `Renderer` so the render loop is backend-agnostic.

#if defined(OPENHTJ2K_USE_METAL)
#  include "metal_renderer.hpp"
#else
#  include "gl_renderer.hpp"
#endif

namespace open_htj2k::rtp_recv {

#if defined(OPENHTJ2K_USE_METAL)
using Renderer = MetalRenderer;
#else
using Renderer = GlRenderer;
#endif

}  // namespace open_htj2k::rtp_recv
