// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Licensed under the same BSD 3-Clause terms as the rest of OpenHTJ2K.

#pragma once

// Native Metal renderer for macOS.  Same public interface as GlRenderer so the
// pipeline code is backend-agnostic via the Renderer typedef in renderer.hpp.
//
// Uses GLFW for window creation (GLFW_CLIENT_API = GLFW_NO_API) and event
// handling.  Rendering goes through CAMetalLayer + MTLDevice directly,
// bypassing the macOS OpenGL→Metal translation layer.

#include <cstdint>

struct GLFWwindow;

// Forward-declare Obj-C types as opaque pointers for the C++ header.
// The actual id<MTL*> types are used in the .mm implementation.
#ifdef __OBJC__
@protocol MTLDevice;
@protocol MTLCommandQueue;
@protocol MTLRenderPipelineState;
@protocol MTLTexture;
@protocol MTLBuffer;
@class CAMetalLayer;
#else
typedef void* id;
#endif

namespace open_htj2k::rtp_recv {

struct ycbcr_coefficients;
struct ColorPipelineParams;

class MetalRenderer {
 public:
  MetalRenderer() = default;
  ~MetalRenderer();

  MetalRenderer(const MetalRenderer&)            = delete;
  MetalRenderer& operator=(const MetalRenderer&) = delete;

  bool init(int window_w, int window_h, const char* title, bool vsync = true);
  void shutdown();
  bool should_close() const;
  void poll_events();

  void upload_and_draw(const uint8_t* rgb, int w, int h);

  void upload_planar_and_draw(const uint8_t* y_plane, const uint8_t* cb_plane,
                              const uint8_t* cr_plane, int w_y, int h_y, int w_c, int h_c,
                              const ycbcr_coefficients* coeffs, bool components_are_rgb,
                              const ColorPipelineParams& pipeline);

  void upload_planar_16_and_draw(const uint16_t* y_plane, const uint16_t* cb_plane,
                                 const uint16_t* cr_plane, int w_y, int h_y, int w_c, int h_c,
                                 int bit_depth, const ycbcr_coefficients* coeffs,
                                 bool components_are_rgb,
                                 const ColorPipelineParams& pipeline);

 private:
  // Opaque pointer to the Obj-C implementation struct.  Defined in metal_renderer.mm.
  struct Impl;
  Impl* impl_ = nullptr;

  GLFWwindow* window_ = nullptr;
};

}  // namespace open_htj2k::rtp_recv
