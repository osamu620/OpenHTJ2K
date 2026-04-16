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
  // Accessor for the owned GLFWwindow* so callers can invoke input APIs
  // (glfwGetCursorPos, glfwSetKeyCallback, …) against the same window.
  // Returns nullptr when init() has not been called or has failed.
  GLFWwindow* get_window() const { return window_; }

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

  // ── Zero-copy plane buffer API ──────────────────────────────────────────
  // The decode thread calls acquire_plane_buffers() to get raw pointers into
  // GPU-visible shared memory.  After decoding, draw_acquired_planes() renders
  // directly from those buffers — no memcpy, no upload.
  //
  // Internally uses a ring of 3 buffer sets so the decode thread always has
  // a free set even while the GPU renders one and another sits in the
  // LatestSlot waiting for the render thread.
  struct PlanePointers {
    void    *y, *cb, *cr;     // Raw pointers into MTLBuffer.contents
    uint32_t stride_y;        // Row stride in samples (= width for packed planes)
    uint32_t stride_c;
    int      ring_index;      // Opaque — pass back to draw_acquired_planes
  };

  // Thread-safe: may be called from the decode thread.
  PlanePointers acquire_plane_buffers(uint32_t w_y, uint32_t h_y,
                                      uint32_t w_c, uint32_t h_c, int bpp);

  // Must be called from the main (render) thread.
  void draw_acquired_planes(int ring_index, int w_y, int h_y, int w_c, int h_c,
                            int bpp, int bit_depth,
                            const ycbcr_coefficients* coeffs, bool components_are_rgb,
                            const ColorPipelineParams& pipeline);

 private:
  // Opaque pointer to the Obj-C implementation struct.  Defined in metal_renderer.mm.
  struct Impl;
  Impl* impl_ = nullptr;

  GLFWwindow* window_ = nullptr;
};

}  // namespace open_htj2k::rtp_recv
