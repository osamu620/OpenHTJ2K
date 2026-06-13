// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Licensed under the same BSD 3-Clause terms as the rest of OpenHTJ2K.

#pragma once

// GLFW + OpenGL 3.3 core renderer for the RFC 9828 receiver.
//
// Three draw paths share one GL context, one VAO, and the same vertex shader:
//
//   - upload_and_draw(rgb, w, h)                     → RGB8 texture +
//     passthrough fragment shader.  Used by the CPU fallback path
//     (decode_to_rgb_buffer) for headless / GL-limited envs.
//   - upload_planar_and_draw(y, cb, cr, ...)        → three R8 textures +
//     YCbCr matrix fragment shader.  Used by the shader color path when the
//     source is 8-bit.  CPU does a clamp+shift to u8 before upload.
//   - upload_planar_16_and_draw(y16, cb16, cr16,    → three R16 textures +
//     ..., bit_depth)                                 the same fragment
//     shader.  Used when the source is 10/12/16-bit so the LSBs survive
//     the CPU -> GPU hop.  CPU does a clamp-only pack to u16; the fragment
//     shader renormalizes via a uNormScale uniform set from bit_depth so the
//     bias/scale/matrix math runs in normalized [0, 1] regardless of
//     source depth.  GL_R16 is unsigned-normalized so GL_LINEAR chroma
//     upsampling still works (GL_R16UI would need nearest+manual bilinear).
//
// init() tries to create a GL 3.3 core context.  Returns false if glfwInit
// or context creation fails; the caller should log and fall back to
// --no-render.  load_functions() resolves GL >=2.0 entry points via
// glfwGetProcAddress in gl_loader.cpp.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>

struct GLFWwindow;

namespace open_htj2k::rtp_recv {

struct ycbcr_coefficients;    // from ycbcr_rgb.hpp
struct ColorPipelineParams;   // from color_pipeline.hpp

class GlRenderer {
 public:
  GlRenderer() = default;
  ~GlRenderer();

  GlRenderer(const GlRenderer&)            = delete;
  GlRenderer& operator=(const GlRenderer&) = delete;

  // Open a window of the given logical size with the given title.  The
  // GL 3.3 core context becomes current on this thread.  Compiles both
  // shader programs and allocates the fullscreen-triangle-strip VBO.
  // Returns false on any failure; the caller should fall back to a
  // headless mode.
  bool init(int window_w, int window_h, const char* title, bool vsync = true);

  // Tear down the window and GL context.  Safe to call multiple times.
  void shutdown();

  // True if the user clicked the close button or hit ESC.
  bool should_close() const;

  // Pump window events.  Call once per frame.
  void poll_events();

  // Accessor for the owned GLFWwindow* so callers can invoke GLFW input
  // APIs (glfwGetCursorPos, glfwSetKeyCallback, …) against the same window.
  // Returns nullptr when init() has not been called or has failed.
  GLFWwindow* get_window() const { return window_; }

  // CPU fallback path: upload an 8-bit interleaved RGB image and draw it
  // once.  Reallocates the RGB texture if (w,h) changed since the last
  // call.  Does NOT own the pixel buffer.
  void upload_and_draw(const uint8_t* rgb, int w, int h);

  // Shader path, 8-bit source: upload three 8-bit planar samples (Y and
  // two chroma planes, already right-shifted to 8 bits on the CPU side)
  // and draw them through the YCbCr->RGB fragment shader.  Cb/Cr widths/
  // heights may be smaller than luma for 4:2:2 / 4:2:0; GL_LINEAR +
  // clamp-to-edge handles the chroma upsample.  `coeffs` selects
  // BT.601/709 and full/narrow range; the matrix + normalization bias/
  // scale are baked into shader uniforms on each call.
  // `components_are_rgb` skips the matrix entirely and treats the three
  // planes as R/G/B.
  void upload_planar_and_draw(const uint8_t* y_plane, const uint8_t* cb_plane,
                              const uint8_t* cr_plane, int w_y, int h_y, int w_c, int h_c,
                              const ycbcr_coefficients* coeffs, bool components_are_rgb,
                              const ColorPipelineParams& pipeline);

  // Shader path, >8-bit source: upload three 16-bit planar samples
  // (unsigned, clamped to [0, (1<<bit_depth)-1], NOT right-shifted) into
  // three GL_R16 textures and draw them through the same YCbCr->RGB
  // fragment shader.  The shader samples each texture as a [0, 1]
  // normalized value and then multiplies by a uNormScale of
  // (65535 / ((1<<bit_depth)-1)) so the sample renormalizes to [0, 1]
  // in the source's native scale before bias/scale/matrix math runs.
  // `bit_depth` is the source luma depth (9..16); for chroma planes at a
  // different depth, pass the luma depth and rely on the streams we've
  // encountered which are always same-depth across planes.  The LSBs
  // the 8-bit path would have truncated are preserved through to the
  // fragment stage, which is the foundational slice for HDR.
  void upload_planar_16_and_draw(const uint16_t* y_plane, const uint16_t* cb_plane,
                                 const uint16_t* cr_plane, int w_y, int h_y, int w_c, int h_c,
                                 int bit_depth, const ycbcr_coefficients* coeffs,
                                 bool components_are_rgb,
                                 const ColorPipelineParams& pipeline);

  // ── Zero-copy plane buffer API (mirrors MetalRenderer) ─────────────────
  // The decode thread calls acquire_plane_buffers() to get raw pointers
  // into a persistently-mapped GL buffer (ARB_buffer_storage, GL 4.4) and
  // writes decoded samples straight into it via PlanarOutputDesc.  The
  // main thread then calls draw_acquired_planes(), whose glTexSubImage2D
  // sources from the bound buffer — a driver-side DMA with no CPU copy on
  // either thread.  This is the staging-free design the BufferSubData PBO
  // ring (reverted in PR #428) was not: the decoder's plane write, which
  // must happen anyway, IS the upload write.
  //
  // Ring discipline (3 slots): acquire() claims a Free slot (atomic CAS,
  // no GL calls — safe on the decode thread).  draw_acquired_planes()
  // inserts a fence after the texture uploads; poll_events() retires
  // signalled fences back to Free.  If no slot is free, or the ring is
  // not built yet (buffer creation needs the GL context, so the first
  // acquire only *requests* dimensions and poll_events() builds the ring
  // on the main thread), acquire returns null pointers and the caller
  // falls back to the plane-vector upload path for that frame.
  struct PlanePointers {
    void    *y = nullptr, *cb = nullptr, *cr = nullptr;
    uint32_t stride_y   = 0;  // row stride in samples (= width, packed)
    uint32_t stride_c   = 0;
    int      ring_index = -1;  // opaque — pass back to draw_acquired_planes
  };

  // Thread-safe: may be called from the decode thread.  Null .y => use
  // the vector fallback for this frame.
  PlanePointers acquire_plane_buffers(uint32_t w_y, uint32_t h_y, uint32_t w_c, uint32_t h_c,
                                      int bpp);

  // Thread-safe, no GL calls: returns an acquired-but-never-drawn slot
  // (e.g. its DecodedFrame was evicted from the render slot, or decode
  // failed mid-frame) to the Free state.
  void release_plane_buffers(int ring_index);

  // Must be called from the main (render) thread.
  void draw_acquired_planes(int ring_index, int w_y, int h_y, int w_c, int h_c, int bpp,
                            int bit_depth, const ycbcr_coefficients* coeffs,
                            bool components_are_rgb, const ColorPipelineParams& pipeline);

 private:
  bool compile_shader_programs();
  bool ensure_rgb_texture(int w, int h);
  // Allocate/resize three planar textures.  `bpp` is 1 for GL_R8 or 2 for
  // GL_R16; swapping bpp across frames forces a fresh `glTexImage2D`
  // allocation on each texture (not just `glTexSubImage2D`).  Each
  // texture caches its own (w, h, bpp) so swapping one bit depth doesn't
  // silently skip reallocation on the others (same class of bug that
  // bit us when Cb/Cr shared (w, h) tracking in the initial draft).
  bool ensure_planar_textures(int w_y, int h_y, int w_c, int h_c, int bpp);
  // Upload all three planes into tex_y_/tex_cb_/tex_cr_ via direct
  // client-memory glTexSubImage2D (see the comment in the definition for
  // why a PBO staging ring was tried and rejected).  `type` is
  // GL_UNSIGNED_BYTE or GL_UNSIGNED_SHORT, matching `bpp` 1 or 2.
  void upload_planar_textures(const void* y_plane, const void* cb_plane, const void* cr_plane,
                              int w_y, int h_y, int w_c, int h_c, unsigned int type, int bpp);
  void draw_ycbcr_program(int w_y, int h_y, const ycbcr_coefficients* coeffs,
                          const float* norm_scale, const ColorPipelineParams& pipeline);
  void draw_planar_rgb_program(int w_y, int h_y, const float* norm_scale,
                               const ColorPipelineParams& pipeline);
  void draw_fullscreen_quad(int fb_w, int fb_h, int content_w, int content_h);
  // Drains the GL error queue and logs each non-zero code with the
  // given context label.  Called only at allocation / program-link
  // sync points so the per-frame draw path isn't forced to sync with
  // the driver.  Returns true if no error was observed.
  bool check_gl_error(const char* context) const;

  GLFWwindow* window_ = nullptr;

  // Vertex array and vertex buffer for a single fullscreen triangle
  // strip.  Filled once in init(); never re-uploaded.
  unsigned int vao_ = 0;
  unsigned int vbo_ = 0;

  // RGB passthrough program (CPU fallback path).
  unsigned int prog_rgb_       = 0;
  int          u_rgb_tex_      = -1;
  int          u_rgb_viewport_ = -1;  // vec4(x0, y0, w, h) in normalized device coords
  unsigned int tex_rgb_        = 0;
  int          tex_rgb_w_      = 0;
  int          tex_rgb_h_      = 0;

  // YCbCr planar program — always applies matrix + bias + scale.
  unsigned int prog_ycbcr_       = 0;
  int          u_yc_y_tex_       = -1;
  int          u_yc_cb_tex_      = -1;
  int          u_yc_cr_tex_      = -1;
  int          u_yc_matrix_      = -1;
  int          u_yc_bias_        = -1;
  int          u_yc_scale_       = -1;
  int          u_yc_norm_scale_  = -1;
  int          u_yc_transfer_    = -1;
  int          u_yc_gamut_       = -1;
  int          u_yc_display_enc_ = -1;
  int          u_yc_tonemap_     = -1;
  int          u_yc_tm_src_pq_   = -1;
  int          u_yc_tm_ks_       = -1;
  int          u_yc_tm_max_lum_  = -1;

  // Planar RGB passthrough program — no matrix/bias/scale, just HDR pipeline.
  unsigned int prog_planar_rgb_       = 0;
  int          u_pr_y_tex_            = -1;
  int          u_pr_cb_tex_           = -1;
  int          u_pr_cr_tex_           = -1;
  int          u_pr_norm_scale_       = -1;
  int          u_pr_transfer_         = -1;
  int          u_pr_gamut_            = -1;
  int          u_pr_display_enc_      = -1;
  int          u_pr_tonemap_          = -1;
  int          u_pr_tm_src_pq_        = -1;
  int          u_pr_tm_ks_            = -1;
  int          u_pr_tm_max_lum_       = -1;
  unsigned int tex_y_            = 0;
  unsigned int tex_cb_           = 0;
  unsigned int tex_cr_           = 0;
  // Per-texture (width, height, bytes-per-pixel) so we can detect resize or
  // format swaps on any of the three planes independently.  Sharing any of
  // these across textures is a bug class -- see the comment above
  // ensure_planar_textures in gl_renderer.cpp.
  int          tex_y_w_          = 0;
  int          tex_y_h_          = 0;
  int          tex_y_bpp_        = 0;
  int          tex_cb_w_         = 0;
  int          tex_cb_h_         = 0;
  int          tex_cb_bpp_       = 0;
  int          tex_cr_w_         = 0;
  int          tex_cr_h_         = 0;
  int          tex_cr_bpp_       = 0;

  // ── Zero-copy ring state ────────────────────────────────────────────────
  // Slot lifecycle: Free → (acquire, decode thread) InUse → (draw, main
  // thread) AwaitFence → (fence signalled, poll_events) Free.  InUse slots
  // whose frame never reaches the renderer go back to Free via
  // release_plane_buffers.  All GL calls stay on the main thread; the
  // decode thread only touches the mapped pointer and the atomics.
  static constexpr int kRingDepth = 3;
  enum SlotState : int { kSlotFree = 0, kSlotInUse = 1, kSlotAwaitFence = 2 };
  struct RingSlot {
    unsigned int     pbo    = 0;
    uint8_t*         mapped = nullptr;
    void*            fence  = nullptr;  // GLsync, opaque here
    std::atomic<int> state{kSlotFree};
  };
  RingSlot ring_[kRingDepth];
  // Geometry the ring was built for (main thread writes, decode thread
  // reads only when ring_ready_ is true — release/acquire ordering below).
  uint32_t ring_w_y_ = 0, ring_h_y_ = 0, ring_w_c_ = 0, ring_h_c_ = 0;
  int      ring_bpp_     = 0;
  size_t   ring_y_bytes_ = 0;
  size_t   ring_c_bytes_ = 0;
  std::atomic<bool> ring_ready_{false};
  // Pending build/rebuild request from the decode thread (guarded by
  // ring_req_mu_): zero w_y means "no request".
  std::mutex ring_req_mu_;
  uint32_t   req_w_y_ = 0, req_h_y_ = 0, req_w_c_ = 0, req_h_c_ = 0;
  int        req_bpp_ = 0;
  bool zero_copy_supported_ = false;  // ARB_buffer_storage present at init

  // Main-thread ring plumbing.
  void service_ring_request();  // called from poll_events()
  void poll_ring_fences();      // called from poll_events()
  void destroy_ring();          // shutdown / rebuild
};

}  // namespace open_htj2k::rtp_recv
