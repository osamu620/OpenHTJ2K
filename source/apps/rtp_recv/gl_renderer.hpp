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

#include <cstdint>

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

  // Planar RGB passthrough program — no matrix/bias/scale, just HDR pipeline.
  unsigned int prog_planar_rgb_       = 0;
  int          u_pr_y_tex_            = -1;
  int          u_pr_cb_tex_           = -1;
  int          u_pr_cr_tex_           = -1;
  int          u_pr_norm_scale_       = -1;
  int          u_pr_transfer_         = -1;
  int          u_pr_gamut_            = -1;
  int          u_pr_display_enc_      = -1;
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
};

}  // namespace open_htj2k::rtp_recv
