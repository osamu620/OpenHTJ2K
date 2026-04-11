// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Licensed under the same BSD 3-Clause terms as the rest of OpenHTJ2K.

#pragma once

// GLFW + OpenGL 3.3 core renderer for the RFC 9828 receiver.
//
// Two draw paths share one GL context, one VAO, and the same vertex shader:
//
//   - upload_and_draw(rgb, w, h)               → RGB8 texture + passthrough
//                                                fragment shader.  Used by
//                                                the CPU fallback path
//                                                (decode_to_rgb_buffer) for
//                                                headless / GL-limited envs.
//   - upload_planar_and_draw(y, cb, cr, ...)  → three R8 textures + YCbCr
//                                                matrix fragment shader.
//                                                Used by the default shader
//                                                path so the CPU only shifts
//                                                samples to 8-bit and the
//                                                color conversion happens on
//                                                the GPU.
//
// init() tries to create a GL 3.3 core context.  Returns false if glfwInit
// or context creation fails; the caller should log and fall back to
// --no-render.  load_functions() resolves GL >=2.0 entry points via
// glfwGetProcAddress in gl_loader.cpp.

#include <cstdint>

struct GLFWwindow;

namespace open_htj2k::rtp_recv {

struct ycbcr_coefficients;  // from ycbcr_rgb.hpp

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

  // Shader path: upload three 8-bit planar samples (Y and two chroma
  // planes, already right-shifted to 8 bits on the CPU side) and draw
  // them through the YCbCr→RGB fragment shader.  Cb/Cr widths/heights
  // may be smaller than luma for 4:2:2 / 4:2:0; GL_LINEAR + clamp-to-
  // edge handles the chroma upsample.  `coeffs` selects BT.601/709 and
  // full/narrow range; the matrix + normalization bias/scale are baked
  // into shader uniforms on each call.  `components_are_rgb` skips the
  // matrix entirely and treats the three planes as R/G/B.
  void upload_planar_and_draw(const uint8_t* y_plane, const uint8_t* cb_plane,
                              const uint8_t* cr_plane, int w_y, int h_y, int w_c, int h_c,
                              const ycbcr_coefficients* coeffs, bool components_are_rgb);

 private:
  bool compile_shader_programs();
  bool ensure_rgb_texture(int w, int h);
  bool ensure_planar_textures(int w_y, int h_y, int w_c, int h_c);
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

  // YCbCr planar program (shader path).
  unsigned int prog_ycbcr_       = 0;
  int          u_yc_y_tex_       = -1;
  int          u_yc_cb_tex_      = -1;
  int          u_yc_cr_tex_      = -1;
  int          u_yc_matrix_      = -1;
  int          u_yc_bias_        = -1;
  int          u_yc_scale_       = -1;
  int          u_yc_rgb_mode_    = -1;  // int: 0=ycbcr, 1=passthrough RGB components
  int          u_yc_viewport_    = -1;
  unsigned int tex_y_            = 0;
  unsigned int tex_cb_           = 0;
  unsigned int tex_cr_           = 0;
  int          tex_y_w_          = 0;
  int          tex_y_h_          = 0;
  int          tex_cb_w_         = 0;
  int          tex_cb_h_         = 0;
  int          tex_cr_w_         = 0;
  int          tex_cr_h_         = 0;
};

}  // namespace open_htj2k::rtp_recv
