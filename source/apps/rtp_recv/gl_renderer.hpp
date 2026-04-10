// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Licensed under the same BSD 3-Clause terms as the rest of OpenHTJ2K.

#pragma once

// Minimal GLFW + OpenGL 2.1 fixed-function renderer.
//
// Holds a single GL_RGB8 2D texture whose storage is re-allocated only when
// the incoming frame dimensions change; every other frame goes through
// glTexSubImage2D.  Draws the texture into a letterboxed quad that fits the
// current window size while preserving aspect ratio.
//
// Uses legacy fixed-function GL (2.1 compat profile) to avoid pulling in a
// function-pointer loader such as glad or GLEW.  Core-profile upgrade with
// a GLSL shader pipeline is a v2 upgrade path when we need tone-mapping or
// post-processing.  All the calls used here (glBegin/glEnd, glOrtho,
// glTexImage2D, glTexSubImage2D, glViewport, glClear) are in the ABI libGL
// exports directly, so no wglGetProcAddress / glXGetProcAddress is needed.

#include <cstdint>

struct GLFWwindow;

namespace open_htj2k::rtp_recv {

class GlRenderer {
 public:
  GlRenderer() = default;
  ~GlRenderer();

  GlRenderer(const GlRenderer&)            = delete;
  GlRenderer& operator=(const GlRenderer&) = delete;

  // Open a window of the given logical size with the given title.  The
  // GL context becomes current on this thread.  Returns false on failure;
  // the caller should fall back to a headless mode.
  bool init(int window_w, int window_h, const char* title);

  // Tear down the window and GL context.  Safe to call multiple times.
  void shutdown();

  // True if the user clicked the close button or hit ESC.
  bool should_close() const;

  // Pump window events (keyboard / close).  Call once per frame.
  void poll_events();

  // Upload an 8-bit RGB image and draw it once.  Reallocates the texture
  // if (w,h) changed since the last call.  Does NOT own the pixel buffer;
  // the caller retains ownership and may reuse it after this returns.
  void upload_and_draw(const uint8_t* rgb, int w, int h);

 private:
  void ensure_texture(int w, int h);
  void draw_textured_quad(int fb_w, int fb_h, int tex_w, int tex_h);

  GLFWwindow*  window_ = nullptr;
  unsigned int tex_    = 0;  // GLuint; kept as unsigned int to avoid including GL headers here
  int          tex_w_  = 0;
  int          tex_h_  = 0;
};

}  // namespace open_htj2k::rtp_recv
