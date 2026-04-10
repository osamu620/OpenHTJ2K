// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Licensed under the same BSD 3-Clause terms as the rest of OpenHTJ2K.

#include "gl_renderer.hpp"

// GLFW must be included before GL so its GL header gating macros take effect.
#include <GLFW/glfw3.h>
#include <GL/gl.h>

#include <cstdio>

namespace open_htj2k::rtp_recv {

namespace {
void glfw_error_callback(int code, const char* description) {
  std::fprintf(stderr, "GLFW error %d: %s\n", code, description ? description : "(null)");
}

void key_callback(GLFWwindow* win, int key, int /*scan*/, int action, int /*mods*/) {
  if (action == GLFW_PRESS && (key == GLFW_KEY_ESCAPE || key == GLFW_KEY_Q)) {
    glfwSetWindowShouldClose(win, GLFW_TRUE);
  }
}
}  // namespace

GlRenderer::~GlRenderer() { shutdown(); }

bool GlRenderer::init(int window_w, int window_h, const char* title) {
  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit()) return false;

  // Request a 2.1 compatibility context — the fixed-function path we use
  // below is not available in core profile, and 2.1 compat is supported on
  // every desktop driver we care about (Mesa, NVIDIA, Apple legacy).
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);

  window_ = glfwCreateWindow(window_w, window_h, title, nullptr, nullptr);
  if (!window_) {
    glfwTerminate();
    return false;
  }
  glfwMakeContextCurrent(window_);
  glfwSwapInterval(1);  // vsync
  glfwSetKeyCallback(window_, key_callback);
  return true;
}

void GlRenderer::shutdown() {
  if (tex_ != 0) {
    const GLuint t = tex_;
    glDeleteTextures(1, &t);
    tex_ = 0;
  }
  if (window_) {
    glfwDestroyWindow(window_);
    window_ = nullptr;
    glfwTerminate();
  }
  tex_w_ = 0;
  tex_h_ = 0;
}

bool GlRenderer::should_close() const {
  return window_ ? glfwWindowShouldClose(window_) != 0 : true;
}

void GlRenderer::poll_events() {
  if (window_) glfwPollEvents();
}

void GlRenderer::ensure_texture(int w, int h) {
  if (tex_ == 0) {
    GLuint t = 0;
    glGenTextures(1, &t);
    tex_ = t;
    glBindTexture(GL_TEXTURE_2D, tex_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  } else {
    glBindTexture(GL_TEXTURE_2D, tex_);
  }
  if (w != tex_w_ || h != tex_h_) {
    // (re)allocate storage
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    tex_w_ = w;
    tex_h_ = h;
  }
}

void GlRenderer::draw_textured_quad(int fb_w, int fb_h, int tex_w, int tex_h) {
  // Letterbox / pillarbox so the aspect ratio is preserved.
  const double fb_aspect  = static_cast<double>(fb_w) / static_cast<double>(fb_h);
  const double tex_aspect = static_cast<double>(tex_w) / static_cast<double>(tex_h);
  int draw_w = fb_w;
  int draw_h = fb_h;
  if (tex_aspect > fb_aspect) {
    // texture is wider than window → letterbox (bars on top/bottom)
    draw_h = static_cast<int>(fb_w / tex_aspect);
  } else {
    // texture is taller than window → pillarbox (bars on sides)
    draw_w = static_cast<int>(fb_h * tex_aspect);
  }
  const int x0 = (fb_w - draw_w) / 2;
  const int y0 = (fb_h - draw_h) / 2;
  const int x1 = x0 + draw_w;
  const int y1 = y0 + draw_h;

  glViewport(0, 0, fb_w, fb_h);
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  // Pixel-space ortho, y-flipped so (0,0) is top-left to match image coords.
  glOrtho(0.0, fb_w, fb_h, 0.0, -1.0, 1.0);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, tex_);
  glColor3f(1.0f, 1.0f, 1.0f);

  glBegin(GL_QUADS);
  glTexCoord2f(0.0f, 0.0f); glVertex2i(x0, y0);
  glTexCoord2f(1.0f, 0.0f); glVertex2i(x1, y0);
  glTexCoord2f(1.0f, 1.0f); glVertex2i(x1, y1);
  glTexCoord2f(0.0f, 1.0f); glVertex2i(x0, y1);
  glEnd();

  glDisable(GL_TEXTURE_2D);
}

void GlRenderer::upload_and_draw(const uint8_t* rgb, int w, int h) {
  if (!window_ || w <= 0 || h <= 0) return;

  ensure_texture(w, h);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, rgb);

  int fb_w = 0;
  int fb_h = 0;
  glfwGetFramebufferSize(window_, &fb_w, &fb_h);
  draw_textured_quad(fb_w, fb_h, w, h);

  glfwSwapBuffers(window_);
}

}  // namespace open_htj2k::rtp_recv
