// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Licensed under the same BSD 3-Clause terms as the rest of OpenHTJ2K.

#include "gl_renderer.hpp"
#include "color_pipeline.hpp"
#include "gl_loader.hpp"
#include "ycbcr_rgb.hpp"

#include <GLFW/glfw3.h>
#include <GL/gl.h>

#include <cstdio>
#include <cstring>

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

// Fullscreen triangle strip.  Four vertices, each (x, y, u, v).  NDC
// positions -1..+1, tex coords 0..1 with v=0 at the top so the source
// image's row 0 lands at the top of the window.
constexpr float kFullscreenQuadVerts[4 * 4] = {
    // x     y     u     v
    -1.0f, +1.0f, 0.0f, 0.0f,  // top-left
    -1.0f, -1.0f, 0.0f, 1.0f,  // bottom-left
    +1.0f, +1.0f, 1.0f, 0.0f,  // top-right
    +1.0f, -1.0f, 1.0f, 1.0f,  // bottom-right
};

constexpr const char* kVertexShaderSrc = R"glsl(
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexCoord;
out vec2 vTexCoord;
void main() {
  gl_Position = vec4(aPos, 0.0, 1.0);
  vTexCoord   = aTexCoord;
}
)glsl";

constexpr const char* kFragmentShaderRgbSrc = R"glsl(
#version 330 core
in  vec2 vTexCoord;
out vec4 fragColor;
uniform sampler2D uTex;
void main() {
  fragColor = vec4(texture(uTex, vTexCoord).rgb, 1.0);
}
)glsl";

constexpr const char* kFragmentShaderYcbcrSrc = R"glsl(
#version 330 core
in  vec2 vTexCoord;
out vec4 fragColor;
uniform sampler2D uY;
uniform sampler2D uCb;
uniform sampler2D uCr;
uniform mat3      uMatrix;     // rgb = uMatrix * ((n - uBias) * uScale)
uniform vec3      uBias;       // per-plane bias in [0,1] after renormalization
uniform vec3      uScale;      // per-plane scale after bias subtraction
uniform vec3      uNormScale;  // per-plane renorm factor (1.0 for R8,
                               // 65535/((1<<depth)-1) for R16 with >8-bit src)
uniform int       uRgbMode;    // 0 = YCbCr -> RGB, 1 = treat planes as R/G/B

// HDR colour pipeline uniforms.  Defaults reproduce v0.12.0 behaviour
// up to a bounded re-encoding drift on SDR sources (see
// smoke_test_color_pipeline for the drift bound).
uniform int  uTransfer;        // 0=gamma2.2, 1=PQ, 2=HLG
uniform mat3 uGamutMatrix;     // identity or BT.2020 -> BT.709
uniform int  uDisplayEncoding; // 0=sRGB, 1=gamma2.2, 2=linear

// SMPTE ST 2084 PQ EOTF constants (BT.2100 Table 4).
const float kPqM1 = 0.1593017578125;
const float kPqM2 = 78.84375;
const float kPqC1 = 0.8359375;
const float kPqC2 = 18.8515625;
const float kPqC3 = 18.6875;

vec3 pq_to_linear(vec3 e) {
  vec3 v   = pow(max(e, vec3(0.0)), vec3(1.0 / kPqM2));
  vec3 num = max(v - vec3(kPqC1), vec3(0.0));
  vec3 den = vec3(kPqC2) - vec3(kPqC3) * v;
  return pow(num / den, vec3(1.0 / kPqM1));
}

// ITU-R BT.2100 HLG inverse OETF.  System OOTF treated as identity for
// display-referred SDR output; a tunable-gamma OOTF is a follow-up.
vec3 hlg_inverse(vec3 e) {
  const float a = 0.17883277;
  const float b = 0.28466892;
  const float c = 0.55991073;
  vec3 lo = (e * e) / 3.0;
  vec3 hi = (exp((e - vec3(c)) / a) + vec3(b)) / 12.0;
  return mix(lo, hi, vec3(greaterThan(e, vec3(0.5))));
}

vec3 apply_inverse_transfer(vec3 e) {
  if (uTransfer == 1) return pq_to_linear(e);
  if (uTransfer == 2) return hlg_inverse(e);
  return pow(max(e, vec3(0.0)), vec3(2.2));
}

// sRGB EOTF^-1 (piecewise).  IEC 61966-2-1.
vec3 linear_to_srgb(vec3 l) {
  vec3  lo  = l * 12.92;
  vec3  hi  = 1.055 * pow(l, vec3(1.0 / 2.4)) - 0.055;
  bvec3 cut = lessThanEqual(l, vec3(0.0031308));
  return mix(hi, lo, vec3(cut));
}

vec3 apply_display_encoding(vec3 l) {
  if (uDisplayEncoding == 1) return pow(max(l, vec3(0.0)), vec3(1.0 / 2.2));
  if (uDisplayEncoding == 2) return l;
  return linear_to_srgb(l);
}

void main() {
  vec3 s;
  s.x = texture(uY,  vTexCoord).r;
  s.y = texture(uCb, vTexCoord).r;
  s.z = texture(uCr, vTexCoord).r;
  // For R8 textures uNormScale is (1,1,1) and this is a no-op; for R16
  // textures uNormScale restores the source's native [0,1] range before
  // bias/scale math runs, so the rest of the shader is depth-agnostic.
  vec3 n = s * uNormScale;
  vec3 rgb_nl;
  if (uRgbMode == 1) {
    rgb_nl = n;
  } else {
    rgb_nl = uMatrix * ((n - uBias) * uScale);
  }
  // Clamp post-matrix before the inverse-transfer stage.  Out-of-range
  // values on a narrow-range BT.709 source would otherwise feed a pow()
  // with negative base.
  rgb_nl      = clamp(rgb_nl, 0.0, 1.0);
  vec3 lin_s  = apply_inverse_transfer(rgb_nl);
  vec3 lin_d  = uGamutMatrix * lin_s;
  // Stage 3: hard clip.  BT.2390 EETF soft-knee is a follow-up PR.
  lin_d       = clamp(lin_d, 0.0, 1.0);
  vec3 out_nl = apply_display_encoding(lin_d);
  fragColor   = vec4(clamp(out_nl, 0.0, 1.0), 1.0);
}
)glsl";

GLuint compile_one_shader(GLenum kind, const char* src, const char* label) {
  GLuint sh = gl::CreateShader(kind);
  gl::ShaderSource(sh, 1, &src, nullptr);
  gl::CompileShader(sh);
  GLint ok = GL_FALSE;
  gl::GetShaderiv(sh, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[2048] = {0};
    GLsizei len    = 0;
    gl::GetShaderInfoLog(sh, sizeof(log) - 1, &len, log);
    std::fprintf(stderr, "gl_renderer: %s compile failed:\n%s\n", label, log);
    gl::DeleteShader(sh);
    return 0;
  }
  return sh;
}

GLuint link_program(GLuint vs, GLuint fs, const char* label) {
  GLuint prog = gl::CreateProgram();
  gl::AttachShader(prog, vs);
  gl::AttachShader(prog, fs);
  gl::LinkProgram(prog);
  GLint ok = GL_FALSE;
  gl::GetProgramiv(prog, GL_LINK_STATUS, &ok);
  if (!ok) {
    char log[2048] = {0};
    GLsizei len    = 0;
    gl::GetProgramInfoLog(prog, sizeof(log) - 1, &len, log);
    std::fprintf(stderr, "gl_renderer: %s link failed:\n%s\n", label, log);
    gl::DeleteProgram(prog);
    return 0;
  }
  return prog;
}

}  // namespace

GlRenderer::~GlRenderer() { shutdown(); }

bool GlRenderer::init(int window_w, int window_h, const char* title, bool vsync) {
  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit()) return false;

  // GL 3.3 core profile — needed for GLSL 330 shaders, VAOs, and the
  // function loader.  Forward-compat keeps Mesa from falling back to
  // 3.1 if the driver is in a weird state.
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);

  window_ = glfwCreateWindow(window_w, window_h, title, nullptr, nullptr);
  if (!window_) {
    std::fprintf(stderr, "gl_renderer: glfwCreateWindow failed (GL 3.3 core)\n");
    glfwTerminate();
    return false;
  }
  glfwMakeContextCurrent(window_);
  glfwSwapInterval(vsync ? 1 : 0);
  glfwSetKeyCallback(window_, key_callback);

  if (!gl::load_functions()) {
    std::fprintf(stderr, "gl_renderer: gl_loader failed to resolve GL 3.3 symbols\n");
    shutdown();
    return false;
  }

  if (!compile_shader_programs()) {
    shutdown();
    return false;
  }

  // Fullscreen-quad VAO/VBO.  Never re-uploaded.
  GLuint vao = 0;
  GLuint vbo = 0;
  gl::GenVertexArrays(1, &vao);
  gl::GenBuffers(1, &vbo);
  vao_ = vao;
  vbo_ = vbo;
  gl::BindVertexArray(vao_);
  gl::BindBuffer(GL_ARRAY_BUFFER, vbo_);
  gl::BufferData(GL_ARRAY_BUFFER, sizeof(kFullscreenQuadVerts), kFullscreenQuadVerts,
                 GL_STATIC_DRAW);
  gl::EnableVertexAttribArray(0);
  gl::VertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                          reinterpret_cast<void*>(0));
  gl::EnableVertexAttribArray(1);
  gl::VertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                          reinterpret_cast<void*>(2 * sizeof(float)));
  gl::BindVertexArray(0);

  return true;
}

bool GlRenderer::compile_shader_programs() {
  GLuint vs = compile_one_shader(GL_VERTEX_SHADER, kVertexShaderSrc, "vertex shader");
  if (!vs) return false;

  GLuint fs_rgb = compile_one_shader(GL_FRAGMENT_SHADER, kFragmentShaderRgbSrc,
                                     "rgb fragment shader");
  if (!fs_rgb) {
    gl::DeleteShader(vs);
    return false;
  }
  GLuint prog_rgb = link_program(vs, fs_rgb, "rgb program");
  gl::DeleteShader(fs_rgb);
  if (!prog_rgb) {
    gl::DeleteShader(vs);
    return false;
  }
  prog_rgb_  = prog_rgb;
  u_rgb_tex_ = gl::GetUniformLocation(prog_rgb_, "uTex");

  GLuint fs_yc = compile_one_shader(GL_FRAGMENT_SHADER, kFragmentShaderYcbcrSrc,
                                    "ycbcr fragment shader");
  if (!fs_yc) {
    gl::DeleteShader(vs);
    gl::DeleteProgram(prog_rgb_);
    prog_rgb_ = 0;
    return false;
  }
  GLuint prog_yc = link_program(vs, fs_yc, "ycbcr program");
  gl::DeleteShader(vs);
  gl::DeleteShader(fs_yc);
  if (!prog_yc) {
    gl::DeleteProgram(prog_rgb_);
    prog_rgb_ = 0;
    return false;
  }
  prog_ycbcr_      = prog_yc;
  u_yc_y_tex_      = gl::GetUniformLocation(prog_ycbcr_, "uY");
  u_yc_cb_tex_     = gl::GetUniformLocation(prog_ycbcr_, "uCb");
  u_yc_cr_tex_     = gl::GetUniformLocation(prog_ycbcr_, "uCr");
  u_yc_matrix_     = gl::GetUniformLocation(prog_ycbcr_, "uMatrix");
  u_yc_bias_       = gl::GetUniformLocation(prog_ycbcr_, "uBias");
  u_yc_scale_      = gl::GetUniformLocation(prog_ycbcr_, "uScale");
  u_yc_norm_scale_ = gl::GetUniformLocation(prog_ycbcr_, "uNormScale");
  u_yc_rgb_mode_   = gl::GetUniformLocation(prog_ycbcr_, "uRgbMode");
  u_yc_transfer_   = gl::GetUniformLocation(prog_ycbcr_, "uTransfer");
  u_yc_gamut_      = gl::GetUniformLocation(prog_ycbcr_, "uGamutMatrix");
  u_yc_display_enc_ = gl::GetUniformLocation(prog_ycbcr_, "uDisplayEncoding");

  // Drain any residual error state from the compile/link sync point so
  // later per-frame glGetError checks (if added) don't see ghosts.
  check_gl_error("compile_shader_programs");
  return true;
}

void GlRenderer::shutdown() {
  if (tex_rgb_ != 0) {
    const GLuint t = tex_rgb_;
    glDeleteTextures(1, &t);
    tex_rgb_ = 0;
  }
  if (tex_y_ != 0) {
    const GLuint t = tex_y_;
    glDeleteTextures(1, &t);
    tex_y_ = 0;
  }
  if (tex_cb_ != 0) {
    const GLuint t = tex_cb_;
    glDeleteTextures(1, &t);
    tex_cb_ = 0;
  }
  if (tex_cr_ != 0) {
    const GLuint t = tex_cr_;
    glDeleteTextures(1, &t);
    tex_cr_ = 0;
  }
  if (vbo_ != 0 && gl::DeleteBuffers != nullptr) {
    const GLuint b = vbo_;
    gl::DeleteBuffers(1, &b);
    vbo_ = 0;
  }
  if (vao_ != 0 && gl::DeleteVertexArrays != nullptr) {
    const GLuint a = vao_;
    gl::DeleteVertexArrays(1, &a);
    vao_ = 0;
  }
  if (prog_rgb_ != 0 && gl::DeleteProgram != nullptr) {
    gl::DeleteProgram(prog_rgb_);
    prog_rgb_ = 0;
  }
  if (prog_ycbcr_ != 0 && gl::DeleteProgram != nullptr) {
    gl::DeleteProgram(prog_ycbcr_);
    prog_ycbcr_ = 0;
  }
  if (window_) {
    glfwDestroyWindow(window_);
    window_ = nullptr;
    glfwTerminate();
  }
  tex_rgb_w_  = 0;
  tex_rgb_h_  = 0;
  tex_y_w_    = 0;
  tex_y_h_    = 0;
  tex_y_bpp_  = 0;
  tex_cb_w_   = 0;
  tex_cb_h_   = 0;
  tex_cb_bpp_ = 0;
  tex_cr_w_   = 0;
  tex_cr_h_   = 0;
  tex_cr_bpp_ = 0;
}

bool GlRenderer::check_gl_error(const char* context) const {
  bool clean = true;
  for (GLenum err = glGetError(); err != GL_NO_ERROR; err = glGetError()) {
    clean = false;
    std::fprintf(stderr, "gl_renderer: GL error 0x%04x at %s\n",
                 static_cast<unsigned>(err), context);
  }
  return clean;
}

bool GlRenderer::should_close() const {
  return window_ ? glfwWindowShouldClose(window_) != 0 : true;
}

void GlRenderer::poll_events() {
  if (window_) glfwPollEvents();
}

bool GlRenderer::ensure_rgb_texture(int w, int h) {
  if (tex_rgb_ == 0) {
    GLuint t = 0;
    glGenTextures(1, &t);
    if (t == 0) {
      std::fprintf(stderr, "gl_renderer: glGenTextures returned 0 (out of memory?)\n");
      check_gl_error("glGenTextures(rgb)");
      return false;
    }
    tex_rgb_ = t;
    glBindTexture(GL_TEXTURE_2D, tex_rgb_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  } else {
    glBindTexture(GL_TEXTURE_2D, tex_rgb_);
  }
  if (w != tex_rgb_w_ || h != tex_rgb_h_) {
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    if (!check_gl_error("glTexImage2D(rgb initial alloc)")) return false;
    tex_rgb_w_ = w;
    tex_rgb_h_ = h;
  }
  return true;
}

bool GlRenderer::ensure_planar_textures(int w_y, int h_y, int w_c, int h_c, int bpp) {
  // A freshly-created texture has no backing store, so the glTexImage2D
  // step must run even when the cached (cur_w, cur_h) already match the
  // requested size.  The earlier version of this function shared Cb/Cr
  // tracking variables, which caused the Cr texture to skip allocation
  // on the first frame (Cb allocation updated the shared counters) and
  // the shader then sampled an incomplete texture — usually reading as
  // all zeros, producing a strongly red-shifted output.  Each texture
  // now owns its (w, h, bpp) triple so a depth swap on one plane cannot
  // silently skip reallocation on the others either.
  const GLint       internal = (bpp == 2) ? GL_R16 : GL_R8;
  const GLenum      type     = (bpp == 2) ? GL_UNSIGNED_SHORT : GL_UNSIGNED_BYTE;
  const char*       label_bpp = (bpp == 2) ? "R16" : "R8";
  auto ensure_one = [&](const char* label, GLuint& tex, int& cur_w, int& cur_h,
                        int& cur_bpp, int w, int h) -> bool {
    const bool fresh = (tex == 0);
    if (fresh) {
      glGenTextures(1, &tex);
      if (tex == 0) {
        std::fprintf(stderr, "gl_renderer: glGenTextures(%s) returned 0 (out of memory?)\n",
                     label);
        check_gl_error("glGenTextures(planar)");
        return false;
      }
      glBindTexture(GL_TEXTURE_2D, tex);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    } else {
      glBindTexture(GL_TEXTURE_2D, tex);
    }
    if (fresh || w != cur_w || h != cur_h || bpp != cur_bpp) {
      // GL_UNPACK_ALIGNMENT matters for row padding: R8 rows are 1-byte
      // aligned, R16 rows are naturally 2-byte aligned.
      glPixelStorei(GL_UNPACK_ALIGNMENT, bpp);
      glTexImage2D(GL_TEXTURE_2D, 0, internal, w, h, 0, GL_RED, type, nullptr);
      if (!check_gl_error("glTexImage2D(planar initial alloc)")) {
        std::fprintf(stderr, "gl_renderer: ensure_planar_textures(%s, %s) alloc failed\n",
                     label, label_bpp);
        return false;
      }
      cur_w   = w;
      cur_h   = h;
      cur_bpp = bpp;
    }
    return true;
  };
  if (!ensure_one("Y",  tex_y_,  tex_y_w_,  tex_y_h_,  tex_y_bpp_,  w_y, h_y)) return false;
  if (!ensure_one("Cb", tex_cb_, tex_cb_w_, tex_cb_h_, tex_cb_bpp_, w_c, h_c)) return false;
  if (!ensure_one("Cr", tex_cr_, tex_cr_w_, tex_cr_h_, tex_cr_bpp_, w_c, h_c)) return false;
  return true;
}

void GlRenderer::draw_fullscreen_quad(int fb_w, int fb_h, int content_w, int content_h) {
  // Letterbox / pillarbox so the aspect ratio of the source image is
  // preserved.  Compute the image rectangle in framebuffer pixels,
  // clear the full framebuffer, then set the GL viewport to the image
  // rect so the fullscreen triangle strip fills exactly that region.
  const double fb_aspect  = static_cast<double>(fb_w) / static_cast<double>(fb_h);
  const double img_aspect = static_cast<double>(content_w) / static_cast<double>(content_h);
  int draw_w = fb_w;
  int draw_h = fb_h;
  if (img_aspect > fb_aspect) {
    draw_h = static_cast<int>(fb_w / img_aspect);
  } else {
    draw_w = static_cast<int>(fb_h * img_aspect);
  }
  const int x0 = (fb_w - draw_w) / 2;
  const int y0 = (fb_h - draw_h) / 2;

  glViewport(0, 0, fb_w, fb_h);
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);

  glViewport(x0, y0, draw_w, draw_h);
  gl::BindVertexArray(vao_);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  gl::BindVertexArray(0);
}

void GlRenderer::upload_and_draw(const uint8_t* rgb, int w, int h) {
  if (!window_ || w <= 0 || h <= 0) return;

  ensure_rgb_texture(w, h);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, rgb);

  gl::UseProgram(prog_rgb_);
  gl::ActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, tex_rgb_);
  gl::Uniform1i(u_rgb_tex_, 0);

  int fb_w = 0;
  int fb_h = 0;
  glfwGetFramebufferSize(window_, &fb_w, &fb_h);
  draw_fullscreen_quad(fb_w, fb_h, w, h);

  glfwSwapBuffers(window_);
}

void GlRenderer::upload_planar_and_draw(const uint8_t* y_plane, const uint8_t* cb_plane,
                                        const uint8_t* cr_plane, int w_y, int h_y, int w_c,
                                        int h_c, const ycbcr_coefficients* coeffs,
                                        bool components_are_rgb,
                                        const ColorPipelineParams& pipeline) {
  if (!window_ || w_y <= 0 || h_y <= 0 || w_c <= 0 || h_c <= 0) return;

  if (!ensure_planar_textures(w_y, h_y, w_c, h_c, /*bpp=*/1)) return;

  // Upload Y.
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glBindTexture(GL_TEXTURE_2D, tex_y_);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w_y, h_y, GL_RED, GL_UNSIGNED_BYTE, y_plane);
  // Upload Cb.
  glBindTexture(GL_TEXTURE_2D, tex_cb_);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w_c, h_c, GL_RED, GL_UNSIGNED_BYTE, cb_plane);
  // Upload Cr.
  glBindTexture(GL_TEXTURE_2D, tex_cr_);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w_c, h_c, GL_RED, GL_UNSIGNED_BYTE, cr_plane);

  // 8-bit source: the texture value already lives in the sample's native
  // [0, 1] range, so uNormScale is the identity.
  const float norm_scale[3] = {1.0f, 1.0f, 1.0f};
  draw_ycbcr_program(w_y, h_y, coeffs, components_are_rgb, norm_scale, pipeline);
}

void GlRenderer::upload_planar_16_and_draw(const uint16_t* y_plane, const uint16_t* cb_plane,
                                           const uint16_t* cr_plane, int w_y, int h_y,
                                           int w_c, int h_c, int bit_depth,
                                           const ycbcr_coefficients* coeffs,
                                           bool components_are_rgb,
                                           const ColorPipelineParams& pipeline) {
  if (!window_ || w_y <= 0 || h_y <= 0 || w_c <= 0 || h_c <= 0) return;
  if (bit_depth < 9 || bit_depth > 16) return;  // caller routes 8-bit to the u8 path

  if (!ensure_planar_textures(w_y, h_y, w_c, h_c, /*bpp=*/2)) return;

  // R16 row pitch is 2-byte aligned.
  glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
  glBindTexture(GL_TEXTURE_2D, tex_y_);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w_y, h_y, GL_RED, GL_UNSIGNED_SHORT, y_plane);
  glBindTexture(GL_TEXTURE_2D, tex_cb_);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w_c, h_c, GL_RED, GL_UNSIGNED_SHORT, cb_plane);
  glBindTexture(GL_TEXTURE_2D, tex_cr_);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w_c, h_c, GL_RED, GL_UNSIGNED_SHORT, cr_plane);

  // GL_R16 reads back as (sample / 65535).  Multiply by this factor in the
  // shader to restore the sample's native [0, 1] range so the bias/scale/
  // matrix math is depth-agnostic.
  const float native_max    = static_cast<float>((1 << bit_depth) - 1);
  const float k             = 65535.0f / native_max;
  const float norm_scale[3] = {k, k, k};
  draw_ycbcr_program(w_y, h_y, coeffs, components_are_rgb, norm_scale, pipeline);
}

void GlRenderer::draw_ycbcr_program(int w_y, int h_y, const ycbcr_coefficients* coeffs,
                                    bool components_are_rgb, const float* norm_scale,
                                    const ColorPipelineParams& pipeline) {
  gl::UseProgram(prog_ycbcr_);
  gl::ActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, tex_y_);
  gl::Uniform1i(u_yc_y_tex_, 0);
  gl::ActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, tex_cb_);
  gl::Uniform1i(u_yc_cb_tex_, 1);
  gl::ActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, tex_cr_);
  gl::Uniform1i(u_yc_cr_tex_, 2);

  gl::Uniform3fv(u_yc_norm_scale_, 1, norm_scale);
  gl::Uniform1i(u_yc_rgb_mode_, components_are_rgb ? 1 : 0);

  // HDR pipeline uniforms.  The defaults (gamma2.2 / identity / sRGB)
  // preserve v0.12.0 behaviour modulo a bounded re-encoding drift on
  // SDR BT.709 sources (quantified in smoke_test_color_pipeline).
  gl::Uniform1i(u_yc_transfer_, pipeline.transfer);
  gl::UniformMatrix3fv(u_yc_gamut_, 1, GL_FALSE, pipeline.gamut_matrix);
  gl::Uniform1i(u_yc_display_enc_, pipeline.display_encoding);

  if (!components_are_rgb && coeffs != nullptr) {
    // Column-major: mat3(col0, col1, col2) where
    //   col0 (Y) = (1, 1, 1)
    //   col1 (Cb) = (0, -cb_to_g, cb_to_b)
    //   col2 (Cr) = (cr_to_r, -cr_to_g, 0)
    // so rgb = uMatrix * vec3(y_n, cb_n, cr_n) computes:
    //   R = y_n + cr_to_r * cr_n
    //   G = y_n - cb_to_g * cb_n - cr_to_g * cr_n
    //   B = y_n + cb_to_b * cb_n
    const float mat[9] = {
        1.0f,           1.0f,            1.0f,
        0.0f,          -coeffs->cb_to_g, coeffs->cb_to_b,
        coeffs->cr_to_r, -coeffs->cr_to_g, 0.0f,
    };
    gl::UniformMatrix3fv(u_yc_matrix_, 1, GL_FALSE, mat);

    // Bias / scale in the sample's native [0, 1] range.  uNormScale above
    // renormalized before this step, so these constants are depth-
    // independent -- the 8-bit narrow-range offsets read as 16/255 and
    // 128/255 regardless of whether the wire depth is 8, 10, 12, or 16.
    float bias[3];
    float scale[3];
    if (coeffs->narrow_range) {
      bias[0]  = 16.0f / 255.0f;
      bias[1]  = 128.0f / 255.0f;
      bias[2]  = 128.0f / 255.0f;
      scale[0] = 255.0f / 219.0f;
      scale[1] = 255.0f / 224.0f;
      scale[2] = 255.0f / 224.0f;
    } else {
      bias[0]  = 0.0f;
      bias[1]  = 0.5f;
      bias[2]  = 0.5f;
      scale[0] = 1.0f;
      scale[1] = 1.0f;
      scale[2] = 1.0f;
    }
    gl::Uniform3fv(u_yc_bias_, 1, bias);
    gl::Uniform3fv(u_yc_scale_, 1, scale);
  }

  int fb_w = 0;
  int fb_h = 0;
  glfwGetFramebufferSize(window_, &fb_w, &fb_h);
  draw_fullscreen_quad(fb_w, fb_h, w_y, h_y);

  glfwSwapBuffers(window_);
}

}  // namespace open_htj2k::rtp_recv
