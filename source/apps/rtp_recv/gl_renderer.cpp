// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Licensed under the same BSD 3-Clause terms as the rest of OpenHTJ2K.

#include "gl_renderer.hpp"
#include "color_pipeline.hpp"
#include "gl_loader.hpp"
#include "ycbcr_rgb.hpp"

#include <GLFW/glfw3.h>

#include <cstdio>
#include <cstdlib>
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

// Shared GLSL helper functions used by both the YCbCr and planar-RGB
// fragment shaders.  Concatenated with each shader's own preamble + main()
// via glShaderSource's array-of-strings interface.
constexpr const char* kFragmentPlanarHelpers = R"glsl(
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

vec3 linear_to_pq(vec3 l) {
  vec3 v = pow(max(l, vec3(0.0)), vec3(kPqM1));
  return pow((vec3(kPqC1) + kPqC2 * v) / (vec3(1.0) + kPqC3 * v), vec3(kPqM2));
}

vec3 hlg_inverse(vec3 e) {
  const float a = 0.17883277;
  const float b = 0.28466892;
  const float c = 0.55991073;
  vec3 lo = (e * e) / 3.0;
  vec3 hi = (exp((e - vec3(c)) / a) + vec3(b)) / 12.0;
  return mix(lo, hi, vec3(greaterThan(e, vec3(0.5))));
}

// ITU-R BT.2390-9 EETF, per channel.  Input: linear light, 1.0 == 10 000
// nits.  Output: display-relative linear in [0, 1] where 1.0 == 203 nits
// (BT.2408 reference white).  uTmSrcPq / uTmKs / uTmMaxLum are
// precomputed on the CPU (compute_bt2390_params in color_pipeline.hpp,
// which also documents the math).
vec3 bt2390_eetf(vec3 lin) {
  const float kDstLinear = 0.0203;  // 203 / 10000
  vec3 e1 = min(linear_to_pq(lin) / uTmSrcPq, vec3(1.0));
  vec3 t  = (e1 - vec3(uTmKs)) / (1.0 - uTmKs);
  vec3 t2 = t * t;
  vec3 t3 = t2 * t;
  vec3 p  = (2.0 * t3 - 3.0 * t2 + 1.0) * uTmKs + (t3 - 2.0 * t2 + t) * (1.0 - uTmKs)
           + (-2.0 * t3 + 3.0 * t2) * uTmMaxLum;
  vec3 e2 = mix(p, e1, vec3(lessThan(e1, vec3(uTmKs))));
  return clamp(pq_to_linear(e2 * uTmSrcPq) / kDstLinear, 0.0, 1.0);
}

vec3 apply_tonemap(vec3 lin) {
  if (uTonemap == 1) return bt2390_eetf(lin);
  return clamp(lin, 0.0, 1.0);
}

vec3 apply_inverse_transfer(vec3 e) {
  if (uTransfer == 1) return pq_to_linear(e);
  if (uTransfer == 2) return hlg_inverse(e);
  return pow(max(e, vec3(0.0)), vec3(2.2));
}

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
)glsl";

// YCbCr planar → RGB conversion.  Always applies matrix + bias + scale;
// no uRgbMode branch (the RGB-passthrough case uses prog_planar_rgb_).
constexpr const char* kFragmentShaderYcbcrSrc = R"glsl(
#version 330 core
in  vec2 vTexCoord;
out vec4 fragColor;
uniform sampler2D uY;
uniform sampler2D uCb;
uniform sampler2D uCr;
uniform mat3      uMatrix;
uniform vec3      uBias;
uniform vec3      uScale;
uniform vec3      uNormScale;
uniform int  uTransfer;
uniform mat3 uGamutMatrix;
uniform int  uDisplayEncoding;
uniform int   uTonemap;
uniform float uTmSrcPq;
uniform float uTmKs;
uniform float uTmMaxLum;
)glsl";

constexpr const char* kFragmentShaderYcbcrMain = R"glsl(
void main() {
  vec3 s;
  s.x = texture(uY,  vTexCoord).r;
  s.y = texture(uCb, vTexCoord).r;
  s.z = texture(uCr, vTexCoord).r;
  vec3 n = s * uNormScale;
  vec3 rgb_nl = uMatrix * ((n - uBias) * uScale);
  rgb_nl      = clamp(rgb_nl, 0.0, 1.0);
  vec3 lin_s  = apply_inverse_transfer(rgb_nl);
  vec3 lin_d  = uGamutMatrix * lin_s;
  lin_d       = apply_tonemap(lin_d);
  vec3 out_nl = apply_display_encoding(lin_d);
  fragColor   = vec4(clamp(out_nl, 0.0, 1.0), 1.0);
}
)glsl";

// Planar RGB passthrough: three R8/R16 planes interpreted directly as
// R, G, B.  No matrix/bias/scale — just renormalization + HDR pipeline.
constexpr const char* kFragmentShaderPlanarRgbSrc = R"glsl(
#version 330 core
in  vec2 vTexCoord;
out vec4 fragColor;
uniform sampler2D uY;
uniform sampler2D uCb;
uniform sampler2D uCr;
uniform vec3      uNormScale;
uniform int  uTransfer;
uniform mat3 uGamutMatrix;
uniform int  uDisplayEncoding;
uniform int   uTonemap;
uniform float uTmSrcPq;
uniform float uTmKs;
uniform float uTmMaxLum;
)glsl";

constexpr const char* kFragmentShaderPlanarRgbMain = R"glsl(
void main() {
  vec3 s;
  s.x = texture(uY,  vTexCoord).r;
  s.y = texture(uCb, vTexCoord).r;
  s.z = texture(uCr, vTexCoord).r;
  vec3 rgb_nl = s * uNormScale;
  rgb_nl      = clamp(rgb_nl, 0.0, 1.0);
  vec3 lin_s  = apply_inverse_transfer(rgb_nl);
  vec3 lin_d  = uGamutMatrix * lin_s;
  lin_d       = apply_tonemap(lin_d);
  vec3 out_nl = apply_display_encoding(lin_d);
  fragColor   = vec4(clamp(out_nl, 0.0, 1.0), 1.0);
}
)glsl";

bool has_gl_extension(const char* name) {
  if (gl::GetStringi == nullptr) return false;
  GLint n = 0;
  glGetIntegerv(GL_NUM_EXTENSIONS, &n);
  for (GLint i = 0; i < n; ++i) {
    const char* e =
        reinterpret_cast<const char*>(gl::GetStringi(GL_EXTENSIONS, static_cast<GLuint>(i)));
    if (e != nullptr && std::strcmp(e, name) == 0) return true;
  }
  return false;
}

GLuint compile_one_shader(GLenum kind, const char* const* srcs, int count, const char* label) {
  GLuint sh = gl::CreateShader(kind);
  gl::ShaderSource(sh, count, srcs, nullptr);
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

  // Optional zero-copy upload capability (ARB_buffer_storage).  Absence
  // is normal — macOS GL stops at 4.1 — and only means uploads go through
  // the plane-vector path instead of decoder-written persistent buffers.
  // OPENHTJ2K_GL_NO_ZEROCOPY=1 forces the vector path; kept as a runtime
  // escape hatch and for single-binary A/B measurements.
  const char* no_zc    = std::getenv("OPENHTJ2K_GL_NO_ZEROCOPY");
  const bool  disabled = (no_zc != nullptr && no_zc[0] != '\0' && no_zc[0] != '0');
  zero_copy_supported_ =
      !disabled && gl::load_zero_copy_functions() && has_gl_extension("GL_ARB_buffer_storage");
  std::fprintf(stderr, "gl_renderer: zero-copy plane upload %s%s\n",
               zero_copy_supported_ ? "available" : "unavailable",
               disabled ? " (disabled by OPENHTJ2K_GL_NO_ZEROCOPY)" : " (ARB_buffer_storage)");

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
  const char* vs_srcs[] = {kVertexShaderSrc};
  GLuint vs = compile_one_shader(GL_VERTEX_SHADER, vs_srcs, 1, "vertex shader");
  if (!vs) return false;

  const char* fs_rgb_srcs[] = {kFragmentShaderRgbSrc};
  GLuint fs_rgb = compile_one_shader(GL_FRAGMENT_SHADER, fs_rgb_srcs, 1,
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

  // YCbCr program: preamble + helpers + main.
  const char* fs_yc_srcs[] = {kFragmentShaderYcbcrSrc, kFragmentPlanarHelpers,
                               kFragmentShaderYcbcrMain};
  GLuint fs_yc = compile_one_shader(GL_FRAGMENT_SHADER, fs_yc_srcs, 3,
                                    "ycbcr fragment shader");
  if (!fs_yc) {
    gl::DeleteShader(vs);
    gl::DeleteProgram(prog_rgb_);
    prog_rgb_ = 0;
    return false;
  }
  GLuint prog_yc = link_program(vs, fs_yc, "ycbcr program");
  gl::DeleteShader(fs_yc);
  if (!prog_yc) {
    gl::DeleteShader(vs);
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
  u_yc_transfer_   = gl::GetUniformLocation(prog_ycbcr_, "uTransfer");
  u_yc_gamut_      = gl::GetUniformLocation(prog_ycbcr_, "uGamutMatrix");
  u_yc_display_enc_ = gl::GetUniformLocation(prog_ycbcr_, "uDisplayEncoding");
  u_yc_tonemap_    = gl::GetUniformLocation(prog_ycbcr_, "uTonemap");
  u_yc_tm_src_pq_  = gl::GetUniformLocation(prog_ycbcr_, "uTmSrcPq");
  u_yc_tm_ks_      = gl::GetUniformLocation(prog_ycbcr_, "uTmKs");
  u_yc_tm_max_lum_ = gl::GetUniformLocation(prog_ycbcr_, "uTmMaxLum");

  // Planar RGB passthrough program: preamble + helpers + main.
  const char* fs_pr_srcs[] = {kFragmentShaderPlanarRgbSrc, kFragmentPlanarHelpers,
                               kFragmentShaderPlanarRgbMain};
  GLuint fs_pr = compile_one_shader(GL_FRAGMENT_SHADER, fs_pr_srcs, 3,
                                    "planar-rgb fragment shader");
  if (!fs_pr) {
    gl::DeleteShader(vs);
    gl::DeleteProgram(prog_rgb_);
    gl::DeleteProgram(prog_ycbcr_);
    prog_rgb_ = 0;
    prog_ycbcr_ = 0;
    return false;
  }
  GLuint prog_pr = link_program(vs, fs_pr, "planar-rgb program");
  gl::DeleteShader(vs);
  gl::DeleteShader(fs_pr);
  if (!prog_pr) {
    gl::DeleteProgram(prog_rgb_);
    gl::DeleteProgram(prog_ycbcr_);
    prog_rgb_ = 0;
    prog_ycbcr_ = 0;
    return false;
  }
  prog_planar_rgb_       = prog_pr;
  u_pr_y_tex_            = gl::GetUniformLocation(prog_planar_rgb_, "uY");
  u_pr_cb_tex_           = gl::GetUniformLocation(prog_planar_rgb_, "uCb");
  u_pr_cr_tex_           = gl::GetUniformLocation(prog_planar_rgb_, "uCr");
  u_pr_norm_scale_       = gl::GetUniformLocation(prog_planar_rgb_, "uNormScale");
  u_pr_transfer_         = gl::GetUniformLocation(prog_planar_rgb_, "uTransfer");
  u_pr_gamut_            = gl::GetUniformLocation(prog_planar_rgb_, "uGamutMatrix");
  u_pr_display_enc_      = gl::GetUniformLocation(prog_planar_rgb_, "uDisplayEncoding");
  u_pr_tonemap_          = gl::GetUniformLocation(prog_planar_rgb_, "uTonemap");
  u_pr_tm_src_pq_        = gl::GetUniformLocation(prog_planar_rgb_, "uTmSrcPq");
  u_pr_tm_ks_            = gl::GetUniformLocation(prog_planar_rgb_, "uTmKs");
  u_pr_tm_max_lum_       = gl::GetUniformLocation(prog_planar_rgb_, "uTmMaxLum");

  check_gl_error("compile_shader_programs");
  return true;
}

void GlRenderer::shutdown() {
  // Ring teardown needs the context (still current — the window is
  // destroyed at the end of this function).  Threads using the ring are
  // joined before shutdown() per run_receiver_threaded's ordering.
  destroy_ring();
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
  if (prog_planar_rgb_ != 0 && gl::DeleteProgram != nullptr) {
    gl::DeleteProgram(prog_planar_rgb_);
    prog_planar_rgb_ = 0;
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
  if (!window_) return;
  glfwPollEvents();
  // Ring housekeeping runs here because poll_events() is the one renderer
  // entry point the main loop hits every iteration with the GL context
  // current — buffer creation and fence retirement both need GL calls,
  // which the decode-thread side of the ring API must never make.
  service_ring_request();
  poll_ring_fences();
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

void GlRenderer::upload_planar_textures(const void* y_plane, const void* cb_plane,
                                        const void* cr_plane, int w_y, int h_y, int w_c,
                                        int h_c, unsigned int type, int bpp) {
  // Deliberately direct client-memory uploads, NOT a pixel-unpack-buffer
  // ring.  A glBufferData(orphan) + glBufferSubData PBO ring was A/B
  // tested on the 4K59.94 4:2:2 1.7 bpp reference stream (Ryzen 9 9950X
  // + RTX 4070 SUPER, driver 595.71.05, 2026-06-13) and lost decisively:
  // 46.7 vs 54.0 fps, main-thread CPU 55% vs 33%, and the extra ~17
  // MB/frame staging memcpy slowed the decode threads too (avg 10.8 ms
  // vs 8.0 ms).  The driver's direct glTexSubImage2D path already does
  // pinned-buffer DMA; staging through BufferSubData just adds a copy.
  // A PBO design can only win here if the decoder writes straight into
  // a persistently-mapped buffer (as the Metal path does) — anything
  // that re-copies an existing client buffer is net negative.
  glPixelStorei(GL_UNPACK_ALIGNMENT, bpp);
  glBindTexture(GL_TEXTURE_2D, tex_y_);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w_y, h_y, GL_RED, type, y_plane);
  glBindTexture(GL_TEXTURE_2D, tex_cb_);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w_c, h_c, GL_RED, type, cb_plane);
  glBindTexture(GL_TEXTURE_2D, tex_cr_);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w_c, h_c, GL_RED, type, cr_plane);
}

// ── Zero-copy ring ──────────────────────────────────────────────────────

GlRenderer::PlanePointers GlRenderer::acquire_plane_buffers(uint32_t w_y, uint32_t h_y,
                                                            uint32_t w_c, uint32_t h_c,
                                                            int bpp) {
  PlanePointers pp;
  if (!zero_copy_supported_ || window_ == nullptr) return pp;
  if (w_y == 0 || h_y == 0 || w_c == 0 || h_c == 0 || (bpp != 1 && bpp != 2)) return pp;

  if (!ring_ready_.load(std::memory_order_acquire) || w_y != ring_w_y_ || h_y != ring_h_y_ ||
      w_c != ring_w_c_ || h_c != ring_h_c_ || bpp != ring_bpp_) {
    // Ring absent or built for different geometry.  Publish a build
    // request for the main thread and use the vector path this frame.
    std::lock_guard<std::mutex> lk(ring_req_mu_);
    req_w_y_ = w_y;
    req_h_y_ = h_y;
    req_w_c_ = w_c;
    req_h_c_ = h_c;
    req_bpp_ = bpp;
    return pp;
  }

  for (int i = 0; i < kRingDepth; ++i) {
    int expected = kSlotFree;
    if (ring_[i].state.compare_exchange_strong(expected, kSlotInUse,
                                               std::memory_order_acq_rel)) {
      pp.y          = ring_[i].mapped;
      pp.cb         = ring_[i].mapped + ring_y_bytes_;
      pp.cr         = ring_[i].mapped + ring_y_bytes_ + ring_c_bytes_;
      pp.stride_y   = w_y;
      pp.stride_c   = w_c;
      pp.ring_index = i;
      return pp;
    }
  }
  return pp;  // every slot busy (GPU behind) — vector path this frame
}

void GlRenderer::release_plane_buffers(int ring_index) {
  if (ring_index < 0 || ring_index >= kRingDepth) return;
  // Only an acquired-but-never-drawn slot may be freed from off-thread;
  // AwaitFence slots belong to the fence poller.
  int expected = kSlotInUse;
  ring_[ring_index].state.compare_exchange_strong(expected, kSlotFree,
                                                  std::memory_order_acq_rel);
}

void GlRenderer::draw_acquired_planes(int ring_index, int w_y, int h_y, int w_c, int h_c,
                                      int bpp, int bit_depth,
                                      const ycbcr_coefficients* coeffs,
                                      bool components_are_rgb,
                                      const ColorPipelineParams& pipeline) {
  if (!window_ || ring_index < 0 || ring_index >= kRingDepth) return;
  RingSlot& slot = ring_[ring_index];
  if (slot.state.load(std::memory_order_acquire) != kSlotInUse) return;
  if (!ensure_planar_textures(w_y, h_y, w_c, h_c, bpp)) {
    slot.state.store(kSlotFree, std::memory_order_release);
    return;
  }

  // Texture uploads source from the slot's buffer — offsets, not client
  // pointers — so the only CPU write of the plane bytes was the decoder's.
  const GLenum type = (bpp == 2) ? GL_UNSIGNED_SHORT : GL_UNSIGNED_BYTE;
  glPixelStorei(GL_UNPACK_ALIGNMENT, bpp);
  gl::BindBuffer(GL_PIXEL_UNPACK_BUFFER, slot.pbo);
  glBindTexture(GL_TEXTURE_2D, tex_y_);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w_y, h_y, GL_RED, type,
                  reinterpret_cast<const void*>(static_cast<uintptr_t>(0)));
  glBindTexture(GL_TEXTURE_2D, tex_cb_);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w_c, h_c, GL_RED, type,
                  reinterpret_cast<const void*>(static_cast<uintptr_t>(ring_y_bytes_)));
  glBindTexture(GL_TEXTURE_2D, tex_cr_);
  glTexSubImage2D(
      GL_TEXTURE_2D, 0, 0, 0, w_c, h_c, GL_RED, type,
      reinterpret_cast<const void*>(static_cast<uintptr_t>(ring_y_bytes_ + ring_c_bytes_)));
  gl::BindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

  // The slot is reusable once the GPU has copied buffer → textures; the
  // fence covers exactly that.  poll_ring_fences() retires it to Free.
  slot.fence = gl::FenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
  slot.state.store(kSlotAwaitFence, std::memory_order_release);

  const float k = (bpp == 2 && bit_depth >= 9 && bit_depth <= 16)
                      ? 65535.0f / static_cast<float>((1 << bit_depth) - 1)
                      : 1.0f;
  const float norm_scale[3] = {k, k, k};
  if (components_are_rgb)
    draw_planar_rgb_program(w_y, h_y, norm_scale, pipeline);
  else
    draw_ycbcr_program(w_y, h_y, coeffs, norm_scale, pipeline);
}

void GlRenderer::service_ring_request() {
  if (!zero_copy_supported_) return;
  uint32_t w_y = 0, h_y = 0, w_c = 0, h_c = 0;
  int      bpp = 0;
  {
    std::lock_guard<std::mutex> lk(ring_req_mu_);
    if (req_w_y_ == 0) return;  // no pending request
    w_y = req_w_y_;
    h_y = req_h_y_;
    w_c = req_w_c_;
    h_c = req_h_c_;
    bpp = req_bpp_;
  }
  if (ring_ready_.load(std::memory_order_relaxed) && w_y == ring_w_y_ && h_y == ring_h_y_ &&
      w_c == ring_w_c_ && h_c == ring_h_c_ && bpp == ring_bpp_) {
    // Duplicate request raced with the build — nothing to do.
    std::lock_guard<std::mutex> lk(ring_req_mu_);
    req_w_y_ = 0;
    return;
  }

  // Block new acquires, then wait until in-flight slots drain.  A slot
  // claimed in the acquire/ready race stays valid — we simply retry on a
  // later poll_events() tick; the old buffers are not touched until every
  // slot is Free again.
  ring_ready_.store(false, std::memory_order_release);
  for (int i = 0; i < kRingDepth; ++i) {
    if (ring_[i].state.load(std::memory_order_acquire) != kSlotFree) return;
  }
  destroy_ring();

  const size_t y_bytes = static_cast<size_t>(w_y) * h_y * static_cast<size_t>(bpp);
  const size_t c_bytes = static_cast<size_t>(w_c) * h_c * static_cast<size_t>(bpp);
  const size_t total   = y_bytes + 2 * c_bytes;
  const GLbitfield storage_flags =
      GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
  bool ok = true;
  for (int i = 0; i < kRingDepth && ok; ++i) {
    GLuint pbo = 0;
    gl::GenBuffers(1, &pbo);
    if (pbo == 0) {
      ok = false;
      break;
    }
    ring_[i].pbo = pbo;
    gl::BindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    gl::BufferStorage(GL_PIXEL_UNPACK_BUFFER, static_cast<GLsizeiptr>(total), nullptr,
                      storage_flags);
    void* mapped = gl::MapBufferRange(GL_PIXEL_UNPACK_BUFFER, 0,
                                      static_cast<GLsizeiptr>(total), storage_flags);
    ring_[i].mapped = static_cast<uint8_t*>(mapped);
    if (mapped == nullptr || !check_gl_error("zero-copy ring slot")) ok = false;
  }
  gl::BindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  if (!ok) {
    // Driver said no — permanently fall back to the vector upload path.
    destroy_ring();
    zero_copy_supported_ = false;
    std::fprintf(stderr, "gl_renderer: zero-copy ring allocation failed; using vector path\n");
    std::lock_guard<std::mutex> lk(ring_req_mu_);
    req_w_y_ = 0;
    return;
  }

  ring_w_y_     = w_y;
  ring_h_y_     = h_y;
  ring_w_c_     = w_c;
  ring_h_c_     = h_c;
  ring_bpp_     = bpp;
  ring_y_bytes_ = y_bytes;
  ring_c_bytes_ = c_bytes;
  {
    std::lock_guard<std::mutex> lk(ring_req_mu_);
    req_w_y_ = 0;
  }
  ring_ready_.store(true, std::memory_order_release);
  std::fprintf(stderr,
               "gl_renderer: zero-copy ring ready (%d x %.1f MB persistent-mapped, "
               "%ux%u Y / %ux%u C, %d bpp)\n",
               kRingDepth, static_cast<double>(total) / (1024.0 * 1024.0), w_y, h_y, w_c, h_c,
               bpp);
}

void GlRenderer::poll_ring_fences() {
  for (int i = 0; i < kRingDepth; ++i) {
    RingSlot& slot = ring_[i];
    if (slot.state.load(std::memory_order_acquire) != kSlotAwaitFence) continue;
    if (slot.fence == nullptr) {  // FenceSync failed at draw time — free defensively
      slot.state.store(kSlotFree, std::memory_order_release);
      continue;
    }
    const GLenum r =
        gl::ClientWaitSync(static_cast<GLsync>(slot.fence), 0, /*timeout_ns=*/0);
    if (r == GL_ALREADY_SIGNALED || r == GL_CONDITION_SATISFIED || r == GL_WAIT_FAILED) {
      gl::DeleteSync(static_cast<GLsync>(slot.fence));
      slot.fence = nullptr;
      slot.state.store(kSlotFree, std::memory_order_release);
    }
    // GL_TIMEOUT_EXPIRED: still in flight — try again next tick.
  }
}

void GlRenderer::destroy_ring() {
  ring_ready_.store(false, std::memory_order_release);
  for (int i = 0; i < kRingDepth; ++i) {
    RingSlot& slot = ring_[i];
    if (slot.fence != nullptr && gl::DeleteSync != nullptr) {
      gl::DeleteSync(static_cast<GLsync>(slot.fence));
      slot.fence = nullptr;
    }
    if (slot.pbo != 0) {
      if (slot.mapped != nullptr && gl::UnmapBuffer != nullptr) {
        gl::BindBuffer(GL_PIXEL_UNPACK_BUFFER, slot.pbo);
        gl::UnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
        gl::BindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
      }
      if (gl::DeleteBuffers != nullptr) {
        const GLuint b = slot.pbo;
        gl::DeleteBuffers(1, &b);
      }
      slot.pbo = 0;
    }
    slot.mapped = nullptr;
    slot.state.store(kSlotFree, std::memory_order_release);
  }
}

void GlRenderer::upload_planar_and_draw(const uint8_t* y_plane, const uint8_t* cb_plane,
                                        const uint8_t* cr_plane, int w_y, int h_y, int w_c,
                                        int h_c, const ycbcr_coefficients* coeffs,
                                        bool components_are_rgb,
                                        const ColorPipelineParams& pipeline) {
  if (!window_ || w_y <= 0 || h_y <= 0 || w_c <= 0 || h_c <= 0) return;

  if (!ensure_planar_textures(w_y, h_y, w_c, h_c, /*bpp=*/1)) return;

  upload_planar_textures(y_plane, cb_plane, cr_plane, w_y, h_y, w_c, h_c,
                         GL_UNSIGNED_BYTE, /*bpp=*/1);

  // 8-bit source: the texture value already lives in the sample's native
  // [0, 1] range, so uNormScale is the identity.
  const float norm_scale[3] = {1.0f, 1.0f, 1.0f};
  if (components_are_rgb)
    draw_planar_rgb_program(w_y, h_y, norm_scale, pipeline);
  else
    draw_ycbcr_program(w_y, h_y, coeffs, norm_scale, pipeline);
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

  upload_planar_textures(y_plane, cb_plane, cr_plane, w_y, h_y, w_c, h_c,
                         GL_UNSIGNED_SHORT, /*bpp=*/2);

  // GL_R16 reads back as (sample / 65535).  Multiply by this factor in the
  // shader to restore the sample's native [0, 1] range so the bias/scale/
  // matrix math is depth-agnostic.
  const float native_max    = static_cast<float>((1 << bit_depth) - 1);
  const float k             = 65535.0f / native_max;
  const float norm_scale[3] = {k, k, k};
  if (components_are_rgb)
    draw_planar_rgb_program(w_y, h_y, norm_scale, pipeline);
  else
    draw_ycbcr_program(w_y, h_y, coeffs, norm_scale, pipeline);
}

void GlRenderer::draw_ycbcr_program(int w_y, int h_y, const ycbcr_coefficients* coeffs,
                                    const float* norm_scale,
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

  gl::Uniform1i(u_yc_transfer_, pipeline.transfer);
  gl::UniformMatrix3fv(u_yc_gamut_, 1, GL_FALSE, pipeline.gamut_matrix);
  gl::Uniform1i(u_yc_display_enc_, pipeline.display_encoding);
  gl::Uniform1i(u_yc_tonemap_, pipeline.tonemap);
  gl::Uniform1f(u_yc_tm_src_pq_, pipeline.tm_src_pq);
  gl::Uniform1f(u_yc_tm_ks_, pipeline.tm_ks);
  gl::Uniform1f(u_yc_tm_max_lum_, pipeline.tm_max_lum);

  if (coeffs != nullptr) {
    const float mat[9] = {
        1.0f,           1.0f,            1.0f,
        0.0f,          -coeffs->cb_to_g, coeffs->cb_to_b,
        coeffs->cr_to_r, -coeffs->cr_to_g, 0.0f,
    };
    gl::UniformMatrix3fv(u_yc_matrix_, 1, GL_FALSE, mat);

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

void GlRenderer::draw_planar_rgb_program(int w_y, int h_y, const float* norm_scale,
                                         const ColorPipelineParams& pipeline) {
  gl::UseProgram(prog_planar_rgb_);
  gl::ActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, tex_y_);
  gl::Uniform1i(u_pr_y_tex_, 0);
  gl::ActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, tex_cb_);
  gl::Uniform1i(u_pr_cb_tex_, 1);
  gl::ActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, tex_cr_);
  gl::Uniform1i(u_pr_cr_tex_, 2);

  gl::Uniform3fv(u_pr_norm_scale_, 1, norm_scale);

  gl::Uniform1i(u_pr_transfer_, pipeline.transfer);
  gl::UniformMatrix3fv(u_pr_gamut_, 1, GL_FALSE, pipeline.gamut_matrix);
  gl::Uniform1i(u_pr_display_enc_, pipeline.display_encoding);
  gl::Uniform1i(u_pr_tonemap_, pipeline.tonemap);
  gl::Uniform1f(u_pr_tm_src_pq_, pipeline.tm_src_pq);
  gl::Uniform1f(u_pr_tm_ks_, pipeline.tm_ks);
  gl::Uniform1f(u_pr_tm_max_lum_, pipeline.tm_max_lum);

  int fb_w = 0;
  int fb_h = 0;
  glfwGetFramebufferSize(window_, &fb_w, &fb_h);
  draw_fullscreen_quad(fb_w, fb_h, w_y, h_y);

  glfwSwapBuffers(window_);
}

}  // namespace open_htj2k::rtp_recv
