// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Licensed under the same BSD 3-Clause terms as the rest of OpenHTJ2K.

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <Cocoa/Cocoa.h>
#include <simd/simd.h>

#include "metal_renderer.hpp"
#include "color_pipeline.hpp"
#include "ycbcr_rgb.hpp"

#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <cstdio>
#include <cstring>
#include <vector>

namespace open_htj2k::rtp_recv {
// Embedded MSL shader source for runtime compilation (fallback when no
// pre-compiled .metallib is available).
const char* kMetalShaderSource = R"msl(
#include <metal_stdlib>
using namespace metal;

struct VertexIn {
  float2 position [[attribute(0)]];
  float2 texcoord [[attribute(1)]];
};

struct VertexOut {
  float4 position [[position]];
  float2 uv;
};

struct FragmentUniforms {
  float3x3 matrix;
  float3   bias;
  float3   scale;
  float3   norm_scale;
  float3x3 gamut_matrix;
  int      transfer;
  int      display_encoding;
};

vertex VertexOut vertex_main(VertexIn in [[stage_in]]) {
  VertexOut out;
  out.position = float4(in.position, 0.0, 1.0);
  out.uv       = in.texcoord;
  return out;
}

constant float kPqM1 = 0.1593017578125;
constant float kPqM2 = 78.84375;
constant float kPqC1 = 0.8359375;
constant float kPqC2 = 18.8515625;
constant float kPqC3 = 18.6875;

static float3 pq_to_linear(float3 e) {
  float3 v   = pow(max(e, float3(0.0)), float3(1.0 / kPqM2));
  float3 num = max(v - float3(kPqC1), float3(0.0));
  float3 den = float3(kPqC2) - float3(kPqC3) * v;
  return pow(num / den, float3(1.0 / kPqM1));
}

static float3 hlg_inverse(float3 e) {
  const float a = 0.17883277;
  const float b = 0.28466892;
  const float c = 0.55991073;
  float3 lo = (e * e) / 3.0;
  float3 hi = (exp((e - float3(c)) / a) + float3(b)) / 12.0;
  float3 sel = float3(e.x > 0.5 ? 1.0 : 0.0,
                      e.y > 0.5 ? 1.0 : 0.0,
                      e.z > 0.5 ? 1.0 : 0.0);
  return mix(lo, hi, sel);
}

static float3 apply_inverse_transfer(float3 e, int transfer) {
  if (transfer == 1) return pq_to_linear(e);
  if (transfer == 2) return hlg_inverse(e);
  return pow(max(e, float3(0.0)), float3(2.2));
}

static float3 linear_to_srgb(float3 l) {
  float3 lo  = l * 12.92;
  float3 hi  = 1.055 * pow(l, float3(1.0 / 2.4)) - 0.055;
  float3 sel = float3(l.x <= 0.0031308 ? 1.0 : 0.0,
                      l.y <= 0.0031308 ? 1.0 : 0.0,
                      l.z <= 0.0031308 ? 1.0 : 0.0);
  return mix(hi, lo, sel);
}

static float3 apply_display_encoding(float3 l, int display_encoding) {
  if (display_encoding == 1) return pow(max(l, float3(0.0)), float3(1.0 / 2.2));
  if (display_encoding == 2) return l;
  return linear_to_srgb(l);
}

fragment float4 fragment_rgb(VertexOut in [[stage_in]],
                             texture2d<float> tex [[texture(0)]]) {
  constexpr sampler s(filter::linear, address::clamp_to_edge);
  return float4(tex.sample(s, in.uv).rgb, 1.0);
}

fragment float4 fragment_ycbcr(VertexOut in [[stage_in]],
                               texture2d<float> texY  [[texture(0)]],
                               texture2d<float> texCb [[texture(1)]],
                               texture2d<float> texCr [[texture(2)]],
                               constant FragmentUniforms& u [[buffer(0)]]) {
  constexpr sampler s(filter::linear, address::clamp_to_edge);
  float3 ycbcr = float3(texY.sample(s, in.uv).r,
                        texCb.sample(s, in.uv).r,
                        texCr.sample(s, in.uv).r);
  float3 n      = ycbcr * u.norm_scale;
  float3 rgb_nl = u.matrix * ((n - u.bias) * u.scale);
  rgb_nl        = saturate(rgb_nl);
  float3 lin_s  = apply_inverse_transfer(rgb_nl, u.transfer);
  float3 lin_d  = u.gamut_matrix * lin_s;
  lin_d         = saturate(lin_d);
  float3 out_nl = apply_display_encoding(lin_d, u.display_encoding);
  return float4(saturate(out_nl), 1.0);
}

fragment float4 fragment_planar_rgb(VertexOut in [[stage_in]],
                                    texture2d<float> texR [[texture(0)]],
                                    texture2d<float> texG [[texture(1)]],
                                    texture2d<float> texB [[texture(2)]],
                                    constant FragmentUniforms& u [[buffer(0)]]) {
  constexpr sampler s(filter::linear, address::clamp_to_edge);
  float3 rgb = float3(texR.sample(s, in.uv).r,
                      texG.sample(s, in.uv).r,
                      texB.sample(s, in.uv).r);
  float3 rgb_nl = rgb * u.norm_scale;
  rgb_nl        = saturate(rgb_nl);
  float3 lin_s  = apply_inverse_transfer(rgb_nl, u.transfer);
  float3 lin_d  = u.gamut_matrix * lin_s;
  lin_d         = saturate(lin_d);
  float3 out_nl = apply_display_encoding(lin_d, u.display_encoding);
  return float4(saturate(out_nl), 1.0);
}
)msl";

// Must match shaders.metal FragmentUniforms exactly.
struct FragmentUniforms {
  simd_float3x3 matrix;
  simd_float3   bias;
  simd_float3   scale;
  simd_float3   norm_scale;
  simd_float3x3 gamut_matrix;
  int32_t       transfer;
  int32_t       display_encoding;
};

// Copy 9 packed floats (column-major) into a padded simd_float3x3.
// simd_float3x3 stores 3 columns of simd_float3, each padded to 16 bytes.
static inline simd_float3x3 mat3_from_packed(const float m[9]) {
  return (simd_float3x3){
      simd_make_float3(m[0], m[1], m[2]),
      simd_make_float3(m[3], m[4], m[5]),
      simd_make_float3(m[6], m[7], m[8]),
  };
}

// ESC/Q key callback.
static void key_callback(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/) {
  if ((key == GLFW_KEY_ESCAPE || key == GLFW_KEY_Q) && action == GLFW_PRESS)
    glfwSetWindowShouldClose(window, GLFW_TRUE);
}

struct MetalRenderer::Impl {
  id<MTLDevice>              device        = nil;
  id<MTLCommandQueue>        commandQueue  = nil;
  id<MTLLibrary>             library       = nil;
  id<MTLRenderPipelineState> psoRgb        = nil;
  id<MTLRenderPipelineState> psoYcbcr      = nil;
  id<MTLRenderPipelineState> psoPlanarRgb  = nil;
  id<MTLBuffer>              vertexBuffer  = nil;
  CAMetalLayer*              layer         = nil;

  // Per-plane textures, recreated on dimension/format change.
  id<MTLTexture> texRgb  = nil;
  int            texRgbW = 0, texRgbH = 0;

  id<MTLTexture> texY  = nil;
  id<MTLTexture> texCb = nil;
  id<MTLTexture> texCr = nil;
  int texYW = 0, texYH = 0, texYBpp = 0;
  int texCbW = 0, texCbH = 0, texCbBpp = 0;
  int texCrW = 0, texCrH = 0, texCrBpp = 0;

  id<MTLTexture> ensureTexture(id<MTLTexture>& tex, int& cachedW, int& cachedH, int& cachedBpp,
                               int w, int h, int bpp) {
    if (tex != nil && cachedW == w && cachedH == h && cachedBpp == bpp)
      return tex;
    MTLPixelFormat fmt = (bpp == 2) ? MTLPixelFormatR16Unorm : MTLPixelFormatR8Unorm;
    MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fmt
                                                                                   width:w
                                                                                  height:h
                                                                               mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;
    tex = [device newTextureWithDescriptor:desc];
    cachedW = w;  cachedH = h;  cachedBpp = bpp;
    return tex;
  }

  id<MTLTexture> ensureRgbTexture(int w, int h) {
    if (texRgb != nil && texRgbW == w && texRgbH == h)
      return texRgb;
    MTLTextureDescriptor* desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                          width:w
                                                         height:h
                                                      mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;
    texRgb = [device newTextureWithDescriptor:desc];
    texRgbW = w;  texRgbH = h;
    return texRgb;
  }

  void drawQuad(id<MTLRenderPipelineState> pso, int fbW, int fbH, int contentW, int contentH,
                id<MTLTexture> t0, id<MTLTexture> t1, id<MTLTexture> t2,
                const FragmentUniforms* uniforms) {
    @autoreleasepool {
      id<CAMetalDrawable> drawable = [layer nextDrawable];
      if (!drawable) return;

      MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
      rpd.colorAttachments[0].texture    = drawable.texture;
      rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
      rpd.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
      rpd.colorAttachments[0].storeAction = MTLStoreActionStore;

      id<MTLCommandBuffer> cmdBuf = [commandQueue commandBuffer];
      id<MTLRenderCommandEncoder> enc = [cmdBuf renderCommandEncoderWithDescriptor:rpd];

      [enc setRenderPipelineState:pso];
      [enc setVertexBuffer:vertexBuffer offset:0 atIndex:0];

      if (t0) [enc setFragmentTexture:t0 atIndex:0];
      if (t1) [enc setFragmentTexture:t1 atIndex:1];
      if (t2) [enc setFragmentTexture:t2 atIndex:2];
      if (uniforms) [enc setFragmentBytes:uniforms length:sizeof(FragmentUniforms) atIndex:0];

      // Letterbox/pillarbox viewport.
      float fbAspect  = static_cast<float>(fbW) / static_cast<float>(fbH);
      float imgAspect = static_cast<float>(contentW) / static_cast<float>(contentH);
      int drawW, drawH, x0, y0;
      if (imgAspect > fbAspect) {
        drawW = fbW;
        drawH = static_cast<int>(static_cast<float>(fbW) / imgAspect);
      } else {
        drawH = fbH;
        drawW = static_cast<int>(static_cast<float>(fbH) * imgAspect);
      }
      x0 = (fbW - drawW) / 2;
      y0 = (fbH - drawH) / 2;

      MTLViewport vp = {static_cast<double>(x0), static_cast<double>(y0),
                        static_cast<double>(drawW), static_cast<double>(drawH), 0.0, 1.0};
      [enc setViewport:vp];

      [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
      [enc endEncoding];

      [cmdBuf presentDrawable:drawable];
      [cmdBuf commit];
    }
  }
};

// ── Public interface ─────────────────────────────────────────────────────

MetalRenderer::~MetalRenderer() { shutdown(); }

bool MetalRenderer::init(int window_w, int window_h, const char* title, bool vsync) {
  glfwSetErrorCallback([](int, const char* desc) {
    std::fprintf(stderr, "GLFW error: %s\n", desc);
  });
  if (!glfwInit()) return false;

  // No OpenGL context — we use Metal directly.
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  window_ = glfwCreateWindow(window_w, window_h, title, nullptr, nullptr);
  if (!window_) { glfwTerminate(); return false; }
  glfwSetKeyCallback(window_, key_callback);

  impl_ = new Impl();

  // Get the native NSWindow and attach a CAMetalLayer.
  NSWindow* nsWin = glfwGetCocoaWindow(window_);
  impl_->device = MTLCreateSystemDefaultDevice();
  if (!impl_->device) {
    std::fprintf(stderr, "metal_renderer: MTLCreateSystemDefaultDevice failed\n");
    shutdown();
    return false;
  }

  impl_->layer = [CAMetalLayer layer];
  impl_->layer.device = impl_->device;
  impl_->layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
  impl_->layer.displaySyncEnabled = vsync ? YES : NO;
  nsWin.contentView.wantsLayer = YES;
  nsWin.contentView.layer = impl_->layer;

  impl_->commandQueue = [impl_->device newCommandQueue];

  // Try loading pre-compiled .metallib first, then fall back to runtime
  // compilation from the embedded shader source string.
  NSError* error = nil;
  impl_->library = [impl_->device newDefaultLibrary];
  if (!impl_->library) {
    // Try loading from the same directory as the executable.
    NSString* exePath = [[NSBundle mainBundle] executablePath];
    NSString* dir = [exePath stringByDeletingLastPathComponent];
    NSString* libPath = [dir stringByAppendingPathComponent:@"default.metallib"];
    NSURL* libURL = [NSURL fileURLWithPath:libPath];
    impl_->library = [impl_->device newLibraryWithURL:libURL error:nil];
  }
  if (!impl_->library) {
    // Runtime compilation from embedded source string.
    extern const char* kMetalShaderSource;
    NSString* src = [NSString stringWithUTF8String:kMetalShaderSource];
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    impl_->library = [impl_->device newLibraryWithSource:src options:opts error:&error];
    if (!impl_->library) {
      std::fprintf(stderr, "metal_renderer: shader compilation failed: %s\n",
                   error ? [[error localizedDescription] UTF8String] : "?");
      shutdown();
      return false;
    }
  }

  // Create render pipeline states for the three fragment programs.
  auto makePSO = [&](const char* fragName) -> id<MTLRenderPipelineState> {
    id<MTLFunction> vertFunc = [impl_->library newFunctionWithName:@"vertex_main"];
    id<MTLFunction> fragFunc = [impl_->library newFunctionWithName:
        [NSString stringWithUTF8String:fragName]];
    if (!vertFunc || !fragFunc) {
      std::fprintf(stderr, "metal_renderer: shader function '%s' not found\n", fragName);
      return nil;
    }

    MTLVertexDescriptor* vd = [[MTLVertexDescriptor alloc] init];
    vd.attributes[0].format = MTLVertexFormatFloat2;
    vd.attributes[0].offset = 0;
    vd.attributes[0].bufferIndex = 0;
    vd.attributes[1].format = MTLVertexFormatFloat2;
    vd.attributes[1].offset = 2 * sizeof(float);
    vd.attributes[1].bufferIndex = 0;
    vd.layouts[0].stride = 4 * sizeof(float);

    MTLRenderPipelineDescriptor* pd = [[MTLRenderPipelineDescriptor alloc] init];
    pd.vertexFunction = vertFunc;
    pd.fragmentFunction = fragFunc;
    pd.vertexDescriptor = vd;
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;

    NSError* err = nil;
    id<MTLRenderPipelineState> pso = [impl_->device newRenderPipelineStateWithDescriptor:pd error:&err];
    if (!pso) {
      std::fprintf(stderr, "metal_renderer: pipeline '%s' failed: %s\n",
                   fragName, err ? [[err localizedDescription] UTF8String] : "?");
    }
    return pso;
  };

  impl_->psoRgb       = makePSO("fragment_rgb");
  impl_->psoYcbcr     = makePSO("fragment_ycbcr");
  impl_->psoPlanarRgb = makePSO("fragment_planar_rgb");

  if (!impl_->psoRgb || !impl_->psoYcbcr || !impl_->psoPlanarRgb) {
    shutdown();
    return false;
  }

  // Vertex buffer — same 4×{x,y,u,v} as the GL renderer.
  constexpr float verts[4 * 4] = {
      -1.0f, +1.0f, 0.0f, 0.0f,
      -1.0f, -1.0f, 0.0f, 1.0f,
      +1.0f, +1.0f, 1.0f, 0.0f,
      +1.0f, -1.0f, 1.0f, 1.0f,
  };
  impl_->vertexBuffer = [impl_->device newBufferWithBytes:verts
                                                   length:sizeof(verts)
                                                  options:MTLResourceStorageModeShared];
  return true;
}

void MetalRenderer::shutdown() {
  if (impl_) {
    // Release all Metal objects (ARC handles the actual dealloc).
    impl_->texRgb = nil;
    impl_->texY = nil;
    impl_->texCb = nil;
    impl_->texCr = nil;
    impl_->vertexBuffer = nil;
    impl_->psoRgb = nil;
    impl_->psoYcbcr = nil;
    impl_->psoPlanarRgb = nil;
    impl_->library = nil;
    impl_->commandQueue = nil;
    impl_->layer = nil;
    impl_->device = nil;
    delete impl_;
    impl_ = nullptr;
  }
  if (window_) {
    glfwDestroyWindow(window_);
    window_ = nullptr;
    glfwTerminate();
  }
}

bool MetalRenderer::should_close() const {
  return window_ ? glfwWindowShouldClose(window_) != 0 : true;
}

void MetalRenderer::poll_events() {
  if (window_) glfwPollEvents();
}

// ── RGB passthrough (CPU fallback) ───────────────────────────────────────

void MetalRenderer::upload_and_draw(const uint8_t* rgb, int w, int h) {
  if (!window_ || !impl_ || w <= 0 || h <= 0) return;

  impl_->ensureRgbTexture(w, h);

  // Convert RGB → RGBA (Metal has no RGB8 texture format).
  // For the CPU fallback path this extra copy is acceptable.
  const size_t npix = static_cast<size_t>(w) * h;
  std::vector<uint8_t> rgba(npix * 4);
  for (size_t i = 0; i < npix; ++i) {
    rgba[4 * i + 0] = rgb[3 * i + 0];
    rgba[4 * i + 1] = rgb[3 * i + 1];
    rgba[4 * i + 2] = rgb[3 * i + 2];
    rgba[4 * i + 3] = 255;
  }
  [impl_->texRgb replaceRegion:MTLRegionMake2D(0, 0, w, h)
                   mipmapLevel:0
                     withBytes:rgba.data()
                   bytesPerRow:w * 4];

  int fbW, fbH;
  glfwGetFramebufferSize(window_, &fbW, &fbH);
  impl_->layer.drawableSize = CGSizeMake(fbW, fbH);

  impl_->drawQuad(impl_->psoRgb, fbW, fbH, w, h, impl_->texRgb, nil, nil, nullptr);
}

// ── Planar 8-bit ─────────────────────────────────────────────────────────

void MetalRenderer::upload_planar_and_draw(const uint8_t* y_plane, const uint8_t* cb_plane,
                                           const uint8_t* cr_plane, int w_y, int h_y,
                                           int w_c, int h_c,
                                           const ycbcr_coefficients* coeffs,
                                           bool components_are_rgb,
                                           const ColorPipelineParams& pipeline) {
  if (!window_ || !impl_ || w_y <= 0 || h_y <= 0 || w_c <= 0 || h_c <= 0) return;

  impl_->ensureTexture(impl_->texY,  impl_->texYW,  impl_->texYH,  impl_->texYBpp,  w_y, h_y, 1);
  impl_->ensureTexture(impl_->texCb, impl_->texCbW, impl_->texCbH, impl_->texCbBpp, w_c, h_c, 1);
  impl_->ensureTexture(impl_->texCr, impl_->texCrW, impl_->texCrH, impl_->texCrBpp, w_c, h_c, 1);

  [impl_->texY  replaceRegion:MTLRegionMake2D(0, 0, w_y, h_y) mipmapLevel:0
                    withBytes:y_plane  bytesPerRow:w_y];
  [impl_->texCb replaceRegion:MTLRegionMake2D(0, 0, w_c, h_c) mipmapLevel:0
                    withBytes:cb_plane bytesPerRow:w_c];
  [impl_->texCr replaceRegion:MTLRegionMake2D(0, 0, w_c, h_c) mipmapLevel:0
                    withBytes:cr_plane bytesPerRow:w_c];

  FragmentUniforms u = {};
  u.norm_scale = simd_make_float3(1.0f, 1.0f, 1.0f);
  u.transfer = pipeline.transfer;
  u.display_encoding = pipeline.display_encoding;
  u.gamut_matrix = mat3_from_packed(pipeline.gamut_matrix);

  if (!components_are_rgb && coeffs) {
    // Column-major 3x3 YCbCr→RGB matrix (same layout as the GLSL version).
    float mat[9] = {
        1.0f,            1.0f,             1.0f,
        0.0f,           -coeffs->cb_to_g,  coeffs->cb_to_b,
        coeffs->cr_to_r, -coeffs->cr_to_g, 0.0f,
    };
    u.matrix = mat3_from_packed(mat);

    if (coeffs->narrow_range) {
      u.bias  = simd_make_float3(16.0f / 255.0f, 128.0f / 255.0f, 128.0f / 255.0f);
      u.scale = simd_make_float3(255.0f / 219.0f, 255.0f / 224.0f, 255.0f / 224.0f);
    } else {
      u.bias  = simd_make_float3(0.0f, 0.5f, 0.5f);
      u.scale = simd_make_float3(1.0f, 1.0f, 1.0f);
    }
  } else {
    // Identity for RGB passthrough.
    const float id3[9] = {1,0,0, 0,1,0, 0,0,1};
    u.matrix = mat3_from_packed(id3);
    u.bias  = simd_make_float3(0.0f, 0.0f, 0.0f);
    u.scale = simd_make_float3(1.0f, 1.0f, 1.0f);
  }

  int fbW, fbH;
  glfwGetFramebufferSize(window_, &fbW, &fbH);
  impl_->layer.drawableSize = CGSizeMake(fbW, fbH);

  id<MTLRenderPipelineState> pso = components_are_rgb ? impl_->psoPlanarRgb : impl_->psoYcbcr;
  impl_->drawQuad(pso, fbW, fbH, w_y, h_y, impl_->texY, impl_->texCb, impl_->texCr, &u);
}

// ── Planar 16-bit ────────────────────────────────────────────────────────

void MetalRenderer::upload_planar_16_and_draw(const uint16_t* y_plane, const uint16_t* cb_plane,
                                              const uint16_t* cr_plane, int w_y, int h_y,
                                              int w_c, int h_c, int bit_depth,
                                              const ycbcr_coefficients* coeffs,
                                              bool components_are_rgb,
                                              const ColorPipelineParams& pipeline) {
  if (!window_ || !impl_ || w_y <= 0 || h_y <= 0 || w_c <= 0 || h_c <= 0) return;
  if (bit_depth < 9 || bit_depth > 16) return;

  impl_->ensureTexture(impl_->texY,  impl_->texYW,  impl_->texYH,  impl_->texYBpp,  w_y, h_y, 2);
  impl_->ensureTexture(impl_->texCb, impl_->texCbW, impl_->texCbH, impl_->texCbBpp, w_c, h_c, 2);
  impl_->ensureTexture(impl_->texCr, impl_->texCrW, impl_->texCrH, impl_->texCrBpp, w_c, h_c, 2);

  [impl_->texY  replaceRegion:MTLRegionMake2D(0, 0, w_y, h_y) mipmapLevel:0
                    withBytes:y_plane  bytesPerRow:w_y * 2];
  [impl_->texCb replaceRegion:MTLRegionMake2D(0, 0, w_c, h_c) mipmapLevel:0
                    withBytes:cb_plane bytesPerRow:w_c * 2];
  [impl_->texCr replaceRegion:MTLRegionMake2D(0, 0, w_c, h_c) mipmapLevel:0
                    withBytes:cr_plane bytesPerRow:w_c * 2];

  const float native_max = static_cast<float>((1 << bit_depth) - 1);
  const float k = 65535.0f / native_max;

  FragmentUniforms u = {};
  u.norm_scale = simd_make_float3(k, k, k);
  u.transfer = pipeline.transfer;
  u.display_encoding = pipeline.display_encoding;
  u.gamut_matrix = mat3_from_packed(pipeline.gamut_matrix);

  if (!components_are_rgb && coeffs) {
    float mat[9] = {
        1.0f,            1.0f,             1.0f,
        0.0f,           -coeffs->cb_to_g,  coeffs->cb_to_b,
        coeffs->cr_to_r, -coeffs->cr_to_g, 0.0f,
    };
    u.matrix = mat3_from_packed(mat);

    if (coeffs->narrow_range) {
      u.bias  = simd_make_float3(16.0f / 255.0f, 128.0f / 255.0f, 128.0f / 255.0f);
      u.scale = simd_make_float3(255.0f / 219.0f, 255.0f / 224.0f, 255.0f / 224.0f);
    } else {
      u.bias  = simd_make_float3(0.0f, 0.5f, 0.5f);
      u.scale = simd_make_float3(1.0f, 1.0f, 1.0f);
    }
  } else {
    const float id3[9] = {1,0,0, 0,1,0, 0,0,1};
    u.matrix = mat3_from_packed(id3);
    u.bias  = simd_make_float3(0.0f, 0.0f, 0.0f);
    u.scale = simd_make_float3(1.0f, 1.0f, 1.0f);
  }

  int fbW, fbH;
  glfwGetFramebufferSize(window_, &fbW, &fbH);
  impl_->layer.drawableSize = CGSizeMake(fbW, fbH);

  id<MTLRenderPipelineState> pso = components_are_rgb ? impl_->psoPlanarRgb : impl_->psoYcbcr;
  impl_->drawQuad(pso, fbW, fbH, w_y, h_y, impl_->texY, impl_->texCb, impl_->texCr, &u);
}

}  // namespace open_htj2k::rtp_recv
