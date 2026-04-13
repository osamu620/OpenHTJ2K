// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Licensed under the same BSD 3-Clause terms as the rest of OpenHTJ2K.

// Metal Shading Language (MSL) shaders for the RFC 9828 RTP receiver.
// Direct port of the GLSL 330 shaders in gl_renderer.cpp.

#include <metal_stdlib>
using namespace metal;

// ── Shared types ─────────────────────────────────────────────────────────

struct VertexIn {
  float2 position [[attribute(0)]];
  float2 texcoord [[attribute(1)]];
};

struct VertexOut {
  float4 position [[position]];
  float2 uv;
};

// Per-draw uniforms passed via setFragmentBytes.
struct FragmentUniforms {
  float3x3 matrix;          // YCbCr→RGB (column-major, same layout as GLSL mat3)
  float3   bias;            // (16/255, 128/255, 128/255) for narrow; (0, 0.5, 0.5) for full
  float3   scale;           // (255/219, 255/224, 255/224) for narrow; (1, 1, 1) for full
  float3   norm_scale;      // Identity for 8-bit; 65535/((1<<bd)-1) for 16-bit
  float3x3 gamut_matrix;    // Identity or BT.2020→BT.709
  int      transfer;        // 0=gamma22, 1=PQ, 2=HLG
  int      display_encoding;// 0=sRGB, 1=gamma22, 2=linear
};

// ── Vertex shader (shared by all fragment programs) ──────────────────────

vertex VertexOut vertex_main(VertexIn in [[stage_in]]) {
  VertexOut out;
  out.position = float4(in.position, 0.0, 1.0);
  out.uv       = in.texcoord;
  return out;
}

// ── HDR pipeline helpers ─────────────────────────────────────────────────
// Direct port of kFragmentPlanarHelpers from gl_renderer.cpp.

// SMPTE ST 2084 PQ EOTF constants (BT.2100 Table 4).
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

// ── Fragment shader: RGB passthrough (CPU fallback path) ─────────────────

fragment float4 fragment_rgb(VertexOut in [[stage_in]],
                             texture2d<float> tex [[texture(0)]]) {
  constexpr sampler s(filter::linear, address::clamp_to_edge);
  return float4(tex.sample(s, in.uv).rgb, 1.0);
}

// ── Fragment shader: YCbCr → RGB with matrix + HDR pipeline ──────────────

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

// ── Fragment shader: Planar RGB passthrough with HDR pipeline ─────────────

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
