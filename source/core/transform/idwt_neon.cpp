// Copyright (c) 2019 - 2021, Osamu Watanabe
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
//    modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
//    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#if defined(OPENHTJ2K_ENABLE_ARM_NEON) || 1
  #include "dwt.hpp"
  #include "utils.hpp"
  #include <cstring>
#include <arm_neon.h>

/********************************************************************************
 * horizontal transforms
 *******************************************************************************/
// irreversible IDWT
auto idwt_irrev97_fixed_neon_hor_step0 = [](const int32_t init_pos, const int32_t simdlen, float *const X,
                                            const int32_t n0, const int32_t n1) {
  auto vvv = vdupq_n_f32(fD);
  for (int32_t n = init_pos, i = simdlen; i > 0; i -= 4, n += 8) {
    auto x0   = vld2q_f32(X + n + n0);
    auto x1   = vld2q_f32(X + n + n1);
    auto tmp  = vaddq_f32(x0.val[0], x1.val[0]);
    tmp       = vmulq_f32(tmp, vvv);
    x0.val[1] = vsubq_f32(x0.val[1], tmp);
    vst2q_f32(X + n + n0, x0);
  }
};

auto idwt_irrev97_fixed_neon_hor_step1 = [](const int32_t init_pos, const int32_t simdlen, float *const X,
                                            const int32_t n0, const int32_t n1) {
  auto vvv = vdupq_n_f32(fC);
  for (int32_t n = init_pos, i = simdlen; i > 0; i -= 4, n += 8) {
    auto x0   = vld2q_f32(X + n + n0);
    auto x1   = vld2q_f32(X + n + n1);
    auto tmp  = vaddq_f32(x0.val[0], x1.val[0]);
    tmp       = vmulq_f32(tmp, vvv);
    x0.val[1] = vsubq_f32(x0.val[1], tmp);
    vst2q_f32(X + n + n0, x0);
  }
};

auto idwt_irrev97_fixed_neon_hor_step2 = [](const int32_t init_pos, const int32_t simdlen, float *const X,
                                            const int32_t n0, const int32_t n1) {
  auto vvv = vdupq_n_f32(fB);
  for (int32_t n = init_pos, i = simdlen; i > 0; i -= 4, n += 8) {
    auto x0   = vld2q_f32(X + n + n0);
    auto x1   = vld2q_f32(X + n + n1);
    auto tmp  = vaddq_f32(x0.val[0], x1.val[0]);
    tmp       = vmulq_f32(tmp, vvv);
    x0.val[1] = vsubq_f32(x0.val[1], tmp);
    vst2q_f32(X + n + n0, x0);
  }
};

auto idwt_irrev97_fixed_neon_hor_step3 = [](const int32_t init_pos, const int32_t simdlen, float *const X,
                                            const int32_t n0, const int32_t n1) {
  auto vvv = vdupq_n_f32(fA);
  for (int32_t n = init_pos, i = simdlen; i > 0; i -= 4, n += 8) {
    auto x0   = vld2q_f32(X + n + n0);
    auto x1   = vld2q_f32(X + n + n1);
    auto tmp  = vaddq_f32(x0.val[0], x1.val[0]);
    tmp       = vmulq_f32(tmp, vvv);
    // tmp       = vsubq_f32(tmp, vaddq_f32(x0.val[0], x1.val[0]));
    x0.val[1] = vsubq_f32(x0.val[1], tmp);
    vst2q_f32(X + n + n0, x0);
  }
};

void idwt_1d_filtr_irrev97_fixed_neon(sprec_t *X, const int32_t left, const int32_t u_i0,
                                      const int32_t u_i1) {
  const auto i0        = static_cast<int32_t>(u_i0);
  const auto i1        = static_cast<int32_t>(u_i1);
  const int32_t start  = i0 / 2;
  const int32_t stop   = i1 / 2;
  const int32_t offset = left - i0 % 2;

  // step 1
  int32_t simdlen = stop + 2 - (start - 1);
  idwt_irrev97_fixed_neon_hor_step0(offset - 2, simdlen, X, -1, 1);

  // step 2
  simdlen = stop + 1 - (start - 1);
  idwt_irrev97_fixed_neon_hor_step1(offset - 2, simdlen, X, 0, 2);

  // step 3
  simdlen = stop + 1 - start;
  idwt_irrev97_fixed_neon_hor_step2(offset, simdlen, X, -1, 1);

  // step 4
  simdlen = stop - start;
  idwt_irrev97_fixed_neon_hor_step3(offset, simdlen, X, 0, 2);
}

// reversible IDWT
void idwt_1d_filtr_rev53_fixed_neon(sprec_t *X, const int32_t left, const int32_t i0, const int32_t i1) {
  //  const auto i0        = static_cast<int32_t>(u_i0);
  //  const auto i1        = static_cast<int32_t>(u_i1);
  const int32_t start  = i0 / 2;
  const int32_t stop   = i1 / 2;
  const int32_t offset = left - i0 % 2;

  // step 1
  sprec_t *sp     = X + offset;
  int32_t simdlen = stop + 1 - start;
  auto xl0 = vld2q_f32(sp - 1);
  auto xl1 = vld2q_f32(sp + 1);
  for (; simdlen > 0; simdlen -= 4) {
    // (xl0.val[0] + xl1.val[0] + 2) >> 2;
    xl0.val[1] = vcvtq_f32_s32(
      vsubq_s32(vcvtq_s32_f32(xl0.val[1]), vrshrq_n_s32(vhaddq_s32(vcvtq_s32_f32(xl0.val[0]), vcvtq_s32_f32(xl1.val[0])), 1))
      );
    vst2q_f32(sp - 1, xl0);
    sp += 8;
    xl0 = vld2q_f32(sp - 1);
    xl1 = vld2q_f32(sp + 1);
  }

  // step 2
  sp      = X + offset;
  simdlen = stop - start;
  xl0     = vld2q_f32(sp);
  xl1     = vld2q_f32(sp + 2);
  for (; simdlen > 0; simdlen -= 4) {
    auto xout  = vhaddq_s32(vcvtq_s32_f32(xl0.val[0]), vcvtq_s32_f32(xl1.val[0]));
    xl0.val[1] = vcvtq_f32_s32(vaddq_s32(vcvtq_s32_f32(xl0.val[1]), xout));
    vst2q_f32(sp, xl0);
    sp += 8;
    xl0 = vld2q_f32(sp);
    xl1 = vld2q_f32(sp + 2);
  }
}

/********************************************************************************
 * vertical transform
 *******************************************************************************/
// irreversible IDWT
auto idwt_irrev97_fixed_neon_ver_step0 = [](const int32_t simdlen, float *const Xin0, float *const Xin1,
                                            float *const Xout) {
  auto vvv = vdupq_n_f32(fD);
  for (int32_t n = 0; n < simdlen; n += 4) {
    auto x0  = vld1q_f32(Xin0 + n);
    auto x2  = vld1q_f32(Xin1 + n);
    auto x1  = vld1q_f32(Xout + n);
    auto tmp = vaddq_f32(x0, x2);
    tmp      = vmulq_f32(tmp, vvv);
    x1       = vsubq_f32(x1, tmp);
    vst1q_f32(Xout + n, x1);
  }
};

auto idwt_irrev97_fixed_neon_ver_step1 = [](const int32_t simdlen, float *const Xin0, float *const Xin1,
                                            float *const Xout) {
  auto vvv = vdupq_n_f32(fC);
  for (int32_t n = 0; n < simdlen; n += 4) {
    auto x0  = vld1q_f32(Xin0 + n);
    auto x2  = vld1q_f32(Xin1 + n);
    auto x1  = vld1q_f32(Xout + n);
    auto tmp = vaddq_f32(x0, x2);
    tmp      = vmulq_f32(tmp, vvv);
    x1       = vsubq_f32(x1, tmp);
    vst1q_f32(Xout + n, x1);
  }
};

auto idwt_irrev97_fixed_neon_ver_step2 = [](const int32_t simdlen, float *const Xin0, float *const Xin1,
                                            float *const Xout) {
  auto vvv = vdupq_n_f32(fB);
  for (int32_t n = 0; n < simdlen; n += 4) {
    auto x0  = vld1q_f32(Xin0 + n);
    auto x2  = vld1q_f32(Xin1 + n);
    auto x1  = vld1q_f32(Xout + n);
    auto tmp = vaddq_f32(x0, x2);
    tmp      = vmulq_f32(tmp, vvv);
    x1       = vsubq_f32(x1, tmp);
    vst1q_f32(Xout + n, x1);
  }
};

auto idwt_irrev97_fixed_neon_ver_step3 = [](const int32_t simdlen, float *const Xin0, float *const Xin1,
                                            float *const Xout) {
  auto vvv = vdupq_n_f32(fA);
  for (int32_t n = 0; n < simdlen; n += 4) {
    auto x0  = vld1q_f32(Xin0 + n);
    auto x2  = vld1q_f32(Xin1 + n);
    auto x1  = vld1q_f32(Xout + n);
    auto tmp = vaddq_f32(x0, x2);
    // x1       = vaddq_f32(x1, tmp);
    tmp      = vmulq_f32(tmp, vvv);
    x1       = vsubq_f32(x1, tmp);
    vst1q_f32(Xout + n, x1);
  }
};


void idwt_irrev_ver_sr_fixed_neon(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                                  const int32_t v1, const int32_t stride) {
  constexpr int32_t num_pse_i0[2] = {3, 4};
  constexpr int32_t num_pse_i1[2] = {4, 3};
  const int32_t top               = num_pse_i0[v0 % 2];
  const int32_t bottom            = num_pse_i1[v1 % 2];
  if (v0 == v1 - 1) {
    // one sample case
    for (int32_t col = 0; col < u1 - u0; ++col) {
      // in[col] >>= (v0 % 2 == 0) ? 0 : 0;
    }
  } else {
    const int32_t len = round_up(stride, SIMD_PADDING);
    auto **buf        = new sprec_t *[static_cast<size_t>(top + v1 - v0 + bottom)];
    for (int32_t i = 1; i <= top; ++i) {
      buf[top - i] =
          static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * static_cast<size_t>(len), 32));
      memcpy(buf[top - i], &in[PSEo(v0 - i, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
    }
    for (int32_t row = 0; row < v1 - v0; ++row) {
      buf[top + row] = &in[row * stride];
    }
    for (int32_t i = 1; i <= bottom; i++) {
      buf[top + (v1 - v0) + i - 1] =
          static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * static_cast<size_t>(len), 32));
      memcpy(buf[top + (v1 - v0) + i - 1], &in[PSEo(v1 - v0 + i - 1 + v0, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
    }
    const int32_t start  = v0 / 2;
    const int32_t stop   = v1 / 2;
    const int32_t offset = top - v0 % 2;

    const int32_t simdlen = (u1 - u0) - (u1 - u0) % 16;
    for (int32_t n = -2 + offset, i = start - 1; i < stop + 2; i++, n += 2) {
      idwt_irrev97_fixed_neon_ver_step0(simdlen, buf[n - 1], buf[n + 1], buf[n]);
      for (int32_t col = simdlen; col < u1 - u0; ++col) {
        float sum = buf[n - 1][col];
        sum += buf[n + 1][col];
        buf[n][col] -= sum * fD; //(sprec_t)((Dcoeff * sum + Doffset) >> Dshift);
      }
    }
    for (int32_t n = -2 + offset, i = start - 1; i < stop + 1; i++, n += 2) {
      idwt_irrev97_fixed_neon_ver_step1(simdlen, buf[n], buf[n + 2], buf[n + 1]);
      for (int32_t col = simdlen; col < u1 - u0; ++col) {
        float sum = buf[n][col];
        sum += buf[n + 2][col];
        buf[n + 1][col] -= sum * fC; //(sprec_t)((Ccoeff * sum + Coffset) >> Cshift);
      }
    }
    for (int32_t n = 0 + offset, i = start; i < stop + 1; i++, n += 2) {
      idwt_irrev97_fixed_neon_ver_step2(simdlen, buf[n - 1], buf[n + 1], buf[n]);
      for (int32_t col = simdlen; col < u1 - u0; ++col) {
        float sum = buf[n - 1][col];
        sum += buf[n + 1][col];
        buf[n][col] -= sum * fB; //(sprec_t)((Bcoeff * sum + Boffset) >> Bshift);
      }
    }
    for (int32_t n = 0 + offset, i = start; i < stop; i++, n += 2) {
      idwt_irrev97_fixed_neon_ver_step3(simdlen, buf[n], buf[n + 2], buf[n + 1]);
      for (int32_t col = simdlen; col < u1 - u0; ++col) {
        float sum = buf[n][col];
        sum += buf[n + 2][col];
        buf[n + 1][col] -= sum * fA; //(sprec_t)((Acoeff * sum + Aoffset) >> Ashift);
      }
    }

    for (int32_t i = 1; i <= top; ++i) {
      aligned_mem_free(buf[top - i]);
    }
    for (int32_t i = 1; i <= bottom; i++) {
      aligned_mem_free(buf[top + (v1 - v0) + i - 1]);
    }
    delete[] buf;
  }
}

// reversible IDWT
void idwt_rev_ver_sr_fixed_neon(sprec_t *in, const int32_t u0, const int32_t u1, const int32_t v0,
                                const int32_t v1, const int32_t stride) {
  constexpr int32_t num_pse_i0[2] = {1, 2};
  constexpr int32_t num_pse_i1[2] = {2, 1};
  const int32_t top               = num_pse_i0[v0 % 2];
  const int32_t bottom            = num_pse_i1[v1 % 2];
  if (v0 == v1 - 1) {
    // one sample case
    for (int32_t col = 0; col < u1 - u0; ++col) {
      in[col] = (v0 % 2 == 0) ? in[col] : (int32_t)(in[col] / 2);
    }
  } else {
    const int32_t len = round_up(stride, SIMD_PADDING);
    auto **buf        = new sprec_t *[static_cast<size_t>(top + v1 - v0 + bottom)];
    for (int32_t i = 1; i <= top; ++i) {
      buf[top - i] =
          static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * static_cast<size_t>(len), 32));
      memcpy(buf[top - i], &in[PSEo(v0 - i, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
    }
    for (int32_t row = 0; row < v1 - v0; ++row) {
      buf[top + row] = &in[row * stride];
    }
    for (int32_t i = 1; i <= bottom; i++) {
      buf[top + (v1 - v0) + i - 1] =
          static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * static_cast<size_t>(len), 32));
      memcpy(buf[top + (v1 - v0) + i - 1], &in[PSEo(v1 - v0 + i - 1 + v0, v0, v1) * stride],
             sizeof(sprec_t) * static_cast<size_t>(stride));
    }
    const int32_t start  = v0 / 2;
    const int32_t stop   = v1 / 2;
    const int32_t offset = top - v0 % 2;

    int32_t simdlen = (u1 - u0) - (u1 - u0) % 8;
    float32x4_t vin00, vout0, vin10, vin01, vout1, vin11;
    for (int32_t n = 0 + offset, i = start; i < stop + 1; ++i, n += 2) {
      float *x0 = buf[n - 1];
      float *x1 = buf[n];
      float *x2 = buf[n + 1];
      // openhtj2k_arm_prefetch(x0);
      // openhtj2k_arm_prefetch(x1);
      // openhtj2k_arm_prefetch(x2);
      for (int32_t col = 0; col < simdlen; col += 8) {
        vin00 = vld1q_f32(x0);
        vin01 = vld1q_f32(x0 + 4);
        vout0 = vld1q_f32(x1);
        vout1 = vld1q_f32(x1 + 4);
        vin10 = vld1q_f32(x2);
        vin11 = vld1q_f32(x2 + 4);
        vout0 = vsubq_f32(vout0, vcvtq_f32_s32(vrshrq_n_s32(vhaddq_s32(vcvtq_s32_f32(vin00), vcvtq_s32_f32(vin10)), 1)));
        vout1 = vsubq_f32(vout1, vcvtq_f32_s32(vrshrq_n_s32(vhaddq_s32(vcvtq_s32_f32(vin01), vcvtq_s32_f32(vin11)), 1)));
        vst1q_f32(x1, vout0);
        vst1q_f32(x1 + 4, vout1);
        x0 += 8;
        x1 += 8;
        x2 += 8;
        // openhtj2k_arm_prefetch(x0);
        // openhtj2k_arm_prefetch(x1);
        // openhtj2k_arm_prefetch(x2);
      }
      for (int32_t col = simdlen; col < u1 - u0; ++col) {
        int32_t sum = (int32_t)*x0++;
        sum += (int32_t)*x2++;
        *x1++ -= static_cast<float>((sum + 2) >> 2);
      }
    }
    for (int32_t n = 0 + offset, i = start; i < stop; ++i, n += 2) {
      float *x0 = buf[n];
      float *x1 = buf[n + 1];
      float *x2 = buf[n + 2];
      // openhtj2k_arm_prefetch(x0);
      // openhtj2k_arm_prefetch(x1);
      // openhtj2k_arm_prefetch(x2);
      for (int32_t col = 0; col < simdlen; col += 8) {
        vin00 = vld1q_f32(x0);
        vin01 = vld1q_f32(x0 + 4);
        vout0 = vld1q_f32(x1);
        vout1 = vld1q_f32(x1 + 4);
        vin10 = vld1q_f32(x2);
        vin11 = vld1q_f32(x2 + 4);
        vout0 = vaddq_f32(vout0, vcvtq_f32_s32(vhaddq_s32(vcvtq_s32_f32(vin00), vcvtq_s32_f32(vin10))));
        vout1 = vaddq_f32(vout1, vcvtq_f32_s32(vhaddq_s32(vcvtq_s32_f32(vin01), vcvtq_s32_f32(vin11))));
        vst1q_f32(x1, vout0);
        vst1q_f32(x1 + 4, vout1);
        x0 += 8;
        x1 += 8;
        x2 += 8;
        // openhtj2k_arm_prefetch(x0);
        // openhtj2k_arm_prefetch(x1);
        // openhtj2k_arm_prefetch(x2);
      }
      for (int32_t col = simdlen; col < u1 - u0; ++col) {
        int32_t sum = (int32_t)*x0++;
        sum += (int32_t)*x2++;
        *x1++ += static_cast<float>(sum >> 1);
      }
    }

    for (int32_t i = 1; i <= top; ++i) {
      aligned_mem_free(buf[top - i]);
    }
    for (int32_t i = 1; i <= bottom; i++) {
      aligned_mem_free(buf[top + (v1 - v0) + i - 1]);
    }
    delete[] buf;
  }
}

#endif