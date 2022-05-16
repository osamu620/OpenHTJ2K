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

#if defined(OPENHTJ2K_ENABLE_ARM_NEON)
  #include "dwt.hpp"
  #include "utils.hpp"
  #include <cstring>

/********************************************************************************
 * horizontal transforms
 *******************************************************************************/
// irreversible IDWT
auto idwt_irrev97_fixed_neon_hor_step0 = [](const int32_t init_pos, const int32_t simdlen, int16_t *const X,
                                            const int32_t n0, const int32_t n1, const int32_t coeff,
                                            const int32_t offset, const int32_t shift) {
  auto vvv = vdupq_n_s16((int16_t)14533);
  for (int32_t n = init_pos, i = simdlen; i > 0; i -= 8, n += 16) {
    auto x0  = vld2q_s16(X + n + n0);
    auto x1  = vld2q_s16(X + n + n1);
    auto tmp = (x0.val[0] + x1.val[0]);
    tmp      = vqrdmulhq_s16(tmp, vvv);
    x0.val[1] -= tmp;
    vst2q_s16(X + n + n0, x0);
  }
};

auto idwt_irrev97_fixed_neon_hor_step1 = [](const int32_t init_pos, const int32_t simdlen, int16_t *const X,
                                            const int32_t n0, const int32_t n1, const int32_t coeff,
                                            const int32_t offset, const int32_t shift) {
  auto vvv = vdupq_n_s16((int16_t)28931);
  for (int32_t n = init_pos, i = simdlen; i > 0; i -= 8, n += 16) {
    auto x0  = vld2q_s16(X + n + n0);
    auto x1  = vld2q_s16(X + n + n1);
    auto tmp = (x0.val[0] + x1.val[0]);
    tmp      = vqrdmulhq_s16(tmp, vvv);
    x0.val[1] -= tmp;
    vst2q_s16(X + n + n0, x0);
  }
};

auto idwt_irrev97_fixed_neon_hor_step2 = [](const int32_t init_pos, const int32_t simdlen, int16_t *const X,
                                            const int32_t n0, const int32_t n1, const int32_t coeff,
                                            const int32_t offset, const int32_t shift) {
  auto vvv = vdupq_n_s16((int16_t)-3472);
  for (int32_t n = init_pos, i = simdlen; i > 0; i -= 8, n += 16) {
    auto x0  = vld2q_s16(X + n + n0);
    auto x1  = vld2q_s16(X + n + n1);
    auto tmp = vhaddq_s16(x0.val[0], x1.val[0]);
    tmp      = vqrdmulhq_s16(tmp, vvv);
    x0.val[1] -= tmp;
    vst2q_s16(X + n + n0, x0);
  }
};

auto idwt_irrev97_fixed_neon_hor_step3 = [](const int32_t init_pos, const int32_t simdlen, int16_t *const X,
                                            const int32_t n0, const int32_t n1, const int32_t coeff,
                                            const int32_t offset, const int32_t shift) {
  auto vvv = vdupq_n_s16((int16_t)-19206);
  for (int32_t n = init_pos, i = simdlen; i > 0; i -= 8, n += 16) {
    auto x0  = vld2q_s16(X + n + n0);
    auto x1  = vld2q_s16(X + n + n1);
    auto tmp = (x0.val[0] + x1.val[0]);
    tmp      = vqrdmulhq_s16(tmp, vvv);
    tmp -= (x0.val[0] + x1.val[0]);
    x0.val[1] -= tmp;
    vst2q_s16(X + n + n0, x0);
  }
};

auto idwt_irrev97_fixed_neon_hor_step = [](const int32_t init_pos, const int32_t simdlen, int16_t *const X,
                                           const int32_t n0, const int32_t n1, const int32_t coeff,
                                           const int32_t offset, const int32_t shift) {
  auto vcoeff  = vdupq_n_s32(coeff);
  auto voffset = vdupq_n_s32(offset);
  for (int32_t n = init_pos, i = 0; i < simdlen; i += 8, n += 16) {
    auto xl0   = vld2q_s16(X + n + n0);
    auto xl1   = vld2q_s16(X + n + n1);
    auto x0    = vreinterpretq_s32_s16(xl0.val[0]);
    auto x0l   = vmovl_s16(vreinterpret_s16_s32(vget_low_s32(x0)));
    auto x0h   = vmovl_s16(vreinterpret_s16_s32(vget_high_s32(x0)));
    auto x2    = vreinterpretq_s32_s16(xl1.val[0]);
    auto x2l   = vmovl_s16(vreinterpret_s16_s32(vget_low_s32(x2)));
    auto x2h   = vmovl_s16(vreinterpret_s16_s32(vget_high_s32(x2)));
    auto xoutl = ((x0l + x2l) * vcoeff + voffset) >> shift;
    auto xouth = ((x0h + x2h) * vcoeff + voffset) >> shift;
    xl0.val[1] -= vcombine_s16(vmovn_s32(xoutl), vmovn_s32(xouth));
    vst2q_s16(X + n + n0, xl0);
  }
};

void idwt_1d_filtr_irrev97_fixed_neon(sprec_t *X, const int32_t left, const int32_t right,
                                      const uint32_t u_i0, const uint32_t u_i1) {
  const auto i0        = static_cast<const int32_t>(u_i0);
  const auto i1        = static_cast<const int32_t>(u_i1);
  const int32_t start  = i0 / 2;
  const int32_t stop   = i1 / 2;
  const int32_t offset = left - i0 % 2;

  // step 1
  int32_t simdlen = stop + 2 - (start - 1);
  idwt_irrev97_fixed_neon_hor_step0(offset - 2, simdlen, X, -1, 1, Dcoeff, Doffset, Dshift);

  // step 2
  simdlen = stop + 1 - (start - 1);
  idwt_irrev97_fixed_neon_hor_step1(offset - 2, simdlen, X, 0, 2, Ccoeff, Coffset, Cshift);

  // step 3
  simdlen = stop + 1 - start;
  idwt_irrev97_fixed_neon_hor_step2(offset, simdlen, X, -1, 1, Bcoeff, Boffset, Bshift);

  // step 4
  simdlen = stop - start;
  idwt_irrev97_fixed_neon_hor_step3(offset, simdlen, X, 0, 2, Acoeff, Aoffset, Ashift);
}

// reversible IDWT
void idwt_1d_filtr_rev53_fixed_neon(sprec_t *X, const int32_t left, const int32_t right,
                                    const uint32_t u_i0, const uint32_t u_i1) {
  const auto i0        = static_cast<const int32_t>(u_i0);
  const auto i1        = static_cast<const int32_t>(u_i1);
  const int32_t start  = i0 / 2;
  const int32_t stop   = i1 / 2;
  const int32_t offset = left - i0 % 2;

  // step 1
  int32_t simdlen = stop + 1 - start;
  for (int32_t n = 0 + offset, i = 0; i < simdlen; i += 8, n += 16) {
    auto xl0 = vld2q_s16(X + n - 1);
    auto xl1 = vld2q_s16(X + n + 1);
    // (xl0.val[0] + xl1.val[0] + 2) >> 2;
    xl0.val[1] -= vrshrq_n_s16(vhaddq_s16(xl0.val[0], xl1.val[0]), 1);
    vst2q_s16(X + n - 1, xl0);
  }

  // step 2
  simdlen = stop - start;
  for (int32_t n = 0 + offset, i = 0; i < simdlen; i += 8, n += 16) {
    auto xl0  = vld2q_s16(X + n);
    auto xl1  = vld2q_s16(X + n + 2);
    auto xout = vhaddq_s16(xl0.val[0], xl1.val[0]);
    xl0.val[1] += xout;
    vst2q_s16(X + n, xl0);
  }
}

/********************************************************************************
 * vertical transform
 *******************************************************************************/
// irreversible IDWT
auto idwt_irrev97_fixed_neon_ver_step = [](const int32_t simdlen, int16_t *const Xin0, int16_t *const Xin1,
                                           int16_t *const Xout, const int32_t coeff, const int32_t offset,
                                           const int32_t shift) {
  auto vcoeff  = vdupq_n_s32(coeff);
  auto voffset = vdupq_n_s32(offset);
  for (int32_t n = 0; n < simdlen; n += 8) {
    auto x0    = vld1q_s16(Xin0 + n);
    auto x2    = vld1q_s16(Xin1 + n);
    auto x1    = vld1q_s16(Xout + n);
    auto x0l   = vmovl_s16(vreinterpret_s16_s32(vget_low_s32(vreinterpretq_s32_s16(x0))));
    auto x0h   = vmovl_s16(vreinterpret_s16_s32(vget_high_s32(vreinterpretq_s32_s16(x0))));
    auto x2l   = vmovl_s16(vreinterpret_s16_s32(vget_low_s32(vreinterpretq_s32_s16(x2))));
    auto x2h   = vmovl_s16(vreinterpret_s16_s32(vget_high_s32(vreinterpretq_s32_s16(x2))));
    auto xoutl = ((x0l + x2l) * vcoeff + voffset) >> shift;
    auto xouth = ((x0h + x2h) * vcoeff + voffset) >> shift;
    x1 -= vcombine_s16(vmovn_s32(xoutl), vmovn_s32(xouth));
    vst1q_s16(Xout + n, x1);
  }
};

void idwt_irrev_ver_sr_fixed_neon(sprec_t *in, const uint32_t u0, const uint32_t u1, const uint32_t v0,
                                  const uint32_t v1) {
  const uint32_t stride           = u1 - u0;
  constexpr int32_t num_pse_i0[2] = {3, 4};
  constexpr int32_t num_pse_i1[2] = {4, 3};
  const int32_t top               = num_pse_i0[v0 % 2];
  const int32_t bottom            = num_pse_i1[v1 % 2];
  if (v0 == v1 - 1) {
    // one sample case
    for (uint32_t col = 0; col < u1 - u0; ++col) {
      in[col] >>= (v0 % 2 == 0) ? 0 : 0;
    }
  } else {
    const uint32_t len = round_up(stride, SIMD_PADDING);
    auto **buf         = new sprec_t *[top + v1 - v0 + bottom];
    for (uint32_t i = 1; i <= top; ++i) {
      buf[top - i] = static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * len, 32));
      memcpy(buf[top - i], &in[(PSEo(v0 - i, v0, v1) - v0) * stride], sizeof(sprec_t) * stride);
    }
    for (uint32_t row = 0; row < v1 - v0; ++row) {
      buf[top + row] = &in[row * stride];
    }
    for (uint32_t i = 1; i <= bottom; i++) {
      buf[top + (v1 - v0) + i - 1] = static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * len, 32));
      memcpy(buf[top + (v1 - v0) + i - 1], &in[(PSEo(v1 - v0 + i - 1 + v0, v0, v1) - v0) * stride],
             sizeof(sprec_t) * stride);
    }
    const int32_t start  = v0 / 2;
    const int32_t stop   = v1 / 2;
    const int32_t offset = top - v0 % 2;

    const int32_t simdlen = (u1 - u0) - (u1 - u0) % 16;
    for (int32_t n = -2 + offset, i = start - 1; i < stop + 2; i++, n += 2) {
      idwt_irrev97_fixed_neon_ver_step(simdlen, buf[n - 1], buf[n + 1], buf[n], Dcoeff, Doffset, Dshift);
      for (uint32_t col = simdlen; col < u1 - u0; ++col) {
        int32_t sum = buf[n - 1][col];
        sum += buf[n + 1][col];
        buf[n][col] -= (sprec_t)((Dcoeff * sum + Doffset) >> Dshift);
      }
    }
    for (int32_t n = -2 + offset, i = start - 1; i < stop + 1; i++, n += 2) {
      idwt_irrev97_fixed_neon_ver_step(simdlen, buf[n], buf[n + 2], buf[n + 1], Ccoeff, Coffset, Cshift);
      for (uint32_t col = simdlen; col < u1 - u0; ++col) {
        int32_t sum = buf[n][col];
        sum += buf[n + 2][col];
        buf[n + 1][col] -= (sprec_t)((Ccoeff * sum + Coffset) >> Cshift);
      }
    }
    for (int32_t n = 0 + offset, i = start; i < stop + 1; i++, n += 2) {
      idwt_irrev97_fixed_neon_ver_step(simdlen, buf[n - 1], buf[n + 1], buf[n], Bcoeff, Boffset, Bshift);
      for (uint32_t col = simdlen; col < u1 - u0; ++col) {
        int32_t sum = buf[n - 1][col];
        sum += buf[n + 1][col];
        buf[n][col] -= (sprec_t)((Bcoeff * sum + Boffset) >> Bshift);
      }
    }
    for (int32_t n = 0 + offset, i = start; i < stop; i++, n += 2) {
      idwt_irrev97_fixed_neon_ver_step(simdlen, buf[n], buf[n + 2], buf[n + 1], Acoeff, Aoffset, Ashift);
      for (uint32_t col = simdlen; col < u1 - u0; ++col) {
        int32_t sum = buf[n][col];
        sum += buf[n + 2][col];
        buf[n + 1][col] -= (sprec_t)((Acoeff * sum + Aoffset) >> Ashift);
      }
    }

    for (uint32_t i = 1; i <= top; ++i) {
      aligned_mem_free(buf[top - i]);
    }
    for (uint32_t i = 1; i <= bottom; i++) {
      aligned_mem_free(buf[top + (v1 - v0) + i - 1]);
    }
    delete[] buf;
  }
}

// reversible IDWT
void idwt_rev_ver_sr_fixed_neon(sprec_t *in, const uint32_t u0, const uint32_t u1, const uint32_t v0,
                                const uint32_t v1) {
  const uint32_t stride           = u1 - u0;
  constexpr int32_t num_pse_i0[2] = {1, 2};
  constexpr int32_t num_pse_i1[2] = {2, 1};
  const int32_t top               = num_pse_i0[v0 % 2];
  const int32_t bottom            = num_pse_i1[v1 % 2];
  if (v0 == v1 - 1) {
    // one sample case
    for (uint32_t col = 0; col < u1 - u0; ++col) {
      in[col] >>= (v0 % 2 == 0) ? 0 : 1;
    }
  } else {
    const uint32_t len = round_up(stride, SIMD_PADDING);
    auto **buf         = new sprec_t *[top + v1 - v0 + bottom];
    for (uint32_t i = 1; i <= top; ++i) {
      buf[top - i] = static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * len, 32));
      memcpy(buf[top - i], &in[(PSEo(v0 - i, v0, v1) - v0) * stride], sizeof(sprec_t) * stride);
    }
    for (uint32_t row = 0; row < v1 - v0; ++row) {
      buf[top + row] = &in[row * stride];
    }
    for (uint32_t i = 1; i <= bottom; i++) {
      buf[top + (v1 - v0) + i - 1] = static_cast<sprec_t *>(aligned_mem_alloc(sizeof(sprec_t) * len, 32));
      memcpy(buf[top + (v1 - v0) + i - 1], &in[(PSEo(v1 - v0 + i - 1 + v0, v0, v1) - v0) * stride],
             sizeof(sprec_t) * stride);
    }
    const int32_t start  = v0 / 2;
    const int32_t stop   = v1 / 2;
    const int32_t offset = top - v0 % 2;

    int32_t simdlen = (u1 - u0) - (u1 - u0) % 8;
    for (int32_t n = 0 + offset, i = start; i < stop + 1; ++i, n += 2) {
      for (uint32_t col = 0; col < simdlen; col += 8) {
        auto X0 = vld1q_s16(&buf[n - 1][col]);
        auto X2 = vld1q_s16(&buf[n + 1][col]);
        auto X1 = vld1q_s16(&buf[n][col]);
        X1 -= vrshrq_n_s16(vhaddq_s16(X0, X2), 1);
        vst1q_s16(&buf[n][col], X1);
      }
      for (uint32_t col = simdlen; col < u1 - u0; ++col) {
        int32_t sum = buf[n - 1][col];
        sum += buf[n + 1][col];
        buf[n][col] -= ((sum + 2) >> 2);
      }
    }
    for (int32_t n = 0 + offset, i = start; i < stop; ++i, n += 2) {
      for (uint32_t col = 0; col < simdlen; col += 8) {
        auto X0 = vld1q_s16(&buf[n][col]);
        auto X2 = vld1q_s16(&buf[n + 2][col]);
        auto X1 = vld1q_s16(&buf[n + 1][col]);
        X1 += vhaddq_s16(X0, X2);
        vst1q_s16(&buf[n + 1][col], X1);
      }
      for (uint32_t col = simdlen; col < u1 - u0; ++col) {
        int32_t sum = buf[n][col];
        sum += buf[n + 2][col];
        buf[n + 1][col] += (sum >> 1);
      }
    }

    for (uint32_t i = 1; i <= top; ++i) {
      aligned_mem_free(buf[top - i]);
    }
    for (uint32_t i = 1; i <= bottom; i++) {
      aligned_mem_free(buf[top + (v1 - v0) + i - 1]);
    }
    delete[] buf;
  }
}

#endif