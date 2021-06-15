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

#if defined(__AVX2__)
  #include "utils.hpp"
  #include "color.hpp"

static constexpr uint32_t NUM_I32_VECTOR_AVX2 = 8;

void cvt_rgb_to_ycbcr_rev_avx2(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t num_tc_samples) {
  for (uint32_t n = 0; n < round_down(num_tc_samples, NUM_I32_VECTOR_AVX2); n += NUM_I32_VECTOR_AVX2) {
    __m256i mR              = *((__m256i *)(sp0 + n));
    __m256i mG              = *((__m256i *)(sp1 + n));
    __m256i mB              = *((__m256i *)(sp2 + n));
    __m256i tmp             = _mm256_slli_epi32(mG, 1);
    *((__m256i *)(sp1 + n)) = _mm256_sub_epi32(mB, mG);
    *((__m256i *)(sp2 + n)) = _mm256_sub_epi32(mR, mG);
    *((__m256i *)(sp0 + n)) = _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(mR, mB), tmp), 2);
  }
  int32_t R, G, B;
  int32_t Y, Cb, Cr;
  for (uint32_t n = round_down(num_tc_samples, NUM_I32_VECTOR_AVX2); n < num_tc_samples; ++n) {
    R      = sp0[n];
    G      = sp1[n];
    B      = sp2[n];
    Y      = (R + 2 * G + B) >> 2;
    Cb     = B - G;
    Cr     = R - G;
    sp0[n] = Y;
    sp1[n] = Cb;
    sp2[n] = Cr;
  }
}

void cvt_rgb_to_ycbcr_irrev_avx2(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t num_tc_samples) {
  __m256 mALPHA_R = _mm256_set1_ps(ALPHA_R);
  __m256 mALPHA_G = _mm256_set1_ps(ALPHA_G);
  __m256 mALPHA_B = _mm256_set1_ps(ALPHA_B);
  __m256 mCB_FACT = _mm256_set1_ps(1.0 / CB_FACT_B);
  __m256 mCR_FACT = _mm256_set1_ps(1.0 / CR_FACT_R);
  for (uint32_t n = 0; n < round_down(num_tc_samples, NUM_I32_VECTOR_AVX2); n += NUM_I32_VECTOR_AVX2) {
    // lossy:ICT
    __m256 mR  = _mm256_cvtepi32_ps(*((__m256i *)(sp0 + n)));
    __m256 mG  = _mm256_cvtepi32_ps(*((__m256i *)(sp1 + n)));
    __m256 mB  = _mm256_cvtepi32_ps(*((__m256i *)(sp2 + n)));
    __m256 mY  = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(mR, mALPHA_R), _mm256_mul_ps(mG, mALPHA_G)),
                              _mm256_mul_ps(mB, mALPHA_B));
    __m256 mCb = _mm256_mul_ps(mCB_FACT, _mm256_sub_ps(mB, mY));
    __m256 mCr = _mm256_mul_ps(mCR_FACT, _mm256_sub_ps(mR, mY));
    *((__m256i *)(sp0 + n)) = _mm256_cvtps_epi32(_mm256_round_ps(mY, 0));
    *((__m256i *)(sp1 + n)) = _mm256_cvtps_epi32(_mm256_round_ps(mCb, 0));
    *((__m256i *)(sp2 + n)) = _mm256_cvtps_epi32(_mm256_round_ps(mCr, 0));
  }
  double fR, fG, fB;
  double fY, fCb, fCr;
  for (uint32_t n = round_down(num_tc_samples, NUM_I32_VECTOR_AVX2); n < num_tc_samples; n++) {
    // lossy:ICT
    fR     = static_cast<double>(sp0[n]);
    fG     = static_cast<double>(sp1[n]);
    fB     = static_cast<double>(sp2[n]);
    fY     = ALPHA_R * fR + ALPHA_G * fG + ALPHA_B * fB;
    fCb    = (1.0 / CB_FACT_B) * (fB - fY);
    fCr    = (1.0 / CR_FACT_R) * (fR - fY);
    sp0[n] = round_d(fY);
    sp1[n] = round_d(fCb);
    sp2[n] = round_d(fCr);
  }
}

void cvt_ycbcr_to_rgb_rev_avx2(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t num_tc_samples) {
  for (uint32_t n = 0; n < round_down(num_tc_samples, NUM_I32_VECTOR_AVX2); n += NUM_I32_VECTOR_AVX2) {
    __m256i mCb             = *((__m256i *)(sp1 + n));
    __m256i mCr             = *((__m256i *)(sp2 + n));
    __m256i mY              = *((__m256i *)(sp0 + n));
    __m256i tmp             = _mm256_add_epi32(mCb, mCr);
    tmp                     = _mm256_srai_epi32(tmp, 2);  //(Cb + Cr) >> 2
    __m256i mG              = _mm256_sub_epi32(mY, tmp);
    *((__m256i *)(sp1 + n)) = mG;
    *((__m256i *)(sp0 + n)) = _mm256_add_epi32(mCr, mG);
    *((__m256i *)(sp2 + n)) = _mm256_add_epi32(mCb, mG);
  }
  int32_t R, G, B;
  int32_t Y, Cb, Cr;
  for (uint32_t n = round_down(num_tc_samples, NUM_I32_VECTOR_AVX2); n < num_tc_samples; ++n) {
    Y      = sp0[n];
    Cb     = sp1[n];
    Cr     = sp2[n];
    G      = Y - ((Cb + Cr) >> 2);
    R      = Cr + G;
    B      = Cb + G;
    sp0[n] = R;
    sp1[n] = G;
    sp2[n] = B;
  }
}

void cvt_ycbcr_to_rgb_irrev_avx2(int32_t *sp0, int32_t *sp1, int32_t *sp2, uint32_t num_tc_samples) {
  __m256 mCR_FACT_R = _mm256_set1_ps(CR_FACT_R);
  __m256 mCR_FACT_G = _mm256_set1_ps(CR_FACT_G);
  __m256 mCB_FACT_B = _mm256_set1_ps(CB_FACT_B);
  __m256 mCB_FACT_G = _mm256_set1_ps(CB_FACT_G);
  for (uint32_t n = 0; n < round_down(num_tc_samples, NUM_I32_VECTOR_AVX2); n += NUM_I32_VECTOR_AVX2) {
    __m256 mY  = _mm256_cvtepi32_ps(*((__m256i *)(sp0 + n)));
    __m256 mCb = _mm256_cvtepi32_ps(*((__m256i *)(sp1 + n)));
    __m256 mCr = _mm256_cvtepi32_ps(*((__m256i *)(sp2 + n)));
    __m256 mR  = _mm256_fmadd_ps(mCr, mCR_FACT_R, mY);
    __m256 mB  = _mm256_fmadd_ps(mCb, mCB_FACT_B, mY);
    __m256 mG  = _mm256_sub_ps(_mm256_fnmadd_ps(mCr, mCR_FACT_G, mY), _mm256_mul_ps(mCb, mCB_FACT_G));
    *((__m256i *)(sp0 + n)) = _mm256_cvtps_epi32(_mm256_round_ps(mR, 0));
    *((__m256i *)(sp1 + n)) = _mm256_cvtps_epi32(_mm256_round_ps(mG, 0));
    *((__m256i *)(sp2 + n)) = _mm256_cvtps_epi32(_mm256_round_ps(mB, 0));
  }
  int32_t R, G, B;
  double fY, fCb, fCr;
  for (uint32_t n = round_down(num_tc_samples, NUM_I32_VECTOR_AVX2); n < num_tc_samples; n++) {
    fY     = static_cast<double>(sp0[n]);
    fCb    = static_cast<double>(sp1[n]);
    fCr    = static_cast<double>(sp2[n]);
    R      = static_cast<int32_t>(round_d(fY + CR_FACT_R * fCr));
    B      = static_cast<int32_t>(round_d(fY + CB_FACT_B * fCb));
    G      = static_cast<int32_t>(round_d(fY - CR_FACT_G * fCr - CB_FACT_G * fCb));
    sp0[n] = R;
    sp1[n] = G;
    sp2[n] = B;
  }
}
#endif
