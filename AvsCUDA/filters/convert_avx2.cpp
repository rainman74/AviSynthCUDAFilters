// Avisynth v2.5.  Copyright 2002-2009 Ben Rudiak-Gould et al.
// http://www.avisynth.org

// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA, or visit
// http://www.gnu.org/copyleft/gpl.html .
//
// Linking Avisynth statically or dynamically with other modules is making a
// combined work based on Avisynth.  Thus, the terms and conditions of the GNU
// General Public License cover the whole combination.
//
// As a special exception, the copyright holders of Avisynth give you
// permission to link Avisynth with independent modules that communicate with
// Avisynth solely through the interfaces defined in avisynth.h, regardless of the license
// terms of these independent modules, and to copy and distribute the
// resulting combined work under terms of your choice, provided that
// every copy of the combined work is accompanied by a complete copy of
// the source code of Avisynth (the version of Avisynth used to produce the
// combined work), being distributed under the terms of the GNU General
// Public License plus this exception.  An independent module is a module
// which is not derived from or based on Avisynth, such as 3rd-party filters,
// import and export plugins, or graphical user interfaces.


#include <avs/alignment.h>
#include "rgy_osdep.h"
#include <emmintrin.h>
#include <immintrin.h>

#if _MSC_VER <= 1900
#define constexpr(expr) (expr)
#endif

#ifndef _mm256_set_m128i
#define _mm256_set_m128i(v0, v1) _mm256_insertf128_si256(_mm256_castsi128_si256(v1), (v0), 1)
#endif

#ifndef _mm256_set_m128
#define _mm256_set_m128(v0, v1) _mm256_insertf128_ps(_mm256_castps128_ps256(v1), (v0), 1)
#endif

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4305 4309)
#endif

template<typename pixel_t, uint8_t targetbits, bool chroma, bool fulls, bool fulld>
void convert_32_to_uintN_avx2(const BYTE *srcp8, BYTE *dstp8, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
  const float *srcp = reinterpret_cast<const float *>(srcp8);
  pixel_t *dstp = reinterpret_cast<pixel_t *>(dstp8);

  src_pitch = src_pitch / sizeof(float);
  dst_pitch = dst_pitch / sizeof(pixel_t);

  int src_width = src_rowsize / sizeof(float);

  const int max_pixel_value = (1 << targetbits) - 1;
  const __m256i max_pixel_value_256 = _mm256_set1_epi16(max_pixel_value);

  const int limit_lo_d = (fulld ? 0 : 16) << (targetbits - 8);
  const int limit_hi_d = fulld ? ((1 << targetbits) - 1) : ((chroma ? 240 : 235) << (targetbits - 8));
  const float range_diff_d = (float)limit_hi_d - limit_lo_d;

  const int limit_lo_s = fulls ? 0 : 16;
  const int limit_hi_s = fulls ? 255 : (chroma ? 240 : 235);
  const float range_diff_s = (limit_hi_s - limit_lo_s) / 255.0f;

  // fulls fulld luma             luma_new   chroma                          chroma_new
  // true  false 0..1              16-235     -0.5..0.5                      16-240       Y = Y * ((235-16) << (bpp-8)) + 16, Chroma= Chroma * ((240-16) << (bpp-8)) + 16
  // true  true  0..1               0-255     -0.5..0.5                      0-128-255
  // false false 16/255..235/255   16-235     (16-128)/255..(240-128)/255    16-240
  // false true  16/255..235/255    0..1      (16-128)/255..(240-128)/255    0-128-255
  const float factor = range_diff_d / range_diff_s;

  const float half_i = (float)(1 << (targetbits - 1));
#ifdef FLOAT_CHROMA_IS_HALF_CENTERED
  const __m256 half_ps = _mm256_set1_ps(0.5f);
#endif
  const __m256 halfint_plus_rounder_ps = _mm256_set1_ps(half_i + 0.5f);
  const __m256 limit_lo_s_ps = _mm256_set1_ps(limit_lo_s / 255.0f);
  const __m256 limit_lo_plus_rounder_ps = _mm256_set1_ps(limit_lo_d + 0.5f);

  __m256 factor_ps = _mm256_set1_ps(factor);

  for (int y = 0; y < src_height; y++)
  {
    for (int x = 0; x < src_width; x += 16) // 16 pixels at a time (64 byte - alignment is OK)
    {
      __m256i result;
      __m256i result_0, result_1;
      __m256 src_0 = _mm256_load_ps(reinterpret_cast<const float *>(srcp + x));
      __m256 src_1 = _mm256_load_ps(reinterpret_cast<const float *>(srcp + x + 8));
      if (chroma) {
#ifdef FLOAT_CHROMA_IS_HALF_CENTERED
        // shift 0.5 before, shift back half_int after. 0.5->exact half of 128/512/...
        src_0 = _mm256_sub_ps(src_0, half_ps);
        src_1 = _mm256_sub_ps(src_1, half_ps);
        //pixel = (srcp0[x] - 0.5f) * factor + half + 0.5f;
#else
        //pixel = (srcp0[x]       ) * factor + half + 0.5f;
#endif
        src_0 = _mm256_fmadd_ps(src_0, factor_ps, halfint_plus_rounder_ps);
        src_1 = _mm256_fmadd_ps(src_1, factor_ps, halfint_plus_rounder_ps);
      }
      else {
        if constexpr(!fulls) {
          src_0 = _mm256_sub_ps(src_0, limit_lo_s_ps);
          src_1 = _mm256_sub_ps(src_1, limit_lo_s_ps);
        }
        src_0 = _mm256_fmadd_ps(src_0, factor_ps, limit_lo_plus_rounder_ps);
        src_1 = _mm256_fmadd_ps(src_1, factor_ps, limit_lo_plus_rounder_ps);
        //pixel = (srcp0[x] - limit_lo_s_ps) * factor + half + limit_lo + 0.5f;
      }
      result_0 = _mm256_cvttps_epi32(src_0); // truncate
      result_1 = _mm256_cvttps_epi32(src_1);
      if constexpr(sizeof(pixel_t) == 2) {
        result = _mm256_packus_epi32(result_0, result_1);
        result = _mm256_permute4x64_epi64(result, (0 << 0) | (2 << 2) | (1 << 4) | (3 << 6));
        if (targetbits > 8 && targetbits < 16) {
          result = _mm256_min_epu16(result, max_pixel_value_256); // extra clamp for 10, 12, 14 bits
        }
          _mm256_store_si256(reinterpret_cast<__m256i *>(dstp + x), result);
      }
      else {
        result = _mm256_packus_epi32(result_0, result_1);
        result = _mm256_permute4x64_epi64(result, (0 << 0) | (2 << 2) | (1 << 4) | (3 << 6));
        __m128i result128_lo = _mm256_castsi256_si128(result);
        __m128i result128_hi = _mm256_extractf128_si256(result, 1);
        __m128i result128 = _mm_packus_epi16(result128_lo, result128_hi);
        _mm_store_si128(reinterpret_cast<__m128i *>(dstp + x), result128);
      }
    }
    dstp += dst_pitch;
    srcp += src_pitch;
  }
  _mm256_zeroupper();
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#define convert_32_to_uintN_avx2_functions_any(type, targetbits) \
template void convert_32_to_uintN_avx2<type, targetbits, false, true, true>(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch); \
template void convert_32_to_uintN_avx2<type, targetbits, true, true, true>(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch); \
template void convert_32_to_uintN_avx2<type, targetbits, false, true, false>(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch); \
template void convert_32_to_uintN_avx2<type, targetbits, true, true, false>(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch); \
template void convert_32_to_uintN_avx2<type, targetbits, false, false, true>(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch); \
template void convert_32_to_uintN_avx2<type, targetbits, true, false, true>(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch); \
template void convert_32_to_uintN_avx2<type, targetbits, false, false, false>(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch); \
template void convert_32_to_uintN_avx2<type, targetbits, true, false, false>(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch);

#define convert_32_to_uintN_avx2_functions_8(targetbits) convert_32_to_uintN_avx2_functions_any(uint8_t, targetbits)
#define convert_32_to_uintN_avx2_functions(targetbits) convert_32_to_uintN_avx2_functions_any(uint16_t, targetbits)

convert_32_to_uintN_avx2_functions_8(8)
convert_32_to_uintN_avx2_functions(10)
convert_32_to_uintN_avx2_functions(12)
convert_32_to_uintN_avx2_functions(14)
convert_32_to_uintN_avx2_functions(16)

// YUV: bit shift 10-12-14-16 <=> 10-12-14-16 bits
// shift right or left, depending on expandrange template param
template<bool expandrange, uint8_t shiftbits>
void convert_uint16_to_uint16_c_avx2(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
    const uint16_t *srcp0 = reinterpret_cast<const uint16_t *>(srcp);
    uint16_t *dstp0 = reinterpret_cast<uint16_t *>(dstp);

    src_pitch = src_pitch / sizeof(uint16_t);
    dst_pitch = dst_pitch / sizeof(uint16_t);

    const int src_width = src_rowsize / sizeof(uint16_t);

    for(int y=0; y<src_height; y++)
    {
        for (int x = 0; x < src_width; x++)
        {
            if(expandrange)
                dstp0[x] = srcp0[x] << shiftbits;  // expand range. No clamp before, source is assumed to have valid range
            else
                dstp0[x] = srcp0[x] >> shiftbits;  // reduce range
        }
        dstp0 += dst_pitch;
        srcp0 += src_pitch;
    }
    _mm256_zeroupper();
}

// instantiate them
template void convert_uint16_to_uint16_c_avx2<false, 2>(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch);
template void convert_uint16_to_uint16_c_avx2<false, 4>(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch);
template void convert_uint16_to_uint16_c_avx2<false, 6>(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch);
template void convert_uint16_to_uint16_c_avx2<true, 2>(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch);
template void convert_uint16_to_uint16_c_avx2<true, 4>(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch);
template void convert_uint16_to_uint16_c_avx2<true, 6>(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch);
