#define _CRT_SECURE_NO_WARNINGS
#include "avisynth.h"

#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <memory>

#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>

#include "CommonFunctions.h"
#include "MVKernel.h"
#include "CudaKernelBase.h"
#include "CudaDebug.h"

#include "ReduceKernel.cuh"
#include "VectorFunctions.cuh"
#include "GenericImageFunctions.cuh"

/////////////////////////////////////////////////////////////////////////////
// MEMCPY
/////////////////////////////////////////////////////////////////////////////

__global__ void memcpy_kernel(uint8_t* dst, const uint8_t* src, int nbytes) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  if (x < nbytes) {
    dst[x] = src[x];
  }
}

/////////////////////////////////////////////////////////////////////////////
// PadFrame
/////////////////////////////////////////////////////////////////////////////


template <typename pixel_t>
__global__ void kl_copy_pad(
    pixel_t* __restrict__ dst, int dst_pitch, const pixel_t* __restrict__ src, int src_pitch, int hPad, int vPad, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x - hPad;
    int y = threadIdx.y + blockIdx.y * blockDim.y - vPad;

    if (x < width + hPad && y < height + vPad) {
        int srcx = x;
        if (srcx < 0) {
            srcx = -srcx - 1;
        } else if (srcx >= width) {
            srcx = width - (srcx - width) - 1;
        }
        int srcy = y;
        if (srcy < 0) {
            srcy = -srcy - 1;
        } else if (srcy >= height) {
            srcy = height - (srcy - height) - 1;
        }
        pixel_t v = src[srcx + srcy * src_pitch];
        dst[x + y * dst_pitch] = v;
    }
}

// width �� Pad ���܂܂Ȃ�����
// block(2, -), threads(hPad, -)
template <typename pixel_t>
__global__ void kl_pad_frame_h(pixel_t* ptr, int pitch, int hPad, int width, int height)
{
  bool isLeft = (blockIdx.x == 0);
  int x = threadIdx.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (y < height) {
    if (isLeft) {
      ptr[x + y * pitch] = ptr[hPad + y * pitch];
    }
    else {
      ptr[(hPad + width + x) + y * pitch] = ptr[(hPad + width - 1) + y * pitch];
    }
  }
}

// height �� Pad ���܂܂Ȃ�����
// block(-, 2), threads(-, vPad)
template <typename pixel_t>
__global__ void kl_pad_frame_v(pixel_t* ptr, int pitch, int vPad, int width, int height)
{
  bool isTop = (blockIdx.y == 0);
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y;

  if (x < width) {
    if (isTop) {
      ptr[x + y * pitch] = ptr[x + vPad * pitch];
    }
    else {
      ptr[x + (vPad + height + y) * pitch] = ptr[x + (vPad + height - 1) * pitch];
    }
  }
}

/////////////////////////////////////////////////////////////////////////////
// Wiener
/////////////////////////////////////////////////////////////////////////////

// so called Wiener interpolation. (sharp, similar to Lanczos ?)
// invarint simplified, 6 taps. Weights: (1, -5, 20, 20, -5, 1)/32 - added by Fizick
template <typename pixel_t>
__global__ void kl_vertical_wiener(pixel_t *pDst, const pixel_t *pSrc, int nDstPitch,
  int nSrcPitch, int nWidth, int nHeight, int max_pixel_value)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < nWidth) {
    if (y < 2) {
      pDst[x + y * nDstPitch] = (pSrc[x + y * nSrcPitch] + pSrc[x + (y + 1) * nSrcPitch] + 1) >> 1;
    }
    else if (y < nHeight - 4) {
      pDst[x + y * nDstPitch] = min(max_pixel_value, max(0,
        (pSrc[x + (y - 2) * nSrcPitch]
          + (-(pSrc[x + (y - 1) * nSrcPitch]) + (pSrc[x + y * nSrcPitch] << 2) +
          (pSrc[x + (y + 1) * nSrcPitch] << 2) - (pSrc[x + (y + 2) * nSrcPitch])) * 5
          + (pSrc[x + (y + 3) * nSrcPitch]) + 16) >> 5));
    }
    else if (y < nHeight - 1) {
      pDst[x + y * nDstPitch] = (pSrc[x + y * nSrcPitch] + pSrc[x + (y + 1) * nSrcPitch] + 1) >> 1;
    }
    else if (y < nHeight) {
      // last row
      pDst[x + y * nDstPitch] = pSrc[x + y * nSrcPitch];
    }
  }
}

template <typename pixel_t>
__global__ void kl_horizontal_wiener(pixel_t *pDst, const pixel_t *pSrc, int nDstPitch,
  int nSrcPitch, int nWidth, int nHeight, int max_pixel_value)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (y < nHeight) {
    if (x < 2) {
      pDst[x + y * nDstPitch] = (pSrc[x + y * nSrcPitch] + pSrc[(x + 1) + y * nSrcPitch] + 1) >> 1;
    }
    else if (x < nWidth - 4) {
      pDst[x + y * nDstPitch] = min(max_pixel_value, max(0,
        (pSrc[(x - 2) + y * nSrcPitch]
          + (-(pSrc[(x - 1) + y * nSrcPitch]) + (pSrc[x + y * nSrcPitch] << 2) +
          (pSrc[(x + 1) + y * nSrcPitch] << 2) - (pSrc[(x + 2) + y * nSrcPitch])) * 5
          + (pSrc[(x + 3) + y * nSrcPitch]) + 16) >> 5));
    }
    else if (x < nWidth - 1) {
      pDst[x + y * nDstPitch] = (pSrc[x + y * nSrcPitch] + pSrc[(x + 1) + y * nSrcPitch] + 1) >> 1;
    }
    else if (x < nWidth) {
      // last column
      pDst[x + y * nDstPitch] = pSrc[x + y * nSrcPitch];
    }
  }
}

/////////////////////////////////////////////////////////////////////////////
// RB2BilinearFilter
/////////////////////////////////////////////////////////////////////////////

enum {
  RB2B_BILINEAR_W = 32,
  RB2B_BILINEAR_H = 16,
};

// BilinearFiltered with 1/8, 3/8, 3/8, 1/8 filter for smoothing and anti-aliasing - Fizick
// threads=(RB2B_BILINEAR_W,RB2B_BILINEAR_H)
// nblocks=(nblocks(nWidth*2, RB2B_BILINEAR_W - 2),nblocks(nHeight,RB2B_BILINEAR_H))
template <typename pixel_t>
__global__ void kl_RB2B_bilinear_filtered(
  pixel_t *pDst, const pixel_t *pSrc, int nDstPitch, int nSrcPitch, int nWidth, int nHeight)
{
  __shared__ pixel_t tmp[RB2B_BILINEAR_H][RB2B_BILINEAR_W];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Vertical�����s
  // Horizontal�ŎQ�Ƃ��邽�ߗ��[1�񂸂]���Ɏ��s
  int x = tx - 1 + blockIdx.x * (RB2B_BILINEAR_W - 2);
  int y = ty + blockIdx.y * RB2B_BILINEAR_H;
  int y2 = y * 2;

  if (x >= 0 && x < nWidth * 2) {
    if (y < 1) {
      tmp[ty][tx] = (pSrc[x + y2 * nSrcPitch] + pSrc[x + (y2 + 1) * nSrcPitch] + 1) >> 1;
    }
    else if (y < nHeight - 1) {
      tmp[ty][tx] = (pSrc[x + (y2 - 1) * nSrcPitch]
        + pSrc[x + y2 * nSrcPitch] * 3
        + pSrc[x + (y2 + 1) * nSrcPitch] * 3
        + pSrc[x + (y2 + 2) * nSrcPitch] + 4) / 8;
    }
    else if (y < nHeight) {
      tmp[ty][tx] = (pSrc[x + y2 * nSrcPitch] + pSrc[x + (y2 + 1) * nSrcPitch] + 1) >> 1;
    }
  }

  __syncthreads();

  // Horizontal�����s
  x = tx + blockIdx.x * ((RB2B_BILINEAR_W - 2) / 2);
  int tx2 = tx * 2;

  if (tx < ((RB2B_BILINEAR_W - 2) / 2) && y < nHeight) {
    // tmp��[0][1]�����_�ł��邱�Ƃɒ���
    if (x < 1) {
      pDst[x + y * nDstPitch] = (tmp[ty][tx2 + 1] + tmp[ty][tx2 + 2] + 1) >> 1;
    }
    else if (x < nWidth - 1) {
      pDst[x + y * nDstPitch] = (tmp[ty][tx2]
        + tmp[ty][tx2 + 1] * 3
        + tmp[ty][tx2 + 2] * 3
        + tmp[ty][tx2 + 3] + 4) / 8;
    }
    else if (x < nWidth) {
      pDst[x + y * nDstPitch] = (tmp[ty][tx2 + 1] + tmp[ty][tx2 + 2] + 1) >> 1;
    }
  }
}


template <typename pixel_t>
__global__ void kl_RB2B_bilinear_filtered_with_pad(
    pixel_t *pDst, const pixel_t *pSrc, int nDstPitch, int nSrcPitch, int nWidth, int nHeight,
    int hpad, int vpad)
{
    __shared__ pixel_t tmp[RB2B_BILINEAR_H][RB2B_BILINEAR_W];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Vertical�����s
    // Horizontal�ŎQ�Ƃ��邽�ߗ��[1�񂸂]���Ɏ��s
    int x = tx - 1 + blockIdx.x * (RB2B_BILINEAR_W - 2);
    const int y = ty + blockIdx.y * RB2B_BILINEAR_H;
    const int y2 = y * 2;

    if (x >= 0 && x < nWidth * 2) {
        pixel_t val;
        pixel_t s1 = pSrc[x + (y2 + 0) * nSrcPitch];
        pixel_t s2 = pSrc[x + (y2 + 1) * nSrcPitch];
        if (y < 1) {
            val = (s1 + s2 + 1) >> 1;
        } else if (y < nHeight - 1) {
            pixel_t s0 = pSrc[x + (y2 - 1) * nSrcPitch];
            pixel_t s3 = pSrc[x + (y2 + 2) * nSrcPitch];
            val = (s0 + s1 * 3 + s2 * 3 + s3 + 4) >> 3;
        } else if (y < nHeight) {
            val = (s1 + s2 + 1) >> 1;
        }
        tmp[ty][tx] = val;
    }

    __syncthreads();

    // Horizontal�����s
    x = tx + blockIdx.x * ((RB2B_BILINEAR_W - 2) / 2);
    const int tx2 = tx * 2;

    if (tx < ((RB2B_BILINEAR_W - 2) / 2) && y < nHeight) {
        pixel_t val;
        pixel_t s1 = tmp[ty][tx2 + 1];
        pixel_t s2 = tmp[ty][tx2 + 2];
        // tmp��[0][1]�����_�ł��邱�Ƃɒ���
        if (x < 1) {
            val = (s1 + s2 + 1) >> 1;
        } else if (x < nWidth - 1) {
            pixel_t s0 = tmp[ty][tx2];
            pixel_t s3 = tmp[ty][tx2 + 3];
            val = (s0 + s1 * 3 + s2 * 3 + s3 + 4) >> 3;
        } else if (x < nWidth) {
            val = (s1 + s2 + 1) >> 1;
        }
        pDst[x + y * nDstPitch] = val;

        // �����܂ł�pad�Ȃ��łƓ���
        // ��������padding�̏���
        int dstx = 0;
        if (x < hpad) {
            dstx = -x - 1;
        } else if (x >= nWidth - hpad) {
            dstx = nWidth + (nWidth - x) - 1;
        }
        if (dstx != 0) {
            pDst[dstx + y * nDstPitch] = val;
        }
        int dsty = 0;
        if (y < vpad) {
            dsty = -y - 1;
        } else if (y >= nHeight - vpad) {
            dsty = nHeight + (nHeight - y) - 1;
        }
        if (dsty != 0) {
            pDst[x + dsty * nDstPitch] = val;
        }
        if (dstx != 0 && dsty != 0) {
            pDst[dstx + dsty * nDstPitch] = val;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////
// SearchMV
/////////////////////////////////////////////////////////////////////////////


typedef int sad_t; // ���float�ɂ���

struct SearchBlock {
  // [0-3]: nDxMax, nDyMax, nDxMin, nDyMin �iMax��Max-1�ɂ��Ă����j
  // [4-9]: Left predictor, Up predictor, bottom-right predictor(from coarse level)
  // �����ȂƂ���͍��Ȃ��悤�ɂ���i�Œ�ł��ǂꂩ�P�͗L���Ȃ̂Ŗ����Ȃ�Ƃ���͂��̃C���f�b�N�X�Ŗ��߂�j
  // [10-11]: predictor �� x, y
  int data[12];
  // [0-3]: penaltyZero, penaltyGlobal, 0(penaltyPredictor), penaltyNew
  // [4]: lambda
  sad_t dataf[5];
};

#define CLIP_RECT data
#define REF_VECTOR_INDEX (&data[4])
#define PRED_X data[10]
#define PRED_Y data[11]
#define PENALTIES dataf
#define PENALTY_NEW dataf[3]
#define LAMBDA dataf[4]

#define LARGE_COST INT_MAX

struct CostResult {
  sad_t cost;
  short2 xy;
};

__device__ void dev_clip_mv(short2& v, const int* rect)
{
  v.x = (v.x > rect[0]) ? rect[0] : (v.x < rect[2]) ? rect[2] : v.x;
  v.y = (v.y > rect[1]) ? rect[1] : (v.y < rect[3]) ? rect[3] : v.y;
}

__device__ bool dev_check_mv(int x, int y, const int* rect)
{
  return (x <= rect[0]) & (y <= rect[1]) & (x >= rect[2]) & (y >= rect[3]);
}

__device__ int dev_sq_norm(int ax, int ay, int bx, int by) {
  return (ax - bx) * (ax - bx) + (ay - by) * (ay - by);
}

// pRef �� �u���b�N�I�t�Z�b�g����\�߈ړ������Ă������|�C���^
// vx,vy �� �T�u�s�N�Z�����܂߂��x�N�g��
template <typename pixel_t, int NPEL>
__device__ const pixel_t* dev_get_ref_block(const pixel_t* pRef, int nPitch, int nImgPitch, int vx, int vy)
{
  if (NPEL == 1) {
    return &pRef[vx + vy * nPitch];
  }
  else if (NPEL == 2) {
    int sx = vx & 1;
    int sy = vy & 1;
    int si = sx + sy * 2;
    int x = vx >> 1;
    int y = vy >> 1;
    return &pRef[x + y * nPitch + si * nImgPitch];
  }
  else { // NPEL == 4
    int sx = vx & 3;
    int sy = vy & 3;
    int si = sx + sy * 4;
    int x = vx >> 2;
    int y = vy >> 2;
    return &pRef[x + y * nPitch + si * nImgPitch];
  }
}

template <typename pixel_t, int BLK_SIZE, bool CHROMA>
__device__ sad_t dev_calc_sad(
  int wi,
  const pixel_t* pSrcY, const pixel_t* pSrcU, const pixel_t* pSrcV,
  const pixel_t* pRefY, const pixel_t* pRefU, const pixel_t* pRefV,
  int nPitchY, int nPitchU, int nPitchV)
{
  enum {
    BLK_SIZE_UV = BLK_SIZE / 2,
    HALF_UV = BLK_SIZE / 4,
  };
  int sad = 0;
  int yx = wi;
  for (int yy = 0; yy < BLK_SIZE; ++yy) { // 16�񃋁[�v
    sad = __sad(pSrcY[yx + yy * BLK_SIZE], pRefY[yx + yy * nPitchY], sad);
  }
  if (CHROMA) {
    // UV��8x8
    int uvx = wi % BLK_SIZE_UV;
    int uvy = wi / BLK_SIZE_UV;
    for (int t = 0; t < HALF_UV; ++t, uvy += 2) { // 4�񃋁[�v
      sad = __sad(pSrcU[uvx + uvy * BLK_SIZE_UV], pRefU[uvx + uvy * nPitchU], sad);
      sad = __sad(pSrcV[uvx + uvy * BLK_SIZE_UV], pRefV[uvx + uvy * nPitchV], sad);
    }
  }
  dev_reduce_warp<int, BLK_SIZE, AddReducer<int>>(wi, sad);
  return sad;
}

#if 0
template <typename pixel_t, int BLK_SIZE, bool CHROMA>
__device__ sad_t dev_calc_sad_debug(
  bool debug,
  int wi,
  const pixel_t* pSrcY, const pixel_t* pSrcU, const pixel_t* pSrcV,
  const pixel_t* pRefY, const pixel_t* pRefU, const pixel_t* pRefV,
  int nPitchY, int nPitchU, int nPitchV)
{
  enum {
    BLK_SIZE_UV = BLK_SIZE / 2,
    HALF_UV = BLK_SIZE / 4,
  };
  int sad = 0;
  int yx = wi;
  for (int yy = 0; yy < BLK_SIZE; ++yy) { // 16�񃋁[�v
    sad = __sad(pSrcY[yx + yy * BLK_SIZE], pRefY[yx + yy * nPitchY], sad);
    if (debug && wi == 0) {
      printf("i=%d,sum=%d\n", yy, sad);
    }
  }
  if (CHROMA) {
    // UV��8x8
    int uvx = wi % BLK_SIZE_UV;
    int uvy = wi / BLK_SIZE_UV;
    for (int t = 0; t < HALF_UV; ++t, uvy += 2) { // 4�񃋁[�v
      sad = __sad(pSrcU[uvx + uvy * BLK_SIZE_UV], pRefU[uvx + uvy * nPitchU], sad);
      sad = __sad(pSrcV[uvx + uvy * BLK_SIZE_UV], pRefV[uvx + uvy * nPitchV], sad);
      if (debug && wi == 0) {
        printf("i=%d,sum=%d\n", uvy, sad);
      }
    }
  }
  dev_reduce_warp<int, BLK_SIZE, AddReducer<int>>(wi, sad);
  return sad;
}
#endif

__device__ void MinCost(CostResult& a, CostResult& b) {
  if (*(volatile sad_t*)&a.cost > *(volatile sad_t*)&b.cost) {
    *(volatile sad_t*)&a.cost = *(volatile sad_t*)&b.cost;
    *(volatile short*)&a.xy.x = *(volatile short*)&b.xy.x;
    *(volatile short*)&a.xy.y = *(volatile short*)&b.xy.y;
  }
}

// MAX - (MAX/4) <= (���ʂ̌�) <= MAX �ł��邱��
// �X���b�h���� (���ʂ̌�) - MAX/2
template <int LEN, bool CPU_EMU>
__device__ void dev_reduce_result(CostResult* tmp, int tid)
{
  if (CPU_EMU) {
    // ���Ԃ�CPU�łɍ��킹��
    if (tid == 0) {
      for (int i = 1; i < LEN; ++i) {
        MinCost(tmp[0], tmp[i]);
      }
    }
  }
  else {
    if (LEN > 8) MinCost(tmp[tid], tmp[tid + 8]);
    MinCost(tmp[tid], tmp[tid + 4]);
    MinCost(tmp[tid], tmp[tid + 2]);
    MinCost(tmp[tid], tmp[tid + 1]);
  }
}

// ���Ԃ�CPU�łɍ��킹��
__constant__ int2 c_expanding_search_1_area[] = {
  { 0, -1 },
  { 0, 1 },
  { -1, 0 },
  { 1, 0 },

  { -1, -1 },
  { -1, 1 },
  { 1, -1 },
  { 1, 1 },
};

// __syncthreads()���Ăяo���Ă���̂őS���ŌĂ�
template <typename pixel_t, int BLK_SIZE, int NPEL, bool CHROMA, bool CPU_EMU>
__device__ void dev_expanding_search_1(
  int debug,
  int tx, int wi, int bx, int cx, int cy,
  const int* data, const sad_t* dataf,
  CostResult& bestResult,
  const pixel_t* pSrcY, const pixel_t* pSrcU, const pixel_t* pSrcV,
  const pixel_t* __restrict__ pRefBY, const pixel_t* __restrict__ pRefBU, const pixel_t* __restrict__ pRefBV,
  int nPitchY, int nPitchU, int nPitchV,
  int nImgPitchY, int nImgPitchU, int nImgPitchV)
{
  __shared__ bool isVectorOK[8];
  __shared__ CostResult result[8];
  __shared__ const pixel_t* pRefY[8];
  __shared__ const pixel_t* pRefU[8];
  __shared__ const pixel_t* pRefV[8];

  if (tx < 8) {
    int x = result[tx].xy.x = cx + c_expanding_search_1_area[tx].x;
    int y = result[tx].xy.y = cy + c_expanding_search_1_area[tx].y;
    bool ok = dev_check_mv(x, y, CLIP_RECT);
    int cost = (LAMBDA * dev_sq_norm(x, y, PRED_X, PRED_Y)) >> 8;

    // no additional SAD calculations if partial sum is already above minCost
    if (cost >= bestResult.cost) {
      ok = false;
    }
#if 0
    if (debug) {
      if (wi == 0) printf("expand1 cx=%d,cy=%d\n", cx, cy);
      printf("expand1 bx=%d,cost=%d,ok=%d\n", tx, cost, ok);
    }
#endif
    isVectorOK[tx] = ok;
    result[tx].cost = ok ? cost : LARGE_COST;

    pRefY[tx] = dev_get_ref_block<pixel_t, NPEL>(pRefBY, nPitchY, nImgPitchY, x, y);
    if (CHROMA) {
      pRefU[tx] = dev_get_ref_block<pixel_t, NPEL>(pRefBU, nPitchU, nImgPitchU, x >> 1, y >> 1);
      pRefV[tx] = dev_get_ref_block<pixel_t, NPEL>(pRefBV, nPitchV, nImgPitchV, x >> 1, y >> 1);
    }
  }

  __syncthreads();

  if (isVectorOK[bx]) {
    sad_t sad = dev_calc_sad<pixel_t, BLK_SIZE, CHROMA>(wi, pSrcY, pSrcU, pSrcV, pRefY[bx], pRefU[bx], pRefV[bx], nPitchY, nPitchU, nPitchV);
    if (wi == 0) {
      result[bx].cost += sad + ((sad * PENALTY_NEW) >> 8);
#if 0
      if (debug) {
        printf("expand1 bx=%d,sad=%d,cost=%d\n", bx, sad, result[bx].cost);
        if (bx == 3) {
          //if(false) {
          printf("srcY: %d,%d,...,%d,%d,... refY: %d,%d,...,%d,%d,... \n",
            pSrcY[0], pSrcY[1], pSrcY[0 + BLK_SIZE], pSrcY[1 + BLK_SIZE],
            pRefY[bx][0], pRefY[bx][1], pRefY[bx][0 + nPitchY], pRefY[bx][1 + nPitchY]
          );
          printf("srcU: %d,%d,...,%d,%d,... refU: %d,%d,...,%d,%d,... \n",
            pSrcU[0], pSrcU[1], pSrcU[0 + BLK_SIZE / 2], pSrcU[1 + BLK_SIZE / 2],
            pRefU[bx][0], pRefU[0][1], pRefU[bx][0 + nPitchU], pRefU[bx][1 + nPitchU]
          );
          printf("srcV: %d,%d,...,%d,%d,... refV: %d,%d,...,%d,%d,... \n",
            pSrcV[0], pSrcV[1], pSrcV[0 + BLK_SIZE / 2], pSrcV[1 + BLK_SIZE / 2],
            pRefV[bx][0], pRefV[bx][1], pRefV[bx][0 + nPitchV], pRefV[bx][1 + nPitchV]
          );
        }
      }
#endif
    }
  }

  __syncthreads();

  // ���ʏW��
  if (tx < 4) { // reduce��8-4=4�X���b�h�ŌĂ�
    dev_reduce_result<8, CPU_EMU>(result, tx);

    if (tx == 0) { // tx == 0�͍Ō�̃f�[�^����������ł���̂ŃA�N�Z�XOK
      if (result[0].cost < bestResult.cost) {
        bestResult = result[0];
      }
    }
  }
}

// ���Ԃ�CPU�łɍ��킹��
__constant__ int2 c_expanding_search_2_area[] = {

  { -1, -2 },
  { -1, 2 },
  { 0, -2 },
  { 0, 2 },
  { 1, -2 },
  { 1, 2 },

  { -2, -1 },
  { 2, -1 },
  { -2, 0 },
  { 2, 0 },
  { -2, 1 },
  { 2, 1 },

  { -2, -2 },
  { -2, 2 },
  { 2, -2 },
  { 2, 2 },
};

// __syncthreads()���Ăяo���Ă���̂őS���ŌĂ�
template <typename pixel_t, int BLK_SIZE, int NPEL, bool CHROMA, bool CPU_EMU>
__device__ void dev_expanding_search_2(
  int debug,
  int tx, int wi, int bx, int cx, int cy,
  const int* data, const sad_t* dataf,
  CostResult& bestResult,
  const pixel_t* pSrcY, const pixel_t* pSrcU, const pixel_t* pSrcV,
  const pixel_t* __restrict__ pRefBY, const pixel_t* __restrict__ pRefBU, const pixel_t* __restrict__ pRefBV,
  int nPitchY, int nPitchU, int nPitchV,
  int nImgPitchY, int nImgPitchU, int nImgPitchV)
{
  __shared__ bool isVectorOK[16];
  __shared__ CostResult result[16];
  __shared__ const pixel_t* pRefY[16];
  __shared__ const pixel_t* pRefU[16];
  __shared__ const pixel_t* pRefV[16];

  if (tx < 16) {
    int x = result[tx].xy.x = cx + c_expanding_search_2_area[tx].x;
    int y = result[tx].xy.y = cy + c_expanding_search_2_area[tx].y;
    bool ok = dev_check_mv(x, y, CLIP_RECT);
    int cost = (LAMBDA * dev_sq_norm(x, y, PRED_X, PRED_Y)) >> 8;

    // no additional SAD calculations if partial sum is already above minCost
    if (cost >= bestResult.cost) {
      ok = false;
    }

    isVectorOK[tx] = ok;
    result[tx].cost = ok ? cost : LARGE_COST;

    pRefY[tx] = dev_get_ref_block<pixel_t, NPEL>(pRefBY, nPitchY, nImgPitchY, x, y);
    if (CHROMA) {
      pRefU[tx] = dev_get_ref_block<pixel_t, NPEL>(pRefBU, nPitchU, nImgPitchU, x >> 1, y >> 1);
      pRefV[tx] = dev_get_ref_block<pixel_t, NPEL>(pRefBV, nPitchV, nImgPitchV, x >> 1, y >> 1);
    }
  }

  __syncthreads();

  if (isVectorOK[bx]) {
    sad_t sad = dev_calc_sad<pixel_t, BLK_SIZE, CHROMA>(wi, pSrcY, pSrcU, pSrcV, pRefY[bx], pRefU[bx], pRefV[bx], nPitchY, nPitchU, nPitchV);
    if (wi == 0) {
      result[bx].cost += sad + ((sad * PENALTY_NEW) >> 8);
    }
  }
  int bx2 = bx + 8;
  if (isVectorOK[bx2]) {
    sad_t sad = dev_calc_sad<pixel_t, BLK_SIZE, CHROMA>(wi, pSrcY, pSrcU, pSrcV, pRefY[bx2], pRefU[bx2], pRefV[bx2], nPitchY, nPitchU, nPitchV);
    if (wi == 0) {
      result[bx2].cost += sad + ((sad * PENALTY_NEW) >> 8);
    }
  }

  __syncthreads();

  // ���ʏW��
  if (tx < 8) { // reduce��16-8=8�X���b�h�ŌĂ�
    dev_reduce_result<16, CPU_EMU>(result, tx);

    if (tx == 0) { // tx == 0�͍Ō�̃f�[�^����������ł���̂ŃA�N�Z�XOK
      if (result[0].cost < bestResult.cost) {
        bestResult = result[0];
      }
    }
  }
}

__constant__ int2 c_hex2_search_1_area[] = {
  { -2, 0 }, { -1, 2 }, { 1, 2 }, { 2, 0 }, { 1, -2 }, { -1, -2 }
};

// __syncthreads()���Ăяo���Ă���̂őS���ŌĂ�
template <typename pixel_t, int BLK_SIZE, int NPEL, bool CHROMA, bool CPU_EMU>
__device__ void dev_hex2_search_1(
  int debug,
  int tx, int wi, int bx, int cx, int cy,
  const int* data, const sad_t* dataf,
  CostResult& bestResult,
  const pixel_t* pSrcY, const pixel_t* pSrcU, const pixel_t* pSrcV,
  const pixel_t* __restrict__ pRefBY, const pixel_t* __restrict__ pRefBU, const pixel_t* __restrict__ pRefBV,
  int nPitchY, int nPitchU, int nPitchV,
  int nImgPitchY, int nImgPitchU, int nImgPitchV)
{
  __shared__ bool isVectorOK[8];
  __shared__ CostResult result[8];
  __shared__ const pixel_t* pRefY[8];
  __shared__ const pixel_t* pRefU[8];
  __shared__ const pixel_t* pRefV[8];

  if (tx < 8) {
    isVectorOK[tx] = false;
  }

  if (tx < 6) {
    int x = result[tx].xy.x = cx + c_hex2_search_1_area[tx].x;
    int y = result[tx].xy.y = cy + c_hex2_search_1_area[tx].y;
    bool ok = dev_check_mv(x, y, CLIP_RECT);
    int cost = (LAMBDA * dev_sq_norm(x, y, PRED_X, PRED_Y)) >> 8;

    // no additional SAD calculations if partial sum is already above minCost
    if (cost >= bestResult.cost) {
      ok = false;
    }

    isVectorOK[tx] = ok;
    result[tx].cost = ok ? cost : LARGE_COST;

    pRefY[tx] = dev_get_ref_block<pixel_t, NPEL>(pRefBY, nPitchY, nImgPitchY, x, y);
    if (CHROMA) {
      pRefU[tx] = dev_get_ref_block<pixel_t, NPEL>(pRefBU, nPitchU, nImgPitchU, x >> 1, y >> 1);
      pRefV[tx] = dev_get_ref_block<pixel_t, NPEL>(pRefBV, nPitchV, nImgPitchV, x >> 1, y >> 1);
    }
  }

  __syncthreads();

  if (isVectorOK[bx]) {
    sad_t sad = dev_calc_sad<pixel_t, BLK_SIZE, CHROMA>(wi, pSrcY, pSrcU, pSrcV, pRefY[bx], pRefU[bx], pRefV[bx], nPitchY, nPitchU, nPitchV);
    if (wi == 0) {
      result[bx].cost += sad + ((sad * PENALTY_NEW) >> 8);
    }
  }

  __syncthreads();

  // ���ʏW��
  if (tx < 2) { // reduce��6-4=2�X���b�h�ŌĂ�
    dev_reduce_result<6, CPU_EMU>(result, tx);

    if (tx == 0) { // tx == 0�͍Ō�̃f�[�^����������ł���̂ŃA�N�Z�XOK
      if (result[0].cost < bestResult.cost) {
        bestResult = result[0];
      }
    }
  }
}

template <typename pixel_t, int BLK_SIZE, bool CHROMA>
__device__ void dev_read_pixels(int tx,
  const pixel_t* pSrcY, const pixel_t* pSrcU, const pixel_t* pSrcV,
  pixel_t *pDstY, pixel_t *pDstU, pixel_t *pDstV,
  int nPitchY, int nPitchU, int nPitchV,
  int offx, int offy)
{
  enum {
    BLK_SIZE_UV = BLK_SIZE / 2,
    HALF_UV = BLK_SIZE / 4,
  };
  int x = tx % BLK_SIZE;
  int y = tx / BLK_SIZE;
  // range(x,y)=(BLK_SIZE,8)
  for (int t = 0, yy = y; t < BLK_SIZE / 8; ++t, yy += 8) {
    pDstY[x + yy * BLK_SIZE] = pSrcY[(x + offx) + (yy + offy) * nPitchY];
  }
  if (CHROMA) {
    offx >>= 1;
    offy >>= 1;
    if (BLK_SIZE_UV == 4) {
      if (x < BLK_SIZE_UV) {
        if (y < BLK_SIZE_UV) { // 0-4
          pDstU[x + y * BLK_SIZE_UV] = pSrcU[(x + offx) + (y + offy) * nPitchU];
        }
        else { // 4-8
          y -= BLK_SIZE_UV;
          pDstV[x + y * BLK_SIZE_UV] = pSrcV[(x + offx) + (y + offy) * nPitchV];
        }
      }
    }
    else if (BLK_SIZE_UV == 8) {
      if (x < BLK_SIZE_UV) {
        pDstU[x + y * BLK_SIZE_UV] = pSrcU[(x + offx) + (y + offy) * nPitchU];
      }
      else {
        x -= BLK_SIZE_UV;
        pDstV[x + y * BLK_SIZE_UV] = pSrcV[(x + offx) + (y + offy) * nPitchV];
      }
    }
    else if (BLK_SIZE_UV == 16) {
      int uvx = tx % BLK_SIZE_UV;
      int uvy = tx / BLK_SIZE_UV;
      // range(uvx,uvy)=(BLK_SIZE_UV,16)
      pDstU[uvx + uvy * BLK_SIZE_UV] = pSrcU[(uvx + offx) + (uvy + offy) * nPitchU];
      pDstV[uvx + uvy * BLK_SIZE_UV] = pSrcV[(uvx + offx) + (uvy + offy) * nPitchV];
    }
  }
}

template <typename pixel_t>
struct SearchBatch {
  VECTOR *out;
  int* dst_sad;
  const SearchBlock* __restrict__ blocks;
  short2* vectors; // [x,y]
  volatile int* prog;
  int* next;
  const pixel_t* __restrict__ pSrcY;
  const pixel_t* __restrict__ pSrcU;
  const pixel_t* __restrict__ pSrcV;
  const pixel_t* __restrict__ pRefY;
  const pixel_t* __restrict__ pRefU;
  const pixel_t* __restrict__ pRefV;
};

template <typename pixel_t>
union SearchBatchData {
  enum {
    LEN = sizeof(SearchBatch<pixel_t>) / sizeof(int)
  };
  SearchBatch<pixel_t> d;
  int data[LEN];
};

// �������@ 0:�����Ȃ��i�f�o�b�O�p�j, 1:1�������i�����x�j, 2:2�������i�ᐸ�x�j
#define ANALYZE_SYNC 1

template <typename pixel_t, int BLK_SIZE, int SEARCH, int NPEL, bool CHROMA, bool CPU_EMU>
__global__ void kl_search(
  SearchBatchData<pixel_t> *pdata,
  int nBlkX, int nBlkY, int nPad,
  int nPitchY, int nPitchUV,
  int nImgPitchY, int nImgPitchUV
)
{
  // threads=BLK_SIZE*8

  enum {
    BLK_SIZE_UV = BLK_SIZE / 2,
    BLK_STEP = BLK_SIZE / 2,
  };

  const int tx = threadIdx.x;
  const int wi = tx % BLK_SIZE;
  const int bx = tx / BLK_SIZE;

  __shared__ SearchBatchData<pixel_t> d;

  if (tx < SearchBatchData<pixel_t>::LEN) {
    d.data[tx] = pdata[blockIdx.x].data[tx];
  }
  __syncthreads();

  __shared__ int blkx;

  //for (int blkx = blockIdx.x; blkx < nBlkX; blkx += gridDim.x) {
  while (true) {
    if (tx == 0) {
      blkx = atomicAdd(d.d.next, 1);
    }
    __syncthreads();

    if (blkx >= nBlkX) {
      break;
    }

    for (int blky = 0; blky < nBlkY; ++blky) {

      // src��shared memory�ɓ]��
      int offx = nPad + blkx * BLK_STEP;
      int offy = nPad + blky * BLK_STEP;

      __shared__ pixel_t srcY[BLK_SIZE * BLK_SIZE];
      __shared__ pixel_t srcU[BLK_SIZE_UV * BLK_SIZE_UV];
      __shared__ pixel_t srcV[BLK_SIZE_UV * BLK_SIZE_UV];

      dev_read_pixels<pixel_t, BLK_SIZE, CHROMA>(tx,
        d.d.pSrcY, d.d.pSrcU, d.d.pSrcV, srcY, srcU, srcV,
        nPitchY, nPitchUV, nPitchUV, offx, offy);

      __shared__ const pixel_t* pRefBY;
      __shared__ const pixel_t* pRefBU;
      __shared__ const pixel_t* pRefBV;

      if (tx == 0) {
        pRefBY = &d.d.pRefY[offx + offy * nPitchY];
        if (CHROMA) {
          pRefBU = &d.d.pRefU[(offx >> 1) + (offy >> 1) * nPitchUV];
          pRefBV = &d.d.pRefV[(offx >> 1) + (offy >> 1) * nPitchUV];
        }
      }

      // �p�����[�^�Ȃǂ̃f�[�^��shared memory�Ɋi�[
      __shared__ int data[12];
      __shared__ sad_t dataf[5];

      if (tx < 12) {
        int blkIdx = blky*nBlkX + blkx;
        data[tx] = d.d.blocks[blkIdx].data[tx];
        if (tx < 5) {
          dataf[tx] = d.d.blocks[blkIdx].dataf[tx];
        }
      }

      // !!!!! �ˑ��u���b�N�̌v�Z���I���̂�҂� !!!!!!
#if ANALYZE_SYNC == 1
      if (tx == 0 && blkx > 0)
      {
        while (d.d.prog[blkx - 1] < blky);
      }
#elif ANALYZE_SYNC == 2
      if (tx == 0 && blkx >= 2)
      {
        while (d.d.prog[blkx - (1 + (blkx & 1))] < blky);
      }
#endif

      __syncthreads();

      // FetchPredictors
      __shared__ CostResult result[8];
      __shared__ const pixel_t* pRefY[8];
      __shared__ const pixel_t* pRefU[8];
      __shared__ const pixel_t* pRefV[8];

      if (tx < 7) {
        __shared__ volatile short pred[7][2]; // x, y

        if (tx < 6) {
          // zero, global, predictor, predictors[1]�`[3]���擾
          short2 vec = d.d.vectors[REF_VECTOR_INDEX[tx]];
          dev_clip_mv(vec, CLIP_RECT);

          if (CPU_EMU) {
            // 3��median�Ȃ̂ŋ󂯂�iCPU�łƍ��킹��j
            int dx = (tx < 3) ? tx : (tx + 1);
            pred[dx][0] = vec.x;
            pred[dx][1] = vec.y;
            // memfence
            if (tx < 2) {
              // Median predictor
              // �v�Z�����������̂ŏ��������E�E�E
              int a = pred[4][tx];
              int b = pred[5][tx];
              int c = pred[6][tx];
              pred[3][tx] = min(max(min(a, b), c), max(a, b));
            }
          }
          else {
            pred[tx][0] = vec.x;
            pred[tx][1] = vec.y;
            // memfence
            if (tx < 2) {
              // Median predictor
              // �v�Z�����������̂ŏ��������E�E�E
              int a = pred[3][tx];
              int b = pred[4][tx];
              int c = pred[5][tx];
              pred[6][tx] = min(max(min(a, b), c), max(a, b));
            }
          }
        }
        // memfence
        int x = result[tx].xy.x = pred[tx][0];
        int y = result[tx].xy.y = pred[tx][1];
        result[tx].cost = (LAMBDA * dev_sq_norm(x, y, PRED_X, PRED_Y)) >> 8;

        pRefY[tx] = dev_get_ref_block<pixel_t, NPEL>(pRefBY, nPitchY, nImgPitchY, x, y);
        if (CHROMA) {
          pRefU[tx] = dev_get_ref_block<pixel_t, NPEL>(pRefBU, nPitchUV, nImgPitchUV, x >> 1, y >> 1);
          pRefV[tx] = dev_get_ref_block<pixel_t, NPEL>(pRefBV, nPitchUV, nImgPitchUV, x >> 1, y >> 1);
        }
      }

      __syncthreads();

      bool debug = (nBlkY == 10 && blkx == 1 && blky == 0);
      //bool debug = false;

      // �܂���7�ӏ����v�Z
      if (bx < 7) {
#if 0
        if (wi == 0 && nBlkY == 10 && blkx == 1 && blky == 0) {
          printf("1:[%d]: x=%d,y=%d,cost=%d\n", bx, result[bx].xy.x, result[bx].xy.y, result[bx].cost);
        }
#endif
        sad_t sad = dev_calc_sad<pixel_t, BLK_SIZE, CHROMA>(wi, srcY, srcU, srcV, pRefY[bx], pRefU[bx], pRefV[bx], nPitchY, nPitchUV, nPitchUV);
        //sad_t sad = dev_calc_sad_debug<pixel_t, BLK_SIZE, CHROMA>(debug && bx == 3, wi, srcY, srcU, srcV, pRefY[bx], pRefU[bx], pRefV[bx], nPitchY, nPitchUV, nPitchUV);

#if 0
        if (blkx == 0 && blky == 0) {
          if (bx == 0) {
            printf("[SAD] wi=%d, sad=%d\n", wi, sad);
          }

          dev_reduce_warp<int, 16, AddReducer<int>>(wi, sad);

          if (bx == 0 && wi == 0) {
            printf("[SAD] reduced sad=%d\n", sad);
          }
        }
#endif

        if (wi == 0) {
          if (bx < 3) {
            // pzero, pglobal, 0
            result[bx].cost = sad + ((sad * PENALTIES[bx]) >> 8);
          }
          else {
            result[bx].cost += sad;
          }
#if 0
          if (blockIdx.y == 1 && nBlkY == 32 && blkx == 0 && blky == 0) {
            if (bx == 0) {
              printf("src: %d,%d,...,%d,%d,... ref: %d,%d,...,%d,%d,... \n",
                srcY[0], srcY[1], srcY[0 + BLK_SIZE], srcY[1 + BLK_SIZE],
                pRefY[0][0], pRefY[0][1], pRefY[0][0 + nPitchY], pRefY[0][1 + nPitchY]
              );
              printf("src: %d,%d,...,%d,%d,... ref: %d,%d,...,%d,%d,... \n",
                srcU[0], srcU[1], srcU[0 + BLK_SIZE / 2], srcU[1 + BLK_SIZE / 2],
                pRefU[0][0], pRefU[0][1], pRefU[0][0 + nPitchUV], pRefU[0][1 + nPitchUV]
              );
              printf("src: %d,%d,...,%d,%d,... ref: %d,%d,...,%d,%d,... \n",
                srcV[0], srcV[1], srcV[0 + BLK_SIZE / 2], srcV[1 + BLK_SIZE / 2],
                pRefV[0][0], pRefV[0][1], pRefV[0][0 + nPitchUV], pRefV[0][1 + nPitchUV]
              );
            }
            printf("bx=%d,sad=%d,cost=%d\n", bx, sad, result[bx].cost);
          }
#endif
        }
        // �Ƃ肠������r��cost�����ł��̂�SAD�͗v��Ȃ�
        // SAD�͒T�����I�������Čv�Z����
      }

      __syncthreads();

      // ���ʏW��
      if (tx < 3) { // 7-4=3�X���b�h�ŌĂ�
        dev_reduce_result<7, CPU_EMU>(result, tx);
      }
#if 0
      if (tx == 0 && nBlkY == 10 && blkx == 1 && blky == 0) {
        printf("1best=(%d,%d,%d)\n", result[0].xy.x, result[0].xy.y, result[0].cost);
      }
#endif

      __syncthreads();

      // Refine
      if (SEARCH == 1) {
        // EXHAUSTIVE
        int bmx = result[0].xy.x;
        int bmy = result[0].xy.y;
        dev_expanding_search_1<pixel_t, BLK_SIZE, NPEL, CHROMA, CPU_EMU>(debug,
          tx, wi, bx, bmx, bmy, data, dataf, result[0],
          srcY, srcU, srcV, pRefBY, pRefBU, pRefBV,
          nPitchY, nPitchUV, nPitchUV, nImgPitchY, nImgPitchUV, nImgPitchUV);
        dev_expanding_search_2<pixel_t, BLK_SIZE, NPEL, CHROMA, CPU_EMU>(debug,
          tx, wi, bx, bmx, bmy, data, dataf, result[0],
          srcY, srcU, srcV, pRefBY, pRefBU, pRefBV,
          nPitchY, nPitchUV, nPitchUV, nImgPitchY, nImgPitchUV, nImgPitchUV);
      }
      else if (SEARCH == 2) {
        // HEX2SEARCH
        dev_hex2_search_1<pixel_t, BLK_SIZE, NPEL, CHROMA, CPU_EMU>(debug,
          tx, wi, bx, result[0].xy.x, result[0].xy.y, data, dataf, result[0],
          srcY, srcU, srcV, pRefBY, pRefBU, pRefBV,
          nPitchY, nPitchUV, nPitchUV, nImgPitchY, nImgPitchUV, nImgPitchUV);
        dev_expanding_search_1<pixel_t, BLK_SIZE, NPEL, CHROMA, CPU_EMU>(debug,
          tx, wi, bx, result[0].xy.x, result[0].xy.y, data, dataf, result[0],
          srcY, srcU, srcV, pRefBY, pRefBU, pRefBV,
          nPitchY, nPitchUV, nPitchUV, nImgPitchY, nImgPitchUV, nImgPitchUV);
      }


      if (tx == 0) {
        // ���ʏ�������
        d.d.vectors[blky*nBlkX + blkx] = result[0].xy;

        // ���ʂ̏������݂��I���̂�҂�
        __threadfence();

        // ��������������
        d.d.prog[blkx] = blky;
      }

      // ���L�������ی�
      __syncthreads();
    }
  }
}

// threads=BLK_SIZE*8,
template <typename pixel_t, int BLK_SIZE, int NPEL, bool CHROMA>
__global__ void kl_calc_all_sad(
  SearchBatchData<pixel_t> *pdata,
  int nBlkX, int nBlkY,
  int nPad,
  int nPitchY, int nPitchUV,
  int nImgPitchY, int nImgPitchUV)
{
  enum {
    BLK_SIZE_UV = BLK_SIZE / 2,
    BLK_STEP = BLK_SIZE / 2,
  };

  int tid = threadIdx.x;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int offx = nPad + bx * BLK_STEP;
  int offy = nPad + by * BLK_STEP;

  __shared__ SearchBatchData<pixel_t> d;

  if (tid < SearchBatchData<pixel_t>::LEN) {
      d.data[tid] = pdata[blockIdx.z].data[tid];
  }
  __syncthreads();

  __shared__ const pixel_t* pRefBY;
  __shared__ const pixel_t* pRefBU;
  __shared__ const pixel_t* pRefBV;
  __shared__ const pixel_t* pSrcBY;
  __shared__ const pixel_t* pSrcBU;
  __shared__ const pixel_t* pSrcBV;

  short2 xy;

  if (tid == 0) {
    pRefBY = &d.d.pRefY[offx + offy * nPitchY];
    if (CHROMA) {
      pRefBU = &d.d.pRefU[(offx >> 1) + (offy >> 1) * nPitchUV];
      pRefBV = &d.d.pRefV[(offx >> 1) + (offy >> 1) * nPitchUV];
    }
    pSrcBY = &d.d.pSrcY[offx + offy * nPitchY];
    if (CHROMA) {
      pSrcBU = &d.d.pSrcU[(offx >> 1) + (offy >> 1) * nPitchUV];
      pSrcBV = &d.d.pSrcV[(offx >> 1) + (offy >> 1) * nPitchUV];
    }

    xy = d.d.vectors[bx + by * nBlkX];

    pRefBY = dev_get_ref_block<pixel_t, NPEL>(pRefBY, nPitchY, nImgPitchY, xy.x, xy.y);
    if (CHROMA) {
      pRefBU = dev_get_ref_block<pixel_t, NPEL>(pRefBU, nPitchUV, nImgPitchUV, xy.x >> 1, xy.y >> 1);
      pRefBV = dev_get_ref_block<pixel_t, NPEL>(pRefBV, nPitchUV, nImgPitchUV, xy.x >> 1, xy.y >> 1);
    }

#if 0
    if (bx == 2 && by == 0) {
      printf("srcY: %d,%d,...,%d,%d,... refY: %d,%d,...,%d,%d,... \n",
        pSrcBY[0], pSrcBY[1], pSrcBY[0 + nPitchY], pSrcBY[1 + nPitchY],
        pRefBY[0], pRefBY[1], pRefBY[0 + nPitchY], pRefBY[1 + nPitchY]
      );
      printf("srcU: %d,%d,...,%d,%d,... refU: %d,%d,...,%d,%d,... \n",
        pSrcBU[0], pSrcBU[1], pSrcBU[0 + nPitchUV], pSrcBU[1 + nPitchUV],
        pRefBU[0], pRefBU[1], pRefBU[0 + nPitchUV], pRefBU[1 + nPitchUV]
      );
      printf("srcV: %d,%d,...,%d,%d,... refV: %d,%d,...,%d,%d,... \n",
        pSrcBV[0], pSrcBV[1], pSrcBV[0 + nPitchUV], pSrcBV[1 + nPitchUV],
        pRefBV[0], pRefBV[1], pRefBV[0 + nPitchUV], pRefBV[1 + nPitchUV]
      );
    }
#endif
  }

  __syncthreads();

  int sad = 0;

  int x = tid % BLK_SIZE;
  int y = tid / BLK_SIZE;
  // range(x,y)=(BLK_SIZE,8)
  for (int t = 0, yy = y; t < BLK_SIZE / 8; ++t, yy += 8) {
    sad = __sad(pSrcBY[x + yy * nPitchY], pRefBY[x + yy * nPitchY], sad);
#if 0
    if (bx == 2 && by == 0) {
      printf("Y,%d,%d,%d\n", yx, yy, __sad(pSrcBY[yx + yy * nPitchY], pRefBY[yx + yy * nPitchY], 0));
    }
#endif
  }
  if (CHROMA) {
    if (BLK_SIZE_UV == 4) {
      if (x < BLK_SIZE_UV) {
        if (y < BLK_SIZE_UV) { // 0-4
          sad = __sad(pSrcBU[x + y * nPitchUV], pRefBU[x + y * nPitchUV], sad);
        }
        else { // 4-8
          y -= BLK_SIZE_UV;
          sad = __sad(pSrcBV[x + y * nPitchUV], pRefBV[x + y * nPitchUV], sad);
        }
      }
    }
    else if (BLK_SIZE_UV == 8) {
      if (x < BLK_SIZE_UV) {
        sad = __sad(pSrcBU[x + y * nPitchUV], pRefBU[x + y * nPitchUV], sad);
      }
      else {
        x -= BLK_SIZE_UV;
        sad = __sad(pSrcBV[x + y * nPitchUV], pRefBV[x + y * nPitchUV], sad);
      }
    }
    else if (BLK_SIZE_UV == 16) {
      int uvx = tid % BLK_SIZE_UV;
      int uvy = tid / BLK_SIZE_UV;
      // range(uvx,uvy)=(BLK_SIZE_UV,16)
      sad = __sad(pSrcBU[uvx + uvy * nPitchUV], pRefBU[uvx + uvy * nPitchUV], sad);
      sad = __sad(pSrcBV[uvx + uvy * nPitchUV], pRefBV[uvx + uvy * nPitchUV], sad);
    }
  }
#if 0
  if (bx == 2 && by == 0) {
    printf("tid=%d,sad=%d\n", tid, sad);
  }
#endif
  __shared__ int buf[BLK_SIZE * 8];
  dev_reduce<int, BLK_SIZE * 8, AddReducer<int>>(tid, sad, buf);

  if (tid == 0) {
      d.d.dst_sad[bx + by * nBlkX] = sad;
      VECTOR vout = { xy.x, xy.y, sad };
      d.d.out[bx + by * nBlkX] = vout;
  }
}

__global__ void kl_prepare_search(
  int nBlkX, int nBlkY, int nBlkSize, int nLogScale,
  sad_t nLambdaLevel, sad_t lsad,
  sad_t penaltyZero, sad_t penaltyGlobal, sad_t penaltyNew,
  int nPel, int nPad, int nBlkSizeOvr, int nExtendedWidth, int nExptendedHeight,
  const short2* vectors, int vectorsPitch, const int* sads, int sadPitch, short2* vectors_copy, SearchBlock* dst_blocks, int* prog, int* next)
{
  int bx = threadIdx.x + blockIdx.x * blockDim.x;
  int by = threadIdx.y + blockIdx.y * blockDim.y;
  int batchid = blockIdx.z;

  vectors += batchid * vectorsPitch;
  sads += batchid * sadPitch;
  vectors_copy += batchid * vectorsPitch;
  dst_blocks += batchid * nBlkX * nBlkY;
  prog += batchid * nBlkX;
  next += batchid;

  if (bx < nBlkX && by < nBlkY) {
    //
    int blkIdx = bx + by*nBlkX;
    int sad = sads[blkIdx];
    SearchBlock *data = &dst_blocks[blkIdx];

    // �i����-1�ɏ��������Ă���
    if (by == 0) {
      prog[bx] = -1;

      // �J�E���^��0�ɂ��Ă���
      if (bx == 0) {
        *next = 0;
      }
    }

    int x = nPad + nBlkSizeOvr * bx;
    int y = nPad + nBlkSizeOvr * by;
    //
    int nPaddingScaled = nPad >> nLogScale;

    int nDxMax = nPel * (nExtendedWidth - x - nBlkSize - nPad + nPaddingScaled) - 1;
    int nDyMax = nPel * (nExptendedHeight - y - nBlkSize - nPad + nPaddingScaled) - 1;
    int nDxMin = -nPel * (x - nPad + nPaddingScaled);
    int nDyMin = -nPel * (y - nPad + nPaddingScaled);

    data->data[0] = nDxMax;
    data->data[1] = nDyMax;
    data->data[2] = nDxMin;
    data->data[3] = nDyMin;

    int p1 = -2; // -2��zero�x�N�^
                 // Left (or right) predictor
#if ANALYZE_SYNC == 0
    if (bx > 0)
    {
      p1 = blkIdx - 1 + nBlkX * nBlkY;
    }
#elif ANALYZE_SYNC == 1
    if (bx > 0)
    {
      p1 = blkIdx - 1;
    }
#else // ANALYZE_SYNC == 2
    if (bx >= 2)
    {
      p1 = blkIdx - (1 + (bx & 1));
    }
#endif

    int p2 = -2;
    // Up predictor
    if (by > 0)
    {
      p2 = blkIdx - nBlkX;
    }
    else {
      // median��left��I�΂��邽��
      p2 = p1;
    }

    int p3 = -2;
    // bottom-right pridictor (from coarse level)
    if ((by < nBlkY - 1) && (bx < nBlkX - 1))
    {
      // ���łɏ���������Ă���\�������肻��ł��v�Z�͉\�����A
      // �f�o�b�O�̂��ߔ񌈒蓮��͔��������̂�
      // �R�s�[���Ă�����̃f�[�^���g��
      p3 = blkIdx + nBlkX + 1 + nBlkX * nBlkY;
    }

    data->data[4] = -2;    // zero
    data->data[5] = -1;    // global
    data->data[6] = blkIdx;// predictor
    data->data[7] = p1;    //  predictors[1]
    data->data[8] = p2;    //  predictors[2]
    data->data[9] = p3;    //  predictors[3]

    short2 pred = vectors[blkIdx];

    // �v�Z���ɑO�̃��x�����狁�߂��x�N�^��ێ��������̂ŃR�s�[���Ă���
    vectors_copy[blkIdx] = pred;

    data->data[10] = pred.x;
    data->data[11] = pred.y;

    data->dataf[0] = penaltyZero;
    data->dataf[1] = penaltyGlobal;
    data->dataf[2] = 0;
    data->dataf[3] = penaltyNew;

    sad_t lambda = nLambdaLevel * lsad / (lsad + (sad >> 1)) * lsad / (lsad + (sad >> 1));
    if (by == 0) lambda = 0;
    data->dataf[4] = lambda;
  }
}

// threads=(1024), blocks=(2,batch)
__global__ void kl_most_freq_mv(const short2* vectors, int vectorsPitch, int nVec, short2* globalMVec)
{
  enum {
    DIMX = 1024,
    // level==1���ő�Ȃ̂ŁA���̃T�C�Y��8K���炢�܂őΉ�
    FREQ_SIZE = DIMX * 8,
    HALF_SIZE = FREQ_SIZE / 2
  };

  int tid = threadIdx.x;

  union SharedBuffer {
    int freq_arr[FREQ_SIZE]; //32KB
    struct {
      int red_cnt[DIMX];
      int red_idx[DIMX];
    };
  };
  __shared__ SharedBuffer b;

  for (int i = 0; i < FREQ_SIZE / DIMX; ++i) {
    b.freq_arr[tid + i * DIMX] = 0;
  }
  __syncthreads();

  vectors += blockIdx.y * vectorsPitch;
  if (blockIdx.x == 0) {
    // x
    for (int i = tid; i < nVec; i += DIMX) {
      atomicAdd(&b.freq_arr[vectors[i].x + HALF_SIZE], 1);
    }
  }
  else {
    // y
    for (int i = tid; i < nVec; i += DIMX) {
      atomicAdd(&b.freq_arr[vectors[i].y + HALF_SIZE], 1);
    }
  }
  __syncthreads();

  int maxcnt = 0;
  int index = 0;
  for (int i = 0; i < FREQ_SIZE / DIMX; ++i) {
    if (b.freq_arr[tid + i * DIMX] > maxcnt) {
      maxcnt = b.freq_arr[tid + i * DIMX];
      index = tid + i * DIMX;
    }
  }
  __syncthreads();

  dev_reduce2<int, int, DIMX, MaxIndexReducer<int>>(tid, maxcnt, index, b.red_cnt, b.red_idx);

  if (tid == 0) {
    if (blockIdx.x == 0) {
      // x
      globalMVec[blockIdx.y].x = index - HALF_SIZE;
    }
    else {
      // y
      globalMVec[blockIdx.y].y = index - HALF_SIZE;
    }
  }
}

__global__ void kl_mean_global_mv(const short2* vectors, int vectorsPitch, int nVec, short2* globalMVec)
{
  enum {
    DIMX = 1024,
  };

  int tid = threadIdx.x;
  int medianx = globalMVec[blockIdx.y].x;
  int mediany = globalMVec[blockIdx.y].y;

  int meanvx = 0;
  int meanvy = 0;
  int num = 0;

  vectors += blockIdx.y * vectorsPitch;
  for (int i = tid; i < nVec; i += DIMX) {
    if (__sad(vectors[i].x, medianx, 0) < 6
      && __sad(vectors[i].y, mediany, 0) < 6)
    {
      meanvx += vectors[i].x;
      meanvy += vectors[i].y;
      num += 1;
    }
  }

  __shared__ int red_vx[DIMX];
  __shared__ int red_vy[DIMX];
  __shared__ int red_num[DIMX];

  red_vx[tid] = meanvx;
  red_vy[tid] = meanvy;
  red_num[tid] = num;

  __syncthreads();
  if (tid < 512) {
    red_vx[tid] += red_vx[tid + 512];
    red_vy[tid] += red_vy[tid + 512];
    red_num[tid] += red_num[tid + 512];
  }
  __syncthreads();
  if (tid < 256) {
    red_vx[tid] += red_vx[tid + 256];
    red_vy[tid] += red_vy[tid + 256];
    red_num[tid] += red_num[tid + 256];
  }
  __syncthreads();
  if (tid < 128) {
    red_vx[tid] += red_vx[tid + 128];
    red_vy[tid] += red_vy[tid + 128];
    red_num[tid] += red_num[tid + 128];
  }
  __syncthreads();
  if (tid < 64) {
    red_vx[tid] += red_vx[tid + 64];
    red_vy[tid] += red_vy[tid + 64];
    red_num[tid] += red_num[tid + 64];
  }
  __syncthreads();
  if (tid < 32) {
    red_vx[tid] += red_vx[tid + 32];
    red_vy[tid] += red_vy[tid + 32];
    red_num[tid] += red_num[tid + 32];
  }
  __syncthreads();
  meanvx = red_vx[tid];
  meanvy = red_vy[tid];
  num = red_num[tid];
  if (tid < 32) {
#if CUDART_VERSION >= 9000
    meanvx += __shfl_down_sync(0xffffffff, meanvx, 16);
    meanvy += __shfl_down_sync(0xffffffff, meanvy, 16);
    num += __shfl_down_sync(0xffffffff, num, 16);
    meanvx += __shfl_down_sync(0xffffffff, meanvx, 8);
    meanvy += __shfl_down_sync(0xffffffff, meanvy, 8);
    num += __shfl_down_sync(0xffffffff, num, 8);
    meanvx += __shfl_down_sync(0xffffffff, meanvx, 4);
    meanvy += __shfl_down_sync(0xffffffff, meanvy, 4);
    num += __shfl_down_sync(0xffffffff, num, 4);
    meanvx += __shfl_down_sync(0xffffffff, meanvx, 2);
    meanvy += __shfl_down_sync(0xffffffff, meanvy, 2);
    num += __shfl_down_sync(0xffffffff, num, 2);
    meanvx += __shfl_down_sync(0xffffffff, meanvx, 1);
    meanvy += __shfl_down_sync(0xffffffff, meanvy, 1);
    num += __shfl_down_sync(0xffffffff, num, 1);
#else
    meanvx += __shfl_down(meanvx, 16);
    meanvy += __shfl_down(meanvy, 16);
    num += __shfl_down(num, 16);
    meanvx += __shfl_down(meanvx, 8);
    meanvy += __shfl_down(meanvy, 8);
    num += __shfl_down(num, 8);
    meanvx += __shfl_down(meanvx, 4);
    meanvy += __shfl_down(meanvy, 4);
    num += __shfl_down(num, 4);
    meanvx += __shfl_down(meanvx, 2);
    meanvy += __shfl_down(meanvy, 2);
    num += __shfl_down(num, 2);
    meanvx += __shfl_down(meanvx, 1);
    meanvy += __shfl_down(meanvy, 1);
    num += __shfl_down(num, 1);
#endif
    if (tid == 0) {
      globalMVec[blockIdx.y].x = 2 * meanvx / num;
      globalMVec[blockIdx.y].y = 2 * meanvy / num;
    }
  }
}

// normFactor = 3 - nLogPel + pob.nLogPel
// normov = (nBlkSizeX - nOverlapX)*(nBlkSizeY - nOverlapY)
// aoddx = (nBlkSizeX * 3 - nOverlapX * 2)
// aevenx = (nBlkSizeX * 3 - nOverlapX * 4);
// aoddy = (nBlkSizeY * 3 - nOverlapY * 2);
// aeveny = (nBlkSizeY * 3 - nOverlapY * 4);
// atotalx = (nBlkSizeX - nOverlapX) * 4
// atotaly = (nBlkSizeY - nOverlapY) * 4
__global__ void kl_interpolate_prediction(
  const short2* src_vector, int srcVectorPitcch, const int* src_sad, int srcSadPitch,
  short2* dst_vector, int dstVectorPitch, int* dst_sad, int dstSadPitch,
  int nSrcBlkX, int nSrcBlkY, int nDstBlkX, int nDstBlkY,
  int normFactor, int normov, int atotal, int aodd, int aeven)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int batchid = blockIdx.z;

  // �o�b�`���i�߂�
  src_vector += batchid * srcVectorPitcch;
  dst_vector += batchid * dstVectorPitch;
  src_sad += batchid * srcSadPitch;
  dst_sad += batchid * dstSadPitch;

  if (x < nDstBlkX && y < nDstBlkY) {
    short2 v1, v2, v3, v4;
    int sad1, sad2, sad3, sad4;
    int i = x;
    int j = y;
    if (i >= 2 * nSrcBlkX)
    {
      i = 2 * nSrcBlkX - 1;
    }
    if (j >= 2 * nSrcBlkY)
    {
      j = 2 * nSrcBlkY - 1;
    }
    int offy = -1 + 2 * (j % 2);
    int offx = -1 + 2 * (i % 2);
    int iper2 = i >> 1;
    int jper2 = j >> 1;

    if ((i == 0) || (i >= 2 * nSrcBlkX - 1))
    {
      if ((j == 0) || (j >= 2 * nSrcBlkY - 1))
      {
        v1 = v2 = v3 = v4 = src_vector[iper2 + (jper2)* nSrcBlkX];
        sad1 = sad2 = sad3 = sad4 = src_sad[iper2 + (jper2)* nSrcBlkX];
      }
      else
      {
        v1 = v2 = src_vector[iper2 + (jper2)* nSrcBlkX];
        sad1 = sad2 = src_sad[iper2 + (jper2)* nSrcBlkX];
        v3 = v4 = src_vector[iper2 + (jper2 + offy) * nSrcBlkX];
        sad3 = sad4 = src_sad[iper2 + (jper2 + offy) * nSrcBlkX];
      }
    }
    else if ((j == 0) || (j >= 2 * nSrcBlkY - 1))
    {
      v1 = v2 = src_vector[iper2 + (jper2)* nSrcBlkX];
      sad1 = sad2 = src_sad[iper2 + (jper2)* nSrcBlkX];
      v3 = v4 = src_vector[iper2 + offx + (jper2)* nSrcBlkX];
      sad3 = sad4 = src_sad[iper2 + offx + (jper2)* nSrcBlkX];
    }
    else
    {
      v1 = src_vector[iper2 + (jper2)* nSrcBlkX];
      sad1 = src_sad[iper2 + (jper2)* nSrcBlkX];
      v2 = src_vector[iper2 + offx + (jper2)* nSrcBlkX];
      sad2 = src_sad[iper2 + offx + (jper2)* nSrcBlkX];
      v3 = src_vector[iper2 + (jper2 + offy) * nSrcBlkX];
      sad3 = src_sad[iper2 + (jper2 + offy) * nSrcBlkX];
      v4 = src_vector[iper2 + offx + (jper2 + offy) * nSrcBlkX];
      sad4 = src_sad[iper2 + offx + (jper2 + offy) * nSrcBlkX];
    }

    int	ax1 = (offx > 0) ? aodd : aeven;
    int ax2 = atotal - ax1;
    int ay1 = (offy > 0) ? aodd : aeven;
    int ay2 = atotal - ay1;
    int a11 = ax1*ay1, a12 = ax1*ay2, a21 = ax2*ay1, a22 = ax2*ay2;
    int vx = (a11*v1.x + a21*v2.x + a12*v3.x + a22*v4.x) / normov;
    int vy = (a11*v1.y + a21*v2.y + a12*v3.y + a22*v4.y) / normov;

    sad_t tmp_sad = ((sad_t)a11*sad1 + (sad_t)a21*sad2 + (sad_t)a12*sad3 + (sad_t)a22*sad4) / normov;

    if (normFactor > 0) {
      vx >>= normFactor;
      vy >>= normFactor;
    }
    else {
      vx <<= -normFactor;
      vy <<= -normFactor;
    }

    int index = x + y * nDstBlkX;
    short2 v = { (short)vx, (short)vy };
    dst_vector[index] = v;
    dst_sad[index] = (int)(tmp_sad >> 4);
  }
}

__global__ void kl_load_mv(
  const VECTOR* in,
  short2* vectors, // [x,y]
  int* sads,
  int nBlk)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if (x < nBlk) {
    VECTOR vin = in[x];
    short2 v = { (short)vin.x, (short)vin.y };
    vectors[x] = v;
    sads[x] = vin.sad;
  }
}

__global__ void kl_store_mv(
  VECTOR* dst,
  const short2* vectors, // [x,y]
  const int* sads,
  int nBlk)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if (x < nBlk) {
    short2 v = vectors[x];
    VECTOR vout = { v.x, v.y, sads[x] };
    dst[x] = vout;
  }
}

__global__ void kl_write_default_mv(VECTOR* dst, int nBlkCount, int verybigSAD)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if (x < nBlkCount) {
    dst[x].x = 0;
    dst[x].y = 0;
    dst[x].x = verybigSAD;
  }
}

__global__ void kl_init_const_vec(short2* vectors, int vectorsPitch, const short2* globalMV, short nPel)
{
  vectors += blockIdx.y * vectorsPitch;
  if (blockIdx.x == 0) {
    vectors[-2] = short2();
  }
  else {
    short2 g = *globalMV;
    short2 c = { short(g.x * nPel), short(g.y * nPel) };
    vectors[-1] = c;
  }
}

/////////////////////////////////////////////////////////////////////////////
// DEGRAIN
/////////////////////////////////////////////////////////////////////////////

// �X���b�h��: sceneChange�̐�
__global__ void kl_init_scene_change(int* sceneChange)
{
  sceneChange[threadIdx.x] = 0;
}

// �X���b�h��: 256�i���v: nBlks�j
__global__ void kl_scene_change(const VECTOR* mv, int nBlks, int nTh1, int* sceneChange)
{
  enum {
    DIMX = 256,
  };

  int tid = threadIdx.x;
  int x = tid + blockIdx.x * blockDim.x;

  int s = 0;
  if (x < nBlks) {
    s = (mv[x].sad > nTh1) ? 1 : 0;
  }

  __shared__ int sbuf[DIMX];
  dev_reduce<int, DIMX, AddReducer<int>>(tid, s, sbuf);

  if (tid == 0) {
    atomicAdd(sceneChange, s);
  }
}

template <typename pixel_t, int N>
struct DegrainBlockData {
  const short *winOver;
  const pixel_t *pSrc;
  const pixel_t *pB[N], *pF[N];
  int WSrc, WRefB[N], WRefF[N];
};

template <typename pixel_t, int N>
union DegrainBlock {
  enum { LEN = sizeof(DegrainBlockData<pixel_t, N>) / 4 };
  DegrainBlockData<pixel_t, N> d;
  uint32_t m[LEN];
};

template <typename pixel_t, int N>
struct DegrainArgData {
  const VECTOR *mvB[N], *mvF[N];
  const pixel_t *pSrc, *pRefB[N], *pRefF[N];
  bool isUsableB[N], isUsableF[N];
};

template <typename pixel_t, int N>
union DegrainArg {
  enum { LEN = sizeof(DegrainArgData<pixel_t, N>) / 4 };
  DegrainArgData<pixel_t, N> d;
  uint32_t m[LEN];
};

static __device__ int dev_degrain_weight(int thSAD, int blockSAD)
{
  // Returning directly prevents a divide by 0 if thSAD == blockSAD == 0.
  if (thSAD <= blockSAD)
  {
    return 0;
  }
  const float sq_thSAD = float(thSAD) * float(thSAD);
  const float sq_blockSAD = float(blockSAD) * float(blockSAD);
  return (int)(256.0f*(sq_thSAD - sq_blockSAD) / (sq_thSAD + sq_blockSAD));
}

// binomial�Ή���4�܂Łi����ȍ~�͊O���̏d�݂�0�ɂȂ�̂ňӖ����Ȃ��j
template<int delta, bool binomial>
static __device__ void dev_norm_weights(int &WSrc, int *WRefB, int *WRefF)
{
  WSrc = 256;
  if (binomial) {
    if (delta == 1) {
      WSrc *= 2;
    }
    else if (delta == 2) {
      WSrc *= 6;
      WRefB[0] *= 4; WRefF[0] *= 4;
    }
    else if (delta == 3) {
      WSrc *= 20;
      WRefB[0] *= 15; WRefF[0] *= 15;
      WRefB[1] *= 6; WRefF[1] *= 6;
    }
    else if (delta == 4) {
      WSrc *= 70;
      WRefB[0] *= 56; WRefF[0] *= 56;
      WRefB[1] *= 28; WRefF[1] *= 28;
      WRefB[2] *= 8; WRefF[2] *= 8;
    }
  }
  int WSum;
  if (delta == 6)
    WSum = WRefB[0] + WRefF[0] + WSrc + WRefB[1] + WRefF[1] + WRefB[2] + WRefF[2] + WRefB[3] + WRefF[3] + WRefB[4] + WRefF[4] + WRefB[5] + WRefF[5] + 1;
  else if (delta == 5)
    WSum = WRefB[0] + WRefF[0] + WSrc + WRefB[1] + WRefF[1] + WRefB[2] + WRefF[2] + WRefB[3] + WRefF[3] + WRefB[4] + WRefF[4] + 1;
  else if (delta == 4)
    WSum = WRefB[0] + WRefF[0] + WSrc + WRefB[1] + WRefF[1] + WRefB[2] + WRefF[2] + WRefB[3] + WRefF[3] + 1;
  else if (delta == 3)
    WSum = WRefB[0] + WRefF[0] + WSrc + WRefB[1] + WRefF[1] + WRefB[2] + WRefF[2] + 1;
  else if (delta == 2)
    WSum = WRefB[0] + WRefF[0] + WSrc + WRefB[1] + WRefF[1] + 1;
  else if (delta == 1)
    WSum = WRefB[0] + WRefF[0] + WSrc + 1;
  WRefB[0] = WRefB[0] * 256 / WSum; // normalize weights to 256
  WRefF[0] = WRefF[0] * 256 / WSum;
  if (delta >= 2) {
    WRefB[1] = WRefB[1] * 256 / WSum; // normalize weights to 256
    WRefF[1] = WRefF[1] * 256 / WSum;
  }
  if (delta >= 3) {
    WRefB[2] = WRefB[2] * 256 / WSum; // normalize weights to 256
    WRefF[2] = WRefF[2] * 256 / WSum;
  }
  if (delta >= 4) {
    WRefB[3] = WRefB[3] * 256 / WSum; // normalize weights to 256
    WRefF[3] = WRefF[3] * 256 / WSum;
  }
  if (delta >= 5) {
    WRefB[4] = WRefB[4] * 256 / WSum; // normalize weights to 256
    WRefF[4] = WRefF[4] * 256 / WSum;
  }
  if (delta >= 6) {
    WRefB[5] = WRefB[5] * 256 / WSum; // normalize weights to 256
    WRefF[5] = WRefF[5] * 256 / WSum;
  }
  if (delta == 6)
    WSrc = 256 - WRefB[0] - WRefF[0] - WRefB[1] - WRefF[1] - WRefB[2] - WRefF[2] - WRefB[3] - WRefF[3] - WRefB[4] - WRefF[4] - WRefB[5] - WRefF[5];
  else if (delta == 5)
    WSrc = 256 - WRefB[0] - WRefF[0] - WRefB[1] - WRefF[1] - WRefB[2] - WRefF[2] - WRefB[3] - WRefF[3] - WRefB[4] - WRefF[4];
  else if (delta == 4)
    WSrc = 256 - WRefB[0] - WRefF[0] - WRefB[1] - WRefF[1] - WRefB[2] - WRefF[2] - WRefB[3] - WRefF[3];
  else if (delta == 3)
    WSrc = 256 - WRefB[0] - WRefF[0] - WRefB[1] - WRefF[1] - WRefB[2] - WRefF[2];
  else if (delta == 2)
    WSrc = 256 - WRefB[0] - WRefF[0] - WRefB[1] - WRefF[1];
  else if (delta == 1)
    WSrc = 256 - WRefB[0] - WRefF[0];
}

template <typename pixel_t, typename vpixel_t, int N, int NPEL, int SHIFT, bool BINOMIAL>
static __global__ void kl_prepare_degrain(
  int nBlkX, int nBlkY, int nPad, int nBlkSize, int nTh2, int thSAD,
  const short* ovrwins,
  const int * __restrict__ sceneChangeB,
  const int * __restrict__ sceneChangeF,
  const DegrainArg<pixel_t, N>* parg,
  DegrainBlock<pixel_t, N>* blocks,
  int nPitch, int nPitchSuper, int nImgPitch
)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int blkx = tx + blockIdx.x * blockDim.x;
  int blky = ty + blockIdx.y * blockDim.y;
  int idx = blkx + blky * nBlkX;

  __shared__ DegrainArg<pixel_t, N> s_arg_buf;

  if (tx < DegrainArg<pixel_t, N>::LEN) {
    s_arg_buf.m[tx] = parg->m[tx];
  }
  __syncthreads();

  if (blkx < nBlkX && blky < nBlkY) {
    DegrainArgData<pixel_t, N>& arg = s_arg_buf.d;
    DegrainBlockData<pixel_t, N>& b = blocks[idx].d;

    // winOver
    int wby = ((blky + nBlkY - 3) / (nBlkY - 2)) * 3;
    int wbx = (blkx + nBlkX - 3) / (nBlkX - 2);
    b.winOver = ovrwins + (wby + wbx) * nBlkSize * nBlkSize;

    int blkStep = nBlkSize / 2;
    int offx = blkx * blkStep;
    int offy = blky * blkStep;
    int offxS = nPad + offx;
    int offyS = nPad + offy;
    int offset = offx + offy * nPitch;
    int offsetS = offxS + offyS * nPitchSuper;

    int WSrc, WRefB[N], WRefF[N];

    // pSrc,pDst,pB,pF
    b.pSrc = arg.pSrc + offset;

    for (int i = 0; i < N; ++i) {
      bool isUsableB = arg.isUsableB[i] && !(sceneChangeB[i] > nTh2);
      if (isUsableB) {
        b.pB[i] = dev_get_ref_block<pixel_t, NPEL>(arg.pRefB[i] + offsetS, nPitchSuper, nImgPitch,
          arg.mvB[i][idx].x >> SHIFT, arg.mvB[i][idx].y >> SHIFT);
        WRefB[i] = dev_degrain_weight(thSAD, arg.mvB[i][idx].sad);
#if 0
        if (blkx == 13 && blky == 0) {
          int off = b.pB[i] - (vpixel_t*)arg.pRefB[i];
          printf("(%d,%d) => off=%d, pitch=%d,imgpitch=%d,v=%d\n",
            arg.mvB[i][idx].x, arg.mvB[i][idx].y, off, nPitchSuper, nImgPitch, b.pB[i][1].y);
        }
#endif
      }
      else {
        // ��������ǂݎ����l�͎g���Ȃ��̂œǂݎ���K���ȃA�h���X������
        //�i�s�b�`��Super�ƈႤ�̂ł����Ƃ���offset�����Ă������Ƃ����摜�͓ǂݎ��Ȃ����Ƃɒ��Ӂj
        b.pB[i] = arg.pSrc;
        WRefB[i] = 0;
      }
      bool isUsableF = arg.isUsableF[i] && !(sceneChangeF[i] > nTh2);
      if (isUsableF) {
        b.pF[i] = dev_get_ref_block<pixel_t, NPEL>(arg.pRefF[i] + offsetS, nPitchSuper, nImgPitch,
          arg.mvF[i][idx].x >> SHIFT, arg.mvF[i][idx].y >> SHIFT);
        WRefF[i] = dev_degrain_weight(thSAD, arg.mvF[i][idx].sad);
      }
      else {
        // ��������ǂݎ����l�͎g���Ȃ��̂œǂݎ���K���ȃA�h���X������
        b.pF[i] = arg.pSrc;
        WRefF[i] = 0;
      }
    }

    // weight
    dev_norm_weights<N, BINOMIAL>(WSrc, WRefB, WRefF);

    b.WSrc = WSrc;
    for (int i = 0; i < N; ++i) {
      b.WRefB[i] = WRefB[i];
      b.WRefF[i] = WRefF[i];
    }
  }
}

// �u���b�N��2x3�O��
template <typename pixel_t, typename vpixel_t, typename vtmp_t, typename vint_t, typename vshort_t, int N, int BLK_SIZE, int M>
static __global__ void kl_degrain_2x3(
  int nPatternX, int nPatternY,
  int nBlkX, int nBlkY, DegrainBlock<pixel_t, N>* data, vtmp_t* pDst, int pitch4, int pitchsuper4
)
{
  enum {
    VLEN = VHelper<vtmp_t>::VLEN,
    SPAN_X = 3,
    SPAN_Y = 2,
    BLK_STEP = BLK_SIZE / 2,
    BLK_SIZE4 = BLK_SIZE / VLEN,
    BLK_STEP4 = BLK_STEP / VLEN,
    THREADS = BLK_SIZE*BLK_SIZE4,
    TMP_W = BLK_SIZE + BLK_STEP * 2, // == BLK_SIZE * 2
    TMP_H = BLK_SIZE + BLK_STEP * 1,
    TMP_W4 = TMP_W / VLEN,
    N_READ_LOOP = DegrainBlock<pixel_t, N>::LEN / THREADS
  };

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = (M == 1) ? 0 : threadIdx.z;

  int tid = tx + ty * BLK_SIZE4;
  int offset = (tx + ty * pitch4) * VLEN;
  int offsetSuper = (tx + ty * pitchsuper4) * VLEN;

  __shared__ vtmp_t tmp[M][TMP_H][TMP_W4];
  __shared__ DegrainBlock<pixel_t, N> info_[M];
  DegrainBlock<pixel_t, N>& info = info_[tz];

  // tmp������
  tmp[tz][ty][tx] = VHelper<vtmp_t>::make(0);
  tmp[tz][ty][tx + BLK_SIZE4] = VHelper<vtmp_t>::make(0);
  if (ty < BLK_STEP) {
    tmp[tz][ty + BLK_SIZE][tx] = VHelper<vtmp_t>::make(0);
    tmp[tz][ty + BLK_SIZE][tx + BLK_SIZE4] = VHelper<vtmp_t>::make(0);
  }

  __syncthreads();

  const bool no_need_round = (sizeof(pixel_t) > 1);

  int largex = blockIdx.x * M + tz;
  int largey = blockIdx.y;
  int basex = (largex * 2 + nPatternX) * SPAN_X;
  int basey = (largey * 2 + nPatternY) * SPAN_Y;

  for (int bby = 0; bby < SPAN_Y; ++bby) {
    int by = basey + bby;
    // by��CUDA�u���b�N���̑S�X���b�h�������l�Ȃ̂ł����OK
    if (by >= nBlkY) break;

    for (int bbx = 0; bbx < SPAN_X; ++bbx) {
      int bx = basex + bbx;

      if (M == 1) {
        // M==1�̏ꍇ��bx��CUDA�u���b�N���̑S�X���b�h�������l�Ȃ̂ł����OK
        if (bx >= nBlkX) break;
      }

      int blkidx = bx + by * nBlkX;

      // M == 1�̂Ƃ��͏���������ȗ�
      if (M == 1 || bx < nBlkX) {
        // �u���b�N����ǂݍ���
        for (int i = 0; i < N_READ_LOOP; ++i) {
          info.m[i * THREADS + tid] = data[blkidx].m[i * THREADS + tid];
        }
        if (THREADS * N_READ_LOOP + tid < DegrainBlock<pixel_t, N>::LEN) {
          info.m[THREADS * N_READ_LOOP + tid] = data[blkidx].m[THREADS * N_READ_LOOP + tid];
        }
      }
      __syncthreads();

      if (M == 1 || bx < nBlkX) {
        vint_t val = { 0 };
        if (N == 1)
          val = to_int(__ldg((const vpixel_t*)&info.d.pSrc[offset])) * info.d.WSrc +
          VLoad<VLEN>::to_int(&info.d.pF[0][offsetSuper]) * info.d.WRefF[0] + VLoad<VLEN>::to_int(&info.d.pB[0][offsetSuper]) * info.d.WRefB[0];
        else if (N == 2)
          val = to_int(__ldg((const vpixel_t*)&info.d.pSrc[offset])) * info.d.WSrc +
          VLoad<VLEN>::to_int(&info.d.pF[0][offsetSuper]) * info.d.WRefF[0] + VLoad<VLEN>::to_int(&info.d.pB[0][offsetSuper]) * info.d.WRefB[0] +
          VLoad<VLEN>::to_int(&info.d.pF[1][offsetSuper]) * info.d.WRefF[1] + VLoad<VLEN>::to_int(&info.d.pB[1][offsetSuper]) * info.d.WRefB[1];

        int dstx = bbx * BLK_STEP4 + tx;
        int dsty = bby * BLK_STEP + ty;

        val = (val + (no_need_round ? 0 : 128)) >> 8;

        if (sizeof(pixel_t) == 1)
          tmp[tz][dsty][dstx] += (val * __ldg(&((const vshort_t*)info.d.winOver)[tid]) + 256) >> 6; // shift 5 in Short2Bytes<uint8_t> in overlap.cpp
        else
          tmp[tz][dsty][dstx] += val * __ldg(&((const vshort_t*)info.d.winOver)[tid]); // shift (5+6); in Short2Bytes16
      }
      __syncthreads();
#if 0
      if (basex == 12 && basey == 0 && dsty == 0 && dstx == 2) {
        auto t = (val * ((const vshort_t*)info.d.winOver)[tid] + 256) >> 6;
        printf("%d,%d*%d+%d*%d+%d*%d=%d*%d=>%d\n",
          tmp[0][0][3].y, info.d.pSrc[offset].y,
          info.d.WSrc, info.d.pF[0][offsetSuper].y,
          info.d.WRefF[0], info.d.pB[0][offsetSuper].y,
          info.d.WRefB[0], val.y, ((const vshort_t*)info.d.winOver)[tid].y,
          t.y);
    }
#endif
  }
}

  // dst�ɏ�������
  vtmp_t *p = &pDst[basex * BLK_STEP4 + tx + (basey * BLK_STEP + ty) * pitch4];
#if 0
  if (basex == 0 && basey == 0 && tx == 0 && ty == 0) {
    printf("%d\n", tmp[0][0][0].x);
  }
#endif
  int bsx = tx / BLK_STEP4 - 1;
  int bsy = ty / BLK_STEP - 1;
  if (basex + bsx < nBlkX && basey + bsy < nBlkY) {
    p[0] += tmp[tz][ty][tx];
    if (basex + bsx + 2 < nBlkX) {
      p[BLK_SIZE4] += tmp[tz][ty][tx + BLK_SIZE4];
    }
    if (ty < BLK_STEP) {
      if (basey + bsy + 2 < nBlkY) {
        p[BLK_SIZE * pitch4] += tmp[tz][ty + BLK_SIZE][tx];
        if (basex + bsx + 2 < nBlkX) {
          p[BLK_SIZE4 + BLK_SIZE * pitch4] += tmp[tz][ty + BLK_SIZE][tx + BLK_SIZE4];
        }
      }
    }
  }
}

template <typename vpixel_t, typename vtmp_t>
__global__ void kl_short_to_byte(
  vpixel_t* dst, const vtmp_t* src, int width4, int height, int pitch4, int max_pixel_value)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width4 && y < height) {
    const int shift = sizeof(dst[0].x) == 1 ? 5 : (5 + 6);
    int4 v = min(to_int(src[x + y * pitch4]) >> shift, max_pixel_value);
    dst[x + y * pitch4] = VHelper<vpixel_t>::cast_to(v);
  }
}

/////////////////////////////////////////////////////////////////////////////
// COMPENSATE
/////////////////////////////////////////////////////////////////////////////

template <typename pixel_t>
struct CompensateBlockData {
  const short *winOver;
  const pixel_t *pRef;
};

template <typename pixel_t>
union CompensateBlock {
  enum { LEN = sizeof(CompensateBlockData<pixel_t>) / 4 };
  CompensateBlockData<pixel_t> d;
  uint32_t m[LEN];
};

template <typename pixel_t, int NPEL, int SHIFT>
static __global__ void kl_prepare_compensate(
  int nBlkX, int nBlkY, int nPad, int nBlkSize, int nTh2, int time256, int thSAD,
  const short* ovrwins,
  const int * __restrict__ sceneChange,
  const VECTOR *mv,
  const pixel_t *pRef0,
  const pixel_t *pRef,
  CompensateBlock<pixel_t>* blocks,
  int nPitchSuper, int nImgPitch
)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int blkx = tx + blockIdx.x * blockDim.x;
  int blky = ty + blockIdx.y * blockDim.y;
  int idx = blkx + blky * nBlkX;

  if (blkx < nBlkX && blky < nBlkY) {
    CompensateBlockData<pixel_t>& b = blocks[idx].d;

    // winOver
    int wby = ((blky + nBlkY - 3) / (nBlkY - 2)) * 3;
    int wbx = (blkx + nBlkX - 3) / (nBlkX - 2);
    b.winOver = ovrwins + (wby + wbx) * nBlkSize * nBlkSize;

    int blkStep = nBlkSize / 2;
    int offxS = nPad + blkx * blkStep;
    int offyS = nPad + blky * blkStep;
    int offsetS = offxS + offyS * nPitchSuper;

    if (!(*sceneChange > nTh2)) {
      VECTOR vec = mv[idx];
      if (vec.sad < thSAD) {
        int mx = (vec.x * time256 / 256) >> SHIFT;
        int my = (vec.y * time256 / 256) >> SHIFT;
        b.pRef = dev_get_ref_block<pixel_t, NPEL>(pRef + offsetS, nPitchSuper, nImgPitch, mx, my);
      }
      else {
        b.pRef = dev_get_ref_block<pixel_t, NPEL>(pRef0 + offsetS, nPitchSuper, nImgPitch, 0, 0);
      }
    }
    else {
      // �V�[���`�F���W
      b.winOver = nullptr;
      b.pRef = nullptr;
    }
  }
}

// �u���b�N��2x3�O��
template <typename pixel_t, typename vtmp_t, typename vint_t, typename vshort_t, int BLK_SIZE, int M>
static __global__ void kl_compensate_2x3(
  int nPatternX, int nPatternY, int nBlkX, int nBlkY,
  CompensateBlock<pixel_t>* __restrict__ data,
  vtmp_t* pDst, int pitch4, int pitchsuper4
)
{
  enum {
    VLEN = VHelper<vtmp_t>::VLEN,
    SPAN_X = 3,
    SPAN_Y = 2,
    BLK_STEP = BLK_SIZE / 2,
    BLK_SIZE4 = BLK_SIZE / VLEN,
    BLK_STEP4 = BLK_STEP / VLEN,
    THREADS = BLK_SIZE*BLK_SIZE4,
    TMP_W = BLK_SIZE + BLK_STEP * 2, // == BLK_SIZE * 2
    TMP_H = BLK_SIZE + BLK_STEP * 1,
    TMP_W4 = TMP_W / VLEN,
    N_READ_LOOP = CompensateBlock<pixel_t>::LEN / THREADS
  };

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = (M == 1) ? 0 : threadIdx.z;

  int tid = tx + ty * BLK_SIZE4;
  int offsetSuper = (tx + ty * pitchsuper4) * VLEN;

  __shared__ vtmp_t tmp[M][TMP_H][TMP_W4];
  __shared__ CompensateBlock<pixel_t> info_[M];
  __shared__ bool isCopySrc;
  CompensateBlock<pixel_t>& info = info_[tz];

  // tmp������
  tmp[tz][ty][tx] = VHelper<vtmp_t>::make(0);
  tmp[tz][ty][tx + BLK_SIZE4] = VHelper<vtmp_t>::make(0);
  if (ty < BLK_STEP) {
    tmp[tz][ty + BLK_SIZE][tx] = VHelper<vtmp_t>::make(0);
    tmp[tz][ty + BLK_SIZE][tx + BLK_SIZE4] = VHelper<vtmp_t>::make(0);
  }

  if (tid == 0 && tz == 0) {
    isCopySrc = (data[0].d.winOver == nullptr);
  }

  __syncthreads();

  if (isCopySrc) {
    // �V�[���`�F���W���͂����ł͏������Ȃ�
    return;
  }

  int largex = blockIdx.x * M + tz;
  int largey = blockIdx.y;

  int basex = (largex * 2 + nPatternX) * SPAN_X;
  int basey = (largey * 2 + nPatternY) * SPAN_Y;

  for (int bby = 0; bby < SPAN_Y; ++bby) {
    int by = basey + bby;
    // by��CUDA�u���b�N���̑S�X���b�h�������l�Ȃ̂ł����OK
    if (by >= nBlkY) break;

    for (int bbx = 0; bbx < SPAN_X; ++bbx) {
      int bx = basex + bbx;

      if (M == 1) {
        // M==1�̏ꍇ��bx��CUDA�u���b�N���̑S�X���b�h�������l�Ȃ̂ł����OK
        if (bx >= nBlkX) break;
      }

      int blkidx = bx + by * nBlkX;

      // M == 1�̂Ƃ��͏���������ȗ�
      if (M == 1 || bx < nBlkX) {
        // �u���b�N����ǂݍ���
        for (int i = 0; i < N_READ_LOOP; ++i) {
          info.m[i * THREADS + tid] = data[blkidx].m[i * THREADS + tid];
        }
        if (THREADS * N_READ_LOOP + tid < CompensateBlock<pixel_t>::LEN) {
          info.m[THREADS * N_READ_LOOP + tid] = data[blkidx].m[THREADS * N_READ_LOOP + tid];
        }
      }
      __syncthreads();

      if (M == 1 || bx < nBlkX) {
        vint_t val = VLoad<VLEN>::to_int(&info.d.pRef[offsetSuper]);

        int dstx = bbx * BLK_STEP4 + tx;
        int dsty = bby * BLK_STEP + ty;

        if (sizeof(pixel_t) == 1)
          tmp[tz][dsty][dstx] += (val * __ldg(&((const vshort_t*)info.d.winOver)[tid]) + 256) >> 6; // shift 5 in Short2Bytes<uint8_t> in overlap.cpp
        else
          tmp[tz][dsty][dstx] += val * __ldg(&((const vshort_t*)info.d.winOver)[tid]); // shift (5+6); in Short2Bytes16
      }
      __syncthreads();
#if 0
      if (basex == 0 && basey == 0 && dsty == 1 && dstx == 0) {
        auto t = (val * ((const vshort_t*)info.d.winOver)[tid] + 256) >> 6;
        printf("%d,%d*%d=>%d\n",
          tmp[0][1][0].x, val.x, ((const vshort_t*)info.d.winOver)[tid].x, t.x);
    }
#endif
  }
}

  // dst�ɏ�������
  vtmp_t *p = &pDst[basex * BLK_STEP4 + tx + (basey * BLK_STEP + ty) * pitch4];
#if 0
  if (basex == 0 && basey == 0 && tx == 0 && ty == 0) {
    printf("%d\n", tmp[0][0][0].x);
  }
#endif
  int bsx = tx / BLK_STEP4 - 1;
  int bsy = ty / BLK_STEP - 1;
  if (basex + bsx < nBlkX && basey + bsy < nBlkY) {
    p[0] += tmp[tz][ty][tx];
    if (basex + bsx + 2 < nBlkX) {
      p[BLK_SIZE4] += tmp[tz][ty][tx + BLK_SIZE4];
    }
    if (ty < BLK_STEP) {
      if (basey + bsy + 2 < nBlkY) {
        p[BLK_SIZE * pitch4] += tmp[tz][ty + BLK_SIZE][tx];
        if (basex + bsx + 2 < nBlkX) {
          p[BLK_SIZE4 + BLK_SIZE * pitch4] += tmp[tz][ty + BLK_SIZE][tx + BLK_SIZE4];
        }
      }
    }
  }
}

template <typename vpixel_t, typename vtmp_t>
__global__ void kl_short_to_byte_or_copy_src(
  const void** __restrict__ pflag,
  vpixel_t* dst, const vpixel_t* src, const vtmp_t* tmp, int width4, int height, int pitch4, int max_pixel_value)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width4 && y < height) {
    if (*pflag) {
      const int shift = sizeof(dst[0].x) == 1 ? 5 : (5 + 6);
      int4 v = min(to_int(tmp[x + y * pitch4]) >> shift, max_pixel_value);
      dst[x + y * pitch4] = VHelper<vpixel_t>::cast_to(v);
    }
    else {
      // �V�[���`�F���W�Ȃ�\�[�X���R�s�[
      dst[x + y * pitch4] = src[x + y * pitch4];
    }
  }
}

/////////////////////////////////////////////////////////////////////////////
// DegrainTypes
/////////////////////////////////////////////////////////////////////////////

template <typename pixel_t> struct DegrainTypes { };

template <> struct DegrainTypes<uint8_t> {
  typedef uint16_t tmp_t;
  typedef uchar4 vpixel_t;
  typedef ushort4 vtmp_t;
};

template <> struct DegrainTypes<uint16_t> {
  typedef int32_t tmp_t;
  typedef ushort4 vpixel_t;
  typedef int4 vtmp_t;
};

/////////////////////////////////////////////////////////////////////////////
// KDeintKernel
/////////////////////////////////////////////////////////////////////////////

template <typename pixel_t>
class KDeintKernel : public CudaKernelBase, public IKDeintKernel<pixel_t>
{
public:
  typedef KDeintKernel Me;
  typedef typename DegrainTypes<pixel_t>::tmp_t tmp_t;
  typedef typename DegrainTypes<pixel_t>::vpixel_t vpixel_t;
  typedef typename DegrainTypes<pixel_t>::vtmp_t vtmp_t;

  virtual bool IsEnabled() const {
    return (env->GetDeviceType() == DEV_TYPE_CUDA);
  }

  void MemCpy(void* dst, const void* src, int nbytes)
  {
    dim3 threads(256);
    dim3 blocks(nblocks(nbytes, threads.x));
    memcpy_kernel << <blocks, threads, 0, stream >> > ((uint8_t*)dst, (uint8_t*)src, nbytes);
    DEBUG_SYNC;
  }

  void Copy(
    pixel_t* dst, int dst_pitch, const pixel_t* src, int src_pitch, int width, int height)
  {
    dim3 threads(32, 16);
    dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
    kl_copy<pixel_t> << <blocks, threads, 0, stream >> > (
      dst, dst_pitch, src, src_pitch, width, height);
    DEBUG_SYNC;
  }

  void PadFrame(pixel_t *ptr, int pitch, int hPad, int vPad, int width, int height)
  {
    { // H����
      dim3 threads(hPad, 32);
      dim3 blocks(2, nblocks(height, threads.y));
      kl_pad_frame_h<pixel_t> << <blocks, threads, 0, stream >> > (
        ptr + vPad * pitch, pitch, hPad, width, height);
      DEBUG_SYNC;
    }
    { // V�����i���ł�Pad���ꂽH���������܂ށj
      dim3 threads(32, vPad);
      dim3 blocks(nblocks(width + hPad * 2, threads.x), 2);
      kl_pad_frame_v<pixel_t> << <blocks, threads, 0, stream >> > (
        ptr, pitch, vPad, width + hPad * 2, height);
      DEBUG_SYNC;
    }
  }

  void CopyPad(
      pixel_t* dst, int dst_pitch, const pixel_t* src, int src_pitch, int hPad, int vPad, int width, int height) {
      dim3 threads(32, 16);
      dim3 blocks(nblocks(width + hPad*2, threads.x), nblocks(height + vPad*2, threads.y));
      kl_copy_pad<pixel_t> << <blocks, threads, 0, stream >> > (
          dst, dst_pitch, src, src_pitch, hPad, vPad, width, height);
      DEBUG_SYNC;
  }

  void VerticalWiener(
    pixel_t *pDst, const pixel_t *pSrc, int nDstPitch,
    int nSrcPitch, int nWidth, int nHeight, int bits_per_pixel)
  {
    const int max_pixel_value = sizeof(pixel_t) == 1 ? 255 : (1 << bits_per_pixel) - 1;

    dim3 threads(32, 16);
    dim3 blocks(nblocks(nWidth, threads.x), nblocks(nHeight, threads.y));
    kl_vertical_wiener<pixel_t> << <blocks, threads, 0, stream >> > (
      pDst, pSrc, nDstPitch, nSrcPitch, nWidth, nHeight, max_pixel_value);
    DEBUG_SYNC;
  }

  void HorizontalWiener(
    pixel_t *pDst, const pixel_t *pSrc, int nDstPitch,
    int nSrcPitch, int nWidth, int nHeight, int bits_per_pixel)
  {
    const int max_pixel_value = sizeof(pixel_t) == 1 ? 255 : (1 << bits_per_pixel) - 1;

    dim3 threads(32, 16);
    dim3 blocks(nblocks(nWidth, threads.x), nblocks(nHeight, threads.y));
    kl_horizontal_wiener<pixel_t> << <blocks, threads, 0, stream >> > (
      pDst, pSrc, nDstPitch, nSrcPitch, nWidth, nHeight, max_pixel_value);
    DEBUG_SYNC;
  }

  void RB2BilinearFiltered(
    pixel_t *pDst, const pixel_t *pSrc, int nDstPitch, int nSrcPitch, int nWidth, int nHeight)
  {
    dim3 threads(RB2B_BILINEAR_W, RB2B_BILINEAR_H);
    dim3 blocks(nblocks(nWidth * 2, RB2B_BILINEAR_W - 2), nblocks(nHeight, RB2B_BILINEAR_H));
    kl_RB2B_bilinear_filtered<pixel_t> << <blocks, threads, 0, stream >> > (
      pDst, pSrc, nDstPitch, nSrcPitch, nWidth, nHeight);
    DEBUG_SYNC;
  }

  void RB2BilinearFilteredPad(
      pixel_t *pDst, const pixel_t *pSrc, int nDstPitch, int nSrcPitch, int hPad, int vPad, int nWidth, int nHeight)
  {
      dim3 threads(RB2B_BILINEAR_W, RB2B_BILINEAR_H);
      dim3 blocks(nblocks(nWidth * 2, RB2B_BILINEAR_W - 2), nblocks(nHeight, RB2B_BILINEAR_H));
      kl_RB2B_bilinear_filtered_with_pad<pixel_t> << <blocks, threads, 0, stream >> > (
          pDst, pSrc, nDstPitch, nSrcPitch, hPad, vPad, nWidth, nHeight);
      DEBUG_SYNC;
  }

  template <int BLK_SIZE, int SEARCH, int NPEL, bool CHROMA, bool CPU_EMU>
  void launch_search(
    int batch,
    SearchBatchData<pixel_t> *pdata,
    int nBlkX, int nBlkY, int nPad,
    int nPitchY, int nPitchUV,
    int nImgPitchY, int nImgPitchUV, cudaStream_t stream)
  {
    dim3 threads(BLK_SIZE * 8);
    // �]���ȃu���b�N�͎d�������ɏI������̂Ŗ��Ȃ�
    dim3 blocks(batch, std::min(nBlkX, nBlkY));
    kl_search<pixel_t, BLK_SIZE, SEARCH, NPEL, CHROMA, CPU_EMU> << <blocks, threads, 0, stream >> > (
      pdata, nBlkX, nBlkY, nPad,
      nPitchY, nPitchUV, nImgPitchY, nImgPitchUV);
    DEBUG_SYNC;
  }

  typedef void (Me::*LAUNCH_SEARCH)(
    int batch,
    SearchBatchData<pixel_t> *pdata,
    int nBlkX, int nBlkY, int nPad,
    int nPitchY, int nPitchUV,
    int nImgPitchY, int nImgPitchUV, cudaStream_t stream);

  template <int BLK_SIZE, int NPEL, bool CHROMA>
  void launch_calc_all_sad(
    SearchBatchData<pixel_t> *pdata, int batch,
    int nBlkX, int nBlkY, int nPad,
    int nPitchY, int nPitchUV,
    int nImgPitchY, int nImgPitchUV, cudaStream_t stream)
  {
    dim3 threads(BLK_SIZE * 8);
    dim3 blocks(nBlkX, nBlkY, batch);
    kl_calc_all_sad <pixel_t, BLK_SIZE, NPEL, CHROMA> << <blocks, threads, 0, stream >> > (
      pdata, nBlkX, nBlkY, nPad,
      nPitchY, nPitchUV, nImgPitchY, nImgPitchUV);
    DEBUG_SYNC;
  }

  typedef void (Me::*LAUNCH_CALC_ALL_SAD)(
    SearchBatchData<pixel_t> *pdata, int batch,
    int nBlkX, int nBlkY, int nPad,
    int nPitchY, int nPitchUV,
    int nImgPitchY, int nImgPitchUV, cudaStream_t stream);

  int GetSearchBlockSize()
  {
    return sizeof(SearchBlock);
  }

  int GetSearchBatchSize()
  {
    return sizeof(SearchBatchData<pixel_t>);
  }

  void Search(
    int batch, VECTOR **out, void* _searchbatch, void* _hsearchbatch,
    int searchType, int nBlkX, int nBlkY, int nBlkSize, int nLogScale,
    int nLambdaLevel, int lsad, int penaltyZero, int penaltyGlobal, int penaltyNew,
    int nPel, bool chroma, int nPad, int nBlkSizeOvr, int nExtendedWidth, int nExptendedHeight,
    const pixel_t** pSrcY, const pixel_t** pSrcU, const pixel_t** pSrcV,
    const pixel_t** pRefY, const pixel_t** pRefU, const pixel_t** pRefV,
    int nPitchY, int nPitchUV, int nImgPitchY, int nImgPitchUV,
    const short2* globalMV, short2* vectors, int vectorsPitch, int* sads, int sadPitch, void* _searchblocks, int* prog, int* next)
  {
    assert(batch <= ANALYZE_MAX_BATCH);

    SearchBlock* searchblocks = (SearchBlock*)_searchblocks;
    SearchBatchData<pixel_t>* searchbatch = (SearchBatchData<pixel_t>*)_searchbatch;
    SearchBatchData<pixel_t>* hsearchbatch = (SearchBatchData<pixel_t>*)_hsearchbatch;

    {
      // set zeroMV and globalMV
      kl_init_const_vec << <dim3(2, batch), 1, 0, stream >> > (vectors, vectorsPitch, globalMV, nPel);
      DEBUG_SYNC;
    }

    { // prepare
      dim3 threads(32, 8);
      dim3 blocks(nblocks(nBlkX, threads.x), nblocks(nBlkY, threads.y), batch);
      kl_prepare_search << <blocks, threads, 0, stream >> > (
        nBlkX, nBlkY, nBlkSize, nLogScale, nLambdaLevel, lsad,
        penaltyZero, penaltyGlobal, penaltyNew,
        nPel, nPad, nBlkSizeOvr, nExtendedWidth, nExptendedHeight,
        vectors, vectorsPitch, sads, sadPitch, &vectors[nBlkX*nBlkY], searchblocks, prog, next);
      DEBUG_SYNC;

      //DataDebug<short2> v(vectors - 2, 2, env);
      //v.Show();

      //DataDebug<SearchBlock> d(searchblocks, nBlkX*nBlkY, env);
      //d.Show();
    }

    int fidx = ((nBlkSize == 8) ? 0 : (nBlkSize == 16) ? 4 : 8) + ((nPel == 1) ? 0 : 2) + (chroma ? 0 : 1);

    { // search
      // �f�o�b�O�p
#define CPU_EMU true
      LAUNCH_SEARCH table[2][12] =
      {
        { // exhaustive
          &Me::launch_search<8, 1, 1, true, CPU_EMU>,
          &Me::launch_search<8, 1, 1, false, CPU_EMU>,
          NULL,
          NULL,
          &Me::launch_search<16, 1, 1, true, CPU_EMU>,
          &Me::launch_search<16, 1, 1, false, CPU_EMU>,
          NULL,
          NULL,
          &Me::launch_search<32, 1, 1, true, CPU_EMU>,
          &Me::launch_search<32, 1, 1, false, CPU_EMU>,
          NULL,
          NULL,
        },
        { // hex
          &Me::launch_search<8, 2, 1, true, CPU_EMU>,
          &Me::launch_search<8, 2, 1, false, CPU_EMU>,
          &Me::launch_search<8, 2, 2, true, CPU_EMU>,
          &Me::launch_search<8, 2, 2, false, CPU_EMU>,
          &Me::launch_search<16, 2, 1, true, CPU_EMU>,
          &Me::launch_search<16, 2, 1, false, CPU_EMU>,
          &Me::launch_search<16, 2, 2, true, CPU_EMU>,
          &Me::launch_search<16, 2, 2, false, CPU_EMU>,
          &Me::launch_search<32, 2, 1, true, CPU_EMU>,
          &Me::launch_search<32, 2, 1, false, CPU_EMU>,
          &Me::launch_search<32, 2, 2, true, CPU_EMU>,
          &Me::launch_search<32, 2, 2, false, CPU_EMU>,
        }
      };
#undef CPU_EMU

      // �p�����[�^�����
      for (int i = 0; i < batch; ++i) {
        hsearchbatch[i].d.out = out[i];
        hsearchbatch[i].d.blocks = searchblocks + nBlkX * nBlkY * i;
        hsearchbatch[i].d.vectors = vectors + vectorsPitch * i;
        hsearchbatch[i].d.dst_sad = sads + sadPitch * i;
        hsearchbatch[i].d.prog = prog + nBlkX * i;
        hsearchbatch[i].d.next = next + i;
        hsearchbatch[i].d.pSrcY = pSrcY[i];
        hsearchbatch[i].d.pSrcU = pSrcU[i];
        hsearchbatch[i].d.pSrcV = pSrcV[i];
        hsearchbatch[i].d.pRefY = pRefY[i];
        hsearchbatch[i].d.pRefU = pRefU[i];
        hsearchbatch[i].d.pRefV = pRefV[i];
      }

      CUDA_CHECK(cudaMemcpyAsync(searchbatch, hsearchbatch, sizeof(searchbatch[0]) * batch, cudaMemcpyHostToDevice, stream));

      // �I�������������R�[���o�b�N��ǉ�
      //env->DeviceAddCallback([](void* arg) {
      //  delete[]((SearchBatchData<pixel_t>*)arg);
      //}, hsearchbatch);

      auto analyzef = table[(searchType == 8) ? 0 : 1][fidx];
      if (analyzef == NULL) {
        env->ThrowError("Unsupported search param combination");
      }
      (this->*analyzef)(batch, searchbatch, nBlkX, nBlkY, nPad,
        nPitchY, nPitchUV, nImgPitchY, nImgPitchUV, stream);

      //DataDebug<short2> d(vectors, nBlkX*nBlkY, env);
      //d.Show();
    }

    { // calc sad
      LAUNCH_CALC_ALL_SAD table[] =
      {
        &Me::launch_calc_all_sad<8, 1, true>,
        &Me::launch_calc_all_sad<8, 1, false>,
        &Me::launch_calc_all_sad<8, 2, true>,
        &Me::launch_calc_all_sad<8, 2, false>,
        &Me::launch_calc_all_sad<16, 1, true>,
        &Me::launch_calc_all_sad<16, 1, false>,
        &Me::launch_calc_all_sad<16, 2, true>,
        &Me::launch_calc_all_sad<16, 2, false>,
        &Me::launch_calc_all_sad<32, 1, true>,
        &Me::launch_calc_all_sad<32, 1, false>,
        &Me::launch_calc_all_sad<32, 2, true>,
        &Me::launch_calc_all_sad<32, 2, false>,
      };
#if 0
      // calc_all_sad�͏\���ȕ��񐫂�����̂Ńo�b�`���̓��[�v�ŉ�
      for (int i = 0; i < batch; ++i) {
        (this->*table[fidx])(nBlkX, nBlkY,
          vectors + vectorsPitch * i, sads + sadPitch * i, nPad,
          pSrcY[i], pSrcU[i], pSrcV[i], pRefY[i], pRefU[i], pRefV[i],
          nPitchY, nPitchUV, nImgPitchY, nImgPitchUV, stream);
        DEBUG_SYNC;
      }
#else
      (this->*table[fidx])(searchbatch, batch,
          nBlkX, nBlkY, nPad,
          nPitchY, nPitchUV, nImgPitchY, nImgPitchUV, stream);
      DEBUG_SYNC;
#endif
      //DataDebug<int> d(sads, nBlkX*nBlkY, env);
      //d.Show();
    }
  }

  void EstimateGlobalMV(int batch, const short2* vectors, int vectorsPitch, int nBlkCount, short2* globalMV)
  {
    kl_most_freq_mv << <dim3(2, batch), 1024, 0, stream >> > (vectors, vectorsPitch, nBlkCount, globalMV);
    DEBUG_SYNC;
    //DataDebug<short2> v1(globalMV, 1, env);
    //v1.Show();
    kl_mean_global_mv << <dim3(1, batch), 1024, 0, stream >> > (vectors, vectorsPitch, nBlkCount, globalMV);
    DEBUG_SYNC;
    //DataDebug<short2> v2(globalMV, 1, env);
    //v2.Show();
  }

  void InterpolatePrediction(
    int batch,
    const short2* src_vector, int srcVectorPitch, const int* src_sad, int srcSadPitch,
    short2* dst_vector, int dstVectorPitch, int* dst_sad, int dstSadPitch,
    int nSrcBlkX, int nSrcBlkY, int nDstBlkX, int nDstBlkY,
    int normFactor, int normov, int atotal, int aodd, int aeven)
  {
    dim3 threads(32, 8);
    dim3 blocks(nblocks(nDstBlkX, threads.x), nblocks(nDstBlkY, threads.y), batch);
    kl_interpolate_prediction << <blocks, threads, 0, stream >> > (
      src_vector, srcVectorPitch, src_sad, srcSadPitch,
      dst_vector, dstVectorPitch, dst_sad, dstSadPitch,
      nSrcBlkX, nSrcBlkY, nDstBlkX, nDstBlkY,
      normFactor, normov, atotal, aodd, aeven);
    DEBUG_SYNC;
  }

  void LoadMV(const VECTOR* in, short2* vectors, int* sads, int nBlkCount)
  {
    dim3 threads(256);
    dim3 blocks(nblocks(nBlkCount, threads.x));
    kl_load_mv << <blocks, threads, 0, stream >> > (in, vectors, sads, nBlkCount);
    DEBUG_SYNC;
  }

  void StoreMV(VECTOR* out, const short2* vectors, const int* sads, int nBlkCount)
  {
    dim3 threads(256);
    dim3 blocks(nblocks(nBlkCount, threads.x));
    kl_store_mv << <blocks, threads, 0, stream >> > (out, vectors, sads, nBlkCount);
    DEBUG_SYNC;
  }

  void WriteDefaultMV(VECTOR* dst, int nBlkCount, int verybigSAD)
  {
    dim3 threads(256);
    dim3 blocks(nblocks(nBlkCount, threads.x));
    kl_write_default_mv << <blocks, threads, 0, stream >> > (dst, nBlkCount, verybigSAD);
    DEBUG_SYNC;
  }
  void GetDegrainStructSize(int N, int& degrainBlock, int& degrainArg)
  {
    switch (N) {
    case 1:
      degrainBlock = sizeof(DegrainBlock<pixel_t, 1>);
      degrainArg = sizeof(DegrainArg<pixel_t, 1>);
      break;
    case 2:
      degrainBlock = sizeof(DegrainBlock<pixel_t, 2>);
      degrainArg = sizeof(DegrainArg<pixel_t, 2>);
      break;
    default:
      env->ThrowError("Degrain: Invalid N (%d)", N);
    }
  }

  template <typename vpixel_t, int N, int NPEL, int SHIFT, bool BINOMIAL>
  void launch_prepare_degrain(
    int nBlkX, int nBlkY, int nPad, int nBlkSize, int nTh2, int thSAD,
    const short* ovrwins,
    const int * sceneChangeB,
    const int * sceneChangeF,
    const DegrainArg<pixel_t, N>* parg,
    DegrainBlock<pixel_t, N>* pblocks,
    int nPitch, int nPitchSuper, int nImgPitch)
  {
    dim3 threads(32, 8);
    dim3 blocks(nblocks(nBlkX, threads.x), nblocks(nBlkY, threads.y));
    kl_prepare_degrain<pixel_t, vpixel_t, N, NPEL, SHIFT, BINOMIAL> << <blocks, threads, 0, stream >> > (
      nBlkX, nBlkY, nPad, nBlkSize, nTh2, thSAD, ovrwins, sceneChangeB, sceneChangeF, parg, pblocks, nPitch, nPitchSuper, nImgPitch);
    DEBUG_SYNC;
  }

  template <int N, int BLK_SIZE, int M>
  void launch_degrain_2x3_small(
    int nPatternX, int nPatternY,
    int nBlkX, int nBlkY, DegrainBlock<pixel_t, N>* data, tmp_t* pDst, int pitch, int pitchsuper)
  {
    dim3 threads(BLK_SIZE, BLK_SIZE, M);
    dim3 blocks(nblocks(nBlkX, 3 * 2 * M), nblocks(nBlkY, 2 * 2));
    kl_degrain_2x3<pixel_t, pixel_t, tmp_t, int, short, N, BLK_SIZE, M> << <blocks, threads, 0, stream >> > (
      nPatternX, nPatternY, nBlkX, nBlkY, data, pDst, pitch, pitchsuper);
    DEBUG_SYNC;
  }

  template <int N, int BLK_SIZE, int M>
  void launch_degrain_2x3(
    int nPatternX, int nPatternY,
    int nBlkX, int nBlkY, DegrainBlock<pixel_t, N>* data, tmp_t* pDst, int pitch4, int pitchsuper4)
  {
    dim3 threads(BLK_SIZE / 4, BLK_SIZE, M);
    dim3 blocks(nblocks(nBlkX, 3 * 2 * M), nblocks(nBlkY, 2 * 2));
    kl_degrain_2x3<pixel_t, vpixel_t, vtmp_t, int4, short4, N, BLK_SIZE, M> << <blocks, threads, 0, stream >> > (
      nPatternX, nPatternY, nBlkX, nBlkY, data, (vtmp_t*)pDst, pitch4, pitchsuper4);
    DEBUG_SYNC;
  }

  template <typename vpixel_t, typename vtmp_t>
  void launch_short_to_byte(
    vpixel_t* dst, const vtmp_t* src, int width4, int height, int pitch4, int max_pixel_value)
  {
    dim3 threads(32, 16);
    dim3 blocks(nblocks(width4, threads.x), nblocks(height, threads.y));
    kl_short_to_byte<vpixel_t, vtmp_t> << <blocks, threads, 0, stream >> > (
      dst, src, width4, height, pitch4, max_pixel_value);
    DEBUG_SYNC;
  }

  typedef void (Me::*DEGRAINN)(
    int nWidth, int nHeight,
    int nBlkX, int nBlkY, int nPad, int nBlkSize, int nPel, int nBitsPerPixel,
    bool* enableYUV, bool* isUsableB, bool* isUsableF,
    int nTh1, int nTh2, int thSAD, int thSADC, bool binomial,
    const short* ovrwins, const short* overwinsUV,
    const VECTOR** mvB, const VECTOR** mvF,
    const pixel_t** pSrc, pixel_t** pDst, tmp_t** pTmp, const pixel_t** pRefB, const pixel_t** pRefF,
    int nPitch, int nPitchUV, int nPitchSuperY, int nPitchSuperUV, int nImgPitch, int nImgPitchUV,
    void* _degrainblocks, void* _degraindarg, int *sceneChangeB, int *sceneChangeF
    );

  // pTmp��pSrc�Ɠ���pitch�ł��邱��
  template <int N>
  void DegrainN(
    int nWidth, int nHeight,
    int nBlkX, int nBlkY, int nPad, int nBlkSize, int nPel, int nBitsPerPixel,
    bool* enableYUV, bool* isUsableB, bool* isUsableF,
    int nTh1, int nTh2, int thSAD, int thSADC, bool binomial,
    const short* ovrwins, const short* overwinsUV,
    const VECTOR** mvB, const VECTOR** mvF,
    const pixel_t** pSrc, pixel_t** pDst, tmp_t** pTmp, const pixel_t** pRefB, const pixel_t** pRefF,
    int nPitchY, int nPitchUV, int nPitchSuperY, int nPitchSuperUV, int nImgPitchY, int nImgPitchUV,
    void* _degrainblocks, void* _degraindarg, int *sceneChangeB, int *sceneChangeF
  )
  {

    int nOverlap = nBlkSize / 2;
    int nWidth_B = nBlkX*(nBlkSize - nOverlap) + nOverlap;
    int nHeight_B = nBlkY*(nBlkSize - nOverlap) + nOverlap;

    // degrainarg�쐬
    DegrainArg<pixel_t, N> *hargs = new DegrainArg<pixel_t, N>[3];
    DegrainArg<pixel_t, N> *dargs = (DegrainArg<pixel_t, N>*)_degraindarg;

    for (int p = 0; p < 3; ++p) {
      DegrainArgData<pixel_t, N>& arg = hargs[p].d;
      if (enableYUV[p]) {
        for (int i = 0; i < N; ++i) {
          arg.mvB[i] = mvB[i];
          arg.mvF[i] = mvF[i];
        }
        arg.pSrc = pSrc[p];
        for (int i = 0; i < N; ++i) {
          arg.pRefB[i] = pRefB[p + i * 3];
          arg.pRefF[i] = pRefF[p + i * 3];
        }
        for (int i = 0; i < N; ++i) {
          arg.isUsableB[i] = isUsableB[i];
          arg.isUsableF[i] = isUsableF[i];
        }
      }
    }

    CUDA_CHECK(cudaMemcpyAsync(dargs, hargs, sizeof(hargs[0]) * 3, cudaMemcpyHostToDevice, stream));

    // �I�������������R�[���o�b�N��ǉ�
    env->DeviceAddCallback([](void* arg) {
      delete[]((DegrainArg<pixel_t, N>*)arg);
    }, hargs);

    typedef void (Me::*PREPARE)(
      int nBlkX, int nBlkY, int nPad, int nBlkSize, int nTh2, int thSAD,
      const short* ovrwins,
      const int * sceneChangeB,
      const int * sceneChangeF,
      const DegrainArg<pixel_t, N>* parg,
      DegrainBlock<pixel_t, N>* blocks,
      int nPitch, int nPitchSuper, int nImgPitch);

    PREPARE prepare_func, prepareuv_func;

    switch (nPel) {
    case 1:
      if (binomial) {
        prepare_func = &Me::launch_prepare_degrain<vpixel_t, N, 1, 0, true>;
        prepareuv_func = &Me::launch_prepare_degrain<vpixel_t, N, 1, 1, true>;
      }
      else {
        prepare_func = &Me::launch_prepare_degrain<vpixel_t, N, 1, 0, false>;
        prepareuv_func = &Me::launch_prepare_degrain<vpixel_t, N, 1, 1, false>;
      }
      break;
    case 2:
      if (binomial) {
        prepare_func = &Me::launch_prepare_degrain<vpixel_t, N, 2, 0, true>;
        prepareuv_func = &Me::launch_prepare_degrain<vpixel_t, N, 2, 1, true>;
      }
      else {
        prepare_func = &Me::launch_prepare_degrain<vpixel_t, N, 2, 0, false>;
        prepareuv_func = &Me::launch_prepare_degrain<vpixel_t, N, 2, 1, false>;
      }
      break;
    default:
      env->ThrowError("[Degrain] ���Ή�Pel");
    }

    DegrainBlock<pixel_t, N>* degrainblocks = (DegrainBlock<pixel_t, N>*)_degrainblocks;
    const int max_pixel_value = (1 << nBitsPerPixel) - 1;

    // YUV���[�v
    for (int p = 0; p < 3; ++p) {

      PREPARE prepare = (p == 0) ? prepare_func : prepareuv_func;
      int shift = (p == 0) ? 0 : 1;
      int blksize = nBlkSize >> shift;
      int width = nWidth >> shift;
      int width_b = nWidth_B >> shift;
      int width4 = width / 4;
      int width_b4 = width_b / 4;
      int height = nHeight >> shift;
      int height_b = nHeight_B >> shift;
      int pitch = (p == 0) ? nPitchY : nPitchUV;
      int pitchsuper = (p == 0) ? nPitchSuperY : nPitchSuperUV;
      int pitch4 = pitch / 4;
      int pitchsuper4 = pitchsuper / 4;
      int imgpitch = (p == 0) ? nImgPitchY : nImgPitchUV;

      if (enableYUV[p]) {

        // DegrainBlockData�쐬
        (this->*prepare)(
          nBlkX, nBlkY, nPad >> shift, blksize, nTh2,
          (p == 0) ? thSAD : thSADC,
          (p == 0) ? ovrwins : overwinsUV,
          sceneChangeB, sceneChangeF, &dargs[p],
          degrainblocks, pitch, pitchsuper, imgpitch);

        // pTmp������
        launch_elementwise<vtmp_t, SetZeroFunction<vtmp_t>>(
          (vtmp_t*)pTmp[p], width_b4, height_b, pitch4, stream);
        DEBUG_SYNC;

        void(Me::*degrain_func)(
          int nPatternX, int nPatternY,
          int nBlkX, int nBlkY, DegrainBlock<pixel_t, N>* data, tmp_t* pDst, int pitchX, int pitchsuperX);

        int pitchX, pitchsuperX;
        if (blksize < 8) {
          pitchX = pitch;
          pitchsuperX = pitchsuper;
        }
        else {
          pitchX = pitch4;
          pitchsuperX = pitchsuper4;
        }

        switch (blksize) {
        case 4:
          degrain_func = &Me::launch_degrain_2x3_small<N, 4, 8>;  // 4x4x16  = 128threads
          break;
        case 8:
          degrain_func = &Me::launch_degrain_2x3<N, 8, 8>;  // 8x2x8  = 128threads
          break;
        case 16:
          degrain_func = &Me::launch_degrain_2x3<N, 16, 1>; // 16x4x1 =  64threads
          break;
        case 32:
          degrain_func = &Me::launch_degrain_2x3<N, 32, 1>; // 32x8x1 = 256threads
          break;
        default:
          env->ThrowError("[Degrain] ���Ή��u���b�N�T�C�Y");
        }

        // 4��J�[�l���Ăяo��
        (this->*degrain_func)(0, 0, nBlkX, nBlkY, degrainblocks, pTmp[p], pitchX, pitchsuperX);
        (this->*degrain_func)(1, 0, nBlkX, nBlkY, degrainblocks, pTmp[p], pitchX, pitchsuperX);
        (this->*degrain_func)(0, 1, nBlkX, nBlkY, degrainblocks, pTmp[p], pitchX, pitchsuperX);
        (this->*degrain_func)(1, 1, nBlkX, nBlkY, degrainblocks, pTmp[p], pitchX, pitchsuperX);

        // tmp_t -> pixel_t �ϊ�
        launch_short_to_byte<vpixel_t, vtmp_t>(
          (vpixel_t*)pDst[p], (const vtmp_t*)pTmp[p], width_b4, height_b, pitch4, max_pixel_value);

#if 0
        DataDebug<tmp_t> dtmp(pTmp[p], height_b * pitch, env);
        DataDebug<pixel_t> ddst(pDst[p], height_b * pitch, env);
        dtmp.Show();
#endif

        // right non-covered region��src����R�s�[
        if (nWidth_B < nWidth) {
          launch_elementwise<vpixel_t, vpixel_t, CopyFunction<vpixel_t>>(
            (vpixel_t*)(pDst[p] + (nWidth_B >> shift)),
            (const vpixel_t*)(pSrc[p] + (nWidth_B >> shift)),
            ((nWidth - nWidth_B) >> shift) / 4, nBlkY * blksize, pitch4, stream);
        }

        // bottom uncovered region��src����R�s�[
        if (nHeight_B < nHeight) {
          launch_elementwise<vpixel_t, vpixel_t, CopyFunction<vpixel_t>>(
            (vpixel_t*)(pDst[p] + (nHeight_B >> shift) * pitch),
            (const vpixel_t*)(pSrc[p] + (nHeight_B >> shift) * pitch),
            width4, (nHeight - nHeight_B) >> shift, pitch4, stream);
        }
      }
      else {
        // src����R�s�[
        launch_elementwise<vpixel_t, vpixel_t, CopyFunction<vpixel_t>>(
          (vpixel_t*)pDst[p], (const vpixel_t*)pSrc[p], width4, height, pitch4, stream);
      }
    }
  }

  void Degrain(
    int N, int nWidth, int nHeight, int nBlkX, int nBlkY, int nPad, int nBlkSize, int nPel, int nBitsPerPixel,
    bool* enableYUV,
    bool* isUsableB, bool* isUsableF,
    int nTh1, int nTh2, int thSAD, int thSADC, bool binomial,
    const short* ovrwins, const short* overwinsUV,
    const VECTOR** mvB, const VECTOR** mvF,
    const pixel_t** pSrc, pixel_t** pDst, tmp_t** pTmp, const pixel_t** pRefB, const pixel_t** pRefF,
    int nPitchY, int nPitchUV,
    int nPitchSuperY, int nPitchSuperUV, int nImgPitchY, int nImgPitchUV,
    void* _degrainblock, void* _degrainarg, int* sceneChange)
  {
    int numRef = N * 2;
    int numBlks = nBlkX * nBlkY;

    // SceneChange���o
    int *sceneChangeB = sceneChange;
    int *sceneChangeF = sceneChange + N;

    kl_init_scene_change << <1, numRef, 0, stream >> > (sceneChange);
    DEBUG_SYNC;
    for (int i = 0; i < N; ++i) {
      dim3 threads(256);
      dim3 blocks(nblocks(numBlks, threads.x));
      if (isUsableB[i]) {
        kl_scene_change << <blocks, threads, 0, stream >> > (mvB[i], numBlks, nTh1, &sceneChangeB[i]);
      }
      if (isUsableF[i]) {
        kl_scene_change << <blocks, threads, 0, stream >> > (mvF[i], numBlks, nTh1, &sceneChangeF[i]);
      }
      DEBUG_SYNC;
    }

    DEGRAINN degrain;
    switch (N) {
    case 1:
      degrain = &Me::DegrainN<1>;
      break;
    case 2:
      degrain = &Me::DegrainN<2>;
      break;
    default:
      env->ThrowError("[Degrain] ���Ή�N�ł�");
    }

    (this->*degrain)(
      nWidth, nHeight, nBlkX, nBlkY, nPad, nBlkSize, nPel, nBitsPerPixel,
      enableYUV, isUsableB, isUsableF,
      nTh1, nTh2, thSAD, thSADC, binomial,
      ovrwins, overwinsUV, mvB, mvF,
      pSrc, pDst, pTmp, pRefB, pRefF,
      nPitchY, nPitchUV, nPitchSuperY, nPitchSuperUV, nImgPitchY, nImgPitchUV,
      _degrainblock, _degrainarg, sceneChangeB, sceneChangeF);
  }

  int GetCompensateStructSize() {
    return sizeof(CompensateBlock<pixel_t>);
  }

  template <int NPEL, int SHIFT>
  void launch_prepare_compensate(
    int nBlkX, int nBlkY, int nPad, int nBlkSize, int nTh2, int time256, int thSAD,
    const short* ovrwins,
    const int * __restrict__ sceneChange,
    const VECTOR *mv,
    const pixel_t *pRef0,
    const pixel_t *pRef,
    CompensateBlock<pixel_t>* pblocks,
    int nPitchSuper, int nImgPitch)
  {
    dim3 threads(32, 8);
    dim3 blocks(nblocks(nBlkX, threads.x), nblocks(nBlkY, threads.y));
    kl_prepare_compensate<pixel_t, NPEL, SHIFT> << <blocks, threads, 0, stream >> > (
      nBlkX, nBlkY, nPad, nBlkSize, nTh2, time256, thSAD, ovrwins, sceneChange,
      mv, pRef0, pRef, pblocks, nPitchSuper, nImgPitch);
    DEBUG_SYNC;
  }

  template <int BLK_SIZE, int M>
  void launch_compensate_2x3_small(
    int nPatternX, int nPatternY,
    int nBlkX, int nBlkY, CompensateBlock<pixel_t>* data, tmp_t* pDst, int pitch, int pitchsuper)
  {
    dim3 threads(BLK_SIZE, BLK_SIZE, M);
    dim3 blocks(nblocks(nBlkX, 3 * 2 * M), nblocks(nBlkY, 2 * 2));
    kl_compensate_2x3<pixel_t, tmp_t, int, short, BLK_SIZE, M> << <blocks, threads, 0, stream >> > (
      nPatternX, nPatternY, nBlkX, nBlkY, data, pDst, pitch, pitchsuper);
    DEBUG_SYNC;
  }

  template <int BLK_SIZE, int M>
  void launch_compensate_2x3(
    int nPatternX, int nPatternY,
    int nBlkX, int nBlkY, CompensateBlock<pixel_t>* data, tmp_t* pDst, int pitch4, int pitchsuper4)
  {
    dim3 threads(BLK_SIZE / 4, BLK_SIZE, M);
    dim3 blocks(nblocks(nBlkX, 3 * 2 * M), nblocks(nBlkY, 2 * 2));
    kl_compensate_2x3<pixel_t, vtmp_t, int4, short4, BLK_SIZE, M> << <blocks, threads, 0, stream >> > (
      nPatternX, nPatternY, nBlkX, nBlkY, data, (vtmp_t*)pDst, pitch4, pitchsuper4);
    DEBUG_SYNC;
  }

  void launch_short_to_byte_or_copy_src(
    const void** __restrict__ pflag,
    vpixel_t* dst, const vpixel_t* src, const vtmp_t* tmp, int width4, int height, int pitch4, int max_pixel_value)
  {
    dim3 threads(32, 16);
    dim3 blocks(nblocks(width4, threads.x), nblocks(height, threads.y));
    kl_short_to_byte_or_copy_src<vpixel_t, vtmp_t> << <blocks, threads, 0, stream >> > (
      pflag, dst, src, tmp, width4, height, pitch4, max_pixel_value);
    DEBUG_SYNC;
  }

  void Compensate(
    int nWidth, int nHeight, int nBlkX, int nBlkY, int nPad, int nBlkSize, int nPel, int nBitsPerPixel,
    int nTh1, int nTh2, int time256, int thSAD,
    const short* ovrwins, const short* overwinsUV,
    const VECTOR* mv,
    const pixel_t** pSrc, pixel_t** pDst, tmp_t** pTmp, const pixel_t** pRef,
    int nPitchY, int nPitchUV,
    int nPitchSuperY, int nPitchSuperUV, int nImgPitchY, int nImgPitchUV,
    void* _compensateblock, int* sceneChange)
  {
    int numBlks = nBlkX * nBlkY;

    // SceneChange���o
    kl_init_scene_change << <1, 1, 0, stream >> > (sceneChange);
    DEBUG_SYNC;
    {
      dim3 threads(256);
      dim3 blocks(nblocks(numBlks, threads.x));
      kl_scene_change << <blocks, threads, 0, stream >> > (mv, numBlks, nTh1, sceneChange);
      DEBUG_SYNC;
    }

    int nOverlap = nBlkSize / 2;
    int nWidth_B = nBlkX*(nBlkSize - nOverlap) + nOverlap;
    int nHeight_B = nBlkY*(nBlkSize - nOverlap) + nOverlap;

    typedef void (Me::*PREPARE)(
      int nBlkX, int nBlkY, int nPad, int nBlkSize, int nTh2, int time256, int thSAD,
      const short* ovrwins,
      const int * __restrict__ sceneChange,
      const VECTOR *mv,
      const pixel_t *pRef0,
      const pixel_t *pRef,
      CompensateBlock<pixel_t>* pblocks,
      int nPitchSuper, int nImgPitch);

    PREPARE prepare_func, prepareuv_func;

    switch (nPel) {
    case 1:
      prepare_func = &Me::launch_prepare_compensate<1, 0>;
      prepareuv_func = &Me::launch_prepare_compensate<1, 1>;
      break;
    case 2:
      prepare_func = &Me::launch_prepare_compensate<2, 0>;
      prepareuv_func = &Me::launch_prepare_compensate<2, 1>;
      break;
    default:
      env->ThrowError("[Compensate] ���Ή�Pel");
    }

    CompensateBlock<pixel_t>* compensateblocks = (CompensateBlock<pixel_t>*)_compensateblock;
    const int max_pixel_value = (1 << nBitsPerPixel) - 1;

    // YUV���[�v
    for (int p = 0; p < 3; ++p) {

      PREPARE prepare = (p == 0) ? prepare_func : prepareuv_func;
      int shift = (p == 0) ? 0 : 1;
      int blksize = nBlkSize >> shift;
      int width = nWidth >> shift;
      int width_b = nWidth_B >> shift;
      int width4 = width / 4;
      int width_b4 = width_b / 4;
      //int height = nHeight >> shift;
      int height_b = nHeight_B >> shift;
      int pitch = (p == 0) ? nPitchY : nPitchUV;
      int pitchsuper = (p == 0) ? nPitchSuperY : nPitchSuperUV;
      int pitch4 = pitch / 4;
      int pitchsuper4 = pitchsuper / 4;
      int imgpitch = (p == 0) ? nImgPitchY : nImgPitchUV;

      // DegrainBlockData�쐬
      (this->*prepare)(
        nBlkX, nBlkY, nPad >> shift, blksize, nTh2, time256, thSAD,
        (p == 0) ? ovrwins : overwinsUV,
        sceneChange, mv, pRef[0 + p], pRef[3 + p],
        compensateblocks, pitchsuper, imgpitch);

      // pTmp������
      launch_elementwise<vtmp_t, SetZeroFunction<vtmp_t>>(
        (vtmp_t*)pTmp[p], width_b4, height_b, pitch4, stream);
      DEBUG_SYNC;

      void(Me::*compensate_func)(
        int nPatternX, int nPatternY,
        int nBlkX, int nBlkY, CompensateBlock<pixel_t>* data, tmp_t* pDst, int pitchX, int pitchsuperX);

      int pitchX, pitchsuperX;
      if (blksize < 8) {
        pitchX = pitch;
        pitchsuperX = pitchsuper;
      }
      else {
        pitchX = pitch4;
        pitchsuperX = pitchsuper4;
      }

      switch (blksize) {
      case 4:
        compensate_func = &Me::launch_compensate_2x3_small<4, 8>;  // 4x4x8  = 128threads
        break;
      case 8:
        compensate_func = &Me::launch_compensate_2x3<8, 8>;  // 8x2x8  = 128threads
        break;
      case 16:
        compensate_func = &Me::launch_compensate_2x3<16, 1>; // 16x4x1 =  64threads
        break;
      case 32:
        compensate_func = &Me::launch_compensate_2x3<32, 1>; // 32x8x1 = 256threads
        break;
      default:
        env->ThrowError("���Ή��u���b�N�T�C�Y");
      }

      // 4��J�[�l���Ăяo��
      (this->*compensate_func)(0, 0, nBlkX, nBlkY, compensateblocks, pTmp[p], pitchX, pitchsuperX);
      (this->*compensate_func)(1, 0, nBlkX, nBlkY, compensateblocks, pTmp[p], pitchX, pitchsuperX);
      (this->*compensate_func)(0, 1, nBlkX, nBlkY, compensateblocks, pTmp[p], pitchX, pitchsuperX);
      (this->*compensate_func)(1, 1, nBlkX, nBlkY, compensateblocks, pTmp[p], pitchX, pitchsuperX);

      // tmp_t -> pixel_t �ϊ�
      launch_short_to_byte_or_copy_src(
        (const void**)&compensateblocks[0].d.winOver,
        (vpixel_t*)pDst[p], (const vpixel_t*)pSrc[p], (const vtmp_t*)pTmp[p], width_b4, height_b, pitch4, max_pixel_value);

#if 0
      DataDebug<tmp_t> dtmp(pTmp[p], height_b * pitch, env);
      DataDebug<pixel_t> ddst(pDst[p], height_b * pitch, env);
      dtmp.Show();
#endif

      // right non-covered region��src����R�s�[
      if (nWidth_B < nWidth) {
        launch_elementwise<vpixel_t, vpixel_t, CopyFunction<vpixel_t>>(
          (vpixel_t*)(pDst[p] + (nWidth_B >> shift)),
          (const vpixel_t*)(pSrc[p] + (nWidth_B >> shift)),
          ((nWidth - nWidth_B) >> shift) / 4, nBlkY * blksize, pitch4, stream);
      }

      // bottom uncovered region��src����R�s�[
      if (nHeight_B < nHeight) {
        launch_elementwise<vpixel_t, vpixel_t, CopyFunction<vpixel_t>>(
          (vpixel_t*)(pDst[p] + (nHeight_B >> shift) * pitch),
          (const vpixel_t*)(pSrc[p] + (nHeight_B >> shift) * pitch),
          width4, (nHeight - nHeight_B) >> shift, pitch4, stream);
      }
    }
  }

};

/////////////////////////////////////////////////////////////////////////////
// IKDeintCUDAImpl
/////////////////////////////////////////////////////////////////////////////

class IMVCUDAImpl : public IMVCUDA
{
  KDeintKernel<uint8_t> k8;
  KDeintKernel<uint16_t> k16;

public:
  virtual void SetEnv(PNeoEnv env) {
    k8.SetEnv(env);
    k16.SetEnv(env);
  }

  virtual bool IsEnabled() const {
    return k8.IsEnabled();
  }

  virtual IKDeintKernel<uint8_t>* get(uint8_t) { return &k8; }
  virtual IKDeintKernel<uint16_t>* get(uint16_t) { return &k16; }
};

IMVCUDA* CreateKDeintCUDA()
{
  return new IMVCUDAImpl();
}
