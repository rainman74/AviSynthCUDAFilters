
#include <stdint.h>
#include <avisynth.h>

#include <algorithm>

#include "CommonFunctions.h"
#include "KFM.h"
#include "TextOut.h"

#include "VectorFunctions.cuh"
#include "ReduceKernel.cuh"
#include "KFMFilterBase.cuh"
#include "Copy.h"

__global__ void kl_init_fmcount(FMCount* dst)
{
  int tx = threadIdx.x;
  dst[tx].move = dst[tx].shima = dst[tx].lshima = 0;
}

enum {
  FM_COUNT_TH_W = 32,
  FM_COUNT_TH_H = 16,
  FM_COUNT_THREADS = FM_COUNT_TH_W * FM_COUNT_TH_H,
};

enum {
  DC_OVERLAP = 4,
  DC_BLOCK_SIZE = 8,
  DC_BLOCK_TH_W = 8,
  DC_BLOCK_TH_H = 4,
};

template <typename pixel_t>
__host__ __device__ int calc_combe(
  pixel_t L0, pixel_t L1, pixel_t L2, pixel_t L3,
  pixel_t L4, pixel_t L5, pixel_t L6, pixel_t L7)
{
  auto diff8 = absdiff(L0, L7);
  auto diffT = absdiff(L0, L1) + absdiff(L1, L2) + absdiff(L2, L3) + absdiff(L3, L4) + absdiff(L4, L5) + absdiff(L5, L6) + absdiff(L6, L7) - diff8;
  auto diffE = absdiff(L0, L2) + absdiff(L2, L4) + absdiff(L4, L6) + absdiff(L6, L7) - diff8;
  auto diffO = absdiff(L0, L1) + absdiff(L1, L3) + absdiff(L3, L5) + absdiff(L5, L7) - diff8;
  return diffT - diffE - diffO;
}

template <typename pixel_t>
__host__ __device__ int calc_diff(
  pixel_t L00, pixel_t L10, pixel_t L01, pixel_t L11,
  pixel_t L02, pixel_t L12, pixel_t L03, pixel_t L13)
{
  return absdiff(L00, L10) + absdiff(L01, L11) + absdiff(L02, L12) + absdiff(L03, L13);
}

template<typename pixel_t, bool parity>
void cpu_analyze_frame(uchar2* __restrict__ flag, int fpitch,
  const pixel_t* __restrict__ f0, const pixel_t* __restrict__ f1,
  int pitch, int nBlkX, int nBlkY, int shift)
{
  for (int by = 0; by < nBlkY - 1; ++by) {
    for (int bx = 0; bx < nBlkX - 1; ++bx) {
      int sum[4] = { 0 };
      for (int tx = 0; tx < DC_BLOCK_SIZE; ++tx) {
        int x = bx * DC_OVERLAP + tx;
        int y = by * DC_OVERLAP;

        {
          auto T00 = f0[x + (y + 0) * pitch];
          auto B00 = f0[x + (y + 1) * pitch];
          auto T01 = f0[x + (y + 2) * pitch];
          auto B01 = f0[x + (y + 3) * pitch];
          auto T02 = f0[x + (y + 4) * pitch];
          auto B02 = f0[x + (y + 5) * pitch];
          auto T03 = f0[x + (y + 6) * pitch];
          auto B03 = f0[x + (y + 7) * pitch];
          int tmp = calc_combe(T00, B00, T01, B01, T02, B02, T03, B03);
          if (parity) { // TFF
            // top
            sum[0] += tmp;
          }
          else { // BFF
            // bottom
            sum[2] += tmp;
          }
        }

        if (parity) { // TFF: B0 <-> T1
          auto T10 = f1[x + (y + 0) * pitch];
          auto B00 = f0[x + (y + 1) * pitch];
          auto T11 = f1[x + (y + 2) * pitch];
          auto B01 = f0[x + (y + 3) * pitch];
          auto T12 = f1[x + (y + 4) * pitch];
          auto B02 = f0[x + (y + 5) * pitch];
          auto T13 = f1[x + (y + 6) * pitch];
          auto B03 = f0[x + (y + 7) * pitch];
          // bottom
          sum[2] += calc_combe(T10, B00, T11, B01, T12, B02, T13, B03);
        }
        else { // BFF: T0 <-> B1
          auto T00 = f0[x + (y + 0) * pitch];
          auto B10 = f1[x + (y + 1) * pitch];
          auto T01 = f0[x + (y + 2) * pitch];
          auto B11 = f1[x + (y + 3) * pitch];
          auto T02 = f0[x + (y + 4) * pitch];
          auto B12 = f1[x + (y + 5) * pitch];
          auto T03 = f0[x + (y + 6) * pitch];
          auto B13 = f1[x + (y + 7) * pitch];
          // top
          sum[0] += calc_combe(T00, B10, T01, B11, T02, B12, T03, B13);
        }

        { // top
          auto T00 = f0[x + (y + 0) * pitch];
          auto T10 = f1[x + (y + 0) * pitch];
          auto T01 = f0[x + (y + 2) * pitch];
          auto T11 = f1[x + (y + 2) * pitch];
          auto T02 = f0[x + (y + 4) * pitch];
          auto T12 = f1[x + (y + 4) * pitch];
          auto T03 = f0[x + (y + 6) * pitch];
          auto T13 = f1[x + (y + 6) * pitch];
          sum[1] += calc_diff(T00, T10, T01, T11, T02, T12, T03, T13);
        }

        { // bottom
          auto B00 = f0[x + (y + 1) * pitch];
          auto B10 = f1[x + (y + 1) * pitch];
          auto B01 = f0[x + (y + 3) * pitch];
          auto B11 = f1[x + (y + 3) * pitch];
          auto B02 = f0[x + (y + 5) * pitch];
          auto B12 = f1[x + (y + 5) * pitch];
          auto B03 = f0[x + (y + 7) * pitch];
          auto B13 = f1[x + (y + 7) * pitch];
          sum[3] += calc_diff(B00, B10, B01, B11, B02, B12, B03, B13);
        }
      }

      flag[(bx + 1) + (2 * (by + 1) + 0) * fpitch].x = (uint8_t)clamp(sum[0] >> shift, 0, 255);
      flag[(bx + 1) + (2 * (by + 1) + 0) * fpitch].y = (uint8_t)clamp(sum[1] >> shift, 0, 255);
      flag[(bx + 1) + (2 * (by + 1) + 1) * fpitch].x = (uint8_t)clamp(sum[2] >> shift, 0, 255);
      flag[(bx + 1) + (2 * (by + 1) + 1) * fpitch].y = (uint8_t)clamp(sum[3] >> shift, 0, 255);
    }
  }
}

template<typename pixel_t, bool parity>
__global__ void kl_analyze_frame(uchar2* __restrict__ flag, int fpitch,
  const pixel_t* __restrict__ f0, const pixel_t* __restrict__ f1,
  int pitch, int nBlkX, int nBlkY, int shift)
{
  int tx = threadIdx.x;
  int bx = blockIdx.x * DC_BLOCK_TH_W + threadIdx.y;
  int by = blockIdx.y * DC_BLOCK_TH_H + threadIdx.z;

  if (bx < nBlkX - 1 && by < nBlkY - 1) {
    int sum[4];
    int x = bx * DC_OVERLAP + tx;
    int y = by * DC_OVERLAP;

    {
      auto T00 = f0[x + (y + 0) * pitch];
      auto B00 = f0[x + (y + 1) * pitch];
      auto T01 = f0[x + (y + 2) * pitch];
      auto B01 = f0[x + (y + 3) * pitch];
      auto T02 = f0[x + (y + 4) * pitch];
      auto B02 = f0[x + (y + 5) * pitch];
      auto T03 = f0[x + (y + 6) * pitch];
      auto B03 = f0[x + (y + 7) * pitch];
      int tmp = calc_combe(T00, B00, T01, B01, T02, B02, T03, B03);
      if (parity) { // TFF
                    // top
        sum[0] = tmp;
      }
      else { // BFF
             // bottom
        sum[2] = tmp;
      }
    }

    if (parity) { // TFF: B0 <-> T1
      auto T10 = f1[x + (y + 0) * pitch];
      auto B00 = f0[x + (y + 1) * pitch];
      auto T11 = f1[x + (y + 2) * pitch];
      auto B01 = f0[x + (y + 3) * pitch];
      auto T12 = f1[x + (y + 4) * pitch];
      auto B02 = f0[x + (y + 5) * pitch];
      auto T13 = f1[x + (y + 6) * pitch];
      auto B03 = f0[x + (y + 7) * pitch];
      // bottom
      sum[2] = calc_combe(T10, B00, T11, B01, T12, B02, T13, B03);
    }
    else { // BFF: T0 <-> B1
      auto T00 = f0[x + (y + 0) * pitch];
      auto B10 = f1[x + (y + 1) * pitch];
      auto T01 = f0[x + (y + 2) * pitch];
      auto B11 = f1[x + (y + 3) * pitch];
      auto T02 = f0[x + (y + 4) * pitch];
      auto B12 = f1[x + (y + 5) * pitch];
      auto T03 = f0[x + (y + 6) * pitch];
      auto B13 = f1[x + (y + 7) * pitch];
      // top
      sum[0] = calc_combe(T00, B10, T01, B11, T02, B12, T03, B13);
    }

    { // top
      auto T00 = f0[x + (y + 0) * pitch];
      auto T10 = f1[x + (y + 0) * pitch];
      auto T01 = f0[x + (y + 2) * pitch];
      auto T11 = f1[x + (y + 2) * pitch];
      auto T02 = f0[x + (y + 4) * pitch];
      auto T12 = f1[x + (y + 4) * pitch];
      auto T03 = f0[x + (y + 6) * pitch];
      auto T13 = f1[x + (y + 6) * pitch];
      sum[1] = calc_diff(T00, T10, T01, T11, T02, T12, T03, T13);
    }

    { // bottom
      auto B00 = f0[x + (y + 1) * pitch];
      auto B10 = f1[x + (y + 1) * pitch];
      auto B01 = f0[x + (y + 3) * pitch];
      auto B11 = f1[x + (y + 3) * pitch];
      auto B02 = f0[x + (y + 5) * pitch];
      auto B12 = f1[x + (y + 5) * pitch];
      auto B03 = f0[x + (y + 7) * pitch];
      auto B13 = f1[x + (y + 7) * pitch];
      sum[3] = calc_diff(B00, B10, B01, B11, B02, B12, B03, B13);
    }

    dev_reduceN_warp<int, 4, DC_BLOCK_SIZE, AddReducer<int>>(tx, sum);

    if (tx == 0) {
      flag[(bx + 1) + (2 * (by + 1) + 0) * fpitch].x = (uint8_t)clamp(sum[0] >> shift, 0, 255);
      flag[(bx + 1) + (2 * (by + 1) + 0) * fpitch].y = (uint8_t)clamp(sum[1] >> shift, 0, 255);
      flag[(bx + 1) + (2 * (by + 1) + 1) * fpitch].x = (uint8_t)clamp(sum[2] >> shift, 0, 255);
      flag[(bx + 1) + (2 * (by + 1) + 1) * fpitch].y = (uint8_t)clamp(sum[3] >> shift, 0, 255);
    }
  }
}

class KFMSuper : public KFMFilterBase
{
  PClip padclip;
  VideoInfo srcvi;

  template <typename pixel_t, bool parity>
  void AnalyzeFrame(Frame& f0, Frame& f1, Frame& combe, PNeoEnv env)
  {
    const pixel_t* f0Y = f0.GetReadPtr<pixel_t>(PLANAR_Y);
    const pixel_t* f1Y = f1.GetReadPtr<pixel_t>(PLANAR_Y);
    const pixel_t* f0U = f0.GetReadPtr<pixel_t>(PLANAR_U);
    const pixel_t* f1U = f1.GetReadPtr<pixel_t>(PLANAR_U);
    const pixel_t* f0V = f0.GetReadPtr<pixel_t>(PLANAR_V);
    const pixel_t* f1V = f1.GetReadPtr<pixel_t>(PLANAR_V);
    uchar2* combeY = combe.GetWritePtr<uchar2>(PLANAR_Y);
    uchar2* combeU = combe.GetWritePtr<uchar2>(PLANAR_U);
    uchar2* combeV = combe.GetWritePtr<uchar2>(PLANAR_V);

    int pitchY = f0.GetPitch<pixel_t>(PLANAR_Y);
    int pitchUV = f0.GetPitch<pixel_t>(PLANAR_U);
    int fpitchY = combe.GetPitch<uchar2>(PLANAR_Y);
    int fpitchUV = combe.GetPitch<uchar2>(PLANAR_U);
    int width = combe.GetWidth<uchar2>(PLANAR_Y);
    int widthUV = combe.GetWidth<uchar2>(PLANAR_U);
    int height = combe.GetHeight(PLANAR_Y) >> 1;
    int heightUV = combe.GetHeight(PLANAR_U) >> 1;

    int shift = srcvi.BitsPerComponent() - 8 + 4;

    if (IS_CUDA) {
			cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());
      dim3 threads(DC_BLOCK_SIZE, DC_BLOCK_TH_W, DC_BLOCK_TH_H);
      dim3 blocks(nblocks(width, DC_BLOCK_TH_W), nblocks(height, DC_BLOCK_TH_H));
      dim3 blocksUV(nblocks(widthUV, DC_BLOCK_TH_W), nblocks(heightUV, DC_BLOCK_TH_H));
      kl_analyze_frame<pixel_t, parity> << <blocks, threads, 0, stream >> > (
        combeY, fpitchY, f0Y, f1Y, pitchY, width, height, shift);
      DEBUG_SYNC;
      kl_analyze_frame<pixel_t, parity> << <blocksUV, threads, 0, stream >> > (
        combeU, fpitchUV, f0U, f1U, pitchUV, widthUV, heightUV, shift);
      DEBUG_SYNC;
      kl_analyze_frame<pixel_t, parity> << <blocksUV, threads, 0, stream >> > (
        combeV, fpitchUV, f0V, f1V, pitchUV, widthUV, heightUV, shift);
      DEBUG_SYNC;
    }
    else {
      cpu_analyze_frame<pixel_t, parity>(
        combeY, fpitchY, f0Y, f1Y, pitchY, width, height, shift);
      cpu_analyze_frame<pixel_t, parity>(
        combeU, fpitchUV, f0U, f1U, pitchUV, widthUV, heightUV, shift);
      cpu_analyze_frame<pixel_t, parity>(
        combeV, fpitchUV, f0V, f1V, pitchUV, widthUV, heightUV, shift);
    }
  }

  template <typename pixel_t>
  PVideoFrame GetFrameT(int n, PNeoEnv env)
  {
    int parity = child->GetParity(n);
    Frame f0 = Frame(padclip->GetFrame(n, env), VPAD);
    Frame f1 = Frame(padclip->GetFrame(n + 1, env), VPAD);
    Frame dst = env->NewVideoFrame(vi);

    if (parity) {
      AnalyzeFrame<pixel_t, true>(f0, f1, dst, env);
    }
    else {
      AnalyzeFrame<pixel_t, false>(f0, f1, dst, env);
    }

    return dst.frame;
  }
public:
  KFMSuper(PClip clip, PClip pad, IScriptEnvironment* env)
    : KFMFilterBase(clip, env)
    , padclip(pad)
    , srcvi(vi)
  {
    if (vi.width & 7) env->ThrowError("[KFMSuper]: width must be multiple of 8");
    if (vi.height & 7) env->ThrowError("[KFMSuper]: height must be multiple of 8");

    vi.width = vi.width / DC_OVERLAP * 2;
    vi.height = vi.height / DC_OVERLAP * 2;
    vi.pixel_type = Get8BitType(vi);
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    int pixelSize = srcvi.ComponentSize();
    switch (pixelSize) {
    case 1:
      return GetFrameT<uint8_t>(n, env);
    case 2:
      return GetFrameT<uint16_t>(n, env);
    default:
      env->ThrowError("[KFMSuper] Unsupported pixel format");
      break;
    }

    return PVideoFrame();
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    if (cachehints == CACHE_GET_MTMODE) {
      return MT_NICE_FILTER;
    }
    return KFMFilterBase::SetCacheHints(cachehints, frame_range);
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    AVSValue clip = new KFMSuper(
      args[0].AsClip(),       // clip
      args[1].AsClip(),       // pad
      env
    );
    clip = env->Invoke("SeparateFields", AVSValue(&clip, 1));
    return env->Invoke("Align", AVSValue(&clip, 1));
  }
};

__global__ void kl_clean_super(uchar2* __restrict__ dst,
  const uchar2* __restrict__ prev, const uchar2* __restrict__ cur,
  int width, int height, int pitch, int thresh)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    uchar2 v = cur[x + y * pitch];
    if (prev[x + y * pitch].y <= thresh && v.y <= thresh) {
      v.x = 0;
    }
    dst[x + y * pitch] = v;
  }
}

void cpu_clean_super(uchar2* __restrict__ dst,
  const uchar2* __restrict__ prev, const uchar2* __restrict__ cur,
  int width, int height, int pitch, int thresh)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      uchar2 v = cur[x + y * pitch];
      if (prev[x + y * pitch].y <= thresh && v.y <= thresh) {
        v.x = 0;
      }
      dst[x + y * pitch] = v;
    }
  }
}

class KCleanSuper : public KFMFilterBase
{
  int thY;
  int thC;

  void CleanSuper(Frame& prev, Frame& cur, Frame& dst, PNeoEnv env)
  {
    const uchar2* prevY = prev.GetReadPtr<uchar2>(PLANAR_Y);
    const uchar2* curY = cur.GetReadPtr<uchar2>(PLANAR_Y);
    const uchar2* prevU = prev.GetReadPtr<uchar2>(PLANAR_U);
    const uchar2* curU = cur.GetReadPtr<uchar2>(PLANAR_U);
    const uchar2* prevV = prev.GetReadPtr<uchar2>(PLANAR_V);
    const uchar2* curV = cur.GetReadPtr<uchar2>(PLANAR_V);
    uchar2* dstY = dst.GetWritePtr<uchar2>(PLANAR_Y);
    uchar2* dstU = dst.GetWritePtr<uchar2>(PLANAR_U);
    uchar2* dstV = dst.GetWritePtr<uchar2>(PLANAR_V);

    int pitchY = cur.GetPitch<uchar2>(PLANAR_Y);
    int pitchUV = cur.GetPitch<uchar2>(PLANAR_U);
    int width = cur.GetWidth<uchar2>(PLANAR_Y);
    int widthUV = cur.GetWidth<uchar2>(PLANAR_U);
    int height = cur.GetHeight(PLANAR_Y);
    int heightUV = cur.GetHeight(PLANAR_U);

    int shift = srcvi.BitsPerComponent() - 8 + 4;

    if (IS_CUDA) {
			cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());
      dim3 threads(32, 16);
      dim3 blocks(nblocks(width, 32), nblocks(height, 16));
      dim3 blocksUV(nblocks(widthUV, 32), nblocks(heightUV, 16));
      kl_clean_super << <blocks, threads, 0, stream >> > (
        dstY, prevY, curY, width, height, pitchY, thY);
      DEBUG_SYNC;
      kl_clean_super << <blocksUV, threads, 0, stream >> > (
        dstU, prevU, curU, widthUV, heightUV, pitchUV, thC);
      DEBUG_SYNC;
      kl_clean_super << <blocksUV, threads, 0, stream >> > (
        dstV, prevV, curV, widthUV, heightUV, pitchUV, thC);
      DEBUG_SYNC;
    }
    else {
      cpu_clean_super(
        dstY, prevY, curY, width, height, pitchY, thY);
      cpu_clean_super(
        dstU, prevU, curU, widthUV, heightUV, pitchUV, thC);
      cpu_clean_super(
        dstV, prevV, curV, widthUV, heightUV, pitchUV, thC);
    }
  }

public:
  KCleanSuper(PClip clip, int thY, int thC, IScriptEnvironment* env)
    : KFMFilterBase(clip, env)
    , thY(thY)
    , thC(thC)
  { }

  PVideoFrame __stdcall GetFrame(int n60, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    Frame prev = child->GetFrame(n60 - 1, env);
    Frame cur = child->GetFrame(n60, env);
    Frame dst = env->NewVideoFrame(vi);
    CleanSuper(prev, cur, dst, env);

    return dst.frame;
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    if (cachehints == CACHE_GET_MTMODE) {
      return MT_NICE_FILTER;
    }
    return KFMFilterBase::SetCacheHints(cachehints, frame_range);
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KCleanSuper(
      args[0].AsClip(),       // clip
      args[1].AsInt(10),       // thY
      args[2].AsInt(8),       // thC
      env
    );
  }
};

struct KCycleAnalyzeParam {
  int threshM;
  int threshS;
  int threshLS;

  KCycleAnalyzeParam(int M, int S, int LS)
    : threshM(M)
    , threshS(S)
    , threshLS(LS)
  { }
};

__global__ void kl_count_cmflags(FMCount* __restrict__ dst,
  const uchar2* __restrict__ combe0, const uchar2* __restrict__ combe1, int pitch,
  int width, int height, int parity,
  int threshM, int threshS, int threshLS)
{
  int bx = threadIdx.x + blockIdx.x * FM_COUNT_TH_W;
  int by = threadIdx.y + blockIdx.y * FM_COUNT_TH_H;
  int tid = threadIdx.x + threadIdx.y * FM_COUNT_TH_W;

  for (int i = 0; i < 2; ++i) {
    int cnt[3] = { 0, 0, 0 };

    if (bx < width && by < height) {
      auto v = ((i == 0) ? combe0 : combe1)[bx + by * pitch];
      if (v.y >= threshM) cnt[0] = 1;
      if (v.x >= threshS) cnt[1] = 1;
      if (v.x >= threshLS) cnt[2] = 1;
    }

    __shared__ int sbuf[FM_COUNT_THREADS * 3];
    dev_reduceN<int, 3, FM_COUNT_THREADS, AddReducer<int>>(tid, cnt, sbuf);

    if (tid == 0) {
      atomicAdd(&dst[i ^ !parity].move, cnt[0]);
      atomicAdd(&dst[i ^ !parity].shima, cnt[1]);
      atomicAdd(&dst[i ^ !parity].lshima, cnt[2]);
    }
  }
}

void cpu_count_cmflags(FMCount* __restrict__ dst,
  const uchar2* __restrict__ combe0, const uchar2* __restrict__ combe1, int pitch,
  int width, int height, int parity,
  int threshM, int threshS, int threshLS)
{
  for (int by = 0; by < height; ++by) {
    for (int bx = 0; bx < width; ++bx) {
      for (int i = 0; i < 2; ++i) {
        auto v = ((i == 0) ? combe0 : combe1)[bx + by * pitch];
        if (v.y >= threshM) dst[i ^ !parity].move++;
        if (v.x >= threshS) dst[i ^ !parity].shima++;
        if (v.x >= threshLS) dst[i ^ !parity].lshima++;
      }
    }
  }
}

class KPreCycleAnalyze : public KFMFilterBase
{
  VideoInfo combevi;

  KCycleAnalyzeParam prmY;
  KCycleAnalyzeParam prmC;

public:
  KPreCycleAnalyze(PClip super, int threshMY, int threshSY, int threshMC, int threshSC, IScriptEnvironment* env)
    : KFMFilterBase(super, env)
    , combevi(vi)
    , prmY(threshMY, threshSY, threshSY * 3)
    , prmC(threshMC, threshSC, threshSC * 3)
  {
    int out_bytes = sizeof(FMCount) * 2;
    vi.pixel_type = VideoInfo::CS_BGR32;
    vi.width = 6;
    vi.height = nblocks(out_bytes, vi.width * 4);
    vi.MulDivFPS(1, 2);
    vi.num_frames /= 2;
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    int parity = child->GetParity(0);
    Frame combe0 = child->GetFrame(2 * n + 0, env);
    Frame combe1 = child->GetFrame(2 * n + 1, env);

    Frame dst = env->NewVideoFrame(vi);

    const uchar2 *combe0Y = combe0.GetReadPtr<uchar2>(PLANAR_Y);
    const uchar2 *combe0U = combe0.GetReadPtr<uchar2>(PLANAR_U);
    const uchar2 *combe0V = combe0.GetReadPtr<uchar2>(PLANAR_V);
    const uchar2 *combe1Y = combe1.GetReadPtr<uchar2>(PLANAR_Y);
    const uchar2 *combe1U = combe1.GetReadPtr<uchar2>(PLANAR_U);
    const uchar2 *combe1V = combe1.GetReadPtr<uchar2>(PLANAR_V);
    FMCount* fmcnt = dst.GetWritePtr<FMCount>();

    int pitch = combe0.GetPitch<uchar2>(PLANAR_Y);
    int pitchUV = combe0.GetPitch<uchar2>(PLANAR_U);
    int width = combe0.GetWidth<uchar2>(PLANAR_Y) - 1;
    int widthUV = combe0.GetWidth<uchar2>(PLANAR_U) - 1;
    int height = combe0.GetHeight(PLANAR_Y) - 1;
    int heightUV = combe0.GetHeight(PLANAR_U) - 1;

    combe0Y += pitch + 1;
    combe0U += pitchUV + 1;
    combe0V += pitchUV + 1;
    combe1Y += pitch + 1;
    combe1U += pitchUV + 1;
    combe1V += pitchUV + 1;

    if (IS_CUDA) {
			cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());
      dim3 threads(FM_COUNT_TH_W, FM_COUNT_TH_H);
      dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
      dim3 blocksUV(nblocks(widthUV, threads.x), nblocks(heightUV, threads.y));
      kl_init_fmcount << <1, 2, 0, stream >> > (fmcnt);
      DEBUG_SYNC;
      kl_count_cmflags << <blocks, threads, 0, stream >> > (
        fmcnt, combe0Y, combe1Y, pitch, width, height, parity, prmY.threshM, prmY.threshS, prmY.threshLS);
      DEBUG_SYNC;
      kl_count_cmflags << <blocksUV, threads, 0, stream >> > (
        fmcnt, combe0U, combe1U, pitchUV, widthUV, heightUV, parity, prmC.threshM, prmC.threshS, prmC.threshLS);
      DEBUG_SYNC;
      kl_count_cmflags << <blocksUV, threads, 0, stream >> > (
        fmcnt, combe0V, combe1V, pitchUV, widthUV, heightUV, parity, prmC.threshM, prmC.threshS, prmC.threshLS);
      DEBUG_SYNC;
    }
    else {
      memset(fmcnt, 0x00, sizeof(FMCount) * 2);
      cpu_count_cmflags(fmcnt, combe0Y, combe1Y, pitch, width, height, parity, prmY.threshM, prmY.threshS, prmY.threshLS);
      cpu_count_cmflags(fmcnt, combe0U, combe1U, pitchUV, widthUV, heightUV, parity, prmC.threshM, prmC.threshS, prmC.threshLS);
      cpu_count_cmflags(fmcnt, combe0V, combe1V, pitchUV, widthUV, heightUV, parity, prmC.threshM, prmC.threshS, prmC.threshLS);
    }

    return dst.frame;
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    if (cachehints == CACHE_GET_MTMODE) {
      return MT_NICE_FILTER;
    }
    return KFMFilterBase::SetCacheHints(cachehints, frame_range);
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KPreCycleAnalyze(
      args[0].AsClip(),       // super
      args[1].AsInt(20),      // threshMY
      args[2].AsInt(12),       // threshSY
      args[3].AsInt(24),      // threshMC
      args[4].AsInt(16),       // threshSC
      env
    );
  }
};

class KPreCycleAnalyzeShow : public KFMFilterBase
{
  PClip fmclip;

public:
  KPreCycleAnalyzeShow(PClip fmclip, PClip source, IScriptEnvironment* env)
    : KFMFilterBase(source, env)
    , fmclip(fmclip)
  { }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    Frame frame = fmclip->GetFrame(n, env);
    const FMCount *fmcount = frame.GetReadPtr<FMCount>();

    Frame dst = child->GetFrame(n, env);
    env->MakeWritable(&dst.frame);

    char buf[100];
    sprintf(buf, "Pre: [0](%d,%d,%d) [1](%d,%d,%d)",
      fmcount[0].move, fmcount[0].shima, fmcount[0].lshima,
      fmcount[1].move, fmcount[1].shima, fmcount[1].lshima);
    DrawText<uint8_t>(dst.frame, vi.BitsPerComponent(), 0, 0, buf, env);

    return dst.frame;
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    if (cachehints == CACHE_GET_MTMODE) {
      return MT_NICE_FILTER;
    }
    return KFMFilterBase::SetCacheHints(cachehints, frame_range);
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KPreCycleAnalyzeShow(
      args[0].AsClip(),       // fmclip
      args[1].AsClip(),       // source
      env
    );
  }
};

class KFMSuperShow : public KFMFilterBase
{
  VideoInfo combvi;

  KCycleAnalyzeParam prmY;
  KCycleAnalyzeParam prmC;

  void VisualizeFlags(Frame& dst, Frame& combe, PNeoEnv env)
  {
    // ���茋�ʂ�\��
    int black[] = { 0, 128, 128 };
    int blue[] = { 73, 230, 111 };
    int gray[] = { 140, 128, 128 };
    int purple[] = { 197, 160, 122 };

    const uchar2* __restrict__ combeY = combe.GetReadPtr<uchar2>(PLANAR_Y);
    const uchar2* __restrict__ combeU = combe.GetReadPtr<uchar2>(PLANAR_U);
    const uchar2* __restrict__ combeV = combe.GetReadPtr<uchar2>(PLANAR_V);
    uint8_t* __restrict__ dstY = dst.GetWritePtr<uint8_t>(PLANAR_Y);
    uint8_t* __restrict__ dstU = dst.GetWritePtr<uint8_t>(PLANAR_U);
    uint8_t* __restrict__ dstV = dst.GetWritePtr<uint8_t>(PLANAR_V);

    int combPitchY = combe.GetPitch<uchar2>(PLANAR_Y);
    int combPitchUV = combe.GetPitch<uchar2>(PLANAR_U);
    int dstPitchY = dst.GetPitch<uint8_t>(PLANAR_Y);
    int dstPitchUV = dst.GetPitch<uint8_t>(PLANAR_U);

    int width = combe.GetWidth<uchar2>(PLANAR_Y);
    int height = combe.GetHeight(PLANAR_Y);

    // ���ŏ��������Ă���
    for (int y = 0; y < vi.height; ++y) {
      for (int x = 0; x < vi.width; ++x) {
        int offY = x + y * dstPitchY;
        int offUV = (x >> logUVx) + (y >> logUVy) * dstPitchUV;
        dstY[offY] = black[0];
        dstU[offUV] = black[1];
        dstV[offUV] = black[2];
      }
    }

    // �F��t����
    for (int by = 0; by < height; ++by) {
      for (int bx = 0; bx < width; ++bx) {
        uchar2 cY = combeY[bx + by * combPitchY];
        uchar2 cU = combeU[(bx >> 1) + (by >> 1) * combPitchUV];
        uchar2 cV = combeV[(bx >> 1) + (by >> 1) * combPitchUV];

        int xbase = bx * DC_OVERLAP;
        int ybase = by * DC_OVERLAP;

        int shimaY = cY.x;
        int moveY = cY.y;
        int shimaUV = std::max(cU.x, cV.x);
        int moveUV = std::max(cU.y, cV.y);

        bool isShima = (shimaY > prmY.threshS || shimaUV > prmC.threshS);
        bool isMove = (moveY > prmY.threshM || moveUV > prmC.threshM);

        int* color = nullptr;
        if (isMove && isShima) {
          color = purple;
        }
        else if (isMove) {
          color = blue;
        }
        else if (isShima) {
          color = gray;
        }

        if (color) {
          for (int tx = 0; tx < DC_OVERLAP; ++tx) {
            for (int ty = 0; ty < DC_OVERLAP; ++ty) {
              int x = xbase + tx;
              int y = ybase + ty;
              int offY = x + y * dstPitchY;
              int offUV = (x >> logUVx) + (y >> logUVy) * dstPitchUV;
              dstY[offY] = color[0];
              dstU[offUV] = color[1];
              dstV[offUV] = color[2];
            }
          }
        }
      }
    }
  }

public:
  KFMSuperShow(PClip combe, int threshMY, int threshSY, int threshMC, int threshSC, IScriptEnvironment* env)
    : KFMFilterBase(combe, env)
    , combvi(vi)
    , prmY(threshMY, threshSY, threshSY * 3)
    , prmC(threshMC, threshSC, threshSC * 3)
  {
    vi.width *= DC_OVERLAP / 2;
    vi.height *= DC_OVERLAP;
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    Frame combe = child->GetFrame(n, env);
    Frame combetmp = env->NewVideoFrame(combvi);
    Frame dst = env->NewVideoFrame(vi);

    env->MakeWritable(&combe.frame);
    ExtendBlocks<uchar4>(combe, combetmp, true, env);
    VisualizeFlags(dst, combe, env);

    return dst.frame;
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    if (cachehints == CACHE_GET_MTMODE) {
      return MT_NICE_FILTER;
    }
    return 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KFMSuperShow(
      args[0].AsClip(),       // combe
      args[1].AsInt(20),      // threshMY
      args[2].AsInt(12),       // threshSY
      args[3].AsInt(24),      // threshMC
      args[4].AsInt(16),       // threshSC
      env
    );
  }
};


class KTelecine : public KFMFilterBase
{
  PClip superclip;
  PClip fmclip;
  bool show;

  PulldownPatterns patterns;

  template <typename pixel_t>
  void CopyField(int top, Frame* const * frames, int fnum, Frame& dst, PNeoEnv env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;
		cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());

    Frame& frame0 = *frames[0];
    const vpixel_t* src0Y = frame0.GetReadPtr<vpixel_t>(PLANAR_Y);
    const vpixel_t* src0U = frame0.GetReadPtr<vpixel_t>(PLANAR_U);
    const vpixel_t* src0V = frame0.GetReadPtr<vpixel_t>(PLANAR_V);
    vpixel_t* dstY = dst.GetWritePtr<vpixel_t>(PLANAR_Y);
    vpixel_t* dstU = dst.GetWritePtr<vpixel_t>(PLANAR_U);
    vpixel_t* dstV = dst.GetWritePtr<vpixel_t>(PLANAR_V);

    int pitchY = frame0.GetPitch<vpixel_t>(PLANAR_Y);
    int pitchUV = frame0.GetPitch<vpixel_t>(PLANAR_U);
    int width4 = vi.width >> 2;
    int width4UV = width4 >> logUVx;
    int heightUV = vi.height >> logUVy;

    if (!top) {
      src0Y += pitchY;
      src0U += pitchUV;
      src0V += pitchUV;
      dstY += pitchY;
      dstU += pitchUV;
      dstV += pitchUV;
    }

    if (fnum == 1) {
      if (IS_CUDA) {
        dim3 threads(32, 16);
        dim3 blocks(nblocks(width4, threads.x), nblocks(srcvi.height / 2, threads.y));
        dim3 blocksUV(nblocks(width4UV, threads.x), nblocks(heightUV / 2, threads.y));
        kl_copy << <blocks, threads, 0, stream >> > (dstY, src0Y, width4, vi.height / 2, pitchY * 2);
        DEBUG_SYNC;
        kl_copy << <blocksUV, threads, 0, stream >> > (dstU, src0U, width4UV, heightUV / 2, pitchUV * 2);
        DEBUG_SYNC;
        kl_copy << <blocksUV, threads, 0, stream >> > (dstV, src0V, width4UV, heightUV / 2, pitchUV * 2);
        DEBUG_SYNC;
      }
      else {
        cpu_copy(dstY, src0Y, width4, vi.height / 2, pitchY * 2);
        cpu_copy(dstU, src0U, width4UV, heightUV / 2, pitchUV * 2);
        cpu_copy(dstV, src0V, width4UV, heightUV / 2, pitchUV * 2);
      }
    }
    else { // fnum == 2
      Frame& frame1 = *frames[1];
      const vpixel_t* src1Y = frame1.GetReadPtr<vpixel_t>(PLANAR_Y);
      const vpixel_t* src1U = frame1.GetReadPtr<vpixel_t>(PLANAR_U);
      const vpixel_t* src1V = frame1.GetReadPtr<vpixel_t>(PLANAR_V);

      if (!top) {
        src1Y += pitchY;
        src1U += pitchUV;
        src1V += pitchUV;
      }

      if (IS_CUDA) {
        dim3 threads(32, 16);
        dim3 blocks(nblocks(width4, threads.x), nblocks(srcvi.height / 2, threads.y));
        dim3 blocksUV(nblocks(width4UV, threads.x), nblocks(heightUV / 2, threads.y));
        kl_average << <blocks, threads, 0, stream >> > (dstY, src0Y, src1Y, width4, vi.height / 2, pitchY * 2);
        DEBUG_SYNC;
        kl_average << <blocksUV, threads, 0, stream >> > (dstU, src0U, src1U, width4UV, heightUV / 2, pitchUV * 2);
        DEBUG_SYNC;
        kl_average << <blocksUV, threads, 0, stream >> > (dstV, src0V, src1V, width4UV, heightUV / 2, pitchUV * 2);
        DEBUG_SYNC;
      }
      else {
        cpu_average(dstY, src0Y, src1Y, width4, vi.height / 2, pitchY * 2);
        cpu_average(dstU, src0U, src1U, width4UV, heightUV / 2, pitchUV * 2);
        cpu_average(dstV, src0V, src1V, width4UV, heightUV / 2, pitchUV * 2);
      }
    }

  }

  template <typename pixel_t>
  Frame CreateWeaveFrame(PClip clip, int n, int fstart, int fnum, int parity, PNeoEnv env)
  {
    // fstart��0or1�ɂ���
    if (fstart < 0 || fstart >= 2) {
      n += fstart / 2;
      fstart &= 1;
    }

    assert(fstart == 0 || fstart == 1);
    assert(fnum == 2 || fnum == 3 || fnum == 4);

    Frame frames[3] = {
      clip->GetFrame(n, env),
      (fstart + fnum > 2) ? clip->GetFrame(n + 1, env) : PVideoFrame(),
      (fstart + fnum > 4) ? clip->GetFrame(n + 2, env) : PVideoFrame()
    };

    if (fstart + fnum == 2) {
      // �t���[�����̂܂�
      return frames[0].frame;
    }
    else {
      Frame dst = env->NewVideoFrame(vi);

      int numFields[2] = { 0 };
      Frame* fields[2][2] = { { 0 } };

      for (int i = 0; i < fnum; ++i) {
        int field_idx = fstart + i;
        int frame_idx = field_idx / 2;
        int isSecond = field_idx & 1;
        fields[isSecond][numFields[isSecond]++] = &frames[frame_idx];
      }

      CopyField<pixel_t>(parity, fields[0], numFields[0], dst, env);
      CopyField<pixel_t>(!parity, fields[1], numFields[1], dst, env);

      return dst;
    }
  }

  template <typename pixel_t>
  void DrawInfo(Frame& dst, int pattern, float cost, int fnum, PNeoEnv env) {
    int number;
    const char* patternName = patterns.PatternToString(pattern, number);
    char buf[100]; sprintf(buf, "KFM: %s-%d (%d) (%.1f) - %d", patternName, number, pattern, cost, fnum);
    DrawText<pixel_t>(dst.frame, vi.BitsPerComponent(), 0, 0, buf, env);
  }

  template <typename pixel_t>
  PVideoFrame GetFrameT(int n, PNeoEnv env)
  {
    int cycleIndex = n / 4;
    int parity = child->GetParity(cycleIndex * 5);
    PDevice cpuDevice = env->GetDevice(DEV_TYPE_CPU, 0);
    KFMResult fm = *(Frame(env->GetFrame(fmclip, cycleIndex, cpuDevice)).GetReadPtr<KFMResult>());
    Frame24Info frameInfo = patterns.GetFrame24(fm.pattern, n);

    int fstart = frameInfo.cycleIndex * 10 + frameInfo.fieldStartIndex;
    Frame out = CreateWeaveFrame<pixel_t>(child, 0, fstart, frameInfo.numFields, parity, env);

    if (show) {
      DrawInfo<pixel_t>(out, fm.pattern, fm.cost, frameInfo.numFields, env);
    }

    return out.frame;
  }

public:
  KTelecine(PClip child, PClip fmclip, bool show, IScriptEnvironment* env)
    : KFMFilterBase(child, env)
    , fmclip(fmclip)
    , show(show)
  {
    // �`�F�b�N
    CycleAnalyzeInfo::GetParam(fmclip->GetVideoInfo(), env);

    // �t���[�����[�g
    vi.MulDivFPS(4, 5);
    vi.num_frames = (vi.num_frames / 5 * 4) + (vi.num_frames % 5);
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    int pixelSize = vi.ComponentSize();
    switch (pixelSize) {
    case 1:
      return GetFrameT<uint8_t>(n, env);
    case 2:
      return GetFrameT<uint16_t>(n, env);
    default:
      env->ThrowError("[KTelecine] Unsupported pixel format");
    }

    return PVideoFrame();
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    if (cachehints == CACHE_GET_MTMODE) {
      return MT_NICE_FILTER;
    }
    return KFMFilterBase::SetCacheHints(cachehints, frame_range);
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KTelecine(
      args[0].AsClip(),       // source
      args[1].AsClip(),       // fmclip
      args[2].AsBool(false),  // show
      env
    );
  }
};

class KTelecineSuper : public KFMFilterBase
{
  PClip fmclip;

  PulldownPatterns patterns;

  Frame CreateWeaveFrame(PClip clip, int fstart, int fnum, PNeoEnv env)
  {
		cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());

    Frame first = child->GetFrame(fstart, env);
    if (fnum == 2) {
      return first;
    }

    Frame dst = env->NewVideoFrame(vi);

    const uint8_t* src0Y = first.GetReadPtr<uint8_t>(PLANAR_Y);
    const uint8_t* src0U = first.GetReadPtr<uint8_t>(PLANAR_U);
    const uint8_t* src0V = first.GetReadPtr<uint8_t>(PLANAR_V);
    uint8_t* dstY = dst.GetWritePtr<uint8_t>(PLANAR_Y);
    uint8_t* dstU = dst.GetWritePtr<uint8_t>(PLANAR_U);
    uint8_t* dstV = dst.GetWritePtr<uint8_t>(PLANAR_V);

    int pitchY = first.GetPitch<uint8_t>(PLANAR_Y);
    int pitchUV = first.GetPitch<uint8_t>(PLANAR_U);
    int width = first.GetWidth<uint8_t>(PLANAR_Y);
    int widthUV = first.GetWidth<uint8_t>(PLANAR_U);
    int height = first.GetHeight(PLANAR_Y);
    int heightUV = first.GetHeight(PLANAR_U);

    for (int i = 1; i < fnum - 1; ++i) {
      Frame frame = child->GetFrame(fstart + i, env);
      const uint8_t* src1Y = frame.GetReadPtr<uint8_t>(PLANAR_Y);
      const uint8_t* src1U = frame.GetReadPtr<uint8_t>(PLANAR_U);
      const uint8_t* src1V = frame.GetReadPtr<uint8_t>(PLANAR_V);

      if (IS_CUDA) {
        dim3 threads(32, 16);
        dim3 blocks(nblocks(width, threads.x), nblocks(srcvi.height, threads.y));
        dim3 blocksUV(nblocks(widthUV, threads.x), nblocks(heightUV, threads.y));
        kl_max << <blocks, threads, 0, stream >> > (dstY, src0Y, src1Y, width, height, pitchY);
        DEBUG_SYNC;
        kl_max << <blocksUV, threads, 0, stream >> > (dstU, src0U, src1U, widthUV, heightUV, pitchUV);
        DEBUG_SYNC;
        kl_max << <blocksUV, threads, 0, stream >> > (dstV, src0V, src1V, widthUV, heightUV, pitchUV);
        DEBUG_SYNC;
      }
      else {
        cpu_max(dstY, src0Y, src1Y, width, height, pitchY);
        cpu_max(dstU, src0U, src1U, widthUV, heightUV, pitchUV);
        cpu_max(dstV, src0V, src1V, widthUV, heightUV, pitchUV);
      }

      src0Y = dstY;
      src0U = dstU;
      src0V = dstV;
    }

    return dst;
  }

public:
  KTelecineSuper(PClip child, PClip fmclip, IScriptEnvironment* env)
    : KFMFilterBase(child, env)
    , fmclip(fmclip)
  {
    // �`�F�b�N
    CycleAnalyzeInfo::GetParam(fmclip->GetVideoInfo(), env);

    // �t���[�����[�g
    vi.MulDivFPS(2, 5);
    vi.num_frames >>= 1;
    vi.num_frames = (vi.num_frames / 5 * 4) + (vi.num_frames % 5);
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    int cycleIndex = n / 4;
    int parity = child->GetParity(cycleIndex * 10);
    PDevice cpuDevice = env->GetDevice(DEV_TYPE_CPU, 0);
    KFMResult fm = *(Frame(env->GetFrame(fmclip, cycleIndex, cpuDevice)).GetReadPtr<KFMResult>());
    Frame24Info frameInfo = patterns.GetFrame24(fm.pattern, n);

    int fstart = frameInfo.cycleIndex * 10 + frameInfo.fieldStartIndex;
    Frame out = CreateWeaveFrame(child, fstart, frameInfo.numFields, env);

    return out.frame;
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    if (cachehints == CACHE_GET_MTMODE) {
      return MT_NICE_FILTER;
    }
    return KFMFilterBase::SetCacheHints(cachehints, frame_range);
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KTelecineSuper(
      args[0].AsClip(),       // super
      args[1].AsClip(),       // fmclip
      env
    );
  }
};


template <typename vpixel_t>
void cpu_copy_first(uint8_t* __restrict__ dst, int dpitch,
  const vpixel_t* __restrict__ src, int width, int height, int spitch)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      dst[x + y * dpitch] = src[x + y * spitch].x;
    }
  }
}

template <typename vpixel_t>
__global__ void kl_copy_first(uint8_t* __restrict__ dst, int dpitch,
  const vpixel_t* __restrict__ src, int width, int height, int spitch)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    dst[x + y * dpitch] = src[x + y * spitch].x;
  }
}

template <typename pixel_t>
void cpu_combe_to_flag(pixel_t* __restrict__ flag, int nBlkX, int nBlkY, int fpitch, const pixel_t* __restrict__ combe, int cpitch)
{
  for (int y = 0; y < nBlkY; ++y) {
    for (int x = 0; x < nBlkX; ++x) {
      flag[x + y * fpitch] =
        (combe[(2 * x + 0) + (2 * y + 0) * cpitch] +
          combe[(2 * x + 1) + (2 * y + 0) * cpitch] +
          combe[(2 * x + 0) + (2 * y + 1) * cpitch] +
          combe[(2 * x + 1) + (2 * y + 1) * cpitch] + 2) >> 2;
    }
  }
}

template <typename pixel_t>
__global__ void kl_combe_to_flag(pixel_t* __restrict__ flag, int nBlkX, int nBlkY, int fpitch, const pixel_t* __restrict__ combe, int cpitch)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < nBlkX && y < nBlkY) {
    flag[x + y * fpitch] =
      (combe[(2 * x + 0) + (2 * y + 0) * cpitch] +
        combe[(2 * x + 1) + (2 * y + 0) * cpitch] +
        combe[(2 * x + 0) + (2 * y + 1) * cpitch] +
        combe[(2 * x + 1) + (2 * y + 1) * cpitch] + 2) >> 2;
  }
}

template <typename pixel_t>
void cpu_sum_box3x3(pixel_t* __restrict__ dst, pixel_t* __restrict__ src, int width, int height, int pitch, int maxv)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      auto sumv = (src[(x - 1) + (y - 1)*pitch] + src[(x + 0) + (y - 1)*pitch] + src[(x + 1) + (y - 1)*pitch] +
        src[(x - 1) + (y + 0)*pitch] + src[(x + 0) + (y + 0)*pitch] + src[(x + 1) + (y + 0)*pitch] +
        src[(x - 1) + (y + 1)*pitch] + src[(x + 0) + (y + 1)*pitch] + src[(x + 1) + (y + 1)*pitch]);
      dst[x + y * pitch] = min(sumv >> 2, maxv); // �K����1/4����
    }
  }
}

template <typename pixel_t>
__global__ void kl_sum_box3x3(pixel_t* __restrict__ dst, pixel_t* __restrict__ src, int width, int height, int pitch, int maxv)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    auto sumv = (src[(x - 1) + (y - 1)*pitch] + src[(x + 0) + (y - 1)*pitch] + src[(x + 1) + (y - 1)*pitch] +
      src[(x - 1) + (y + 0)*pitch] + src[(x + 0) + (y + 0)*pitch] + src[(x + 1) + (y + 0)*pitch] +
      src[(x - 1) + (y + 1)*pitch] + src[(x + 0) + (y + 1)*pitch] + src[(x + 1) + (y + 1)*pitch]);
    dst[x + y * pitch] = min(sumv >> 2, maxv); // �K����1/4����
  }
}

void cpu_binary_flag(
  uint8_t* __restrict__ dst, const uint8_t* __restrict__ srcY, const uint8_t* __restrict__ srcC,
  int nBlkX, int nBlkY, int pitch, int thY, int thC)
{
  for (int y = 0; y < nBlkY; ++y) {
    for (int x = 0; x < nBlkX; ++x) {
      auto Y = srcY[x + y * pitch];
      auto C = srcC[x + y * pitch];
      dst[x + y * pitch] = ((Y >= thY || C >= thC) ? 128 : 0);
    }
  }
}

__global__ void kl_binary_flag(
  uint8_t* __restrict__ dst, const uint8_t* __restrict__ srcY, const uint8_t* __restrict__ srcC,
  int nBlkX, int nBlkY, int pitch, int thY, int thC)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < nBlkX && y < nBlkY) {
    auto Y = srcY[x + y * pitch];
    auto C = srcC[x + y * pitch];
    dst[x + y * pitch] = ((Y >= thY || C >= thC) ? 128 : 0);
  }
}

template <int SCALE, int SHIFT>
void cpu_bilinear_v(
  uint8_t* __restrict__  dst, int width, int height, int dpitch,
  const uint8_t* __restrict__  src, int spitch, PNeoEnv env)
{
  enum { HALF = SCALE / 2 };
  for (int y = 0; y < height; ++y) {
    int y0 = ((y - HALF) >> SHIFT);
    int c0 = ((y0 + 1) << SHIFT) - (y - HALF);
    int c1 = SCALE - c0;
    for (int x = 0; x < width; ++x) {
      auto s0 = src[x + (y0 + 0) * spitch];
      auto s1 = src[x + (y0 + 1) * spitch];
      dst[x + y * dpitch] = (s0 * c0 + s1 * c1 + HALF) >> SHIFT;
    }
  }
}

template <int SCALE, int SHIFT>
__global__ void kl_bilinear_v(
  uint8_t* __restrict__ dst, int width, int height, int dpitch,
  const uint8_t* __restrict__ src, int spitch)
{
  enum { HALF = SCALE / 2 };
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    int y0 = ((y - HALF) >> SHIFT);
    int c0 = ((y0 + 1) << SHIFT) - (y - HALF);
    int c1 = SCALE - c0;
    auto s0 = src[x + (y0 + 0) * spitch];
    auto s1 = src[x + (y0 + 1) * spitch];
    dst[x + y * dpitch] = (s0 * c0 + s1 * c1 + HALF) >> SHIFT;
  }
}

template <int SCALE, int SHIFT>
void launch_bilinear_v(
  uint8_t* __restrict__ dst, int width, int height, int dpitch,
  const uint8_t* __restrict__ src, int spitch, PNeoEnv env)
{
	cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());
  dim3 threads(32, 8);
  dim3 h_blocks(nblocks(width, threads.x), nblocks(height, threads.y));
  kl_bilinear_v<SCALE, SHIFT> << <h_blocks, threads, 0, stream >> > (dst, width, height, dpitch, src, spitch);
  DEBUG_SYNC;
}

template <int SCALE, int SHIFT>
void cpu_bilinear_h(
  uint8_t* __restrict__  dst, int width, int height, int dpitch,
  const uint8_t* __restrict__  src, int spitch, PNeoEnv env)
{
  enum { HALF = SCALE / 2 };
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int x0 = ((x - HALF) >> SHIFT);
      int c0 = ((x0 + 1) << SHIFT) - (x - HALF);
      int c1 = SCALE - c0;
      auto s0 = src[(x0 + 0) + y * spitch];
      auto s1 = src[(x0 + 1) + y * spitch];
      dst[x + y * dpitch] = (s0 * c0 + s1 * c1 + HALF) >> SHIFT;
    }
  }
}

template <int SCALE, int SHIFT>
__global__ void kl_bilinear_h(
  uint8_t* __restrict__ dst, int width, int height, int dpitch,
  const uint8_t* __restrict__ src, int spitch)
{
  enum { HALF = SCALE / 2 };
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    int x0 = ((x - HALF) >> SHIFT);
    int c0 = ((x0 + 1) << SHIFT) - (x - HALF);
    int c1 = SCALE - c0;
    auto s0 = src[(x0 + 0) + y * spitch];
    auto s1 = src[(x0 + 1) + y * spitch];
    dst[x + y * dpitch] = (s0 * c0 + s1 * c1 + HALF) >> SHIFT;
  }
}

template <int SCALE, int SHIFT>
void launch_bilinear_h(
  uint8_t* __restrict__ dst, int width, int height, int dpitch,
  const uint8_t* __restrict__ src, int spitch, PNeoEnv env)
{
	cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());
  dim3 threads(32, 8);
  dim3 h_blocks(nblocks(width, threads.x), nblocks(height, threads.y));
  kl_bilinear_h<SCALE, SHIFT> << <h_blocks, threads, 0, stream >> > (dst, width, height, dpitch, src, spitch);
  DEBUG_SYNC;
}

template <typename pixel_t>
void cpu_temporal_soften(pixel_t* __restrict__ dst,
  const pixel_t* __restrict__ src0, const pixel_t* __restrict__ src1, const pixel_t* __restrict__ src2,
  int width, int height, int pitch)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      auto t = to_float(src0[x + y * pitch]) + to_float(src1[x + y * pitch]) + to_float(src2[x + y * pitch]);
      dst[x + y * pitch] = VHelper<pixel_t>::cast_to(to_int(t * (1.0f / 3.0f)));
    }
  }
}

template <typename pixel_t>
__global__ void kl_temporal_soften(pixel_t* __restrict__ dst,
  const pixel_t* __restrict__ src0, const pixel_t* __restrict__ src1, const pixel_t* __restrict__ src2,
  int width, int height, int pitch)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    auto t = to_float(src0[x + y * pitch]) + to_float(src1[x + y * pitch]) + to_float(src2[x + y * pitch]);
    dst[x + y * pitch] = VHelper<pixel_t>::cast_to(to_int(t * (1.0f / 3.0f)));
  }
}

class KSwitchFlag : public KFMFilterBase
{
  VideoInfo srcvi;
  VideoInfo combvi;

  float thY;
  float thC;

  Frame TemporalSoften(const Frame& src0, const Frame& src1, const Frame& src2, PNeoEnv env)
  {
    Frame dst = env->NewVideoFrame(srcvi);

    const uchar4* src0Y = src0.GetReadPtr<uchar4>(PLANAR_Y);
    const uchar4* src0U = src0.GetReadPtr<uchar4>(PLANAR_U);
    const uchar4* src0V = src0.GetReadPtr<uchar4>(PLANAR_V);
    const uchar4* src1Y = src1.GetReadPtr<uchar4>(PLANAR_Y);
    const uchar4* src1U = src1.GetReadPtr<uchar4>(PLANAR_U);
    const uchar4* src1V = src1.GetReadPtr<uchar4>(PLANAR_V);
    const uchar4* src2Y = src2.GetReadPtr<uchar4>(PLANAR_Y);
    const uchar4* src2U = src2.GetReadPtr<uchar4>(PLANAR_U);
    const uchar4* src2V = src2.GetReadPtr<uchar4>(PLANAR_V);
    uchar4* dstY = dst.GetWritePtr<uchar4>(PLANAR_Y);
    uchar4* dstU = dst.GetWritePtr<uchar4>(PLANAR_U);
    uchar4* dstV = dst.GetWritePtr<uchar4>(PLANAR_V);

    int pitchY = src0.GetPitch<uchar4>(PLANAR_Y);
    int pitchUV = src0.GetPitch<uchar4>(PLANAR_U);
    int width = src0.GetWidth<uchar4>(PLANAR_Y);
    int widthUV = src0.GetWidth<uchar4>(PLANAR_U);
    int height = src0.GetHeight(PLANAR_Y);
    int heightUV = src0.GetHeight(PLANAR_U);

    if (IS_CUDA) {
			cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());
      dim3 threads(32, 16);
      dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
      dim3 blocksUV(nblocks(widthUV, threads.x), nblocks(heightUV, threads.y));
      kl_temporal_soften << <blocks, threads, 0, stream >> > (dstY, src0Y, src1Y, src2Y, width, height, pitchY);
      DEBUG_SYNC;
      kl_temporal_soften << <blocksUV, threads, 0, stream >> > (dstU, src0U, src1U, src2U, widthUV, heightUV, pitchUV);
      DEBUG_SYNC;
      kl_temporal_soften << <blocksUV, threads, 0, stream >> > (dstV, src0V, src1V, src2V, widthUV, heightUV, pitchUV);
      DEBUG_SYNC;
    }
    else {
      cpu_temporal_soften(dstY, src0Y, src1Y, src2Y, width, height, pitchY);
      cpu_temporal_soften(dstU, src0U, src1U, src2U, widthUV, heightUV, pitchUV);
      cpu_temporal_soften(dstV, src0V, src1V, src2V, widthUV, heightUV, pitchUV);
    }

    return dst;
  }

  Frame MakeSwitchFlag(Frame& src, PNeoEnv env)
  {
		cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());

    Frame combe = env->NewVideoFrame(combvi);
    Frame combetmp = env->NewVideoFrame(combvi);
    Frame flagY = NewSwitchFlagFrame(vi, env);
    Frame flagC = NewSwitchFlagFrame(vi, env);
    Frame flagtmp = NewSwitchFlagFrame(vi, env);

    const uchar2* srcY = src.GetReadPtr<uchar2>(PLANAR_Y);
    const uchar2* srcU = src.GetReadPtr<uchar2>(PLANAR_U);
    const uchar2* srcV = src.GetReadPtr<uchar2>(PLANAR_V);
    uint8_t* combeY = combe.GetWritePtr<uint8_t>(PLANAR_Y);
    uint8_t* combeU = combe.GetWritePtr<uint8_t>(PLANAR_U);
    uint8_t* combeV = combe.GetWritePtr<uint8_t>(PLANAR_V);

    int spitchY = src.GetPitch<uchar2>(PLANAR_Y);
    int spitchUV = src.GetPitch<uchar2>(PLANAR_U);
    int pitchY = combe.GetPitch<uint8_t>(PLANAR_Y);
    int pitchUV = combe.GetPitch<uint8_t>(PLANAR_U);
    int width = combe.GetWidth<uint8_t>(PLANAR_Y);
    int widthUV = combe.GetWidth<uint8_t>(PLANAR_U);
    int height = combe.GetHeight(PLANAR_Y);
    int heightUV = combe.GetHeight(PLANAR_U);

    // �����͂���Ȃ��̂ŎȂ������o
    if (IS_CUDA) {
      dim3 threads(32, 16);
      dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
      dim3 blocksUV(nblocks(widthUV, threads.x), nblocks(heightUV, threads.y));
      kl_copy_first << <blocks, threads, 0, stream >> > (combeY, pitchY, srcY, width, height, spitchY);
      DEBUG_SYNC;
      kl_copy_first << <blocksUV, threads, 0, stream >> > (combeU, pitchUV, srcU, widthUV, heightUV, spitchUV);
      DEBUG_SYNC;
      kl_copy_first << <blocksUV, threads, 0, stream >> > (combeV, pitchUV, srcV, widthUV, heightUV, spitchUV);
      DEBUG_SYNC;
    }
    else {
      cpu_copy_first(combeY, pitchY, srcY, width, height, spitchY);
      cpu_copy_first(combeU, pitchUV, srcU, widthUV, heightUV, spitchUV);
      cpu_copy_first(combeV, pitchUV, srcV, widthUV, heightUV, spitchUV);
    }

    // UV��U�Ƀ}�[�W
    if (IS_CUDA) {
      dim3 threads(32, 16);
      dim3 blocksUV(nblocks(widthUV, threads.x), nblocks(heightUV, threads.y));
      kl_max << <blocksUV, threads, 0, stream >> > (combeU, combeU, combeV, widthUV, heightUV, pitchUV);
      DEBUG_SYNC;
    }
    else {
      cpu_max(combeU, combeU, combeV, widthUV, heightUV, pitchUV);
    }

    // ���̂܂�4�_���ςŏk������ƈʒu�������̂�
    // �k������v���[���͍����extend����
    ExtendBlocks<uint8_t>(combe, combetmp, width == widthUV, env);

    uint8_t* flagpY = flagY.GetWritePtr<uint8_t>();
    uint8_t* flagpC = flagC.GetWritePtr<uint8_t>();
    uint8_t* flagtmpp = flagtmp.GetWritePtr<uint8_t>();

    int fpitch = flagY.GetPitch<uint8_t>();
    int fwidth = flagY.GetWidth<uint8_t>();
    int fheight = flagY.GetHeight();

    uint8_t* flagptrs[2] = { flagpY, flagpC };
    uint8_t* combeptrs[2] = { combeY, combeU };

    for (int i = 0; i < 2; ++i) {
      if (i == 0 || width == widthUV) {
        // 4�_���ςŏk�� src��dst����1��Ə�1�s�̓X�L�b�v
        if (IS_CUDA) {
          dim3 threads(32, 8);
          dim3 blocks(nblocks(fwidth, threads.x), nblocks(fheight, threads.y));
          kl_combe_to_flag << <blocks, threads, 0, stream >> > (
            flagptrs[i] + fpitch + 1, fwidth - 1, fheight - 1, fpitch, combeptrs[i] + pitchY + 1, pitchY);
          DEBUG_SYNC;
        }
        else {
          cpu_combe_to_flag(
            flagptrs[i] + fpitch + 1, fwidth - 1, fheight - 1, fpitch, combeptrs[i] + pitchY + 1, pitchY);
        }
      }
      else {
        // �k�����Ȃ��ꍇ�̓R�s�[
        assert(fpitch == pitchUV);
        assert(fwidth == widthUV);
        // ��1��Ə�1�s�̓X�L�b�v
        Copy(flagpC + fpitch + 1, fpitch, combeU + pitchUV + 1, pitchUV, fwidth - 1, fheight - 1, env);
      }
    }

    // 3x3box�t�B���^�łڂ���
    if (IS_CUDA) {
      dim3 threads(32, 8);
      dim3 blocks(nblocks(fwidth, threads.x), nblocks(fheight, threads.y));
      for (int i = 0; i < 2; ++i) {
        kl_sum_box3x3 << <blocks, threads, 0, stream >> > (
          flagtmpp, flagptrs[i], fwidth, fheight, fpitch, 255);
        DEBUG_SYNC;
        kl_sum_box3x3 << <blocks, threads, 0, stream >> > (
          flagptrs[i], flagtmpp, fwidth, fheight, fpitch, 255);
        DEBUG_SYNC;
      }
    }
    else {
      for (int i = 0; i < 2; ++i) {
        cpu_sum_box3x3(flagtmpp, flagptrs[i], fwidth, fheight, fpitch, 255);
        cpu_sum_box3x3(flagptrs[i], flagtmpp, fwidth, fheight, fpitch, 255);
      }
    }

    // 臒l��2�l��
    if (IS_CUDA) {
      dim3 threads(32, 8);
      dim3 binary_blocks(nblocks(fwidth, threads.x), nblocks(fheight, threads.y));
      kl_binary_flag << <binary_blocks, threads, 0, stream >> > (
        flagpY, flagpY, flagpC, fwidth, fheight, fpitch, (int)thY, (int)thC);
      DEBUG_SYNC;
    }
    else {
      cpu_binary_flag(flagpY, flagpY, flagpC, fwidth, fheight, fpitch, (int)thY, (int)thC);
    }

    // �����extend
    ExtendBlocks<uint8_t>(flagY, flagtmp, false, env);

    return flagY;
  }

public:
  KSwitchFlag(PClip combe, float thY, float thC, IScriptEnvironment* env)
    : KFMFilterBase(combe, env)
    , srcvi(combe->GetVideoInfo())
    , combvi(combe->GetVideoInfo())
    , thY(thY)
    , thC(thC)
  {
    if (combvi.GetPlaneHeightSubsampling(PLANAR_U) != combvi.GetPlaneWidthSubsampling(PLANAR_U)) {
      env->ThrowError("[KSwitchFlag] Only 444 or 420 is supported");
    }

    combvi.width /= 2;

    vi.width = nblocks(combvi.width, 2) + COMBE_FLAG_PAD_W * 2;
    vi.height = nblocks(combvi.height, 2) + COMBE_FLAG_PAD_H * 2;
    vi.pixel_type = VideoInfo::CS_Y8;
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    Frame src = TemporalSoften(
      child->GetFrame(n - 1, env),
      child->GetFrame(n + 0, env),
      child->GetFrame(n + 1, env), env);
    Frame flag = MakeSwitchFlag(src, env);

    return flag.frame;
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    if (cachehints == CACHE_GET_MTMODE) {
      return MT_NICE_FILTER;
    }
    return KFMFilterBase::SetCacheHints(cachehints, frame_range);
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KSwitchFlag(
      args[0].AsClip(),       // combe
      (float)args[1].AsFloat(60),  // thY
      (float)args[2].AsFloat(80),  // thC
      env
    );
  }
};

bool cpu_contains_durty_block(const uint8_t* flagp, int width, int height, int pitch, int* work)
{
  for (int by = 0; by < height; ++by) {
    for (int bx = 0; bx < width; ++bx) {
      if (flagp[bx + by * pitch]) return true;
    }
  }
  return false;
}

__global__ void kl_init_contains_durty_block(int* work)
{
  *work = 0;
}

__global__ void kl_contains_durty_block(const uint8_t* flagp, int width, int height, int pitch, int* work)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    if (flagp[x + y * pitch]) {
      *work = 1;
    }
  }
}

class KContainsCombe : public GenericVideoFilter
{
  VideoInfo workvi;

  void ContainsDurtyBlock(Frame& flag, Frame& work, PNeoEnv env)
  {
    const uint8_t* flagp = flag.GetReadPtr<uint8_t>();
    int* pwork = work.GetWritePtr<int>();

    int pitch = flag.GetPitch<uint8_t>();
    int width = flag.GetWidth<uint8_t>();
    int height = flag.GetHeight();

    if (IS_CUDA) {
			cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());
      dim3 threads(32, 16);
      dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
      kl_init_contains_durty_block << <1, 1, 0, stream >> > (pwork);
      kl_contains_durty_block << <blocks, threads, 0, stream >> > (flagp, width, height, pitch, pwork);
    }
    else {
      *pwork = cpu_contains_durty_block(flagp, width, height, pitch, pwork);
    }
  }

public:
  KContainsCombe(PClip flag, IScriptEnvironment* env)
    : GenericVideoFilter(flag)
  {
    int work_bytes = sizeof(int);
    workvi.pixel_type = VideoInfo::CS_BGR32;
    workvi.width = 4;
    workvi.height = nblocks(work_bytes, workvi.width * 4);
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    Frame src = child->GetFrame(n, env);
    Frame dst = env->NewVideoFrame(workvi);
    ContainsDurtyBlock(src, dst, env);

    return dst.frame;
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    if (cachehints == CACHE_GET_DEV_TYPE) {
      return GetDeviceTypes(child) &
        (DEV_TYPE_CPU | DEV_TYPE_CUDA);
    }
    else if (cachehints == CACHE_GET_MTMODE) {
      return MT_NICE_FILTER;
    }
    return 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KContainsCombe(
      args[0].AsClip(),       // flag
      env
    );
  }
};

class KCombeMask : public KFMFilterBase
{
  PClip flagclip;

  VideoInfo combvi;

  void BilinearImage(
    uint8_t* dst, uint8_t* dsttmp, int dpitch, int scalew, int scaleh,
    const uint8_t* src, int spitch, int swidth, int sheight, PNeoEnv env)
  {
    void(*table[2][2][2])(uint8_t* dst, int width, int height, int dpitch, const uint8_t* src, int spitch, PNeoEnv env) = {
      {
        { cpu_bilinear_h<4, 2>, cpu_bilinear_h<8, 3> },
        { cpu_bilinear_v<4, 2>, cpu_bilinear_v<8, 3> }
      },
      {
        { launch_bilinear_h<4, 2>, launch_bilinear_h<8, 3> },
        { launch_bilinear_v<4, 2>, launch_bilinear_v<8, 3> }
      }
    };

    int dwidth = swidth * scalew;
    int dheight = sheight * scaleh;

    int is_cuda = IS_CUDA;

    assert(scalew == 8 || scalew == 4);
    assert(scaleh == 8 || scaleh == 4);

    // �㉺�p�f�B���O1�s�����܂߂ď���
    table[is_cuda][0][scalew == 8](dsttmp, dwidth, sheight + 2, dpitch, src - spitch, spitch, env);
    // �\�[�X�̓p�f�B���O1�s�����X�L�b�v���ēn��
    table[is_cuda][1][scaleh == 8](dst, dwidth, dheight, dpitch, dsttmp + dpitch, dpitch, env);
  }

  Frame MakeMask(Frame& flag, PNeoEnv env)
  {
    // SwitchFlag����Mask�쐬
    Frame dst = env->NewVideoFrame(vi);
    Frame dsttmp = env->NewVideoFrame(vi);

    env->MakeWritable(&flag.frame);
    uint8_t* flagp = flag.GetWritePtr<uint8_t>(PLANAR_Y);
    uint8_t* dstY = dst.GetWritePtr<uint8_t>(PLANAR_Y);
    uint8_t* dstUV = dst.GetWritePtr<uint8_t>(PLANAR_U);
    uint8_t* dsttmpY = dsttmp.GetWritePtr<uint8_t>(PLANAR_Y);
    uint8_t* dsttmpUV = dsttmp.GetWritePtr<uint8_t>(PLANAR_U);

    int fpitch = flag.GetPitch<uint8_t>();
    int fwidth = flag.GetWidth<uint8_t>();
    int fheight = flag.GetHeight();

    int pitchY = dst.GetPitch<uint8_t>(PLANAR_Y);
    int pitchUV = dst.GetPitch<uint8_t>(PLANAR_U);
    int width = dst.GetWidth<uint8_t>(PLANAR_Y);
    int widthUV = dst.GetWidth<uint8_t>(PLANAR_U);
    int height = dst.GetHeight(PLANAR_Y);
    int heightUV = dst.GetHeight(PLANAR_U);

    int scaleUVw = widthUV / fwidth;
    int scaleUVh = heightUV / fheight;
    assert(scaleUVw == 8 || scaleUVw == 4);
    assert(scaleUVh == 8 || scaleUVh == 4);

    if (IS_CUDA) {
			cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());
      {
        dim3 threads(32, 1);
        dim3 blocks(nblocks(fwidth, threads.x));
        kl_padv << <blocks, threads, 0, stream >> > (flagp, fwidth, fheight, fpitch, 1);
        DEBUG_SYNC;
      }
      {
        dim3 threads(1, 32);
        dim3 blocks(1, nblocks(fheight + 1 * 2, threads.y));
        kl_padh << <blocks, threads, 0, stream >> > (flagp - fpitch, fwidth, fheight + 1 * 2, fpitch, 1);
        DEBUG_SYNC;
      }
    }
    else {
      cpu_padv(flagp, fwidth, fheight, fpitch, 1);
      cpu_padh(flagp - fpitch, fwidth, fheight + 1 * 2, fpitch, 1);
    }

    BilinearImage(dstY, dsttmpY, pitchY, DC_BLOCK_SIZE, DC_BLOCK_SIZE, flagp, fpitch, fwidth, fheight, env);
    BilinearImage(dstUV, dsttmpUV, pitchUV, scaleUVw, scaleUVh, flagp, fpitch, fwidth, fheight, env);

    return dst;
  }

public:
  KCombeMask(PClip source, PClip flag, IScriptEnvironment* env)
    : KFMFilterBase(source, env)
    , flagclip(flag)
  {
    VideoInfo flagvi = flagclip->GetVideoInfo();

    if ((flagvi.width - COMBE_FLAG_PAD_W * 2) * 8 != vi.width) {
      env->ThrowError("[KCombeMask] width unmatch");
    }
    if ((flagvi.height - COMBE_FLAG_PAD_H * 2) * 8 != vi.height) {
      env->ThrowError("[KCombeMask] height unmatch");
    }
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    Frame flag = WrapSwitchFragFrame(flagclip->GetFrame(n, env));
    Frame dst = MakeMask(flag, env);

    return dst.frame;
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    if (cachehints == CACHE_GET_MTMODE) {
      return MT_NICE_FILTER;
    }
    return KFMFilterBase::SetCacheHints(cachehints, frame_range);
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KCombeMask(
      args[0].AsClip(),       // source
      args[1].AsClip(),       // flag
      env
    );
  }
};

__device__ __host__ int BinomialMerge(int a, int b, int c)
{
  return (a + 2 * b + c + 2) >> 2;
}

template <typename pixel_t>
void cpu_remove_combe2(pixel_t* __restrict__ dst,
  const pixel_t* __restrict__ src, int width, int height, int pitch,
  const uchar2* __restrict__ combe, int c_pitch, int thcombe)
{
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int score = combe[(x >> 2) + (y >> 2) * c_pitch].x;
      if (score >= thcombe) {
        dst[x + y * pitch] = BinomialMerge(
          src[x + (y - 1) * pitch],
          src[x + y * pitch],
          src[x + (y + 1) * pitch]);
      }
      else {
        dst[x + y * pitch] = src[x + y * pitch];
      }
    }
  }
}

template <typename pixel_t>
__global__ void kl_remove_combe2(pixel_t* __restrict__ dst,
  const pixel_t* __restrict__ src, int width, int height, int pitch,
  const uchar2* __restrict__ combe, int c_pitch, int thcombe)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width && y < height) {
    int score = combe[(x >> 2) + (y >> 2) * c_pitch].x;
    if (score >= thcombe) {
      dst[x + y * pitch] = BinomialMerge(
        src[x + (y - 1) * pitch],
        src[x + y * pitch],
        src[x + (y + 1) * pitch]);
    }
    else {
      dst[x + y * pitch] = src[x + y * pitch];
    }
  }
}

class KRemoveCombe : public KFMFilterBase
{
  PClip superclip;

  float thY;
  float thC;

  template <typename pixel_t>
  void RemoveCombe(Frame& dst, Frame& src, Frame& combe, int thY, int thC, PNeoEnv env)
  {
    const uchar2* combeY = combe.GetReadPtr<uchar2>(PLANAR_Y);
    const uchar2* combeU = combe.GetReadPtr<uchar2>(PLANAR_U);
    const uchar2* combeV = combe.GetReadPtr<uchar2>(PLANAR_V);
    const pixel_t* srcY = src.GetReadPtr<pixel_t>(PLANAR_Y);
    const pixel_t* srcU = src.GetReadPtr<pixel_t>(PLANAR_U);
    const pixel_t* srcV = src.GetReadPtr<pixel_t>(PLANAR_V);
    pixel_t* dstY = dst.GetWritePtr<pixel_t>(PLANAR_Y);
    pixel_t* dstU = dst.GetWritePtr<pixel_t>(PLANAR_U);
    pixel_t* dstV = dst.GetWritePtr<pixel_t>(PLANAR_V);

    int pitchY = src.GetPitch<pixel_t>(PLANAR_Y);
    int pitchUV = src.GetPitch<pixel_t>(PLANAR_U);
    int fpitchY = combe.GetPitch<uchar2>(PLANAR_Y);
    int fpitchUV = combe.GetPitch<uchar2>(PLANAR_U);
    int width = src.GetWidth<pixel_t>(PLANAR_Y);
    int widthUV = src.GetWidth<pixel_t>(PLANAR_U);
    int height = src.GetHeight(PLANAR_Y);
    int heightUV = src.GetHeight(PLANAR_U);

    if (IS_CUDA) {
			cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());
      dim3 threads(32, 16);
      dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
      dim3 blocksUV(nblocks(widthUV, threads.x), nblocks(heightUV, threads.y));
      kl_remove_combe2 << <blocks, threads, 0, stream >> > (dstY, srcY, width, height, pitchY, combeY, fpitchY, thY);
      DEBUG_SYNC;
      kl_remove_combe2 << <blocksUV, threads, 0, stream >> > (dstU, srcU, widthUV, heightUV, pitchUV, combeU, fpitchUV, thC);
      DEBUG_SYNC;
      kl_remove_combe2 << <blocksUV, threads, 0, stream >> > (dstV, srcV, widthUV, heightUV, pitchUV, combeV, fpitchUV, thC);
      DEBUG_SYNC;
    }
    else {
      cpu_remove_combe2(dstY, srcY, width, height, pitchY, combeY, fpitchY, thY);
      cpu_remove_combe2(dstU, srcU, widthUV, heightUV, pitchUV, combeU, fpitchUV, thC);
      cpu_remove_combe2(dstV, srcV, widthUV, heightUV, pitchUV, combeV, fpitchUV, thC);
    }
  }

  template <typename pixel_t>
  PVideoFrame GetFrameT(int n, PNeoEnv env)
  {
    Frame src = Frame(child->GetFrame(n, env), VPAD);
    Frame combe = superclip->GetFrame(n, env);
    Frame dst = env->NewVideoFrame(vi);

    RemoveCombe<pixel_t>(dst, src, combe, (int)thY, (int)thC, env);

    return dst.frame;
  }

public:
  KRemoveCombe(PClip pad, PClip super, float thY, float thC, IScriptEnvironment* env)
    : KFMFilterBase(pad, env)
    , superclip(super)
    , thY(thY)
    , thC(thC)
  {
    if (vi.width & 7) env->ThrowError("[KRemoveCombe]: width must be multiple of 8");
    if (vi.height & 7) env->ThrowError("[KRemoveCombe]: height must be multiple of 8");

    // superclip���`�F�b�N
    VideoInfo supervi = superclip->GetVideoInfo();
    if (supervi.num_frames != vi.num_frames) {
      env->ThrowError("[KRemoveCombe]: padclip and superclip should have the same num_frames");
    }

    vi.height -= VPAD * 2;
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    int pixelSize = vi.ComponentSize();
    switch (pixelSize) {
    case 1:
      return GetFrameT<uint8_t>(n, env);
    case 2:
      return GetFrameT<uint16_t>(n, env);
    default:
      env->ThrowError("[KRemoveCombe] Unsupported pixel format");
    }

    return PVideoFrame();
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    if (cachehints == CACHE_GET_MTMODE) {
      return MT_NICE_FILTER;
    }
    return KFMFilterBase::SetCacheHints(cachehints, frame_range);
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KRemoveCombe(
      args[0].AsClip(),       // source
      args[1].AsClip(),       // super
      (float)args[2].AsFloat(6), // thY
      (float)args[3].AsFloat(6), // thC
      env
    );
  }
};

void AddFuncCombingAnalyze(IScriptEnvironment* env)
{
  env->AddFunction("KFMSuper", "cc", KFMSuper::Create, 0);
  env->AddFunction("KCleanSuper", "c[thY]i[thC]i", KCleanSuper::Create, 0);
  env->AddFunction("KPreCycleAnalyze", "c[threshMY]i[threshSY]i[threshMC]i[threshSC]i", KPreCycleAnalyze::Create, 0);
  env->AddFunction("KPreCycleAnalyzeShow", "cc", KPreCycleAnalyzeShow::Create, 0);
  env->AddFunction("KFMSuperShow", "c[threshMY]i[threshSY]i[threshMC]i[threshSC]i", KFMSuperShow::Create, 0);

  env->AddFunction("KTelecine", "cc[show]b", KTelecine::Create, 0);
  env->AddFunction("KTelecineSuper", "cc", KTelecineSuper::Create, 0);
  env->AddFunction("KSwitchFlag", "c[thY]f[thC]f", KSwitchFlag::Create, 0);
  env->AddFunction("KContainsCombe", "c", KContainsCombe::Create, 0);

  env->AddFunction("KCombeMask", "cc", KCombeMask::Create, 0);
  env->AddFunction("KRemoveCombe", "cc[thY]f[thC]f", KRemoveCombe::Create, 0);
}
