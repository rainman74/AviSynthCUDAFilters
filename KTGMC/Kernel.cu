#define _CRT_SECURE_NO_WARNINGS
#include "avisynth.h"

#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <memory>

#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>

#include "CommonFunctions.h"
#include "DeviceLocalData.h"
#include "DebugWriter.h"
#include "CudaDebug.h"
#include "ReduceKernel.cuh"
#include "VectorFunctions.cuh"
#include "GenericImageFunctions.cuh"
#include "Misc.h"

#define LOG_PRINT 0

// ���̃t�@�C���Ɋւ��ẮA����Ŏ��s�ł���kernel�������Ȃ��̂ŁA���܂荂�������Ȃ�
// �܂��x���Ȃ�̂ŁAstream�ɂ�����͖���������
#define ENABLE_MULTI_STREAM 0

class CUDAFilterBase : public GenericVideoFilter {
  std::unique_ptr<cudaPlaneStreams> planeStreams;
public:
  CUDAFilterBase(PClip _child, IScriptEnvironment* env_) : GenericVideoFilter(_child), planeStreams(std::make_unique<cudaPlaneStreams>()) {
      planeStreams->initStream((cudaStream_t)((PNeoEnv)env_)->GetDeviceStream());
  }
  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    if (cachehints == CACHE_GET_DEV_TYPE) {
      return DEV_TYPE_CUDA;
    }
    else if (cachehints == CACHE_GET_MTMODE) {
      return MT_NICE_FILTER;
    }
    return 0;
  };
  virtual cudaEventPlanes *CreateEventPlanes() {
      return (ENABLE_MULTI_STREAM) ? planeStreams->CreateEventPlanes() : nullptr;
  }
  virtual void *GetDeviceStreamPlane(int idx) {
      if (ENABLE_MULTI_STREAM) {
          switch (idx) {
          case 1: return planeStreams->GetDeviceStreamU();
          case 2: return planeStreams->GetDeviceStreamV();
          case 0:
          default: return planeStreams->GetDeviceStreamY();
          }
      }
      return planeStreams->GetDeviceStreamDefault();
  }
};

#pragma region resample

struct ResamplingProgram {
  PNeoEnv env;
  int source_size, target_size;
  double crop_start, crop_size;
  int filter_size;

  // Array of Integer indicate starting point of sampling
  std::unique_ptr<DeviceLocalData<int>> pixel_offset;

  // Array of array of coefficient for each pixel
  // {{pixel[0]_coeff}, {pixel[1]_coeff}, ...}
  std::unique_ptr<DeviceLocalData<float>> pixel_coefficient_float;

  ResamplingProgram(int filter_size, int source_size, int target_size, double crop_start, double crop_size,
    int* ppixel_offset, float* ppixel_coefficient_float, PNeoEnv env)
    : filter_size(filter_size), source_size(source_size), target_size(target_size), crop_start(crop_start), crop_size(crop_size),
    env(env)
  {
    pixel_offset = std::unique_ptr<DeviceLocalData<int>>(
      new DeviceLocalData<int>(ppixel_offset, target_size, env));
    pixel_coefficient_float = std::unique_ptr<DeviceLocalData<float>>(
      new DeviceLocalData<float>(ppixel_coefficient_float, target_size * filter_size, env));
  };
};

class ResamplingFunction
  /**
  * Pure virtual base class for resampling functions
  */
{
public:
  virtual double f(double x) = 0;
  virtual double support() = 0;

  virtual std::unique_ptr<ResamplingProgram> GetResamplingProgram(int source_size, double crop_start, double crop_size, int target_size, PNeoEnv env);
};

std::unique_ptr<ResamplingProgram> ResamplingFunction::GetResamplingProgram(int source_size, double crop_start, double crop_size, int target_size, PNeoEnv env)
{
  double filter_scale = double(target_size) / crop_size;
  double filter_step = min(filter_scale, 1.0);
  double filter_support = support() / filter_step;
  int fir_filter_size = int(ceil(filter_support * 2));

  std::unique_ptr<int[]> pixel_offset = std::unique_ptr<int[]>(new int[target_size]);
  std::unique_ptr<float[]> pixel_coefficient_float = std::unique_ptr<float[]>(new float[target_size * fir_filter_size]);
  //ResamplingProgram* program = new ResamplingProgram(fir_filter_size, source_size, target_size, crop_start, crop_size, env);

  // this variable translates such that the image center remains fixed
  double pos;
  double pos_step = crop_size / target_size;

  if (source_size <= filter_support) {
    env->ThrowError("Resize: Source image too small for this resize method. Width=%d, Support=%d", source_size, int(ceil(filter_support)));
  }

  if (fir_filter_size == 1) // PointResize
    pos = crop_start;
  else
    pos = crop_start + ((crop_size - target_size) / (target_size * 2)); // TODO this look wrong, gotta check

  for (int i = 0; i < target_size; ++i) {
    // Clamp start and end position such that it does not exceed frame size
    int end_pos = int(pos + filter_support);

    if (end_pos > source_size - 1)
      end_pos = source_size - 1;

    int start_pos = end_pos - fir_filter_size + 1;

    if (start_pos < 0)
      start_pos = 0;

    pixel_offset[i] = start_pos;

    // the following code ensures that the coefficients add to exactly FPScale
    double total = 0.0;

    // Ensure that we have a valid position
    double ok_pos = clamp(pos, 0.0, (double)(source_size - 1));

    // Accumulate all coefficients for weighting
    for (int j = 0; j < fir_filter_size; ++j) {
      total += f((start_pos + j - ok_pos) * filter_step);
    }

    if (total == 0.0) {
      // Shouldn't happened for valid positions.
      total = 1.0;
    }

    double value = 0.0;

    // Now we generate real coefficient
    for (int k = 0; k < fir_filter_size; ++k) {
      double new_value = value + f((start_pos + k - ok_pos) * filter_step) / total;
      pixel_coefficient_float[i*fir_filter_size + k] = float(new_value - value); // no scaling for float
      value = new_value;
    }

    pos += pos_step;
  }

  return std::unique_ptr<ResamplingProgram>(new ResamplingProgram(
    fir_filter_size, source_size, target_size, crop_start, crop_size,
    pixel_offset.get(), pixel_coefficient_float.get(), env));
}

/*********************************
*** Mitchell-Netravali filter ***
*********************************/

class MitchellNetravaliFilter : public ResamplingFunction
  /**
  * Mitchell-Netraveli filter, used in BicubicResize
  **/
{
public:
  MitchellNetravaliFilter(double b = 1. / 3., double c = 1. / 3.);
  double f(double x);
  double support() { return 2.0; }

private:
  double p0, p2, p3, q0, q1, q2, q3;
};

MitchellNetravaliFilter::MitchellNetravaliFilter(double b, double c) {
  p0 = (6. - 2.*b) / 6.;
  p2 = (-18. + 12.*b + 6.*c) / 6.;
  p3 = (12. - 9.*b - 6.*c) / 6.;
  q0 = (8.*b + 24.*c) / 6.;
  q1 = (-12.*b - 48.*c) / 6.;
  q2 = (6.*b + 30.*c) / 6.;
  q3 = (-b - 6.*c) / 6.;
}

double MitchellNetravaliFilter::f(double x) {
  x = fabs(x);
  return (x < 1) ? (p0 + x*x*(p2 + x*p3)) : (x < 2) ? (q0 + x*(q1 + x*(q2 + x*q3))) : 0.0;
}

/***********************
*** Gaussian filter ***
***********************/

/* Solve taps from p*value*value < 9 as pow(2.0, -9.0) == 1.0/512.0 i.e 0.5 bit
value*value < 9/p       p = param*0.1;
value*value < 90/param
value*value < 90/{0.1, 22.5, 30.0, 100.0}
value*value < {900, 4.0, 3.0, 0.9}
value       < {30, 2.0, 1.73, 0.949}         */

class GaussianFilter : public ResamplingFunction
  /**
  * GaussianFilter, from swscale.
  **/
{
public:
  GaussianFilter(double p = 30.0);
  double f(double x);
  double support() { return 4.0; };

private:
  double param;
};

GaussianFilter::GaussianFilter(double p) {
  param = clamp(p, 0.1, 100.0);
}

double GaussianFilter::f(double value) {
  double p = param*0.1;
  return pow(2.0, -p*value*value);
}

template <typename vpixel_t, int filter_size>
__global__ void kl_resample_v(
  const vpixel_t* __restrict__ src0, const vpixel_t* __restrict__ src1,
  vpixel_t* dst0, vpixel_t* dst1,
  int src_pitch4, int dst_pitch4,
  int width4, int height,
  const int* __restrict__ offset, const float* __restrict__ coef)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  const vpixel_t* __restrict__ src = (blockIdx.z) ? src1 : src0;
  vpixel_t* dst = (blockIdx.z) ? dst1 : dst0;

  if (x < width4 && y < height) {
    int begin = offset[y];
    float4 result = { 0 };
    for (int i = 0; i < filter_size; ++i) {
      result += to_float(src[x + (begin + i) * src_pitch4]) * coef[y * filter_size + i];
    }
    result = clamp(result, 0, (sizeof(src[0].x) == 1) ? 255 : 65535);
#if 0
    if (x == 1200 && y == 0) {
      printf("! %f\n", result);
    }
#endif
    dst[x + y * dst_pitch4] = VHelper<vpixel_t>::cast_to(result + 0.5f);
  }
}

template <typename vpixel_t, int filter_size>
void launch_resmaple_v(
  const vpixel_t* src0, const vpixel_t* src1, vpixel_t* dst0, vpixel_t* dst1,
  int src_pitch4, int dst_pitch4,
  int width4, int height,
  const int* offset, const float* coef, cudaStream_t stream)
{
  dim3 threads(32, 16);
  dim3 blocks(nblocks(width4, threads.x), nblocks(height, threads.y), src1 && dst1 ? 2 : 1);
  kl_resample_v<vpixel_t, filter_size> << <blocks, threads, 0, stream >> > (
    src0, src1, dst0, dst1, src_pitch4, dst_pitch4, width4, height, offset, coef);
}

enum {
  RESAMPLE_H_W = 32,
  RESAMPLE_H_H = 16,
};

template <typename pixel_t, int filter_size>
__global__ void kl_resample_h(
  const pixel_t* __restrict__ src0, const pixel_t* __restrict__ src1,
  pixel_t* dst0, pixel_t* dst1,
  int src_pitch, int dst_pitch,
  int width, int height,
  const int* __restrict__ offset, const float* __restrict__ coef)
{
  enum {
    THREADS = RESAMPLE_H_W * RESAMPLE_H_H,
    COEF_BUF_LEN = RESAMPLE_H_W * 4 * filter_size,
    N_READ_LOOP = COEF_BUF_LEN / THREADS
  };

  const pixel_t* __restrict__ src = (blockIdx.z) ? src1 : src0;
  pixel_t* dst = (blockIdx.z) ? dst1 : dst0;

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = tx + ty * RESAMPLE_H_W;
  int xbase = blockIdx.x * RESAMPLE_H_W * 4;
  int ybase = blockIdx.y * RESAMPLE_H_H;
  int x = tx + xbase;
  int y = ty + ybase;

  __shared__ float scoef[COEF_BUF_LEN];
  __shared__ pixel_t ssrc[RESAMPLE_H_H][RESAMPLE_H_W * 4 + filter_size + 1];

  int src_base = xbase - filter_size / 2;

  int start_pos = tx + src_base;
  if (blockIdx.x == 0 || blockIdx.x == gridDim.x - 1) {
    // x���[�̏ꍇ�͏������낢��
    if (y < height) {
      if (start_pos + RESAMPLE_H_W * 0 < width) {
        if (start_pos >= 0) {
          ssrc[ty][tx + RESAMPLE_H_W * 0] = src[start_pos + RESAMPLE_H_W * 0 + y * src_pitch];
        }
      }
      if (start_pos + RESAMPLE_H_W * 1 < width) {
        ssrc[ty][tx + RESAMPLE_H_W * 1] = src[start_pos + RESAMPLE_H_W * 1 + y * src_pitch];
      }
      if (start_pos + RESAMPLE_H_W * 2 < width) {
        ssrc[ty][tx + RESAMPLE_H_W * 2] = src[start_pos + RESAMPLE_H_W * 2 + y * src_pitch];
      }
      if (start_pos + RESAMPLE_H_W * 3 < width) {
        ssrc[ty][tx + RESAMPLE_H_W * 3] = src[start_pos + RESAMPLE_H_W * 3 + y * src_pitch];
      }
      // �Ō�̔��[����
      if (tx < filter_size + 1 && start_pos + RESAMPLE_H_W * 4 < width) {
        ssrc[ty][tx + RESAMPLE_H_W * 4] = src[start_pos + RESAMPLE_H_W * 4 + y * src_pitch];
      }
    }
    int lend = min((width - xbase) * filter_size, COEF_BUF_LEN);
    for (int lx = tid; lx < lend; lx += THREADS) {
      scoef[lx] = coef[xbase * filter_size + lx];
    }
  }
  else {
    // x���[�ł͂Ȃ��̂ŏ����Ȃ�
    if (y < height) {
      ssrc[ty][tx + RESAMPLE_H_W * 0] = src[start_pos + RESAMPLE_H_W * 0 + y * src_pitch];
      ssrc[ty][tx + RESAMPLE_H_W * 1] = src[start_pos + RESAMPLE_H_W * 1 + y * src_pitch];
      ssrc[ty][tx + RESAMPLE_H_W * 2] = src[start_pos + RESAMPLE_H_W * 2 + y * src_pitch];
      ssrc[ty][tx + RESAMPLE_H_W * 3] = src[start_pos + RESAMPLE_H_W * 3 + y * src_pitch];
      // �Ō�̔��[����
      if (tx < filter_size + 1 && start_pos + RESAMPLE_H_W * 4 < width) {
        ssrc[ty][tx + RESAMPLE_H_W * 4] = src[start_pos + RESAMPLE_H_W * 4 + y * src_pitch];
      }
    }
    for (int i = 0; i < N_READ_LOOP; ++i) {
      scoef[i * THREADS + tid] = coef[xbase * filter_size + i * THREADS + tid];
    }
    if (THREADS * N_READ_LOOP + tid < COEF_BUF_LEN) {
      scoef[THREADS * N_READ_LOOP + tid] = coef[xbase * filter_size + THREADS * N_READ_LOOP + tid];
    }
  }
  __syncthreads();

  if (y < height) {
    for (int v = 0; v < 4; ++v) {
      if (x + v * RESAMPLE_H_W < width) {
        int begin = offset[x + v * RESAMPLE_H_W] - src_base;
#if 0
        if (begin < 0 || begin >= RESAMPLE_H_W * 4 + 1) {
          printf("[resample_v kernel] Unexpected offset %d - %d at %d\n", offset[x + v * RESAMPLE_H_W], src_base, x + v * RESAMPLE_H_W);
        }
#endif
        float result = 0;
        for (int i = 0; i < filter_size; ++i) {
          result += ssrc[ty][begin + i] * scoef[(tx + v * RESAMPLE_H_W) * filter_size + i];
        }
        result = clamp<float>(result, 0, (sizeof(pixel_t) == 1) ? 255 : 65535);
        dst[x + v * RESAMPLE_H_W + y * dst_pitch] = pixel_t(result + 0.5f);
      }
    }
  }
}

template <typename pixel_t, int filter_size>
void launch_resmaple_h(
  const pixel_t* src0, const pixel_t* src1, pixel_t* dst0, pixel_t* dst1,
  int src_pitch, int dst_pitch,
  int width, int height,
  const int* offset, const float* coef, cudaStream_t stream)
{
  dim3 threads(RESAMPLE_H_W, RESAMPLE_H_H);
  dim3 blocks(nblocks(width / 4, threads.x), nblocks(height, threads.y), src1 && dst1 ? 2 : 1);
  kl_resample_h<pixel_t, filter_size> << <blocks, threads, 0, stream >> > (
    src0, src1, dst0, dst1, src_pitch, dst_pitch, width, height, offset, coef);
}

#pragma endregion

class KTGMC_Bob : public CUDAFilterBase {
  std::unique_ptr<ResamplingProgram> program_e_y;
  std::unique_ptr<ResamplingProgram> program_e_uv;
  std::unique_ptr<ResamplingProgram> program_o_y;
  std::unique_ptr<ResamplingProgram> program_o_uv;

  bool parity;
  int logUVx;
  int logUVy;

  // 1�t���[�����L���b�V�����Ă���
  std::mutex mtx_;
  int cacheN;
  PVideoFrame cache[2];

  template <typename pixel_t>
  void MakeFrameT(bool top, PVideoFrame& src, PVideoFrame& dst,
    ResamplingProgram* program_y, ResamplingProgram* program_uv, PNeoEnv env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;
    cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());

    const int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };

    auto planeEvent = CreateEventPlanes();

    const bool uvSamePitch = src->GetPitch(PLANAR_U) == src->GetPitch(PLANAR_V);

    for (int p = 0; p < (uvSamePitch ? 2 : 3); ++p) {
      const vpixel_t* srcptr = reinterpret_cast<const vpixel_t*>(src->GetReadPtr(planes[p]));
      const vpixel_t* srcptr1 = (p > 0 && uvSamePitch) ? reinterpret_cast<const vpixel_t*>(src->GetReadPtr(planes[p+1])) : nullptr;
      int src_pitch4 = src->GetPitch(planes[p]) / sizeof(pixel_t) / 4;

      // separate field
      srcptr += top ? 0 : src_pitch4;
      srcptr1 += top ? 0 : src_pitch4;
      src_pitch4 *= 2;

      vpixel_t* dstptr = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(planes[p]));
      vpixel_t* dstptr1 = (p > 0 && uvSamePitch) ? reinterpret_cast<vpixel_t*>(dst->GetWritePtr(planes[p+1])) : nullptr;
      int dst_pitch4 = dst->GetPitch(planes[p]) / sizeof(pixel_t) / 4;

      ResamplingProgram* prog = (p == 0) ? program_y : program_uv;

      int width4 = vi.width / 4;
      int height = vi.height;

      if (p > 0) {
        width4 >>= logUVx;
        height >>= logUVy;
      }

      launch_resmaple_v<vpixel_t, 4>(
        srcptr, srcptr1, dstptr, dstptr1, src_pitch4, dst_pitch4, width4, height,
        prog->pixel_offset->GetData(env), prog->pixel_coefficient_float->GetData(env), stream);

      DEBUG_SYNC;
    }
    if (planeEvent) planeEvent->finPlane();
  }

  void MakeFrame(bool top, PVideoFrame& src, PVideoFrame& dst,
    ResamplingProgram* program_y, ResamplingProgram* program_uv, PNeoEnv env)
  {
    int pixelSize = vi.ComponentSize();
    switch (pixelSize) {
    case 1:
      MakeFrameT<uint8_t>(top, src, dst, program_y, program_uv, env);
      break;
    case 2:
      MakeFrameT<uint16_t>(top, src, dst, program_y, program_uv, env);
      break;
    default:
      env->ThrowError("[KTGMC_Bob] Unsupported pixel format");
    }
  }

public:
  KTGMC_Bob(PClip _child, double b, double c, IScriptEnvironment* env_)
    : CUDAFilterBase(_child, env_)
    , parity(_child->GetParity(0))
    , cacheN(-1)
    , logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
    , logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
  {
    PNeoEnv env = env_;

    // �t���[�����AFPS��2�{
    vi.num_frames *= 2;
    vi.MulDivFPS(2, 1);

    double shift = parity ? 0.25 : -0.25;

    int y_height = vi.height;
    int uv_height = vi.height >> logUVy;

    program_e_y = MitchellNetravaliFilter(b, c).GetResamplingProgram(y_height / 2, shift, y_height / 2, y_height, env);
    program_e_uv = MitchellNetravaliFilter(b, c).GetResamplingProgram(uv_height / 2, shift, uv_height / 2, uv_height, env);
    program_o_y = MitchellNetravaliFilter(b, c).GetResamplingProgram(y_height / 2, -shift, y_height / 2, y_height, env);
    program_o_uv = MitchellNetravaliFilter(b, c).GetResamplingProgram(uv_height / 2, -shift, uv_height / 2, uv_height, env);
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    if (!IS_CUDA) {
      env->ThrowError("[KTGMC_Bob] CUDA�t���[������͂��Ă�������");
    }
#if LOG_PRINT
    if (IS_CUDA) {
      printf("KTGMC_Bob[CUDA]: N=%d\n", n);
    }
#endif

    int srcN = n >> 1;

    {
      std::lock_guard<std::mutex> lock(mtx_);

      if (cacheN >= 0 && srcN == cacheN) {
        return cache[n % 2];
      }
    }

    PVideoFrame src = child->GetFrame(srcN, env);

    PVideoFrame bobE = env->NewVideoFrame(vi);
    PVideoFrame bobO = env->NewVideoFrame(vi);

    MakeFrame(parity, src, bobE, program_e_y.get(), program_e_uv.get(), env);
    MakeFrame(!parity, src, bobO, program_o_y.get(), program_o_uv.get(), env);

    {
      std::lock_guard<std::mutex> lock(mtx_);

      cacheN = n / 2;
      cache[0] = bobE;
      cache[1] = bobO;

      return cache[n % 2];
    }
  }

  bool __stdcall GetParity(int n) {
    return child->GetParity(0);
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    if (cachehints == CACHE_GET_MTMODE) {
      return MT_NICE_FILTER;
    }
    return CUDAFilterBase::SetCacheHints(cachehints, frame_range);
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return new KTGMC_Bob(
      args[0].AsClip(),
      args[1].AsFloat(0),
      args[2].AsFloat(0.5),
      env);
  }
};

enum { CALC_SAD_THREADS = 256 };

__global__ void kl_init_sad(float *sad)
{
  sad[threadIdx.x] = 0;
}

template <typename vpixel_t>
__global__ void kl_calculate_sad(
    const vpixel_t* pSrcA, const vpixel_t* pSrcB,
    const vpixel_t* pPrv1A, const vpixel_t* pPrv1B,
    const vpixel_t* pFwd1A, const vpixel_t* pFwd1B,
    const vpixel_t* pPrv2A, const vpixel_t* pPrv2B,
    const vpixel_t* pFwd2A, const vpixel_t* pFwd2B,
    int width4, int height, int pitch4,
    float* sad0, float* sad1)
{
  int y = blockIdx.x;
  const vpixel_t* pA = (blockIdx.z) ? pSrcB : pSrcA;
  const vpixel_t* pB = nullptr;
  switch (blockIdx.y) {
  case 0: pB = (blockIdx.z) ? pPrv1B : pPrv1A; break;
  case 1: pB = (blockIdx.z) ? pFwd1B : pFwd1A; break;
  case 2: pB = (blockIdx.z) ? pPrv2B : pPrv2A; break;
  case 3: pB = (blockIdx.z) ? pFwd2B : pFwd2A; break;
  }
  float *sad = ((blockIdx.z) ? sad1 : sad0) + blockIdx.y;

  int sum = 0;
  for (int x = threadIdx.x; x < width4; x += blockDim.x) {
      if (sizeof(vpixel_t) == sizeof(unsigned int)) {
          sum = __vabsdiff4(*(unsigned int*)&pA[x + y * pitch4], *(unsigned int*)&pB[x + y * pitch4], sum);
      } else {
          int4 p = absdiff(pA[x + y * pitch4], pB[x + y * pitch4]);
          sum += p.x + p.y + p.z + p.w;
      }
  }

  float tmpsad = sum;

  __shared__ float sbuf[CALC_SAD_THREADS];
  dev_reduce<float, CALC_SAD_THREADS, AddReducer<float>>(threadIdx.x, tmpsad, sbuf);

  if (threadIdx.x == 0) {
    atomicAdd(sad, tmpsad);
  }
}

template <typename vpixel_t>
__global__ void kl_binomial_temporal_soften_1(
  vpixel_t* pDstA, vpixel_t* pDstB,
  const vpixel_t* __restrict__ pSrcA, const vpixel_t* __restrict__ pSrcB,
  const vpixel_t* __restrict__ pRef0A, const vpixel_t* __restrict__ pRef0B,
  const vpixel_t* __restrict__ pRef1A, const vpixel_t* __restrict__ pRef1B,
  const float* __restrict__ sadA, const float* __restrict__ sadB,
  float scenechange, int width4, int height, int pitch4)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  const vpixel_t* __restrict__ pSrc  = (blockIdx.z) ? pSrcB : pSrcA;
  vpixel_t*                    pDst  = (blockIdx.z) ? pDstB : pDstA;
  const vpixel_t* __restrict__ pRef0 = (blockIdx.z) ? pRef0B : pRef0A;
  const vpixel_t* __restrict__ pRef1 = (blockIdx.z) ? pRef1B : pRef1A;
  const float*    __restrict__ sad   = (blockIdx.z) ? sadB : sadA;

  __shared__ bool isSC[2];
  if (threadIdx.x < 2 && threadIdx.y == 0) {
    isSC[threadIdx.x] = (sad[threadIdx.x] >= scenechange);
  }
  __syncthreads();

  if (x < width4 && y < height) {
    int4 src = to_int(pSrc[x + y * pitch4]);
    int4 ref0 = isSC[0] ? src : to_int(pRef0[x + y * pitch4]);
    int4 ref1 = isSC[1] ? src : to_int(pRef1[x + y * pitch4]);

    int4 tmp = (ref0 + src * 2 + ref1 + 2) >> 2;
    tmp = clamp(tmp, 0, (sizeof(pSrc[0].x) == 1) ? 255 : 65535);
    pDst[x + y * pitch4] = VHelper<vpixel_t>::cast_to(tmp);
  }
}

template <typename vpixel_t>
__global__ void kl_binomial_temporal_soften_2(
  vpixel_t* pDstA, vpixel_t* pDstB,
  const vpixel_t* __restrict__ pSrcA, const vpixel_t* __restrict__ pSrcB,
  const vpixel_t* __restrict__ pRef0A, const vpixel_t* __restrict__ pRef0B,
  const vpixel_t* __restrict__ pRef1A, const vpixel_t* __restrict__ pRef1B,
  const vpixel_t* __restrict__ pRef2A, const vpixel_t* __restrict__ pRef2B,
  const vpixel_t* __restrict__ pRef3A, const vpixel_t* __restrict__ pRef3B,
  const float* __restrict__ sadA, const float* __restrict__ sadB,
  float scenechange, int width4, int height, int pitch4)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  const vpixel_t* __restrict__ pSrc = (blockIdx.z) ? pSrcB : pSrcA;
  vpixel_t*                    pDst = (blockIdx.z) ? pDstB : pDstA;
  const vpixel_t* __restrict__ pRef0 = (blockIdx.z) ? pRef0B : pRef0A;
  const vpixel_t* __restrict__ pRef1 = (blockIdx.z) ? pRef1B : pRef1A;
  const vpixel_t* __restrict__ pRef2 = (blockIdx.z) ? pRef2B : pRef2A;
  const vpixel_t* __restrict__ pRef3 = (blockIdx.z) ? pRef3B : pRef3A;
  const float*    __restrict__ sad   = (blockIdx.z) ? sadB : sadA;

  __shared__ bool isSC[4];
  if (threadIdx.x < 4 && threadIdx.y == 0) {
    isSC[threadIdx.x] = (sad[threadIdx.x] >= scenechange);
  }
  __syncthreads();

  if (x < width4 && y < height) {
    int4 src = to_int(pSrc[x + y * pitch4]);
    int4 ref0 = isSC[0] ? src : to_int(pRef0[x + y * pitch4]);
    int4 ref1 = isSC[1] ? src : to_int(pRef1[x + y * pitch4]);
    int4 ref2 = isSC[2] ? src : to_int(pRef2[x + y * pitch4]);
    int4 ref3 = isSC[3] ? src : to_int(pRef3[x + y * pitch4]);

    int4 tmp = (ref2 + ref0 * 4 + src * 6 + ref1 * 4 + ref3 + 4) >> 4;
    tmp = clamp(tmp, 0, (sizeof(pSrc[0].x) == 1) ? 255 : 65535);
    pDst[x + y * pitch4] = VHelper<vpixel_t>::cast_to(tmp);
  }
}

class KBinomialTemporalSoften : public CUDAFilterBase {

  int radius;
  int scenechange;
  bool chroma;

  int logUVx;
  int logUVy;

  PVideoFrame GetRefFrame(int ref, PNeoEnv env)
  {
    ref = clamp(ref, 0, vi.num_frames);
    return child->GetFrame(ref, env);
  }

  template <typename pixel_t>
  PVideoFrame Proc(int n, PNeoEnv env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;
    cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());

    PVideoFrame prv2, prv1, src, fwd1, fwd2;

    if (radius >= 2) {
      prv2 = GetRefFrame(n - 2, env);
    }
    prv1 = GetRefFrame(n - 1, env);
    src = GetRefFrame(n, env);
    fwd1 = GetRefFrame(n + 1, env);
    if (radius >= 2) {
      fwd2 = GetRefFrame(n + 2, env);
    }

    PVideoFrame work;
    int work_bytes = sizeof(float) * radius * 2 * 3;
    VideoInfo workvi = VideoInfo();
    workvi.pixel_type = VideoInfo::CS_BGR32;
    workvi.width = 2048;
    workvi.height = nblocks(work_bytes, workvi.width * 4);
    work = env->NewVideoFrame(workvi);
    float* sad = reinterpret_cast<float*>(work->GetWritePtr());

    PVideoFrame dst = env->NewVideoFrame(vi);

    kl_init_sad << <1, radius * 2 * 3, 0, stream >> > (sad);
    DEBUG_SYNC;

    const bool uvSamePitch = src->GetPitch(PLANAR_U) == src->GetPitch(PLANAR_V);

    auto planeEvent = CreateEventPlanes();

    int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };
    for (int p = 0; p < ((uvSamePitch && chroma) ? 2 : 3); ++p) {

      const auto planeStream = (cudaStream_t)GetDeviceStreamPlane(p);
      const vpixel_t* pSrcA = (const vpixel_t*)(src->GetReadPtr(planes[p]));
      const vpixel_t* pSrcB = (p > 0 && uvSamePitch) ? (const vpixel_t*)(src->GetReadPtr(planes[p + 1])) : nullptr;
      vpixel_t* pDstA = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(planes[p]));
      vpixel_t* pDstB = (p > 0 && uvSamePitch) ? reinterpret_cast<vpixel_t*>(dst->GetWritePtr(planes[p + 1])) : nullptr;

      int pitch = src->GetPitch(planes[p]) / sizeof(pixel_t);
      int width = vi.width;
      int height = vi.height;

      if (p > 0) {
        width >>= logUVx;
        height >>= logUVy;
      }

      int width4 = width >> 2;
      int pitch4 = pitch >> 2;

      if (chroma == false && p > 0) {
        cudaMemcpy2DAsync(pDstA, pitch * sizeof(pixel_t), pSrcA, pitch * sizeof(pixel_t), width * sizeof(pixel_t), height, cudaMemcpyDeviceToDevice, planeStream);
        DEBUG_SYNC;
        continue;
      }

      const vpixel_t* pPrv1A = (const vpixel_t*)(prv1->GetReadPtr(planes[p]));
      const vpixel_t* pFwd1A = (const vpixel_t*)(fwd1->GetReadPtr(planes[p]));
      const vpixel_t* pPrv2A = (radius >= 2) ? (const vpixel_t*)(prv2->GetReadPtr(planes[p])) : nullptr;
      const vpixel_t* pFwd2A = (radius >= 2) ? (const vpixel_t*)(fwd2->GetReadPtr(planes[p])) : nullptr;
      const vpixel_t* pPrv1B = (p > 0 && uvSamePitch) ? (const vpixel_t*)(prv1->GetReadPtr(planes[p + 1])) : nullptr;
      const vpixel_t* pFwd1B = (p > 0 && uvSamePitch) ? (const vpixel_t*)(fwd1->GetReadPtr(planes[p + 1])) : nullptr;
      const vpixel_t* pPrv2B = (radius >= 2 && p > 0 && uvSamePitch) ? (const vpixel_t*)(prv2->GetReadPtr(planes[p + 1])) : nullptr;
      const vpixel_t* pFwd2B = (radius >= 2 && p > 0 && uvSamePitch) ? (const vpixel_t*)(fwd2->GetReadPtr(planes[p + 1])) : nullptr;

      float* pSadA = sad + p * radius * 2;
      float* pSadB = sad + (p+1) * radius * 2;
      dim3 sadblocks(height, radius * 2, (p > 0 && uvSamePitch) ? 2 : 1);
      kl_calculate_sad << <sadblocks, CALC_SAD_THREADS, 0, planeStream >> > (pSrcA, pSrcB, pPrv1A, pPrv1B, pFwd1A, pFwd1B, pPrv2A, pPrv2B, pFwd2A, pFwd2B, width4, height, pitch4, pSadA, pSadB);
      DEBUG_SYNC;

      //DataDebug<float> dsad(pSad, 2, env);
      //dsad.Show();


      float fsc = (float)scenechange * width * height;

      dim3 threads(32, 16);
      dim3 blocks(nblocks(width4, threads.x), nblocks(height, threads.y), (p > 0 && uvSamePitch) ? 2 : 1);

      switch (radius) {
      case 1:
        kl_binomial_temporal_soften_1 << <blocks, threads, 0, planeStream >> > (
          pDstA, pDstB, pSrcA, pSrcB, pPrv1A, pPrv1B, pFwd1A, pFwd1B, pSadA, pSadB, fsc, width4, height, pitch4);
        DEBUG_SYNC;
        break;
      case 2:
        kl_binomial_temporal_soften_2 << <blocks, threads, 0, planeStream >> > (
          pDstA, pDstB, pSrcA, pSrcB, pPrv1A, pPrv1B, pFwd1A, pFwd1B, pPrv2A, pPrv2B, pFwd2A, pFwd2B, pSadA, pSadB, fsc, width4, height, pitch4);
        DEBUG_SYNC;
        break;
      }
    }
    if (planeEvent) planeEvent->finPlane();

    return dst;
  }

public:
  KBinomialTemporalSoften(PClip _child, int radius, int scenechange, bool chroma, IScriptEnvironment* env_)
    : CUDAFilterBase(_child, env_)
    , radius(radius)
    , scenechange(scenechange)
    , chroma(chroma)
    , logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
    , logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
  {
    PNeoEnv env = env_;

    if (radius != 1 && radius != 2) {
      env->ThrowError("[KBinomialTemporalSoften] radius��1��2�ł�");
    }
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    if (!IS_CUDA) {
      env->ThrowError("[KBinomialTemporalSoften] CUDA�t���[������͂��Ă�������");
    }

    int pixelSize = vi.ComponentSize();
    switch (pixelSize) {
    case 1:
      return Proc<uint8_t>(n, env);
    case 2:
      return Proc<uint16_t>(n, env);
    default:
      env->ThrowError("[KBinomialTemporalSoften] Unsupported pixel format");
    }
    return PVideoFrame();
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return new KBinomialTemporalSoften(
      args[0].AsClip(),
      args[1].AsInt(),
      args[2].AsInt(0),
      args[3].AsBool(true),
      env);
  }
};

template <typename pixel_t>
__global__ void kl_copy_boarder1(
  pixel_t* pDst, const pixel_t* __restrict__ pSrc, int width, int height, int pitch
)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  switch (blockIdx.y) {
  case 0: // top
    if (x < width) pDst[x] = pSrc[x];
    break;
  case 1: // left
    if (x < height) pDst[x * pitch] = pSrc[x * pitch];
    break;
  case 2: // bottom
    if (x < width) pDst[x + (height - 1) * pitch] = pSrc[x + (height - 1) * pitch];
    break;
  case 3: // right
    if (x < height) pDst[(width - 1) + x * pitch] = pSrc[(width - 1) + x * pitch];
    break;
  }
}

template <typename pixel_t>
__global__ void kl_copy_boarder1_v(
  pixel_t* pDst, const pixel_t* __restrict__ pSrc, int width, int height, int pitch
)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  switch (blockIdx.y) {
  case 0: // top
    if (x < width) pDst[x] = pSrc[x];
    break;
  case 1: // bottom
    if (x < width) pDst[x + (height - 1) * pitch] = pSrc[x + (height - 1) * pitch];
    break;
  }
}

template <typename pixel_t, typename Horizontal, typename Vertical>
__global__ void kl_box3x3_filter(
  pixel_t* pDstA, pixel_t* pDstB,
  const pixel_t* __restrict__ pSrcA, const pixel_t* __restrict__ pSrcB,
  int width, int height, int pitch
)
{
  Horizontal horizontal;
  Vertical vertical;

  const int x = threadIdx.x + blockDim.x * blockIdx.x;
  const int y = threadIdx.y + blockDim.y * blockIdx.y;
  pixel_t* pDst = (blockIdx.z) ? pDstB : pDstA;
  const pixel_t* __restrict__ pSrc = (blockIdx.z) ? pSrcB : pSrcA;

  if (x < width && y < height) {
      const int s = pSrc[x + y * pitch];
      int tmp = s;
      if (1 <= x && x < width - 1 && 1 <= y && y < height - 1) {
          tmp = vertical(
              horizontal(pSrc[x - 1 + (y - 1) * pitch], pSrc[x + (y - 1) * pitch], pSrc[x + 1 + (y - 1) * pitch]),
              horizontal(pSrc[x - 1 + y * pitch], s, pSrc[x + 1 + y * pitch]),
              horizontal(pSrc[x - 1 + (y + 1) * pitch], pSrc[x + (y + 1) * pitch], pSrc[x + 1 + (y + 1) * pitch]));
          tmp = clamp(tmp, 0, (sizeof(pixel_t) == 1) ? 255 : 65535);
      }
      pDst[x + y * pitch] = tmp;
  }
}

struct RG11Horizontal {
  __device__ int operator()(int a, int b, int c) {
    return a + b * 2 + c;
  }
};
struct RG11Vertical {
  __device__ int operator()(int a, int b, int c) {
    return (a + b * 2 + c + 8) >> 4;
  }
};

struct RG20Horizontal {
  __device__ int operator()(int a, int b, int c) {
    return a + b + c;
  }
};
struct RG20Vertical {
  __device__ int operator()(int a, int b, int c) {
    return (a + b + c + 4) / 9;
  }
};

template<typename T, typename CompareAndSwap>
__device__ void dev_sort_8elem(T& a0, T& a1, T& a2, T& a3, T& a4, T& a5, T& a6, T& a7)
{
  CompareAndSwap cas;

  // Batcher's odd-even mergesort
  // 8�v�f�Ȃ�19comparison�Ȃ̂ōŏ��̃\�[�e�B���O�l�b�g���[�N�ɂȂ���ۂ�
  cas(a0, a1);
  cas(a2, a3);
  cas(a4, a5);
  cas(a6, a7);

  cas(a0, a2);
  cas(a1, a3);
  cas(a4, a6);
  cas(a5, a7);

  cas(a1, a2);
  cas(a5, a6);

  cas(a0, a4);
  cas(a1, a5);
  cas(a2, a6);
  cas(a3, a7);

  cas(a2, a4);
  cas(a3, a5);

  cas(a1, a2);
  cas(a3, a4);
  cas(a5, a6);
}

template<typename T, typename CompareAndSwap>
__device__ void dev_sort_9elem(T& a0, T& a1, T& a2, T& a3, T& a4, T& a5, T& a6, T& a7, T& a8)
{
  CompareAndSwap cas;

  // 25 comparison

  cas(a0, a1);
  cas(a3, a4);
  cas(a6, a7);

  cas(a1, a2);
  cas(a4, a5);
  cas(a7, a8);

  cas(a0, a1);
  cas(a3, a4);
  cas(a6, a7);

  cas(a0, a3);
  cas(a1, a4);
  cas(a2, a5);

  cas(a3, a6);
  cas(a4, a7);
  cas(a5, a8);

  cas(a0, a3);
  cas(a1, a4);
  cas(a2, a5);

  cas(a1, a3);
  cas(a5, a7);
  cas(a2, a6);
  cas(a4, a6);
  cas(a2, a4);
  cas(a2, a3);
  cas(a5, a6);
}

struct IntCompareAndSwap {
  __device__ void operator()(int& a, int& b) {
    int a_ = min(a, b);
    int b_ = max(a, b);
    a = a_; b = b_;
  }
};

template <typename pixel_t, int N>
__global__ void kl_rg_clip(
  pixel_t* pDstA, pixel_t* pDstB,
  const pixel_t* __restrict__ pSrcA, const pixel_t* __restrict__ pSrcB,
  int width, int height, int pitch)
{
  const int x = threadIdx.x + blockDim.x * blockIdx.x;
  const int y = threadIdx.y + blockDim.y * blockIdx.y;
  pixel_t* pDst = (blockIdx.z) ? pDstB : pDstA;
  const pixel_t* __restrict__ pSrc = (blockIdx.z) ? pSrcB : pSrcA;
  if (x < width && y < height) {
    const int s = pSrc[x + y * pitch];
    int tmp = s;
    if (1 <= x && x < width-1 && 1 <= y && y < height - 1) {
      int a0 = pSrc[x - 1 + (y - 1) * pitch];
      int a1 = pSrc[x + (y - 1) * pitch];
      int a2 = pSrc[x + 1 + (y - 1) * pitch];
      int a3 = pSrc[x - 1 + y * pitch];
      int a4 = pSrc[x + 1 + y * pitch];
      int a5 = pSrc[x - 1 + (y + 1) * pitch];
      int a6 = pSrc[x + (y + 1) * pitch];
      int a7 = pSrc[x + 1 + (y + 1) * pitch];

      dev_sort_8elem<int, IntCompareAndSwap>(a0, a1, a2, a3, a4, a5, a6, a7);

      switch (N) {
      case 1: // 1st
          tmp = clamp(s, a0, a7);
          break;
      case 2: // 2nd
          tmp = clamp(s, a1, a6);
          break;
      case 3: // 3rd
          tmp = clamp(s, a2, a5);
          break;
      case 4: // 4th
          tmp = clamp(s, a3, a4);
          break;
      }
    }
    pDst[x + y * pitch] = tmp;
  }
}

template <typename pixel_t, int N>
__global__ void kl_repair_clip(
  pixel_t* pDstA, pixel_t* pDstB,
  const pixel_t* __restrict__ pSrcA, const pixel_t* __restrict__ pSrcB,
  const pixel_t* __restrict__ pRefA, const pixel_t* __restrict__ pRefB,
  int width, int height, int pitch)
{
  const int x = threadIdx.x + blockDim.x * blockIdx.x;
  const int y = threadIdx.y + blockDim.y * blockIdx.y;
  pixel_t* pDst = (blockIdx.z) ? pDstB : pDstA;
  const pixel_t* __restrict__ pSrc = (blockIdx.z) ? pSrcB : pSrcA;
  const pixel_t* __restrict__ pRef = (blockIdx.z) ? pRefB : pRefA;

  if (x < width && y < height) {
    const int s = pSrc[x + y * pitch];
    int tmp = s;
    if (1 <= x && x < width - 1 && 1 <= y && y < height - 1) {
        int a0 = pRef[x - 1 + (y - 1) * pitch];
        int a1 = pRef[x + (y - 1) * pitch];
        int a2 = pRef[x + 1 + (y - 1) * pitch];
        int a3 = pRef[x - 1 + y * pitch];
        int a4 = s;
        int a5 = pRef[x + 1 + y * pitch];
        int a6 = pRef[x - 1 + (y + 1) * pitch];
        int a7 = pRef[x + (y + 1) * pitch];
        int a8 = pRef[x + 1 + (y + 1) * pitch];

        dev_sort_9elem<int, IntCompareAndSwap>(a0, a1, a2, a3, a4, a5, a6, a7, a8);

        switch (N) {
        case 1: // 1st
            tmp = clamp(s, a0, a8);
            break;
        case 2: // 2nd
            tmp = clamp(s, a1, a7);
            break;
        case 3: // 3rd
            tmp = clamp(s, a2, a6);
            break;
        case 4: // 4th
            tmp = clamp(s, a3, a5);
            break;
        }
    }
    pDst[x + y * pitch] = tmp;
  }
}

class KRemoveGrain : public CUDAFilterBase {

  int mode;
  int modeU;
  int modeV;

  int logUVx;
  int logUVy;

  template <typename pixel_t>
  PVideoFrame Proc(int n, PNeoEnv env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;
    cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());

    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame dst = env->NewVideoFrame(vi);

    int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };
    int modes[] = { mode, modeU, modeV };

    auto planeEvent = CreateEventPlanes();

    const bool uvSamePitch = src->GetPitch(PLANAR_U) == src->GetPitch(PLANAR_V);

    for (int p = 0; p < ((uvSamePitch && modeU > 0 && modeV > 0) ? 2 : 3); p++) {
      int mode = modes[p];
      if (mode == -1) continue;

      const auto planeStream = (cudaStream_t)GetDeviceStreamPlane(p);
      const pixel_t* pSrc = reinterpret_cast<const pixel_t*>(src->GetReadPtr(planes[p]));
      const pixel_t* pSrc1 = (p > 0 && uvSamePitch) ? reinterpret_cast<const pixel_t*>(src->GetReadPtr(planes[p + 1])) : nullptr;
      pixel_t* pDst = reinterpret_cast<pixel_t*>(dst->GetWritePtr(planes[p]));
      pixel_t* pDst1 = (p > 0 && uvSamePitch) ? reinterpret_cast<pixel_t*>(dst->GetWritePtr(planes[p + 1])) : nullptr;

      int pitch = src->GetPitch(planes[p]) / sizeof(pixel_t);
      int width = vi.width;
      int height = vi.height;

      if (p > 0) {
        width >>= logUVx;
        height >>= logUVy;
      }

      int width4 = width >> 2;
      int pitch4 = pitch >> 2;

      if (mode == 0) {
        //launch_elementwise<vpixel_t, vpixel_t, CopyFunction<vpixel_t>>(
        //  (vpixel_t*)pDst, (const vpixel_t*)pSrc, width4, height, pitch4, planeStream);
        cudaMemcpy2DAsync(pDst, pitch * sizeof(pixel_t), pSrc, pitch * sizeof(pixel_t), width * sizeof(pixel_t), height, cudaMemcpyDeviceToDevice, planeStream);
        DEBUG_SYNC;
        continue;
      }

      dim3 threads(32, 16);
      dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y), (p > 0 && uvSamePitch) ? 2 : 1);

      switch (mode) {
      case 1:
        // Clips the pixel with the minimum and maximum of the 8 neighbour pixels.
        kl_rg_clip<pixel_t, 1>
          << <blocks, threads, 0, planeStream >> > (pDst, pDst1, pSrc, pSrc1, width, height, pitch);
        DEBUG_SYNC;
        break;
      case 2:
        // Clips the pixel with the second minimum and maximum of the 8 neighbour pixels
        kl_rg_clip<pixel_t, 2>
          << <blocks, threads, 0, planeStream >> > (pDst, pDst1, pSrc, pSrc1, width, height, pitch);
        DEBUG_SYNC;
        break;
      case 3:
        // Clips the pixel with the third minimum and maximum of the 8 neighbour pixels.
        kl_rg_clip<pixel_t, 3>
          << <blocks, threads, 0, planeStream >> > (pDst, pDst1, pSrc, pSrc1, width, height, pitch);
        DEBUG_SYNC;
        break;
      case 4:
        // Clips the pixel with the fourth minimum and maximum of the 8 neighbour pixels, which is equivalent to a median filter.
        kl_rg_clip<pixel_t, 4>
          << <blocks, threads, 0, planeStream >> > (pDst, pDst1, pSrc, pSrc1, width, height, pitch);
        DEBUG_SYNC;
        break;
      case 11:
      case 12:
        // [1 2 1] horizontal and vertical kernel blur
        kl_box3x3_filter<pixel_t, RG11Horizontal, RG11Vertical>
          << <blocks, threads, 0, planeStream >> > (pDst, pDst1, pSrc, pSrc1, width, height, pitch);
        DEBUG_SYNC;
        break;

      case 20:
        // Averages the 9 pixels ([1 1 1] horizontal and vertical blur)
        kl_box3x3_filter<pixel_t, RG20Horizontal, RG20Vertical>
          << <blocks, threads, 0, planeStream >> > (pDst, pDst1, pSrc, pSrc1, width, height, pitch);
        DEBUG_SYNC;
        break;

      default:
        env->ThrowError("[KRemoveGrain] Unsupported mode %d", modes[p]);
      }
    }
    if (planeEvent) planeEvent->finPlane();

    return dst;
  }

public:
  KRemoveGrain(PClip _child, int mode, int modeU, int modeV, IScriptEnvironment* env_)
    : CUDAFilterBase(_child, env_)
    , mode(mode)
    , modeU(modeU)
    , modeV(modeV)
    , logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
    , logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
  {
    PNeoEnv env = env_;

    int modes[] = { mode, modeU, modeV };
    for (int p = 0; p < 3; ++p) {
      switch (modes[p]) {
      case -1:
      case 0:
      case 1:
      case 2:
      case 3:
      case 4:
      case 11:
      case 12:
      case 20:
        break;
      default:
        env->ThrowError("[KRemoveGrain] Unsupported mode %d", modes[p]);
      }
    }
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    if (!IS_CUDA) {
      env->ThrowError("[KRemoveGrain] CUDA�t���[������͂��Ă�������");
    }

    int pixelSize = vi.ComponentSize();
    switch (pixelSize) {
    case 1:
      return Proc<uint8_t>(n, env);
    case 2:
      return Proc<uint16_t>(n, env);
    default:
      env->ThrowError("[KRemoveGrain] Unsupported pixel format");
    }
    return PVideoFrame();
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env) {
    int mode = args[1].AsInt(2);
    int modeU = args[2].AsInt(mode);
    int modeV = args[3].AsInt(modeU);
    return new KRemoveGrain(
      args[0].AsClip(),
      mode,
      modeU,
      modeV,
      env);
  }
};

class KRepair : public CUDAFilterBase {

  PClip refclip;

  int mode;
  int modeU;
  int modeV;

  int logUVx;
  int logUVy;

  template <typename pixel_t>
  PVideoFrame Proc(int n, PNeoEnv env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;
    cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());

    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame ref = refclip->GetFrame(n, env);
    PVideoFrame dst = env->NewVideoFrame(vi);

    int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };
    int modes[] = { mode, modeU, modeV };
    auto planeEvent = CreateEventPlanes();

    const bool uvSamePitch = src->GetPitch(PLANAR_U) == src->GetPitch(PLANAR_V);

    for (int p = 0; p < ((uvSamePitch && modeU > 0 && modeV > 0) ? 2 : 3); p++) {
      int mode = modes[p];
      if (mode == -1) continue;

      const auto planeStream = (cudaStream_t)GetDeviceStreamPlane(p);
      const pixel_t* pSrc = reinterpret_cast<const pixel_t*>(src->GetReadPtr(planes[p]));
      const pixel_t* pSrc1 = (p > 0 && uvSamePitch) ? reinterpret_cast<const pixel_t*>(src->GetReadPtr(planes[p + 1])) : nullptr;
      const pixel_t* pRef = reinterpret_cast<const pixel_t*>(ref->GetReadPtr(planes[p]));
      const pixel_t* pRef1 = (p > 0 && uvSamePitch) ? reinterpret_cast<const pixel_t*>(ref->GetReadPtr(planes[p + 1])) : nullptr;
      pixel_t* pDst = reinterpret_cast<pixel_t*>(dst->GetWritePtr(planes[p]));
      pixel_t* pDst1 = (p > 0 && uvSamePitch) ? reinterpret_cast<pixel_t*>(dst->GetWritePtr(planes[p + 1])) : nullptr;

      int pitch = src->GetPitch(planes[p]) / sizeof(pixel_t);
      int width = vi.width;
      int height = vi.height;

      if (p > 0) {
        width >>= logUVx;
        height >>= logUVy;
      }

      int width4 = width >> 2;
      int pitch4 = pitch >> 2;

      if (mode == 0) {
        //launch_elementwise<vpixel_t, vpixel_t, CopyFunction<vpixel_t>>(
        //  (vpixel_t*)pDst, (const vpixel_t*)pSrc, width4, height, pitch4, planeStream);
        cudaMemcpy2DAsync(pDst, pitch * sizeof(pixel_t), pSrc, pitch * sizeof(pixel_t), width * sizeof(pixel_t), height, cudaMemcpyDeviceToDevice, planeStream);
        DEBUG_SYNC;
        continue;
      }

      dim3 threads(32, 16);
      dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y), (p > 0 && uvSamePitch) ? 2 : 1);

      switch (mode) {
      case 1:
        // Clips the source pixel with the Nth minimum and maximum found on the 3�~3-pixel square from the reference clip.
        kl_repair_clip<pixel_t, 1>
          << <blocks, threads, 0, planeStream >> > (pDst, pDst1, pSrc, pSrc1, pRef, pRef1, width, height, pitch);
        DEBUG_SYNC;
        break;
      case 2:
        // Clips the source pixel with the Nth minimum and maximum found on the 3�~3-pixel square from the reference clip.
        kl_repair_clip<pixel_t, 2>
          << <blocks, threads, 0, planeStream >> > (pDst, pDst1, pSrc, pSrc1, pRef, pRef1, width, height, pitch);
        DEBUG_SYNC;
        break;
      case 3:
        // Clips the source pixel with the Nth minimum and maximum found on the 3�~3-pixel square from the reference clip.
        kl_repair_clip<pixel_t, 3>
          << <blocks, threads, 0, planeStream >> > (pDst, pDst1, pSrc, pSrc1, pRef, pRef1, width, height, pitch);
        DEBUG_SYNC;
        break;
      case 4:
        // Clips the source pixel with the Nth minimum and maximum found on the 3�~3-pixel square from the reference clip.
        kl_repair_clip<pixel_t, 4>
          << <blocks, threads, 0, planeStream >> > (pDst, pDst1, pSrc, pSrc1, pRef, pRef1, width, height, pitch);
        DEBUG_SYNC;
        break;

      default:
        env->ThrowError("[KRepair] Unsupported mode %d", modes[p]);
      }
    }
    if (planeEvent) planeEvent->finPlane();

    return dst;
  }

public:
  KRepair(PClip _child, PClip refclip, int mode, int modeU, int modeV, IScriptEnvironment* env_)
    : CUDAFilterBase(_child, env_)
    , refclip(refclip)
    , mode(mode)
    , modeU(modeU)
    , modeV(modeV)
    , logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
    , logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
  {
    PNeoEnv env = env_;

    int modes[] = { mode, modeU, modeV };
    for (int p = 0; p < 3; ++p) {
      switch (modes[p]) {
      case -1:
      case 0:
      case 1:
      case 2:
      case 3:
      case 4:
        break;
      default:
        env->ThrowError("[KRepair] Unsupported mode %d", modes[p]);
      }
    }
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    if (!IS_CUDA) {
      env->ThrowError("[KRepair] CUDA�t���[������͂��Ă�������");
    }

    int pixelSize = vi.ComponentSize();
    switch (pixelSize) {
    case 1:
      return Proc<uint8_t>(n, env);
    case 2:
      return Proc<uint16_t>(n, env);
    default:
      env->ThrowError("[KRepair] Unsupported pixel format");
    }
    return PVideoFrame();
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env) {
    int mode = args[2].AsInt(2);
    int modeU = args[3].AsInt(mode);
    int modeV = args[4].AsInt(modeU);
    return new KRepair(
      args[0].AsClip(),
      args[1].AsClip(),
      mode,
      modeU,
      modeV,
      env);
  }
};

template <typename vpixel_t>
__global__ void kl_vertical_cleaner_median(
  vpixel_t* dst0, vpixel_t* dst1,
  const vpixel_t* __restrict__ pSrc0, const vpixel_t* __restrict__ pSrc1,
  int width4, int height, int pitch4
)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  vpixel_t* dst = (blockIdx.z) ? dst1 : dst0;
  const vpixel_t* __restrict__ pSrc = (blockIdx.z) ? pSrc1 : pSrc0;

  if (x < width4 && y < height) {
    int4 tmp = to_int(pSrc[x + y * pitch4]);
    if (1 <= y && y < height - 1) {
        int4 a = to_int(pSrc[x + (y - 1) * pitch4]);
        int4 b = tmp;
        int4 c = to_int(pSrc[x + (y + 1) * pitch4]);
        tmp = min(max(min(a, b), c), max(a, b));
    }
    dst[x + y * pitch4] = VHelper<vpixel_t>::cast_to(tmp);
  }
}

class KVerticalCleaner : public CUDAFilterBase {

  int mode;
  int modeU;
  int modeV;

  int logUVx;
  int logUVy;

  template <typename pixel_t>
  PVideoFrame Proc(int n, PNeoEnv env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;
    cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());

    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame dst = env->NewVideoFrame(vi);

    int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };
    int modes[] = { mode, modeU, modeV };

    auto planeEvent = CreateEventPlanes();

    const bool uvSamePitch = src->GetPitch(PLANAR_U) == src->GetPitch(PLANAR_V);

    for (int p = 0; p < ((uvSamePitch && modeU > 0 && modeV > 0) ? 2 : 3); p++) {
      int mode = modes[p];
      if (mode == -1) continue;

      const auto planeStream = (cudaStream_t)GetDeviceStreamPlane(p);
      const pixel_t* pSrc = reinterpret_cast<const pixel_t*>(src->GetReadPtr(planes[p]));
      const pixel_t* pSrc1 = (p > 0 && uvSamePitch) ? reinterpret_cast<const pixel_t*>(src->GetReadPtr(planes[p + 1])) : nullptr;
      pixel_t* pDst = reinterpret_cast<pixel_t*>(dst->GetWritePtr(planes[p]));
      pixel_t* pDst1 = (p > 0 && uvSamePitch) ? reinterpret_cast<pixel_t*>(dst->GetWritePtr(planes[p + 1])) : nullptr;

      int pitch = src->GetPitch(planes[p]) / sizeof(pixel_t);
      int width = vi.width;
      int height = vi.height;

      if (p > 0) {
        width >>= logUVx;
        height >>= logUVy;
      }

      int width4 = width >> 2;
      int pitch4 = pitch >> 2;

      if (mode == 0) {
        //launch_elementwise<vpixel_t, vpixel_t, CopyFunction<vpixel_t>>(
        //  (vpixel_t*)pDst, (const vpixel_t*)pSrc, width4, height, pitch4, planeStream);
        cudaMemcpy2DAsync(pDst, pitch * sizeof(pixel_t), pSrc, pitch * sizeof(pixel_t), width * sizeof(pixel_t), height, cudaMemcpyDeviceToDevice, planeStream);
        DEBUG_SYNC;
        continue;
      }

      dim3 threads(32, 16);
      dim3 blocks(nblocks(width4, threads.x), nblocks(height, threads.y), (p > 0 && uvSamePitch) ? 2 : 1);

      switch (mode) {
      case 1:
        // vertical median
        kl_vertical_cleaner_median<vpixel_t>
          << <blocks, threads, 0, planeStream >> > (
          (vpixel_t*)(pDst), (vpixel_t*)(pDst1), (const vpixel_t*)(pSrc), (const vpixel_t*)(pSrc1), width4, height, pitch4);
        DEBUG_SYNC;
        break;

      default:
        env->ThrowError("[KVerticalCleaner] Unsupported mode %d", modes[p]);
      }
#if 0
      {
        dim3 threads(256);
        dim3 blocks(nblocks(width, threads.x), 2);
        kl_copy_boarder1_v << <blocks, threads, 0, planeStream >> > (pDst, pSrc, width, height, pitch);
        DEBUG_SYNC;
      }
#endif
    }
    if (planeEvent) planeEvent->finPlane();

    return dst;
  }

public:
  KVerticalCleaner(PClip _child, int mode, int modeU, int modeV, IScriptEnvironment* env_)
    : CUDAFilterBase(_child, env_)
    , mode(mode)
    , modeU(modeU)
    , modeV(modeV)
    , logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
    , logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
  {
    PNeoEnv env = env_;

    int modes[] = { mode, modeU, modeV };
    for (int p = 0; p < 3; ++p) {
      switch (modes[p]) {
      case 0:
      case 1:
        break;
      default:
        env->ThrowError("[KVerticalCleaner] Unsupported mode %d", modes[p]);
      }
    }
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    if (!IS_CUDA) {
      env->ThrowError("[KVerticalCleaner] CUDA�t���[������͂��Ă�������");
    }

    int pixelSize = vi.ComponentSize();
    switch (pixelSize) {
    case 1:
      return Proc<uint8_t>(n, env);
    case 2:
      return Proc<uint16_t>(n, env);
    default:
      env->ThrowError("[KVerticalCleaner] Unsupported pixel format");
    }
    return PVideoFrame();
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env) {
    int mode = args[1].AsInt(2);
    int modeU = args[2].AsInt(mode);
    int modeV = args[3].AsInt(modeU);
    return new KVerticalCleaner(
      args[0].AsClip(),
      mode,
      modeU,
      modeV,
      env);
  }
};

class KGaussResize : public CUDAFilterBase {
  std::unique_ptr<ResamplingProgram> progVert;
  std::unique_ptr<ResamplingProgram> progVertUV;
  std::unique_ptr<ResamplingProgram> progHori;
  std::unique_ptr<ResamplingProgram> progHoriUV;

  bool chroma;
  int logUVx;
  int logUVy;

  template <typename pixel_t>
  PVideoFrame Proc(int n, PNeoEnv env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;
    cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());

    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame tmp = env->NewVideoFrame(vi);
    PVideoFrame dst = env->NewVideoFrame(vi);

    const int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };

    typedef void(*RESAMPLE_V)(
        const vpixel_t* src0, const vpixel_t* src1, vpixel_t* dst0, vpixel_t* dst1,
      int src_pitch4, int dst_pitch4,
      int width4, int height,
      const int* offset, const float* coef, cudaStream_t);

    typedef void(*RESAMPLE_H)(
      const pixel_t* src0, const pixel_t* src1, pixel_t* dst0, pixel_t* dst1,
      int src_pitch, int dst_pitch,
      int width, int height,
      const int* offset, const float* coef, cudaStream_t);

    auto planeEvent = CreateEventPlanes();

    const bool uvSamePitch = src->GetPitch(PLANAR_U) == src->GetPitch(PLANAR_V);

    for (int p = 0; p < (chroma ? (uvSamePitch ? 2 : 3) : 1); ++p) {
      const auto planeStream = (cudaStream_t)GetDeviceStreamPlane(p);
      const pixel_t* srcptr = reinterpret_cast<const pixel_t*>(src->GetReadPtr(planes[p]));
      const pixel_t* srcptr1 = (p > 0 && uvSamePitch) ? reinterpret_cast<const pixel_t*>(src->GetReadPtr(planes[p + 1])) : nullptr;
      pixel_t* tmpptr = reinterpret_cast<pixel_t*>(tmp->GetWritePtr(planes[p]));
      pixel_t* tmpptr1 = (p > 0 && uvSamePitch) ? reinterpret_cast<pixel_t*>(tmp->GetWritePtr(planes[p + 1])) : nullptr;
      pixel_t* dstptr = reinterpret_cast<pixel_t*>(dst->GetWritePtr(planes[p]));
      pixel_t* dstptr1 = (p > 0 && uvSamePitch) ? reinterpret_cast<pixel_t*>(dst->GetWritePtr(planes[p + 1])) : nullptr;

      int pitch = src->GetPitch(planes[p]) / sizeof(pixel_t);
      int pitch4 = pitch / 4;

      ResamplingProgram* progV = (p == 0) ? progVert.get() : progVertUV.get();
      ResamplingProgram* progH = (p == 0) ? progHori.get() : progHoriUV.get();

      int width = vi.width;
      int height = vi.height;

      if (p > 0) {
        width >>= logUVx;
        height >>= logUVy;
      }

      int width4 = width / 4;

      RESAMPLE_V resample_v;
      RESAMPLE_H resample_h;

      switch (progV->filter_size) {
      case 8:
        resample_v = launch_resmaple_v<vpixel_t, 8>;
        break;
      case 9:
        resample_v = launch_resmaple_v<vpixel_t, 9>;
        break;
      default:
        env->ThrowError("[KGaussResize] Unexpected filter_size %d", progV->filter_size);
        break;
      }
      switch (progH->filter_size) {
      case 8:
        resample_h = launch_resmaple_h<pixel_t, 8>;
        break;
      case 9:
        resample_h = launch_resmaple_h<pixel_t, 9>;
        break;
      default:
        env->ThrowError("[KGaussResize] Unexpected filter_size %d", progV->filter_size);
        break;
      }

      resample_v(
        (const vpixel_t*)srcptr, (const vpixel_t*)srcptr1, (vpixel_t*)tmpptr, (vpixel_t*)tmpptr1, pitch4, pitch4, width4, height,
        progV->pixel_offset->GetData(env), progV->pixel_coefficient_float->GetData(env), planeStream);
      DEBUG_SYNC;

      resample_h(
        tmpptr, tmpptr1, dstptr, dstptr1, pitch, pitch, width, height,
        progH->pixel_offset->GetData(env), progH->pixel_coefficient_float->GetData(env), planeStream);
      DEBUG_SYNC;
    }
    if (planeEvent) planeEvent->finPlane();

    return dst;
  }

public:
  KGaussResize(PClip _child, double p, bool chroma, IScriptEnvironment* env_)
    : CUDAFilterBase(_child, env_)
    , logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
    , logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
    , chroma(chroma)
  {
    PNeoEnv env = env_;

    int width = vi.width;
    int height = vi.height;

    // QTGMC�ɍ��킹��
    double crop_width = width + 0.0001;
    double crop_height = height + 0.0001;

    int divUVx = (1 << logUVx);
    int divUVy = (1 << logUVy);

    progVert = GaussianFilter(p).GetResamplingProgram(height, 0, crop_height, height, env);
    progVertUV = GaussianFilter(p).GetResamplingProgram(height / divUVy, 0, crop_height / divUVy, height / divUVy, env);
    progHori = GaussianFilter(p).GetResamplingProgram(width, 0, crop_width, width, env);
    progHoriUV = GaussianFilter(p).GetResamplingProgram(width / divUVx, 0, crop_width / divUVx, width / divUVx, env);
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    if (!IS_CUDA) {
      env->ThrowError("[KGaussResize] CUDA�t���[������͂��Ă�������");
    }

    int pixelSize = vi.ComponentSize();
    switch (pixelSize) {
    case 1:
      return Proc<uint8_t>(n, env);
    case 2:
      return Proc<uint16_t>(n, env);
    default:
      env->ThrowError("[KGaussResize] Unsupported pixel format");
    }
    return PVideoFrame();
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return new KGaussResize(
      args[0].AsClip(),
      args[1].AsFloat(),
      args[2].AsBool(true),
      env);
  }
};

class KMasktoolFilterBase : public CUDAFilterBase {
protected:
  int numChilds;
  PClip childs[4];

  int Y, U, V;
  int logUVx;
  int logUVy;

  virtual void ProcPlane(int p, uint8_t* pDstA, uint8_t* pDstB,
    const uint8_t* pSrc0A, const uint8_t* pSrc0B,
    const uint8_t* pSrc1A, const uint8_t* pSrc1B,
    const uint8_t* pSrc2A, const uint8_t* pSrc2B,
    const uint8_t* pSrc3A, const uint8_t* pSrc3B,
    int width, int height, int pitch, cudaStream_t stream) { }

  virtual void ProcPlane(int p, uint16_t* pDstA, uint16_t* pDstB,
    const uint16_t* pSrc0A, const uint16_t* pSrc0B,
    const uint16_t* pSrc1A, const uint16_t* pSrc1B,
    const uint16_t* pSrc2A, const uint16_t* pSrc2B,
    const uint16_t* pSrc3A, const uint16_t* pSrc3B,
    int width, int height, int pitch, cudaStream_t stream) { }

  template <typename pixel_t>
  PVideoFrame Proc(int n, PNeoEnv env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;
	cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());

    PVideoFrame src0 = childs[0]->GetFrame(n, env);
    PVideoFrame src1 = (numChilds >= 2) ? childs[1]->GetFrame(n, env) : PVideoFrame();
    PVideoFrame src2 = (numChilds >= 3) ? childs[2]->GetFrame(n, env) : PVideoFrame();
    PVideoFrame src3 = (numChilds >= 4) ? childs[3]->GetFrame(n, env) : PVideoFrame();
    PVideoFrame dst = env->NewVideoFrame(vi);

    const int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };
    const int modes[] = { Y, U, V };

    auto planeEvent = CreateEventPlanes();

    const bool uvSameProc = src0->GetPitch(PLANAR_U) == src0->GetPitch(PLANAR_V) && modes[1] == 3 && modes[1] == 3;

    for (int p = 0; p < (uvSameProc ? 2 : 3); p++) {
      const int mode = modes[p];
      if (mode == 1) continue;

      const auto planeStream = (cudaStream_t)GetDeviceStreamPlane(p);
      const pixel_t* pSrc0A =                    reinterpret_cast<const pixel_t*>(src0->GetReadPtr(planes[p]));
      const pixel_t* pSrc1A = (numChilds >= 2) ? reinterpret_cast<const pixel_t*>(src1->GetReadPtr(planes[p])) : nullptr;
      const pixel_t* pSrc2A = (numChilds >= 3) ? reinterpret_cast<const pixel_t*>(src2->GetReadPtr(planes[p])) : nullptr;
      const pixel_t* pSrc3A = (numChilds >= 4) ? reinterpret_cast<const pixel_t*>(src3->GetReadPtr(planes[p])) : nullptr;
      const pixel_t* pSrc0B = (                  p > 0 && uvSameProc) ? reinterpret_cast<const pixel_t*>(src0->GetReadPtr(planes[p + 1])) : nullptr;
      const pixel_t* pSrc1B = (numChilds >= 2 && p > 0 && uvSameProc) ? reinterpret_cast<const pixel_t*>(src1->GetReadPtr(planes[p + 1])) : nullptr;
      const pixel_t* pSrc2B = (numChilds >= 3 && p > 0 && uvSameProc) ? reinterpret_cast<const pixel_t*>(src2->GetReadPtr(planes[p + 1])) : nullptr;
      const pixel_t* pSrc3B = (numChilds >= 4 && p > 0 && uvSameProc) ? reinterpret_cast<const pixel_t*>(src3->GetReadPtr(planes[p + 1])) : nullptr;
      pixel_t* pDstA = reinterpret_cast<pixel_t*>(dst->GetWritePtr(planes[p]));
      pixel_t* pDstB = (p > 0 && uvSameProc) ? reinterpret_cast<pixel_t*>(dst->GetWritePtr(planes[p + 1])) : nullptr;

      const int pitch = src0->GetPitch(planes[p]) / sizeof(pixel_t);
      int width = vi.width;
      int height = vi.height;

      if (p > 0) {
        width >>= logUVx;
        height >>= logUVy;
      }

      const int width4 = width >> 2;
      const int pitch4 = pitch >> 2;

      if (mode == 0) {
        //launch_elementwise<vpixel_t, vpixel_t, CopyFunction<vpixel_t>>(
        //  (vpixel_t*)pDst, (const vpixel_t*)pSrc0, width4, height, pitch4, planeStream);
        cudaMemcpy2DAsync(pDstA, pitch * sizeof(pixel_t), pSrc0A, pitch * sizeof(pixel_t), width * sizeof(pixel_t), height, cudaMemcpyDeviceToDevice, planeStream);
        DEBUG_SYNC;
        continue;
      }

      switch (modes[p]) {
      case 2:
        //launch_elementwise<vpixel_t, vpixel_t, CopyFunction<vpixel_t>>(
        //  (vpixel_t*)pDst, (const vpixel_t*)pSrc0, width4, height, pitch4, planeStream);
        cudaMemcpy2DAsync(pDstA, pitch * sizeof(pixel_t), pSrc0A, pitch * sizeof(pixel_t), width * sizeof(pixel_t), height, cudaMemcpyDeviceToDevice, planeStream);
        DEBUG_SYNC;
        break;
      case 3:
        ProcPlane(p, pDstA, pDstB, pSrc0A, pSrc0B, pSrc1A, pSrc1B, pSrc2A, pSrc2B, pSrc3A, pSrc3B, width, height, pitch, planeStream);
        break;
      case 4:
        //launch_elementwise<vpixel_t, vpixel_t, CopyFunction<vpixel_t>>(
        //  (vpixel_t*)pDst, (const vpixel_t*)pSrc1, width4, height, pitch4, planeStream);
        cudaMemcpy2DAsync(pDstA, pitch * sizeof(pixel_t), pSrc1A, pitch * sizeof(pixel_t), width * sizeof(pixel_t), height, cudaMemcpyDeviceToDevice, planeStream);
        DEBUG_SYNC;
        break;
      case 5:
        //launch_elementwise<vpixel_t, vpixel_t, CopyFunction<vpixel_t>>(
        //  (vpixel_t*)pDst, (const vpixel_t*)pSrc2, width4, height, pitch4, planeStream);
        cudaMemcpy2DAsync(pDstA, pitch * sizeof(pixel_t), pSrc2A, pitch * sizeof(pixel_t), width * sizeof(pixel_t), height, cudaMemcpyDeviceToDevice, planeStream);
        DEBUG_SYNC;
        break;
      }
    }
    if (planeEvent) planeEvent->finPlane();

    return dst;
  }

public:
  KMasktoolFilterBase(PClip child, int Y, int U, int V, IScriptEnvironment* env_)
    : CUDAFilterBase(child, env_)
    , numChilds(1)
    , Y(Y), U(U), V(V)
    , logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
    , logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
  {
    childs[0] = child;
  }

  KMasktoolFilterBase(PClip child0, PClip child1, int Y, int U, int V, IScriptEnvironment* env_)
    : CUDAFilterBase(child0, env_)
    , numChilds(2)
    , Y(Y), U(U), V(V)
    , logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
    , logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
  {
    childs[0] = child0;
    childs[1] = child1;
  }

  KMasktoolFilterBase(PClip child0, PClip child1, PClip child2, int Y, int U, int V, IScriptEnvironment* env_)
    : CUDAFilterBase(child0, env_)
    , numChilds(3)
    , Y(Y), U(U), V(V)
    , logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
    , logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
  {
    childs[0] = child0;
    childs[1] = child1;
    childs[2] = child2;
  }

  KMasktoolFilterBase(PClip child0, PClip child1, PClip child2, PClip child3, int Y, int U, int V, IScriptEnvironment* env_)
    : CUDAFilterBase(child0, env_)
    , numChilds(4)
    , Y(Y), U(U), V(V)
    , logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
    , logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
  {
    childs[0] = child0;
    childs[1] = child1;
    childs[2] = child2;
    childs[3] = child3;
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    if (!IS_CUDA) {
      env->ThrowError("[KMasktoolFilterBase] CUDA�t���[������͂��Ă�������");
    }

    int pixelSize = vi.ComponentSize();
    switch (pixelSize) {
    case 1:
      return Proc<uint8_t>(n, env);
    case 2:
      return Proc<uint16_t>(n, env);
    default:
      env->ThrowError("[KMasktoolFilterBase] Unsupported pixel format");
    }
    return PVideoFrame();
  }
};

template <typename vpixel_t, typename Op>
__global__ void kl_makediff(
  vpixel_t* pDstA, vpixel_t* pDstB,
  const vpixel_t* __restrict__ pA0,const vpixel_t* __restrict__ pA1,
  const vpixel_t* __restrict__ pB0,const vpixel_t* __restrict__ pB1,
  int width4, int height, int pitch4, int range_half
)
{
  Op op;

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  vpixel_t* pDst = (blockIdx.z) ? pDstB : pDstA;
  const vpixel_t* __restrict__ pA = (blockIdx.z) ? pA1 : pA0;
  const vpixel_t* __restrict__ pB = (blockIdx.z) ? pB1 : pB0;

  if (x < width4 && y < height) {
    auto tmp = op(to_int(pA[x + y * pitch4]), to_int(pB[x + y * pitch4]), range_half);
    tmp = clamp(tmp, 0, (sizeof(pA[0].x) == 1) ? 255 : 65535);
    pDst[x + y * pitch4] = VHelper<vpixel_t>::cast_to(tmp);
  }
}

struct MakeDiffOp {
  __device__ int4 operator()(int4 a, int4 b, int range_half) {
    return a - b + range_half;
  }
};
struct AddDiffOp {
  __device__ int4 operator()(int4 a, int4 b, int range_half) {
    return a + b + (-range_half);
  }
};

template <typename Op>
class KMakeDiff : public KMasktoolFilterBase
{
protected:

  virtual void ProcPlane(int p, uint8_t* pDstA, uint8_t* pDstB,
      const uint8_t* pSrc0A, const uint8_t* pSrc0B,
      const uint8_t* pSrc1A, const uint8_t* pSrc1B,
      const uint8_t* pSrc2A, const uint8_t* pSrc2B,
      const uint8_t* pSrc3A, const uint8_t* pSrc3B,
    int width, int height, int pitch, cudaStream_t stream)
  {
    ProcPlane_(pDstA, pDstB, pSrc0A, pSrc0B, pSrc1A, pSrc1B, pSrc2A, pSrc2B, width, height, pitch, stream);
  }

  virtual void ProcPlane(int p, uint16_t* pDstA, uint16_t* pDstB,
      const uint16_t* pSrc0A, const uint16_t* pSrc0B,
      const uint16_t* pSrc1A, const uint16_t* pSrc1B,
      const uint16_t* pSrc2A, const uint16_t* pSrc2B,
      const uint16_t* pSrc3A, const uint16_t* pSrc3B,
    int width, int height, int pitch, cudaStream_t stream)
  {
    ProcPlane_(pDstA, pDstB, pSrc0A, pSrc0B, pSrc1A, pSrc1B, pSrc2A, pSrc2B, width, height, pitch, stream);
  }

  template <typename pixel_t>
  void ProcPlane_(pixel_t* pDstA, pixel_t* pDstB,
    const pixel_t* pSrc0A, const pixel_t* pSrc0B,
    const pixel_t* pSrc1A, const pixel_t* pSrc1B,
    const pixel_t* pSrc2A, const pixel_t* pSrc2B,
    int width, int height, int pitch, cudaStream_t stream)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;

    int width4 = width / 4;
    int pitch4 = pitch / 4;

    int bits = vi.BitsPerComponent();

    dim3 threads(32, 16);
    dim3 blocks(nblocks(width4, threads.x), nblocks(height, threads.y), pDstB ? 2 : 1);
    kl_makediff<vpixel_t, Op> << <blocks, threads, 0, stream >> > (
      (vpixel_t*)pDstA, (vpixel_t*)pDstB, (const vpixel_t*)pSrc0A, (const vpixel_t*)pSrc0B, (const vpixel_t*)pSrc1A, (const vpixel_t*)pSrc1B, width4, height, pitch4, 1 << (bits - 1));
    DEBUG_SYNC;
  }

public:
  KMakeDiff(PClip src0, PClip src1, int y, int u, int v, IScriptEnvironment* env_)
    : KMasktoolFilterBase(src0, src1, y, u, v, env_)
  { }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return new KMakeDiff(
      args[0].AsClip(),
      args[1].AsClip(),
      args[2].AsInt(3),
      args[3].AsInt(1),
      args[4].AsInt(1),
      env);
  }
};


// �㉺2���C�����͏����ēn��
template <typename vpixel_t, typename F>
__global__ void kl_box5_v_and_border(
  vpixel_t* pDstA, vpixel_t* pDstB,
  const vpixel_t* __restrict__ pSrcA, const vpixel_t* __restrict__ pSrcB,
  int width4, int height, int pitch4
)
{
  F f;

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  vpixel_t* pDst = (blockIdx.z) ? pDstB : pDstA;
  const vpixel_t* __restrict__ pSrc = (blockIdx.z) ? pSrcB : pSrcA;

  if (x < width4 && y < height) {
    auto v2 = to_int(pSrc[x + (y + 0) * pitch4]);
    auto v0 = (y - 2 >= 0)     ? to_int(pSrc[x + (y - 2) * pitch4]) : v2;
    auto v1 = (y - 1 >= 0)     ? to_int(pSrc[x + (y - 1) * pitch4]) : v2;
    auto v3 = (y + 1 < height) ? to_int(pSrc[x + (y + 1) * pitch4]) : v2;
    auto v4 = (y + 2 < height) ? to_int(pSrc[x + (y + 2) * pitch4]) : v2;

    auto tmp = f(v0, v1, v2, v3, v4);
    pDst[x + y * pitch4] = VHelper<vpixel_t>::cast_to(tmp);
  }
}

struct Min5 {
  __device__ int4 operator()(int4 a, int4 b, int4 c, int4 d, int4 e) {
    return min(min(min(a, b), min(c, d)), e);
  }
};
struct Max5 {
  __device__ int4 operator()(int4 a, int4 b, int4 c, int4 d, int4 e) {
    return max(max(max(a, b), max(c, d)), e);
  }
};

template <typename F>
class KXpandVerticalX2 : public KMasktoolFilterBase
{
protected:

  virtual void ProcPlane(int p, uint8_t* pDstA, uint8_t* pDstB,
      const uint8_t* pSrc0A, const uint8_t* pSrc0B,
      const uint8_t* pSrc1A, const uint8_t* pSrc1B,
      const uint8_t* pSrc2A, const uint8_t* pSrc2B,
      const uint8_t* pSrc3A, const uint8_t* pSrc3B,
    int width, int height, int pitch, cudaStream_t stream)
  {
    ProcPlane_(pDstA, pDstB, pSrc0A, pSrc0B, pSrc1A, pSrc1B, pSrc2A, pSrc2B, width, height, pitch, stream);
  }

  virtual void ProcPlane(int p, uint16_t* pDstA, uint16_t* pDstB,
      const uint16_t* pSrc0A, const uint16_t* pSrc0B,
      const uint16_t* pSrc1A, const uint16_t* pSrc1B,
      const uint16_t* pSrc2A, const uint16_t* pSrc2B,
      const uint16_t* pSrc3A, const uint16_t* pSrc3B,
    int width, int height, int pitch, cudaStream_t stream)
  {
    ProcPlane_(pDstA, pDstB, pSrc0A, pSrc0B, pSrc1A, pSrc1B, pSrc2A, pSrc2B, width, height, pitch, stream);
  }

  template <typename pixel_t>
  void ProcPlane_(pixel_t* pDstA, pixel_t* pDstB,
      const pixel_t* pSrc0A, const pixel_t* pSrc0B,
      const pixel_t* pSrc1A, const pixel_t* pSrc1B,
      const pixel_t* pSrc2A, const pixel_t* pSrc2B,
    int width, int height, int pitch, cudaStream_t stream)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;

    int width4 = width / 4;
    int pitch4 = pitch / 4;
    dim3 threads(32, 16);
    dim3 blocks(nblocks(width4, threads.x), nblocks(height, threads.y), pDstB ? 2 : 1);
    kl_box5_v_and_border<vpixel_t, F> << <blocks, threads, 0, stream >> > (
        (vpixel_t*)pDstA, (vpixel_t*)pDstB, (const vpixel_t*)pSrc0A, (const vpixel_t*)pSrc0B, width4, height, pitch4);
    DEBUG_SYNC;
  }

public:
  KXpandVerticalX2(PClip src0, int y, int u, int v, IScriptEnvironment* env_)
    : KMasktoolFilterBase(src0, y, u, v, env_)
  { }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return new KXpandVerticalX2<F>(
      args[0].AsClip(),
      args[1].AsInt(3),
      args[2].AsInt(1),
      args[3].AsInt(1),
      env);
  }
};

template <typename vpixel_t, typename F>
__global__ void kl_logic1(
  vpixel_t* pDstA, vpixel_t* pDstB,
  const vpixel_t* __restrict__ pSrcA, const vpixel_t* __restrict__ pSrcB,
  int width4, int height, int pitch4
)
{
  F f;
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  vpixel_t* pDst = (blockIdx.z) ? pDstB : pDstA;
  const vpixel_t* __restrict__ pSrc = (blockIdx.z) ? pSrcB : pSrcA;

  if (x < width4 && y < height) {
    int4 src = to_int(pSrc[x + y * pitch4]);
    auto tmp = f(src);
    pDst[x + y * pitch4] = VHelper<vpixel_t>::cast_to(tmp);
  }
}

template <typename vpixel_t, typename F>
__global__ void kl_logic2(
  vpixel_t* pDstA, vpixel_t* pDstB,
  const vpixel_t* __restrict__ pSrc0A, const vpixel_t* __restrict__ pSrc0B,
  const vpixel_t* __restrict__ pSrc1A, const vpixel_t* __restrict__ pSrc1B,
  int width4, int height, int pitch4
)
{
  F f;
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  vpixel_t* pDst = (blockIdx.z) ? pDstB : pDstA;
  const vpixel_t* __restrict__ pSrc0 = (blockIdx.z) ? pSrc0B : pSrc0A;
  const vpixel_t* __restrict__ pSrc1 = (blockIdx.z) ? pSrc1B : pSrc1A;

  if (x < width4 && y < height) {
    int4 src0 = to_int(pSrc0[x + y * pitch4]);
    int4 src1 = to_int(pSrc1[x + y * pitch4]);
    auto tmp = f(src0, src1);
    pDst[x + y * pitch4] = VHelper<vpixel_t>::cast_to(tmp);
  }
}

template <typename vpixel_t, typename F>
__global__ void kl_logic3(
  vpixel_t* pDstA, vpixel_t* pDstB,
  const vpixel_t* __restrict__ pSrc0A, const vpixel_t* __restrict__ pSrc0B,
  const vpixel_t* __restrict__ pSrc1A, const vpixel_t* __restrict__ pSrc1B,
  const vpixel_t* __restrict__ pSrc2A, const vpixel_t* __restrict__ pSrc2B,
  int width4, int height, int pitch4
)
{
  F f;
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  vpixel_t* pDst = (blockIdx.z) ? pDstB : pDstA;
  const vpixel_t* __restrict__ pSrc0 = (blockIdx.z) ? pSrc0B : pSrc0A;
  const vpixel_t* __restrict__ pSrc1 = (blockIdx.z) ? pSrc1B : pSrc1A;
  const vpixel_t* __restrict__ pSrc2 = (blockIdx.z) ? pSrc2B : pSrc2A;

  if (x < width4 && y < height) {
    int4 src0 = to_int(pSrc0[x + y * pitch4]);
    int4 src1 = to_int(pSrc1[x + y * pitch4]);
    int4 src2 = to_int(pSrc2[x + y * pitch4]);
    auto tmp = f(src0, src1, src2);
    pDst[x + y * pitch4] = VHelper<vpixel_t>::cast_to(tmp);
  }
}

struct LogicMin {
  __device__ int4 operator()(int4 a, int4 b) {
    return min(a, b);
  }
};
struct LogicMax {
  __device__ int4 operator()(int4 a, int4 b) {
    return max(a, b);
  }
};

template <typename F>
class KLogic1 : public KMasktoolFilterBase
{
protected:

  virtual void ProcPlane(int p, uint8_t* pDstA, uint8_t* pDstB,
      const uint8_t* pSrc0A, const uint8_t* pSrc0B,
      const uint8_t* pSrc1A, const uint8_t* pSrc1B,
      const uint8_t* pSrc2A, const uint8_t* pSrc2B,
      const uint8_t* pSrc3A, const uint8_t* pSrc3B,
    int width, int height, int pitch, cudaStream_t stream)
  {
    ProcPlane_(pDstA, pDstB, pSrc0A, pSrc0B, pSrc1A, pSrc1B, pSrc2A, pSrc2B, width, height, pitch, stream);
  }

  virtual void ProcPlane(int p, uint16_t* pDstA, uint16_t* pDstB,
      const uint16_t* pSrc0A, const uint16_t* pSrc0B,
      const uint16_t* pSrc1A, const uint16_t* pSrc1B,
      const uint16_t* pSrc2A, const uint16_t* pSrc2B,
      const uint16_t* pSrc3A, const uint16_t* pSrc3B,
    int width, int height, int pitch, cudaStream_t stream)
  {
    ProcPlane_(pDstA, pDstB, pSrc0A, pSrc0B, pSrc1A, pSrc1B, pSrc2A, pSrc2B, width, height, pitch, stream);
  }

  template <typename pixel_t>
  void ProcPlane_(pixel_t* pDstA, pixel_t* pDstB,
    const pixel_t* pSrc0A, const pixel_t* pSrc0B,
    const pixel_t* pSrc1A, const pixel_t* pSrc1B,
    const pixel_t* pSrc2A, const pixel_t* pSrc2B,
    int width, int height, int pitch, cudaStream_t stream)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;

    int width4 = width / 4;
    int pitch4 = pitch / 4;

    dim3 threads(32, 16);
    dim3 blocks(nblocks(width4, threads.x), nblocks(height, threads.y), pDstB ? 2 : 1);
    kl_logic1<vpixel_t, F> << <blocks, threads, 0, stream >> > (
      (vpixel_t*)pDstA, (vpixel_t*)pDstB, (const vpixel_t*)pSrc0A, (const vpixel_t*)pSrc0B, width4, height, pitch4);
    DEBUG_SYNC;
  }

public:
  KLogic1(PClip src0, int y, int u, int v, IScriptEnvironment* env_)
    : KMasktoolFilterBase(src0, y, u, v, env_)
  { }
};

template <typename F>
class KLogic2 : public KMasktoolFilterBase
{
protected:

  virtual void ProcPlane(int p, uint8_t* pDstA, uint8_t* pDstB,
      const uint8_t* pSrc0A, const uint8_t* pSrc0B,
      const uint8_t* pSrc1A, const uint8_t* pSrc1B,
      const uint8_t* pSrc2A, const uint8_t* pSrc2B,
      const uint8_t* pSrc3A, const uint8_t* pSrc3B,
    int width, int height, int pitch, cudaStream_t stream)
  {
    ProcPlane_(pDstA, pDstB, pSrc0A, pSrc0B, pSrc1A, pSrc1B, pSrc2A, pSrc2B, width, height, pitch, stream);
  }

  virtual void ProcPlane(int p, uint16_t* pDstA, uint16_t* pDstB,
      const uint16_t* pSrc0A, const uint16_t* pSrc0B,
      const uint16_t* pSrc1A, const uint16_t* pSrc1B,
      const uint16_t* pSrc2A, const uint16_t* pSrc2B,
      const uint16_t* pSrc3A, const uint16_t* pSrc3B,
    int width, int height, int pitch, cudaStream_t stream)
  {
    ProcPlane_(pDstA, pDstB, pSrc0A, pSrc0B, pSrc1A, pSrc1B, pSrc2A, pSrc2B, width, height, pitch, stream);
  }

  template <typename pixel_t>
  void ProcPlane_(pixel_t* pDstA, pixel_t* pDstB,
    const pixel_t* pSrc0A, const pixel_t* pSrc0B,
    const pixel_t* pSrc1A, const pixel_t* pSrc1B,
    const pixel_t* pSrc2A, const pixel_t* pSrc2B,
    int width, int height, int pitch, cudaStream_t stream)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;

    int width4 = width / 4;
    int pitch4 = pitch / 4;

    dim3 threads(32, 16);
    dim3 blocks(nblocks(width4, threads.x), nblocks(height, threads.y), pDstB ? 2 : 1);
    kl_logic2<vpixel_t, F> << <blocks, threads, 0, stream >> > (
        (vpixel_t*)pDstA, (vpixel_t*)pDstB, (const vpixel_t*)pSrc0A, (const vpixel_t*)pSrc0B, (const vpixel_t*)pSrc1A, (const vpixel_t*)pSrc1B, width4, height, pitch4);
    DEBUG_SYNC;
  }

public:
  KLogic2(PClip src0, PClip src1, int y, int u, int v, IScriptEnvironment* env_)
    : KMasktoolFilterBase(src0, src1, y, u, v, env_)
  { }
};

template <typename F>
class KLogic3 : public KMasktoolFilterBase
{
protected:

  virtual void ProcPlane(int p, uint8_t* pDstA, uint8_t* pDstB,
      const uint8_t* pSrc0A, const uint8_t* pSrc0B,
      const uint8_t* pSrc1A, const uint8_t* pSrc1B,
      const uint8_t* pSrc2A, const uint8_t* pSrc2B,
      const uint8_t* pSrc3A, const uint8_t* pSrc3B,
    int width, int height, int pitch, PNeoEnv env)
  {
    ProcPlane_(pDstA, pDstB, pSrc0A, pSrc0B, pSrc1A, pSrc1B, pSrc2A, pSrc2B, width, height, pitch, env);
  }

  virtual void ProcPlane(int p, uint16_t* pDstA, uint16_t* pDstB,
      const uint16_t* pSrc0A, const uint16_t* pSrc0B,
      const uint16_t* pSrc1A, const uint16_t* pSrc1B,
      const uint16_t* pSrc2A, const uint16_t* pSrc2B,
      const uint16_t* pSrc3A, const uint16_t* pSrc3B,
    int width, int height, int pitch, PNeoEnv env)
  {
    ProcPlane_(pDstA, pDstB, pSrc0A, pSrc0B, pSrc1A, pSrc1B, pSrc2A, pSrc2B, width, height, pitch, env);
  }

  template <typename pixel_t>
  void ProcPlane_(pixel_t* pDstA, pixel_t* pDstB,
    const pixel_t* pSrc0A, const pixel_t* pSrc0B,
    const pixel_t* pSrc1A, const pixel_t* pSrc1B,
    const pixel_t* pSrc2A, const pixel_t* pSrc2B,
    int width, int height, int pitch, cudaStream_t stream)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;

    int width4 = width / 4;
    int pitch4 = pitch / 4;

    dim3 threads(32, 16);
    dim3 blocks(nblocks(width4, threads.x), nblocks(height, threads.y), pDstB ? 2 : 1);
    kl_logic3<vpixel_t, F> << <blocks, threads, 0, stream >> > (
        (vpixel_t*)pDstA, (vpixel_t*)pDstB,
        (const vpixel_t*)pSrc0A, (const vpixel_t*)pSrc0B,
        (const vpixel_t*)pSrc1A, (const vpixel_t*)pSrc1B,
        (const vpixel_t*)pSrc2A, (const vpixel_t*)pSrc2B, width4, height, pitch4);
    DEBUG_SYNC;
  }

public:
  KLogic3(PClip src0, PClip src1, PClip src2, int y, int u, int v, IScriptEnvironment* env_)
    : KMasktoolFilterBase(src0, src1, src2, y, u, v, env_)
  { }
};

static AVSValue __cdecl KLogicCreate(AVSValue args, void* user_data, IScriptEnvironment* env) {

  PClip src0 = args[0].AsClip();
  PClip src1 = args[1].AsClip();
  int Y = args[3].AsInt(3);
  int U = args[4].AsInt(1);
  int V = args[5].AsInt(1);

  auto modestr = std::string(args[2].AsString());
  if (modestr == "min") {
    return new KLogic2<LogicMin>(src0, src1, Y, U, V, env);
  }
  else if (modestr == "max") {
    return new KLogic2<LogicMax>(src0, src1, Y, U, V, env);
  }
  else {
    env->ThrowError("[KLogicCreate] Unsupported mode %s", modestr.c_str());
  }

  return AVSValue();
}

__device__ int dev_bobshimmerfixes_merge(
  int src, int diff, int c1, int c2, int scale, int maxval
)
{
  const int h = (128 << scale);
  diff = (diff < (129 << scale)) ? diff : (c1 < h) ? h : c1;
  diff = (diff > (127 << scale)) ? diff : (c2 > h) ? h : c2;
  int dst = src + diff - h;
  return clamp(dst, 0, maxval);
}

template <typename vpixel_t>
__global__ void kl_bobshimmerfixes_merge(
  vpixel_t* pDstA,                       vpixel_t* pDstB,
  const vpixel_t* __restrict__ pSrcA,    const vpixel_t* __restrict__ pSrcB,
  const vpixel_t* __restrict__ pDiffA,   const vpixel_t* __restrict__ pDiffB,
  const vpixel_t* __restrict__ pChoke1A, const vpixel_t* __restrict__ pChoke1B,
  const vpixel_t* __restrict__ pChoke2A, const vpixel_t* __restrict__ pChoke2B,
  int width4, int height, int pitch4,
  int scale
)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  vpixel_t* pDst = (blockIdx.z) ? pDstB : pDstA;
  const vpixel_t* __restrict__ pSrc = (blockIdx.z) ? pSrcB : pSrcA;
  const vpixel_t* __restrict__ pDiff = (blockIdx.z) ? pDiffB : pDiffA;
  const vpixel_t* __restrict__ pChoke1 = (blockIdx.z) ? pChoke1B : pChoke1A;
  const vpixel_t* __restrict__ pChoke2 = (blockIdx.z) ? pChoke2B : pChoke2A;

  if (x < width4 && y < height) {
    auto src = to_int(pSrc[x + y * pitch4]);
    auto diff = to_int(pDiff[x + y * pitch4]);
    auto c1 = to_int(pChoke1[x + y * pitch4]);
    auto c2 = to_int(pChoke2[x + y * pitch4]);
    int4 tmp = {
      dev_bobshimmerfixes_merge(src.x, diff.x, c1.x, c2.x, scale, ((sizeof(pSrc[0].x) == 1) ? 255 : 65535)),
      dev_bobshimmerfixes_merge(src.y, diff.y, c1.y, c2.y, scale, ((sizeof(pSrc[0].x) == 1) ? 255 : 65535)),
      dev_bobshimmerfixes_merge(src.z, diff.z, c1.z, c2.z, scale, ((sizeof(pSrc[0].x) == 1) ? 255 : 65535)),
      dev_bobshimmerfixes_merge(src.w, diff.w, c1.w, c2.w, scale, ((sizeof(pSrc[0].x) == 1) ? 255 : 65535))
    };
    pDst[x + y * pitch4] = VHelper<vpixel_t>::cast_to(tmp);
  }
}

class KTGMC_BobShimmerFixesMerge : public KMasktoolFilterBase
{
protected:

  virtual void ProcPlane(int p, uint8_t* pDstA, uint8_t* pDstB,
      const uint8_t* pSrc0A, const uint8_t* pSrc0B,
      const uint8_t* pSrc1A, const uint8_t* pSrc1B,
      const uint8_t* pSrc2A, const uint8_t* pSrc2B,
      const uint8_t* pSrc3A, const uint8_t* pSrc3B,
    int width, int height, int pitch, cudaStream_t stream)
  {
    ProcPlane_(pDstA, pDstB, pSrc0A, pSrc0B, pSrc1A, pSrc1B, pSrc2A, pSrc2B, pSrc3A, pSrc3B, width, height, pitch, stream);
  }

  virtual void ProcPlane(int p, uint16_t* pDstA, uint16_t* pDstB,
      const uint16_t* pSrc0A, const uint16_t* pSrc0B,
      const uint16_t* pSrc1A, const uint16_t* pSrc1B,
      const uint16_t* pSrc2A, const uint16_t* pSrc2B,
      const uint16_t* pSrc3A, const uint16_t* pSrc3B,
    int width, int height, int pitch, cudaStream_t stream)
  {
    ProcPlane_(pDstA, pDstB, pSrc0A, pSrc0B, pSrc1A, pSrc1B, pSrc2A, pSrc2B, pSrc3A, pSrc3B, width, height, pitch, stream);
  }

  template <typename pixel_t>
  void ProcPlane_(pixel_t* pDstA, pixel_t* pDstB,
    const pixel_t* pSrcA,    const pixel_t* pSrcB,
    const pixel_t* pDiffA,   const pixel_t* pDiffB,
    const pixel_t* pChoke1A, const pixel_t* pChoke1B,
    const pixel_t* pChoke2A, const pixel_t* pChoke2B,
    int width, int height, int pitch, cudaStream_t stream)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;

    int width4 = width / 4;
    int pitch4 = pitch / 4;

    int scale = vi.BitsPerComponent() - 8;

    dim3 threads(32, 16);
    dim3 blocks(nblocks(width4, threads.x), nblocks(height, threads.y), pDstB ? 2 : 1);
    kl_bobshimmerfixes_merge<vpixel_t> << <blocks, threads, 0, stream >> > ((vpixel_t*)pDstA, (vpixel_t*)pDstB,
      (const vpixel_t*)pSrcA, (const vpixel_t*)pSrcB,
      (const vpixel_t*)pDiffA, (const vpixel_t*)pDiffB,
      (const vpixel_t*)pChoke1A, (const vpixel_t*)pChoke1B,
      (const vpixel_t*)pChoke2A, (const vpixel_t*)pChoke2B,
      width4, height, pitch4, scale);
    DEBUG_SYNC;
  }

public:
  KTGMC_BobShimmerFixesMerge(PClip src, PClip diff, PClip choke1, PClip choke2, int y, int u, int v, IScriptEnvironment* env_)
    : KMasktoolFilterBase(src, diff, choke1, choke2, y, u, v, env_)
  { }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return new KTGMC_BobShimmerFixesMerge(
      args[0].AsClip(),
      args[1].AsClip(),
      args[2].AsClip(),
      args[3].AsClip(),
      args[4].AsInt(3),
      args[5].AsInt(1),
      args[6].AsInt(1),
      env);
  }
};

template <typename vpixel_t, typename F>
__global__ void kl_box3_v(
    vpixel_t* pDstA, vpixel_t* pDstB,
    const vpixel_t* __restrict__ pSrcA, const vpixel_t* __restrict__ pSrcB,
    int width4, int height, int pitch4
)
{
    F f;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    vpixel_t* pDst = (blockIdx.z) ? pDstB : pDstA;
    const vpixel_t* __restrict__ pSrc = (blockIdx.z) ? pSrcB : pSrcA;

    if (x < width4 && y < height) {
        auto v1 = to_int(pSrc[x + (y + 0) * pitch4]);
        auto v0 = (y == 0)          ? v1 : to_int(pSrc[x + (y - 1) * pitch4]);
        auto v2 = (y == height - 1) ? v1 : to_int(pSrc[x + (y + 1) * pitch4]);
        auto tmp = f(v0, v1, v2);
        pDst[x + y * pitch4] = VHelper<vpixel_t>::cast_to(tmp);
    }
}

struct Resharpen {
  __device__ int4 operator()(int4 a, int4 b, int4 c) {
    auto inpand = min(a, min(b, c));
    auto expand = max(a, max(b, c));
    return (inpand + expand + 1) >> 1;
  }
};

class KTGMC_VResharpen : public KMasktoolFilterBase
{
protected:

  virtual void ProcPlane(int p, uint8_t* pDstA, uint8_t* pDstB,
      const uint8_t* pSrc0A, const uint8_t* pSrc0B,
      const uint8_t* pSrc1A, const uint8_t* pSrc1B,
      const uint8_t* pSrc2A, const uint8_t* pSrc2B,
      const uint8_t* pSrc3A, const uint8_t* pSrc3B,
    int width, int height, int pitch, cudaStream_t stream)
  {
    ProcPlane_(pDstA, pDstB, pSrc0A, pSrc0B, width, height, pitch, stream);
  }

  virtual void ProcPlane(int p, uint16_t* pDstA, uint16_t* pDstB,
      const uint16_t* pSrc0A, const uint16_t* pSrc0B,
      const uint16_t* pSrc1A, const uint16_t* pSrc1B,
      const uint16_t* pSrc2A, const uint16_t* pSrc2B,
      const uint16_t* pSrc3A, const uint16_t* pSrc3B,
    int width, int height, int pitch, cudaStream_t stream)
  {
    ProcPlane_(pDstA, pDstB, pSrc0A, pSrc0B, width, height, pitch, stream);
  }

  template <typename pixel_t>
  void ProcPlane_(pixel_t* pDstA, pixel_t* pDstB,
    const pixel_t* pSrc0A, const pixel_t* pSrc0B, int width, int height, int pitch, cudaStream_t stream)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;

    int width4 = width / 4;
    int pitch4 = pitch / 4;

    {
      dim3 threads(32, 16);
      dim3 blocks(nblocks(width4, threads.x), nblocks(height, threads.y), pDstB ? 2 : 1);
      kl_box3_v<vpixel_t, Resharpen> << <blocks, threads, 0, stream >> > (
        (vpixel_t*)(pDstA), (vpixel_t*)(pDstB), (const vpixel_t*)(pSrc0A), (const vpixel_t*)(pSrc0B), width4, height, pitch4);
      DEBUG_SYNC;
    }
  }

public:
  KTGMC_VResharpen(PClip src, int y, int u, int v, IScriptEnvironment* env_)
    : KMasktoolFilterBase(src, y, u, v, env_)
  { }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return new KTGMC_VResharpen(
      args[0].AsClip(),
      args[1].AsInt(3),
      args[2].AsInt(1),
      args[3].AsInt(1),
      env);
  }
};

template <typename vpixel_t>
__global__ void kl_resharpen(
  vpixel_t* pDstA, vpixel_t* pDstB,
  const vpixel_t* __restrict__ pSrc0A, const vpixel_t* __restrict__ pSrc0B,
  const vpixel_t* __restrict__ pSrc1A, const vpixel_t* __restrict__ pSrc1B,
  int width4, int height, int pitch4,
  float sharpAdj
)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  vpixel_t* pDst = (blockIdx.z) ? pDstB : pDstA;
  const vpixel_t* __restrict__ pSrc0 = (blockIdx.z) ? pSrc0B : pSrc0A;
  const vpixel_t* __restrict__ pSrc1 = (blockIdx.z) ? pSrc1B : pSrc1A;

  if (x < width4 && y < height) {
    auto srcx = to_float(pSrc0[x + y * pitch4]);
    auto srcy = to_float(pSrc1[x + y * pitch4]);
    auto lut = srcx + (srcx - srcy) * sharpAdj;
    auto tmp = clamp(lut + 0.5f, 0, ((sizeof(pSrc0[0].x) == 1) ? 255 : 65535));
    pDst[x + y * pitch4] = VHelper<vpixel_t>::cast_to(to_int(tmp));
  }
}

// mt_lutxy("clamp_f x x y - sharpAdj * +")���v�Z
class KTGMC_Resharpen : public KMasktoolFilterBase
{
  float sharpAdj;
protected:

  virtual void ProcPlane(int p, uint8_t* pDstA, uint8_t* pDstB,
      const uint8_t* pSrc0A, const uint8_t* pSrc0B,
      const uint8_t* pSrc1A, const uint8_t* pSrc1B,
      const uint8_t* pSrc2A, const uint8_t* pSrc2B,
      const uint8_t* pSrc3A, const uint8_t* pSrc3B,
    int width, int height, int pitch, cudaStream_t stream)
  {
    ProcPlane_(pDstA, pDstB, pSrc0A, pSrc0B, pSrc1A, pSrc1B, width, height, pitch, stream);
  }

  virtual void ProcPlane(int p, uint16_t* pDstA, uint16_t* pDstB,
      const uint16_t* pSrc0A, const uint16_t* pSrc0B,
      const uint16_t* pSrc1A, const uint16_t* pSrc1B,
      const uint16_t* pSrc2A, const uint16_t* pSrc2B,
      const uint16_t* pSrc3A, const uint16_t* pSrc3B,
    int width, int height, int pitch, cudaStream_t stream)
  {
    ProcPlane_(pDstA, pDstB, pSrc0A, pSrc0B, pSrc1A, pSrc1B, width, height, pitch, stream);
  }

  template <typename pixel_t>
  void ProcPlane_(pixel_t* pDstA, pixel_t* pDstB,
    const pixel_t* pSrc0A, const pixel_t* pSrc0B,
    const pixel_t* pSrc1A, const pixel_t* pSrc1B,
    int width, int height, int pitch, cudaStream_t stream)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;

    int width4 = width / 4;
    int pitch4 = pitch / 4;

    dim3 threads(32, 16);
    dim3 blocks(nblocks(width4, threads.x), nblocks(height, threads.y), pDstB ? 2 : 1);
    kl_resharpen<vpixel_t> << <blocks, threads, 0, stream >> > ((vpixel_t*)pDstA, (vpixel_t*)pDstB,
      (const vpixel_t*)pSrc0A, (const vpixel_t*)pSrc0B, (const vpixel_t*)pSrc1A, (const vpixel_t*)pSrc1B,
      width4, height, pitch4, sharpAdj);
    DEBUG_SYNC;
  }

public:
  KTGMC_Resharpen(PClip src, PClip vresharp, float sharpAdj, int y, int u, int v, IScriptEnvironment* env_)
    : KMasktoolFilterBase(src, vresharp, y, u, v, env_)
    , sharpAdj(sharpAdj)
  { }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return new KTGMC_Resharpen(
      args[0].AsClip(),
      args[1].AsClip(),
      (float)args[2].AsFloat(),
      args[3].AsInt(3),
      args[4].AsInt(1),
      args[5].AsInt(1),
      env);
  }
};

__device__ int dev_limit_over_sharpen(
  int src, int ref, int compb, int compf, int osv
)
{
  auto tMin = min(ref, min(compb, compf));
  auto tMax = max(ref, max(compb, compf));
  return clamp(src, tMin - osv, tMax + osv);
}

template <typename vpixel_t>
__global__ void kl_limit_over_sharpen(
  vpixel_t* pDstA,                      vpixel_t* pDstB,
  const vpixel_t* __restrict__ pSrcA,   const vpixel_t* __restrict__ pSrcB,
  const vpixel_t* __restrict__ pRefA,   const vpixel_t* __restrict__ pRefB,
  const vpixel_t* __restrict__ pCompBA, const vpixel_t* __restrict__ pCompBB,
  const vpixel_t* __restrict__ pCompFA, const vpixel_t* __restrict__ pCompFB,
  int width4, int height, int pitch4,
  int osv
)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  vpixel_t* pDst = (blockIdx.z) ? pDstB : pDstA;
  const vpixel_t* __restrict__ pSrc = (blockIdx.z) ? pSrcB : pSrcA;
  const vpixel_t* __restrict__ pRef = (blockIdx.z) ? pRefB : pRefA;
  const vpixel_t* __restrict__ pCompB = (blockIdx.z) ? pCompBB : pCompBA;
  const vpixel_t* __restrict__ pCompF = (blockIdx.z) ? pCompFB : pCompFA;

  if (x < width4 && y < height) {
    auto src = to_int(pSrc[x + y * pitch4]);
    auto ref = to_int(pRef[x + y * pitch4]);
    auto compb = to_int(pCompB[x + y * pitch4]);
    auto compf = to_int(pCompF[x + y * pitch4]);
    int4 tmp = {
      dev_limit_over_sharpen(src.x, ref.x, compb.x, compf.x, osv),
      dev_limit_over_sharpen(src.y, ref.y, compb.y, compf.y, osv),
      dev_limit_over_sharpen(src.z, ref.z, compb.z, compf.z, osv),
      dev_limit_over_sharpen(src.w, ref.w, compb.w, compf.w, osv)
    };
    pDst[x + y * pitch4] = VHelper<vpixel_t>::cast_to(tmp);
  }
}

class KTGMC_LimitOverSharpen : public KMasktoolFilterBase
{
  int ovs;
protected:

  virtual void ProcPlane(int p, uint8_t* pDstA, uint8_t* pDstB,
      const uint8_t* pSrc0A, const uint8_t* pSrc0B,
      const uint8_t* pSrc1A, const uint8_t* pSrc1B,
      const uint8_t* pSrc2A, const uint8_t* pSrc2B,
      const uint8_t* pSrc3A, const uint8_t* pSrc3B,
    int width, int height, int pitch, cudaStream_t stream)
  {
    ProcPlane_(pDstA, pDstB, pSrc0A, pSrc0B, pSrc1A, pSrc1B, pSrc2A, pSrc2B, pSrc3A, pSrc3B, width, height, pitch, stream);
  }

  virtual void ProcPlane(int p, uint16_t* pDstA, uint16_t* pDstB,
      const uint16_t* pSrc0A, const uint16_t* pSrc0B,
      const uint16_t* pSrc1A, const uint16_t* pSrc1B,
      const uint16_t* pSrc2A, const uint16_t* pSrc2B,
      const uint16_t* pSrc3A, const uint16_t* pSrc3B,
    int width, int height, int pitch, cudaStream_t stream)
  {
    ProcPlane_(pDstA, pDstB, pSrc0A, pSrc0B, pSrc1A, pSrc1B, pSrc2A, pSrc2B, pSrc3A, pSrc3B, width, height, pitch, stream);
  }

  template <typename pixel_t>
  void ProcPlane_(pixel_t* pDstA, pixel_t* pDstB,
    const pixel_t* pSrcA, const pixel_t* pSrcB,
    const pixel_t* pRefA, const pixel_t* pRefB,
    const pixel_t* pCompBA, const pixel_t* pCompBB,
    const pixel_t* pCompFA, const pixel_t* pCompFB,
    int width, int height, int pitch, cudaStream_t stream)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;

    int width4 = width / 4;
    int pitch4 = pitch / 4;

    int scale = vi.BitsPerComponent() - 8;

    dim3 threads(32, 16);
    dim3 blocks(nblocks(width4, threads.x), nblocks(height, threads.y), pDstB ? 2 : 1);
    kl_limit_over_sharpen<vpixel_t> << <blocks, threads, 0, stream >> > ((vpixel_t*)pDstA, (vpixel_t*)pDstB,
      (const vpixel_t*)pSrcA, (const vpixel_t*)pSrcB, (const vpixel_t*)pRefA, (const vpixel_t*)pRefB,
      (const vpixel_t*)pCompBA, (const vpixel_t*)pCompBB, (const vpixel_t*)pCompFA, (const vpixel_t*)pCompFB,
      width4, height, pitch4, ovs << scale);
    DEBUG_SYNC;
  }

public:
  KTGMC_LimitOverSharpen(PClip src, PClip ref, PClip compb, PClip compf, int ovs, int y, int u, int v, IScriptEnvironment* env_)
    : KMasktoolFilterBase(src, ref, compb, compf, y, u, v, env_)
    , ovs(ovs)
  { }

#if LOG_PRINT
  virtual PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) {
    printf("KTGMC_LimitOverSharpen[CUDA]: N=%d\n", n);
    return KMasktoolFilterBase::GetFrame(n, env);
  }
#endif

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return new KTGMC_LimitOverSharpen(
      args[0].AsClip(),
      args[1].AsClip(),
      args[2].AsClip(),
      args[3].AsClip(),
      args[4].AsInt(0),
      args[5].AsInt(3),
      args[6].AsInt(1),
      args[7].AsInt(1),
      env);
  }
};

template <typename vpixel_t, bool uv>
__global__ void kl_to_full_range(
  vpixel_t* pDstA, vpixel_t* pDstB,
  const vpixel_t* __restrict__ pSrcA, const vpixel_t* __restrict__ pSrcB,
  int width4, int height, int pitch4
)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  vpixel_t* pDst = (blockIdx.z) ? pDstB : pDstA;
  const vpixel_t* __restrict__ pSrc = (blockIdx.z) ? pSrcB : pSrcA;

  if (x < width4 && y < height) {
    auto src = to_float(pSrc[x + y * pitch4]);
    float4 d;
    if (uv == false) {
      d = (src + (-16)) * (255.0f / 219.0f);
    } else {
      d = (src + (-128)) * (128.0f / 112.0f) + 128.0f;
    }
    auto tmp = clamp(d + 0.5f, 0, ((sizeof(pSrc[0].x) == 1) ? 255 : 65535));
    pDst[x + y * pitch4] = VHelper<vpixel_t>::cast_to(to_int(tmp));
  }
}

class KTGMC_ToFullRange : public KMasktoolFilterBase
{
protected:

  virtual void ProcPlane(int p, uint8_t* pDstA, uint8_t* pDstB,
      const uint8_t* pSrc0A, const uint8_t* pSrc0B,
      const uint8_t* pSrc1A, const uint8_t* pSrc1B,
      const uint8_t* pSrc2A, const uint8_t* pSrc2B,
      const uint8_t* pSrc3A, const uint8_t* pSrc3B,
    int width, int height, int pitch, cudaStream_t stream)
  {
    ProcPlane_(p, pDstA, pDstB, pSrc0A, pSrc0B, width, height, pitch, stream);
  }

  virtual void ProcPlane(int p, uint16_t* pDstA, uint16_t* pDstB,
      const uint16_t* pSrc0A, const uint16_t* pSrc0B,
      const uint16_t* pSrc1A, const uint16_t* pSrc1B,
      const uint16_t* pSrc2A, const uint16_t* pSrc2B,
      const uint16_t* pSrc3A, const uint16_t* pSrc3B,
    int width, int height, int pitch, cudaStream_t stream)
  {
    ProcPlane_(p, pDstA, pDstB, pSrc0A, pSrc0B, width, height, pitch, stream);
  }

  template <typename pixel_t>
  void ProcPlane_(int p, pixel_t* pDstA, pixel_t* pDstB,
    const pixel_t* pSrcA, const pixel_t* pSrcB, int width, int height, int pitch, cudaStream_t stream)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;

    int width4 = width / 4;
    int pitch4 = pitch / 4;

    dim3 threads(32, 16);
    dim3 blocks(nblocks(width4, threads.x), nblocks(height, threads.y), pDstB ? 2 : 1);
    if (p == 0) {
      kl_to_full_range<vpixel_t, false> << <blocks, threads, 0, stream >> > (
        (vpixel_t*)pDstA, (vpixel_t*)pDstB, (const vpixel_t*)pSrcA, (const vpixel_t*)pSrcB, width4, height, pitch4);
    }
    else {
      kl_to_full_range<vpixel_t, true> << <blocks, threads, 0, stream >> > (
          (vpixel_t*)pDstA, (vpixel_t*)pDstB, (const vpixel_t*)pSrcA, (const vpixel_t*)pSrcB, width4, height, pitch4);
    }
    DEBUG_SYNC;
  }

public:
  KTGMC_ToFullRange(PClip src, int y, int u, int v, IScriptEnvironment* env_)
    : KMasktoolFilterBase(src, y, u, v, env_)
  { }

#if LOG_PRINT
  virtual PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) {
    printf("KTGMC_ToFullRange[CUDA]: N=%d\n", n);
    return KMasktoolFilterBase::GetFrame(n, env);
  }
#endif

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return new KTGMC_ToFullRange(
      args[0].AsClip(),
      args[1].AsInt(3),
      args[2].AsInt(1),
      args[3].AsInt(1),
      env);
  }
};

__device__ float dev_lossless_proc(
  float x, float y, float half, float maxval
)
{
  float dst;
  if ((x - half) * (y - half) < 0) {
    dst = half;
  } else if (fabsf(x - half) < fabsf(y - half)) {
    dst = x;
  } else {
    dst = y;
  }
  return clamp(dst, 0.0f, maxval);
}

template <typename vpixel_t>
__global__ void kl_lossless_proc(
  vpixel_t* pDstA, vpixel_t* pDstB,
  const vpixel_t* __restrict__ pXA, const vpixel_t* __restrict__ pXB,
  const vpixel_t* __restrict__ pYA, const vpixel_t* __restrict__ pYB,
  int width4, int height, int pitch4,
  float half, float maxval
)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  vpixel_t* pDst = (blockIdx.z) ? pDstB : pDstA;
  const vpixel_t* __restrict__ pX = (blockIdx.z) ? pXB : pXA;
  const vpixel_t* __restrict__ pY = (blockIdx.z) ? pYB : pYA;

  if (x < width4 && y < height) {
    auto valx = to_float(pX[x + y * pitch4]);
    auto valy = to_float(pY[x + y * pitch4]);
    float4 tmp = {
      dev_lossless_proc(valx.x, valy.x, half, maxval),
      dev_lossless_proc(valx.y, valy.y, half, maxval),
      dev_lossless_proc(valx.z, valy.z, half, maxval),
      dev_lossless_proc(valx.w, valy.w, half, maxval)
    };
    pDst[x + y * pitch4] = VHelper<vpixel_t>::cast_to(tmp);
  }
}

class KTGMC_LosslessProc : public KMasktoolFilterBase
{
protected:

  virtual void ProcPlane(int p, uint8_t* pDstA, uint8_t* pDstB,
      const uint8_t* pSrc0A, const uint8_t* pSrc0B,
      const uint8_t* pSrc1A, const uint8_t* pSrc1B,
      const uint8_t* pSrc2A, const uint8_t* pSrc2B,
      const uint8_t* pSrc3A, const uint8_t* pSrc3B,
    int width, int height, int pitch, cudaStream_t stream)
  {
    ProcPlane_(pDstA, pDstB, pSrc0A, pSrc0B, pSrc1A, pSrc1B, width, height, pitch, stream);
  }

  virtual void ProcPlane(int p, uint16_t* pDstA, uint16_t* pDstB,
      const uint16_t* pSrc0A, const uint16_t* pSrc0B,
      const uint16_t* pSrc1A, const uint16_t* pSrc1B,
      const uint16_t* pSrc2A, const uint16_t* pSrc2B,
      const uint16_t* pSrc3A, const uint16_t* pSrc3B,
    int width, int height, int pitch, cudaStream_t stream)
  {
    ProcPlane_(pDstA, pDstB, pSrc0A, pSrc0B, pSrc1A, pSrc1B, width, height, pitch, stream);
  }

  template <typename pixel_t>
  void ProcPlane_(pixel_t* pDstA, pixel_t* pDstB,
    const pixel_t* pXA, const pixel_t* pXB, const pixel_t* pYA, const pixel_t* pYB,
    int width, int height, int pitch, cudaStream_t stream)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;

    int width4 = width / 4;
    int pitch4 = pitch / 4;

    int bits = vi.BitsPerComponent();
    float range_half = (float)(1 << (bits - 1));
    float maxval = (float)((1 << bits) - 1);

    dim3 threads(32, 16);
    dim3 blocks(nblocks(width4, threads.x), nblocks(height, threads.y), pDstB ? 2 : 1);
    kl_lossless_proc<vpixel_t> << <blocks, threads, 0, stream >> > ((vpixel_t*)pDstA, (vpixel_t*)pDstB,
      (const vpixel_t*)pXA, (const vpixel_t*)pXB, (const vpixel_t*)pYA,  (const vpixel_t*)pYB,
      width4, height, pitch4, range_half, maxval);
    DEBUG_SYNC;
  }

public:
  KTGMC_LosslessProc(PClip srcx, PClip srcy, int y, int u, int v, IScriptEnvironment* env_)
    : KMasktoolFilterBase(srcx, srcy, y, u, v, env_)
  { }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return new KTGMC_LosslessProc(
      args[0].AsClip(),
      args[1].AsClip(),
      args[2].AsInt(3),
      args[3].AsInt(1),
      args[4].AsInt(1),
      env);
  }
};

template <typename vpixel_t>
__global__ void kl_merge(
  vpixel_t* pDstA, vpixel_t* pDstB,
  const vpixel_t* __restrict__ pSrc0A, const vpixel_t* __restrict__ pSrc0B,
  const vpixel_t* __restrict__ pSrc1A, const vpixel_t* __restrict__ pSrc1B,
  int width4, int height, int pitch4,
  int weight, int invweight
)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  vpixel_t* pDst = (blockIdx.z) ? pDstB : pDstA;
  const vpixel_t* __restrict__ pSrc0 = (blockIdx.z) ? pSrc0B : pSrc0A;
  const vpixel_t* __restrict__ pSrc1 = (blockIdx.z) ? pSrc1B : pSrc1A;

  if (x < width4 && y < height) {
    auto src0 = to_int(pSrc0[x + y * pitch4]);
    auto src1 = to_int(pSrc1[x + y * pitch4]);
    int4 tmp = (src0 * invweight + src1 * weight + 16384) >> 15;
    pDst[x + y * pitch4] = VHelper<vpixel_t>::cast_to(tmp);
  }
}

class KMerge : public KMasktoolFilterBase
{
  float weight;
protected:

  virtual void ProcPlane(int p, uint8_t* pDstA, uint8_t* pDstB,
      const uint8_t* pSrc0A, const uint8_t* pSrc0B,
      const uint8_t* pSrc1A, const uint8_t* pSrc1B,
      const uint8_t* pSrc2A, const uint8_t* pSrc2B,
      const uint8_t* pSrc3A, const uint8_t* pSrc3B,
    int width, int height, int pitch, cudaStream_t stream)
  {
    ProcPlane_(pDstA, pDstB, pSrc0A, pSrc0B, pSrc1A, pSrc1B, width, height, pitch, stream);
  }

  virtual void ProcPlane(int p, uint16_t* pDstA, uint16_t* pDstB,
      const uint16_t* pSrc0A, const uint16_t* pSrc0B,
      const uint16_t* pSrc1A, const uint16_t* pSrc1B,
      const uint16_t* pSrc2A, const uint16_t* pSrc2B,
      const uint16_t* pSrc3A, const uint16_t* pSrc3B,
    int width, int height, int pitch, cudaStream_t stream)
  {
    ProcPlane_(pDstA, pDstB, pSrc0A, pSrc0B, pSrc1A, pSrc1B, width, height, pitch, stream);
  }

  template <typename pixel_t>
  void ProcPlane_(pixel_t* pDstA, pixel_t* pDstB,
    const pixel_t* pSrc0A, const pixel_t* pSrc0B, const pixel_t* pSrc1A, const pixel_t* pSrc1B,
    int width, int height, int pitch, cudaStream_t stream)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;

    int width4 = width / 4;
    int pitch4 = pitch / 4;

    dim3 threads(32, 16);
    dim3 blocks(nblocks(width4, threads.x), nblocks(height, threads.y), pDstB ? 2 : 1);
    kl_merge<vpixel_t> << <blocks, threads, 0, stream >> > ((vpixel_t*)pDstA, (vpixel_t*)pDstB,
      (const vpixel_t*)pSrc0A, (const vpixel_t*)pSrc0B, (const vpixel_t*)pSrc1A, (const vpixel_t*)pSrc1B,
      width4, height, pitch4, (int)(weight*32767.0f), 32767 - (int)(weight*32767.0f));
    DEBUG_SYNC;
  }

public:
  KMerge(PClip child, PClip clip, float weight, int y, int u, int v, IScriptEnvironment* env_)
    : KMasktoolFilterBase(child, clip, y, u, v, env_)
    , weight(weight)
  { }

  static AVSValue __cdecl CreateMerge(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return new KMerge(
      args[0].AsClip(),
      args[1].AsClip(),
      (float)args[2].AsFloat(0.5),
      3, 3, 3,
      env);
  }
  static AVSValue __cdecl CreateMergeLuma(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return new KMerge(
      args[0].AsClip(),
      args[1].AsClip(),
      (float)args[2].AsFloat(0.5),
      3, 1, 1,
      env);
  }
  static AVSValue __cdecl CreateMergeChroma(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return new KMerge(
      args[0].AsClip(),
      args[1].AsClip(),
      (float)args[2].AsFloat(0.5),
      1, 3, 3,
      env);
  }
};

__device__ float dev_tweak_search_clip(
  float repair, float bobbed, float blur, float scale, float invscale
)
{
  // ���e�������X�P�[��������͖̂ʓ|�Ȃ̂Œl�̕����X�P�[��������
  repair *= invscale;
  bobbed *= invscale;
  blur *= invscale;

  //float tweaked = ((repair + 3) < bobbed) ? (repair + 3) : ((repair - 3) > bobbed) ? (repair - 3) : bobbed;
  // �����ȏ����ɏ����ς�
  float tweaked = clamp(bobbed, repair - 3, repair + 3);

  float ret = ((blur + 7) < tweaked) ? (blur + 2) : ((blur - 7) > tweaked) ? (blur - 2) : (((blur * 51) + (tweaked * 49)) * (1.0f / 100.0f));

  return ret * scale;
}

template <typename vpixel_t>
__global__ void kl_tweak_search_clip(
  vpixel_t* pDstA, vpixel_t* pDstB,
  const vpixel_t* __restrict__ pRepairA, const vpixel_t* __restrict__ pRepairB,
  const vpixel_t* __restrict__ pBobbedA, const vpixel_t* __restrict__ pBobbedB,
  const vpixel_t* __restrict__ pBlurA,   const vpixel_t* __restrict__ pBlurB,
  int width4, int height, int pitch4,
  float scale, float invscale
)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  vpixel_t* pDst = (blockIdx.z) ? pDstB : pDstA;
  const vpixel_t* __restrict__ pRepair = (blockIdx.z) ? pRepairB : pRepairA;
  const vpixel_t* __restrict__ pBobbed = (blockIdx.z) ? pBobbedB : pBobbedA;
  const vpixel_t* __restrict__ pBlur = (blockIdx.z) ? pBlurB : pBlurA;

  if (x < width4 && y < height) {
    auto repair = to_float(pRepair[x + y * pitch4]);
    auto bobbed = to_float(pBobbed[x + y * pitch4]);
    auto blur = to_float(pBlur[x + y * pitch4]);
    float4 d = {
      dev_tweak_search_clip(repair.x, bobbed.x, blur.x, scale, invscale),
      dev_tweak_search_clip(repair.y, bobbed.y, blur.y, scale, invscale),
      dev_tweak_search_clip(repair.z, bobbed.z, blur.z, scale, invscale),
      dev_tweak_search_clip(repair.w, bobbed.w, blur.w, scale, invscale)
    };
    auto tmp = clamp(d + 0.5f, 0, ((sizeof(pRepair[0].x) == 1) ? 255 : 65535));
    pDst[x + y * pitch4] = VHelper<vpixel_t>::cast_to(to_int(tmp));
  }
}

class KTGMC_TweakSearchClip : public KMasktoolFilterBase
{
protected:

  virtual void ProcPlane(int p, uint8_t* pDstA, uint8_t* pDstB,
      const uint8_t* pSrc0A, const uint8_t* pSrc0B,
      const uint8_t* pSrc1A, const uint8_t* pSrc1B,
      const uint8_t* pSrc2A, const uint8_t* pSrc2B,
      const uint8_t* pSrc3A, const uint8_t* pSrc3B,
    int width, int height, int pitch, cudaStream_t stream)
  {
    ProcPlane_(pDstA, pDstB, pSrc0A, pSrc0B, pSrc1A, pSrc1B, pSrc2A, pSrc2B, width, height, pitch, stream);
  }

  virtual void ProcPlane(int p, uint16_t* pDstA, uint16_t* pDstB,
      const uint16_t* pSrc0A, const uint16_t* pSrc0B,
      const uint16_t* pSrc1A, const uint16_t* pSrc1B,
      const uint16_t* pSrc2A, const uint16_t* pSrc2B,
      const uint16_t* pSrc3A, const uint16_t* pSrc3B,
    int width, int height, int pitch, cudaStream_t stream)
  {
    ProcPlane_(pDstA, pDstB, pSrc0A, pSrc0B, pSrc1A, pSrc1B, pSrc2A, pSrc2B, width, height, pitch, stream);
  }

  template <typename pixel_t>
  void ProcPlane_(pixel_t* pDstA, pixel_t* pDstB,
    const pixel_t* pRepairA, const pixel_t* pRepairB, const pixel_t* pBobbedA, const pixel_t* pBobbedB, const pixel_t* pBlurA, const pixel_t* pBlurB,
    int width, int height, int pitch, cudaStream_t stream)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;

    int width4 = width / 4;
    int pitch4 = pitch / 4;

    int bits = vi.BitsPerComponent();
    float scale = (float)(1 << (bits - 8));
    float invscale = 1.f / scale;

    dim3 threads(32, 16);
    dim3 blocks(nblocks(width4, threads.x), nblocks(height, threads.y), pDstB ? 2 : 1);
    kl_tweak_search_clip<vpixel_t> << <blocks, threads, 0, stream >> > ((vpixel_t*)pDstA, (vpixel_t*)pDstB,
      (const vpixel_t*)pRepairA, (const vpixel_t*)pRepairB, (const vpixel_t*)pBobbedA, (const vpixel_t*)pBobbedB, (const vpixel_t*)pBlurA, (const vpixel_t*)pBlurB,
      width4, height, pitch4, scale, invscale);
    DEBUG_SYNC;
  }

public:
  KTGMC_TweakSearchClip(PClip repair, PClip bobbed, PClip blur, int y, int u, int v, IScriptEnvironment* env_)
    : KMasktoolFilterBase(repair, bobbed, blur, y, u, v, env_)
  { }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return new KTGMC_TweakSearchClip(
      args[0].AsClip(),
      args[1].AsClip(),
      args[2].AsClip(),
      args[3].AsInt(3),
      args[4].AsInt(1),
      args[5].AsInt(1),
      env);
  }
};

template <typename vpixel_t>
__global__ void kl_error_adjust(
  vpixel_t* pDstA, vpixel_t* pDstB,
  const vpixel_t* __restrict__ pSrc0A, const vpixel_t* __restrict__ pSrc0B,
  const vpixel_t* __restrict__ pSrc1A, const vpixel_t* __restrict__ pSrc1B,
  int width4, int height, int pitch4,
  float errorAdj
)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  vpixel_t* pDst = (blockIdx.z) ? pDstB : pDstA;
  const vpixel_t* __restrict__ pSrc0 = (blockIdx.z) ? pSrc0B : pSrc0A;
  const vpixel_t* __restrict__ pSrc1 = (blockIdx.z) ? pSrc1B : pSrc1A;

  if (x < width4 && y < height) {
    auto srcx = to_float(pSrc0[x + y * pitch4]);
    auto srcy = to_float(pSrc1[x + y * pitch4]);
    auto lut = (srcx * (errorAdj + 1)) - (srcy * errorAdj);
    auto tmp = clamp(lut + 0.5f, 0, ((sizeof(pSrc0[0].x) == 1) ? 255 : 65535));
    pDst[x + y * pitch4] = VHelper<vpixel_t>::cast_to(to_int(tmp));
  }
}

// mt_lutxy("clamp_f x " + string(errorAdjust1 + 1) + " * y " + string(errorAdjust1) + " * -")���v�Z
class KTGMC_ErrorAdjust : public KMasktoolFilterBase
{
  float errorAdj;
protected:

  virtual void ProcPlane(int p, uint8_t* pDstA, uint8_t* pDstB,
      const uint8_t* pSrc0A, const uint8_t* pSrc0B,
      const uint8_t* pSrc1A, const uint8_t* pSrc1B,
      const uint8_t* pSrc2A, const uint8_t* pSrc2B,
      const uint8_t* pSrc3A, const uint8_t* pSrc3B,
    int width, int height, int pitch, cudaStream_t stream)
  {
    ProcPlane_(pDstA, pDstB, pSrc0A, pSrc0B, pSrc1A, pSrc1B, width, height, pitch, stream);
  }

  virtual void ProcPlane(int p, uint16_t* pDstA, uint16_t* pDstB,
      const uint16_t* pSrc0A, const uint16_t* pSrc0B,
      const uint16_t* pSrc1A, const uint16_t* pSrc1B,
      const uint16_t* pSrc2A, const uint16_t* pSrc2B,
      const uint16_t* pSrc3A, const uint16_t* pSrc3B,
    int width, int height, int pitch, cudaStream_t stream)
  {
    ProcPlane_(pDstA, pDstB, pSrc0A, pSrc0B, pSrc1A, pSrc1B, width, height, pitch, stream);
  }

  template <typename pixel_t>
  void ProcPlane_(pixel_t* pDstA, pixel_t* pDstB,
    const pixel_t* pSrc0A, const pixel_t* pSrc0B, const pixel_t* pSrc1A, const pixel_t* pSrc1B,
    int width, int height, int pitch, cudaStream_t stream)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;

    int width4 = width / 4;
    int pitch4 = pitch / 4;

    dim3 threads(32, 16);
    dim3 blocks(nblocks(width4, threads.x), nblocks(height, threads.y), pDstB ? 2 : 1);
    kl_error_adjust<vpixel_t> << <blocks, threads, 0, stream >> > ((vpixel_t*)pDstA, (vpixel_t*)pDstB,
      (const vpixel_t*)pSrc0A, (const vpixel_t*)pSrc0B, (const vpixel_t*)pSrc1A, (const vpixel_t*)pSrc1B,
      width4, height, pitch4, errorAdj);
    DEBUG_SYNC;
  }

public:
  KTGMC_ErrorAdjust(PClip src, PClip match, float errorAdj, int y, int u, int v, IScriptEnvironment* env_)
    : KMasktoolFilterBase(src, match, y, u, v, env_)
    , errorAdj(errorAdj)
  { }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return new KTGMC_ErrorAdjust(
      args[0].AsClip(),
      args[1].AsClip(),
      (float)args[2].AsFloat(),
      args[3].AsInt(3),
      args[4].AsInt(1),
      args[5].AsInt(1),
      env);
  }
};

template <typename vpixel_t>
__global__ void kl_weave(
  vpixel_t* dst, int dst_pitch4,
  const vpixel_t* top, int top_pitch4,
  const vpixel_t* bottom, int bottom_pitch4,
  int width4, int height2)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < width4 && y < height2) {
    dst[x + (2 * y + 0) * dst_pitch4] = top[x + y * top_pitch4];
    dst[x + (2 * y + 1) * dst_pitch4] = bottom[x + y * bottom_pitch4];
  }
}

class KDoubleWeave : public CUDAFilterBase
{
  int logUVx;
  int logUVy;

  template <typename pixel_t>
  void Proc(PVideoFrame& dst, PVideoFrame& top, PVideoFrame& bottom, PNeoEnv env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;
    cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());

    const int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };

    auto planeEvent = CreateEventPlanes();

    for (int p = 0; p < 3; ++p) {
      const auto planeStream = (cudaStream_t)GetDeviceStreamPlane(p);
      const vpixel_t* pTop = reinterpret_cast<const vpixel_t*>(top->GetReadPtr(planes[p]));
      const vpixel_t* pBottom = reinterpret_cast<const vpixel_t*>(bottom->GetReadPtr(planes[p]));
      vpixel_t* pDst = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(planes[p]));
      int topPitch4 = top->GetPitch(planes[p]) / sizeof(pixel_t) / 4;
      int bottomPitch4 = bottom->GetPitch(planes[p]) / sizeof(pixel_t) / 4;
      int dstPitch4 = dst->GetPitch(planes[p]) / sizeof(pixel_t) / 4;

      int width4 = vi.width / 4;
      int height2 = vi.height / 2;

      if (p > 0) {
        width4 >>= logUVx;
        height2 >>= logUVy;
      }

      dim3 threads(32, 16);
      dim3 blocks(nblocks(width4, threads.x), nblocks(height2, threads.y));
      kl_weave << <blocks, threads, 0, planeStream >> > (
        pDst, dstPitch4, pTop, topPitch4, pBottom, bottomPitch4, width4, height2);
      DEBUG_SYNC;
    }
    if (planeEvent) planeEvent->finPlane();
  }

public:
  KDoubleWeave(PClip child, IScriptEnvironment* env_)
    : CUDAFilterBase(child, env_)
    , logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
    , logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
  {
    vi.height *= 2;
    vi.SetFieldBased(false);
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    if (!IS_CUDA) {
      env->ThrowError("[KDoubleWeave] CUDA�t���[������͂��Ă�������");
    }
#if LOG_PRINT
    if (IS_CUDA) {
      printf("KDoubleWeave[CUDA]: N=%d\n", n);
    }
#endif
    PVideoFrame a = child->GetFrame(n, env);
    PVideoFrame b = child->GetFrame(n + 1, env);
    PVideoFrame dst = env->NewVideoFrame(vi);
    const bool parity = child->GetParity(n);

    if (!parity) {
      std::swap(a, b);
    }

    int pixelSize = vi.ComponentSize();
    switch (pixelSize) {
    case 1:
      Proc<uint8_t>(dst, a, b, env);
      break;
    case 2:
      Proc<uint16_t>(dst, a, b, env);
      break;
    default:
      env->ThrowError("[KDoubleWeave] Unsupported pixel format");
    }

    return dst;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return new KDoubleWeave(
      args[0].AsClip(),
      env);
  }

  static AVSValue __cdecl CreateWeave(AVSValue args, void* user_data, IScriptEnvironment* env) {
    PClip clip = args[0].AsClip();
    if (!clip->GetVideoInfo().IsFieldBased()) {
      env->ThrowError("KWeave: Weave should be applied on field-based material: use AssumeFieldBased() beforehand");
    }
    AVSValue selectargs[3] = { KDoubleWeave::Create(args, 0, env), 2, 0 };
    return env->Invoke("SelectEvery", AVSValue(selectargs, 3));
  }
};

class KCopy : public GenericVideoFilter
{
  int logUVx;
  int logUVy;
  std::unique_ptr<cudaPlaneStreams> planeStreams;

  cudaEventPlanes *CreateEventPlanes() {
      return planeStreams->CreateEventPlanes();
  }
  void *GetDeviceStreamPlane(int idx) {
      switch (idx) {
      case 1: return planeStreams->GetDeviceStreamU();
      case 2: return planeStreams->GetDeviceStreamV();
      case 0:
      default: return planeStreams->GetDeviceStreamY();
      }
      return nullptr;
  }

  template <typename pixel_t>
  void Proc(PVideoFrame& dst, PVideoFrame& src, PNeoEnv env)
  {
    typedef typename VectorType<pixel_t>::type vpixel_t;
    cudaStream_t stream = static_cast<cudaStream_t>(env->GetDeviceStream());

    const int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };

    auto planeEvent = (IS_CUDA && ENABLE_MULTI_STREAM) ? CreateEventPlanes() : nullptr;

    for (int p = 0; p < 3; ++p) {
      if (IS_CUDA) {
        const auto planeStream = (ENABLE_MULTI_STREAM) ? (cudaStream_t)GetDeviceStreamPlane(p) : stream;
        const vpixel_t* pSrc = reinterpret_cast<const vpixel_t*>(src->GetReadPtr(planes[p]));
        vpixel_t* pDst = reinterpret_cast<vpixel_t*>(dst->GetWritePtr(planes[p]));
        int srcPitch4 = src->GetPitch(planes[p]) / sizeof(vpixel_t);
        int dstPitch4 = dst->GetPitch(planes[p]) / sizeof(vpixel_t);

        int width4 = vi.width / 4;
        int height = vi.height;

        if (p > 0) {
          width4 >>= logUVx;
          height >>= logUVy;
        }

        dim3 threads(32, 16);
        dim3 blocks(nblocks(width4, threads.x), nblocks(height, threads.y));
        kl_copy << <blocks, threads, 0, planeStream >> > (
          pDst, dstPitch4, pSrc, srcPitch4, width4, height);
        DEBUG_SYNC;
      }
      else {
        const uint8_t* pSrc = src->GetReadPtr(planes[p]);
        uint8_t* pDst = dst->GetWritePtr(planes[p]);
        int srcPitch = src->GetPitch(planes[p]);
        int dstPitch = dst->GetPitch(planes[p]);
        int rowSize = src->GetRowSize(planes[p]);
        int height = src->GetHeight(planes[p]);

        env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowSize, height);
      }
    }
    if (planeEvent) if (planeEvent) planeEvent->finPlane();
  }

public:
  KCopy(PClip child, IScriptEnvironment* env_)
    : GenericVideoFilter(child)
    , logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
    , logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
  {
      auto env = (PNeoEnv)env_;
      if (IS_CUDA && ENABLE_MULTI_STREAM) {
          planeStreams = std::make_unique<cudaPlaneStreams>();
          planeStreams->initStream((cudaStream_t)((PNeoEnv)env_)->GetDeviceStream());
      }
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;
    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame dst = env->NewVideoFrame(vi);

    int pixelSize = vi.ComponentSize();
    switch (pixelSize) {
    case 1:
      Proc<uint8_t>(dst, src, env);
      break;
    case 2:
      Proc<uint16_t>(dst, src, env);
      break;
    default:
      env->ThrowError("[KCopy] Unsupported pixel format");
    }

    return dst;
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    if (cachehints == CACHE_GET_DEV_TYPE) {
      return GetDeviceTypes(child) & (DEV_TYPE_CPU | DEV_TYPE_CUDA);
    }
    return 0;
  };

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env) {
    return new KCopy(
      args[0].AsClip(),
      env);
  }
};

void AddFuncKernel(IScriptEnvironment* env)
{
  env->AddFunction("KTGMC_Bob", "c[b]f[c]f", KTGMC_Bob::Create, 0);
  env->AddFunction("KBinomialTemporalSoften", "ci[scenechange]i[chroma]b", KBinomialTemporalSoften::Create, 0);
  env->AddFunction("KRemoveGrain", "c[mode]i[modeU]i[modeV]i", KRemoveGrain::Create, 0);
  env->AddFunction("KRepair", "cc[mode]i[modeU]i[modeV]i", KRepair::Create, 0);
  env->AddFunction("KVerticalCleaner", "c[mode]i[modeU]i[modeV]i", KVerticalCleaner::Create, 0);
  env->AddFunction("KGaussResize", "c[p]f[chroma]b", KGaussResize::Create, 0);

  env->AddFunction("KInpandVerticalX2", "c[y]i[u]i[v]i", KXpandVerticalX2<Min5>::Create, 0);
  env->AddFunction("KExpandVerticalX2", "c[y]i[u]i[v]i", KXpandVerticalX2<Max5>::Create, 0);

  env->AddFunction("KMakeDiff", "cc[y]i[u]i[v]i", KMakeDiff<MakeDiffOp>::Create, 0);
  env->AddFunction("KAddDiff", "cc[y]i[u]i[v]i", KMakeDiff<AddDiffOp>::Create, 0);
  env->AddFunction("KLogic", "cc[mode]s[y]i[u]i[v]i", KLogicCreate, 0);

  env->AddFunction("KTGMC_BobShimmerFixesMerge", "cccc[y]i[u]i[v]i", KTGMC_BobShimmerFixesMerge::Create, 0);
  env->AddFunction("KTGMC_VResharpen", "c[y]i[u]i[v]i", KTGMC_VResharpen::Create, 0);
  // mt_lutxy("clamp_f x x y - sharpAdj * +")���v�Z
  env->AddFunction("KTGMC_Resharpen", "cc[sharpAdj]f[y]i[u]i[v]i", KTGMC_Resharpen::Create, 0);
  env->AddFunction("KTGMC_LimitOverSharpen", "cccc[ovs]i[y]i[u]i[v]i", KTGMC_LimitOverSharpen::Create, 0);
  env->AddFunction("KTGMC_ToFullRange", "c[y]i[u]i[v]i", KTGMC_ToFullRange::Create, 0);
  env->AddFunction("KTGMC_TweakSearchClip", "ccc[y]i[u]i[v]i", KTGMC_TweakSearchClip::Create, 0);
  env->AddFunction("KTGMC_LosslessProc", "cc[y]i[u]i[v]i", KTGMC_LosslessProc::Create, 0);

  env->AddFunction("KMerge", "cc[weight]f", KMerge::CreateMerge, 0);
  env->AddFunction("KMergeLuma", "cc[weight]f", KMerge::CreateMergeLuma, 0);
  env->AddFunction("KMergeChroma", "cc[weight]f", KMerge::CreateMergeChroma, 0);

  env->AddFunction("KDoubleWeave", "c", KDoubleWeave::Create, 0);
  env->AddFunction("KWeave", "c", KDoubleWeave::CreateWeave, 0);
  env->AddFunction("KCopy", "c", KCopy::Create, 0);

  // mt_lutxy( XX, YY, "clamp_f x " + string(errorAdj + 1) + " * y " + string(errorAdj) + " * -" )
  env->AddFunction("KTGMC_ErrorAdjust", "cc[errorAdj]f[y]i[u]i[v]i", KTGMC_ErrorAdjust::Create, 0);
}
