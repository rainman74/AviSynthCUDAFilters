#pragma once
#include "avisynth.h"

#include "rgy_osdep.h"

#include "CommonFunctions.h"

// CUDAカーネル実装の共通処理
class CudaKernelBase
{
protected:
  PNeoEnv env;
  cudaStream_t stream;
public:

  void SetEnv(PNeoEnv env)
  {
    this->env = env;
    stream = static_cast<cudaStream_t>(env->GetDeviceStream());
  }

  void VerifyCUDAPointer(void* ptr)
  {
#ifndef NDEBUG
    cudaPointerAttributes attr;
    CUDA_CHECK(cudaPointerGetAttributes(&attr, ptr));
    if (attr.type != cudaMemoryTypeDevice) {
      env->ThrowError("[CUDA Error] Not valid devicce pointer");
    }
#endif
  }
};

