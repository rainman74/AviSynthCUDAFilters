#pragma once
#include "avisynth.h"

#define NOMINMAX
#include <windows.h>

#include "CommonFunctions.h"
#include "KMV.h"

enum {
  ANALYZE_MAX_BATCH = 8
};

class IMVCUDA;

template <typename pixel_t>
class IKDeintKernel
{
public:
  typedef typename std::conditional <sizeof(pixel_t) == 1, unsigned short, int>::type tmp_t;

  virtual bool IsEnabled() const = 0;

  virtual void MemCpy(void* dst, const void* src, int nbytes) = 0;
  virtual void Copy(pixel_t* dst, int dst_pitch, const pixel_t* src, int src_pitch, int width, int height) = 0;
  virtual void PadFrame(pixel_t *refFrame, int refPitch, int hPad, int vPad, int width, int height) = 0;
  virtual void CopyPad(
      pixel_t* dst, int dst_pitch, const pixel_t* src, int src_pitch, int hPad, int vPad, int width, int height, void *stream_) = 0;
  virtual void VerticalWiener(
    pixel_t *pDst, const pixel_t *pSrc, int nDstPitch,
    int nSrcPitch, int nWidth, int nHeight, int bits_per_pixel, void *stream_) = 0;
  virtual void HorizontalWiener(
    pixel_t *pDst, const pixel_t *pSrc, int nDstPitch,
    int nSrcPitch, int nWidth, int nHeight, int bits_per_pixel, void *stream_) = 0;
  virtual void RB2BilinearFiltered(
    pixel_t *pDst, const pixel_t *pSrc, int nDstPitch, int nSrcPitch, int nWidth, int nHeight) = 0;
  virtual void RB2BilinearFilteredPad(
      pixel_t *pDst, const pixel_t *pSrc, int nDstPitch, int nSrcPitch, int hPad, int vPad, int nWidth, int nHeight, void *stream_) = 0;

  // Analyze //
  virtual int GetSearchBlockSize() = 0;
  virtual int GetSearchBatchSize() = 0;
  virtual int GetLoadMVBatchSize() = 0;
  virtual void EstimateGlobalMV(int batch, const short2* vectors, int vectorsPitch, int nBlkCount, short2* globalMV) = 0;
  virtual void InterpolatePrediction(
    int batch,
    const short2* src_vector, int srcVectorPitch, const int* src_sad, int srcSadPitch,
    short2* dst_vector, int dstVectorPitch, int* dst_sad, int dstSadPitch,
    int nSrcBlkX, int nSrcBlkY, int nDstBlkX, int nDstBlkY,
    int normFactor, int normov, int atotal, int aodd, int aeven) = 0;
  virtual void LoadMV(const VECTOR* in, short2* vectors, int* sads, int nBlkCount) = 0;
  virtual void StoreMV(VECTOR* out, const short2* vectors, const int* sads, int nBlkCount) = 0;
  virtual void LoadMVBatch(void* _loadmvbatch, void* _hloadmvbatch, int batch,
      const VECTOR** in, VECTOR** out, short2* vectors, int vectorsPitch, int* sads, int sadPitch, int nBlkCount) = 0;
  virtual void WriteDefaultMV(VECTOR* dst, int nBlkCount, int verybigSAD) = 0;

  // 36 args
  virtual void Search(
    int batch, VECTOR **out, void* _searchbatch, void* _hsearchbatch,
    int searchType, int nBlkX, int nBlkY, int nBlkSize, int nLogScale,
    int nLambdaLevel, int lsad, int penaltyZero, int penaltyGlobal, int penaltyNew,
    int nPel, bool chroma, int nPad, int nBlkSizeOvr, int nExtendedWidth, int nExptendedHeight,
    const pixel_t** pSrcY, const pixel_t** pSrcU, const pixel_t** pSrcV,
    const pixel_t** pRefY, const pixel_t** pRefU, const pixel_t** pRefV,
    int nPitchY, int nPitchUV, int nImgPitchY, int nImgPitchUV,
    const short2* globalMV, short2* vectors, int vectorsPitch, int* sads, int sadPitch, void* blocks, int* prog, int* next) = 0;

  // Degrain //
  virtual void GetDegrainStructSize(int N, int& degrainBlock, int& degrainArg) = 0;

  //35 args
  virtual void Degrain(
    int N, int nWidth, int nHeight, int nBlkX, int nBlkY, int nPad, int nBlkSize, int nPel, int nBitsPerPixel,
    bool* enableYUV, bool* isUsableB, bool* isUsableF,
    int nTh1, int nTh2, int thSAD, int thSADC, bool binomial,
    const short* ovrwins, const short* overwinsUV,
    const VECTOR** mvB, const VECTOR** mvF,
    const pixel_t** pSrc, pixel_t** pDst, tmp_t** pTmp, const pixel_t** pRefB, const pixel_t** pRefF,
    int nPitchY, int nPitchUV,
    int nPitchSuperY, int nPitchSuperUV, int nImgPitchY, int nImgPitchUV,
    void* _degrainblock, void* _degrainarg, int* sceneChange, IMVCUDA *cuda) = 0;

  // Compensate //
  virtual int GetCompensateStructSize() = 0;

  //31 args
  virtual void Compensate(
    int nWidth, int nHeight, int nBlkX, int nBlkY, int nPad, int nBlkSize, int nPel, int nBitsPerPixel,
    int nTh1, int nTh2, int time256, int thSAD,
    const short* ovrwins, const short* overwinsUV, const VECTOR* mv,
    const pixel_t** pSrc, pixel_t** pDst, tmp_t** pTmp, const pixel_t** pRef,
    int nPitchY, int nPitchUV,
    int nPitchSuperY, int nPitchSuperUV, int nImgPitchY, int nImgPitchUV,
    void* _compensateblock, int* sceneChange) = 0;
};

class cudaEventPlanes {
protected:
    cudaEvent_t start;
    cudaEvent_t endY;
    cudaEvent_t endU;
    cudaEvent_t endV;
    cudaStream_t streamMain;
    cudaStream_t streamY;
    cudaStream_t streamU;
    cudaStream_t streamV;
public:
    cudaEventPlanes();
    ~cudaEventPlanes();
    void init();
    void startPlane(cudaStream_t sMain, cudaStream_t sY, cudaStream_t sU, cudaStream_t sV);
    void finPlane();
    bool planeYFin();
    bool planeUFin();
    bool planeVFin();
};

class IMVCUDA
{
public:
  virtual void SetEnv(PNeoEnv env) = 0;
  virtual bool IsEnabled() const = 0;
  virtual IKDeintKernel<uint8_t>* get(uint8_t) = 0;
  virtual IKDeintKernel<uint16_t>* get(uint16_t) = 0;
  virtual cudaEventPlanes *CreateEventPlanes() = 0;
  virtual void *GetDeviceStreamY() = 0;
  virtual void *GetDeviceStreamU() = 0;
  virtual void *GetDeviceStreamV() = 0;
  virtual void *GetDeviceStreamPlane(int idx) = 0;
};

IMVCUDA* CreateKDeintCUDA();
