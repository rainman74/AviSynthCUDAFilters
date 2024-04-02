#pragma once

#include "avisynth.h"
#include <deque>

int GetDeviceTypes(const PClip& clip);

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

class CudaPlaneEventsPool {
protected:
    std::deque<std::unique_ptr<cudaEventPlanes>> events;
public:
    CudaPlaneEventsPool();
    ~CudaPlaneEventsPool();

    cudaEventPlanes *PlaneStreamStart(cudaStream_t sMain, cudaStream_t sY, cudaStream_t sU, cudaStream_t sV);
};

class cudaPlaneStreams {
    cudaStream_t stream;
    cudaStream_t streamY;
    cudaStream_t streamU;
    cudaStream_t streamV;
    CudaPlaneEventsPool eventPool;
public:
    cudaPlaneStreams();
    ~cudaPlaneStreams();
    void initStream(cudaStream_t stream_);
    cudaEventPlanes *CreateEventPlanes();
    void *GetDeviceStreamY();
    void *GetDeviceStreamU();
    void *GetDeviceStreamV();
    void *GetDeviceStreamDefault();
    void *GetDeviceStreamPlane(int idx);
};

class cudaHostBatchParam {
protected:
    void *ptr;
    size_t size;
    cudaEvent_t event;
public:
    void alloc(size_t size);
    size_t getSize() const { return size; }
    void *getPtr() { return ptr; }
    cudaHostBatchParam();
    ~cudaHostBatchParam();
    void recordEvent(cudaStream_t stream);
    bool hasFinished();
};

class cudaHostBatchParams {
protected:
    std::deque<std::unique_ptr<cudaHostBatchParam>> params;
public:
    cudaHostBatchParams();
    ~cudaHostBatchParams();

    cudaHostBatchParam *getNewParam(size_t size);
};
