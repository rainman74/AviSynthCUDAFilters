#pragma once

#include "avisynth.h"
#include <deque>

int GetDeviceTypes(const PClip& clip);

class cudaEventPlanes {
protected:
    cudaEvent_t start;
    cudaEvent_t endU;
    cudaEvent_t endV;
    cudaStream_t streamMain;
    cudaStream_t streamU;
    cudaStream_t streamV;
public:
    cudaEventPlanes();
    ~cudaEventPlanes();
    void init();
    void startPlane(cudaStream_t sMain, cudaStream_t sU, cudaStream_t sV);
    void finPlane();
    bool planeUFin();
    bool planeVFin();
};

class CudaPlaneEventsPool {
protected:
    std::deque<std::unique_ptr<cudaEventPlanes>> events;
public:
    CudaPlaneEventsPool();
    ~CudaPlaneEventsPool();

    cudaEventPlanes *PlaneStreamStart(cudaStream_t sMain, cudaStream_t sU, cudaStream_t sV);
};

class cudaPlaneStreams {
    cudaStream_t stream;
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
