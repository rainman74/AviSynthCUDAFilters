#pragma once
#ifndef __KUTIL_H__
#define __KUTIL_H__

#define AVISYNTH_NEO (1)
#define AVISYNTH_PLUS (2) 

#if defined(_WIN32) || defined(_WIN64)
#define AVISYNTH_MODE AVISYNTH_NEO
#else
#define AVISYNTH_MODE AVISYNTH_PLUS
#endif

#if AVISYNTH_MODE == AVISYNTH_PLUS
#define GetProperty GetEnvProperty
#define CopyFrameProps copyFrameProps
#endif

#if !defined(KFM_API)
  #if defined(_WIN32) || defined(_WIN64)
    #define KFM_API __declspec(dllexport)
    #define WINAPI __stdcall
    #define WINAPI_CDECL __cdecl
  #else
    #define KFM_API __attribute__((visibility("default")))
    #define WINAPI
    #define WINAPI_CDECL
  #endif
#endif

#define KFM_MAX_PATH 512



#endif //__KUTIL_H__
