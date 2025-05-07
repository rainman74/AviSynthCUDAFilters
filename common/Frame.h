#pragma once

#include <stdint.h>
#include <avisynth.h>
#include "KUtil.h"

// ROI��������PVideoFrame�̃��b�p
struct Frame {
  PVideoFrame frame;
  // �S�ăo�C�g�P��
  int offsetX, offsetY, offsetUVx, offsetUVy;
  int width, height, widthUV, heightUV;
  Frame() { }
  Frame(const PVideoFrame& frame)
    : frame(frame)
    , offsetX(0)
    , offsetY(0)
    , width(0)
    , height(0)
  {
    init();
  }
  Frame(const PVideoFrame& frame, int cropY)
    : frame(frame)
    , offsetX(0)
    , offsetY(cropY)
    , width(frame->GetRowSize() - offsetX * 2)
    , height(frame->GetHeight() - offsetY * 2)
  {
    init();
  }
  Frame(const PVideoFrame& frame, int cropX, int cropY, int pixelsize)
    : frame(frame)
    , offsetX(cropX * pixelsize)
    , offsetY(cropY)
    , width(frame->GetRowSize() - offsetX * 2)
    , height(frame->GetHeight() - offsetY * 2)
  {
    init();
  }
  Frame(const PVideoFrame& frame, int offsetX, int offsetY, int width, int height, int pixelsize)
    : frame(frame)
    , offsetX(offsetX * pixelsize)
    , offsetY(offsetY)
    , width(width * pixelsize)
    , height(height)
  {
    init();
  }

  // for conditional expressions
  operator void*() const { return frame; }
  bool operator!() const { return !frame; }

  template <typename T> int GetPitch(int plane = 0) const {
		if (!frame) return 0;
    return frame->GetPitch(plane) / sizeof(T);
  }
  int GetRowSize(int plane = 0) const {
    return (plane & (PLANAR_U | PLANAR_V)) ? widthUV : width;
  }
  int GetHeight(int plane = 0) const {
    return (plane & (PLANAR_U | PLANAR_V)) ? heightUV : height;
  }
  template <typename T> int GetWidth(int plane = 0) const {
    return GetRowSize(plane) / sizeof(T);
  }
  template <typename T> const T* GetReadPtr(int plane = 0) const {
		if (!frame) return nullptr;
    const BYTE* ptr = frame->GetReadPtr(plane);
    if (ptr) {
      if (plane & (PLANAR_U | PLANAR_V)) {
        if (offsetUVx > 0 || offsetUVy > 0) {
          ptr += offsetUVx + offsetUVy * frame->GetPitch(plane);
        }
      }
      else {
        if (offsetX > 0 || offsetY > 0) {
          ptr += offsetX + offsetY * frame->GetPitch(plane);
        }
      }
    }
    return reinterpret_cast<const T*>(ptr);
  }
  template <typename T> T* GetWritePtr(int plane = 0) {
		if (!frame) return nullptr;
    BYTE* ptr = frame->GetWritePtr(plane);
    if (ptr) {
      if (plane & (PLANAR_U | PLANAR_V)) {
        if (offsetUVx > 0 || offsetUVy > 0) {
          ptr += offsetUVx + offsetUVy * frame->GetPitch(plane);
        }
      }
      else {
        if (offsetX > 0 || offsetY > 0) {
          ptr += offsetX + offsetY * frame->GetPitch(plane);
        }
      }
    }
    return reinterpret_cast<T*>(ptr);
  }
#if AVISYNTH_MODE == AVISYNTH_NEO
  void SetProperty(const char* key, const AVSMapValue& value) { frame->SetProperty(key, value); }
  const AVSMapValue* GetProperty(const char* key) const { return frame->GetProperty(key); }
  PVideoFrame GetProperty(const char* key, const PVideoFrame& def) const { return frame->GetProperty(key, def); }
  int GetProperty(const char* key, int def) const { return frame->GetProperty(key, def); }
  double GetProperty(const char* key, double def) const { return frame->GetProperty(key, def); }
#elif AVISYNTH_MODE == AVISYNTH_PLUS
  PVideoFrame GetPropertyFrame(PNeoEnv env, const char* key, const PVideoFrame& def) const { int error; auto val = env->propGetFrame(env->getFramePropsRO(frame), key, 0, &error); if (error) return def; else return val; }
  void SetPropertyInt(PNeoEnv env, const char* key, const int value) { env->propSetInt(env->getFramePropsRW(frame), key, value, AVSPropAppendMode::PROPAPPENDMODE_REPLACE); }
  void SetPropertyDouble(PNeoEnv env, const char* key, const double value) { env->propSetFloat(env->getFramePropsRW(frame), key, value, AVSPropAppendMode::PROPAPPENDMODE_REPLACE); }
  void SetPropertyFrame(PNeoEnv env, const char* key, const PVideoFrame &value) { env->propSetFrame(env->getFramePropsRW(frame), key, value, AVSPropAppendMode::PROPAPPENDMODE_REPLACE); }
  int GetPropertyInt(PNeoEnv env, const char* key, int def) const { int error; auto val = env->propGetInt(env->getFramePropsRO(frame), key, 0, &error); if (error) return def; else return (int)val; }
  double GetPropertyDouble(PNeoEnv env, const char* key, double def) const { int error; auto val = env->propGetFloat(env->getFramePropsRO(frame), key, 0, &error); if (error) return def; else return val; }
#endif
  PDevice GetDevice() const { return frame->GetDevice(); }
  int CheckMemory() const { return frame->CheckMemory(); }

  void Crop(int x, int y, int pixelsize) {
    offsetX += x * pixelsize;
    offsetY += y;
    offsetUVx += x * pixelsize >> ((widthUV < width) ? 1 : 0);
    offsetUVy += y >> ((heightUV < height) ? 1 : 0);
    widthUV -= x * pixelsize * ((widthUV < width) ? 1 : 2);
    heightUV -= y * ((heightUV < height) ? 1 : 2);
    width -= x * pixelsize * 2;
    height -= y * 2;
  }

private:
  void init()
  {
    if (!frame) return;
    if (width == 0) {
      width = frame->GetRowSize() - offsetX;
    }
    if (height == 0) {
      height = frame->GetHeight() - offsetY;
    }
    if (frame->GetRowSize(PLANAR_U) < frame->GetRowSize()) {
      // UV�͉�����
      widthUV = width / 2;
      offsetUVx = offsetX / 2;
    }
    else {
      widthUV = width;
      offsetUVx = offsetX;
    }
    if (frame->GetHeight(PLANAR_U) < frame->GetHeight()) {
      // UV�͏c����
      heightUV = height / 2;
      offsetUVy = offsetY / 2;
    }
    else {
      heightUV = height;
      offsetUVy = offsetY;
    }
  }
};
