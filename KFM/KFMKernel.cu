
#include <stdint.h>
#include <avisynth.h>

#include <algorithm>
#include <numeric>
#include <memory>
#include <vector>
#include <mutex>

#include "CommonFunctions.h"
#include "KFM.h"
#include "TextOut.h"

#include "VectorFunctions.cuh"
#include "ReduceKernel.cuh"
#include "KFMFilterBase.cuh"

class KPatchCombe : public KFMFilterBase
{
  PClip clip60;
  PClip combemaskclip;
  PClip containscombeclip;
  PClip fmclip;

  bool is24; // 24p or 30p
  PulldownPatterns patterns;

  template <typename pixel_t>
  PVideoFrame GetFrameT(int n, PNeoEnv env)
  {
    PDevice cpuDevice = env->GetDevice(DEV_TYPE_CPU, 0);

    {
      Frame containsframe = env->GetFrame(containscombeclip, n, cpuDevice);
      if (*containsframe.GetReadPtr<int>() == 0) {
        // �_���ȃu���b�N�͂Ȃ��̂ł��̂܂ܕԂ�
        return child->GetFrame(n, env);
      }
    }

    int n60;

    if (is24) {
      int cycleIndex = n / 4;
      KFMResult fm = *(Frame(env->GetFrame(fmclip, cycleIndex, cpuDevice)).GetReadPtr<KFMResult>());
      Frame24Info frameInfo = patterns.GetFrame24(fm.pattern, n);

      int fieldIndex[] = { 1, 3, 6, 8 };
      // �W���ʒu
      n60 = fieldIndex[n % 4];
      // �t�B�[���h�Ώ۔͈͂ɕ␳
      n60 = clamp(n60, frameInfo.fieldStartIndex, frameInfo.fieldStartIndex + frameInfo.numFields - 1);
      n60 += cycleIndex * 10;
    }
    else {
      n60 = n * 2;
    }

    Frame baseFrame = child->GetFrame(n, env);
    Frame frame60 = clip60->GetFrame(n60, env);
    Frame mflag = combemaskclip->GetFrame(n, env);

    // �_���ȃu���b�N��bob�t���[������R�s�[
    Frame dst = env->NewVideoFrame(vi);
    MergeBlock<pixel_t>(baseFrame, frame60, mflag, dst, env);

    return dst.frame;
  }

public:
  KPatchCombe(PClip clip24, PClip clip60, PClip fmclip, PClip combemaskclip, PClip containscombeclip, IScriptEnvironment* env)
    : KFMFilterBase(clip24, env)
    , clip60(clip60)
    , combemaskclip(combemaskclip)
    , containscombeclip(containscombeclip)
    , fmclip(fmclip)
  {
    auto vi24 = child->GetVideoInfo();
    auto vi60 = clip60->GetVideoInfo();

    auto rate = (double)(vi60.fps_numerator * vi24.fps_denominator) / (vi60.fps_denominator * vi24.fps_numerator);
    if (rate != 2.5 && rate != 2.0) {
      env->ThrowError("[KPatchCombe] Unsupported frame rate combination: %f", rate);
    }
    is24 = (rate == 2.5);

    // �`�F�b�N
    CycleAnalyzeInfo::GetParam(fmclip->GetVideoInfo(), env);
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
      env->ThrowError("[KPatchCombe] Unsupported pixel format");
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
    return new KPatchCombe(
      args[0].AsClip(),       // clip24
      args[1].AsClip(),       // clip60
      args[2].AsClip(),       // fmclip
      args[3].AsClip(),       // combemaskclip
      args[4].AsClip(),       // containscombeclip
      env
    );
  }
};

enum KFMSWTICH_FLAG {
  FRAME_60 = 1,
  FRAME_30,
  FRAME_24,
  FRAME_UCF,
};

class KFMSwitch : public KFMFilterBase
{
  typedef uint8_t pixel_t;

  enum Mode {
    NORMAL = 0,
    WITH_FRAME_DURATION = 1,
    ONLY_FRAME_DURATION = 2,
  };

  struct FrameDurationInfo {
      int duration;
      bool isFrame24;
  };

  VideoInfo srcvi;

  PClip clip24;
  PClip mask24;
  PClip cc24;

  PClip clip30;
  PClip mask30;
  PClip cc30;

  PClip fmclip;
  PClip combemaskclip;
  PClip containscombeclip;
  PClip ucfclip;
  float thswitch; // <0��60fps����
  int mode; // 0:�ʏ� 1:�ʏ�+timecode���� 2:timecode�����̂�
  bool show;
  bool showflag;

  int analyzeMode;

  int logUVx;
  int logUVy;
  int nBlkX, nBlkY;

  bool is30_60;
  bool is24_60;
  bool is120;

  PulldownPatterns patterns;

	// timecode�����p�e���|����
	std::string filepath;
	std::vector<FrameDurationInfo> durations;
    std::mutex mtxGetFrameTop;
	int current;
	bool complete;

  template <typename pixel_t>
  void VisualizeFlag(Frame& dst, Frame& flag, PNeoEnv env)
  {
    // ���茋�ʂ�\��
    int blue[] = { 73, 230, 111 };

    pixel_t* dstY = dst.GetWritePtr<pixel_t>(PLANAR_Y);
    pixel_t* dstU = dst.GetWritePtr<pixel_t>(PLANAR_U);
    pixel_t* dstV = dst.GetWritePtr<pixel_t>(PLANAR_V);
    const uint8_t* flagY = flag.GetReadPtr<uint8_t>(PLANAR_Y);
    const uint8_t* flagC = flag.GetReadPtr<uint8_t>(PLANAR_U);

    int dstPitchY = dst.GetPitch<pixel_t>(PLANAR_Y);
    int dstPitchUV = dst.GetPitch<pixel_t>(PLANAR_U);
    int fpitchY = flag.GetPitch<uint8_t>(PLANAR_Y);
    int fpitchUV = flag.GetPitch<uint8_t>(PLANAR_U);

    // �F��t����
    for (int y = 0; y < srcvi.height; ++y) {
      for (int x = 0; x < srcvi.width; ++x) {
        int coefY = flagY[x + y * fpitchY];
        int offY = x + y * dstPitchY;
        dstY[offY] = (blue[0] * coefY + dstY[offY] * (128 - coefY)) >> 7;

        int coefC = flagC[(x >> logUVx) + (y >> logUVy) * fpitchUV];
        int offUV = (x >> logUVx) + (y >> logUVy) * dstPitchUV;
        dstU[offUV] = (blue[1] * coefC + dstU[offUV] * (128 - coefC)) >> 7;
        dstV[offUV] = (blue[2] * coefC + dstV[offUV] * (128 - coefC)) >> 7;
      }
    }
  }

  struct FrameInfo {
    int baseType;
    int maskType;
    int n24;
  };

  Frame GetBaseFrame(int n60, FrameInfo& info, PNeoEnv env)
  {
    switch (info.baseType) {
    case FRAME_60:
      return child->GetFrame(n60, env);
    case FRAME_UCF:
      return ucfclip->GetFrame(n60, env);
    case FRAME_30:
      return clip30->GetFrame(is30_60 ? n60 : (n60 >> 1), env);
    case FRAME_24:
      return clip24->GetFrame(is24_60 ? n60 : info.n24, env);
    }
    return Frame();
  }

  Frame GetMaskFrame(int n60, FrameInfo& info, PNeoEnv env)
  {
    switch (info.maskType) {
    case FRAME_30:
      return mask30->GetFrame(n60 >> 1, env);
    case FRAME_24:
      return mask24->GetFrame(info.n24, env);
    }
    return Frame();
  }

  template <typename pixel_t>
  Frame InternalGetFrame(int n60, FrameInfo& info, PNeoEnv env)
  {
    Frame baseFrame = GetBaseFrame(n60, info, env);
    if (info.maskType == 0) {
      return baseFrame;
    }

    Frame mflag = GetMaskFrame(n60, info, env);
    Frame frame60 = child->GetFrame(n60, env);

    if (!IS_CUDA && srcvi.ComponentSize() == 1 && showflag) {
      env->MakeWritable(&baseFrame.frame);
      VisualizeFlag<pixel_t>(baseFrame, mflag, env);
      return baseFrame;
    }

    // �_���ȃu���b�N��bob�t���[������R�s�[
    Frame dst = env->NewVideoFrame(srcvi);
    MergeBlock<pixel_t>(baseFrame, frame60, mflag, dst, env);

    // �v���p�e�B���R�s�[
    env->CopyFrameProps(baseFrame.frame, dst.frame);

    return dst;
  }

  FrameInfo GetFrameInfo(int n60, KFMResult fm, PNeoEnv env)
  {
    int cycleIndex = n60 / 10;
    Frame baseFrame;
    FrameInfo info = { 0 };

    // 60p����� 1�p�X�̏ꍇ�̓R�X�g 2�p�X�̏ꍇ��KFMCycleAnalyze�̌��� ���g��
    if (thswitch >= 0 && ((analyzeMode == 0 && fm.cost > thswitch) || (analyzeMode != 0 && fm.is60p))) {
      // �R�X�g�������̂�60p�Ɣ��f
      info.baseType = ucfclip ? FRAME_UCF : FRAME_60;

      if (mode == ONLY_FRAME_DURATION) {
        // FrameDuration�݂̂Ȃ�UCF��60����ʂ���K�v�͂Ȃ��̂�
        // �t���[���̐���������邽�߂����ŋA��
        return info;
      }

      if (ucfclip) {
        // �t���[���ɃA�N�Z�X����������̂Œ���
        // �����ł�60fps�Ɍ��肵�Ă�̂ŁA
        // ����GetFrame�ł��̃t���[�����K�v�Ȃ��Ƃ͌��肵�Ă���
        baseFrame = ucfclip->GetFrame(n60, env);
        auto prop = baseFrame.GetProperty(DECOMB_UCF_FLAG_STR);
        if (prop == nullptr) {
          env->ThrowError("Invalid UCF clip");
        }
        auto flag = (DECOMB_UCF_FLAG)prop->GetInt();
        // �t���[���u�������ꂽ�ꍇ�́A60p�����}�[�W���������s����
        if (flag != DECOMB_UCF_NEXT && flag != DECOMB_UCF_PREV) {
          return info;
        }
      }
      else {
        return info;
      }
    }

    // �����ł�type�� 24 or 30 or UCF

    if (PulldownPatterns::Is30p(fm.pattern)) {
      // 30p
      int n30 = n60 >> 1;

      if (!baseFrame) {
        info.baseType = FRAME_30;
      }

      Frame containsframe = env->GetFrame(cc30, n30, env->GetDevice(DEV_TYPE_CPU, 0));
      info.maskType = *containsframe.GetReadPtr<int>() ? FRAME_30 : 0;
    }
    else {
      // 24p�t���[���ԍ����擾
      Frame24Info frameInfo = patterns.GetFrame60(fm.pattern, n60);
      // fieldShift�ŃT�C�N�����܂������Ƃ�����̂ŁAframeIndex��fieldShift���Ōv�Z
      int frameIndex = frameInfo.frameIndex + frameInfo.fieldShift;
      int n24 = frameInfo.cycleIndex * 4 + frameIndex;

      if (frameIndex < 0) {
        // �O�ɋ󂫂�����̂őO�̃T�C�N��
        n24 = frameInfo.cycleIndex * 4 - 1;
      }
      else if (frameIndex >= 4) {
        // ���̃T�C�N���̃p�^�[�����擾
        PDevice cpudev = env->GetDevice(DEV_TYPE_CPU, 0);
        auto nextfm = *(Frame(env->GetFrame(fmclip, cycleIndex + 1, cpudev)).GetReadPtr<KFMResult>());
        int fstart = patterns.GetFrame24(nextfm.pattern, 0).fieldStartIndex;
        if (fstart > 0) {
          // �O�ɋ󂫂�����̂őO�̃T�C�N��
          n24 = frameInfo.cycleIndex * 4 + 3;
        }
        else {
          // �O�ɋ󂫂��Ȃ��̂Ō��̃T�C�N��
          n24 = frameInfo.cycleIndex * 4 + 4;
        }
      }

      if (!baseFrame) {
        info.baseType = FRAME_24;
      }

      Frame containsframe = env->GetFrame(cc24, n24, env->GetDevice(DEV_TYPE_CPU, 0));
      info.maskType = *containsframe.GetReadPtr<int>() ? FRAME_24 : 0;
      info.n24 = n24;
    }

    return info;
  }

  static const char* FrameTypeStr(int frameType)
  {
    switch (frameType) {
    case FRAME_60: return "60p";
    case FRAME_30: return "30p";
    case FRAME_24: return "24p";
    case FRAME_UCF: return "UCF";
    }
    return "???";
  }

  FrameDurationInfo GetFrameDuration(int n60, FrameInfo& info, PNeoEnv env)
  {
    int duration = 1;
    bool isFrame24 = false;
    // 60fps�}�[�W����������ꍇ��60fps
    if (thswitch < 0 || info.maskType == 0) {
      int source;
      // �ő�duration��ݒ�
      switch (info.baseType) {
      case FRAME_60:
      case FRAME_UCF:
        duration = 1;
        source = n60;
        break;
      case FRAME_30:
        duration = 2;
        source = n60 >> 1;
        break;
      case FRAME_24:
        duration = 4;
        source = info.n24;
        break;
      }
      PDevice cpudev = env->GetDevice(DEV_TYPE_CPU, 0);
      for (int i = 1; i < duration; ++i) {
        // �����ł� FRAME_30 or FRAME_24
        if (n60 + i >= vi.num_frames) {
          // �t���[�����𒴂��Ă�
          duration = i;
          isFrame24 = info.baseType == FRAME_24;
          break;
        }
        int cycleIndex = (n60 + i) / 10;
        KFMResult fm = *(Frame(env->GetFrame(fmclip, cycleIndex, cpudev)).GetReadPtr<KFMResult>());
        FrameInfo next = GetFrameInfo(n60 + i, fm, env);
        if (next.baseType != info.baseType) {
          // �x�[�X�^�C�v��������瓯���t���[���łȂ�
          duration = i;
          isFrame24 = info.baseType == FRAME_24;
          break;
        }
        else {
          int nextsource = -1;
          switch (next.baseType) {
          case FRAME_30:
            nextsource = (n60 + i) >> 1;
            break;
          case FRAME_24:
            nextsource = next.n24;
            break;
          }
          if (nextsource != source) {
            // �\�[�X�t���[����������瓯���t���[���łȂ�
            duration = i;
            isFrame24 = info.baseType == FRAME_24;
            break;
          }
        }
      }
    }
    return FrameDurationInfo{ duration, isFrame24 };
  }

  FrameDurationInfo GetFrameDuration(int n60, PNeoEnv env)
  {
  	PDevice cpudev = env->GetDevice(DEV_TYPE_CPU, 0);
  	int cycleIndex = n60 / 10;
  	KFMResult fm = *(Frame(env->GetFrame(fmclip, cycleIndex, cpudev)).GetReadPtr<KFMResult>());
  	FrameInfo info = GetFrameInfo(n60, fm, env);
  	return GetFrameDuration(n60, info, env);
  }

  void WriteToFile(PNeoEnv env)
  {
    std::vector<FrameDurationInfo> frame_list; // 120fps�p��duration���X�g
    frame_list.reserve(durations.size());

  	auto file = std::unique_ptr<TextFile>(new TextFile(filepath + ".duration.txt", "w", env));
  	for (int i = 0; i < (int)durations.size(); i++) {
  		fprintf(file->fp, "%d\n", durations[i].duration);
        frame_list.push_back({ durations[i].duration * 2, durations[i].isFrame24 });
  	}
    file.reset();

    if (is120) {
        for (int i = 0; i < (int)durations.size()-1; i++) {
            if (frame_list[i].isFrame24 && frame_list[i+1].isFrame24) {
                if (frame_list[i + 0].duration + frame_list[i + 1].duration == 10) {
                    frame_list[i + 0].duration = 5;
                    frame_list[i + 1].duration = 5;
                    i++;
                }
            }
        }
    }
  	file = std::unique_ptr<TextFile>(new TextFile(filepath + ".timecode.txt", "w", env));
  	double elapsed = 0.0;
  	double tick = (double)vi.fps_denominator / vi.fps_numerator * 0.5;
  	fprintf(file->fp, "# timecode format v2\n");
  	for (int i = 0; i < (int)frame_list.size(); i++) {
  		fprintf(file->fp, "%d\n", (int)std::round(elapsed * 1000));
  		elapsed += frame_list[i].duration * tick;
  	}
    file.reset();
  }

  template <typename pixel_t>
  PVideoFrame GetFrameTop(int n60, PNeoEnv env)
  {
    std::lock_guard<std::mutex> lock(mtxGetFrameTop);
    PDevice cpudev = env->GetDevice(DEV_TYPE_CPU, 0);
    int cycleIndex = n60 / 10;
    KFMResult fm = *(Frame(env->GetFrame(fmclip, cycleIndex, cpudev)).GetReadPtr<KFMResult>());
    FrameInfo info = GetFrameInfo(n60, fm, env);

    Frame dst;
    if (mode != ONLY_FRAME_DURATION) {
      dst = InternalGetFrame<pixel_t>(n60, info, env);

      if (dst.GetProperty("KFM_SourceStart") == nullptr) {
        // �v���p�e�B���Ȃ��ꍇ�͂����Œǉ�����
        int start, end;
        switch (info.baseType) {
        case FRAME_60:
        case FRAME_UCF:
        case FRAME_30:
          start = n60 >> 1;
          end = start + 1;
          break;
        case FRAME_24:
          Frame24Info frameInfo = patterns.GetFrame60(fm.pattern, n60);
					int cycleStart = (n60 / 10) * 5;
          start = cycleStart + frameInfo.fieldStartIndex / 2;
          end = cycleStart + (frameInfo.fieldStartIndex + frameInfo.numFields + 1) / 2;
          break;
        }
        env->MakePropertyWritable(&dst.frame);
        dst.SetProperty("KFM_SourceStart", start);
        dst.SetProperty("KFM_NumSourceFrames", end - start);
      }
    }
    else {
      dst = env->NewVideoFrame(srcvi);
    }

    FrameDurationInfo duration = { 0, false };
    if (mode != NORMAL) {
	  for (; current < n60; ) {
	  	durations.push_back(GetFrameDuration(current, env));
	  	current += durations.back().duration;
	  }
      duration = GetFrameDuration(n60, info, env);
	  if (current == n60) {
	  	durations.push_back(duration);
	  	current += durations.back().duration;
	  }
	  if (current >= vi.num_frames && complete == false) {
	  	// �t�@�C���ɏ�������
	  	WriteToFile(env);
	  	complete = true;
	  }
    }

    if (show) {
      const char* fps = FrameTypeStr(info.baseType);
      char buf[100]; sprintf(buf, "KFMSwitch: %s dur: %d pattern:%2d cost:%.3f", fps, duration.duration, fm.pattern, fm.cost);
      DrawText<pixel_t>(dst.frame, srcvi.BitsPerComponent(), 0, 0, buf, env);
      return dst.frame;
    }

    return dst.frame;
  }

public:
  KFMSwitch(PClip clip60, PClip fmclip,
    PClip clip24, PClip mask24, PClip cc24,
    PClip clip30, PClip mask30, PClip cc30,
    PClip ucfclip,
    float thswitch, int mode, const std::string& filepath, bool is120,
		bool show, bool showflag, IScriptEnvironment* env)
    : KFMFilterBase(clip60, env)
    , srcvi(vi)
    , fmclip(fmclip)
    , clip24(clip24)
    , mask24(mask24)
    , cc24(cc24)
    , clip30(clip30)
    , mask30(mask30)
    , cc30(cc30)
    , ucfclip(ucfclip)
    , thswitch(thswitch)
    , mode(mode)
    , is120(is120)
    , show(show)
    , showflag(showflag)
    , logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
    , logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
		, filepath(GetFullPath(filepath)) // GetFrame���ƃJ�����g�f�B���N�g�����Ⴄ�̂Ńt���p�X�ɂ��Ă���
		, current(0)
		, complete(false)
  {
    if (vi.width & 7) env->ThrowError("[KFMSwitch]: width must be multiple of 8");
    if (vi.height & 7) env->ThrowError("[KFMSwitch]: height must be multiple of 8");

    nBlkX = nblocks(vi.width, OVERLAP);
    nBlkY = nblocks(vi.height, OVERLAP);

    if (mode < 0 || mode > 2) {
      env->ThrowError("[KFMSwitch] mode(%d) must be in range 0-2", mode);
    }

    auto info = CycleAnalyzeInfo::GetParam(fmclip->GetVideoInfo(), env);
    analyzeMode = info->mode;

    // check clip device
    if (!(GetDeviceTypes(fmclip) & DEV_TYPE_CPU)) {
      env->ThrowError("[KFMSwitch]: fmclip must be CPU device");
    }
    if (!(GetDeviceTypes(cc24) & DEV_TYPE_CPU)) {
      env->ThrowError("[KFMSwitch]: cc24 must be CPU device");
    }
    if (!(GetDeviceTypes(cc30) & DEV_TYPE_CPU)) {
      env->ThrowError("[KFMSwitch]: cc30 must be CPU device");
    }

    auto devs = GetDeviceTypes(clip60);
    if (!(GetDeviceTypes(clip24) & devs)) {
      env->ThrowError("[KFMSwitch]: clip24 device unmatch");
    }
    if (!(GetDeviceTypes(clip30) & devs)) {
      env->ThrowError("[KFMSwitch]: clip30 device unmatch");
    }
    if (!(GetDeviceTypes(mask24) & devs)) {
      env->ThrowError("[KFMSwitch]: mask24 device unmatch");
    }
    if (!(GetDeviceTypes(mask30) & devs)) {
      env->ThrowError("[KFMSwitch]: mask30 device unmatch");
    }
    if (ucfclip && !(GetDeviceTypes(ucfclip) & devs)) {
      env->ThrowError("[KFMSwitch]: ucfclip device unmatch");
    }

    // VideoInfo�`�F�b�N
    VideoInfo vi60 = clip60->GetVideoInfo();
    VideoInfo vifm = fmclip->GetVideoInfo();
    VideoInfo vi24 = clip24->GetVideoInfo();
    VideoInfo vimask24 = mask24->GetVideoInfo();
    VideoInfo vicc24 = cc24->GetVideoInfo();
    VideoInfo vi30 = clip30->GetVideoInfo();
    VideoInfo vimask30 = mask30->GetVideoInfo();
    VideoInfo vicc30 = cc30->GetVideoInfo();
    VideoInfo viucf = ucfclip ? ucfclip->GetVideoInfo() : VideoInfo();

    // 24/30�N���b�v�͕�Ԃ��ꂽ60fps������
    is24_60 = (vi24.fps_numerator == vi60.fps_numerator) && (vi24.fps_denominator == vi60.fps_denominator);
    is30_60 = (vi30.fps_numerator == vi60.fps_numerator) && (vi30.fps_denominator == vi60.fps_denominator);

    // fps�`�F�b�N
    if (is24_60 == false) {
      if (vi24.fps_denominator != vimask24.fps_denominator)
        env->ThrowError("[KFMSwitch]: vi24.fps_denominator != vimask24.fps_denominator");
      if (vi24.fps_numerator != vimask24.fps_numerator)
        env->ThrowError("[KFMSwitch]: vi24.fps_numerator != vimask24.fps_numerator");
    }
    if (vicc24.fps_denominator != vimask24.fps_denominator)
      env->ThrowError("[KFMSwitch]: vicc24.fps_denominator != vimask24.fps_denominator");
    if (vicc24.fps_numerator != vimask24.fps_numerator)
      env->ThrowError("[KFMSwitch]: vicc24.fps_numerator != vimask24.fps_numerator");
    if (is30_60 == false) {
      if (vi30.fps_denominator != vimask30.fps_denominator)
        env->ThrowError("[KFMSwitch]: vi30.fps_denominator != vimask30.fps_denominator");
      if (vi30.fps_numerator != vimask30.fps_numerator)
        env->ThrowError("[KFMSwitch]: vi30.fps_numerator != vimask30.fps_numerator");
    }
    if (vicc30.fps_denominator != vimask30.fps_denominator)
      env->ThrowError("[KFMSwitch]: vicc30.fps_denominator != vimask30.fps_denominator");
    if (vicc30.fps_numerator != vimask30.fps_numerator)
      env->ThrowError("[KFMSwitch]: vicc30.fps_numerator != vimask30.fps_numerator");
    if (ucfclip) {
      if (vi60.fps_denominator != viucf.fps_denominator)
        env->ThrowError("[KFMSwitch]: vi60.fps_denominator != viucf.fps_denominator");
      if (vi60.fps_numerator != viucf.fps_numerator)
        env->ThrowError("[KFMSwitch]: vi60.fps_numerator != viucf.fps_numerator");
    }

    // �T�C�Y�`�F�b�N
    if (vi60.width != vi24.width)
      env->ThrowError("[KFMSwitch]: vi60.width != vi24.width");
    if (vi60.height != vi24.height)
      env->ThrowError("[KFMSwitch]: vi60.height != vi24.height");
    if (vi60.width != vimask24.width)
      env->ThrowError("[KFMSwitch]: vi60.width != vimask24.width");
    if (vi60.height != vimask24.height)
      env->ThrowError("[KFMSwitch]: vi60.height != vimask24.height");
    if (vi60.width != vi30.width)
      env->ThrowError("[KFMSwitch]: vi60.width != vi30.width");
    if (vi60.height != vi30.height)
      env->ThrowError("[KFMSwitch]: vi60.height != vi30.height");
    if (vi60.width != vimask30.width)
      env->ThrowError("[KFMSwitch]: vi60.width != vimask30.width");
    if (vi60.height != vimask30.height)
      env->ThrowError("[KFMSwitch]: vi60.height != vimask30.height");
    if (ucfclip) {
      if (vi60.width != viucf.width)
        env->ThrowError("[KFMSwitch]: vi60.width != viucf.width");
      if (vi60.height != viucf.height)
        env->ThrowError("[KFMSwitch]: vi60.height != viucf.height");
    }

    // UCF�N���b�v�`�F�b�N
    if (ucfclip) {
      if (DecombUCFInfo::GetParam(viucf, env)->fpsType != 60)
        env->ThrowError("[KFMSwitch]: Invalid UCF clip (KDecombUCF60 clip is required)");
    }
  }

  PVideoFrame __stdcall GetFrame(int n60, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    int pixelSize = srcvi.ComponentSize();
    switch (pixelSize) {
    case 1:
      return GetFrameTop<uint8_t>(n60, env);
    case 2:
      return GetFrameTop<uint16_t>(n60, env);
    default:
      env->ThrowError("[KFMSwitch] Unsupported pixel format");
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
    return new KFMSwitch(
      args[0].AsClip(),           // clip60
      args[1].AsClip(),           // fmclip
      args[2].AsClip(),           // clip24
      args[3].AsClip(),           // mask24
      args[4].AsClip(),           // cc24
      args[5].AsClip(),           // clip30
      args[6].AsClip(),           // mask30
      args[7].AsClip(),           // cc30
      args[8].Defined() ? args[8].AsClip() : nullptr,           // ucfclip
      (float)args[9].AsFloat(3.0f),// thswitch
			args[10].AsInt(0),           // mode
			args[11].AsString("kfmswitch"),        // filepath
	  args[12].AsBool(false),        // is120
      args[13].AsBool(false),      // show
      args[14].AsBool(false),      // showflag
      env
    );
  }
};

class KFMPad : public KFMFilterBase
{
  VideoInfo srcvi;

  template <typename pixel_t>
  PVideoFrame GetFrameT(int n, PNeoEnv env)
  {
    Frame src = child->GetFrame(n, env);
    Frame dst = Frame(env->NewVideoFrame(vi), VPAD);

    if (true) {
        CopyFrameAndPad<pixel_t>(src, dst, env);
    } else {
        CopyFrame<pixel_t>(src, dst, env);
        PadFrame<pixel_t>(dst, env);
    }

    return dst.frame;
  }
public:
  KFMPad(PClip src, IScriptEnvironment* env)
    : KFMFilterBase(src, env)
    , srcvi(vi)
  {
    if (srcvi.width & 3) env->ThrowError("[KFMPad]: width must be multiple of 4");
    if (srcvi.height & 3) env->ThrowError("[KFMPad]: height must be multiple of 4");

    vi.height += VPAD * 2;
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
      env->ThrowError("[KFMPad] Unsupported pixel format");
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
    return new KFMPad(
      args[0].AsClip(),       // src
      env
    );
  }
};

class KFMDecimate : public GenericVideoFilter
{
  std::vector<int> durations;
  std::vector<int> framesMap;
public:
  KFMDecimate(PClip source, const std::string& filepath, IScriptEnvironment* env)
    : GenericVideoFilter(source)
  {
    auto file = std::unique_ptr<TextFile>(new TextFile(filepath + ".duration.txt", "r", env));
    char buf[100];
    while (fgets(buf, sizeof(buf), file->fp) && *buf) {
      durations.push_back(std::atoi(buf));
    }
    int numSourceFrames = std::accumulate(durations.begin(), durations.end(), 0);
    if (vi.num_frames != numSourceFrames) {
      env->ThrowError("[KFMDecimate] # of frames does not match. %d(%s) vs %d(source clip)",
        (int)numSourceFrames, filepath.c_str(), vi.num_frames);
    }
    vi.num_frames = (int)durations.size();
    framesMap.resize(durations.size());
    framesMap[0] = 0;
    for (int i = 0; i < (int)durations.size() - 1; ++i) {
      framesMap[i + 1] = framesMap[i] + durations[i];
    }
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env)
  {
    return child->GetFrame(framesMap[std::max(0, std::min(n, vi.num_frames - 1))], env);
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    if (cachehints == CACHE_GET_DEV_TYPE) {
      return GetDeviceTypes(child) & (DEV_TYPE_CPU | DEV_TYPE_CUDA);
    }
    return 0;
  };

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KFMDecimate(
      args[0].AsClip(),       // source
      args[1].AsString("kfmswitch"),     // filepath
      env
    );
  }
};


class AssumeDevice : public GenericVideoFilter
{
  int devices;
public:
  AssumeDevice(PClip clip, int devices)
    : GenericVideoFilter(clip)
    , devices(devices)
  { }

  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    if (cachehints == CACHE_GET_DEV_TYPE) {
      return devices;
    }
    else if (cachehints == CACHE_GET_MTMODE) {
      return MT_NICE_FILTER;
    }
    return 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new AssumeDevice(args[0].AsClip(), args[1].AsInt());
  }
};

void AddFuncFMKernel(IScriptEnvironment* env)
{
  env->AddFunction("KPatchCombe", "ccccc", KPatchCombe::Create, 0);
  env->AddFunction("KFMSwitch", "cccccccc[ucfclip]c[thswitch]f[mode]i[filepath]s[is120]b[show]b[showflag]b", KFMSwitch::Create, 0);
  env->AddFunction("KFMPad", "c", KFMPad::Create, 0);
  env->AddFunction("KFMDecimate", "c[filepath]s", KFMDecimate::Create, 0);
  env->AddFunction("AssumeDevice", "ci", AssumeDevice::Create, 0);
}
