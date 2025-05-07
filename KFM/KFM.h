#pragma once

#include "Frame.h"
#include <string>

enum {
  OVERLAP = 8,
  VPAD = 4,

  MOVE = 1,
  SHIMA = 2,
  LSHIMA = 4,

  NUM_PATTERNS = 21,
};

struct FMCount {
  int move, shima, lshima;
};

struct PulldownPatternField {
  bool split; // ���̃t�B�[���h�Ƃ͕ʃt���[��
  bool merge; // 3�t�B�[���h�̍ŏ��̃t�B�[���h
  bool shift; // ���24p�t���[�����Q�Ƃ���t�B�[���h
};

struct PulldownPattern {
  PulldownPatternField fields[10 * 4];
  int cycle;

  PulldownPattern(int nf0, int nf1, int nf2, int nf3); // 24p
  PulldownPattern(); // 30p

  // �p�^�[����10�t�B�[���h+�O��2�t�B�[���h���̍��킹��
  // 14�t�B�[���h�����݂�z��B14�t�B�[���h�̑O���ւ̃|�C���^��Ԃ�
  const PulldownPatternField* GetPattern(int n) const {
    return &fields[10 + n - 2];
  }
  int GetCycleLength() const {
    return cycle;
  }
};

struct Frame24Info {
  int cycleIndex;
  int frameIndex; // �T�C�N�����̃t���[���ԍ�
  int fieldStartIndex; // �\�[�X�t�B�[���h�J�n�ԍ�
  int numFields; // �\�[�X�t�B�[���h��
  int fieldShift; // 2224�p�^�[����2323�ϊ�����ꍇ�̂��炵���K�v�ȃt���[��
};

struct FMData {
  // �ȂƓ����̘a
  float mft[14];
  float mftr[14];
  float mftcost[14];
};

struct FMMatch {
  float shima[NUM_PATTERNS];
  float costs[NUM_PATTERNS];
  float reliability[NUM_PATTERNS];
};

struct KFMResult {
  int pattern;
  int is60p;
  float score;
  float cost;
  float reliability;

  KFMResult() { }

  KFMResult(int pattern, float score, float cost, float reliability)
    : pattern(pattern)
    , is60p()
    , score(score)
    , cost(cost)
    , reliability(reliability)
  { }

  KFMResult(FMMatch& match, int pattern)
    : pattern(pattern)
    , is60p()
    , score(match.shima[pattern])
    , cost(match.costs[pattern])
    , reliability(match.reliability[pattern])
  { }
};

class PulldownPatterns
{
public:
private:
  PulldownPattern p2323, p2233, p2224, p30;
  int patternOffsets[5];
  const PulldownPatternField* allpatterns[NUM_PATTERNS];
public:
  PulldownPatterns();

  const PulldownPatternField* GetPattern(int patternIndex) const {
    return allpatterns[patternIndex];
  }

  const char* PatternToString(int patternIndex, int& index) const;

  // �p�^�[����24fps�̃t���[���ԍ�����t���[�������擾
  Frame24Info GetFrame24(int patternIndex, int n24) const;

  // �p�^�[����60fps�̃t���[���ԍ�����t���[�������擾
  // frameIndex < 0 or frameIndex >= 4�̏ꍇ�A
  // fieldStartIndex��numFields�͐������Ȃ��\��������̂Œ���
  Frame24Info GetFrame60(int patternIndex, int n60) const;

  FMMatch Matching(const FMData& data, int width, int height, float costth, float adj2224, float adj30) const;

  static bool Is30p(int patternIndex) { return patternIndex == NUM_PATTERNS - 1; }
  static bool Is60p(int patternIndex) { return patternIndex == NUM_PATTERNS; }
};

struct CycleAnalyzeInfo {
  enum
  {
    VERSION = 1,
    MAGIC_KEY = 0x6180EDF8,
  };
  int nMagicKey;
  int nVersion;

  int mode;

  CycleAnalyzeInfo(int mode)
    : nMagicKey(MAGIC_KEY)
    , nVersion(VERSION)
    , mode(mode)
  { }

  static const CycleAnalyzeInfo* GetParam(const VideoInfo& vi, PNeoEnv env)
  {
    if (vi.sample_type != MAGIC_KEY) {
      env->ThrowError("Invalid source (sample_type signature does not match)");
    }
    const CycleAnalyzeInfo* param = (const CycleAnalyzeInfo*)(void*)vi.num_audio_samples;
    if (param->nMagicKey != MAGIC_KEY) {
      env->ThrowError("Invalid source (magic key does not match)");
    }
    return param;
  }

  static void SetParam(VideoInfo& vi, const CycleAnalyzeInfo* param)
  {
    vi.audio_samples_per_second = 0; // kill audio
    vi.sample_type = MAGIC_KEY;
    vi.num_audio_samples = (size_t)param;
  }
};

enum {
  COMBE_FLAG_PAD_W = 4,
  COMBE_FLAG_PAD_H = 2,
};

static Frame WrapSwitchFragFrame(const PVideoFrame& frame) {
  return Frame(frame, COMBE_FLAG_PAD_W, COMBE_FLAG_PAD_H, 1);
}

#define DECOMB_UCF_FLAG_STR "KDecombUCF_Flag"

enum DECOMB_UCF_FLAG {
  DECOMB_UCF_NONE,  // ���Ȃ�
  DECOMB_UCF_PREV,  // �O�̃t���[��
  DECOMB_UCF_NEXT,  // ���̃t���[��
  DECOMB_UCF_FIRST, // 1�Ԗڂ̃t�B�[���h��bob
  DECOMB_UCF_SECOND,// 2�Ԗڂ̃t�B�[���h��bob
  DECOMB_UCF_NR,    // �����t���[��
};

struct DecombUCFInfo {
  enum
  {
    VERSION = 1,
    MAGIC_KEY = 0x7080EDF8,
  };
  int nMagicKey;
  int nVersion;

  int fpsType;

  DecombUCFInfo(int fpsType)
    : nMagicKey(MAGIC_KEY)
    , nVersion(VERSION)
    , fpsType(fpsType)
  { }

  static const DecombUCFInfo* GetParam(const VideoInfo& vi, PNeoEnv env)
  {
    if (vi.sample_type != MAGIC_KEY) {
      env->ThrowError("Invalid source (sample_type signature does not match)");
    }
    const DecombUCFInfo* param = (const DecombUCFInfo*)(void*)vi.num_audio_samples;
    if (param->nMagicKey != MAGIC_KEY) {
      env->ThrowError("Invalid source (magic key does not match)");
    }
    return param;
  }

  static void SetParam(VideoInfo& vi, const DecombUCFInfo* param)
  {
    vi.audio_samples_per_second = 0; // kill audio
    vi.sample_type = MAGIC_KEY;
    vi.num_audio_samples = (size_t)param;
  }
};

class TextFile
{
public:
	FILE* fp;
	TextFile(const std::string& fname, const char* mode, IScriptEnvironment* env)
#if defined(_WIN32) || defined(_WIN64)
		: fp(_fsopen(fname.c_str(), mode, _SH_DENYNO))
#else
		: fp(fopen(fname.c_str(), mode))
#endif
	{
		if (fp == nullptr) {
			env->ThrowError("Failed to open file ... %s", fname.c_str());
		}
	}
	~TextFile() {
		fclose(fp);
	}
};

int GetDeviceTypes(const PClip& clip);
bool IsAligned(Frame& frame, const VideoInfo& vi, PNeoEnv env);
