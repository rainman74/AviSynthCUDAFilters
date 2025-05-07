
#include <stdint.h>
#include <avisynth.h>

#include <algorithm>
#include <numeric>
#include <memory>
#include <vector>
#include <deque>
#include <cmath>

#include "CommonFunctions.h"
#include "TextOut.h"
#include "Frame.h"
#include "KMV.h"
#include "KFM.h"
#include "Copy.h"
#include "KUtil.h"
#include "rgy_filesystem.h"

void OnCudaError(cudaError_t err) {
#if 1 // �f�o�b�O�p�i�{�Ԃ͎�菜���j
  printf("[CUDA Error] %s (code: %d)\n", cudaGetErrorString(err), err);
#endif
}

bool IsAligned(Frame& frame, const VideoInfo& vi, PNeoEnv env) {
	int frame_align = (int)env->GetProperty(AEP_FRAME_ALIGN);
	int plane_align = (int)env->GetProperty(AEP_PLANE_ALIGN);

	const int planesYUV[] = { 0/*PLANAR_Y*/, PLANAR_U, PLANAR_V, PLANAR_A };
	const int planesRGB[] = { 0/*PLANAR_G*/, PLANAR_B, PLANAR_R, PLANAR_A };
	const int* planes = vi.IsRGB() ? planesRGB : planesYUV;
	int numPlanes = vi.IsPlanar() ? vi.NumComponents() : 1;
	for (int p = 0; p < numPlanes; p++) {
		const int plane = planes[p];
		const BYTE* ptr = frame.GetReadPtr<BYTE>(plane);
		int pitch = frame.GetPitch<BYTE>(plane);
		if ((uintptr_t)ptr & (plane_align - 1)) return false;
		if (pitch & (frame_align - 1)) return false;
	}

	return true;
}

class File
{
public:
  File(const std::string& path, const char* mode, IScriptEnvironment* env) {
#if defined(_WIN32) || defined(_WIN64)
    fp_ = _fsopen(path.c_str(), mode, _SH_DENYNO);
#else
    fp_ = fopen(path.c_str(), mode);
#endif
    if (fp_ == NULL) {
      env->ThrowError("failed to open file %s", path.c_str());
    }
  }
  ~File() {
    fclose(fp_);
  }
  void write(uint8_t* ptr, size_t sz, IScriptEnvironment* env) const {
    if (sz == 0) return;
    if (fwrite(ptr, sz, 1, fp_) != 1) {
      env->ThrowError("failed to write to file");
    }
  }
  template <typename T>
  void writeValue(T v, IScriptEnvironment* env) const {
    write((uint8_t*)&v, sizeof(T), env);
  }
  size_t read(uint8_t* ptr, size_t sz, IScriptEnvironment* env) const {
    if (sz == 0) return 0;
    size_t ret = fread(ptr, 1, sz, fp_);
    if (ret <= 0) {
      env->ThrowError("failed to read from file");
    }
    return ret;
  }
  template <typename T>
  T readValue(IScriptEnvironment* env) const {
    T v;
    if (read((uint8_t*)&v, sizeof(T), env) != sizeof(T)) {
      env->ThrowError("failed to read value from file");
    }
    return v;
  }
  static bool exists(const std::string& path) {
#if defined(_WIN32) || defined(_WIN64)
    FILE* fp_ = _fsopen(path.c_str(), "rb", _SH_DENYNO);
#else
    FILE* fp_ = fopen(path.c_str(), "rb");
#endif
    if (fp_) {
      fclose(fp_);
      return true;
    }
    return false;
  }
private:
  FILE * fp_;
};

int GetDeviceTypes(const PClip& clip)
{
  int devtypes = (clip->GetVersion() >= 5) ? clip->SetCacheHints(CACHE_GET_DEV_TYPE, 0) : 0;
  if (devtypes == 0) {
    return DEV_TYPE_CPU;
  }
  return devtypes;
}

int GetCommonDevices(const std::vector<PClip>& clips)
{
  int devs = DEV_TYPE_ANY;
  for (const PClip& c : clips) {
    if (c) {
      devs &= GetDeviceTypes(c);
    }
  }
  return devs;
}

class KShowStatic : public GenericVideoFilter
{
  typedef uint8_t pixel_t;

  PClip sttclip;

  int logUVx;
  int logUVy;

  void CopyFrame(Frame& src, Frame& dst, PNeoEnv env)
  {
    const pixel_t* srcY = src.GetReadPtr<pixel_t>(PLANAR_Y);
    const pixel_t* srcU = src.GetReadPtr<pixel_t>(PLANAR_U);
    const pixel_t* srcV = src.GetReadPtr<pixel_t>(PLANAR_V);
    pixel_t* dstY = dst.GetWritePtr<pixel_t>(PLANAR_Y);
    pixel_t* dstU = dst.GetWritePtr<pixel_t>(PLANAR_U);
    pixel_t* dstV = dst.GetWritePtr<pixel_t>(PLANAR_V);

    int pitchY = src.GetPitch<pixel_t>(PLANAR_Y);
    int pitchUV = src.GetPitch<pixel_t>(PLANAR_U);
    int widthUV = vi.width >> logUVx;
    int heightUV = vi.height >> logUVy;

    Copy(dstY, pitchY, srcY, pitchY, vi.width, vi.height, env);
    Copy(dstU, pitchUV, srcU, pitchUV, widthUV, heightUV, env);
    Copy(dstV, pitchUV, srcV, pitchUV, widthUV, heightUV, env);
  }

  void MaskFill(pixel_t* dstp, int dstPitch,
    const uint8_t* flagp, int flagPitch, int width, int height, int val)
  {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int coef = flagp[x + y * flagPitch];
        pixel_t& v = dstp[x + y * dstPitch];
        v = (coef * v + (128 - coef) * val + 64) >> 7;
      }
    }
  }

  void VisualizeBlock(Frame& flag, Frame& dst)
  {
    const pixel_t* flagY = flag.GetReadPtr<pixel_t>(PLANAR_Y);
    const pixel_t* flagU = flag.GetReadPtr<pixel_t>(PLANAR_U);
    const pixel_t* flagV = flag.GetReadPtr<pixel_t>(PLANAR_V);
    pixel_t* dstY = dst.GetWritePtr<pixel_t>(PLANAR_Y);
    pixel_t* dstU = dst.GetWritePtr<pixel_t>(PLANAR_U);
    pixel_t* dstV = dst.GetWritePtr<pixel_t>(PLANAR_V);

    int flagPitchY = flag.GetPitch<uint8_t>(PLANAR_Y);
    int flagPitchUV = flag.GetPitch<uint8_t>(PLANAR_U);
    int dstPitchY = dst.GetPitch<pixel_t>(PLANAR_Y);
    int dstPitchUV = dst.GetPitch<pixel_t>(PLANAR_U);
    int widthUV = vi.width >> logUVx;
    int heightUV = vi.height >> logUVy;

    int blue[] = { 73, 230, 111 };

    MaskFill(dstY, dstPitchY, flagY, flagPitchY, vi.width, vi.height, blue[0]);
    MaskFill(dstU, dstPitchUV, flagU, flagPitchUV, widthUV, heightUV, blue[1]);
    MaskFill(dstV, dstPitchUV, flagV, flagPitchUV, widthUV, heightUV, blue[2]);
  }

public:
  KShowStatic(PClip sttclip, PClip clip30, PNeoEnv env)
    : GenericVideoFilter(clip30)
    , sttclip(sttclip)
    , logUVx(vi.GetPlaneWidthSubsampling(PLANAR_U))
    , logUVy(vi.GetPlaneHeightSubsampling(PLANAR_U))
  {
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;

    Frame flag = sttclip->GetFrame(n, env);
    Frame frame30 = child->GetFrame(n, env);
    Frame dst = env->NewVideoFrame(vi);

    CopyFrame(frame30, dst, env);
    VisualizeBlock(flag, dst);

    return dst.frame;
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    if (cachehints == CACHE_GET_MTMODE) {
      return MT_NICE_FILTER;
    }
    return 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;
    return new KShowStatic(
      args[0].AsClip(),       // sttclip
      args[1].AsClip(),       // clip30
      env);
  }
};

float RSplitScore(const PulldownPatternField* pattern, const float* fv) {
  float sumsplit = 0, sumnsplit = 0;

  for (int i = 0; i < 14; ++i) {
    if (pattern[i].split) {
      sumsplit += fv[i];
    }
    else {
      sumnsplit += fv[i];
    }
  }

  return sumsplit - sumnsplit;
}

float RSplitCost(const PulldownPatternField* pattern, const float* fv, const float* fvcost, float costth) {
  int nsplit = 0;
  float sumcost = 0;

  for (int i = 0; i < 14; ++i) {
    if (pattern[i].split) {
      nsplit++;
      if (fv[i] < costth) {
        // ���ΓI��fv���d���������̂ŁAfvcost��log�ŗ}����
        //�u�S�؂�ւ��|�C���g��fv���������v->��i�������������S��������Ώ��j
        //�u����؂�ւ��|�C���g�ł̂݃m�C�Y�������v->�� �ɂ�����
        sumcost += (costth - fv[i]) * std::log2f(fvcost[i] + 1.0f);
      }
    }
  }

  return sumcost / nsplit;
}

float RSplitReliability(const PulldownPatternField* pattern, const float* fv, float costth) {
  int nsplit = 0;
  float sumcost = 0;

  for (int i = 0; i < 14; ++i) {
    if (pattern[i].split) {
      nsplit++;
      if (fv[i] < costth) {
        sumcost += (costth - fv[i]);
      }
    }
  }

  return sumcost / nsplit;
}

#pragma region PulldownPatterns

PulldownPattern::PulldownPattern(int nf0, int nf1, int nf2, int nf3)
  : fields()
  , cycle(10)
{
  // 24p
  if (nf0 + nf1 + nf2 + nf3 != 10) {
    printf("Error: sum of nfields must be 10.\n");
  }
  if (nf0 == nf2 && nf1 == nf3) {
    cycle = 5;
  }
  int nfields[] = { nf0, nf1, nf2, nf3 };
  for (int c = 0, fstart = 0; c < 4; ++c) {
    for (int i = 0; i < 4; ++i) {
      int nf = nfields[i];
      for (int f = 0; f < nf - 2; ++f) {
        fields[fstart + f].merge = true;
      }
      fields[fstart + nf - 1].split = true;

      // �ȂȂ�24fps�Ή�
      if (nf0 == 4 && nf1 == 2 && nf2 == 2 && nf3 == 2) {
        if (i < 2) {
          fields[fstart + nf - 1].shift = true;
        }
      }

      fstart += nf;
    }
  }
}

PulldownPattern::PulldownPattern()
  : fields()
  , cycle(2)
{
  // 30p
  for (int c = 0, fstart = 0; c < 4; ++c) {
    for (int i = 0; i < 4; ++i) {
      int nf = 2;
      fields[fstart + nf - 1].split = true;
      fstart += nf;
    }
  }
}

PulldownPatterns::PulldownPatterns()
  : p2323(2, 3, 2, 3)
  , p2233(2, 2, 3, 3)
  , p2224(4, 2, 2, 2)
  , p30()
{
  const PulldownPattern* patterns[] = { &p2323, &p2233, &p2224, &p30 };
  int steps[] = { 1, 1, 2, 2 };

  int pi = 0;
  for (int p = 0; p < 4; ++p) {
    patternOffsets[p] = pi;
    for (int i = 0; i < patterns[p]->GetCycleLength(); i += steps[p]) {
      allpatterns[pi++] = patterns[p]->GetPattern(i);
    }
  }
  patternOffsets[4] = pi;

  if (pi != NUM_PATTERNS) {
    throw "Error !!!";
  }
}

const char* PulldownPatterns::PatternToString(int patternIndex, int& index) const
{
  const char* names[] = { "2323", "2233", "2224", "30p" };
  auto pattern = std::upper_bound(patternOffsets, patternOffsets + 4, patternIndex) - patternOffsets - 1;
  index = patternIndex - patternOffsets[pattern];
  return names[pattern];
}

Frame24Info PulldownPatterns::GetFrame24(int patternIndex, int n24) const {
  Frame24Info info = { 0 };
  info.cycleIndex = n24 / 4;
  info.frameIndex = n24 % 4;

  int searchFrame = info.frameIndex;

  // �p�^�[����30p�̏ꍇ�́A5�����^�񒆂�1��������
  // 30p�̏ꍇ�́A24p�ɂ������_��5����1���������Ă��܂��̂ŁA������60p�ɕ������邱�Ƃ͂ł��Ȃ�
  // 30p������60p�N���b�v����擾�����̂Ŋ�{�I�ɂ͖��Ȃ����A
  // �O��̃T�C�N����24p�ŁA�T�C�N�����E�̋󂫃t���[���Ƃ���30p�������擾����邱�Ƃ�����
  // �Ȃ̂ŁA5�����A�ŏ��ƍŌ�̃t���[�������͐�����60p�ɕ�������K�v������
  // �ȉ��̏������Ȃ��ƍŌ�̃t���[��(4����)���Y���Ă��܂�
  if (Is30p(patternIndex)) {
    if (searchFrame >= 2) ++searchFrame;
  }

  const PulldownPatternField* ptn = allpatterns[patternIndex];
  int fldstart = 0;
  int nframes = 0;
  for (int i = 0; i < 16; ++i) {
    if (ptn[i].split) {
      if (fldstart >= 1) {
        if (nframes++ == searchFrame) {
          int nextfldstart = i + 1;
          info.fieldStartIndex = fldstart - 2;
          info.numFields = nextfldstart - fldstart;
          return info;
        }
      }
      fldstart = i + 1;
    }
  }

  throw "Error !!!";
}

Frame24Info PulldownPatterns::GetFrame60(int patternIndex, int n60) const {
  Frame24Info info = { 0 };
  info.cycleIndex = n60 / 10;

  // split�Ńt���[���̐؂�ւ�������
  // split�͑O�̃t���[���̍Ō�̃t�B�[���h�ƂȂ�̂�
  // -2�t�B�[���h�ڂ��猩��ƌ��o�ł���ŏ��̃t���[����
  // �ł�������-1�t�B�[���h�X�^�[�g
  // �����O��ɃA���S���Y�����g�܂�Ă��邱�Ƃɒ���

  const PulldownPatternField* ptn = allpatterns[patternIndex];
  int fldstart = 0;
  int nframes = -1;
  int findex = n60 % 10;

  // �ő�4�t�B�[���h������̂ŁA2+10+4�����񂵂Ă�
  for (int i = 0; i < 16; ++i) {
    if (ptn[i].split) {
      if (fldstart >= 1) {
        ++nframes;
      }
      int nextfldstart = i + 1;
      if (findex < nextfldstart - 2) {
        info.frameIndex = nframes;
        info.fieldStartIndex = fldstart - 2;
        info.numFields = nextfldstart - fldstart;
        info.fieldShift = ptn[findex + 2].shift ? 1 : 0;
        return info;
      }
      fldstart = i + 1;
    }
  }

  throw "Error !!!";
}

FMMatch PulldownPatterns::Matching(const FMData& data, int width, int height, float costth, float adj2224, float adj30) const
{
  FMMatch match;

  // �e�X�R�A���v�Z
  for (int i = 0; i < NUM_PATTERNS; ++i) {
    match.shima[i] = RSplitScore(allpatterns[i], data.mftr);
    match.costs[i] = RSplitCost(allpatterns[i], data.mftr, data.mftcost, costth);
    match.reliability[i] = RSplitReliability(allpatterns[i], data.mftr, costth);
  }

  // ����
  for (int i = patternOffsets[2]; i < patternOffsets[3]; ++i) {
    match.shima[i] -= adj2224;
  }
  for (int i = patternOffsets[3]; i < patternOffsets[4]; ++i) {
    match.shima[i] -= adj30;
  }

  return match;
}

#pragma endregion

class KFMCycleAnalyze : public GenericVideoFilter
{

  enum Mode {
    REALTIME = 0,
    GEN_PATTERN = 1,
    READ_PATTERN = 2,
  };

  PClip source;
  VideoInfo srcvi;
  int numCycles;
  PulldownPatterns patterns;
  int mode; // 0:���A���^�C���ŗ�, 1:1�p�X��, 2:2�p�X��

  CycleAnalyzeInfo info;

  // ��{�p�����[�^
  float lscale;
  float costth;
  float adj2224;
  float adj30;

  // 2�p�X�p
  int cycleRange;
  float NGThresh;
  int pastCycles;

  // 60p����p
  float th60;  // 60p���肵�����l
  float th24;  // 24p���肵�����l
  float rel24; // 24p����M�����������l

  std::string filepath;
  int debug;

  std::vector<KFMResult> results;
  std::unique_ptr<TextFile> debugFile;

  // �e���|����
  std::deque<KFMResult> recentBest;
  int pattern;
  int current;

  PVideoFrame MakeFrame(KFMResult result, IScriptEnvironment* env)
  {
    Frame dst = env->NewVideoFrame(vi);
    *dst.GetWritePtr<KFMResult>() = result;
    return dst.frame;
  }

  FMData GetFMData(int cycle, IScriptEnvironment* env)
  {
    FMCount fmcnt[18];
    for (int i = -2; i <= 6; ++i) {
      Frame frame = child->GetFrame(cycle * 5 + i, env);
      memcpy(fmcnt + (i + 2) * 2, frame.GetReadPtr<uint8_t>(), sizeof(fmcnt[0]) * 2);
    }

    // shima, lshima, move�̉�f�����}�`�}�`�Ȃ̂ő傫���̈Ⴂ�ɂ��d�݂̈Ⴂ���o��
    // shima, lshima��move�ɍ��킹��i���ς������ɂȂ�悤�ɂ���j

    int mft[18] = { 0 };
    for (int i = 1; i < 17; ++i) {
      int split = std::min(fmcnt[i - 1].move, fmcnt[i].move);
      mft[i] = split + fmcnt[i].shima + (int)(fmcnt[i].lshima * lscale);
    }

    FMData data = { 0 };
    int vbase = (int)(srcvi.width * srcvi.height * 0.001f) >> 4;
    for (int i = 0; i < 14; ++i) {
      data.mft[i] = (float)mft[i + 2];
      data.mftr[i] = (mft[i + 2] + vbase) * 2.0f / (mft[i + 1] + mft[i + 3] + vbase * 2.0f) - 1.0f;
      data.mftcost[i] = (float)(mft[i + 1] + mft[i + 3]) / vbase;
    }

    return data;
  }

  int BestPattern(FMMatch& match)
  {
    auto it = std::max_element(match.shima, match.shima + NUM_PATTERNS);
    return (int)(it - match.shima);
  }

  PVideoFrame RealTimeGetFrame(int cycle, IScriptEnvironment* env)
  {
    auto result = patterns.Matching(GetFMData(cycle, env),
      srcvi.width, srcvi.height, costth, adj2224, adj30);
    return MakeFrame(KFMResult(result, BestPattern(result)), env);
  }

  void Make60p()
  {
    bool is60p = true;
    for (int i = 0; i < (int)results.size(); ++i) {
      auto& cur = results[i];
      if (is60p) {
        // 60p���[�h
        if (cur.cost < th24) {
          if (cur.reliability < rel24) {
            // 24p�ֈڍs
            is60p = false;
          }
        }
        else {
          cur.is60p = true;
        }
      }
      else {
        // 24p���[�h
        if (cur.cost >= th60) {
          // 60p�ֈڍs
          is60p = true;
          // �k����th24�ȏ�Ȃ�60p��
          for (int t = i; t >= 0; --t) {
            auto& cur = results[t];
            if (cur.cost < th24) {
              if (cur.reliability < rel24) {
                // 24p�ֈڍs����|�C���g
                break;
              }
            }
            else {
              cur.is60p = true;
            }
          }
        }
      }
    }
  }

  PVideoFrame ExecuteOnePath(int cycle, IScriptEnvironment* env)
  {
    if (cycle < (int)results.size()) {
      return MakeFrame(results[cycle], env);
    }

    FMMatch match = { 0 };
    int last = (cycle == numCycles - 1) ? (numCycles + cycleRange) : (cycle + 1);
    for (; current < last; ++current) {
      if (current < numCycles) {
        match = patterns.Matching(GetFMData(current, env),
          srcvi.width, srcvi.height, costth, adj2224, adj30);
      }

      results.emplace_back(KFMResult(match, pattern));
      recentBest.emplace_front(KFMResult(match, BestPattern(match)));

      if (debug) {
        auto cur = results.back();
        auto best = recentBest[0];
        fprintf(debugFile->fp, "%d,%d,%.2f,%.2f,%.2f,%d,%.2f,%.2f,%.2f", current,
          cur.pattern, cur.score, cur.cost, cur.reliability,
          best.pattern, best.score, best.cost, best.reliability);
      }

      if (results.back().pattern != recentBest[0].pattern) {
        // ���݂̃p�^�[�����ŗǃp�^�[���łȂ��Ȃ�p�^�[���؂�ւ�����
        float NGScore = 0;
        for (int i = 0; i < std::min(cycleRange, (int)recentBest.size()); ++i) {
          NGScore += recentBest[i].score - results[current - i].score;
        }
        if (debug) {
          fprintf(debugFile->fp, ",%.2f,%s", NGScore, (NGScore > NGThresh) ? "@" : "-");
        }
        if (NGScore > NGThresh) {
          // �p�^�[���؂�ւ�
          pattern = recentBest[0].pattern;

          // �k���Đ؂�ւ���̃p�^�[�����ŗǃp�^�[���Ȃ珑������
          for (int i = 0; i < (int)recentBest.size(); ++i) {
            if (recentBest[i].pattern != pattern) {
              break;
            }
            results[current - i] = recentBest[i];
          }
        }
      }

      if (debug) {
        fprintf(debugFile->fp, "\n");
      }

      if ((int)recentBest.size() > pastCycles) {
        recentBest.pop_back();
      }
    }

    if (last == numCycles + cycleRange) {
      // 60p����
      Make60p();
      // �Ō�̓t�@�C���ɏ�������
      File file(filepath + ".result.dat", "wb", env);
      for (int i = 0; i < numCycles; ++i) {
        file.writeValue(results[i], env);
      }
      if (debug) {
        debugFile = nullptr;
        auto file = std::unique_ptr<TextFile>(new TextFile(filepath + ".pattern.txt", "w", env));
        int cur = -1;
        for (int i = 0; i < numCycles; ++i) {
          int next = results[i].is60p ? NUM_PATTERNS : results[i].pattern;
          if (cur != next) {
            fprintf(file->fp, "%d,%d\n", i * 5, next);
            cur = next;
          }
        }
      }
    }

    return MakeFrame(results[cycle], env);
  }

public:
  KFMCycleAnalyze(PClip fmframe, PClip source, int mode,
    float lscale, float costth, float adj2224, float adj30,
    int cycleRange, float NGThresh, int pastCycles,
    float th60, float th24, float rel24,
    const std::string& filepath, int debug, IScriptEnvironment* env)
    : GenericVideoFilter(fmframe)
    , source(source)
    , srcvi(source->GetVideoInfo())
    , numCycles(nblocks(vi.num_frames, 5))
    , mode(mode)
    , info(mode)
    , lscale(lscale)
    , costth(costth)
    , adj2224(adj2224)
    , adj30(adj30)
    , cycleRange(cycleRange)
    , NGThresh(NGThresh * cycleRange)
    , pastCycles(pastCycles)
    , th60(th60)
    , th24(th24)
    , rel24(rel24)
    , filepath(GetFullPathFrom(filepath.c_str())) // GetFrame���ƃJ�����g�f�B���N�g�����Ⴄ�̂Ńt���p�X�ɂ��Ă���
    , debug(debug)
    , pattern(0)
    , current(0)
  {
    int out_bytes = sizeof(KFMResult);
    vi.pixel_type = VideoInfo::CS_BGR32;
    vi.width = 4;
    vi.height = nblocks(out_bytes, vi.width * 4);
    vi.num_frames = numCycles;

    if (mode < 0 || mode > 2) {
      env->ThrowError("[KFMCycleAnalyze] mode(%d) must be in range 0-2", mode);
    }

    if (mode == GEN_PATTERN) {
      if (debug) {
        debugFile = std::unique_ptr<TextFile>(new TextFile(filepath + ".debug.txt", "w", env));
      }
    }
    else if (mode == READ_PATTERN) {
      File file(filepath + ".result.dat", "rb", env);
      for (int i = 0; i < numCycles; ++i) {
        results.push_back(file.readValue<KFMResult>(env));
      }
      if ((int)results.size() != numCycles) {
        env->ThrowError("[KFMCycleAnalyze] # of cycles does not match. please generate again.");
      }
    }

    CycleAnalyzeInfo::SetParam(vi, &info);
  }

  PVideoFrame __stdcall GetFrame(int cycle, IScriptEnvironment* env)
  {
    if (mode == REALTIME) {
      return RealTimeGetFrame(cycle, env);
    }
    else if (mode == GEN_PATTERN) {
      return ExecuteOnePath(cycle, env);
    }
    else if (mode == READ_PATTERN) {
      return MakeFrame(results[cycle], env);
    }
    return PVideoFrame();
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    if (cachehints == CACHE_GET_MTMODE) {
      return MT_SERIALIZED;
    }
    return 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    PClip src = args[1].AsClip();
    float costthDef = (src->GetVideoInfo().height >= 720) ? 1.5f : 1.0f;
    return new KFMCycleAnalyze(
      args[0].AsClip(),       // fmframe
      src,                    // source
      args[2].AsInt(0),       // mode
      (float)args[3].AsFloat(5.0f), // lscale
      (float)args[4].AsFloat(costthDef), // costth
      (float)args[5].AsFloat(0.5f), // adj2224
      (float)args[6].AsFloat(1.5f), // adj30
      args[7].AsInt(5),             // range
      (float)args[8].AsFloat(1.0f), // thresh
      args[9].AsInt(180),           // past
      (float)args[10].AsFloat(3.0f),           // th60
      (float)args[11].AsFloat(0.1f),           // th30
      (float)args[12].AsFloat(0.2f),           // rell24
      args[13].AsString("kfm"),                // filepath
      args[14].AsInt(0),           // debug
      env
    );
  }
};

class Print : public GenericVideoFilter
{
  std::string str;
  int x, y;
public:
  Print(PClip clip, const char* str, int x, int y, IScriptEnvironment* env)
    : GenericVideoFilter(clip)
    , str(str)
    , x(x)
    , y(y)
  {
    int cs = vi.ComponentSize();
    if (cs != 1 && cs != 2)
      env->ThrowError("[Print] Unsupported pixel format");
    if (vi.IsRGB() || !vi.IsPlanar())
      env->ThrowError("[Print] Unsupported pixel format");
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    PNeoEnv env = env_;
    PVideoFrame src = child->GetFrame(n, env);

    switch (vi.ComponentSize()) {
    case 1:
      DrawText<uint8_t>(src, vi.BitsPerComponent(), x, y, str, env);
      break;
    case 2:
      DrawText<uint16_t>(src, vi.BitsPerComponent(), x, y, str, env);
      break;
    }

    return src;
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
    return new Print(
      args[0].AsClip(),      // clip
      args[1].AsString(),    // str
      args[2].AsInt(0),       // x
      args[3].AsInt(0),       // y
      env
    );
  }
};

class KFMDumpFM : public GenericVideoFilter
{
  int nOut;
  std::unique_ptr<TextFile> outFile;
public:
  KFMDumpFM(PClip fmframe, const std::string& filepath, IScriptEnvironment* env)
    : GenericVideoFilter(fmframe)
    , nOut(0)
  {
    outFile = std::unique_ptr<TextFile>(new TextFile(filepath, "w", env));
    fprintf(outFile->fp, "#shima,large shima,move\n");
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env)
  {
    if (n < nOut) {
      return child->GetFrame(n, env);
    }

    FMMatch match = { 0 };
    for (int current = n; current < n + 1; ++current) {
      Frame frame = child->GetFrame(n, env);
      const FMCount* ptr = frame.GetReadPtr<FMCount>();
      for (int i = 0; i < 2; ++i) {
        fprintf(outFile->fp, "%d,%d,%d\n", ptr[i].shima, ptr[i].lshima, ptr[i].move);
      }
      ++nOut;
    }

    if (nOut == vi.num_frames) {
      // �Ō�܂�GetFrame����
      outFile = nullptr;
    }

    return child->GetFrame(n, env);
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    if (cachehints == CACHE_GET_DEV_TYPE) {
      return GetDeviceTypes(child) &
        (DEV_TYPE_CPU | DEV_TYPE_CUDA);
    }
    else if (cachehints == CACHE_GET_MTMODE) {
      return MT_SERIALIZED;
    }
    return 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env)
  {
    return new KFMDumpFM(
      args[0].AsClip(),      // clip
      args[1].AsString("kfm.txt"),    // filepath
      env
    );
  }
};

void AddFuncFM(IScriptEnvironment* env)
{
  env->AddFunction("KShowStatic", "cc", KShowStatic::Create, 0);

  env->AddFunction("KFMCycleAnalyze", "cc[mode]i[lscale]f[costth]f[adj2224]f[adj30]f[range]i[thresh]f[past]i[th60]f[th24]f[rel24]f[filepath]s[debug]i", KFMCycleAnalyze::Create, 0);
  env->AddFunction("Print", "cs[x]i[y]i", Print::Create, 0);

  env->AddFunction("KFMDumpFM", "c[filepath]s", KFMDumpFM::Create, 0);
}

#include "rgy_osdep.h"

void AddFuncFMKernel(IScriptEnvironment* env);
void AddFuncMergeStatic(IScriptEnvironment* env);
void AddFuncCombingAnalyze(IScriptEnvironment* env);
void AddFuncDebandKernel(IScriptEnvironment* env);
void AddFuncUCF(IScriptEnvironment* env);
void AddFuncDeblock(IScriptEnvironment* env);

#if 0
static void init_console()
{
  AllocConsole();
  freopen("CONOUT$", "w", stdout);
  freopen("CONIN$", "r", stdin);
}
#endif

const AVS_Linkage *AVS_linkage = 0;

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors)
{
  AVS_linkage = vectors;
  //init_console();

  AddFuncFM(env);
  AddFuncFMKernel(env);
  AddFuncMergeStatic(env);
  AddFuncCombingAnalyze(env);
  AddFuncDebandKernel(env);
  AddFuncUCF(env);
  AddFuncDeblock(env);

  return "K Field Matching Plugin";
}
