#pragma once

#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>

// int�g���͖����I�ɏ��� //

// to_int(uchar)
static __device__ __host__ int to_int(unsigned char a) {
	return a;
}

// to_int(ushort)
static __device__ __host__ int to_int(unsigned short a) {
	return a;
}

// to_int(uchar4)
static __device__ __host__ int4 to_int(uchar4 a) {
  int4 r = { a.x, a.y, a.z, a.w };
  return r;
}

// to_int(ushort4)
static __device__ __host__ int4 to_int(ushort4 a) {
  int4 r = { a.x, a.y, a.z, a.w };
  return r;
}

// to_int(short4)
static __device__ __host__ int4 to_int(short4 a) {
  int4 r = { a.x, a.y, a.z, a.w };
  return r;
}

// to_int(int4)
static __device__ __host__ int4 to_int(int4 a) {
  return a;
}

// to_int(int4)
static __device__ __host__ int4 to_int(float4 a) {
  int4 r = { (int)a.x, (int)a.y, (int)a.z, (int)a.w };
  return r;
}

// to_float(uchar4)
static __device__ __host__ float4 to_float(uchar4 a) {
  float4 r = { (float)a.x, (float)a.y, (float)a.z, (float)a.w };
  return r;
}

// to_float(ushort4)
static __device__ __host__ float4 to_float(ushort4 a) {
  float4 r = { (float)a.x, (float)a.y, (float)a.z, (float)a.w };
  return r;
}

// to_float(int4)
static __device__ __host__ float4 to_float(int4 a) {
  float4 r = { (float)a.x, (float)a.y, (float)a.z, (float)a.w };
  return r;
}

// int4 + int
static __device__ __host__ int4 operator+(int4 a, int b) {
  int4 r = { a.x + b, a.y + b, a.z + b, a.w + b };
  return r;
}

// float4 + float
static __device__ __host__ float4 operator+(float4 a, float b) {
  float4 r = { a.x + b, a.y + b, a.z + b, a.w + b };
  return r;
}

// int4 + int4
static __device__ __host__ int4 operator+(int4 a, int4 b) {
  int4 r = { a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
  return r;
}

// float4 + float4
static __device__ __host__ float4 operator+(float4 a, float4 b) {
  float4 r = { a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
  return r;
}

// int4 - int4
static __device__ __host__ int4 operator-(int4 a, int4 b) {
  int4 r = { a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w };
  return r;
}

// int - int4
static __device__ __host__ int4 operator-(int a, int4 b) {
  int4 r = { a - b.x, a - b.y, a - b.z, a - b.w };
  return r;
}

// float4 - float4
static __device__ __host__ float4 operator-(float4 a, float4 b) {
  float4 r = { a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w };
  return r;
}

// int4 * int
static __device__ __host__ int4 operator*(int4 a, int b) {
  int4 r = { a.x * b, a.y * b, a.z * b, a.w * b };
  return r;
}

// int4 * int4
static __device__ __host__ int4 operator*(int4 a, int4 b) {
  int4 r = { a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w };
  return r;
}

// float4 * float
static __device__ __host__ float4 operator*(float4 a, float b) {
  float4 r = { a.x * b, a.y * b, a.z * b, a.w * b };
  return r;
}

// float4 * float4
static __device__ __host__ float4 operator*(float4 a, float4 b) {
  float4 r = { a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w };
  return r;
}

// int4 * short4
static __device__ __host__ int4 operator*(int4 a, short4 b) {
  int4 r = { a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w };
  return r;
}

// int4 >> int
static __device__ __host__ int4 operator>>(int4 a, int b) {
  int4 r = { a.x >> b, a.y >> b, a.z >> b, a.w >> b };
  return r;
}

// int4 / int
static __device__ __host__ int4 operator/(int4 a, int b) {
  int4 r = { a.x / b, a.y / b, a.z / b, a.w / b };
  return r;
}

// float4 / float4
static __device__ __host__ float4 operator/(float4 a, float4 b) {
  float4 r = { a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w };
  return r;
}

// int4 += int4
static __device__ __host__ void operator+=(int4& a, int4 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}

// float4 += float4
static __device__ __host__ void operator+=(float4& a, float4 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}

// ushort4 += int4
static __device__ __host__ void operator+=(ushort4& a, int4 b) {
  a.x += (unsigned short)b.x;
  a.y += (unsigned short)b.y;
  a.z += (unsigned short)b.z;
  a.w += (unsigned short)b.w;
}

// ushort4 += ushort4
static __device__ __host__ void operator+=(ushort4& a, ushort4 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}

// min(int4, int)
static __device__ __host__ int4 min(int4 a, int b) {
  int4 r = { min(a.x, b), min(a.y, b), min(a.z, b), min(a.w, b) };
  return r;
}

// min(int4, int4)
static __device__ __host__ int4 min(int4 a, int4 b) {
  int4 r = { min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w) };
  return r;
}

// min(float4, float)
static __device__ __host__ float4 min(float4 a, float b) {
	float4 r = { min(a.x, b), min(a.y, b), min(a.z, b), min(a.w, b) };
	return r;
}

// max(uchar4, uchar4)
static __device__ __host__ uchar4 max(uchar4 a, uchar4 b) {
  uchar4 r = { (unsigned char)max(a.x, b.x), (unsigned char)max(a.y, b.y), (unsigned char)max(a.z, b.z), (unsigned char)max(a.w, b.w) };
  return r;
}

// max(ushort4, ushort4)
static __device__ __host__ ushort4 max(ushort4 a, ushort4 b) {
  ushort4 r = { (unsigned short)max(a.x, b.x), (unsigned short)max(a.y, b.y), (unsigned short)max(a.z, b.z), (unsigned short)max(a.w, b.w) };
  return r;
}

// max(int4, int4)
static __device__ __host__ int4 max(int4 a, int4 b) {
  int4 r = { max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w) };
  return r;
}

// max(float4, float)
static __device__ __host__ float4 max(float4 a, float b) {
  float4 r = { max(a.x, b), max(a.y, b), max(a.z, b), max(a.w, b) };
  return r;
}

// clamp(float4, float, float)
static __device__ __host__ float4 clamp(float4 a, float b, float c) {
  float4 r = { clamp(a.x, b, c), clamp(a.y, b, c), clamp(a.z, b, c), clamp(a.w, b, c) };
  return r;
}

// clamp(int4, int, int)
static __device__ __host__ int4 clamp(int4 a, int b, int c) {
  int4 r = { clamp(a.x, b, c), clamp(a.y, b, c), clamp(a.z, b, c), clamp(a.w, b, c) };
  return r;
}

static __device__ __host__ float4 abs(float4 a) {
  float4 r = { fabsf(a.x), fabsf(a.y), fabsf(a.z), fabsf(a.w) };
  return r;
}

static __device__ __host__ int4 abs(int4 a) {
	int4 r = { abs(a.x), abs(a.y), abs(a.z), abs(a.w) };
	return r;
}

static __device__ __host__ int absdiff(int a, int b) {
  return abs(a - b);
}

static __device__ __host__ int4 absdiff(uchar4 a, uchar4 b) {
  int4 r = { a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w };
  int4 r2 = {
    (r.x >= 0) ? r.x : -r.x,
    (r.y >= 0) ? r.y : -r.y,
    (r.z >= 0) ? r.z : -r.z,
    (r.w >= 0) ? r.w : -r.w
  };
  return r2;
}

static __device__ unsigned int __vabsdiff4(unsigned int srcA, unsigned int srcB, unsigned int c) {
	unsigned int d;
	asm volatile("vabsdiff4.u32.u32.u32.add %0,%1,%2,%3;":"=r"(d) : "r"(srcA), "r"(srcB), "r"(c));
	return d;
}

static __device__ __host__ int4 absdiff(ushort4 a, ushort4 b) {
  int4 r = { a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w };
  int4 r2 = {
    (r.x >= 0) ? r.x : -r.x,
    (r.y >= 0) ? r.y : -r.y,
    (r.z >= 0) ? r.z : -r.z,
    (r.w >= 0) ? r.w : -r.w
  };
  return r2;
}

// �O���[�o���֐��̃I�[�o�[���[�h�ŏ����Ȃ����� //

template <typename V> struct VHelper { };

template <> struct VHelper<unsigned char> {
	enum { VLEN = 1 };
  static __device__ __host__ unsigned char make(int a) {
    return (unsigned char)a;
  }
};

template <> struct VHelper<unsigned short> {
	enum { VLEN = 1 };
  static __device__ __host__ unsigned short make(int a) {
    return (unsigned short)a;
  }
};

template <> struct VHelper<int> {
	enum { VLEN = 1 };
	static __device__ __host__ unsigned short make(int a) {
		return (unsigned short)a;
	}
};

template <> struct VHelper<uchar4> {
	enum { VLEN = 4 };
  static __device__ __host__ uchar4 make(int a) {
    typedef unsigned char uchar;
    uchar4 r = { (uchar)a, (uchar)a, (uchar)a, (uchar)a };
    return r;
  }
  static __device__ __host__ uchar4 cast_to(int4 a) {
    typedef unsigned char uchar;
    uchar4 r = { (uchar)a.x, (uchar)a.y, (uchar)a.z, (uchar)a.w };
    return r;
  }
  static __device__ __host__ uchar4 cast_to(float4 a) {
    typedef unsigned char uchar;
    uchar4 r = { (uchar)a.x, (uchar)a.y, (uchar)a.z, (uchar)a.w };
    return r;
  }
};

template <> struct VHelper<ushort4> {
	enum { VLEN = 4 };
  static __device__ __host__ ushort4 make(int a) {
    typedef unsigned short ushort;
    ushort4 r = { (ushort)a, (ushort)a, (ushort)a, (ushort)a };
    return r;
  }
  static __device__ __host__ ushort4 cast_to(int4 a) {
    typedef unsigned short ushort;
    ushort4 r = { (ushort)a.x, (ushort)a.y, (ushort)a.z, (ushort)a.w };
    return r;
  }
  static __device__ __host__ ushort4 cast_to(float4 a) {
    typedef unsigned short ushort;
    ushort4 r = { (ushort)a.x, (ushort)a.y, (ushort)a.z, (ushort)a.w };
    return r;
  }
};

template <> struct VHelper<int4> {
	enum { VLEN = 4 };
  static __device__ __host__ int4 make(int a) {
    int4 r = { a, a, a, a };
    return r;
  }
};

template <> struct VHelper<float4> {
	enum { VLEN = 4 };
  static __device__ __host__ float4 make(float a) {
    float4 r = { a, a, a, a };
    return r;
  }
  static __device__ __host__ float4 cast_to(float4 a) {
    return a;
  }
};

template <int VLEN> struct VLoad { };

template <> struct VLoad<1> {
	static __device__ int to_int(const unsigned char* p) {
		return __ldg(&p[0]);
	}
	static __device__ int to_int(const unsigned short* p) {
		return __ldg(&p[0]);
	}
	static __device__ int to_int(const int* p) {
		return __ldg(&p[0]);
	}
};

template <> struct VLoad<4> {
	static __device__ int4 to_int(const unsigned char* p) {
		int4 r = { __ldg(&p[0]), __ldg(&p[1]), __ldg(&p[2]), __ldg(&p[3]) };
		return r;
	}
	static __device__ int4 to_int(const unsigned short* p) {
		int4 r = { __ldg(&p[0]), __ldg(&p[1]), __ldg(&p[2]), __ldg(&p[3]) };
		return r;
	}
	static __device__ int4 to_int(const int* p) {
		int4 r = { __ldg(&p[0]), __ldg(&p[1]), __ldg(&p[2]), __ldg(&p[3]) };
		return r;
	}
};

template <typename T> struct VectorType {};

template <> struct VectorType<unsigned char> {
  typedef uchar4 type;
};

template <> struct VectorType<unsigned short> {
  typedef ushort4 type;
};

template <> struct VectorType<float> {
  typedef float4 type;
};
