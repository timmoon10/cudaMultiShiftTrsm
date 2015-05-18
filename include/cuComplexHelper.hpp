#include <cuComplex.h>

#ifndef __CUCOMPLEXHELPER_HPP__
#define __CUCOMPLEXHELPER_HPP__

// ===============================================
// Overloading operators
// ===============================================
__host__ __device__ inline
cuFloatComplex operator+(const cuFloatComplex & x, const cuFloatComplex & y) {
  return cuCaddf(x,y);
}
__host__ __device__ inline
cuFloatComplex operator+(const float x, const cuFloatComplex & y) {
  return make_cuFloatComplex(x+cuCrealf(y), cuCimagf(y));
}
__host__ __device__ inline
cuFloatComplex operator+(const int x, const cuFloatComplex & y) {
  return make_cuFloatComplex(x+cuCrealf(y), cuCimagf(y));
}
__host__ __device__ inline
cuFloatComplex operator+(const cuFloatComplex & x,const float y) {
  return make_cuFloatComplex(cuCrealf(x)+y, cuCimagf(x));
}
__host__ __device__ inline
cuFloatComplex operator+(const cuFloatComplex & x,const double y) {
  return make_cuFloatComplex(cuCrealf(x)+y, cuCimagf(x));
}
__host__ __device__ inline
cuFloatComplex operator+(const cuFloatComplex & x,const int y) {
  return make_cuFloatComplex(cuCrealf(x)+y, cuCimagf(x));
}
__host__ __device__ inline
cuFloatComplex operator-(const cuFloatComplex & x, const cuFloatComplex & y) {
  return cuCsubf(x,y);
}
__host__ __device__ inline
cuFloatComplex operator*(const cuFloatComplex & x, const cuFloatComplex & y) {
  return cuCmulf(x,y);
}
__host__ __device__ inline
cuFloatComplex operator/(const cuFloatComplex & x, const cuFloatComplex & y) {
  return cuCdivf(x,y);
}
__host__ __device__ inline
cuFloatComplex operator*(const float x, const cuFloatComplex & y) {
  return make_cuFloatComplex(x*cuCrealf(y), x*cuCimagf(y));
}
__host__ __device__ inline
cuFloatComplex operator*(const double x, const cuFloatComplex & y) {
  return make_cuFloatComplex(x*cuCrealf(y), x*cuCimagf(y));
}
__host__ __device__ inline
cuFloatComplex operator*(const int x, const cuFloatComplex & y) {
  return make_cuFloatComplex(x*cuCrealf(y), x*cuCimagf(y));
}
__host__ __device__ inline
cuFloatComplex operator*(const cuFloatComplex & x, const float & y) {
  return make_cuFloatComplex(cuCrealf(x)*y, cuCimagf(x)*y);
}
__host__ __device__ inline
cuFloatComplex operator*(const cuFloatComplex & x, const double & y) {
  return make_cuFloatComplex(cuCrealf(x)*y, cuCimagf(x)*y);
}
__host__ __device__ inline
cuFloatComplex operator*(const cuFloatComplex & x, const int & y) {
  return make_cuFloatComplex(cuCrealf(x)*y, cuCimagf(x)*y);
}
__host__ __device__ inline
cuFloatComplex operator/(const float x, const cuFloatComplex & y) {
  return cuCdivf(make_cuFloatComplex(x,0),y);
}
__host__ __device__ inline
cuFloatComplex operator/(const cuFloatComplex & x, const float y) {
  return make_cuFloatComplex(cuCrealf(x)/y, cuCimagf(x)/y);
}
__host__ __device__ inline
cuFloatComplex operator/(const cuFloatComplex & x, const double y) {
  return make_cuFloatComplex(cuCrealf(x)/y, cuCimagf(x)/y);
}
__host__ __device__ inline
cuFloatComplex operator/(const cuFloatComplex & x, const int y) {
  return make_cuFloatComplex(cuCrealf(x)/y, cuCimagf(x)/y);
}

// ===============================================
// Child classes with additional constructors
// ===============================================
struct cuFloatComplexFull : public cuFloatComplex {

public:

  // Constructors
  __host__ __device__ cuFloatComplexFull() {}
  __host__ __device__ cuFloatComplexFull(const cuFloatComplex & z) {
    x = z.x;
    y = z.y;
  }
  __host__ __device__ cuFloatComplexFull(const cuDoubleComplex & z) {
    x = z.x;
    y = z.y;
  }
  __host__ __device__ cuFloatComplexFull(const float z) {
    x = z;
    y = 0.f;
  }
  __host__ __device__ cuFloatComplexFull(const double z) {
    x = z;
    y = 0.f;
  }
  __host__ __device__ cuFloatComplexFull(const int z) {
    x = z;
    y = 0.f;
  }

};

#endif
