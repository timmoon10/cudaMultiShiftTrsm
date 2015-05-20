#include <cuComplex.h>

#ifndef __CUCOMPLEXHELPER_HPP__
#define __CUCOMPLEXHELPER_HPP__

// ===============================================
// Overloading operators
// ===============================================

// Addition
__host__ __device__ inline
cuFloatComplex operator+(const cuFloatComplex & a,
			 const cuFloatComplex & b) {
  return cuCaddf(a,b);
}
__host__ __device__ inline
cuFloatComplex operator+(const float a, const cuFloatComplex & b) {
  return make_cuFloatComplex(a+cuCrealf(b), cuCimagf(b));
}
__host__ __device__ inline
cuFloatComplex operator+(const double a, const cuFloatComplex & b) {
  return make_cuFloatComplex(float(a+cuCrealf(b)), cuCimagf(b));
}
__host__ __device__ inline
cuFloatComplex operator+(const int a, const cuFloatComplex & b) {
  return make_cuFloatComplex(a+cuCrealf(b), cuCimagf(b));
}
__host__ __device__ inline
cuFloatComplex operator+(const cuFloatComplex & a,const float b) {
  return make_cuFloatComplex(cuCrealf(a)+b, cuCimagf(a));
}
__host__ __device__ inline
cuFloatComplex operator+(const cuFloatComplex & a,const double b) {
  return make_cuFloatComplex(float(cuCrealf(a)+b), cuCimagf(a));
}
__host__ __device__ inline
cuFloatComplex operator+(const cuFloatComplex & a,const int b) {
  return make_cuFloatComplex(cuCrealf(a)+b, cuCimagf(a));
}
__host__ __device__ inline
cuDoubleComplex operator+(const cuDoubleComplex & a,
			  const cuDoubleComplex & b) {
  return cuCadd(a,b);
}
__host__ __device__ inline
cuDoubleComplex operator+(const float a, const cuDoubleComplex & b) {
  return make_cuDoubleComplex(a+cuCreal(b), cuCimag(b));
}
__host__ __device__ inline
cuDoubleComplex operator+(const double a, const cuDoubleComplex & b) {
  return make_cuDoubleComplex(a+cuCreal(b), cuCimag(b));
}
__host__ __device__ inline
cuDoubleComplex operator+(const int a, const cuDoubleComplex & b) {
  return make_cuDoubleComplex(a+cuCreal(b), cuCimag(b));
}
__host__ __device__ inline
cuDoubleComplex operator+(const cuDoubleComplex & a, const float b) {
  return make_cuDoubleComplex(cuCreal(a)+b, cuCimag(a));
}
__host__ __device__ inline
cuDoubleComplex operator+(const cuDoubleComplex & a, const double b) {
  return make_cuDoubleComplex(cuCreal(a)+b, cuCimag(a));
}
__host__ __device__ inline
cuDoubleComplex operator+(const cuDoubleComplex & a, const int b) {
  return make_cuDoubleComplex(cuCreal(a)+b, cuCimag(a));
}
__host__ __device__ inline
cuDoubleComplex operator+(const cuDoubleComplex & a,
			  const cuFloatComplex  & b) {
  return make_cuDoubleComplex(cuCreal(a)+cuCrealf(b),
			      cuCimag(a)+cuCimagf(b));
}
__host__ __device__ inline
cuDoubleComplex operator+(const cuFloatComplex & a,
			  const cuDoubleComplex & b) {
  return make_cuDoubleComplex(cuCrealf(a)+cuCreal(b),
			      cuCimagf(a)+cuCimag(b));
}

// Subtraction
__host__ __device__ inline
cuFloatComplex operator-(const cuFloatComplex & a,
			 const cuFloatComplex & b) {
  return cuCsubf(a,b);
}
__host__ __device__ inline
cuFloatComplex operator-(const float a, const cuFloatComplex & b) {
  return make_cuFloatComplex(a-cuCrealf(b), -cuCimagf(b));
}
__host__ __device__ inline
cuFloatComplex operator-(const double a, const cuFloatComplex & b) {
  return make_cuFloatComplex(float(a-cuCrealf(b)), -cuCimagf(b));
}
__host__ __device__ inline
cuFloatComplex operator-(const int a, const cuFloatComplex & b) {
  return make_cuFloatComplex(a-cuCrealf(b), -cuCimagf(b));
}
__host__ __device__ inline
cuFloatComplex operator-(const cuFloatComplex & a,const float b) {
  return make_cuFloatComplex(cuCrealf(a)-b, cuCimagf(a));
}
__host__ __device__ inline
cuFloatComplex operator-(const cuFloatComplex & a,const double b) {
  return make_cuFloatComplex(float(cuCrealf(a)-b), cuCimagf(a));
}
__host__ __device__ inline
cuFloatComplex operator-(const cuFloatComplex & a,const int b) {
  return make_cuFloatComplex(cuCrealf(a)-b, cuCimagf(a));
}
__host__ __device__ inline
cuDoubleComplex operator-(const cuDoubleComplex & a,
			  const cuDoubleComplex & b) {
  return cuCsub(a,b);
}
__host__ __device__ inline
cuDoubleComplex operator-(const float a, const cuDoubleComplex & b) {
  return make_cuDoubleComplex(a-cuCreal(b), -cuCimag(b));
}
__host__ __device__ inline
cuDoubleComplex operator-(const double a, const cuDoubleComplex & b) {
  return make_cuDoubleComplex(a-cuCreal(b), -cuCimag(b));
}
__host__ __device__ inline
cuDoubleComplex operator-(const int a, const cuDoubleComplex & b) {
  return make_cuDoubleComplex(a-cuCreal(b), -cuCimag(b));
}
__host__ __device__ inline
cuDoubleComplex operator-(const cuDoubleComplex & a, const float b) {
  return make_cuDoubleComplex(cuCreal(a)-b, cuCimag(a));
}
__host__ __device__ inline
cuDoubleComplex operator-(const cuDoubleComplex & a, const double b) {
  return make_cuDoubleComplex(cuCreal(a)-b, cuCimag(a));
}
__host__ __device__ inline
cuDoubleComplex operator-(const cuDoubleComplex & a, const int b) {
  return make_cuDoubleComplex(cuCreal(a)-b, cuCimag(a));
}
__host__ __device__ inline
cuDoubleComplex operator-(const cuDoubleComplex & a,
			  const cuFloatComplex  & b) {
  return make_cuDoubleComplex(cuCreal(a)-cuCrealf(b),
			      cuCimag(a)-cuCimagf(b));
}
__host__ __device__ inline
cuDoubleComplex operator-(const cuFloatComplex  & a,
			  const cuDoubleComplex & b) {
  return make_cuDoubleComplex(cuCrealf(a)-cuCreal(b),
			      cuCimagf(a)-cuCimag(b));
}

// Multiplication
__host__ __device__ inline
cuFloatComplex operator*(const cuFloatComplex & a,
			 const cuFloatComplex & b) {
  return cuCmulf(a,b);
}
__host__ __device__ inline
cuFloatComplex operator*(const float a, const cuFloatComplex & b) {
  return make_cuFloatComplex(a*cuCrealf(b), a*cuCimagf(b));
}
__host__ __device__ inline
cuFloatComplex operator*(const double a, const cuFloatComplex & b) {
  return make_cuFloatComplex(float(a*cuCrealf(b)), float(a*cuCimagf(b)));
}
__host__ __device__ inline
cuFloatComplex operator*(const int a, const cuFloatComplex & b) {
  return make_cuFloatComplex(a*cuCrealf(b), a*cuCimagf(b));
}
__host__ __device__ inline
cuFloatComplex operator*(const cuFloatComplex & a, const float & b) {
  return make_cuFloatComplex(cuCrealf(a)*b, cuCimagf(a)*b);
}
__host__ __device__ inline
cuFloatComplex operator*(const cuFloatComplex & a, const double & b) {
  return make_cuFloatComplex(float(cuCrealf(a)*b), float(cuCimagf(a)*b));
}
__host__ __device__ inline
cuFloatComplex operator*(const cuFloatComplex & a, const int & b) {
  return make_cuFloatComplex(cuCrealf(a)*b, cuCimagf(a)*b);
}
__host__ __device__ inline
cuDoubleComplex operator*(const cuDoubleComplex & a,
			  const cuDoubleComplex & b) {
  return cuCmul(a,b);
}
__host__ __device__ inline
cuDoubleComplex operator*(const float a, const cuDoubleComplex & b) {
  return make_cuDoubleComplex(a*cuCreal(b), a*cuCimag(b));
}
__host__ __device__ inline
cuDoubleComplex operator*(const double a, const cuDoubleComplex & b) {
  return make_cuDoubleComplex(a*cuCreal(b), a*cuCimag(b));
}
__host__ __device__ inline
cuDoubleComplex operator*(const int a, const cuDoubleComplex & b) {
  return make_cuDoubleComplex(a*cuCreal(b), a*cuCimag(b));
}
__host__ __device__ inline
cuDoubleComplex operator*(const cuDoubleComplex & a, const float b) {
  return make_cuDoubleComplex(cuCreal(a)*b, cuCimag(a)*b);
}
__host__ __device__ inline
cuDoubleComplex operator*(const cuDoubleComplex & a, const double b) {
  return make_cuDoubleComplex(cuCreal(a)*b, cuCimag(a)*b);
}
__host__ __device__ inline
cuDoubleComplex operator*(const cuDoubleComplex & a, const int b) {
  return make_cuDoubleComplex(cuCreal(a)*b, cuCimag(a)*b);
}
__host__ __device__ inline
cuDoubleComplex operator*(const cuDoubleComplex & a,
			  const cuFloatComplex  & b) {
  cuDoubleComplex temp;
  temp.x = b.x;
  temp.y = b.y;
  return a*temp;
}
__host__ __device__ inline
cuDoubleComplex operator*(const cuFloatComplex  & a,
			  const cuDoubleComplex & b) {
  cuDoubleComplex temp;
  temp.x = a.x;
  temp.y = a.y;
  return temp*b;
}

// Division
__host__ __device__ inline
cuFloatComplex operator/(const cuFloatComplex & a,
			 const cuFloatComplex & b) {
  return cuCdivf(a,b);
}
__host__ __device__ inline
cuFloatComplex operator/(const float a, const cuFloatComplex & b) {
  return cuCdivf(make_cuFloatComplex(a,0.f),b);
}
__host__ __device__ inline
cuFloatComplex operator/(const double a, const cuFloatComplex & b) {
  return cuCdivf(make_cuFloatComplex(float(a),0.f),b);
}
__host__ __device__ inline
cuFloatComplex operator/(const int a, const cuFloatComplex & b) {
  return cuCdivf(make_cuFloatComplex(float(a),0.f),b);
}
__host__ __device__ inline
cuFloatComplex operator/(const cuFloatComplex & a, const float b) {
  return make_cuFloatComplex(cuCrealf(a)/b, cuCimagf(a)/b);
}
__host__ __device__ inline
cuFloatComplex operator/(const cuFloatComplex & a, const double b) {
  return make_cuFloatComplex(float(cuCrealf(a)/b), float(cuCimagf(a)/b));
}
__host__ __device__ inline
cuFloatComplex operator/(const cuFloatComplex & a, const int b) {
  return make_cuFloatComplex(cuCrealf(a)/b, cuCimagf(a)/b);
}
__host__ __device__ inline
cuDoubleComplex operator/(const cuDoubleComplex & a,
			  const cuDoubleComplex & b) {
  return cuCdiv(a,b);
}
__host__ __device__ inline
cuDoubleComplex operator/(const float a, const cuDoubleComplex & b) {
  return cuCdiv(make_cuDoubleComplex(double(a),0.),b);
}
__host__ __device__ inline
cuDoubleComplex operator/(const double a, const cuDoubleComplex & b) {
  return cuCdiv(make_cuDoubleComplex(a,0.),b);
}
__host__ __device__ inline
cuDoubleComplex operator/(const int a, const cuDoubleComplex & b) {
  return cuCdiv(make_cuDoubleComplex(double(a),0.),b);
}
__host__ __device__ inline
cuDoubleComplex operator/(const cuDoubleComplex & a, const float b) {
  return make_cuDoubleComplex(cuCreal(a)/b, cuCimag(a)/b);
}
__host__ __device__ inline
cuDoubleComplex operator/(const cuDoubleComplex & a, const double b) {
  return make_cuDoubleComplex(cuCreal(a)/b, cuCimag(a)/b);
}
__host__ __device__ inline
cuDoubleComplex operator/(const cuDoubleComplex & a, const int b) {
  return make_cuDoubleComplex(cuCreal(a)/b, cuCimag(a)/b);
}
__host__ __device__ inline
cuDoubleComplex operator/(const cuDoubleComplex & a,
			  const cuFloatComplex  & b) {
  cuDoubleComplex temp;
  temp.x = b.x;
  temp.y = b.y;
  return a/temp;
}
__host__ __device__ inline
cuDoubleComplex operator/(const cuFloatComplex  & a,
			  const cuDoubleComplex & b) {
  cuDoubleComplex temp;
  temp.x = a.x;
  temp.y = a.y;
  return temp/b;
}

// Addition assignment
__host__ __device__ inline
cuFloatComplex& operator+=(cuFloatComplex & a, const cuFloatComplex & b) {
  a.x += b.x;
  a.y += b.y;
  return a;
}
__host__ __device__ inline
cuFloatComplex& operator+=(cuFloatComplex & a, const cuDoubleComplex & b) {
  a.x += b.x;
  a.y += b.y;
  return a;
}
__host__ __device__ inline
cuFloatComplex& operator+=(cuFloatComplex & a, const float b) {
  a.x += b;
  return a;
}
__host__ __device__ inline
cuFloatComplex& operator+=(cuFloatComplex & a, const double b) {
  a.x += b;
  return a;
}
__host__ __device__ inline
cuFloatComplex& operator+=(cuFloatComplex & a, const int b) {
  a.x += b;
  return a;
}
__host__ __device__ inline
cuDoubleComplex& operator+=(cuDoubleComplex & a, const cuFloatComplex & b) {
  a.x += b.x;
  a.y += b.y;
  return a;
}
__host__ __device__ inline
cuDoubleComplex& operator+=(cuDoubleComplex & a, const cuDoubleComplex & b) {
  a.x += b.x;
  a.y += b.y;
  return a;
}
__host__ __device__ inline
cuDoubleComplex& operator+=(cuDoubleComplex & a, const float b) {
  a.x += b;
  return a;
}
__host__ __device__ inline
cuDoubleComplex& operator+=(cuDoubleComplex & a, const double b) {
  a.x += b;
  return a;
}
__host__ __device__ inline
cuDoubleComplex& operator+=(cuDoubleComplex & a, const int b) {
  a.x += b;
  return a;
}

// Subtraction assignment
__host__ __device__ inline
cuFloatComplex& operator-=(cuFloatComplex & a, const cuFloatComplex & b) {
  a.x -= b.x;
  a.y -= b.y;
  return a;
}
__host__ __device__ inline
cuFloatComplex& operator-=(cuFloatComplex & a, const cuDoubleComplex & b) {
  a.x -= b.x;
  a.y -= b.y;
  return a;
}
__host__ __device__ inline
cuFloatComplex& operator-=(cuFloatComplex & a, const float b) {
  a.x -= b;
  return a;
}
__host__ __device__ inline
cuFloatComplex& operator-=(cuFloatComplex & a, const double b) {
  a.x -= b;
  return a;
}
__host__ __device__ inline
cuFloatComplex& operator-=(cuFloatComplex & a, const int b) {
  a.x -= b;
  return a;
}
__host__ __device__ inline
cuDoubleComplex& operator-=(cuDoubleComplex & a, const cuFloatComplex & b) {
  a.x -= b.x;
  a.y -= b.y;
  return a;
}
__host__ __device__ inline
cuDoubleComplex& operator-=(cuDoubleComplex & a, const cuDoubleComplex & b) {
  a.x -= b.x;
  a.y -= b.y;
  return a;
}
__host__ __device__ inline
cuDoubleComplex& operator-=(cuDoubleComplex & a, const float b) {
  a.x -= b;
  return a;
}
__host__ __device__ inline
cuDoubleComplex& operator-=(cuDoubleComplex & a, const double b) {
  a.x -= b;
  return a;
}
__host__ __device__ inline
cuDoubleComplex& operator-=(cuDoubleComplex & a, const int b) {
  a.x -= b;
  return a;
}

// Multiplication assignment
__host__ __device__ inline
cuFloatComplex& operator*=(cuFloatComplex & a, const cuFloatComplex & b) {
  a = a*b;
  return a;
}
__host__ __device__ inline
cuFloatComplex& operator*=(cuFloatComplex & a, const cuDoubleComplex & b) {
  cuDoubleComplex temp = a*b;
  a.x = temp.x;
  a.y = temp.y;
  return a;
}
__host__ __device__ inline
cuFloatComplex& operator*=(cuFloatComplex & a, const float b) {
  a.x *= b;
  a.y *= b;
  return a;
}
__host__ __device__ inline
cuFloatComplex& operator*=(cuFloatComplex & a, const double b) {
  a.x *= b;
  a.y *= b;
  return a;
}
__host__ __device__ inline
cuFloatComplex& operator*=(cuFloatComplex & a, const int b) {
  a.x *= b;
  a.y *= b;
  return a;
}
__host__ __device__ inline
cuDoubleComplex& operator*=(cuDoubleComplex & a, const cuFloatComplex & b) {
  a = a*b;
  return a;
}
__host__ __device__ inline
cuDoubleComplex& operator*=(cuDoubleComplex & a, const cuDoubleComplex & b) {
  a = a*b;
  return a;
}
__host__ __device__ inline
cuDoubleComplex& operator*=(cuDoubleComplex & a, const float b) {
  a.x *= b;
  a.y *= b;
  return a;
}
__host__ __device__ inline
cuDoubleComplex& operator*=(cuDoubleComplex & a, const double b) {
  a.x *= b;
  a.y *= b;
  return a;
}
__host__ __device__ inline
cuDoubleComplex& operator*=(cuDoubleComplex & a, const int b) {
  a.x *= b;
  a.y *= b;
  return a;
}

// Division assignment
__host__ __device__ inline
cuFloatComplex& operator/=(cuFloatComplex & a, const cuFloatComplex & b) {
  a = a/b;
  return a;
}
__host__ __device__ inline
cuFloatComplex& operator/=(cuFloatComplex & a, const cuDoubleComplex & b) {
  cuDoubleComplex temp = a/b;
  a.x = temp.x;
  a.y = temp.y;
  return a;
}
__host__ __device__ inline
cuFloatComplex& operator/=(cuFloatComplex & a, const float b) {
  a.x /= b;
  a.y /= b;
  return a;
}
__host__ __device__ inline
cuFloatComplex& operator/=(cuFloatComplex & a, const double b) {
  a.x /= b;
  a.y /= b;
  return a;
}
__host__ __device__ inline
cuFloatComplex& operator/=(cuFloatComplex & a, const int b) {
  a.x /= b;
  a.y /= b;
  return a;
}
__host__ __device__ inline
cuDoubleComplex& operator/=(cuDoubleComplex & a, const cuFloatComplex & b) {
  a = a/b;
  return a;
}
__host__ __device__ inline
cuDoubleComplex& operator/=(cuDoubleComplex & a, const cuDoubleComplex & b) {
  a = a/b;
  return a;
}
__host__ __device__ inline
cuDoubleComplex& operator/=(cuDoubleComplex & a, const float b) {
  a.x /= b;
  a.y /= b;
  return a;
}
__host__ __device__ inline
cuDoubleComplex& operator/=(cuDoubleComplex & a, const double b) {
  a.x /= b;
  a.y /= b;
  return a;
}
__host__ __device__ inline
cuDoubleComplex& operator/=(cuDoubleComplex & a, const int b) {
  a.x /= b;
  a.y /= b;
  return a;
}

// ===============================================
// Derived classes with additional constructors
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

struct cuDoubleComplexFull : public cuDoubleComplex {

public:

  // Constructors
  __host__ __device__ cuDoubleComplexFull() {}
  __host__ __device__ cuDoubleComplexFull(const cuFloatComplex & z) {
    x = z.x;
    y = z.y;
  }
  __host__ __device__ cuDoubleComplexFull(const cuDoubleComplex & z) {
    x = z.x;
    y = z.y;
  }
  __host__ __device__ cuDoubleComplexFull(const float z) {
    x = z;
    y = 0.;
  }
  __host__ __device__ cuDoubleComplexFull(const double z) {
    x = z;
    y = 0.;
  }
  __host__ __device__ cuDoubleComplexFull(const int z) {
    x = z;
    y = 0.;
  }

};

#endif
