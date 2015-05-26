#include <cuComplex.h>

#ifndef __CUCOMPLEXHELPER_HPP__
#define __CUCOMPLEXHELPER_HPP__

// ===============================================
// Derived structs with additional functionality
// ===============================================

// Single-precision complex data type
struct cuFloatComplexFull : public cuFloatComplex {

public:

  // Constructors
  __host__ __device__ cuFloatComplexFull() {}
  __host__ __device__ cuFloatComplexFull(const cuFloatComplex & z) : cuFloatComplex(z) {}
  __host__ __device__ cuFloatComplexFull(volatile const cuFloatComplex & z) {
    x = z.x;
    y = z.y;
  }
  __host__ __device__ cuFloatComplexFull(const cuDoubleComplex z) {
    x = float(z.x);
    y = float(z.y);
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

  // Assignment operator
  __host__ __device__ inline
  cuFloatComplexFull & operator=(const cuFloatComplex z) {
    x = z.x;
    y = z.y;
    return *this;
  }
  __host__ __device__ inline
  cuFloatComplexFull & operator=(const cuDoubleComplex z) {
    x = z.x;
    y = z.y;
    return *this;
  }
  __host__ __device__ inline
  cuFloatComplexFull & operator=(const int z) {
    x = z;
    y = 0;
    return *this;
  }
  __host__ __device__ inline
  cuFloatComplexFull & operator=(const float z) {
    x = z;
    y = 0;
    return *this;
  }
  __host__ __device__ inline
  cuFloatComplexFull & operator=(const double z) {
    x = z;
    y = 0;
    return *this;
  }
  __host__ __device__ inline
  volatile cuFloatComplexFull & operator=(const cuFloatComplex z) volatile {
    x = z.x;
    y = z.y;
    return *this;
  }
  __host__ __device__ inline
  volatile cuFloatComplexFull & operator=(const cuDoubleComplex z) volatile {
    x = z.x;
    y = z.y;
    return *this;
  }
  __host__ __device__ inline
  volatile cuFloatComplexFull & operator=(const int z) volatile {
    x = z;
    y = 0;
    return *this;
  }
  __host__ __device__ inline
  volatile cuFloatComplexFull & operator=(const float z) volatile {
    x = z;
    y = 0;
    return *this;
  }
  __host__ __device__ inline
  volatile cuFloatComplexFull & operator=(const double z) volatile {
    x = z;
    y = 0;
    return *this;
  }

};

// Double-precision complex data type
struct cuDoubleComplexFull : public cuDoubleComplex {

public:

  // Constructors
  __host__ __device__ cuDoubleComplexFull() {}
  __host__ __device__ cuDoubleComplexFull(const cuFloatComplex z) {
    x = z.x;
    y = z.y;
  }
  __host__ __device__ cuDoubleComplexFull(const cuDoubleComplex & z) : cuDoubleComplex(z) {}
  __host__ __device__ cuDoubleComplexFull(volatile const cuDoubleComplex & z) {
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

  // Assignment operator
  __host__ __device__ inline
  cuDoubleComplexFull & operator=(const cuFloatComplex z) {
    x = z.x;
    y = z.y;
    return *this;
  }
  __host__ __device__ inline
  cuDoubleComplexFull & operator=(const cuDoubleComplex z) {
    x = z.x;
    y = z.y;
    return *this;
  }
  __host__ __device__ inline
  cuDoubleComplexFull & operator=(const int z) {
    x = z;
    y = 0;
    return *this;
  }
  __host__ __device__ inline
  cuDoubleComplexFull & operator=(const float z) {
    x = z;
    y = 0;
    return *this;
  }
  __host__ __device__ inline
  cuDoubleComplexFull & operator=(const double z) {
    x = z;
    y = 0;
    return *this;
  }
  __host__ __device__ inline
  volatile cuDoubleComplexFull & operator=(const cuFloatComplex z) volatile {
    x = z.x;
    y = z.y;
    return *this;
  }
  __host__ __device__ inline
  volatile cuDoubleComplexFull & operator=(const cuDoubleComplex z) volatile {
    x = z.x;
    y = z.y;
    return *this;
  }
  __host__ __device__ inline
  volatile cuDoubleComplexFull & operator=(const int z) volatile {
    x = z;
    y = 0;
    return *this;
  }
  __host__ __device__ inline
  volatile cuDoubleComplexFull & operator=(const float z) volatile {
    x = z;
    y = 0;
    return *this;
  }
  __host__ __device__ inline
  volatile cuDoubleComplexFull & operator=(const double z) volatile {
    x = z;
    y = 0;
    return *this;
  }

};

// ===============================================
// Overloaded operators
// ===============================================

// Conjugation
__host__ __device__ inline
cuFloatComplexFull conjugate(const cuFloatComplexFull a) {
  return cuConjf(a);
}
__host__ __device__ inline
cuDoubleComplexFull conjugate(const cuDoubleComplexFull a) {
  return cuConj(a);
}
__host__ __device__ inline
float conjugate(const float a) {
  return a;
}
__host__ __device__ inline
double conjugate(const double a) {
  return a;
}

// Negation
__host__ __device__ inline
cuFloatComplexFull operator-(const cuFloatComplexFull a) {
  return make_cuFloatComplex(-cuCrealf(a),-cuCimagf(a));
}
__host__ __device__ inline
cuDoubleComplexFull operator-(const cuDoubleComplexFull a) {
  return make_cuDoubleComplex(-cuCreal(a),-cuCimag(a));
}

// Addition
__host__ __device__ inline
cuFloatComplexFull operator+(const cuFloatComplexFull a,
			     const cuFloatComplexFull b) {
  return cuCaddf(a,b);
}
__host__ __device__ inline
cuFloatComplexFull operator+(const float a, const cuFloatComplexFull b) {
  return make_cuFloatComplex(a+cuCrealf(b), cuCimagf(b));
}
__host__ __device__ inline
cuFloatComplexFull operator+(const double a, const cuFloatComplexFull b) {
  return make_cuFloatComplex(float(a+cuCrealf(b)), cuCimagf(b));
}
__host__ __device__ inline
cuFloatComplexFull operator+(const int a, const cuFloatComplexFull b) {
  return make_cuFloatComplex(a+cuCrealf(b), cuCimagf(b));
}
__host__ __device__ inline
cuFloatComplexFull operator+(const cuFloatComplexFull a,const float b) {
  return b+a;
}
__host__ __device__ inline
cuFloatComplexFull operator+(const cuFloatComplexFull a,const double b) {
  return b+a;
}
__host__ __device__ inline
cuFloatComplexFull operator+(const cuFloatComplexFull a,const int b) {
  return b+a;
}
__host__ __device__ inline
cuDoubleComplexFull operator+(const cuDoubleComplexFull a,
			      const cuDoubleComplexFull b) {
  return cuCadd(a,b);
}
__host__ __device__ inline
cuDoubleComplexFull operator+(const float a, const cuDoubleComplexFull b) {
  return make_cuDoubleComplex(a+cuCreal(b), cuCimag(b));
}
__host__ __device__ inline
cuDoubleComplexFull operator+(const double a, const cuDoubleComplexFull b) {
  return make_cuDoubleComplex(a+cuCreal(b), cuCimag(b));
}
__host__ __device__ inline
cuDoubleComplexFull operator+(const int a, const cuDoubleComplexFull b) {
  return make_cuDoubleComplex(a+cuCreal(b), cuCimag(b));
}
__host__ __device__ inline
cuDoubleComplexFull operator+(const cuDoubleComplexFull a, const float b) {
  return b+a;
}
__host__ __device__ inline
cuDoubleComplexFull operator+(const cuDoubleComplexFull a, const double b) {
  return b+a;
}
__host__ __device__ inline
cuDoubleComplexFull operator+(const cuDoubleComplexFull a, const int b) {
  return b+a;
}
__host__ __device__ inline
cuDoubleComplexFull operator+(const cuDoubleComplexFull a,
			      const cuFloatComplexFull  b) {
  return make_cuDoubleComplex(cuCreal(a)+cuCrealf(b),
			      cuCimag(a)+cuCimagf(b));
}
__host__ __device__ inline
cuDoubleComplexFull operator+(const cuFloatComplexFull a,
			      const cuDoubleComplexFull b) {
  return make_cuDoubleComplex(cuCrealf(a)+cuCreal(b),
			      cuCimagf(a)+cuCimag(b));
}

// Subtraction
__host__ __device__ inline
cuFloatComplexFull operator-(const cuFloatComplexFull a,
			     const cuFloatComplexFull b) {
  return cuCsubf(a,b);
}
__host__ __device__ inline
cuFloatComplexFull operator-(const float a, const cuFloatComplexFull b) {
  return make_cuFloatComplex(a-cuCrealf(b), -cuCimagf(b));
}
__host__ __device__ inline
cuFloatComplexFull operator-(const double a, const cuFloatComplexFull b) {
  return make_cuFloatComplex(float(a-cuCrealf(b)), -cuCimagf(b));
}
__host__ __device__ inline
cuFloatComplexFull operator-(const int a, const cuFloatComplexFull b) {
  return make_cuFloatComplex(a-cuCrealf(b), -cuCimagf(b));
}
__host__ __device__ inline
cuFloatComplexFull operator-(const cuFloatComplexFull a,const float b) {
  return make_cuFloatComplex(cuCrealf(a)-b, cuCimagf(a));
}
__host__ __device__ inline
cuFloatComplexFull operator-(const cuFloatComplexFull a,const double b) {
  return make_cuFloatComplex(float(cuCrealf(a)-b), cuCimagf(a));
}
__host__ __device__ inline
cuFloatComplexFull operator-(const cuFloatComplexFull a,const int b) {
  return make_cuFloatComplex(cuCrealf(a)-b, cuCimagf(a));
}
__host__ __device__ inline
cuDoubleComplexFull operator-(const cuDoubleComplexFull a,
			      const cuDoubleComplexFull b) {
  return cuCsub(a,b);
}
__host__ __device__ inline
cuDoubleComplexFull operator-(const float a, const cuDoubleComplexFull b) {
  return make_cuDoubleComplex(a-cuCreal(b), -cuCimag(b));
}
__host__ __device__ inline
cuDoubleComplexFull operator-(const double a, const cuDoubleComplexFull b) {
  return make_cuDoubleComplex(a-cuCreal(b), -cuCimag(b));
}
__host__ __device__ inline
cuDoubleComplexFull operator-(const int a, const cuDoubleComplexFull b) {
  return make_cuDoubleComplex(a-cuCreal(b), -cuCimag(b));
}
__host__ __device__ inline
cuDoubleComplexFull operator-(const cuDoubleComplexFull a, const float b) {
  return make_cuDoubleComplex(cuCreal(a)-b, cuCimag(a));
}
__host__ __device__ inline
cuDoubleComplexFull operator-(const cuDoubleComplexFull a, const double b) {
  return make_cuDoubleComplex(cuCreal(a)-b, cuCimag(a));
}
__host__ __device__ inline
cuDoubleComplexFull operator-(const cuDoubleComplexFull a, const int b) {
  return make_cuDoubleComplex(cuCreal(a)-b, cuCimag(a));
}
__host__ __device__ inline
cuDoubleComplexFull operator-(const cuDoubleComplexFull a,
			      const cuFloatComplexFull  b) {
  return make_cuDoubleComplex(cuCreal(a)-cuCrealf(b),
			      cuCimag(a)-cuCimagf(b));
}
__host__ __device__ inline
cuDoubleComplexFull operator-(const cuFloatComplexFull  a,
			      const cuDoubleComplexFull b) {
  return make_cuDoubleComplex(cuCrealf(a)-cuCreal(b),
			      cuCimagf(a)-cuCimag(b));
}

// Multiplication
__host__ __device__ inline
cuFloatComplexFull operator*(const cuFloatComplexFull a,
			     const cuFloatComplexFull b) {
  return cuCmulf(a,b);
}
__host__ __device__ inline
cuFloatComplexFull operator*(const float a, const cuFloatComplexFull b) {
  return make_cuFloatComplex(a*cuCrealf(b), a*cuCimagf(b));
}
__host__ __device__ inline
cuFloatComplexFull operator*(const double a, const cuFloatComplexFull b) {
  return make_cuFloatComplex(float(a*cuCrealf(b)), float(a*cuCimagf(b)));
}
__host__ __device__ inline
cuFloatComplexFull operator*(const int a, const cuFloatComplexFull b) {
  return make_cuFloatComplex(a*cuCrealf(b), a*cuCimagf(b));
}
__host__ __device__ inline
cuFloatComplexFull operator*(const cuFloatComplexFull a, const float b) {
  return b*a;
}
__host__ __device__ inline
cuFloatComplexFull operator*(const cuFloatComplexFull a, const double b) {
  return b*a;
}
__host__ __device__ inline
cuFloatComplexFull operator*(const cuFloatComplexFull a, const int b) {
  return b*a;
}
__host__ __device__ inline
cuDoubleComplexFull operator*(const cuDoubleComplexFull a,
			      const cuDoubleComplexFull b) {
  return cuCmul(a,b);
}
__host__ __device__ inline
cuDoubleComplexFull operator*(const float a, const cuDoubleComplexFull b) {
  return make_cuDoubleComplex(a*cuCreal(b), a*cuCimag(b));
}
__host__ __device__ inline
cuDoubleComplexFull operator*(const double a, const cuDoubleComplexFull b) {
  return make_cuDoubleComplex(a*cuCreal(b), a*cuCimag(b));
}
__host__ __device__ inline
cuDoubleComplexFull operator*(const int a, const cuDoubleComplexFull b) {
  return make_cuDoubleComplex(a*cuCreal(b), a*cuCimag(b));
}
__host__ __device__ inline
cuDoubleComplexFull operator*(const cuDoubleComplexFull a, const float b) {
  return b*a;
}
__host__ __device__ inline
cuDoubleComplexFull operator*(const cuDoubleComplexFull a, const double b) {
  return b*a;
}
__host__ __device__ inline
cuDoubleComplexFull operator*(const cuDoubleComplexFull a, const int b) {
  return b*a;
}
__host__ __device__ inline
cuDoubleComplexFull operator*(const cuDoubleComplexFull a,
			      const cuFloatComplexFull  b) {
  cuDoubleComplexFull temp;
  temp.x = b.x;
  temp.y = b.y;
  return a*temp;
}
__host__ __device__ inline
cuDoubleComplexFull operator*(const cuFloatComplexFull  a,
			      const cuDoubleComplexFull b) {
  cuDoubleComplexFull temp;
  temp.x = a.x;
  temp.y = a.y;
  return temp*b;
}

// Division
__host__ __device__ inline
cuFloatComplexFull operator/(const cuFloatComplexFull a,
			     const cuFloatComplexFull b) {
  return cuCdivf(a,b);
}
__host__ __device__ inline
cuFloatComplexFull operator/(const float a, const cuFloatComplexFull b) {
  return cuCdivf(make_cuFloatComplex(a,0.f),b);
}
__host__ __device__ inline
cuFloatComplexFull operator/(const double a, const cuFloatComplexFull b) {
  return cuCdivf(make_cuFloatComplex(float(a),0.f),b);
}
__host__ __device__ inline
cuFloatComplexFull operator/(const int a, const cuFloatComplexFull b) {
  return cuCdivf(make_cuFloatComplex(float(a),0.f),b);
}
__host__ __device__ inline
cuFloatComplexFull operator/(const cuFloatComplexFull a, const float b) {
  return make_cuFloatComplex(cuCrealf(a)/b, cuCimagf(a)/b);
}
__host__ __device__ inline
cuFloatComplexFull operator/(const cuFloatComplexFull a, const double b) {
  return make_cuFloatComplex(float(cuCrealf(a)/b), float(cuCimagf(a)/b));
}
__host__ __device__ inline
cuFloatComplexFull operator/(const cuFloatComplexFull a, const int b) {
  return make_cuFloatComplex(cuCrealf(a)/b, cuCimagf(a)/b);
}
__host__ __device__ inline
cuDoubleComplexFull operator/(const cuDoubleComplexFull a,
			      const cuDoubleComplexFull b) {
  return cuCdiv(a,b);
}
__host__ __device__ inline
cuDoubleComplexFull operator/(const float a, const cuDoubleComplexFull b) {
  return cuCdiv(make_cuDoubleComplex(double(a),0.),b);
}
__host__ __device__ inline
cuDoubleComplexFull operator/(const double a, const cuDoubleComplexFull b) {
  return cuCdiv(make_cuDoubleComplex(a,0.),b);
}
__host__ __device__ inline
cuDoubleComplexFull operator/(const int a, const cuDoubleComplexFull b) {
  return cuCdiv(make_cuDoubleComplex(double(a),0.),b);
}
__host__ __device__ inline
cuDoubleComplexFull operator/(const cuDoubleComplexFull a, const float b) {
  return make_cuDoubleComplex(cuCreal(a)/b, cuCimag(a)/b);
}
__host__ __device__ inline
cuDoubleComplexFull operator/(const cuDoubleComplexFull a, const double b) {
  return make_cuDoubleComplex(cuCreal(a)/b, cuCimag(a)/b);
}
__host__ __device__ inline
cuDoubleComplexFull operator/(const cuDoubleComplexFull a, const int b) {
  return make_cuDoubleComplex(cuCreal(a)/b, cuCimag(a)/b);
}
__host__ __device__ inline
cuDoubleComplexFull operator/(const cuDoubleComplexFull a,
			      const cuFloatComplexFull  b) {
  cuDoubleComplexFull temp;
  temp.x = b.x;
  temp.y = b.y;
  return a/temp;
}
__host__ __device__ inline
cuDoubleComplexFull operator/(const cuFloatComplexFull  a,
			      const cuDoubleComplexFull b) {
  cuDoubleComplexFull temp;
  temp.x = a.x;
  temp.y = a.y;
  return temp/b;
}

// Addition assignment
__host__ __device__ inline
cuFloatComplexFull& operator+=(cuFloatComplexFull & a, const cuFloatComplexFull b) {
  a.x += b.x;
  a.y += b.y;
  return a;
}
__host__ __device__ inline
cuFloatComplexFull& operator+=(cuFloatComplexFull & a, const cuDoubleComplexFull b) {
  a.x += b.x;
  a.y += b.y;
  return a;
}
__host__ __device__ inline
cuFloatComplexFull& operator+=(cuFloatComplexFull & a, const float b) {
  a.x += b;
  return a;
}
__host__ __device__ inline
cuFloatComplexFull& operator+=(cuFloatComplexFull & a, const double b) {
  a.x += b;
  return a;
}
__host__ __device__ inline
cuFloatComplexFull& operator+=(cuFloatComplexFull & a, const int b) {
  a.x += b;
  return a;
}
__host__ __device__ inline
cuDoubleComplexFull& operator+=(cuDoubleComplexFull & a, const cuFloatComplexFull b) {
  a.x += b.x;
  a.y += b.y;
  return a;
}
__host__ __device__ inline
cuDoubleComplexFull& operator+=(cuDoubleComplexFull & a, const cuDoubleComplexFull b) {
  a.x += b.x;
  a.y += b.y;
  return a;
}
__host__ __device__ inline
cuDoubleComplexFull& operator+=(cuDoubleComplexFull & a, const float b) {
  a.x += b;
  return a;
}
__host__ __device__ inline
cuDoubleComplexFull& operator+=(cuDoubleComplexFull & a, const double b) {
  a.x += b;
  return a;
}
__host__ __device__ inline
cuDoubleComplexFull& operator+=(cuDoubleComplexFull & a, const int b) {
  a.x += b;
  return a;
}
__host__ __device__ inline
volatile cuFloatComplexFull& operator+=(volatile cuFloatComplexFull & a,
					const cuFloatComplexFull b) {
  a.x += b.x;
  a.y += b.y;
  return a;
}
__host__ __device__ inline
volatile cuFloatComplexFull& operator+=(volatile cuFloatComplexFull & a,
					const cuDoubleComplexFull b) {
  a.x += b.x;
  a.y += b.y;
  return a;
}
__host__ __device__ inline
volatile cuFloatComplexFull& operator+=(volatile cuFloatComplexFull & a,
					const float b) {
  a.x += b;
  return a;
}
__host__ __device__ inline
volatile cuFloatComplexFull& operator+=(volatile cuFloatComplexFull & a,
					const double b) {
  a.x += b;
  return a;
}
__host__ __device__ inline
volatile cuFloatComplexFull& operator+=(volatile cuFloatComplexFull & a,
					const int b) {
  a.x += b;
  return a;
}
__host__ __device__ inline
volatile cuDoubleComplexFull& operator+=(volatile cuDoubleComplexFull & a,
					 const cuFloatComplexFull b) {
  a.x += b.x;
  a.y += b.y;
  return a;
}
__host__ __device__ inline
volatile cuDoubleComplexFull& operator+=(volatile cuDoubleComplexFull & a,
					 const cuDoubleComplexFull b) {
  a.x += b.x;
  a.y += b.y;
  return a;
}
__host__ __device__ inline
volatile cuDoubleComplexFull& operator+=(volatile cuDoubleComplexFull & a,
					 const float b) {
  a.x += b;
  return a;
}
__host__ __device__ inline
volatile cuDoubleComplexFull& operator+=(volatile cuDoubleComplexFull & a,
					 const double b) {
  a.x += b;
  return a;
}
__host__ __device__ inline
volatile cuDoubleComplexFull& operator+=(volatile cuDoubleComplexFull & a,
					 const int b) {
  a.x += b;
  return a;
}

// Subtraction assignment
__host__ __device__ inline
cuFloatComplexFull& operator-=(cuFloatComplexFull & a, const cuFloatComplexFull b) {
  a.x -= b.x;
  a.y -= b.y;
  return a;
}
__host__ __device__ inline
cuFloatComplexFull& operator-=(cuFloatComplexFull & a, const cuDoubleComplexFull b) {
  a.x -= b.x;
  a.y -= b.y;
  return a;
}
__host__ __device__ inline
cuFloatComplexFull& operator-=(cuFloatComplexFull & a, const float b) {
  a.x -= b;
  return a;
}
__host__ __device__ inline
cuFloatComplexFull& operator-=(cuFloatComplexFull & a, const double b) {
  a.x -= b;
  return a;
}
__host__ __device__ inline
cuFloatComplexFull& operator-=(cuFloatComplexFull & a, const int b) {
  a.x -= b;
  return a;
}
__host__ __device__ inline
cuDoubleComplexFull& operator-=(cuDoubleComplexFull & a, const cuFloatComplexFull b) {
  a.x -= b.x;
  a.y -= b.y;
  return a;
}
__host__ __device__ inline
cuDoubleComplexFull& operator-=(cuDoubleComplexFull & a, const cuDoubleComplexFull b) {
  a.x -= b.x;
  a.y -= b.y;
  return a;
}
__host__ __device__ inline
cuDoubleComplexFull& operator-=(cuDoubleComplexFull & a, const float b) {
  a.x -= b;
  return a;
}
__host__ __device__ inline
cuDoubleComplexFull& operator-=(cuDoubleComplexFull & a, const double b) {
  a.x -= b;
  return a;
}
__host__ __device__ inline
cuDoubleComplexFull& operator-=(cuDoubleComplexFull & a, const int b) {
  a.x -= b;
  return a;
}
__host__ __device__ inline
volatile cuFloatComplexFull& operator-=(volatile cuFloatComplexFull & a,
					const cuFloatComplexFull b) {
  a.x -= b.x;
  a.y -= b.y;
  return a;
}
__host__ __device__ inline
volatile cuFloatComplexFull& operator-=(volatile cuFloatComplexFull & a,
					const cuDoubleComplexFull b) {
  a.x -= b.x;
  a.y -= b.y;
  return a;
}
__host__ __device__ inline
volatile cuFloatComplexFull& operator-=(volatile cuFloatComplexFull & a,
					const float b) {
  a.x -= b;
  return a;
}
__host__ __device__ inline
volatile cuFloatComplexFull& operator-=(volatile cuFloatComplexFull & a,
					const double b) {
  a.x -= b;
  return a;
}
__host__ __device__ inline
volatile cuFloatComplexFull& operator-=(volatile cuFloatComplexFull & a,
					const int b) {
  a.x -= b;
  return a;
}
__host__ __device__ inline
volatile cuDoubleComplexFull& operator-=(volatile cuDoubleComplexFull & a,
					 const cuFloatComplexFull b) {
  a.x -= b.x;
  a.y -= b.y;
  return a;
}
__host__ __device__ inline
volatile cuDoubleComplexFull& operator-=(volatile cuDoubleComplexFull & a,
					 const cuDoubleComplexFull b) {
  a.x -= b.x;
  a.y -= b.y;
  return a;
}
__host__ __device__ inline
volatile cuDoubleComplexFull& operator-=(volatile cuDoubleComplexFull & a,
					 const float b) {
  a.x -= b;
  return a;
}
__host__ __device__ inline
volatile cuDoubleComplexFull& operator-=(volatile cuDoubleComplexFull & a,
					 const double b) {
  a.x -= b;
  return a;
}
__host__ __device__ inline
volatile cuDoubleComplexFull& operator-=(volatile cuDoubleComplexFull & a,
					 const int b) {
  a.x -= b;
  return a;
}

// Multiplication assignment
__host__ __device__ inline
cuFloatComplexFull& operator*=(cuFloatComplexFull & a, const cuFloatComplexFull b) {
  a = a*b;
  return a;
}
__host__ __device__ inline
cuFloatComplexFull& operator*=(cuFloatComplexFull & a, const cuDoubleComplexFull b) {
  cuDoubleComplexFull temp = a*b;
  a.x = temp.x;
  a.y = temp.y;
  return a;
}
__host__ __device__ inline
cuFloatComplexFull& operator*=(cuFloatComplexFull & a, const float b) {
  a.x *= b;
  a.y *= b;
  return a;
}
__host__ __device__ inline
cuFloatComplexFull& operator*=(cuFloatComplexFull & a, const double b) {
  a.x *= b;
  a.y *= b;
  return a;
}
__host__ __device__ inline
cuFloatComplexFull& operator*=(cuFloatComplexFull & a, const int b) {
  a.x *= b;
  a.y *= b;
  return a;
}
__host__ __device__ inline
cuDoubleComplexFull& operator*=(cuDoubleComplexFull & a, const cuFloatComplexFull b) {
  a = a*b;
  return a;
}
__host__ __device__ inline
cuDoubleComplexFull& operator*=(cuDoubleComplexFull & a, const cuDoubleComplexFull b) {
  a = a*b;
  return a;
}
__host__ __device__ inline
cuDoubleComplexFull& operator*=(cuDoubleComplexFull & a, const float b) {
  a.x *= b;
  a.y *= b;
  return a;
}
__host__ __device__ inline
cuDoubleComplexFull& operator*=(cuDoubleComplexFull & a, const double b) {
  a.x *= b;
  a.y *= b;
  return a;
}
__host__ __device__ inline
cuDoubleComplexFull& operator*=(cuDoubleComplexFull & a, const int b) {
  a.x *= b;
  a.y *= b;
  return a;
}
__host__ __device__ inline
volatile cuFloatComplexFull& operator*=(volatile cuFloatComplexFull & a,
					const cuFloatComplexFull b) {
  float temp = a.x;
  a.x = temp*b.x - a.y*b.y;
  a.y = temp*b.y + a.y*b.x;
  return a;
}
__host__ __device__ inline
volatile cuFloatComplexFull& operator*=(volatile cuFloatComplexFull & a,
					const cuDoubleComplexFull b) {
  float temp = a.x;
  a.x = float(temp*b.x - a.y*b.y);
  a.y = float(temp*b.y + a.y*b.x);
  return a;
}
__host__ __device__ inline
volatile cuFloatComplexFull& operator*=(volatile cuFloatComplexFull & a,
					const float b) {
  a.x *= b;
  a.y *= b;
  return a;
}
__host__ __device__ inline
volatile cuFloatComplexFull& operator*=(volatile cuFloatComplexFull & a,
					const double b) {
  a.x *= b;
  a.y *= b;
  return a;
}
__host__ __device__ inline
volatile cuFloatComplexFull& operator*=(volatile cuFloatComplexFull & a,
					const int b) {
  a.x *= b;
  a.y *= b;
  return a;
}
__host__ __device__ inline
volatile cuDoubleComplexFull& operator*=(volatile cuDoubleComplexFull & a,
					 const cuFloatComplexFull b) {
  double temp = a.x;
  a.x = temp*b.x - a.y*b.y;
  a.y = temp*b.y + a.y*b.x;
  return a;
}
__host__ __device__ inline
volatile cuDoubleComplexFull& operator*=(volatile cuDoubleComplexFull & a,
					 const cuDoubleComplexFull b) {
  double temp = a.x;
  a.x = temp*b.x - a.y*b.y;
  a.y = temp*b.y + a.y*b.x;
  return a;
}
__host__ __device__ inline
volatile cuDoubleComplexFull& operator*=(volatile cuDoubleComplexFull & a,
					 const float b) {
  a.x *= b;
  a.y *= b;
  return a;
}
__host__ __device__ inline
volatile cuDoubleComplexFull& operator*=(volatile cuDoubleComplexFull & a,
					 const double b) {
  a.x *= b;
  a.y *= b;
  return a;
}
__host__ __device__ inline
volatile cuDoubleComplexFull& operator*=(volatile cuDoubleComplexFull & a,
					 const int b) {
  a.x *= b;
  a.y *= b;
  return a;
}

// Division assignment
__host__ __device__ inline
cuFloatComplexFull& operator/=(cuFloatComplexFull & a, const cuFloatComplexFull b) {
  a = a/b;
  return a;
}
__host__ __device__ inline
cuFloatComplexFull& operator/=(cuFloatComplexFull & a, const cuDoubleComplexFull b) {
  cuDoubleComplexFull temp = a/b;
  a.x = temp.x;
  a.y = temp.y;
  return a;
}
__host__ __device__ inline
cuFloatComplexFull& operator/=(cuFloatComplexFull & a, const float b) {
  a.x /= b;
  a.y /= b;
  return a;
}
__host__ __device__ inline
cuFloatComplexFull& operator/=(cuFloatComplexFull & a, const double b) {
  a.x /= b;
  a.y /= b;
  return a;
}
__host__ __device__ inline
cuFloatComplexFull& operator/=(cuFloatComplexFull & a, const int b) {
  a.x /= b;
  a.y /= b;
  return a;
}
__host__ __device__ inline
cuDoubleComplexFull& operator/=(cuDoubleComplexFull & a, const cuFloatComplexFull b) {
  a = a/b;
  return a;
}
__host__ __device__ inline
cuDoubleComplexFull& operator/=(cuDoubleComplexFull & a, const cuDoubleComplexFull b) {
  a = a/b;
  return a;
}
__host__ __device__ inline
cuDoubleComplexFull& operator/=(cuDoubleComplexFull & a, const float b) {
  a.x /= b;
  a.y /= b;
  return a;
}
__host__ __device__ inline
cuDoubleComplexFull& operator/=(cuDoubleComplexFull & a, const double b) {
  a.x /= b;
  a.y /= b;
  return a;
}
__host__ __device__ inline
cuDoubleComplexFull& operator/=(cuDoubleComplexFull & a, const int b) {
  a.x /= b;
  a.y /= b;
  return a;
}
__host__ __device__ inline
volatile cuFloatComplexFull& operator/=(volatile cuFloatComplexFull & a,
					const cuFloatComplexFull b) {
  float temp = a.x;
  a.x = (temp*b.x + a.y*b.y)/(b.x*b.x+b.y*b.y);
  a.y = (-temp*b.y + a.y*b.x)/(b.x*b.x+b.y*b.y);
  return a;
}
__host__ __device__ inline
volatile cuFloatComplexFull& operator/=(volatile cuFloatComplexFull & a,
					const cuDoubleComplexFull b) {
  float temp = a.x;
  a.x = float((temp*b.x + a.y*b.y)/(b.x*b.x+b.y*b.y));
  a.y = float((-temp*b.y + a.y*b.x)/(b.x*b.x+b.y*b.y));
  return a;
}
__host__ __device__ inline
volatile cuFloatComplexFull& operator/=(volatile cuFloatComplexFull & a,
					const float b) {
  a.x /= b;
  a.y /= b;
  return a;
}
__host__ __device__ inline
volatile cuFloatComplexFull& operator/=(volatile cuFloatComplexFull & a,
					const double b) {
  a.x /= b;
  a.y /= b;
  return a;
}
__host__ __device__ inline
volatile cuFloatComplexFull& operator/=(volatile cuFloatComplexFull & a,
					const int b) {
  a.x /= b;
  a.y /= b;
  return a;
}
__host__ __device__ inline
volatile cuDoubleComplexFull& operator/=(volatile cuDoubleComplexFull & a,
					 const cuFloatComplexFull b) {
  double temp = a.x;
  a.x = (temp*b.x + a.y*b.y)/(b.x*b.x+b.y*b.y);
  a.y = (-temp*b.y + a.y*b.x)/(b.x*b.x+b.y*b.y);
  return a;
}
__host__ __device__ inline
volatile cuDoubleComplexFull& operator/=(volatile cuDoubleComplexFull & a,
					 const cuDoubleComplexFull b) {
  double temp = a.x;
  a.x = (temp*b.x + a.y*b.y)/(b.x*b.x+b.y*b.y);
  a.y = (-temp*b.y + a.y*b.x)/(b.x*b.x+b.y*b.y);
  return a;
}
__host__ __device__ inline
volatile cuDoubleComplexFull& operator/=(volatile cuDoubleComplexFull & a,
					 const float b) {
  a.x /= b;
  a.y /= b;
  return a;
}
__host__ __device__ inline
volatile cuDoubleComplexFull& operator/=(volatile cuDoubleComplexFull & a,
					 const double b) {
  a.x /= b;
  a.y /= b;
  return a;
}
__host__ __device__ inline
volatile cuDoubleComplexFull& operator/=(volatile cuDoubleComplexFull & a,
					 const int b) {
  a.x /= b;
  a.y /= b;
  return a;
}

#endif
