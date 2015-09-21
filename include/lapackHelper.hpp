#pragma once

#include <complex>

// ===============================================
// BLAS routines
// ===============================================

// NRM2
extern "C" {
  float snrm2_(int *n, void *x, int *incx);
  double dnrm2_(int *n, void *x, int *incx);
  float scnrm2_(int *n, void *x, int *incx);
  double dznrm2_(int *n, void *x, int *incx);
}
inline
double nrm2(int n, float * x, int incx) {
  return snrm2_(&n,x,&incx);
}
inline
double nrm2(int n, double * x, int incx) {
  return dnrm2_(&n,x,&incx);
}
inline
double nrm2(int n, std::complex<float> * x, int incx) {
  return scnrm2_(&n,x,&incx);
}
inline
double nrm2(int n, std::complex<double> * x, int incx) {
  return dznrm2_(&n,x,&incx);
}

// AXPY
extern "C" {
  void saxpy_(int *n, void *a, void *x, int *incx, void *y, int *incy);
  void daxpy_(int *n, void *a, void *x, int *incx, void *y, int *incy);
  void caxpy_(int *n, void *a, void *x, int *incx, void *y, int *incy);
  void zaxpy_(int *n, void *a, void *x, int *incx, void *y, int *incy);
}
inline
void axpy(int n, float a, float * x, int incx, float * y, int incy) {
  saxpy_(&n,&a,x,&incx,y,&incy); 
}
inline
void axpy(int n, double a, double * x, int incx,
	  double * y, int incy) {
  daxpy_(&n,&a,x,&incx,y,&incy);
}
inline
void axpy(int n, std::complex<float> a,
	  std::complex<float> * x, int incx,
	  std::complex<float> * y, int incy) {
  caxpy_(&n,&a,x,&incx,y,&incy);
}
inline
void axpy(int n, std::complex<double> a,
	  std::complex<double> * x, int incx,
	  std::complex<double> * y, int incy) {
  zaxpy_(&n,&a,x,&incx,y,&incy);
}

// SYRK/HERK
extern "C" {
  void ssyrk_(char *uplo, char *trans, int *n, int *k,
	      void *alpha, void *A, int *lda,
	      void *beta, void *C, int *ldc);
  void dsyrk_(char *uplo, char *trans, int *n, int *k,
	      void *alpha, void *A, int *lda,
	      void *beta, void *C, int *ldc);
  void cherk_(char *uplo, char *trans, int *n, int *k,
	      void *alpha, void *A, int *lda,
	      void *beta, void *C, int *ldc);
  void zherk_(char *uplo, char *trans, int *n, int *k,
	      void *alpha, void *A, int *lda,
	      void *beta, void *C, int *ldc);
}
inline
void herk(char uplo, char trans, int n, int k,
	  float alpha, float * A, int lda,
	  float beta, float * C, int ldc) {
  ssyrk_(&uplo,&trans,&n,&k,&alpha,A,&lda,&beta,C,&ldc);
}
inline
void herk(char uplo, char trans, int n, int k,
	  double alpha, double * A, int lda,
	  double beta, double * C, int ldc) {
  dsyrk_(&uplo,&trans,&n,&k,&alpha,A,&lda,&beta,C,&ldc);
}
inline
void herk(char uplo, char trans, int n, int k,
	  std::complex<float> alpha,
	  std::complex<float> * A, int lda,
	  std::complex<float> beta,
	  std::complex<float> * C, int ldc) {
  cherk_(&uplo,&trans,&n,&k,&alpha,A,&lda,&beta,C,&ldc);
}
inline
void herk(char uplo, char trans, int n, int k,
	  std::complex<double> alpha,
	  std::complex<double> * A, int lda,
	  std::complex<double> beta,
	  std::complex<double> * C, int ldc) {
  zherk_(&uplo,&trans,&n,&k,&alpha,A,&lda,&beta,C,&ldc);
}

// TRMM
extern "C" {
  void strmm_(char *side, char *uplo, char *transa, char *diag, 
	      int *n, int *m, void *alpha, void *A, int *lda,
	      void *B, int *ldb);
  void dtrmm_(char *side, char *uplo, char *transa, char *diag, 
	      int *n, int *m, void *alpha, void *A, int *lda,
	      void *B, int *ldb);
  void ctrmm_(char *side, char *uplo, char *transa, char *diag, 
	      int *n, int *m, void *alpha, void *A, int *lda,
	      void *B, int *ldb);
  void ztrmm_(char *side, char *uplo, char *transa, char *diag, 
	      int *n, int *m, void *alpha, void *A, int *lda,
	      void *B, int *ldb);
}
inline
void trmm(char side, char uplo, char transa, char diag, 
	  int n, int m, float alpha, float * A, int lda,
	  float * B, int ldb) {
  strmm_(&side, &uplo, &transa, &diag, &n, &m, &alpha, A, &lda, B, &ldb);
}
inline
void trmm(char side, char uplo, char transa, char diag, 
	  int n, int m, double alpha, double * A, int lda,
	  double * B, int ldb) {
  dtrmm_(&side, &uplo, &transa, &diag, &n, &m, &alpha, A, &lda, B, &ldb);
}
inline
void trmm(char side, char uplo, char transa, char diag, 
	  int n, int m, std::complex<float> alpha,
	  std::complex<float> * A, int lda,
	  std::complex<float> * B, int ldb) {
  ctrmm_(&side, &uplo, &transa, &diag, &n, &m, &alpha, A, &lda, B, &ldb);
}
inline
void trmm(char side, char uplo, char transa, char diag, 
	  int n, int m, std::complex<double> alpha,
	  std::complex<double> * A, int lda,
	  std::complex<double> * B, int ldb) {
  ztrmm_(&side, &uplo, &transa, &diag, &n, &m, &alpha, A, &lda, B, &ldb);
}

// ===============================================
// LAPACK routines
// ===============================================

// POTRF
extern "C" {
  void spotrf_(char *uplo, int *n, void *A, int *lda, int *info);
  void dpotrf_(char *uplo, int *n, void *A, int *lda, int *info);
  void cpotrf_(char *uplo, int *n, void *A, int *lda, int *info);
  void zpotrf_(char *uplo, int *n, void *A, int *lda, int *info);
}
inline
void potrf(char uplo, int n, float * A, int lda, int &info) {
  spotrf_(&uplo,&n,A,&lda,&info);
}
inline
void potrf(char uplo, int n, double * A, int lda, int &info) {
  dpotrf_(&uplo,&n,A,&lda,&info);
}
inline
void potrf(char uplo, int n,
	   std::complex<float> * A, int lda, int &info) {
  cpotrf_(&uplo,&n,A,&lda,&info);
}
inline
void potrf(char uplo, int n,
	   std::complex<double> * A, int lda, int &info) {
  zpotrf_(&uplo,&n,A,&lda,&info);
}
