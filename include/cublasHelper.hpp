#include <complex>
#include <cublas_v2.h>

#ifndef __CUBLASHELPER_HPP__
#define __CUBLASHELPER_HPP__

// ===============================================
// Unified interface for cuBLAS calls
// TODO: Add all cuBLAS routines
// ===============================================

// SCAL
inline
cublasStatus_t cublasScal(cublasHandle_t handle, int n,
			  const float *alpha,
			  float *x, int incx) {
  return cublasSscal(handle,n,alpha,x,incx);
}
inline
cublasStatus_t cublasScal(cublasHandle_t handle, int n,
			  const double *alpha,
			  double *x, int incx) {
  return cublasDscal(handle,n,alpha,x,incx);
}
inline
cublasStatus_t cublasScal(cublasHandle_t handle, int n,
			  const cuFloatComplex *alpha, 
			  cuFloatComplex *x, int incx) {
  return cublasCscal(handle,n,alpha,x,incx);
}
inline
cublasStatus_t cublasScal(cublasHandle_t handle, int n,
			  const cuDoubleComplex *alpha, 
			  cuDoubleComplex *x, int incx) {
  return cublasZscal(handle,n,alpha,x,incx);
}
inline
cublasStatus_t cublasScal(cublasHandle_t handle, int n,
			  const std::complex<float> *alpha, 
			  std::complex<float> *x, int incx) {
  return cublasCscal(handle,n,(cuFloatComplex*)alpha,
		     (cuFloatComplex*)x,incx);
}
inline
cublasStatus_t cublasScal(cublasHandle_t handle, int n,
			  const std::complex<double> *alpha, 
			  std::complex<double> *x, int incx) {
  return cublasZscal(handle,n,(cuDoubleComplex*)alpha,
		     (cuDoubleComplex*)x,incx);
}

// GEMM
inline
cublasStatus_t cublasGemm(cublasHandle_t handle,
			  cublasOperation_t transa,
			  cublasOperation_t transb,
			  int m, int n, int k,
			  const float *alpha,
			  const float *A, int lda,
			  const float *B, int ldb,
			  const float *beta,
			  float *C, int ldc) {
  return cublasSgemm(handle,transa,transb,m,n,k,
		     alpha,A,lda,B,ldb,beta,C,ldc);
}
inline
cublasStatus_t cublasGemm(cublasHandle_t handle,
			  cublasOperation_t transa,
			  cublasOperation_t transb,
			  int m, int n, int k,
			  const double *alpha,
			  const double *A, int lda,
			  const double *B, int ldb,
			  const double *beta,
			  double *C, int ldc) {
  return cublasDgemm(handle,transa,transb,m,n,k,
		     alpha,A,lda,B,ldb,beta,C,ldc);
}
inline
cublasStatus_t cublasGemm(cublasHandle_t handle,
			  cublasOperation_t transa,
			  cublasOperation_t transb,
			  int m, int n, int k,
			  const cuFloatComplex *alpha,
			  const cuFloatComplex *A, int lda,
			  const cuFloatComplex *B, int ldb,
			  const cuFloatComplex *beta,
			  cuFloatComplex *C, int ldc) {
  return cublasCgemm(handle,transa,transb,m,n,k,
		     alpha,A,lda,B,ldb,beta,C,ldc);
}
inline
cublasStatus_t cublasGemm(cublasHandle_t handle,
			  cublasOperation_t transa,
			  cublasOperation_t transb,
			  int m, int n, int k,
			  const cuDoubleComplex *alpha,
			  const cuDoubleComplex *A, int lda,
			  const cuDoubleComplex *B, int ldb,
			  const cuDoubleComplex *beta,
			  cuDoubleComplex *C, int ldc) {
  return cublasZgemm(handle,transa,transb,m,n,k,
		     alpha,A,lda,B,ldb,beta,C,ldc);
}
inline
cublasStatus_t cublasGemm(cublasHandle_t handle,
			  cublasOperation_t transa,
			  cublasOperation_t transb,
			  int m, int n, int k,
			  const std::complex<float> *alpha,
			  const std::complex<float> *A, int lda,
			  const std::complex<float> *B, int ldb,
			  const std::complex<float> *beta,
			  std::complex<float> *C, int ldc) {
  return cublasCgemm(handle,transa,transb,m,n,k,
		     (cuFloatComplex*)alpha,(cuFloatComplex*)A,lda,
		     (cuFloatComplex*)B,ldb,
		     (cuFloatComplex*)beta,(cuFloatComplex*)C,ldc);
}
inline
cublasStatus_t cublasGemm(cublasHandle_t handle,
			  cublasOperation_t transa,
			  cublasOperation_t transb,
			  int m, int n, int k,
			  const std::complex<double> *alpha,
			  const std::complex<double> *A, int lda,
			  const std::complex<double> *B, int ldb,
			  const std::complex<double> *beta,
			  std::complex<double> *C, int ldc) {
  return cublasZgemm(handle,transa,transb,m,n,k,
		     (cuDoubleComplex*)alpha,(cuDoubleComplex*)A,lda,
		     (cuDoubleComplex*)B,ldb,
		     (cuDoubleComplex*)beta,(cuDoubleComplex*)C,ldc);
}

// TRSM
inline
cublasStatus_t cublasTrsm(cublasHandle_t handle,
			  cublasSideMode_t side,
			  cublasFillMode_t uplo,
			  cublasOperation_t trans,
			  cublasDiagType_t diag,
			  int m, int n,
			  const float *alpha,
			  const float *A, int lda,
			  float * B, int ldb) {
  return cublasStrsm(handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb);
}
inline
cublasStatus_t cublasTrsm(cublasHandle_t handle,
			  cublasSideMode_t side,
			  cublasFillMode_t uplo,
			  cublasOperation_t trans,
			  cublasDiagType_t diag,
			  int m, int n,
			  const double *alpha,
			  const double *A, int lda,
			  double * B, int ldb) {
  return cublasDtrsm(handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb);
}
inline
cublasStatus_t cublasTrsm(cublasHandle_t handle,
			  cublasSideMode_t side,
			  cublasFillMode_t uplo,
			  cublasOperation_t trans,
			  cublasDiagType_t diag,
			  int m, int n,
			  const cuFloatComplex *alpha,
			  const cuFloatComplex *A, int lda,
			  cuFloatComplex * B, int ldb) {
  return cublasCtrsm(handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb);
}
inline
cublasStatus_t cublasTrsm(cublasHandle_t handle,
			  cublasSideMode_t side,
			  cublasFillMode_t uplo,
			  cublasOperation_t trans,
			  cublasDiagType_t diag,
			  int m, int n,
			  const cuDoubleComplex *alpha,
			  const cuDoubleComplex *A, int lda,
			  cuDoubleComplex * B, int ldb) {
  return cublasZtrsm(handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb);
}
inline
cublasStatus_t cublasTrsm(cublasHandle_t handle,
			  cublasSideMode_t side,
			  cublasFillMode_t uplo,
			  cublasOperation_t trans,
			  cublasDiagType_t diag,
			  int m, int n,
			  const std::complex<float> *alpha,
			  const std::complex<float> *A, int lda,
			  std::complex<float> * B, int ldb) {
  return cublasCtrsm(handle,side,uplo,trans,diag,m,n,
		     (cuFloatComplex*)alpha,(cuFloatComplex*)A,lda,
		     (cuFloatComplex*)B,ldb);
}
inline
cublasStatus_t cublasTrsm(cublasHandle_t handle,
			  cublasSideMode_t side,
			  cublasFillMode_t uplo,
			  cublasOperation_t trans,
			  cublasDiagType_t diag,
			  int m, int n,
			  const std::complex<double> *alpha,
			  const std::complex<double> *A, int lda,
			  std::complex<double> * B, int ldb) {
  return cublasZtrsm(handle,side,uplo,trans,diag,m,n,
		     (cuDoubleComplex*)alpha,(cuDoubleComplex*)A,lda,
		     (cuDoubleComplex*)B,ldb);
}


#endif
