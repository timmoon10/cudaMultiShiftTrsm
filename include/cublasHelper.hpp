#pragma once

#include <iostream>
#include <complex>

#include <cublas_v2.h>
#include <thrust/complex.h>

// ===============================================
// Error checking
// ===============================================

static const char* cublasGetErrorString(cublasStatus_t status) {
  switch(status) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
  default:
    return "unknown error";
  }
}

#define CUBLAS_CHECK(status)					\
  do {								\
    cublasStatus_t e = (status);				\
    if(e != CUBLAS_STATUS_SUCCESS) {				\
      std::cerr << "cuBLAS error "				\
		<< "(" << __FILE__ << ":" << __LINE__ << ")"	\
		<< ": " << cublasGetErrorString(e)		\
		<< std::endl;					\
      exit(EXIT_FAILURE); /* TODO: find better way to fail */	\
    }								\
  } while(0)

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
inline
cublasStatus_t cublasScal(cublasHandle_t handle, int n,
			  const thrust::complex<float> *alpha, 
			  thrust::complex<float> *x, int incx) {
  return cublasCscal(handle,n,(cuFloatComplex*)alpha,
		     (cuFloatComplex*)x,incx);
}
inline
cublasStatus_t cublasScal(cublasHandle_t handle, int n,
			  const thrust::complex<double> *alpha, 
			  thrust::complex<double> *x, int incx) {
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
inline
cublasStatus_t cublasGemm(cublasHandle_t handle,
			  cublasOperation_t transa,
			  cublasOperation_t transb,
			  int m, int n, int k,
			  const thrust::complex<float> *alpha,
			  const thrust::complex<float> *A, int lda,
			  const thrust::complex<float> *B, int ldb,
			  const thrust::complex<float> *beta,
			  thrust::complex<float> *C, int ldc) {
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
			  const thrust::complex<double> *alpha,
			  const thrust::complex<double> *A, int lda,
			  const thrust::complex<double> *B, int ldb,
			  const thrust::complex<double> *beta,
			  thrust::complex<double> *C, int ldc) {
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
inline
cublasStatus_t cublasTrsm(cublasHandle_t handle,
			  cublasSideMode_t side,
			  cublasFillMode_t uplo,
			  cublasOperation_t trans,
			  cublasDiagType_t diag,
			  int m, int n,
			  const thrust::complex<float> *alpha,
			  const thrust::complex<float> *A, int lda,
			  thrust::complex<float> * B, int ldb) {
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
			  const thrust::complex<double> *alpha,
			  const thrust::complex<double> *A, int lda,
			  thrust::complex<double> * B, int ldb) {
  return cublasZtrsm(handle,side,uplo,trans,diag,m,n,
		     (cuDoubleComplex*)alpha,(cuDoubleComplex*)A,lda,
		     (cuDoubleComplex*)B,ldb);
}

// GEAM
inline
cublasStatus_t cublasGeam(cublasHandle_t handle,
			  cublasOperation_t transa,
			  cublasOperation_t transb,
			  int m, int n,
			  const float *alpha,
			  const float *A, int lda,
			  const float *beta,
			  const float *B, int ldb,
			  float *C, int ldc) {
  return cublasSgeam(handle, transa, transb, m, n,
		     alpha, A, lda, beta, B, ldb, C, ldc);
}
inline
cublasStatus_t cublasGeam(cublasHandle_t handle,
			  cublasOperation_t transa,
			  cublasOperation_t transb,
			  int m, int n,
			  const double *alpha,
			  const double *A, int lda,
			  const double *beta,
			  const double *B, int ldb,
			  double *C, int ldc) {
  return cublasDgeam(handle, transa, transb, m, n,
		     alpha, A, lda, beta, B, ldb, C, ldc);
}
inline
cublasStatus_t cublasGeam(cublasHandle_t handle,
			  cublasOperation_t transa,
			  cublasOperation_t transb,
			  int m, int n,
			  const thrust::complex<float> *alpha,
			  const thrust::complex<float> *A, int lda,
			  const thrust::complex<float> *beta,
			  const thrust::complex<float> *B, int ldb,
			  thrust::complex<float> *C, int ldc) {
  return cublasCgeam(handle, transa, transb, m, n,
		     (cuFloatComplex*)alpha, (cuFloatComplex*)A, lda,
		     (cuFloatComplex*)beta, (cuFloatComplex*)B, ldb,
		     (cuFloatComplex*)C, ldc);
}
inline
cublasStatus_t cublasGeam(cublasHandle_t handle,
			  cublasOperation_t transa,
			  cublasOperation_t transb,
			  int m, int n,
			  const thrust::complex<double> *alpha,
			  const thrust::complex<double> *A, int lda,
			  const thrust::complex<double> *beta,
			  const thrust::complex<double> *B, int ldb,
			  thrust::complex<double> *C, int ldc) {
  return cublasZgeam(handle, transa, transb, m, n,
		     (cuDoubleComplex*)alpha,
		     (cuDoubleComplex*)A, lda,
		     (cuDoubleComplex*)beta,
		     (cuDoubleComplex*)B, ldb,
		     (cuDoubleComplex*)C, ldc);
}
