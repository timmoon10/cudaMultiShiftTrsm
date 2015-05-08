#ifndef __GPU_TRSMRS__
#define __GPU_TRSMRS__

#include <stdio.h>
#include <cuda.h>
#include <cublas_v2.h>

cublasStatus_t gpuStrsms(cublasHandle_t handle,
			 cublasSideMode_t side, cublasFillMode_t uplo,
			 cublasOperation_t trans, cublasDiagType_t diag,
			 int m, int n,
			 const float * __restrict__ alpha,
			 const float * __restrict__ A, int lda,
			 float * __restrict__ B, int ldb,
			 const float * __restrict__ shifts);
cublasStatus_t gpuDtrsms(cublasHandle_t handle,
			 cublasSideMode_t side, cublasFillMode_t uplo,
			 cublasOperation_t trans, cublasDiagType_t diag,
			 int m, int n,
			 const double * __restrict__ alpha,
			 const double * __restrict__ A, int lda,
			 double * __restrict__ B, int ldb,
			 const double * __restrict__ shifts);
cublasStatus_t gpuCtrsms(cublasHandle_t handle,
			 cublasSideMode_t side, cublasFillMode_t uplo,
			 cublasOperation_t trans, cublasDiagType_t diag,
			 int m, int n,
			 const cuComplex * __restrict__ alpha,
			 const cuComplex * __restrict__ A, int lda,
			 cuComplex * __restrict__ B, int ldb,
			 const cuComplex * __restrict__ shifts);
cublasStatus_t gpuZtrsms(cublasHandle_t handle,
			 cublasSideMode_t side, cublasFillMode_t uplo,
			 cublasOperation_t trans, cublasDiagType_t diag,
			 int m, int n,
			 const cuDoubleComplex * __restrict__ alpha,
			 const cuDoubleComplex * __restrict__ A, int lda,
			 cuDoubleComplex * __restrict__ B, int ldb,
			 const cuDoubleComplex * __restrict__ shifts);

#endif
