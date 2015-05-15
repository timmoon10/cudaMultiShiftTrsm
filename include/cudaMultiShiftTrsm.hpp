#ifndef __CUDA_MULTISHIFTTRSM__
#define __CUDA_MULTISHIFTTRSM__

#include <stdio.h>
#include <cuda.h>
#include <cublas_v2.h>

template<typename F>
cublasStatus_t cudaMultiShiftTrsm(cublasHandle_t handle, cublasSideMode_t side,
				  cublasFillMode_t uplo, cublasOperation_t trans,
				  cublasDiagType_t diag, int m, int n,
				  const F * __restrict__ alpha,
				  const F * __restrict__ A, int lda,
				  F * __restrict__ B, int ldb,
				  const F * __restrict__ shifts);

#endif
