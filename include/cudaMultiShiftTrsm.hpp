#pragma once

#include <cublas_v2.h>

namespace cudaMstrsm {

  // -------------------------------------------
  // Multi-shift triangular solve
  // -------------------------------------------

  /// Solve triangular systems with multiple shifts
  template<typename F>
  cublasStatus_t cudaMultiShiftTrsm(cublasHandle_t handle,
				    const cublasSideMode_t side,
				    const cublasFillMode_t uplo,
				    const cublasOperation_t trans,
				    const cublasDiagType_t diag,
				    const int m, const int n,
				    const F * __restrict__ alpha,
				    const F * __restrict__ A, const int lda,
				    F * __restrict__ B, const int ldb,
				    const F * __restrict__ shifts);
}
