#pragma once

#include <cublas_v2.h>

namespace cudaMstrsm {

  // -------------------------------------------
  // Multi-shift triangular solve
  // -------------------------------------------

  /// Solve triangular systems with multiple shifts
  template<typename F>
  cublasStatus_t cudaMultiShiftTrsm(cublasHandle_t handle,
				    cublasFillMode_t uplo,
				    cublasOperation_t trans,
				    int m, int n,
				    const F * alpha,
				    const F * __restrict__ A, int lda,
				    F * __restrict__ B, int ldb,
				    const F * __restrict__ shifts);
}
