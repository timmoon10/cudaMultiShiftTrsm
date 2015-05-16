#ifndef __CUDA_MULTISHIFTTRSM_HPP__
#define __CUDA_MULTISHIFTTRSM_HPP__

#include <stdio.h>
#include <cuda.h>
#include <cublas_v2.h>

#define BSIZE 32 // CUDA warp size
#define IDX(i,j,ld) ((i)+(j)*(ld))

namespace cudaMstrsm {
  
  // -------------------------------------------
  // cuBLAS Functions
  // (For use in templated functions)
  // -------------------------------------------
  inline
  cublasStatus_t cublasGscal(cublasHandle_t handle, int n,
			     const float *alpha, float *x, int incx) {
    return cublasSscal(handle,n,alpha,x,incx);
  }
  inline
  cublasStatus_t cublasGscal(cublasHandle_t handle, int n,
			     const double *alpha, double *x, int incx) {
    return cublasDscal(handle,n,alpha,x,incx);
  }
  inline
  cublasStatus_t cublasGscal(cublasHandle_t handle, int n,
			     const cuComplex *alpha, cuComplex *x, int incx) {
    return cublasCscal(handle,n,alpha,x,incx);
  }
  inline
  cublasStatus_t cublasGscal(cublasHandle_t handle, int n,
			     const cuDoubleComplex *alpha, cuDoubleComplex *x, int incx) {
    return cublasZscal(handle,n,alpha,x,incx);
  }
  inline
  cublasStatus_t cublasGgemm(cublasHandle_t handle, cublasOperation_t transa,
			     cublasOperation_t transb, int m, int n, int k,
			     const float *alpha, const float *A, int lda,
			     const float *B, int ldb,
			     const float *beta, float *C, int ldc) {
    return cublasSgemm(handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
  }
  inline
  cublasStatus_t cublasGgemm(cublasHandle_t handle, cublasOperation_t transa,
			     cublasOperation_t transb, int m, int n, int k,
			     const double *alpha, const double *A, int lda,
			     const double *B, int ldb,
			     const double *beta, double *C, int ldc) {
    return cublasDgemm(handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
  }
  inline
  cublasStatus_t cublasGgemm(cublasHandle_t handle, cublasOperation_t transa,
			     cublasOperation_t transb, int m, int n, int k,
			     const cuComplex *alpha, const cuComplex *A, int lda,
			     const cuComplex *B, int ldb,
			     const cuComplex *beta, cuComplex *C, int ldc) {
    return cublasCgemm(handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
  }
  inline
  cublasStatus_t cublasGgemm(cublasHandle_t handle, cublasOperation_t transa,
			     cublasOperation_t transb, int m, int n, int k,
			     const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda,
			     const cuDoubleComplex *B, int ldb,
			     const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
    return cublasZgemm(handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
  }

  // -------------------------------------------
  // CUDA Kernels
  // -------------------------------------------

  /// Solve triangular systems with multiple shifts
  template <typename F>
  __global__ void mstrsmBlock(const bool diag,
			      const int m,
			      const int n,
			      const F * __restrict__ A,
			      const int lda,
			      F * __restrict__ B,
			      const int ldb,
			      const F * __restrict__ shifts) {

    // Initialize indices
    int tid = threadIdx.x;

    // Copy global memory to shared memory
    __shared__ F shared_B[BSIZE];
    __shared__ F shared_shift;
    if(tid < m)
      shared_B[tid] = B[IDX(tid,blockIdx.x,ldb)];
    if(tid == 0) {
      if(shifts == 0)
	shared_shift = 0.f;
      else
	shared_shift = shifts[blockIdx.x];
    }

    // Perform forward substitution
    for(int i=0; i<m; ++i) {

      if(i<=tid && tid<m) {

	// Copy global memory to private memory
	__syncthreads();
	F private_A = A[IDX(tid,i,lda)];

	// Obtain ith row of solution
	if(tid==i) {
	  if(diag)
	    shared_B[tid] /= (private_A+shared_shift);
	  else {
	    // If matrix is unit diagonal
	    shared_B[tid] /= (1.f+shared_shift);
	  }
	}

	// Update remaining rows of RHS matrix
	__syncthreads();
	if(tid>i)
	  shared_B[tid] -= private_A*shared_B[i];
      }
    }

    // Copy shared memory to global memory
    if(tid < m)
      B[IDX(tid,blockIdx.x,ldb)] = shared_B[tid];
  }

  // -------------------------------------------
  // Multi-shift triangular solve
  // -------------------------------------------

  /// Solve triangular systems with multiple shifts
  template<typename F>
  cublasStatus_t cudaMultiShiftTrsm(const cublasHandle_t handle,
				    const cublasSideMode_t side,
				    const cublasFillMode_t uplo,
				    const cublasOperation_t trans,
				    const cublasDiagType_t diag,
				    const int m, const int n,
				    const F * __restrict__ alpha,
				    const F * __restrict__ A, const int lda,
				    F * __restrict__ B, const int ldb,
				    const F * __restrict__ shifts) {

    // Initialize CUDA and cuBLAS objects
    cublasStatus_t status;
    cudaStream_t stream;
    status = cublasGetStream(handle, &stream);
    if(status != CUBLAS_STATUS_SUCCESS)
      return status;

    // Report invalid parameters
    if(m < 0) {
      printf("Error (gpuStrsms): argument 6 is invalid (m<0)\n");
      return CUBLAS_STATUS_INVALID_VALUE;
    }
    if(n < 0) {
      printf("Error (gpuStrsms): argument 7 is invalid (n<0)\n");
      return CUBLAS_STATUS_INVALID_VALUE;
    }
    if(lda < max(1,m)){
      printf("Error (gpuStrsms): argument 10 is invalid (lda<max(1,m))\n");
      return CUBLAS_STATUS_INVALID_VALUE;
    }
    if(ldb < max(1,m)) {
      printf("Error (gpuStrsms): argument 12 is invalid (lda<max(1,m))\n");
      return CUBLAS_STATUS_INVALID_VALUE;
    }

    // Error if an unimplemented feature is called
    // TODO: remove this section when possible
    if(side != CUBLAS_SIDE_LEFT) {
      fprintf(stderr,
	      "ERROR in gpuStrsms: invalid input in argument 2\n"
	      "  side=CUBLAS_SIDE_RIGHT is not yet implemented\n");
      return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    if(uplo != CUBLAS_FILL_MODE_LOWER) {
      fprintf(stderr,
	      "ERROR in gpuStrsms: invalid input in argument 3\n"
	      "  uplo=CUBLAS_FILL_MODE_UPPER is not yet implemented\n");
      return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    if(trans != CUBLAS_OP_N) {
      fprintf(stderr,
	      "ERROR in gpuStrsms: invalid input in argument 4\n"
	      "  trans=CUBLAS_OP_T and trans=CUBLAS_OP_C are not yet implemented\n");
      return CUBLAS_STATUS_NOT_SUPPORTED;
    }

    // Return zero if right hand side is zero
    if(alpha == 0) {
      for(int i=0; i<n; ++i) {
	cudaError_t cudaStatus = cudaMemset(B+i*ldb,0,m*sizeof(F));
	if(cudaStatus != cudaSuccess) {
	  cudaDeviceSynchronize();
	  return CUBLAS_STATUS_INTERNAL_ERROR;
	}
      }
      cudaDeviceSynchronize();
      return CUBLAS_STATUS_SUCCESS;
    }

    // Scale right hand side
    for(int i=0; i<n; ++i) {
      status = cublasGscal(handle, m, alpha, B+i*ldb, 1);
      if(status != CUBLAS_STATUS_SUCCESS)
	return status;
    }

    // Misc initialization
    bool nonUnitDiag = (diag  == CUBLAS_DIAG_NON_UNIT);
    F one    = 1;
    F negOne = -1;
  
    // Perform blocked triangular solve
    int numBlocks = (m+BSIZE-1)/BSIZE;  // Number of subblocks in A
    int i = 0;                          // Current row in A
    for(int b=0; b<numBlocks-1; ++b) {
      mstrsmBlock <<< n, BSIZE, 0, stream >>>
	(nonUnitDiag, BSIZE,n,A+i+i*lda,lda,B+i,ldb,shifts);
      status = cublasGgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
			   m-(i+BSIZE),n,BSIZE,
			   &negOne, A+i+BSIZE+i*lda, lda,
			   B+i,ldb, &one, B+i+BSIZE, ldb);
      if(status != CUBLAS_STATUS_SUCCESS)
	return status;
      i += BSIZE;
    }
    mstrsmBlock <<< n, BSIZE, 0, stream >>>
      (nonUnitDiag,m-i,n,A+i+i*lda,lda,B+i,ldb,shifts);

    // Function has completed successfully
    return CUBLAS_STATUS_SUCCESS;

  }

}

#endif
