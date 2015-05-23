#include <stdio.h>
#include <complex>
#include <cuda.h>
#include <cublas_v2.h>
#include "cublasHelper.hpp"
#include "cuComplexHelper.hpp"

#ifndef __CUDA_MULTISHIFTTRSM_HPP__
#define __CUDA_MULTISHIFTTRSM_HPP__

namespace cudaMstrsm {

  // -------------------------------------------
  // Misc
  // -------------------------------------------

  // CUDA warp size
  const int BSIZE = 32;
  // Matrix entry index with Fortran ordering
  __host__ __device__ inline
  int idx(int i,int j,int ld) {return i+j*ld;}

  // -------------------------------------------
  // CUDA Kernels
  // -------------------------------------------

  /// Solve triangular systems with multiple shifts (LLN case)
  template <typename F>
  __global__ void LLN_block(const bool unitDiag,
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
      shared_B[tid] = B[idx(tid,blockIdx.x,ldb)];
    if(tid == 0) {
      if(shifts == 0)
	shared_shift = 0;
      else
	shared_shift = shifts[blockIdx.x];
    }

    // Perform forward substitution
    for(int i=0; i<m; ++i) {

      // Copy global memory to private memory
      F private_A;
      __syncthreads();
      if(i<=tid && tid<m)
	private_A = A[idx(tid,i,lda)];
      if(unitDiag && tid==i)
	private_A = 1;

      // Obtain ith row of solution
      if(tid==i)
	shared_B[i] /= private_A + shared_shift;

      // Update remaining rows of RHS matrix
      __syncthreads();
      if(tid>i)
	shared_B[tid] -= private_A*shared_B[i];

    }

    // Copy shared memory to global memory
    __syncthreads();
    if(tid < m)
      B[idx(tid,blockIdx.x,ldb)] = shared_B[tid];
  }

  /// Solve triangular systems with multiple shifts (LUN case)
  template <typename F>
  __global__ void LUN_block(const bool unitDiag,
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
      shared_B[tid] = B[idx(tid,blockIdx.x,ldb)];
    if(tid == 0) {
      if(shifts == 0)
	shared_shift = 0;
      else
	shared_shift = shifts[blockIdx.x];
    }

    // Perform backward substitution
    for(int i=m-1; i>=0; --i) {

      // Copy global memory to private memory
      F private_A;
      __syncthreads();
      if(tid<=i)
	private_A = A[idx(tid,i,lda)];
      if(unitDiag && tid==i)
	private_A = 1;

      // Obtain ith row of solution
      if(tid==i)
	shared_B[tid] /= private_A+shared_shift;

      // Update remaining rows of RHS matrix
      __syncthreads();
      if(tid<i)
	shared_B[tid] -= private_A*shared_B[i];
      
    }

    // Copy shared memory to global memory
    __syncthreads();
    if(tid < m)
      B[idx(tid,blockIdx.x,ldb)] = shared_B[tid];
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
      fprintf(stderr,"Error (cudaMultiShiftTrsm): argument 6 is invalid (m<0)\n");
      return CUBLAS_STATUS_INVALID_VALUE;
    }
    if(n < 0) {
      fprintf(stderr,"Error (cudaMultiShiftTrsm): argument 7 is invalid (n<0)\n");
      return CUBLAS_STATUS_INVALID_VALUE;
    }
    if(lda < max(1,m)){
      fprintf(stderr,"Error (cudaMultiShiftTrsm): argument 10 is invalid (lda<max(1,m))\n");
      return CUBLAS_STATUS_INVALID_VALUE;
    }
    if(ldb < max(1,m)) {
      fprintf(stderr,"Error (cudaMultiShiftTrsm): argument 12 is invalid (lda<max(1,m))\n");
      return CUBLAS_STATUS_INVALID_VALUE;
    }

    // Error if an unimplemented feature is called
    // TODO: remove this section when possible
    if(side != CUBLAS_SIDE_LEFT) {
      fprintf(stderr,
	      "Error (cudaMultiShiftTrsm): invalid input in argument 2\n"
	      "  side=CUBLAS_SIDE_RIGHT is not yet implemented\n");
      return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    if(trans != CUBLAS_OP_N) {
      fprintf(stderr,
	      "Error (cudaMultiShiftTrsm): invalid input in argument 4\n"
	      "  trans=CUBLAS_OP_T and trans=CUBLAS_OP_C are not yet implemented\n");
      return CUBLAS_STATUS_NOT_SUPPORTED;
    }

    // Return zero if right hand side is zero
    if(alpha == 0) {
      for(int i=0; i<n; ++i) {
	cudaError_t cudaStatus = cudaMemset(B+idx(0,i,ldb),0,m*sizeof(F));
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
      status = cublasScal(handle, m, alpha, B+idx(0,i,ldb), 1);
      if(status != CUBLAS_STATUS_SUCCESS)
	return status;
    }

    // Misc initialization
    bool unitDiag = (diag==CUBLAS_DIAG_UNIT);
    F one    = 1;
    F negOne = -1;
    int numBlocks = (m+BSIZE-1)/BSIZE;  // Number of subblocks in A
  
    // LLN case
    if(side==CUBLAS_SIDE_LEFT
       && uplo==CUBLAS_FILL_MODE_LOWER
       && trans==CUBLAS_OP_N) {
      
      int i = 0;  // Current row in A
      for(int b=0; b<numBlocks-1; ++b) {
	LLN_block <<< n, BSIZE, 0, stream >>>
	  (unitDiag, BSIZE,n,A+idx(i,i,lda),lda,B+i,ldb,shifts);
	status = cublasGemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
			    m-(i+BSIZE),n,BSIZE,
			    &negOne, A+idx(i+BSIZE,i,lda), lda,
			    B+i,ldb, &one, B+i+BSIZE, ldb);
	if(status != CUBLAS_STATUS_SUCCESS)
	  return status;
	i += BSIZE;
      }
      LLN_block <<< n, BSIZE, 0, stream >>>
	(unitDiag,m-i,n,A+idx(i,i,lda),lda,B+i,ldb,shifts);
    }

    // LUN case
    else if(side==CUBLAS_SIDE_LEFT
       && uplo==CUBLAS_FILL_MODE_UPPER
       && trans==CUBLAS_OP_N) {

      int i = m-BSIZE; // Current row in A
      for(int b=numBlocks-1; b>0; --b) {
	LUN_block <<< n, BSIZE, 0, stream >>>
	  (unitDiag,BSIZE,n,A+idx(i,i,lda),lda,B+i,ldb,shifts);
	status = cublasGemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
			    i,n,BSIZE,&negOne,A+idx(0,i,lda),lda,
			    B+i,ldb,&one,B,ldb);
	if(status != CUBLAS_STATUS_SUCCESS)
	  return status;
	i -= BSIZE;
      }
      LUN_block <<< n, BSIZE, 0, stream >>>
	(unitDiag,i+BSIZE,n,A,lda,B,ldb,shifts);

    }
    
    // Function has completed successfully
    return CUBLAS_STATUS_SUCCESS;

  }

  template <> inline
  cublasStatus_t 
  cudaMultiShiftTrsm<std::complex<float> >
  (const cublasHandle_t handle,
   const cublasSideMode_t side, const cublasFillMode_t uplo,
   const cublasOperation_t trans, const cublasDiagType_t diag,
   const int m, const int n, const std::complex<float> * __restrict__ alpha,
   const std::complex<float> * __restrict__ A, const int lda,
   std::complex<float> * __restrict__ B, const int ldb,
   const std::complex<float> * __restrict__ shifts) {
    return cudaMultiShiftTrsm<cuFloatComplexFull>
      (handle,side,uplo,trans,diag,m,n,(cuFloatComplexFull*)alpha,
       (cuFloatComplexFull*)A,lda,(cuFloatComplexFull*)B,ldb,
       (cuFloatComplexFull*)shifts);
  }
  template <> inline
  cublasStatus_t 
  cudaMultiShiftTrsm<cuFloatComplex>
  (const cublasHandle_t handle,
   const cublasSideMode_t side, const cublasFillMode_t uplo,
   const cublasOperation_t trans, const cublasDiagType_t diag,
   const int m, const int n, const cuFloatComplex * __restrict__ alpha,
   const cuFloatComplex * __restrict__ A, const int lda,
   cuFloatComplex * __restrict__ B, const int ldb,
   const cuFloatComplex * __restrict__ shifts) {
    return cudaMultiShiftTrsm<cuFloatComplexFull>
      (handle,side,uplo,trans,diag,m,n,(cuFloatComplexFull*)alpha,
       (cuFloatComplexFull*)A,lda,(cuFloatComplexFull*)B,ldb,
       (cuFloatComplexFull*)shifts);
  }
  template <> inline
  cublasStatus_t 
  cudaMultiShiftTrsm<std::complex<double> >
  (const cublasHandle_t handle,
   const cublasSideMode_t side, const cublasFillMode_t uplo,
   const cublasOperation_t trans, const cublasDiagType_t diag,
   const int m, const int n, const std::complex<double> * __restrict__ alpha,
   const std::complex<double> * __restrict__ A, const int lda,
   std::complex<double> * __restrict__ B, const int ldb,
   const std::complex<double> * __restrict__ shifts) {
    return cudaMultiShiftTrsm<cuDoubleComplexFull>
      (handle,side,uplo,trans,diag,m,n,(cuDoubleComplexFull*)alpha,
       (cuDoubleComplexFull*)A,lda,(cuDoubleComplexFull*)B,ldb,
       (cuDoubleComplexFull*)shifts);
  }
  template <> inline
  cublasStatus_t 
  cudaMultiShiftTrsm<cuDoubleComplex>
  (const cublasHandle_t handle,
   const cublasSideMode_t side, const cublasFillMode_t uplo,
   const cublasOperation_t trans, const cublasDiagType_t diag,
   const int m, const int n, const cuDoubleComplex * __restrict__ alpha,
   const cuDoubleComplex * __restrict__ A, const int lda,
   cuDoubleComplex * __restrict__ B, const int ldb,
   const cuDoubleComplex * __restrict__ shifts) {
    return cudaMultiShiftTrsm<cuDoubleComplexFull>
      (handle,side,uplo,trans,diag,m,n,(cuDoubleComplexFull*)alpha,
       (cuDoubleComplexFull*)A,lda,(cuDoubleComplexFull*)B,ldb,
       (cuDoubleComplexFull*)shifts);
  }

}

#endif
