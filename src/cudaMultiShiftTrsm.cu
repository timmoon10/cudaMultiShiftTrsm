#include <iostream>
#include <complex>

#include <cuda.h>
#include <cublas_v2.h>
#include <thrust/complex.h>

#include "cudaMultiShiftTrsm.hpp"
#include "cudaHelper.hpp"
#include "cublasHelper.hpp"

#define BSIZE 128

namespace cudaMstrsm {

  namespace {

    // -------------------------------------------
    // Misc
    // -------------------------------------------

    /// Complex conjugate of real numbers
    __host__ __device__ inline
    float conj(float x) {return x;}
    __host__ __device__ inline
    double conj(double x) {return x;}

    // -------------------------------------------
    // CUDA Kernels
    // -------------------------------------------

    /// Solve triangular systems with multiple shifts (LLN case)
    /** Assumes blockDim.x = m
     */
    template <typename F>
    __global__ void LLN_block(const bool unitDiagonal,
			      const int m,
			      const int n,
			      const F * __restrict__ A,
			      const int lda,
			      F * __restrict__ B,
			      const int ldb,
			      const F * __restrict__ shifts) {

      // Initialize indices
      int tid = threadIdx.x;

      // Shared memory
      __shared__ F shared_B[BSIZE];
      __shared__ F shared_shift;

      // Perform forward substitution for each RHS
      for(int bid = blockIdx.x; bid<n; bid+=gridDim.x) {

	// Copy global memory to private and shared memory
	F private_B = B[IDX(tid,bid,ldb)];
	if(tid == 0) {
	  if(shifts == 0)
	    shared_shift = 0;
	  else
	    shared_shift = shifts[bid];
	}
	__threadfence_block();
	// __syncthreads();  // Not needed due to warp-level parallelism

	// Perform forward substitution
	for(int i=0; i<m; ++i) {

	  // Copy global memory to private memory
	  F private_A;
	  if(tid>=i)
	    private_A = A[IDX(tid,i,lda)];

	  // Obtain ith row of solution
	  if(tid==i) {
	    if(unitDiagonal)
	      private_A = 1;
	    private_B /= private_A + shared_shift;
	    shared_B[i] = private_B;
	  }

	  // Update remaining rows of RHS matrix
	  __syncthreads();
	  if(tid>i)
	    private_B -= private_A*shared_B[i];

	}

	// Copy shared memory to global memory
	B[IDX(tid,bid,ldb)] = private_B;
	__syncthreads();

      }

    }

    /// Solve triangular systems with multiple shifts (LUN case)
    /** Assumes blockDim.x = m
     */
    template <typename F>
    __global__ void LUN_block(const bool unitDiagonal,
			      const int m,
			      const int n,
			      const F * __restrict__ A,
			      const int lda,
			      F * __restrict__ B,
			      const int ldb,
			      const F * __restrict__ shifts) {

      // Initialize indices
      int tid = threadIdx.x;

      // Shared memory
      __shared__ F shared_B[BSIZE];
      __shared__ F shared_shift;

      // Perform backward substitution for each RHS
      for(int bid = blockIdx.x; bid<n; bid+=gridDim.x) {

	// Copy global memory to private and shared memory
	F private_B = B[IDX(tid,bid,ldb)];
	if(tid == m-1) {
	  if(shifts == 0)
	    shared_shift = 0;
	  else
	    shared_shift = shifts[bid];
	}
	__threadfence_block();
	// __syncthreads();  // Not needed due to warp-level parallelism

	// Perform backward substitution
	for(int i=m-1; i>=0; --i) {

	  // Copy global memory to private memory
	  F private_A;
	  if(tid<=i)
	    private_A = A[IDX(tid,i,lda)];

	  // Obtain ith row of solution
	  if(tid==i) {
	    if(unitDiagonal)
	      private_A = 1;
	    private_B /= private_A + shared_shift;
	    shared_B[i] = private_B;
	  }

	  // Update remaining rows of RHS matrix
	  __syncthreads();
	  if(tid<i)
	    private_B -= private_A*shared_B[i];
      
	}

	// Copy shared memory to global memory
	B[IDX(tid,bid,ldb)] = private_B;
	__syncthreads();

      }

    }

    /// Solve triangular systems with multiple shifts (LUT and LUC case)
    /** Assumes blockDim.x = m
     *  TODO: slow because memory access is not coalesced
     */
    template <typename F>
    __global__ void LUT_block(const bool unitDiagonal,
			      const bool conjugate,
			      const int m,
			      const int n,
			      const F * __restrict__ A,
			      const int lda,
			      F * __restrict__ B,
			      const int ldb,
			      const F * __restrict__ shifts) {

      using namespace thrust;

      // Initialize indices
      int tid = threadIdx.x;

      // Shared memory
      __shared__ F shared_B[BSIZE];
      __shared__ F shared_shift;

      // Perform forward substitution for each RHS
      for(int bid = blockIdx.x; bid<n; bid+=gridDim.x) {

	// Copy global memory to private and shared memory
	F private_B = B[IDX(tid,bid,ldb)];
	if(tid == 0) {
	  if(shifts == 0)
	    shared_shift = 0;
	  else
	    shared_shift = shifts[bid];
	}
	__threadfence_block();
	// __syncthreads();  // Not needed due to warp-level parallelism

	// Perform forward substitution
	for(int i=0; i<m; ++i) {

	  // Copy global memory to private memory
	  F private_A;
	  if(tid>=i) {
	    if(conjugate)
	      private_A = conj(A[IDX(i,tid,lda)]);
	    else
	      private_A = A[IDX(i,tid,lda)];
	  }

	  // Obtain ith row of solution
	  if(tid==i) {
	    if(unitDiagonal)
	      private_A = 1;
	    private_B /= private_A + shared_shift;
	    shared_B[i] = private_B;
	  }

	  // Update remaining rows of RHS matrix
	  __syncthreads();
	  if(tid>i)
	    private_B -= private_A*shared_B[i];

	}

	// Copy shared memory to global memory
	B[IDX(tid,bid,ldb)] = private_B;

      }

      __syncthreads();

    }

    /// Solve triangular systems with multiple shifts (LLT and LLC case)
    /** Assumes blockDim.x = m
     *  TODO: slow because memory access is not coalesced
     */
    template <typename F>
    __global__ void LLT_block(const bool unitDiagonal,
			      const bool conjugate,
			      const int m,
			      const int n,
			      const F * __restrict__ A,
			      const int lda,
			      F * __restrict__ B,
			      const int ldb,
			      const F * __restrict__ shifts) {

      using namespace thrust;

      // Initialize indices
      int tid = threadIdx.x;

      // Shared memory
      __shared__ F shared_B[BSIZE];
      __shared__ F shared_shift;

      // Perform backward substitution for each RHS
      for(int bid = blockIdx.x; bid<n; bid+=gridDim.x) {

	// Copy global memory to private and shared memory
	F private_B = B[IDX(tid,bid,ldb)];
	if(tid == m-1) {
	  if(shifts == 0)
	    shared_shift = 0;
	  else
	    shared_shift = shifts[bid];
	}
	__threadfence_block();
	// __syncthreads();  // Not needed due to warp-level parallelism

	// Perform backward substitution
	for(int i=m-1; i>=0; --i) {

	  // Copy global memory to private memory
	  F private_A;
	  if(tid<=i) {
	    if(conjugate)
	      private_A = conj(A[IDX(i,tid,lda)]);
	    else
	      private_A = A[IDX(i,tid,lda)];
	  }

	  // Obtain ith row of solution
	  if(tid==i) {
	    if(unitDiagonal)
	      private_A = 1;
	    private_B /= private_A + shared_shift;
	    shared_B[i] = private_B;
	  }

	  // Update remaining rows of RHS matrix
	  __syncthreads();
	  if(tid<i)
	    private_B -= private_A*shared_B[i];
      
	}

	// Copy shared memory to global memory
	B[IDX(tid,bid,ldb)] = private_B;
	__syncthreads();

      }

    }

  }

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
				    const F * __restrict__ shifts) {

    // Initialize CUDA and cuBLAS objects
    cublasStatus_t status;
    cudaError_t cudaStatus;    
    cudaStream_t stream;
    int device;
    cudaDeviceProp prop;
    status = cublasGetStream(handle, &stream);
    if(status != CUBLAS_STATUS_SUCCESS)
      return status;
    cudaStatus = cudaGetDevice(&device);
    if(cudaStatus != cudaSuccess)
      return CUBLAS_STATUS_EXECUTION_FAILED;
    cudaStatus = cudaGetDeviceProperties(&prop, device);
    if(cudaStatus != cudaSuccess)
      return CUBLAS_STATUS_EXECUTION_FAILED;

    // Report invalid parameters
    if(m < 0) {
      WARNING("argument 6 is invalid (m<0)");
      return CUBLAS_STATUS_INVALID_VALUE;
    }
    if(n < 0) {
      WARNING("argument 7 is invalid (n<0)");
      return CUBLAS_STATUS_INVALID_VALUE;
    }
    if(lda < max(1,m)){
      WARNING("argument 10 is invalid (lda<max(1,m))");
      return CUBLAS_STATUS_INVALID_VALUE;
    }
    if(ldb < max(1,m)) {
      WARNING("argument 12 is invalid (lda<max(1,m))");
      return CUBLAS_STATUS_INVALID_VALUE;
    }

    // Error if an unimplemented feature is called
    // TODO: remove this section when possible
    if(side != CUBLAS_SIDE_LEFT) {
      WARNING("invalid input in argument 2 "
	      "(side=CUBLAS_SIDE_RIGHT is not yet implemented)");
      return CUBLAS_STATUS_NOT_SUPPORTED;
    }

    // Return zero if right hand side is zero
    if(alpha == 0) {
      for(int i=0; i<n; ++i) {
	cudaStatus = cudaMemsetAsync(B+IDX(0,i,ldb),0,m*sizeof(F),stream);
	if(cudaStatus != cudaSuccess) {
	  return CUBLAS_STATUS_INTERNAL_ERROR;
	}
      }
      return CUBLAS_STATUS_SUCCESS;
    }

    // Scale right hand side
    for(int i=0; i<n; ++i) {
      status = cublasScal(handle, m, alpha, B+IDX(0,i,ldb), 1);
      if(status != CUBLAS_STATUS_SUCCESS)
	return status;
    }

    // Misc initialization
    bool unitDiagonal = (diag==CUBLAS_DIAG_UNIT);
    bool conjugate    = (trans==CUBLAS_OP_C);
    const F one    = 1;
    const F negOne = -1;
    int numBlocks = (m+BSIZE-1)/BSIZE;  // Number of subblocks in A
    int gridDim = min(n,prop.maxGridSize[0]);

    // LLN case
    if(side==CUBLAS_SIDE_LEFT
       && uplo==CUBLAS_FILL_MODE_LOWER
       && trans==CUBLAS_OP_N) {
      
      // Current row in A
      int i = 0;

      // Partition matrix into subblocks
      for(int b=0; b<numBlocks-1; ++b) {
	LLN_block <<< gridDim, BSIZE, 0, stream >>>
	  (unitDiagonal,BSIZE,n,A+IDX(i,i,lda),lda,B+i,ldb,shifts);
	status = cublasGemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
			    m-(i+BSIZE),n,BSIZE,
			    &negOne,A+IDX(i+BSIZE,i,lda),lda,
			    B+i,ldb,&one,B+i+BSIZE,ldb);
	if(status != CUBLAS_STATUS_SUCCESS)
	  return status;
	i += BSIZE;
      }

      // Final subblock
      LLN_block <<< gridDim, m-i, 0, stream >>>
	(unitDiagonal,m-i,n,A+IDX(i,i,lda),lda,B+i,ldb,shifts);

    }

    // LUN case
    else if(side==CUBLAS_SIDE_LEFT
	    && uplo==CUBLAS_FILL_MODE_UPPER
	    && trans==CUBLAS_OP_N) {

      // Current row in A
      int i = m-BSIZE;

      // Partition matrix into subblocks
      for(int b=numBlocks-1; b>0; --b) {
	LUN_block <<< gridDim, BSIZE, 0, stream >>>
	  (unitDiagonal,BSIZE,n,A+IDX(i,i,lda),lda,B+i,ldb,shifts);
	status = cublasGemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
			    i,n,BSIZE,&negOne,A+IDX(0,i,lda),lda,
			    B+i,ldb,&one,B,ldb);
	if(status != CUBLAS_STATUS_SUCCESS)
	  return status;
	i -= BSIZE;
      }

      // Final subblock
      LUN_block <<< gridDim, i+BSIZE, 0, stream >>>
	(unitDiagonal,i+BSIZE,n,A,lda,B,ldb,shifts);

    }

    // LUT and LUC cases
    else if(side==CUBLAS_SIDE_LEFT
	    && uplo==CUBLAS_FILL_MODE_UPPER
	    && trans!=CUBLAS_OP_N) {

      // Current column in A
      int i = 0;

      // Partition matrix into subblocks
      for(int b=0; b<numBlocks-1; ++b) {
	LUT_block <<< gridDim, BSIZE, 0, stream >>>
	  (unitDiagonal,conjugate,BSIZE,n,A+IDX(i,i,lda),lda,B+i,ldb,shifts);
	status = cublasGemm(handle,trans,CUBLAS_OP_N,
			    m-(i+BSIZE),n,BSIZE,
			    &negOne,A+IDX(i,i+BSIZE,lda),lda,
			    B+i,ldb,&one,B+i+BSIZE,ldb);
	if(status != CUBLAS_STATUS_SUCCESS)
	  return status;
	i += BSIZE;
      }

      // Final subblock
      LUT_block <<< gridDim, m-i, 0, stream >>>
	(unitDiagonal,conjugate,m-i,n,A+IDX(i,i,lda),lda,B+i,ldb,shifts);

    }

    // LLT and LLC cases
    else if(side==CUBLAS_SIDE_LEFT
	    && uplo==CUBLAS_FILL_MODE_LOWER
	    && trans!=CUBLAS_OP_N) {

      // Current column in A
      int i = m-BSIZE;

      // Partition matrix into subblocks
      for(int b=numBlocks-1; b>0; --b) {
	LLT_block <<< gridDim, BSIZE, 0, stream >>>
	  (unitDiagonal,conjugate,BSIZE,n,A+IDX(i,i,lda),lda,B+i,ldb,shifts);
	status = cublasGemm(handle,trans,CUBLAS_OP_N,
			    i,n,BSIZE,&negOne,A+IDX(i,0,lda),lda,
			    B+i,ldb,&one,B,ldb);
	if(status != CUBLAS_STATUS_SUCCESS)
	  return status;
	i -= BSIZE;
      }

      // Final subblock
      LLT_block <<< gridDim, i+BSIZE, 0, stream >>>
	(unitDiagonal,conjugate,i+BSIZE,n,A,lda,B,ldb,shifts);

    }
    
    
    // Function has completed successfully
    return CUBLAS_STATUS_SUCCESS;

  }

  // -------------------------------------------
  // Cast complex types to Thrust complex types
  // -------------------------------------------
  template <>
  cublasStatus_t cudaMultiShiftTrsm<std::complex<float> >
  (cublasHandle_t handle,
   cublasSideMode_t side, cublasFillMode_t uplo,
   cublasOperation_t trans, cublasDiagType_t diag,
   int m, int n, const std::complex<float> * __restrict__ alpha,
   const std::complex<float> * __restrict__ A, int lda,
   std::complex<float> * __restrict__ B, int ldb,
   const std::complex<float> * __restrict__ shifts) {
    return cudaMultiShiftTrsm<thrust::complex<float> >
      (handle,side,uplo,trans,diag,m,n,(thrust::complex<float>*)alpha,
       (thrust::complex<float>*)A,lda,(thrust::complex<float>*)B,ldb,
       (thrust::complex<float>*)shifts);
  }
  template <>
  cublasStatus_t cudaMultiShiftTrsm<cuFloatComplex>
  (cublasHandle_t handle,
   cublasSideMode_t side, cublasFillMode_t uplo,
   cublasOperation_t trans, cublasDiagType_t diag,
   int m, int n, const cuFloatComplex * __restrict__ alpha,
   const cuFloatComplex * __restrict__ A, int lda,
   cuFloatComplex * __restrict__ B, int ldb,
   const cuFloatComplex * __restrict__ shifts) {
    return cudaMultiShiftTrsm<thrust::complex<float> >
      (handle,side,uplo,trans,diag,m,n,(thrust::complex<float>*)alpha,
       (thrust::complex<float>*)A,lda,(thrust::complex<float>*)B,ldb,
       (thrust::complex<float>*)shifts);
  }
  template <>
  cublasStatus_t cudaMultiShiftTrsm<std::complex<double> >
  (cublasHandle_t handle,
   cublasSideMode_t side, cublasFillMode_t uplo,
   cublasOperation_t trans, cublasDiagType_t diag,
   int m, int n, const std::complex<double> * __restrict__ alpha,
   const std::complex<double> * __restrict__ A, int lda,
   std::complex<double> * __restrict__ B, int ldb,
   const std::complex<double> * __restrict__ shifts) {
    return cudaMultiShiftTrsm<thrust::complex<double> >
      (handle,side,uplo,trans,diag,m,n,(thrust::complex<double>*)alpha,
       (thrust::complex<double>*)A,lda,(thrust::complex<double>*)B,ldb,
       (thrust::complex<double>*)shifts);
  }
  template <>
  cublasStatus_t cudaMultiShiftTrsm<cuDoubleComplex>
  (cublasHandle_t handle,
   cublasSideMode_t side, cublasFillMode_t uplo,
   cublasOperation_t trans, cublasDiagType_t diag,
   int m, int n, const cuDoubleComplex * __restrict__ alpha,
   const cuDoubleComplex * __restrict__ A, int lda,
   cuDoubleComplex * __restrict__ B, int ldb,
   const cuDoubleComplex * __restrict__ shifts) {
    return cudaMultiShiftTrsm<thrust::complex<double> >
      (handle,side,uplo,trans,diag,m,n,(thrust::complex<double>*)alpha,
       (thrust::complex<double>*)A,lda,(thrust::complex<double>*)B,ldb,
       (thrust::complex<double>*)shifts);
  }

  // -------------------------------------------
  // Explicit instantiation
  // -------------------------------------------
  template cublasStatus_t cudaMultiShiftTrsm<float>
  (cublasHandle_t handle,
   cublasSideMode_t side, cublasFillMode_t uplo,
   cublasOperation_t trans, cublasDiagType_t diag,
   int m, int n, const float * __restrict__ alpha,
   const float * __restrict__ A, int lda,
   float * __restrict__ B, int ldb,
   const float * __restrict__ shifts);
  template cublasStatus_t cudaMultiShiftTrsm<double>
  (cublasHandle_t handle,
   cublasSideMode_t side, cublasFillMode_t uplo,
   cublasOperation_t trans, cublasDiagType_t diag,
   int m, int n, const double * __restrict__ alpha,
   const double * __restrict__ A, int lda,
   double * __restrict__ B, int ldb,
   const double * __restrict__ shifts);
  template cublasStatus_t cudaMultiShiftTrsm<thrust::complex<float> >
  (cublasHandle_t handle,
   cublasSideMode_t side, cublasFillMode_t uplo,
   cublasOperation_t trans, cublasDiagType_t diag,
   int m, int n, const thrust::complex<float> * __restrict__ alpha,
   const thrust::complex<float> * __restrict__ A, int lda,
   thrust::complex<float> * __restrict__ B, int ldb,
   const thrust::complex<float> * __restrict__ shifts);
  template cublasStatus_t cudaMultiShiftTrsm<thrust::complex<double> >
  (cublasHandle_t handle,
   cublasSideMode_t side, cublasFillMode_t uplo,
   cublasOperation_t trans, cublasDiagType_t diag,
   int m, int n, const thrust::complex<double> * __restrict__ alpha,
   const thrust::complex<double> * __restrict__ A, int lda,
   thrust::complex<double> * __restrict__ B, int ldb,
   const thrust::complex<double> * __restrict__ shifts);

}
