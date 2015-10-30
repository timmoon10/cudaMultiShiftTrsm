#include <iostream>
#include <complex>

#include <cuda.h>
#include <cublas_v2.h>
#include <thrust/complex.h>

#include "cudaMultiShiftTrsm.hpp"
#include "cudaHelper.hpp"
#include "cublasHelper.hpp"

#define BLOCK_SIZE 32

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
    /** Assumes blockDim=(32,1,1). Optimally, gridDim=(1,n,1),
     *  although it will suffice for gridDim to be one-dimensional in
     *  the y-dimension.
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
      const int tidx = threadIdx.x;
      int bidy = threadIdx.y + blockIdx.y*blockDim.y;

      // Private memory
      F private_A;
      F private_B;
      F private_shift;

      // Shared memory
      __shared__ F shared_B;

      // Each thread y-index corresponds to a RHS
      while(bidy < n) {

	// Transfer data from global to private memory
	if(tidx < m) {
	  private_B = B[IDX(tidx,bidy,ldb)];
	  private_shift = shifts[bidy];
	}

	// Iterate through columns of triangular matrix
	for(int i=0; i<m; ++i) {

	  // Transfer matrix column from global to private memory
	  if(unitDiagonal) {
	    // Matrix is unit triangular
	    if(tidx==i)
	      private_A = 1;
	    else if(i<tidx && tidx<m)
	      private_A = A[IDX(tidx,i,lda)];
	  }
	  else {
	    // Matrix is not unit triangular
	    if(i<=tidx && tidx<m)
	      private_A = A[IDX(tidx,i,lda)];
	  }

	  // Obtain solution at index i
	  __syncthreads();
	  if(tidx == i) {
	    private_B /= private_A + private_shift;
	    shared_B = private_B;
	  }
	  __syncthreads();

	  // Update RHS
	  if(i<tidx && tidx<m)
	    private_B -= private_A*shared_B;
	  
	}

	// Transfer solution from private to global memory
	if(tidx < m)
	  B[IDX(tidx,bidy,ldb)] = private_B;

	// Move to next RHS
	bidy += gridDim.y;

      }

    }

    /// Solve triangular systems with multiple shifts (LUN case)
    /** Assumes blockDim=(32,1,1). Optimally, gridDim=(1,n,1),
     *  although it will suffice for gridDim to be one-dimensional in
     *  the y-dimension.
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
      const int tidx = threadIdx.x;
      int bidy = threadIdx.y + blockIdx.y*blockDim.y;

      // Private memory
      F private_A;
      F private_B;
      F private_shift;

      // Shared memory
      __shared__ F shared_B;

      // Each thread y-index corresponds to a RHS
      while(bidy < n) {

	// Transfer data from global to private memory
	if(tidx < m) {
	  private_B = B[IDX(tidx,bidy,ldb)];
	  private_shift = shifts[bidy];
	}

	// Iterate through columns of triangular matrix
	for(int i=m-1; i>=0; --i) {

	  // Transfer matrix column from global to private memory
	  if(unitDiagonal) {
	    // Matrix is unit triangular
	    if(tidx==i)
	      private_A = 1;
	    else if(tidx<i)
	      private_A = A[IDX(tidx,i,lda)];
	  }
	  else {
	    // Matrix is not unit triangular
	    if(tidx<=i)
	      private_A = A[IDX(tidx,i,lda)];
	  }

	  // Obtain solution at index i
	  __syncthreads();
	  if(tidx == i) {
	    private_B /= private_A + private_shift;
	    shared_B = private_B;
	  }
	  __syncthreads();

	  // Update RHS
	  if(tidx < i)
	    private_B -= private_A*shared_B;
	  
	}

	// Transfer solution from private to global memory
	if(tidx < m)
	  B[IDX(tidx,bidy,ldb)] = private_B;

	// Move to next RHS
	bidy += gridDim.y;

      }

    }

    /// Solve triangular systems with multiple shifts (LUT, LUC case)
    /** Assumes blockDim=(32,1,1). Optimally, gridDim=(1,n,1),
     *  although it will suffice for gridDim to be one-dimensional in
     *  the y-dimension.
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

      // Initialize indices
      const int tidx = threadIdx.x;
      int bidy = threadIdx.y + blockIdx.y*blockDim.y;

      // Private memory
      F private_A;
      F private_B;
      F private_shift;

      // Shared memory
      __shared__ F shared_red[BLOCK_SIZE];

      // Each thread y-index corresponds to a RHS
      while(bidy < n) {

	// Transfer data from global to private memory
	if(tidx < m) {
	  private_B = B[IDX(tidx,bidy,ldb)];
	  private_shift = shifts[bidy];
	}

	// Iterate through columns of triangular matrix
	for(int i=0; i<m; ++i) {

	  // Transfer matrix column from global to private memory
	  if(unitDiagonal) {
	    // Matrix is unit triangular
	    if(tidx==i)
	      private_A = 1;
	    else if(tidx<i)
	      private_A = A[IDX(tidx,i,lda)];
	  }
	  else {
	    // Matrix is not unit triangular
	    if(tidx<=i)
	      private_A = A[IDX(tidx,i,lda)];
	  }

	  // Conjugate matrix if option is selected
	  if(conjugate)
	    private_A = conj(private_A);

	  // Obtain solution at index i
	  __syncthreads();
	  if(tidx<i)
	    shared_red[tidx] = private_A*private_B;
	  else
	    shared_red[tidx] = 0;
	  __syncthreads();
	  for(int b=BLOCK_SIZE/2; b>0; b/=2) {
	    if(tidx<b)
	      shared_red[tidx] += shared_red[tidx+b];
	    __syncthreads();
	  }
	  if(tidx==i) {
	    private_B -= shared_red[0];
	    private_B /= private_A+private_shift;
	  }
	  
	}

	// Transfer solution from private to global memory
	if(tidx < m)
	  B[IDX(tidx,bidy,ldb)] = private_B;

	// Move to next RHS
	bidy += gridDim.y;

      }

    }

    /// Solve triangular systems with multiple shifts (LLT, LLC case)
    /** Assumes blockDim=(32,1,1). Optimally, gridDim=(1,n,1),
     *  although it will suffice for gridDim to be one-dimensional in
     *  the y-dimension.
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

      // Initialize indices
      const int tidx = threadIdx.x;
      int bidy = threadIdx.y + blockIdx.y*blockDim.y;

      // Private memory
      F private_A;
      F private_B;
      F private_shift;

      // Shared memory
      __shared__ F shared_red[BLOCK_SIZE];

      // Each thread y-index corresponds to a RHS
      while(bidy < n) {

	// Transfer data from global to private memory
	if(tidx < m) {
	  private_B = B[IDX(tidx,bidy,ldb)];
	  private_shift = shifts[bidy];
	}

	// Iterate through columns of triangular matrix
	for(int i=m-1; i>=0; --i) {

	  // Transfer matrix column from global to private memory
	  if(unitDiagonal) {
	    // Matrix is unit triangular
	    if(tidx==i)
	      private_A = 1;
	    else if(i<tidx && tidx<m)
	      private_A = A[IDX(tidx,i,lda)];
	  }
	  else {
	    // Matrix is not unit triangular
	    if(i<=tidx && tidx<m)
	      private_A = A[IDX(tidx,i,lda)];
	  }

	  // Conjugate matrix if option is selected
	  if(conjugate)
	    private_A = conj(private_A);

	  // Obtain solution at index i
	  __syncthreads();
	  if(i<tidx && tidx<m)
	    shared_red[tidx] = private_A*private_B;
	  else
	    shared_red[tidx] = 0;
	  __syncthreads();
	  for(int b=BLOCK_SIZE/2; b>0; b/=2) {
	    if(tidx<b)
	      shared_red[tidx] += shared_red[tidx+b];
	    __syncthreads();
	  }
	  if(tidx==i) {
	    private_B -= shared_red[0];
	    private_B /= private_A+private_shift;
	  }
	  
	}

	// Transfer solution from private to global memory
	if(tidx < m)
	  B[IDX(tidx,bidy,ldb)] = private_B;

	// Move to next RHS
	bidy += gridDim.y;

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

    // Initialize CUDA grid dimensions
    dim3 blockDim, gridDim;
    blockDim.x = BLOCK_SIZE;
    blockDim.y = 1;
    blockDim.z = 1;
    gridDim.x  = 1;
    gridDim.y  = min(n, prop.maxGridSize[1]);
    gridDim.z  = 1;

    // Number matrix subblocks
    int nb = (m+BLOCK_SIZE-1)/BLOCK_SIZE;

    // LLN case
    if(side==CUBLAS_SIDE_LEFT
       && uplo==CUBLAS_FILL_MODE_LOWER
       && trans==CUBLAS_OP_N) {
      
      // Current row in A
      int i = 0;

      // Partition matrix into subblocks
      for(int b=0; b<nb-1; ++b) {
	LLN_block <<< gridDim, blockDim, 0, stream >>>
	  (unitDiagonal,BLOCK_SIZE,n,A+IDX(i,i,lda),lda,B+i,ldb,shifts);
	status = cublasGemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
			    m-(i+BLOCK_SIZE),n,BLOCK_SIZE,
			    &negOne,A+IDX(i+BLOCK_SIZE,i,lda),lda,
			    B+i,ldb,&one,B+i+BLOCK_SIZE,ldb);
	if(status != CUBLAS_STATUS_SUCCESS)
	  return status;
	i += BLOCK_SIZE;
      }

      // Final subblock
      LLN_block <<< gridDim, blockDim, 0, stream >>>
	(unitDiagonal,m-i,n,A+IDX(i,i,lda),lda,B+i,ldb,shifts);

    }

    // LUN case
    else if(side==CUBLAS_SIDE_LEFT
	    && uplo==CUBLAS_FILL_MODE_UPPER
	    && trans==CUBLAS_OP_N) {

      // Current row in A
      int i = m-BLOCK_SIZE;

      // Partition matrix into subblocks
      for(int b=nb-1; b>0; --b) {
	LUN_block <<< gridDim, blockDim, 0, stream >>>
	  (unitDiagonal,BLOCK_SIZE,n,A+IDX(i,i,lda),lda,B+i,ldb,shifts);
	status = cublasGemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
			    i,n,BLOCK_SIZE,&negOne,A+IDX(0,i,lda),lda,
			    B+i,ldb,&one,B,ldb);
	if(status != CUBLAS_STATUS_SUCCESS)
	  return status;
	i -= BLOCK_SIZE;
      }

      // Final subblock
      LUN_block <<< gridDim, blockDim, 0, stream >>>
	(unitDiagonal,i+BLOCK_SIZE,n,A,lda,B,ldb,shifts);

    }

    // LUT and LUC cases
    else if(side==CUBLAS_SIDE_LEFT
	    && uplo==CUBLAS_FILL_MODE_UPPER
	    && trans!=CUBLAS_OP_N) {

      // Current column in A
      int i = 0;

      // Partition matrix into subblocks
      for(int b=0; b<nb-1; ++b) {
	LUT_block <<< gridDim, BLOCK_SIZE, 0, stream >>>
	  (unitDiagonal,conjugate,BLOCK_SIZE,n,
	   A+IDX(i,i,lda),lda,B+IDX(i,0,ldb),ldb,shifts);
	status = cublasGemm(handle,trans,CUBLAS_OP_N,
			    m-(i+BLOCK_SIZE),n,BLOCK_SIZE,
			    &negOne,A+IDX(i,i+BLOCK_SIZE,lda),lda,
			    B+IDX(i,0,ldb),ldb,
			    &one,B+IDX(i+BLOCK_SIZE,0,ldb),ldb);
	if(status != CUBLAS_STATUS_SUCCESS)
	  return status;
	i += BLOCK_SIZE;
      }

      // Final subblock
      LUT_block <<< gridDim, blockDim, 0, stream >>>
	(unitDiagonal,conjugate,m-i,n,
	 A+IDX(i,i,lda),lda,B+IDX(i,0,ldb),ldb,shifts);

    }

    // LLT and LLC cases
    else if(side==CUBLAS_SIDE_LEFT
	    && uplo==CUBLAS_FILL_MODE_LOWER
	    && trans!=CUBLAS_OP_N) {

      // Current column in A
      int i = m-BLOCK_SIZE;

      // Partition matrix into subblocks
      for(int b=nb-1; b>0; --b) {
	LLT_block <<< gridDim, BLOCK_SIZE, 0, stream >>>
	  (unitDiagonal,conjugate,BLOCK_SIZE,n,
	   A+IDX(i,i,lda),lda,B+i,ldb,shifts);
	status = cublasGemm(handle,trans,CUBLAS_OP_N,
			    i,n,BLOCK_SIZE,&negOne,A+IDX(i,0,lda),lda,
			    B+i,ldb,&one,B,ldb);
	if(status != CUBLAS_STATUS_SUCCESS)
	  return status;
	i -= BLOCK_SIZE;
      }

      // Final subblock
      LLT_block <<< gridDim, blockDim, 0, stream >>>
	(unitDiagonal,conjugate,i+BLOCK_SIZE,n,A,lda,B,ldb,shifts);

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
