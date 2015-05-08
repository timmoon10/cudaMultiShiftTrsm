#include "gpuTrsms.h"

#define BSIZE 32 // CUDA warp size
#define IDX(i,j,ld) ((i)+(j)*(ld))

using namespace std;

// Function prototypes
__global__
void strsmsBlock(const bool diag,
		 int m, int n,
		 const float * __restrict__ A, int lda,
		 float * __restrict__ B, int ldb,
		 const float * __restrict__ shifts);

/// Solve a shifted triangular matrix equation
cublasStatus_t gpuStrsms(cublasHandle_t handle,
			 cublasSideMode_t side, cublasFillMode_t uplo,
			 cublasOperation_t trans, cublasDiagType_t diag,
			 const int m, const int n,
			 const float * __restrict__ alpha,
			 const float * __restrict__ A, const int lda,
			 float * __restrict__ B, const int ldb,
			 const float * __restrict__ shifts) {

  // Initialize CUDA and cuBLAS objects
  cublasStatus_t status;
  cudaStream_t stream;
  status = cublasGetStream(handle, &stream);
  if(status != CUBLAS_STATUS_SUCCESS)
    return status;

  // Initialize parameters
  bool sideLeft    = (side  == CUBLAS_SIDE_LEFT);
  bool lower       = (uplo  == CUBLAS_FILL_MODE_LOWER);
  bool noTranspose = (trans == CUBLAS_OP_N);
  bool nonUnitDiag = (diag  == CUBLAS_DIAG_NON_UNIT);

  // Misc initialization
  float one = 1.f;
  float negOne = -1.f;

  // Report invalid parameters
  if(!sideLeft && side!=CUBLAS_SIDE_RIGHT) {
    printf("Error (gpuStrsms): argument 2 is invalid\n");
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  if(!lower && uplo!=CUBLAS_FILL_MODE_UPPER) {
    printf("Error (gpuStrsms): argument 3 is invalid\n");
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  if(!noTranspose && trans!=CUBLAS_OP_T && trans!=CUBLAS_OP_C) {
    printf("Error (gpuStrsms): argument 4 is invalid\n");
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  if(!nonUnitDiag && diag!=CUBLAS_DIAG_UNIT) {
    printf("Error (gpuStrsms): argument 5 is invalid\n");
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  if(m < 0) {
    printf("Error (gpuStrsms): argument 6 is invalid (m<0)\n");
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  if(n < 0) {
    printf("Error (gpuStrsms): argument 7 is invalid (n<0)\n");
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  if(lda < m) {
    printf("Error (gpuStrsms): argument 10 is invalid (lda<m)\n");
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  if(ldb < m) {
    printf("Error (gpuStrsms): argument 12 is invalid (lda<m)\n");
    return CUBLAS_STATUS_INVALID_VALUE;
  }

  // Display error and exit if an unimplemented feature is called
  // TODO: remove this section when possible
  if(side != CUBLAS_SIDE_LEFT) {
    fprintf(stderr,
	    "ERROR in gpuStrsms: invalid input in argument 2\n"
	    "  side=CUBLAS_SIDE_RIGHT is not yet implemented\n");
    exit(EXIT_FAILURE);
  }
  if(uplo != CUBLAS_FILL_MODE_LOWER) {
    fprintf(stderr,
	    "ERROR in gpuStrsms: invalid input in argument 3\n"
	    "  uplo=CUBLAS_FILL_MODE_UPPER is not yet implemented\n");
    exit(EXIT_FAILURE);
  }
  if(trans != CUBLAS_OP_N) {
    fprintf(stderr,
	    "ERROR in gpuStrsms: invalid input in argument 4\n"
	    "  trans=CUBLAS_OP_T and trans=CUBLAS_OP_C are not yet implemented\n");
    exit(EXIT_FAILURE);
  }

  // Return zero if right hand side is zero
  if(alpha == 0) {
    float zero = 0.f;
    for(int i=0; i<n; ++i) {
      status = cublasSscal(handle, m, &zero, B+i*ldb, 1);
      if(status != CUBLAS_STATUS_SUCCESS)
	return status;
    }
    return CUBLAS_STATUS_SUCCESS;
  }

  // Scale right hand side
  for(int i=0; i<n; ++i) {
    status = cublasSscal(handle, m, alpha, B+i*ldb, 1);
    if(status != CUBLAS_STATUS_SUCCESS)
      return status;
  }

  // Perform blocked triangular solve
  int numBlocks = (m+BSIZE-1)/BSIZE;  // Number of subblocks in A
  int i = 0;                          // Current row in A
  for(int b=0; b<numBlocks-1; ++b) {
    strsmsBlock <<< n, BSIZE, 0, stream >>>
      (nonUnitDiag, BSIZE,n,A+i+i*lda,lda,B+i,ldb,shifts);
    status = cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
			 m-(i+BSIZE),n,BSIZE,
			 &negOne, A+i+BSIZE+i*lda, lda,
			 B+i,ldb, &one, B+i+BSIZE, ldb);
    if(status != CUBLAS_STATUS_SUCCESS)
      return status;
    i += BSIZE;
  }
  strsmsBlock <<< n, BSIZE, 0, stream >>>
    (nonUnitDiag, m-i,n,A+i+i*lda,lda,B+i,ldb,shifts);

  // Function has completed successfully
  return CUBLAS_STATUS_SUCCESS;
  
}

/// Solve a shifted triangular matrix equation
__global__
void strsmsBlock(const bool diag,
		 const int m, const int n,
		 const float * __restrict__ A, const int lda,
		 float * __restrict__ B, const int ldb,
		 const float * __restrict__ shifts) {

  // Initialize indices
  int tid = threadIdx.x;

  // Copy global memory to shared memory
  __shared__ float shared_B[BSIZE];
  __shared__ float shared_shift;
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
      float private_A = A[IDX(tid,i,lda)];

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
