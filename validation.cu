#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>
#include <cublas_v2.h>
extern "C"
{
#include <cblas.h>
}
#include "gpuTrsms.h"

#define IDX(i,j,ld) ((i)+(j)*(ld))

using namespace std;

// Function prototypes
int main(int argc, char **argv);
double randn();
void gaussianRandomMatrix(int m, float *A);
void choleskyRandomMatrix(int m, float *A);
extern "C" void spotrf_(char *UPLO, int *N, float *A, int *LDA, int *INFO);

/// Main function
int main(int argc, char **argv) {
  
  // -------------------------------------------------
  // Initialization
  // -------------------------------------------------

  // Default parameters
  int  m           = 4;
  int  n           = 1;
  bool sideLeft    = true;
  bool lower       = true;
  bool noTranspose = true;
  bool nonUnitDiag = true;
  bool verbose     = false;

  // User-provided parameters
  if(argc > 1) {
    m = atoi(argv[1]);
    n = atoi(argv[2]);
  }
  if(argc > 3) {
    sideLeft    = (bool) atoi(argv[3]);
    lower       = (bool) atoi(argv[4]);
    noTranspose = (bool) atoi(argv[5]);
    nonUnitDiag = (bool) atoi(argv[6]);
  }
  if(argc > 7)
    verbose = (bool) atoi(argv[7]);

  // Initialization
  timeval timeStart, timeEnd;

  // Report parameters
  printf("========================================\n");
  printf("  SHIFTED TRIANGULAR SOLVE VALIDATION\n");
  printf("========================================\n");
  printf("m = %d\n", m);
  printf("n = %d\n", n);
  printf("\n");
  printf("BLAS Options\n");
  printf("----------------------------------------\n");  
  printf("sideLeft    = %d\n", sideLeft);
  printf("lower       = %d\n", lower);
  printf("noTranspose = %d\n", noTranspose);
  printf("nonUnitDiag = %d\n", nonUnitDiag);

  // Initialize memory on host
  float *A = (float*) malloc(m*m*sizeof(float));
  float *B = (float*) malloc(m*n*sizeof(float));
  float *shifts = (float*) malloc(n*sizeof(float));
  float *X = (float*) malloc(m*n*sizeof(float));
  float *residual = (float*) malloc(m*n*sizeof(float));

  // Initialize matrices on host
  float alpha = randn();
  choleskyRandomMatrix(m,A);
#pragma omp parallel for
  for(int i=0;i<m*n;++i)
    B[i] = randn();
#pragma omp parallel for
  for(int i=0;i<n;++i) {
    shifts[i] = randn();
  }

  // Initialize memory on device
  float *cuda_A, *cuda_B, *cuda_B_cublas, *cuda_shifts;
  cudaMalloc(&cuda_A, m*m*sizeof(float));
  cudaMalloc(&cuda_B, m*n*sizeof(float));
  cudaMalloc(&cuda_B_cublas, m*n*sizeof(float));
  cudaMalloc(&cuda_shifts, n*sizeof(float));
  cudaMemcpy(cuda_A, A, m*m*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_B, B, m*n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_B_cublas, B, m*n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_shifts, shifts, n*sizeof(float), cudaMemcpyHostToDevice);

  // Initialize cuBLAS
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Initialize BLAS options (cuBLAS and CBLAS)
  CBLAS_SIDE      sideCblas  = CblasLeft;
  CBLAS_UPLO      uploCblas  = CblasLower;
  CBLAS_TRANSPOSE transCblas = CblasNoTrans;
  CBLAS_DIAG      diagCblas  = CblasNonUnit;
  cublasSideMode_t  side  = CUBLAS_SIDE_LEFT;
  cublasFillMode_t  uplo  = CUBLAS_FILL_MODE_LOWER;
  cublasOperation_t trans = CUBLAS_OP_N;
  cublasDiagType_t  diag  = CUBLAS_DIAG_NON_UNIT;
  if(!sideLeft) {
    side = CUBLAS_SIDE_RIGHT;
    sideCblas = CblasRight;
  }
  if(!lower) {
    uplo = CUBLAS_FILL_MODE_UPPER;
    uploCblas = CblasLower;
  }
  if(!noTranspose) {
    trans = CUBLAS_OP_T;
    transCblas = CblasTrans;
  }
  if(!nonUnitDiag) {
    diag = CUBLAS_DIAG_UNIT;
    diagCblas = CblasUnit;
  }

  // -------------------------------------------------
  // Test gpuStrsms
  // -------------------------------------------------

  // Solve triangular system
  cudaDeviceSynchronize();
  gettimeofday(&timeStart, NULL);
  gpuStrsms(handle, side, uplo, trans, diag, m, n,
	    &alpha, cuda_A, m, cuda_B, m, cuda_shifts);
  cudaDeviceSynchronize();
  gettimeofday(&timeEnd, NULL);
  double gpuStrsmsTime
    = timeEnd.tv_sec - timeStart.tv_sec
    + (timeEnd.tv_usec - timeStart.tv_usec)/1e6;

  // Transfer result to host
  cudaMemcpy(X, cuda_B, m*n*sizeof(float), 
	     cudaMemcpyDeviceToHost);

  // -------------------------------------------------
  // cuBLAS triangular solve
  //   For performance comparison
  // -------------------------------------------------

  // Solve triangular system
  cudaDeviceSynchronize();
  gettimeofday(&timeStart, NULL);
  cublasStrsm(handle,side,uplo,trans,diag,m,n,
	      &alpha,cuda_A,m,cuda_B_cublas,m);
  cudaDeviceSynchronize();
  gettimeofday(&timeEnd, NULL);
  double cublasTime
    = timeEnd.tv_sec - timeStart.tv_sec
    + (timeEnd.tv_usec - timeStart.tv_usec)/1e6;

  // -------------------------------------------------
  // Output results
  // -------------------------------------------------

  // Report time for matrix multiplication
  printf("\n");
  printf("Timings\n");
  printf("----------------------------------------\n");
  printf("  gpuStrsms : %g sec\n",gpuStrsmsTime);
  printf("  cuBLAS    : %g sec\n",cublasTime);

  // Report FLOPS
  double gflopCount = 1e-9*m*m*n; // Approximate
  printf("\n");
  printf("Performance\n");
  printf("----------------------------------------\n");
  printf("  gpuStrsms : %g GFLOPS\n", gflopCount/gpuStrsmsTime);
  printf("  cuBLAS    : %g GFLOPS\n", gflopCount/cublasTime);
  
  if(verbose) {
    // Print matrices
    printf("\n");
    printf("Matrix entries\n");
    printf("----------------------------------------\n");
    printf("  alpha = %g\n", alpha);
    printf("  shifts = [");
    for(int i=0;i<n;++i)
      printf("%g ", shifts[i]);
    printf("]\n");
    printf("  A =\n    [[");
    for(int i=0;i<m;++i) {
      for(int j=0;j<=i;++j)
	printf("%g ", A[IDX(i,j,m)]);
      for(int j=i+1;j<m;++j)
	printf("0 ");
      printf("]");
      if(i<m-1)
	printf("\n    [");
      else
	printf("]\n");
    }
    printf("  B =\n    [[");
    for(int i=0;i<m;++i) {
      for(int j=0;j<n;++j)
	printf("%g ", B[IDX(i,j,m)]);
      printf(" ]");
      if(i<m-1)
	printf("\n    [");
      else
	printf("]\n");
    }
    printf("  X =\n    [[");
    for(int i=0;i<m;++i) {
      for(int j=0;j<n;++j)
	printf("%g ", X[IDX(i,j,m)]);
      printf(" ]");
      if(i<m-1)
	printf("\n    [");
      else
	printf("]\n");
    }
  }

  // Check error in solution
  float normB = cblas_snrm2(m*n,B,1);
  memcpy(residual,X,m*n*sizeof(float));
  cblas_strmm(CblasColMajor,sideCblas,uploCblas,transCblas,diagCblas,
  	      m, n, 1., A, m, residual, m);
#pragma omp parallel for
  for(int i=0;i<n;++i)
    cblas_saxpy(m, shifts[i], X+i*m, 1, residual+i*m, 1);
  cblas_saxpy(m*n, -alpha, B, 1, residual, 1);
  float relResidual = cblas_snrm2(m*n,residual,1)/normB;
  printf("\n");
  printf("Relative error (Frobenius norm)\n");
  printf("----------------------------------------\n");
  printf("  gpuStrsms : %g\n", relResidual);

  // -------------------------------------------------
  // Clean up and finish
  // -------------------------------------------------
  free(A);
  free(B);
  free(shifts);
  free(X);
  free(residual);
  cudaFree(cuda_A);
  cudaFree(cuda_B);
  cudaFree(cuda_B_cublas);
  cudaFree(cuda_shifts);
  cublasDestroy(handle);
  return EXIT_SUCCESS;

}

/// Compute Gaussian random variable
/** Uses Box-Muller transform to convert uniform distribution to
 *  Gaussian distribution
 */
double randn() {
  double u1 = ((double)rand())/RAND_MAX;
  double u2 = ((double)rand())/RAND_MAX;
  return sqrt(-2*log(u1))*cos(2*M_PI*u2);
}

/// Generate matrix with Gaussian random variables
/** Diagonal entries are increased to improve conditioning. Viswanath
 *  and Trefethen (1998) find that the lower triangle of this matrix
 *  has a condition number on the order of 2^m.
 */
void gaussianRandomMatrix(int m, float *A) {

  // Each matrix entry is a Gaussian random variable
#pragma omp parallel for
  for(int i=0;i<m*m;++i)
    A[i] = randn();
  
  // Improve conditioning by increasing diagonal
#pragma omp parallel for
  for(int i=0;i<m;++i)
    A[IDX(i,i,m)] += 4.;

}

/// Generate matrix with Cholesky factorization of random matrix
void choleskyRandomMatrix(int m, float *A) {

  float *B = (float*) malloc(m*m*sizeof(float));

  // Generate matrix with Gaussian random variables
#pragma omp parallel for
  for(int i=0;i<m*m;++i)
    B[i] = randn();

  // Construct positive definite matrix
  cblas_ssyrk(CblasColMajor,CblasLower,CblasNoTrans,
	      m,m,1.,B,m,0.,A,m);

  // Perform Cholesky factorization
  char charL = 'L';
  int info;
  spotrf_(&charL, &m, A, &m, &info);

  // Clean up
  free(B);

}
