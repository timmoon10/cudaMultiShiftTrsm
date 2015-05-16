#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <complex>
#include <cuda.h>
#include <cublas_v2.h>
extern "C"
{
#include <cblas.h>
}
#include "cudaMultiShiftTrsm.hpp"

#define datafloat float
#define IDX(i,j,ld) ((i)+(j)*(ld))

using namespace std;

// ===============================================
// BLAS and LAPACK routines
// ===============================================
extern "C" 
{
  float snrm2_(int *n, void *x, int *incx);
  double dnrm2_(int *n, void *x, int *incx);
  float scnrm2_(int *n, void *x, int *incx);
  double dznrm2_(int *n, void *x, int *incx);
  void ssyrk_(char *uplo, char *trans, int *n, int *k,
	      void *alpha, void *A, int *lda,
	      void *beta, void *C, int *ldc);
  void dsyrk_(char *uplo, char *trans, int *n, int *k,
	      void *alpha, void *A, int *lda,
	      void *beta, void *C, int *ldc);
  void csyrk_(char *uplo, char *trans, int *n, int *k,
	      void *alpha, void *A, int *lda,
	      void *beta, void *C, int *ldc);
  void zsyrk_(char *uplo, char *trans, int *n, int *k,
	      void *alpha, void *A, int *lda,
	      void *beta, void *C, int *ldc);
  void spotrf_(char *uplo, int *n, void *A, int *lda, int *info);
  void dpotrf_(char *uplo, int *n, void *A, int *lda, int *info);
  void cpotrf_(char *uplo, int *n, void *A, int *lda, int *info);
  void zpotrf_(char *uplo, int *n, void *A, int *lda, int *info);
}
template <typename F> inline
double nrm2(int n, F * x, int incx);
template <> inline
double nrm2<float>(int n, float * x, int incx) {
  return snrm2_(&n,x,&incx);
}
template <> inline
double nrm2<double>(int n, double * x, int incx) {
  return dnrm2_(&n,x,&incx);
}
template <> inline
double nrm2<complex<float> >(int n, complex<float> * x, int incx) {
  return scnrm2_(&n,x,&incx);
}
template <> inline
double nrm2<complex<double> >(int n, complex<double> * x, int incx) {
  return dznrm2_(&n,x,&incx);
}

template <typename F> inline
void syrk(char uplo, char trans, int n, int k,
	  F alpha, F * A, int lda,
	  F beta, F * C, int ldc);
template <> inline
void syrk<float>(char uplo, char trans, int n, int k,
		 float alpha, float * A, int lda,
		 float beta, float * C, int ldc) {
  ssyrk_(&uplo,&trans,&n,&k,&alpha,A,&lda,&beta,C,&ldc);
}
template <> inline
void syrk<double>(char uplo, char trans, int n, int k,
		  double alpha, double * A, int lda,
		  double beta, double * C, int ldc) {
  dsyrk_(&uplo,&trans,&n,&k,&alpha,A,&lda,&beta,C,&ldc);
}
template <> inline
void syrk<complex<float> >(char uplo, char trans, int n, int k,
			  complex<float> alpha,
			  complex<float> * A, int lda,
			  complex<float> beta,
			  complex<float> * C, int ldc) {
  csyrk_(&uplo,&trans,&n,&k,&alpha,A,&lda,&beta,C,&ldc);
}
template <> inline
void syrk<complex<double> >(char uplo, char trans, int n, int k,
			   complex<double> alpha,
			   complex<double> * A, int lda,
			   complex<double> beta,
			   complex<double> * C, int ldc) {
  csyrk_(&uplo,&trans,&n,&k,&alpha,A,&lda,&beta,C,&ldc);
}
template <typename F> inline
void potrf(char uplo, int n, F * A, int lda, int &info);
template <> inline
void potrf<float>(char uplo, int n, float * A, int lda, int &info) {
  spotrf_(&uplo,&n,A,&lda,&info);
}
template <> inline
void potrf<double>(char uplo, int n, double * A, int lda, int &info) {
  dpotrf_(&uplo,&n,A,&lda,&info);
}
template <> inline
void potrf<complex<float> >(char uplo, int n, complex<float> * A, int lda, int &info) {
  cpotrf_(&uplo,&n,A,&lda,&info);
}
template <> inline
void potrf<complex<double> >(char uplo, int n, complex<double> * A, int lda, int &info) {
  zpotrf_(&uplo,&n,A,&lda,&info);
}

// ===============================================
// Random matrix generation
// ===============================================

/// Compute Gaussian random variable
/** Uses Box-Muller transform to convert uniform distribution to
 *  Gaussian distribution
 */
template <typename F>
F randn() {
  F u1 = ((F)rand())/RAND_MAX;
  F u2 = ((F)rand())/RAND_MAX;
  return sqrt(-2*log(u1))*cos(2*M_PI*u2);
}
template <template<typename> class complex, typename T>
complex<T> randn() {
  T u1 = ((T)rand())/RAND_MAX;
  T u2 = ((T)rand())/RAND_MAX;
  return complex<T>(sqrt(-2*log(u1))*cos(2*M_PI*u2),
		    sqrt(-2*log(u1))*sin(2*M_PI*u2));
}

/// Generate matrix with Gaussian random variables
/** Diagonal entries are increased to improve conditioning. Viswanath
 *  and Trefethen (1998) find that the lower triangle of this matrix
 *  has a condition number on the order of 2^m.
 */
template <typename F>
void gaussianRandomMatrix(int m, F *A) {
#pragma omp parallel for
  for(int i=0;i<m*m;++i)
    A[i] = randn<F>();
}

/// Generate matrix with Cholesky factorization of random matrix
template <typename F>
void choleskyRandomMatrix(int m, F *A) {

  F *B = (F*) malloc(m*m*sizeof(F));

  // Generate matrix with Gaussian random variables
  gaussianRandomMatrix<F>(m,B);

  // Construct positive definite matrix
  syrk<F>('L','N',m,m,1,B,m,0,A,m);

  // Perform Cholesky factorization
  int info;
  potrf<F>('L', m, A, m, info);

  // Clean up
  free(B);

}

// ===============================================
// Validation program
// ===============================================

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
  datafloat *A = (datafloat*) malloc(m*m*sizeof(datafloat));
  datafloat *B = (datafloat*) malloc(m*n*sizeof(datafloat));
  datafloat *shifts = (datafloat*) malloc(n*sizeof(datafloat));
  datafloat *X = (datafloat*) malloc(m*n*sizeof(datafloat));
  datafloat *residual = (datafloat*) malloc(m*n*sizeof(datafloat));

  // Initialize matrices on host
  datafloat alpha = randn<datafloat>();
  choleskyRandomMatrix<datafloat>(m,A);
#pragma omp parallel for
  for(int i=0;i<m*n;++i)
    B[i] = randn<datafloat>();
#pragma omp parallel for
  for(int i=0;i<n;++i) {
    shifts[i] = randn<datafloat>();
  }

  // Initialize memory on device
  datafloat *cuda_A, *cuda_B, *cuda_B_cublas, *cuda_shifts;
  cudaMalloc(&cuda_A, m*m*sizeof(datafloat));
  cudaMalloc(&cuda_B, m*n*sizeof(datafloat));
  cudaMalloc(&cuda_B_cublas, m*n*sizeof(datafloat));
  cudaMalloc(&cuda_shifts, n*sizeof(datafloat));
  cudaMemcpy(cuda_A, A, m*m*sizeof(datafloat), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_B, B, m*n*sizeof(datafloat), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_B_cublas, B, m*n*sizeof(datafloat), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_shifts, shifts, n*sizeof(datafloat), cudaMemcpyHostToDevice);

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
  // Test cudaMultiShiftTrsm
  // -------------------------------------------------

  // Solve triangular system
  cudaDeviceSynchronize();
  gettimeofday(&timeStart, NULL);
  cudaMstrsm::cudaMultiShiftTrsm<datafloat>(handle, side, uplo, trans, diag, m, n,
					    &alpha, cuda_A, m, cuda_B, m, cuda_shifts);
  cudaDeviceSynchronize();
  gettimeofday(&timeEnd, NULL);
  double cudaMstrsmTime
    = timeEnd.tv_sec - timeStart.tv_sec
    + (timeEnd.tv_usec - timeStart.tv_usec)/1e6;

  // Transfer result to host
  cudaMemcpy(X, cuda_B, m*n*sizeof(datafloat), 
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
  printf("  cudaMstrsm : %g sec\n",cudaMstrsmTime);
  printf("  cuBLAS     : %g sec\n",cublasTime);

  // Report FLOPS
  double gflopCount = 1e-9*m*m*n; // Approximate
  printf("\n");
  printf("Performance\n");
  printf("----------------------------------------\n");
  printf("  cudaMstrsm : %g GFLOPS\n", gflopCount/cudaMstrsmTime);
  printf("  cuBLAS     : %g GFLOPS\n", gflopCount/cublasTime);
  
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
  double normB = nrm2<datafloat>(m*n,B,1);
  memcpy(residual,X,m*n*sizeof(datafloat));
  cblas_strmm(CblasColMajor,sideCblas,uploCblas,transCblas,diagCblas,
  	      m, n, 1., A, m, residual, m);
#pragma omp parallel for
  for(int i=0;i<n;++i)
    cblas_saxpy(m, shifts[i], X+i*m, 1, residual+i*m, 1);
  cblas_saxpy(m*n, -alpha, B, 1, residual, 1);
  double relResidual = nrm2<datafloat>(m*n,residual,1)/normB;
  printf("\n");
  printf("Relative error (Frobenius norm)\n");
  printf("----------------------------------------\n");
  printf("  cudaMstrsm : %g\n", relResidual);

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
