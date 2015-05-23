#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <sys/time.h>
#include <complex>
#include <cuda.h>
#include <cublas_v2.h>
#include "cublasHelper.hpp"
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
  void saxpy_(int *n, void *a, void *x, int *incx, void *y, int *incy);
  void daxpy_(int *n, void *a, void *x, int *incx, void *y, int *incy);
  void caxpy_(int *n, void *a, void *x, int *incx, void *y, int *incy);
  void zaxpy_(int *n, void *a, void *x, int *incx, void *y, int *incy);
  void ssyrk_(char *uplo, char *trans, int *n, int *k,
	      void *alpha, void *A, int *lda,
	      void *beta, void *C, int *ldc);
  void dsyrk_(char *uplo, char *trans, int *n, int *k,
	      void *alpha, void *A, int *lda,
	      void *beta, void *C, int *ldc);
  void cherk_(char *uplo, char *trans, int *n, int *k,
	      void *alpha, void *A, int *lda,
	      void *beta, void *C, int *ldc);
  void zherk_(char *uplo, char *trans, int *n, int *k,
	      void *alpha, void *A, int *lda,
	      void *beta, void *C, int *ldc);
  void strmm_(char *side, char *uplo, char *transa, char *diag, 
	      int *n, int *m, void *alpha, void *A, int *lda,
	      void *B, int *ldb);
  void dtrmm_(char *side, char *uplo, char *transa, char *diag, 
	      int *n, int *m, void *alpha, void *A, int *lda,
	      void *B, int *ldb);
  void ctrmm_(char *side, char *uplo, char *transa, char *diag, 
	      int *n, int *m, void *alpha, void *A, int *lda,
	      void *B, int *ldb);
  void ztrmm_(char *side, char *uplo, char *transa, char *diag, 
	      int *n, int *m, void *alpha, void *A, int *lda,
	      void *B, int *ldb);
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
void axpy(int n, F a, F * x, int incx, F * y, int incy);
template <> inline
void axpy<float>(int n, float a, float * x, int incx,
		 float * y, int incy) {
  saxpy_(&n,&a,x,&incx,y,&incy);
}
template <> inline
void axpy<double>(int n, double a, double * x, int incx,
		  double * y, int incy) {
  daxpy_(&n,&a,x,&incx,y,&incy);
}
template <> inline
void axpy<complex<float> >(int n, complex<float> a,
			   complex<float> * x, int incx,
			   complex<float> * y, int incy) {
  caxpy_(&n,&a,x,&incx,y,&incy);
}
template <> inline
void axpy<complex<double> >(int n, complex<double> a,
			    complex<double> * x, int incx,
			    complex<double> * y, int incy) {
  zaxpy_(&n,&a,x,&incx,y,&incy);
}
template <typename F> inline
void herk(char uplo, char trans, int n, int k,
	  F alpha, F * A, int lda,
	  F beta, F * C, int ldc);
template <> inline
void herk<float>(char uplo, char trans, int n, int k,
		 float alpha, float * A, int lda,
		 float beta, float * C, int ldc) {
  ssyrk_(&uplo,&trans,&n,&k,&alpha,A,&lda,&beta,C,&ldc);
}
template <> inline
void herk<double>(char uplo, char trans, int n, int k,
		  double alpha, double * A, int lda,
		  double beta, double * C, int ldc) {
  dsyrk_(&uplo,&trans,&n,&k,&alpha,A,&lda,&beta,C,&ldc);
}
template <> inline
void herk<complex<float> >(char uplo, char trans, int n, int k,
			   complex<float> alpha,
			   complex<float> * A, int lda,
			   complex<float> beta,
			   complex<float> * C, int ldc) {
  cherk_(&uplo,&trans,&n,&k,&alpha,A,&lda,&beta,C,&ldc);
}
template <> inline
void herk<complex<double> >(char uplo, char trans, int n, int k,
			    complex<double> alpha,
			    complex<double> * A, int lda,
			    complex<double> beta,
			    complex<double> * C, int ldc) {
  zherk_(&uplo,&trans,&n,&k,&alpha,A,&lda,&beta,C,&ldc);
}
template <typename F> inline
void trmm(char side, char uplo, char transa, char diag, 
	  int n, int m, F alpha, F * A, int lda,
	  F * B, int ldb);
template <> inline
void trmm<float>(char side, char uplo, char transa, char diag, 
		 int n, int m, float alpha, float * A, int lda,
		 float * B, int ldb) {
  strmm_(&side, &uplo, &transa, &diag, &n, &m, &alpha, A, &lda, B, &ldb);
}
template <> inline
void trmm<double>(char side, char uplo, char transa, char diag, 
		  int n, int m, double alpha, double * A, int lda,
		  double * B, int ldb) {
  dtrmm_(&side, &uplo, &transa, &diag, &n, &m, &alpha, A, &lda, B, &ldb);
}
template <> inline
void trmm<complex<float> >(char side, char uplo, char transa, char diag, 
			   int n, int m, complex<float> alpha,
			   complex<float> * A, int lda,
			   complex<float> * B, int ldb) {
  ctrmm_(&side, &uplo, &transa, &diag, &n, &m, &alpha, A, &lda, B, &ldb);
}
template <> inline
void trmm<complex<double> >(char side, char uplo, char transa, char diag, 
			    int n, int m, complex<double> alpha,
			    complex<double> * A, int lda,
			    complex<double> * B, int ldb) {
  ztrmm_(&side, &uplo, &transa, &diag, &n, &m, &alpha, A, &lda, B, &ldb);
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
void potrf<complex<float> >(char uplo, int n,
			    complex<float> * A, int lda, int &info) {
  cpotrf_(&uplo,&n,A,&lda,&info);
}
template <> inline
void potrf<complex<double> >(char uplo, int n,
			     complex<double> * A, int lda, int &info) {
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
  F u1 = ((F)rand()+1)/(RAND_MAX+2);
  F u2 = ((F)rand()+1)/(RAND_MAX+2);
  return sqrt(-2*log(u1))*cos(2*M_PI*u2);
}
template <>
complex<float> randn<complex<float> >() {
  float u1 = ((float)rand()+1)/(RAND_MAX+2);
  float u2 = ((float)rand()+1)/(RAND_MAX+2);
  return complex<float>(sqrt(-2*log(u1))*cos(2*M_PI*u2),
			sqrt(-2*log(u1))*sin(2*M_PI*u2));
}
template <>
complex<double> randn<complex<double> >() {
  double u1 = ((double)rand()+1)/(RAND_MAX+2);
  double u2 = ((double)rand()+1)/(RAND_MAX+2);
  return complex<double>(sqrt(-2*log(u1))*cos(2*M_PI*u2),
			 sqrt(-2*log(u1))*sin(2*M_PI*u2));
}

/// Generate matrix with Gaussian random variables
/** Diagonal entries are increased to improve conditioning. Viswanath
 *  and Trefethen (1998) find that the lower triangle of this matrix
 *  has a condition number on the order of 2^m.
 */
template <typename F>
void randn(int n, F *A) {
#pragma omp parallel for
  for(int i=0;i<n;++i)
    A[i] = randn<F>();
}

/// Generate matrix with Cholesky factorization of random matrix
template <typename F>
void choleskyRandomMatrix(char uplo, int m, F *A) {

  // Generate matrix with Gaussian random variables
  F *temp = (F*) malloc(m*m*sizeof(F));
  randn<F>(m*m,temp);

  // Construct positive definite matrix
  herk<F>(uplo,'N',m,m,1,temp,m,0,A,m);

  // Shift diagonal to improve condition number
  for(int i=0;i<m;++i)
    A[IDX(i,i,m)] += sqrt(m);

  // Perform Cholesky factorization
  int info;
  potrf<F>(uplo, m, A, m, info);

  // Clean up
  free(temp);

}

/// Output matrix entries to stream
template <typename F>
void printMatrix(ostream & os,
		 const char uplo, const char diag,
		 const int m, const int n, 
		 const F * A, const int lda) {

  os << "    [[";
  for(int i=0;i<m;++i) {
    if(toupper(uplo) == 'L') {
      for(int j=0;j<i;++j)
	os << A[IDX(i,j,lda)] << " ";
      if(toupper(diag)=='U')
	os << "1 ";
      else
	os << A[IDX(i,i,lda)] << " ";
      for(int j=i+1;j<n;++j)
	os << "0 ";
    }
    else if(toupper(uplo) == 'U') {
      for(int j=0;j<i;++j)
	os << "0 ";
      if(toupper(diag)=='U')
	os << "1 ";
      else
	os << A[IDX(i,i,lda)] << " ";
      for(int j=i+1;j<n;++j)
	os << A[IDX(i,j,lda)] << " ";
    }
    else {
      for(int j=0;j<n;++j)
	os << A[IDX(i,j,lda)] << " ";
    }
    os << "]";
    if(i<m-1)
      os << std::endl << "    [";
    else
      os << "]" << std::endl;
  }
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
  int  m       = 4;
  int  n       = 1;
  char side    = 'L';
  char uplo    = 'L';
  char trans   = 'N';
  char diag    = 'N';
  bool verbose = false;

  // User-provided parameters
  if(argc > 1)
    m = atoi(argv[1]);
  if(argc > 2)
    n = atoi(argv[2]);
  if(argc > 3)
    side = toupper(argv[3][0]);
  if(argc > 4)
    uplo = toupper(argv[4][0]);
  if(argc > 5)
    trans = toupper(argv[5][0]);
  if(argc > 6)
    diag = toupper(argv[6][0]);
  if(argc > 7)
    verbose = atoi(argv[7]);

  // Initialization
  timeval timeStart, timeEnd;

  // Report parameters
  std::cout << "========================================" << std::endl
	    << "  SHIFTED TRIANGULAR SOLVE VALIDATION" << std::endl
	    << "========================================" << std::endl
	    << "m = " << m << std::endl
	    << "n = " << n << std::endl
	    << std::endl
	    << "BLAS Options" << std::endl
	    << "----------------------------------------" << std::endl
	    << "side  = " << side << std::endl
	    << "uplo  = " << uplo << std::endl
	    << "trans = " << trans << std::endl
	    << "diag  = " << diag << std::endl;

  // Initialize memory on host
  datafloat *A = (datafloat*) malloc(m*m*sizeof(datafloat));
  datafloat *B = (datafloat*) malloc(m*n*sizeof(datafloat));
  datafloat *shifts = (datafloat*) malloc(n*sizeof(datafloat));
  datafloat *X = (datafloat*) malloc(m*n*sizeof(datafloat));
  datafloat *residual = (datafloat*) malloc(m*n*sizeof(datafloat));

  // Initialize matrices on host
  datafloat alpha = randn<datafloat>();
  choleskyRandomMatrix<datafloat>(uplo,m,A);
  randn<datafloat>(m*n,B);
  randn<datafloat>(n,shifts);

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
  cublasStatus_t status;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSideMode_t  cublasSide  = CUBLAS_SIDE_LEFT;
  cublasFillMode_t  cublasUplo  = CUBLAS_FILL_MODE_LOWER;
  cublasOperation_t cublasTrans = CUBLAS_OP_N;
  cublasDiagType_t  cublasDiag  = CUBLAS_DIAG_NON_UNIT;
  if(side=='R')
    cublasSide = CUBLAS_SIDE_RIGHT;
  if(uplo=='U')
    cublasUplo = CUBLAS_FILL_MODE_UPPER;
  if(trans=='T')
    cublasTrans = CUBLAS_OP_T;
  else if(trans=='C') 
    cublasTrans = CUBLAS_OP_C;
  if(diag=='U')
    cublasDiag = CUBLAS_DIAG_UNIT;

  // -------------------------------------------------
  // Test cudaMultiShiftTrsm
  // -------------------------------------------------

  // Solve triangular system
  cudaDeviceSynchronize();
  gettimeofday(&timeStart, NULL);
  status = cudaMstrsm::cudaMultiShiftTrsm<datafloat>
    (handle, cublasSide, cublasUplo, cublasTrans, cublasDiag,
     m, n, &alpha, cuda_A, m, cuda_B, m, cuda_shifts);
  cudaDeviceSynchronize();
  gettimeofday(&timeEnd, NULL);
  double cudaMstrsmTime
    = timeEnd.tv_sec - timeStart.tv_sec
    + (timeEnd.tv_usec - timeStart.tv_usec)/1e6;
  if(status != CUBLAS_STATUS_SUCCESS)
    std::cout << "cudaMstrsm FAILED" << std::endl;

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
  status = cublasTrsm(handle,cublasSide,cublasUplo,cublasTrans,cublasDiag,
		      m,n,&alpha,cuda_A,m,cuda_B_cublas,m);
  cudaDeviceSynchronize();
  gettimeofday(&timeEnd, NULL);
  double cublasTime
    = timeEnd.tv_sec - timeStart.tv_sec
    + (timeEnd.tv_usec - timeStart.tv_usec)/1e6;
  if(status != CUBLAS_STATUS_SUCCESS)
    std::cout << "cublasTrsm FAILED" << std::endl;

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
    std::cout << std::endl
	      << "Matrix entries" << std::endl
	      << "----------------------------------------" << std::endl
	      << "  alpha = " << alpha << std::endl;
    std::cout << "  shifts =" << std::endl;
    printMatrix<datafloat>(std::cout,'N','N',1,n,shifts,1);
    std::cout << "  A =" << std::endl;
    printMatrix<datafloat>(std::cout,uplo,diag,m,m,A,m);
    std::cout << "  B =" << std::endl;
    printMatrix<datafloat>(std::cout,'N','N',m,n,B,m);
    std::cout << "  X =" << std::endl;
    printMatrix<datafloat>(std::cout,'N','N',m,n,X,m);
  }

  // Check error in solution
  double normB = nrm2<datafloat>(m*n,B,1);
  memcpy(residual,X,m*n*sizeof(datafloat));
  trmm<datafloat>(side,uplo,trans,diag,m,n,1,A,m,residual,m);
#pragma omp parallel for
  for(int i=0;i<n;++i)
    axpy<datafloat>(m, shifts[i], X+i*m, 1, residual+i*m, 1);
  axpy<datafloat>(m*n, -alpha, B, 1, residual, 1);
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
