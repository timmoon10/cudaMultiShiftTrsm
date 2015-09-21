#include <cstdlib>
#include <cstring>
#include <string>
#include <cmath>
#include <iostream>
#include <cstdio>
#include <complex>
#include <limits>
#include <sys/time.h> // Not implemented in Windows

#include <cuda.h>
#include <cublas_v2.h>

#include "gtest/gtest.h"
#include "cudaHelper.hpp"
#include "cublasHelper.hpp"
#include "lapackHelper.hpp"
#include "cudaMultiShiftTrsm.hpp"

using namespace std;

// ===============================================
// Test matrices
// ===============================================

/// Compute Gaussian random variable
/** Uses Box-Muller transform to convert uniform distribution to
 *  Gaussian distribution
 */
template <typename F>
F randn() {
  F u1 = ((F)std::rand())/RAND_MAX;
  F u2 = ((F)std::rand())/RAND_MAX;
  return std::sqrt(-2*std::log(u1))*std::cos(2*M_PI*u2);
}
template <>
complex<float> randn<complex<float> >() {
  float u1 = ((float)std::rand())/RAND_MAX;
  float u2 = ((float)std::rand())/RAND_MAX;
  return complex<float>(std::sqrt(-2*std::log(u1))*std::cos(2*M_PI*u2),
			std::sqrt(-2*std::log(u1))*std::sin(2*M_PI*u2));
}
template <>
complex<double> randn<complex<double> >() {
  double u1 = ((double)std::rand())/RAND_MAX;
  double u2 = ((double)std::rand())/RAND_MAX;
  return complex<double>(std::sqrt(-2*std::log(u1))*std::cos(2*M_PI*u2),
			 std::sqrt(-2*std::log(u1))*std::sin(2*M_PI*u2));
}

/// Generate matrix with Gaussian random variables
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
  F *temp = (F*) std::malloc(m*m*sizeof(F));
  randn<F>(m*m,temp);

  // Construct positive definite matrix
  herk(uplo,'N',m,m,1,temp,m,0,A,m);

  // Shift diagonal to improve condition number
#pragma omp parallel for
  for(int i=0;i<m;++i)
    A[i+i*m] += std::sqrt(m);

  // Perform Cholesky factorization
  int info;
  potrf(uplo, m, A, m, info);

  // Clean up
  std::free(temp);

}

/// Output matrix to stream
template <typename F>
void printMatrix(ostream & os,
		 const char uplo, const char diag,
		 const int m, const int n, 
		 const F * A, const int lda) {

  os << "    [[";
  for(int i=0;i<m;++i) {
    if(std::toupper(uplo) == 'L') {
      for(int j=0;j<i;++j)
	os << A[i+j*lda] << " ";
      if(std::toupper(diag)=='U')
	os << "1 ";
      else
	os << A[i+i*lda] << " ";
      for(int j=i+1;j<n;++j)
	os << "0 ";
    }
    else if(std::toupper(uplo) == 'U') {
      for(int j=0;j<i;++j)
	os << "0 ";
      if(std::toupper(diag)=='U')
	os << "1 ";
      else
	os << A[i+i*lda] << " ";
      for(int j=i+1;j<n;++j)
	os << A[i+j*lda] << " ";
    }
    else {
      for(int j=0;j<n;++j)
	os << A[i+j*lda] << " ";
    }
    os << "]";
    if(i<m-1)
      os << "\n" << "    [";
    else
      os << "]\n";
  }
}

// ===============================================
// Validation program
// ===============================================

/// cuBLAS handle
static cublasHandle_t cublasHandle;

/// Run validation program
template <typename F>
double validation(int m, int n,
		  char side, char uplo, char trans, char diag,
		  bool benchmark, bool verbose) {

  // -------------------------------------------------
  // Variable declarations
  // -------------------------------------------------

  // Host memory
  F alpha;
  std::vector<F> A(m*m);
  std::vector<F> B(m*n);
  std::vector<F> shifts(n);
  std::vector<F> X(m*n);
  std::vector<F> residual(m*n);

  // Device memory
  F * A_device;
  F * B_device;
  F * shifts_device;

  // cuBLAS objects
  cublasStatus_t cublasStatus;
  cublasSideMode_t  cublasSide;
  cublasFillMode_t  cublasUplo;
  cublasOperation_t cublasTrans;
  cublasDiagType_t  cublasDiag;

  // Timing
  timeval timeStart, timeEnd;
  double cudaMstrsmTime, cublasTime;

  // Error in solution
  double normB, normResidual;
  double relError;

  // -------------------------------------------------
  // Initialization
  // -------------------------------------------------

  // Initialize matrices on host
  alpha = randn<F>();
  choleskyRandomMatrix<F>(uplo, m, A.data());
  randn<F>(m*n, B.data());
  randn<F>(n, shifts.data());

  // Initialize memory on device
  CUDA_CHECK(cudaMalloc(&A_device, m*m*sizeof(F)));
  CUDA_CHECK(cudaMalloc(&B_device, m*n*sizeof(F)));
  CUDA_CHECK(cudaMalloc(&shifts_device, n*sizeof(F)));
  CUDA_CHECK(cudaMemcpy(A_device, A.data(), m*m*sizeof(F),
			cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B_device, B.data(), m*n*sizeof(F),
			cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(shifts_device, shifts.data(), n*sizeof(F),
			cudaMemcpyHostToDevice));

  // Initialize BLAS parameters
  side  = std::toupper(side);
  uplo  = std::toupper(uplo);
  trans = std::toupper(trans);
  diag  = std::toupper(diag);
  switch(side) {
  case 'L': cublasSide = CUBLAS_SIDE_LEFT; break;
  case 'R': cublasSide = CUBLAS_SIDE_RIGHT; break;
  default: WARNING("invalid parameter for side"); exit(EXIT_FAILURE);
  }
  switch(uplo) {
  case 'L': cublasUplo = CUBLAS_FILL_MODE_LOWER; break;
  case 'U': cublasUplo = CUBLAS_FILL_MODE_UPPER; break;
  default: WARNING("invalid parameter for uplo"); exit(EXIT_FAILURE);
  }
  switch(trans) {
  case 'N': cublasTrans = CUBLAS_OP_N; break;
  case 'T': cublasTrans = CUBLAS_OP_T; break;
  case 'C': cublasTrans = CUBLAS_OP_C; break;
  default: WARNING("invalid parameter for trans"); exit(EXIT_FAILURE);
  }
  switch(diag) {
  case 'N': cublasDiag = CUBLAS_DIAG_NON_UNIT; break;
  case 'U': cublasDiag = CUBLAS_DIAG_UNIT; break;
  default: WARNING("invalid parameter for diag"); exit(EXIT_FAILURE);
  }

  // -------------------------------------------------
  // Test cudaMultiShiftTrsm
  // -------------------------------------------------

  // Solve shifted triangular system
  cudaDeviceSynchronize();
  gettimeofday(&timeStart, NULL);
  cublasStatus = cudaMstrsm::cudaMultiShiftTrsm<F>
    (cublasHandle, cublasSide, cublasUplo, cublasTrans, cublasDiag,
     m, n, &alpha, A_device, m, B_device, m, shifts_device);
  cudaDeviceSynchronize();
  gettimeofday(&timeEnd, NULL);
  cudaMstrsmTime = ((timeEnd.tv_sec - timeStart.tv_sec)
		    + (timeEnd.tv_usec - timeStart.tv_usec)/1e6);
  EXPECT_EQ(CUBLAS_STATUS_SUCCESS, cublasStatus);

  // Transfer result to host
  CUDA_CHECK(cudaMemcpy(X.data(), B_device, m*n*sizeof(F),
			cudaMemcpyDeviceToHost));

  // Check error in solution
  normB = nrm2(m*n, B.data(), 1);
  std::memcpy(residual.data(), X.data(), m*n*sizeof(F));
  trmm(side, uplo, trans, diag, m, n,
       1, A.data(), m, residual.data(), m);
#pragma omp parallel for
  for(int i=0;i<n;++i)
    axpy(m, shifts[i], X.data()+i*m, 1, residual.data()+i*m, 1);
  axpy(m*n, -alpha, B.data(), 1, residual.data(), 1);
  normResidual = nrm2(m*n,residual.data(),1);
  relError = normResidual/normB;

  // -------------------------------------------------
  // cuBLAS triangular solve
  //   For performance comparison if benchmark mode
  //   is activated
  // -------------------------------------------------

  if(benchmark) {

    // Transfer right-hand-side to device
    CUDA_CHECK(cudaMemcpy(B_device, B.data(), m*n*sizeof(F),
			  cudaMemcpyHostToDevice));

    // Solve triangular system
    cudaDeviceSynchronize();
    gettimeofday(&timeStart, NULL);
    CUBLAS_CHECK(cublasTrsm(cublasHandle, cublasSide, cublasUplo,
			    cublasTrans, cublasDiag, m, n,
			    &alpha, A_device, m, B_device, m));
    cudaDeviceSynchronize();
    gettimeofday(&timeEnd, NULL);
    cublasTime = ((timeEnd.tv_sec - timeStart.tv_sec)
		  + (timeEnd.tv_usec - timeStart.tv_usec)/1e6);

  }

  // -------------------------------------------------
  // Output results
  // -------------------------------------------------

  // Output benchmark results if benchmark mode is activated
  if(benchmark) {

    // Report error
    std::cout << "\n"
	      << "Relative error (Frobenius norm)\n"
	      << "----------------------------------------\n"
	      << "  cudaMstrsm : " << relError << "\n";

    // Report time for matrix multiplication
    std::cout << "\n"
	      << "Timings\n"
	      << "----------------------------------------\n"
	      << "  cudaMstrsm : " << cudaMstrsmTime << " sec\n"
	      << "  cuBLAS     : " << cublasTime     << " sec\n";

    // Report FLOPS
    double gflopCount = 1e-9*m*m*n; // Approximate
    std::cout << "\n"
	      << "Performance\n"
	      << "----------------------------------------\n"
	      << "  cudaMstrsm : " << gflopCount/cudaMstrsmTime << " GFLOPS\n"
	      << "  cuBLAS     : " << gflopCount/cublasTime     << " GFLOPS\n";

  }

  // Output matrix entries if verbose output is activated
  if(verbose) {
    // Print matrices
    std::cout << "\n"
	      << "Matrix entries\n"
	      << "----------------------------------------\n"
	      << "  alpha = " << alpha << "\n";
    std::cout << "  shifts =\n";
    printMatrix<F>(std::cout,'N','N',1,n,shifts.data(),1);
    std::cout << "  A =\n";
    printMatrix<F>(std::cout,uplo,diag,m,m,A.data(),m);
    std::cout << "  B =\n";
    printMatrix<F>(std::cout,'N','N',m,n,B.data(),m);
    std::cout << "  X =\n";
    printMatrix<F>(std::cout,'N','N',m,n,X.data(),m);
  }

  // Clean up and return
  CUDA_CHECK(cudaFree(A_device));
  CUDA_CHECK(cudaFree(B_device));
  CUDA_CHECK(cudaFree(shifts_device));
  return relError;

}

/// Run validation program for each data type
void unitTest(char side, char uplo, char trans, char diag) {

  // Parameters
  int m = 723;
  int n = 19;

  // Relative error
  double relError_S, relError_D, relError_C, relError_Z;

  // Machine precision
  double eps_float  = std::numeric_limits<float>::epsilon();
  double eps_double = std::numeric_limits<double>::epsilon();

  // Float (S)
  relError_S = validation<float>(m, n, side, uplo, trans, diag,
				 false, false);
  EXPECT_LT(relError_S, 100*eps_float);

  // Double (D)
  relError_D = validation<double>(m, n, side, uplo, trans, diag,
				  false, false);
  EXPECT_LT(relError_D, 100*eps_double);
  
  // Single-precision complex (C)
  relError_C
    = validation<std::complex<float> >(m, n, side, uplo, trans, diag,
				       false, false);
  EXPECT_LT(relError_C, 100*eps_float);

  // Double-precision complex (Z)
  relError_Z 
    = validation<std::complex<double> >(m, n, side, uplo, trans, diag,
					false, false);
  EXPECT_LT(relError_Z, 100*eps_double);
  
}

// ===============================================
// Unit tests
// ===============================================

TEST(cudaMultiShiftTrsmTest, LLNN) { unitTest('L','L','N','N'); }
TEST(cudaMultiShiftTrsmTest, RLNN) { unitTest('R','L','N','N'); }
TEST(cudaMultiShiftTrsmTest, LUNN) { unitTest('L','U','N','N'); }
TEST(cudaMultiShiftTrsmTest, RUNN) { unitTest('R','U','N','N'); }
TEST(cudaMultiShiftTrsmTest, LLTN) { unitTest('L','L','T','N'); }
TEST(cudaMultiShiftTrsmTest, RLTN) { unitTest('R','L','T','N'); }
TEST(cudaMultiShiftTrsmTest, LUTN) { unitTest('L','U','T','N'); }
TEST(cudaMultiShiftTrsmTest, RUTN) { unitTest('R','U','T','N'); }
TEST(cudaMultiShiftTrsmTest, LLCN) { unitTest('L','L','C','N'); }
TEST(cudaMultiShiftTrsmTest, RLCN) { unitTest('R','L','C','N'); }
TEST(cudaMultiShiftTrsmTest, LUCN) { unitTest('L','U','C','N'); }
TEST(cudaMultiShiftTrsmTest, RUCN) { unitTest('R','U','C','N'); }
TEST(cudaMultiShiftTrsmTest, LLNU) { unitTest('L','L','N','U'); }
TEST(cudaMultiShiftTrsmTest, RLNU) { unitTest('R','L','N','U'); }
TEST(cudaMultiShiftTrsmTest, LUNU) { unitTest('L','U','N','U'); }
TEST(cudaMultiShiftTrsmTest, RUNU) { unitTest('R','U','N','U'); }
TEST(cudaMultiShiftTrsmTest, LLTU) { unitTest('L','L','T','U'); }
TEST(cudaMultiShiftTrsmTest, RLTU) { unitTest('R','L','T','U'); }
TEST(cudaMultiShiftTrsmTest, LUTU) { unitTest('L','U','T','U'); }
TEST(cudaMultiShiftTrsmTest, RUTU) { unitTest('R','U','T','U'); }
TEST(cudaMultiShiftTrsmTest, LLCU) { unitTest('L','L','C','U'); }
TEST(cudaMultiShiftTrsmTest, RLCU) { unitTest('R','L','C','U'); }
TEST(cudaMultiShiftTrsmTest, LUCU) { unitTest('L','U','C','U'); }
TEST(cudaMultiShiftTrsmTest, RUCU) { unitTest('R','U','C','U'); }

// ===============================================
// Main function
// ===============================================

int main(int argc, char **argv) {
  
  // Default parameters
  int  m         = 1000;
  int  n         = 50;
  char dataType  = 'S';
  char side      = 'L';
  char uplo      = 'L';
  char trans     = 'N';
  char diag      = 'N';
  bool benchmark = false;
  bool verbose   = false;

  // Status flag
  int status;

  // Check for help command line argument
  for(int i=1; i<argc; ++i) {
    if(strcmp(argv[i],"--help")==0 || strcmp(argv[i],"-h")==0) {

      // Display help page
      std::cout << "\n"
		<< "Testing suite for CUDA implementation of triangular solve with" << "\n"
		<< "multiple shifts. Unit tests are performed with Google Test." << "\n"
		<< "Command line arguments are ignored if benchmark mode is not activated." << "\n"
		<< "\n"
		<< "Usage: cuda_mstrsm_test [options]" << "\n"
		<< "\n"
		<< "Options" << "\n"
		<< "========================================" << "\n"
		<< "--help                Display this help page." << "\n"
		<< "--benchmark           Activate benchmark mode and disable Google Test."  << "\n"
		<< "                      Performance of triangular solve is compared with" << "\n" 
		<< "                      performance of cublas<T>trsm." << "\n"
		<< "--verbose             Enable verbose output." << "\n"
		<< "--m=[#]               Dimension of triangular matrix." << "\n"
		<< "--n=[#]               Number of right-hand-side vectors." << "\n"
		<< "--datatype=[S/D/C/Z]  Floating-point data type." << "\n"
		<< "--side=[L/R]          Side to apply triangular matrix." << "\n"
		<< "--uplo=[L/U]          Type of triangular matrix." << "\n"
		<< "--trans=[N/T/C]       Whether to transpose triangular matrix." << "\n"
		<< "--diag=[N/U]          Whether to transpose triangular matrix." << "\n";
      std::cout << std::flush;

      // Display help page for Google Test
      std::cout << "\n"
		<< "Help page for Google Test" << "\n"
		<< "========================================" << "\n";
      std::cout << std::flush;
      ::testing::InitGoogleTest(&argc, argv);
      std::cout << std::flush;

      // Exit
      exit(EXIT_SUCCESS);
      
    }
  }

  // Initialize Google Test
  ::testing::InitGoogleTest(&argc, argv);

  // Interpret command line arguments
  for(int i=1; i<argc; ++i) {

    std::string currArg(argv[i]);
    if(currArg.compare("--benchmark")==0)
      benchmark = true;
    else if(currArg.compare("--verbose")==0)
      verbose = true;
    else if(currArg.find("--m=")==0)
      m = std::atoi(currArg.c_str()+std::strlen("--m="));
    else if(currArg.find("--n=")==0)
      n = std::atoi(currArg.c_str()+std::strlen("--n="));
    else if(currArg.find("--datatype=")==0)
      dataType = std::toupper(currArg[std::strlen("--datatype=")]);
    else if(currArg.find("--side=")==0)
      side = std::toupper(currArg[std::strlen("--side=")]);
    else if(currArg.find("--uplo=")==0)
      uplo = std::toupper(currArg[std::strlen("--uplo=")]);
    else if(currArg.find("--trans=")==0)
      trans = std::toupper(currArg[std::strlen("--trans=")]);
    else if(currArg.find("--diag=")==0)
      diag = std::toupper(currArg[std::strlen("--diag=")]);
    else {
      char message[512];
      std::sprintf(message, "invalid argument (%s)", currArg.c_str());
      WARNING(message);
    }

  }

  // Report parameters
  if(benchmark) {
    std::cout << "========================================\n"
	      << "  SHIFTED TRIANGULAR SOLVE BENCHMARK\n"
	      << "========================================\n"
	      << "\n"
	      << "Parameters\n"
	      << "----------------------------------------\n"
	      << "m         = " << m << "\n"
	      << "n         = " << n << "\n"
	      << "Data type = " << dataType << "\n"
	      << "side      = " << side << "\n"
	      << "uplo      = " << uplo << "\n"
	      << "trans     = " << trans << "\n"
	      << "diag      = " << diag << "\n";
  }

  // Initialize cuBLAS
  CUBLAS_CHECK(cublasCreate(&cublasHandle));

  // Perform benchmark
  if(benchmark) {
    if(dataType == 'S')
      validation<float>(m,n,side,uplo,trans,diag,true,verbose);
    else if(dataType == 'D')
      validation<double>(m,n,side,uplo,trans,diag,true,verbose);
    else if(dataType == 'C')
      validation<std::complex<float> >(m,n,side,uplo,trans,diag,
				       true,verbose);
    else if(dataType == 'Z')
      validation<std::complex<double> >(m,n,side,uplo,trans,diag,
					true,verbose);
    else
      WARNING("invalid data type");

    status = EXIT_SUCCESS;
  }

  // Run Google Test
  else
    status = RUN_ALL_TESTS();

  // Clean up and exit
  CUBLAS_CHECK(cublasDestroy(cublasHandle));
  return status;

}
