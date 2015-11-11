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
#include <cuda_profiler_api.h>
#include <cublas_v2.h>

#include "gtest/gtest.h"
#include "cudaHelper.hpp"
#include "cublasHelper.hpp"
#include "lapackHelper.hpp"
#include "cudaMultiShiftTrsm.hpp"

using namespace std;

/// cuBLAS handle
static cublasHandle_t cublasHandle;

// ===============================================
// Helper functions
// ===============================================

/// Output matrix to stream
template <typename F>
void printMatrix(ostream & os,
		 const char uplo,
		 const int m, const int n, 
		 const F * A, const int lda) {

  os << "    [[";
  for(int i=0;i<m;++i) {
    if(std::toupper(uplo) == 'L') {
      for(int j=0;j<=i;++j)
	os << A[i+j*lda] << " ";
      for(int j=i+1;j<n;++j)
	os << "0 ";
    }
    else if(std::toupper(uplo) == 'U') {
      for(int j=0;j<i;++j)
	os << "0 ";
      for(int j=i;j<n;++j)
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

/// Run validation program
template <typename F>
double validation(int m, int n,
		  char uplo, char trans,
		  bool benchmark, bool profile, bool verbose) {

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
  std::vector<int> work_int(m);

  // Device memory
  F * A_device;
  F * B_device;
  F * shifts_device;

  // LAPACK objects
  int iseed[4] = {1234,321,2345,43}; // TODO: pick better seed

  // cuBLAS objects
  cublasStatus_t cublasStatus;
  cublasFillMode_t  cublasUplo;
  cublasOperation_t cublasTrans;

  // Timing
  timeval timeStart, timeEnd;
  double cudaMstrsmTime, cublasTime;

  // Error in solution
  double normB, normResidual;
  double relError;

  // -------------------------------------------------
  // Initialization
  // -------------------------------------------------

  // Initialize BLAS parameters
  uplo  = std::toupper(uplo);
  trans = std::toupper(trans);
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

  // Generate triangular matrix
  //   Perform LU factorization and copy L to upper triangle (diagonal
  //   of U is preserved). This method is relatively well-conditioned.
  larnv(3, iseed, m*m, A.data());
  getrf(m, m, A.data(), m, work_int.data());
  for(int j=0; j<m; ++j)
    for(int i=0; i<j; ++i)
      A[IDX(i,j,m)] = A[IDX(j,i,m)];

  // Initialize shifts
  larnv(3, iseed, n, shifts.data());

  // Initialize remaining matrices on host
  larnv(3, iseed, 1,   &alpha);
  larnv(3, iseed, m*n, B.data());

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

  // -------------------------------------------------
  // Test cudaMultiShiftTrsm
  // -------------------------------------------------


  // Solve shifted triangular system
  if(profile)
    cudaProfilerStart();
  cudaDeviceSynchronize();
  gettimeofday(&timeStart, NULL);
  cublasStatus = cudaMstrsm::cudaMultiShiftTrsm<F>
    (cublasHandle, cublasUplo, cublasTrans,
     m, n, &alpha, A_device, m, B_device, m, shifts_device);
  cudaDeviceSynchronize();
  gettimeofday(&timeEnd, NULL);
  if(profile)
    cudaProfilerStop();
  cudaMstrsmTime = ((timeEnd.tv_sec - timeStart.tv_sec)
		    + (timeEnd.tv_usec - timeStart.tv_usec)/1e6);
  EXPECT_EQ(CUBLAS_STATUS_SUCCESS, cublasStatus);

  // Transfer result to host
  CUDA_CHECK(cudaMemcpy(X.data(), B_device, m*n*sizeof(F),
			cudaMemcpyDeviceToHost));

  // Check error in solution
  normB = nrm2(m*n, B.data(), 1);
  std::memcpy(residual.data(), X.data(), m*n*sizeof(F));
  trmm('L', uplo, trans, 'N', m, n,
       1, A.data(), m, residual.data(), m);
  for(int i=0;i<n;++i)
    axpy(m, shifts[i], X.data()+IDX(0,i,m), 1,
	 residual.data()+IDX(0,i,m), 1);
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
    CUBLAS_CHECK(cublasTrsm(cublasHandle, CUBLAS_SIDE_LEFT, cublasUplo,
			    cublasTrans, CUBLAS_DIAG_NON_UNIT, m, n,
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
    printMatrix<F>(std::cout,'N',1,n,shifts.data(),1);
    std::cout << "  A =\n";
    printMatrix<F>(std::cout,uplo,m,m,A.data(),m);
    std::cout << "  B =\n";
    printMatrix<F>(std::cout,'N',m,n,B.data(),m);
    std::cout << "  X =\n";
    printMatrix<F>(std::cout,'N',m,n,X.data(),m);
  }

  // Clean up and return
  CUDA_CHECK(cudaFree(A_device));
  CUDA_CHECK(cudaFree(B_device));
  CUDA_CHECK(cudaFree(shifts_device));
  return relError;

}

// ===============================================
// Unit tests
// ===============================================

/// Run validation program for each data type
void unitTest(char uplo, char trans) {

  // Parameters
  int m = 723;
  int n = 19;

  // Relative error
  double relError_S, relError_D, relError_C, relError_Z;

  // Machine precision
  double eps_float  = std::numeric_limits<float>::epsilon();
  double eps_double = std::numeric_limits<double>::epsilon();

  // Float (S)
  relError_S = validation<float>(m, n, uplo, trans,
				 false, false, false);
  EXPECT_LT(relError_S, 100*eps_float);

  // Double (D)
  relError_D = validation<double>(m, n, uplo, trans,
				  false, false, false);
  EXPECT_LT(relError_D, 100*eps_double);
  
  // Single-precision complex (C)
  relError_C
    = validation<std::complex<float> >(m, n, uplo, trans,
				       false, false, false);
  EXPECT_LT(relError_C, 100*eps_float);

  // Double-precision complex (Z)
  relError_Z 
    = validation<std::complex<double> >(m, n, uplo, trans,
					false, false, false);
  EXPECT_LT(relError_Z, 100*eps_double);
  
}

TEST(cudaMultiShiftTrsmTest, LLNN) { unitTest('L','N'); }
TEST(cudaMultiShiftTrsmTest, LUNN) { unitTest('U','N'); }
TEST(cudaMultiShiftTrsmTest, LLTN) { unitTest('L','T'); }
TEST(cudaMultiShiftTrsmTest, LUTN) { unitTest('U','T'); }
TEST(cudaMultiShiftTrsmTest, LLCN) { unitTest('L','C'); }
TEST(cudaMultiShiftTrsmTest, LUCN) { unitTest('U','C'); }

// ===============================================
// Main function
// ===============================================

int main(int argc, char **argv) {
  
  // Default parameters
  int  m         = 1000;
  int  n         = 50;
  char dataType  = 'S';
  char uplo      = 'L';
  char trans     = 'N';
  bool benchmark = false;
  bool profile   = false;
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
		<< "--profile             Activate focused profiling for nvprof or nvvp." << "\n"
		<< "--verbose             Enable verbose output." << "\n"
		<< "--m=[#]               Dimension of triangular matrix." << "\n"
		<< "--n=[#]               Number of right-hand-side vectors." << "\n"
		<< "--datatype=[S/D/C/Z]  Floating-point data type." << "\n"
		<< "--uplo=[L/U]          Type of triangular matrix." << "\n"
		<< "--trans=[N/T/C]       Whether to transpose triangular matrix." << "\n";
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
    else if(currArg.compare("--profile")==0)
      profile = true;
    else if(currArg.compare("--verbose")==0)
      verbose = true;
    else if(currArg.find("--m=")==0)
      m = std::atoi(currArg.c_str()+std::strlen("--m="));
    else if(currArg.find("--n=")==0)
      n = std::atoi(currArg.c_str()+std::strlen("--n="));
    else if(currArg.find("--datatype=")==0)
      dataType = std::toupper(currArg[std::strlen("--datatype=")]);
    else if(currArg.find("--uplo=")==0)
      uplo = std::toupper(currArg[std::strlen("--uplo=")]);
    else if(currArg.find("--trans=")==0)
      trans = std::toupper(currArg[std::strlen("--trans=")]);
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
	      << "uplo      = " << uplo << "\n"
	      << "trans     = " << trans << "\n";
  }

  // Initialize cuBLAS
  CUBLAS_CHECK(cublasCreate(&cublasHandle));

  // Perform benchmark
  if(benchmark) {
    if(dataType == 'S')
      validation<float>(m,n,uplo,trans,
			true,profile,verbose);
    else if(dataType == 'D')
      validation<double>(m,n,uplo,trans,
			 true,profile,verbose);
    else if(dataType == 'C')
      validation<std::complex<float> >(m,n,uplo,trans,
				       true,profile,verbose);
    else if(dataType == 'Z')
      validation<std::complex<double> >(m,n,uplo,trans,
					true,profile,verbose);
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
