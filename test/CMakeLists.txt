# Initialize CMake
#   CMake should be called by parent project (cudaMultiShiftTrsm)
cmake_minimum_required(VERSION 2.8)
project(cudaMultiShiftTrsm_test)

# Dependencies
find_package(LAPACK QUIET REQUIRED)

# Header files
include_directories(${cudaMultiShiftTrsm_SOURCE_DIR}/include)

# Compile test program
cuda_add_executable(cuda_mstrsm_test cudaMultiShiftTrsmTest.cu)
cuda_add_cublas_to_target(cuda_mstrsm_test)
target_link_libraries(cuda_mstrsm_test
  ${CUDA_LIBRARIES} ${CUBLAS_LIBRARIES} ${LAPACK_LIBRARIES}
  cuda_mstrsm)
