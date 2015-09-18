# Initialize CMake
cmake_minimum_required(VERSION 2.8)
project(cudaMultiShiftTrsm)

# Dependencies
find_package(CUDA QUIET REQUIRED)

# Header files
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files
file(GLOB cudaMultiShiftTrsm_SRC "src/*.cpp" "src/*.c" "src/*.cu")

# Compile cudaMultiShiftTrsm into a static library
cuda_add_library(cuda_mstrsm ${cudaMultiShiftTrsm_SRC})
cuda_add_cublas_to_target(cuda_mstrsm)
target_link_libraries(cuda_mstrsm ${CUDA_LIBRARIES} ${CUBLAS_LIBRARIES})

# Compile test program
add_subdirectory(test)
