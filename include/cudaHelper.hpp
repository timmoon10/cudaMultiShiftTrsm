#pragma once

#include <iostream>
#include <cuda.h>

// Macro for error-checking CUDA calls
#define CUDA_CHECK(status)					\
  do {								\
    cudaError_t e = (status);					\
    if(e != cudaSuccess) {					\
      std::cerr << "CUDA error "				\
		<< "(" << __FILE__ << ":" << __LINE__ << ")"	\
		<< ": " << cudaGetErrorString(e)		\
		<< std::endl;					\
      exit(EXIT_FAILURE); /* TODO: find better way to fail */	\
    }								\
  } while(0)

// Macro to display warning messages
#define WARNING(message)					\
  do {								\
    std::cerr << "Warning "					\
	      << "(" << __FILE__ << ":" << __LINE__ << ")"	\
	      << ": " << message				\
	      << std::endl;					\
  } while(0)

// Macro to obtain Fortran matrix index
#define IDX(i,j,ld) ((i)+(j)*(ld))
