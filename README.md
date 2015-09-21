# cudaMultiShiftTrsm
Multi-Shift Triangular Solve in CUDA

## Compilation
A static library can be compiled with CMake by typing the following
into the command line:
```
mkdir build
cd build
cmake ..
make
cd ..
```
The static library will be located at `build/libcuda_mstrsm.a`.

## Validation
A validation program can be compiled by running CMake with the
argument `-DTEST=1`. The executable will be located at
`build/test/cuda_mstrsm_test`. To see the help page, type the
following into the command line:
```
build/test/cuda_mstrsm_test --help
```
