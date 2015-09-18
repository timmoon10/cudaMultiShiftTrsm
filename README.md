# cudaMultiShiftTrsm
Multi-Shift Triangular Solve in CUDA

## Compilation
A static library and a validation program can be compiled with CMake
by typing the following into the command line:
```
mkdir build
cd build
cmake ..
make
```

The static library will be located at `build/libcuda_mstrsm.a` and the
validation program at `build/test/cuda_mstrsm_test`

## Validation program
The validation program takes up to eight arguments:
```
build/test/cuda_mstrsm_test m n dataType side uplo trans diag verbose
```

Argument   | Default | Description
-----------|---------|---------------------------------------------
`m`        | `4`     | Matrix dimension
`n`        | `1`     | Number of right hand sides
`dataType` | `S`     | Data type (`S`, `D`, `C`, or `Z`)
`side`     | `L`     | Side to apply matrix (`L` or `R`)
`uplo`     | `L`     | Type of triangular matrix (`L` or `U`)
`trans`    | `N`     | Whether to apply matrix transpose (`N`, `T`, or `C`)
`diag`     | `N`     | Whether matrix is unit triangular (`N` or `U`)
`verbose`  | `0`     | Whether to output matrix entries
