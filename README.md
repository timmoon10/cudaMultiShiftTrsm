# cudaMultiShiftTrsm
Multi-Shift Triangular Solve in CUDA

## Validation
Compile the validation program by typing `make` into the command
line. The following arguments can be applied:

Argument        | Description
----------------|---------------------------------------------
`DEBUG=1`       | Activate debugging flags (`-g` and `-pg`)
`O=#`           | Activate compiler optimization flag (`-O0`, `-O1`, `-O2`, or `-O3`, depending on input)

The validation program takes up to eight arguments:
```
./validation m n dataType side uplo trans diag verbose
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
