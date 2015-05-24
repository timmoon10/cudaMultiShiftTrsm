# cudaMultiShiftTrsm
Multi-Shift Triangular Solve in CUDA

## Validation
Compile the validation program by typing `make` into the command
line. The following arguments can be applied:

Argument        | Description
----------------|---------------------------------------------
`DATAFLOAT="#"` | Choose data type (`float`,`double`,`complex<float>`,`complex<double>`). `float` is default.
`DEBUG=1`       | Activate debugging flags (`-g`, `-pg`)
`O=1`           | Activate compiler optimization flag (`-O2`)

The validation program takes up to seven arguments:
```
./validation m n side uplo trans diag verbose
```

Argument   | Default | Description
-----------|---------|---------------------------------------------
`m`        | `4`     | Matrix dimension
`n`        | `1`     | Number of right hand sides
`side`     | `L`     | Side to apply matrix
`uplo`     | `L`     | Whether matrix is upper or lower triangular
`trans`    | `N`     | Whether to apply matrix transpose
`diag`     | `N`     | Whether matrix is unit triangular
`verbose`  | `0`     | Whether to output matrix entries
