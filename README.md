# cudaMultiShiftTrsm
Multi-Shift Triangular Solve in CUDA

# Validation
The validation program takes seven arguments:
    ./validation m n side uplo trans diag verbose

    Argument	 Default  Description
    m		 4	  Matrix dimension
    n		 1	  Number of right hand sides
    side	 L	  Side to apply matrix
    uplo	 L	  Whether matrix is upper or lower triangular
    trans	 N	  Whether to apply matrix transpose
    diag	 N	  Whether matrix is unit triangular
    verbose	 0	  Whether to output matrix entries
