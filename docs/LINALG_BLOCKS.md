# GNU Radio Linalg Blocks

List of public block headers in `include/gnuradio/linalg/` (excluding internal `detail/`).

## Decomposition
- `decomp_cholesky.h` – Cholesky factorization
- `decomp_eigen.h` – Eigenvalue decomposition
- `decomp_hessenberg.h` – Hessenberg reduction
- `decomp_lu.h` – LU decomposition
- `decomp_qr.h` – QR decomposition
- `decomp_schur.h` – Schur decomposition
- `decomp_svd.h` – Singular Value Decomposition
- (Experimental / legacy) `decomp_qr_corrupted.h`, `decomp_qr_new.h`

## Core / Base
- `linalg.h` – Top-level convenience includes
- `linalg_base.h` – Common base functionality
- `linalg_base_pdu.h` – PDU-based base class
- `linalg_base_sync.h` – Stream (sync) base class
- `array_slice.h` – Array slicing utilities
- `types.h` – Type aliases and traits
- `utils.h` – General utilities
- `utils_fixed.h` – Fixed-size helpers
- `performance_profiler.h` – Profiling helpers
- `api.h` – Export macros

## Matrix Creation & Source
- `eye.h` – Identity matrix source
- `ones.h` – Ones matrix source
- `zeros.h` – Zeros matrix source
- `matrix_source_const.h` – Constant matrix source

## Matrix Elementwise / Basic Ops
- `matrix_add.h`
- `matrix_subtract.h`
- `matrix_elementwise_multiply.h`
- `matrix_elementwise_divide.h`
- `matrix_diag.h` – Extract / form diagonal
- `matrix_transpose.h`
- `matrix_trace.h`
- `matrix_reshape.h`
- `matrix_power.h`
- `matrix_exp.h`
- `matrix_log.h`
- `matrix_sqrt.h`
- `matrix_trig.h` – Trigonometric (elementwise)
- `matrix_hermitian.h` – Hermitian (conjugate transpose)

## Matrix Products & Advanced Ops
- `matrix_multiply.h`
- `matrix_kronecker_product.h`
- `matrix_inverse.h`
- `matrix_pseudo_inverse.h`
- `matrix_solve.h`
- `solve_least_squares.h`
- `solve_triangular.h`
- `matrix_rank.h`
- `matrix_condition_number.h`
- `matrix_pseudo_determinant.h`
- `matrix_determinant.h`
- `matrix_norm.h`

## Vector Operations
- `vector_dot.h`
- `vector_cross_product.h`
- `vector_outer_product.h`
- `vector_product_inner.h`
- `vector_product_outer.h`
- `vector_projection.h`
- `vector_angle.h`
- `vector_correlate.h`
- `vector_norm.h`
- `vector_normalize.h`
- `vector_to_matrix.h`

## Mixed / Other
- `dot_product.h` – Generic dot product (matrix/vector)
- `matrix_elementwise_multiply.h` / `matrix_elementwise_divide.h` (also apply to vectors via broadcasting in implementations)
