# GNU Radio Linear Algebra Blocks - Comprehensive List

This document provides a detailed description of all linear algebra blocks available in the gr-linalg GNU Radio Out-of-Tree module.

## Block Categories

### Matrix Operations

#### matrix_add

**Purpose**: Performs element-wise addition of matrices
**Input**: Two or more matrices of the same shape
**Output**: Single matrix containing the sum
**Execution Models**: Stream-based (`linalg_base_sync`) and PDU-based (`linalg_base_pdu`)
**Supported Types**: float, double, complex<float>, complex<double>
**Broadcasting**: Supports various broadcasting policies (NONE, VECTOR, MATRIX, CUSTOM)
**Use Cases**: Signal combination, offset addition, matrix arithmetic operations

#### matrix_subtract

**Purpose**: Performs element-wise subtraction of matrices
**Input**: Two matrices of the same shape (minuend and subtrahend)
**Output**: Single matrix containing the difference
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Use Cases**: Error calculation, differential analysis, signal processing

#### matrix_multiply

**Purpose**: Standard matrix multiplication (not element-wise)
**Input**: Two matrices where first matrix columns equal second matrix rows
**Output**: Single matrix with dimensions [first_rows × second_cols]
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Mathematical Operation**: A × B where A is m×n and B is n×p, result is m×p
**Use Cases**: Linear transformations, system equations, filter implementations

#### matrix_transpose

**Purpose**: Transposes a matrix (rows become columns, columns become rows)
**Input**: Single matrix of any shape
**Output**: Transposed matrix with swapped dimensions
**Execution Models**: Stream-based and PDU-based
**Supported Types**: All supported scalar types
**Mathematical Operation**: A^T where (A^T)ij = Aji
**Use Cases**: Linear algebra operations, conjugate transpose for complex matrices

#### matrix_trace

**Purpose**: Computes the sum of diagonal elements of a square matrix
**Input**: Square matrix (n×n)
**Output**: Single scalar value
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Mathematical Operation**: tr(A) = Σ Aii
**Use Cases**: Matrix characterization, eigenvalue estimation, system analysis

#### matrix_determinant

**Purpose**: Calculates the determinant of a square matrix
**Input**: Square matrix (n×n)
**Output**: Single scalar value representing the determinant
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Error Handling**: Handles singular matrices and numerical instability
**Use Cases**: Matrix invertibility testing, system solvability, volume scaling

#### matrix_rank

**Purpose**: Computes the rank of a matrix (number of linearly independent rows/columns)
**Input**: Matrix of any shape
**Output**: Single integer representing the rank
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Algorithm**: Uses singular value decomposition with numerical tolerance
**Use Cases**: System analysis, dimensionality assessment, linear independence testing

#### matrix_norm

**Purpose**: Calculates various matrix norms (Frobenius, spectral, etc.)
**Input**: Matrix of any shape
**Output**: Single scalar norm value
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Norm Types**: Frobenius norm, spectral norm, 1-norm, infinity-norm
**Use Cases**: Matrix magnitude measurement, condition number calculation, optimization

#### matrix_inverse

**Purpose**: Computes the inverse of a square matrix
**Input**: Square invertible matrix (n×n)
**Output**: Inverse matrix of same dimensions
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Error Handling**: Detects and reports singular matrices
**Singularity Checks**: Built-in detection of non-invertible matrices
**Use Cases**: System solving, linear transformations, control systems

#### matrix_pseudo_inverse

**Purpose**: Computes the Moore-Penrose pseudoinverse for non-square or singular matrices
**Input**: Matrix of any shape (m×n)
**Output**: Pseudoinverse matrix (n×m)
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Algorithm**: Uses SVD-based computation for numerical stability
**Use Cases**: Least-squares solutions, overdetermined systems, signal processing

#### matrix_source_const

**Purpose**: Generates a constant matrix with specified values
**Input**: Configuration parameters (shape, values)
**Output**: Constant matrix stream
**Execution Models**: Stream-based source block
**Supported Types**: All supported scalar types
**Configuration**: Supports various initialization patterns
**Use Cases**: Signal generation, test matrix creation, system stimulation

### Vector Operations

#### dot_product

**Purpose**: Computes the inner product of two vectors
**Input**: Two vectors of the same length
**Output**: Single scalar value
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Mathematical Operation**: a·b = Σ aibi
**Use Cases**: Correlation calculation, similarity measurement, signal analysis

#### vector_dot

**Purpose**: Alternative implementation of vector dot product
**Input**: Two vectors of equal dimensions
**Output**: Scalar dot product result
**Execution Models**: Stream-based and PDU-based
**Supported Types**: All supported scalar types
**Use Cases**: Signal correlation, pattern matching, vector projections

#### vector_norm

**Purpose**: Calculates various vector norms (L1, L2, L∞)
**Input**: Single vector of any length
**Output**: Single scalar norm value
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Norm Types**: L1 (Manhattan), L2 (Euclidean), L∞ (Maximum)
**Use Cases**: Signal magnitude measurement, distance calculation, normalization

#### vector_normalize

**Purpose**: Normalizes a vector to unit length
**Input**: Single vector of any length
**Output**: Normalized vector with the same dimensions
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Mathematical Operation**: v_norm = v / ||v||
**Error Handling**: Handles zero vectors gracefully
**Use Cases**: Direction vectors, unit vector conversion, signal normalization

#### vector_Kronecker

**Purpose**: Computes the Kronecker product of two vectors
**Input**: Two vectors of any lengths
**Output**: Matrix representing the Kronecker product
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Mathematical Operation**: a ⊗ b creates matrix where (a⊗b)ij = ai × bj
**Use Cases**: Tensor operations, signal processing, system modeling

### Matrix Decompositions

#### decomp_cholesky

**Purpose**: Performs Cholesky decomposition of positive definite matrices
**Input**: Square positive definite matrix
**Output**: Lower triangular matrix L where A = L×L^T
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Variants**: LLT (standard) and LDLT (pivoting) decompositions
**Error Handling**: Detects non-positive-definite matrices
**Use Cases**: System solving, optimization, covariance matrix processing

#### decomp_eigen

**Purpose**: Computes eigenvalues and eigenvectors of a matrix
**Input**: Square matrix (preferably symmetric for real eigenvalues)
**Output**: Eigenvalue vector and eigenvector matrix
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Algorithm Options**: Different solvers for symmetric/general matrices
**Use Cases**: Principal component analysis, stability analysis, modal analysis

#### decomp_lu

**Purpose**: Performs LU factorization with partial pivoting
**Input**: Square matrix (m×m)
**Output**: Lower triangular L, upper triangular U, and permutation matrix P
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Mathematical Result**: P×A = L×U
**Use Cases**: System solving, determinant calculation, matrix inversion

#### decomp_qr

**Purpose**: Computes QR factorization of a matrix
**Input**: Matrix of any shape (m×n)
**Output**: Orthogonal matrix Q and upper triangular matrix R
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Mathematical Result**: A = Q×R where Q^T×Q = I
**Use Cases**: Least-squares problems, orthogonalization, system solving

#### decomp_svd

**Purpose**: Performs Singular Value Decomposition
**Input**: Matrix of any shape (m×n)
**Output**: U matrix, singular values vector, V^T matrix
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Mathematical Result**: A = U×Σ×V^T
**Algorithm Features**: Enhanced algorithm handling, improved shape validation
**Use Cases**: Pseudoinverse calculation, dimensionality reduction, signal processing

### Utility Functions

#### eye

**Purpose**: Generates identity matrices
**Input**: Size parameter (n for n×n matrix)
**Output**: Identity matrix with ones on diagonal, zeros elsewhere
**Execution Models**: Stream-based source block
**Supported Types**: All supported scalar types
**Use Cases**: Identity transformations, initialization, system modeling

#### ones

**Purpose**: Creates matrices filled with ones
**Input**: Shape parameters (rows, columns)
**Output**: Matrix of specified shape with all elements equal to one
**Execution Models**: Stream-based source block
**Supported Types**: All supported scalar types
**Use Cases**: Matrix initialization, uniform weighting, test patterns

#### zeros

**Purpose**: Generates matrices filled with zeros
**Input**: Shape parameters (rows, columns)
**Output**: Zero matrix of specified dimensions
**Execution Models**: Stream-based source block
**Supported Types**: All supported scalar types
**Use Cases**: Matrix initialization, padding, null transformations

## Proposed New Blocks (Not Yet Implemented)

### Advanced Matrix Operations

#### matrix_elementwise_multiply

**Purpose**: Performs element-wise (Hadamard) multiplication of matrices
**Input**: Two or more matrices of the same shape
**Output**: Single matrix with element-wise products
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Mathematical Operation**: (A ⊙ B)ij = Aij × Bij
**Use Cases**: Signal modulation, masking operations, component-wise scaling
**Status**: **NOT IMPLEMENTED**

#### matrix_elementwise_divide

**Purpose**: Performs element-wise division of matrices
**Input**: Two matrices of the same shape (dividend and divisor)
**Output**: Single matrix with element-wise quotients
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Error Handling**: Division by zero detection and handling
**Use Cases**: Signal demodulation, normalization, ratio calculations
**Status**: **NOT IMPLEMENTED**

#### matrix_power

**Purpose**: Raises matrix elements to a power or computes matrix power (A^n)
**Input**: Matrix and power parameter (scalar or matrix for element-wise)
**Output**: Matrix with powered elements or matrix power
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Variants**: Element-wise power, matrix power for square matrices
**Use Cases**: Nonlinear transformations, iterative algorithms, power calculations
**Status**: **NOT IMPLEMENTED**

#### matrix_exponential

**Purpose**: Computes matrix exponential e^A for square matrices
**Input**: Square matrix (n×n)
**Output**: Matrix exponential of same dimensions
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Algorithm**: Uses scaling and squaring with Padé approximation
**Use Cases**: Differential equation solutions, system responses, Lie group operations
**Status**: **NOT IMPLEMENTED**

#### matrix_logarithm

**Purpose**: Computes matrix logarithm for square matrices
**Input**: Square matrix (n×n) with positive eigenvalues
**Output**: Matrix logarithm of same dimensions
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Error Handling**: Handles matrices with negative or zero eigenvalues
**Use Cases**: Matrix scaling, inverse exponential operations, system analysis
**Status**: **NOT IMPLEMENTED**

#### matrix_condition_number

**Purpose**: Computes the condition number of a matrix
**Input**: Square matrix (n×n)
**Output**: Single scalar condition number
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Algorithm**: Uses SVD to compute σ_max/σ_min
**Use Cases**: Numerical stability assessment, system conditioning, optimization
**Status**: **NOT IMPLEMENTED**

#### matrix_reshape

**Purpose**: Reshapes a matrix to new dimensions while preserving total elements
**Input**: Matrix and new shape parameters
**Output**: Matrix with new shape, same total elements
**Execution Models**: Stream-based and PDU-based
**Supported Types**: All supported scalar types
**Ordering**: Row-major and column-major options
**Use Cases**: Data restructuring, dimension manipulation, tensor operations
**Status**: **NOT IMPLEMENTED**

#### matrix_concatenate

**Purpose**: Concatenates matrices along specified axis
**Input**: Multiple matrices and axis parameter (horizontal/vertical)
**Output**: Single concatenated matrix
**Execution Models**: Stream-based and PDU-based
**Supported Types**: All supported scalar types
**Axis Options**: Row concatenation (vertical stack), column concatenation (horizontal stack)
**Use Cases**: Data aggregation, matrix assembly, signal combining
**Status**: **NOT IMPLEMENTED**

#### matrix_split

**Purpose**: Splits a matrix into multiple submatrices
**Input**: Matrix and split parameters (indices or sizes)
**Output**: Multiple matrices from split operation
**Execution Models**: Stream-based and PDU-based
**Supported Types**: All supported scalar types
**Split Types**: Equal splits, custom index splits, block splits
**Use Cases**: Data partitioning, parallel processing, signal separation
**Status**: **NOT IMPLEMENTED**

### Advanced Vector Operations

#### vector_cross_product

**Purpose**: Computes cross product of two 3D vectors
**Input**: Two vectors of length 3
**Output**: Vector of length 3 representing cross product
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Mathematical Operation**: a × b = (a₂b₃-a₃b₂, a₃b₁-a₁b₃, a₁b₂-a₂b₁)
**Use Cases**: 3D rotations, normal vector calculation, angular momentum
**Status**: **NOT IMPLEMENTED**

#### vector_outer_product

**Purpose**: Computes outer product of two vectors
**Input**: Two vectors of any lengths
**Output**: Matrix representing the outer product
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Mathematical Operation**: (a ⊗ b)ij = ai × bj
**Use Cases**: Rank-1 matrix construction, projection matrices, tensor products
**Status**: **NOT IMPLEMENTED**

#### vector_angle

**Purpose**: Computes angle between two vectors
**Input**: Two vectors of the same length
**Output**: Scalar angle in radians
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double (real vectors only)
**Mathematical Operation**: θ = arccos((a·b)/(||a|| ||b||))
**Use Cases**: Signal correlation, geometric analysis, pattern matching
**Status**: **NOT IMPLEMENTED**

#### vector_projection

**Purpose**: Projects one vector onto another
**Input**: Two vectors (vector to project and projection target)
**Output**: Projected vector with same dimensions as target
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Mathematical Operation**: proj_b(a) = ((a·b)/(b·b)) × b
**Use Cases**: Signal decomposition, component analysis, orthogonalization
**Status**: **NOT IMPLEMENTED**

#### vector_gram_schmidt

**Purpose**: Performs Gram-Schmidt orthogonalization on a set of vectors
**Input**: Multiple vectors (matrix with vectors as columns)
**Output**: Orthogonalized vectors (matrix with orthogonal columns)
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Variants**: Classical and modified Gram-Schmidt algorithms
**Use Cases**: Orthogonal basis creation, signal space analysis, beamforming
**Status**: **NOT IMPLEMENTED**

### Specialized Decompositions

#### decomp_schur

**Purpose**: Computes Schur decomposition of a square matrix
**Input**: Square matrix (n×n)
**Output**: Unitary matrix Q and upper triangular matrix T
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Mathematical Result**: A = Q×T×Q^H where T is upper triangular
**Use Cases**: Eigenvalue computation, matrix functions, stability analysis
**Status**: **NOT IMPLEMENTED**

#### decomp_polar

**Purpose**: Computes polar decomposition of a matrix
**Input**: Matrix of any shape (m×n)
**Output**: Unitary factor U and positive semi-definite factor P
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Mathematical Result**: A = U×P where U is unitary, P is positive semi-definite
**Use Cases**: Matrix analysis, closest unitary matrix, mechanical engineering
**Status**: **NOT IMPLEMENTED**

#### decomp_hessenberg

**Purpose**: Reduces a matrix to Hessenberg form
**Input**: Square matrix (n×n)
**Output**: Orthogonal matrix Q and Hessenberg matrix H
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Mathematical Result**: A = Q×H×Q^T where H is upper Hessenberg
**Use Cases**: Eigenvalue algorithms, matrix reduction, numerical analysis
**Status**: **NOT IMPLEMENTED**

#### decomp_jordan

**Purpose**: Computes Jordan canonical form (when possible)
**Input**: Square matrix (n×n)
**Output**: Transformation matrix P and Jordan form J
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Mathematical Result**: A = P×J×P^(-1) where J is Jordan canonical form
**Error Handling**: Handles non-diagonalizable matrices
**Use Cases**: System analysis, differential equations, matrix powers
**Status**: **NOT IMPLEMENTED**

### Statistical and Signal Processing Operations

#### matrix_covariance

**Purpose**: Computes covariance matrix from data matrix
**Input**: Data matrix with observations as rows or columns
**Output**: Covariance matrix
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Options**: Sample vs population covariance, observation axis selection
**Use Cases**: Statistical analysis, principal component analysis, signal processing
**Status**: **NOT IMPLEMENTED**

#### matrix_correlation

**Purpose**: Computes correlation matrix from data matrix
**Input**: Data matrix with observations as rows or columns
**Output**: Correlation matrix (normalized covariance)
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Normalization**: Pearson correlation coefficients
**Use Cases**: Signal correlation, pattern recognition, statistical analysis
**Status**: **NOT IMPLEMENTED**

#### matrix_convolution

**Purpose**: Performs 2D convolution between matrices
**Input**: Two matrices (signal and kernel)
**Output**: Convolved matrix
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Padding Options**: Valid, same, full convolution modes
**Use Cases**: Image processing, filtering, feature detection
**Status**: **NOT IMPLEMENTED**

#### matrix_fft2d

**Purpose**: Computes 2D Fast Fourier Transform of a matrix
**Input**: Matrix (preferably power-of-2 dimensions)
**Output**: 2D FFT transformed matrix
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Options**: Forward/inverse transform, normalization options
**Use Cases**: Spectral analysis, image processing, signal transformation
**Status**: **NOT IMPLEMENTED**

### Optimization and Solver Operations

#### matrix_least_squares

**Purpose**: Solves least squares problems Ax = b
**Input**: Coefficient matrix A and right-hand side vector/matrix b
**Output**: Least squares solution x
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Algorithm**: Uses QR decomposition or SVD for robust solutions
**Use Cases**: Parameter estimation, curve fitting, overdetermined systems
**Status**: **NOT IMPLEMENTED**

#### matrix_solve

**Purpose**: Solves linear system Ax = b for square matrices
**Input**: Coefficient matrix A and right-hand side vector/matrix b
**Output**: Solution vector/matrix x
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Algorithm**: Uses LU decomposition with partial pivoting
**Error Handling**: Detects singular systems
**Use Cases**: System solving, control theory, engineering applications
**Status**: **NOT IMPLEMENTED**

#### matrix_eigenvalues_only

**Purpose**: Computes only eigenvalues (not eigenvectors) for faster computation
**Input**: Square matrix (n×n)
**Output**: Vector of eigenvalues
**Execution Models**: Stream-based and PDU-based
**Supported Types**: float, double, complex<float>, complex<double>
**Performance**: Faster than full eigendecomposition when eigenvectors not needed
**Use Cases**: Stability analysis, spectral radius computation, matrix characterization
**Status**: **NOT IMPLEMENTED**

### Utility and Construction Operations

#### matrix_diag

**Purpose**: Extracts diagonal elements or constructs diagonal matrix
**Input**: Matrix (for extraction) or vector (for construction)
**Output**: Vector (extracted diagonal) or matrix (diagonal matrix)
**Execution Models**: Stream-based and PDU-based
**Supported Types**: All supported scalar types
**Modes**: Extract diagonal, construct diagonal matrix from vector
**Use Cases**: Diagonal manipulation, matrix construction, eigenvalue extraction
**Status**: **NOT IMPLEMENTED**

#### matrix_tril

**Purpose**: Extracts lower triangular part of a matrix
**Input**: Square matrix (n×n)
**Output**: Lower triangular matrix with upper triangle zeroed
**Execution Models**: Stream-based and PDU-based
**Supported Types**: All supported scalar types
**Options**: Include/exclude diagonal, offset parameter
**Use Cases**: Matrix decomposition post-processing, triangular system solving
**Status**: **NOT IMPLEMENTED**

#### matrix_triu

**Purpose**: Extracts upper triangular part of a matrix
**Input**: Square matrix (n×n)
**Output**: Upper triangular matrix with lower triangle zeroed
**Execution Models**: Stream-based and PDU-based
**Supported Types**: All supported scalar types
**Options**: Include/exclude diagonal, offset parameter
**Use Cases**: Matrix decomposition post-processing, triangular system solving
**Status**: **NOT IMPLEMENTED**

#### matrix_flip

**Purpose**: Flips matrix along specified axis (horizontal/vertical)
**Input**: Matrix and flip axis parameter
**Output**: Flipped matrix
**Execution Models**: Stream-based and PDU-based
**Supported Types**: All supported scalar types
**Options**: Horizontal flip (left-right), vertical flip (up-down)
**Use Cases**: Data manipulation, image processing, signal reversal
**Status**: **NOT IMPLEMENTED**

#### matrix_rotate

**Purpose**: Rotates matrix by 90, 180, or 270 degrees
**Input**: Matrix and rotation angle parameter
**Output**: Rotated matrix
**Execution Models**: Stream-based and PDU-based
**Supported Types**: All supported scalar types
**Rotation Angles**: 90°, 180°, 270° clockwise rotations
**Use Cases**: Image processing, coordinate transformations, data orientation
**Status**: **NOT IMPLEMENTED**

#### matrix_block_diagonal

**Purpose**: Constructs block diagonal matrix from multiple input matrices
**Input**: Multiple matrices of any shapes
**Output**: Block diagonal matrix with inputs as diagonal blocks
**Execution Models**: Stream-based and PDU-based
**Supported Types**: All supported scalar types
**Structure**: Creates matrix with input matrices on diagonal, zeros elsewhere
**Use Cases**: System modeling, decoupled systems, structured matrices
**Status**: **NOT IMPLEMENTED**

## Technical Features

### Execution Models

- **Stream Blocks**: Inherit from `linalg_base_sync<T>`, process continuous data streams
- **PDU Blocks**: Inherit from `linalg_base_pdu<T>`, message-based processing with PMT data exchange

### Type Support

- **float**: Single-precision floating point
- **double**: Double-precision floating point
- **complex<float>**: Single-precision complex numbers
- **complex<double>**: Double-precision complex numbers
- **int**: Limited integer support for specific operations

### Broadcasting Policies

- **NONE**: No broadcasting, exact shape matching required
- **VECTOR**: Vector broadcasting to matrix dimensions
- **MATRIX**: Matrix broadcasting for compatible operations
- **CUSTOM**: User-defined broadcasting rules

### Error Handling

- **Tag-based**: Stream tag error indicators for stream blocks
- **PDU-based**: Error message PDUs for message blocks
- **Return codes**: SUCCESS, FAILURE, INVALID_SHAPE, NOT_IMPLEMENTED

### Performance Features

- **Zero-copy operations**: Eigen maps for efficient memory access
- **Template specialization**: Optimized implementations for common types
- **Vectorization**: Leverages Eigen's SIMD capabilities
- **In-place operations**: Minimal memory allocation where possible

## Integration with GNU Radio

All blocks are fully integrated with GNU Radio Companion (GRC) and include:

- Standardized YAML block definitions
- Parameter validation and type checking
- Comprehensive documentation strings
- Categorized under "Linalg" in GRC block tree
- Python bindings support (when enabled)
- Consistent naming conventions following `linalg_<operation>` pattern

This comprehensive block library provides a complete foundation for linear algebra operations in GNU Radio signal processing applications.
