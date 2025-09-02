# Linear Algebra Operations for GNU Radio OOT Module

This document outlines the linear algebra operations that would be beneficial to implement in the `gr-linalg` GNU Radio Out-of-Tree (OOT) module. Operations are categorized by functionality and priority.

## **Vector Operations**

### Basic Vector Operations

- [x] **dot_product** - Inner product of vectors (already implemented)
  - Computes the sum of element-wise products of two vectors
  - Essential for correlation, energy calculations, and projections
  - **Use cases**: Signal correlation, matched filtering, energy detection

- [ ] **vector_norm** - L1, L2, L∞ norms
  - L1: Sum of absolute values, L2: Euclidean norm, L∞: Maximum absolute value
  - **Use cases**: Signal power calculation, distance metrics, normalization

- [ ] **vector_normalize** - Normalize vectors to unit length
  - Scales vector to have unit norm (typically L2 norm)
  - **Use cases**: Direction vectors, normalized correlation

- [ ] **vector_add** - Element-wise vector addition
  - Adds corresponding elements of two vectors
  - **Use cases**: Signal combination, superposition

- [ ] **vector_subtract** - Element-wise vector subtraction
  - Subtracts corresponding elements of two vectors
  - **Use cases**: Error calculation, signal differencing

- [ ] **vector_scale** - Scalar multiplication of vectors
  - Multiplies each element by a scalar value
  - **Use cases**: Gain application, amplitude scaling

### Advanced Vector Operations

- [ ] **vector_angle** - Angle between two vectors
  - Computes the angle using dot product and norms
  - **Use cases**: Signal similarity, phase relationships

- [ ] **vector_projection** - Project one vector onto another
  - Projects vector a onto vector b: proj_b(a) = (a·b/|b|²)b
  - **Use cases**: Signal decomposition, component extraction

- [ ] **outer_product** - Outer product of two vectors
  - Creates a matrix from two vectors: A_ij = a_i * b_j
  - **Use cases**: Covariance matrices, rank-1 updates

## **Matrix Operations**

### Basic Matrix Operations

- [ ] **matrix_multiply** - Standard matrix multiplication (A × B)
  - Fundamental linear algebra operation
  - **Use cases**: Linear transformations, filter banks, beamforming

- [ ] **matrix_add** - Element-wise matrix addition
  - Adds corresponding elements of two matrices
  - **Use cases**: Combining transformations, superposition

- [ ] **matrix_subtract** - Element-wise matrix subtraction
  - Subtracts corresponding elements of two matrices
  - **Use cases**: Error matrices, difference calculations

- [ ] **matrix_transpose** - Matrix transposition
  - Swaps rows and columns: (A^T)_ij = A_ji
  - **Use cases**: Hermitian operations, conjugate transpose

- [ ] **matrix_inverse** - Matrix inversion (with singularity checks)
  - Computes A^(-1) such that A * A^(-1) = I
  - **Use cases**: Solving linear systems, adaptive filtering

### Matrix Properties

- [ ] **matrix_determinant** - Compute determinant
  - Scalar value characterizing the matrix
  - **Use cases**: Singularity detection, volume scaling

- [ ] **matrix_trace** - Sum of diagonal elements
  - tr(A) = Σ A_ii
  - **Use cases**: Matrix characterization, optimization

- [ ] **matrix_rank** - Matrix rank computation
  - Number of linearly independent rows/columns
  - **Use cases**: System analysis, dimensionality assessment

- [ ] **matrix_norm** - Frobenius, spectral, nuclear norms
  - Different ways to measure matrix "size"
  - **Use cases**: Optimization, convergence criteria

- [ ] **hadamard_product** - Element-wise matrix multiplication
  - Multiplies corresponding elements: (A ⊙ B)_ij = A_ij * B_ij
  - **Use cases**: Masking, windowing, element-wise operations

## **Matrix Decompositions**

- [ ] **svd** - Singular Value Decomposition
  - Decomposes A = UΣV^T where U, V are orthogonal and Σ is diagonal
  - **Use cases**: Principal component analysis, low-rank approximation, noise reduction

- [ ] **eigen_decomposition** - Eigenvalues and eigenvectors
  - Finds λ and v such that Av = λv
  - **Use cases**: System stability, modal analysis, spectral analysis

- [ ] **lu_decomposition** - LU factorization
  - Decomposes A = LU where L is lower triangular, U is upper triangular
  - **Use cases**: Efficient linear system solving

- [ ] **qr_decomposition** - QR factorization
  - Decomposes A = QR where Q is orthogonal, R is upper triangular
  - **Use cases**: Least squares, Gram-Schmidt orthogonalization

- [ ] **cholesky_decomposition** - For positive definite matrices
  - Decomposes A = LL^T for positive definite A
  - **Use cases**: Efficient solving for covariance matrices

- [ ] **schur_decomposition** - Schur decomposition
  - Decomposes A = QTQ^T where Q is orthogonal, T is upper triangular
  - **Use cases**: Eigenvalue computation, matrix functions

## **Linear Systems**

- [ ] **solve_linear_system** - Solve Ax = b
  - Finds x given A and b
  - **Use cases**: Parameter estimation, system inversion

- [ ] **least_squares** - Overdetermined system solver
  - Minimizes ||Ax - b||² for overdetermined systems
  - **Use cases**: Parameter fitting, regression

- [ ] **conjugate_gradient** - Iterative solver for large sparse systems
  - Efficient iterative method for symmetric positive definite systems
  - **Use cases**: Large-scale problems, sparse matrices

- [ ] **gauss_seidel** - Iterative solver method
  - Iterative method for solving linear systems
  - **Use cases**: Large systems, numerical stability

- [ ] **jacobi_method** - Another iterative solver
  - Simple iterative method for linear systems
  - **Use cases**: Parallel computation, educational purposes

## **Signal Processing Specific**

### Correlation and Covariance

- [ ] **covariance_matrix** - Sample covariance estimation
  - Estimates Cov(X,Y) = E[(X-μ_X)(Y-μ_Y)^T]
  - **Use cases**: Statistical signal processing, adaptive algorithms

- [ ] **correlation_matrix** - Cross-correlation matrix
  - Computes correlation between different signals/channels
  - **Use cases**: MIMO processing, array signal processing

- [ ] **autocorrelation** - Autocorrelation function
  - Correlation of signal with delayed version of itself
  - **Use cases**: Spectral analysis, periodicity detection

### Advanced Signal Processing

- [ ] **whitening_transform** - Data whitening/sphering
  - Transforms data to have identity covariance matrix
  - **Use cases**: Preprocessing, decorrelation

- [ ] **principal_components** - PCA transformation
  - Finds principal components using eigendecomposition
  - **Use cases**: Dimensionality reduction, feature extraction

- [ ] **beamforming_weights** - Array processing weights
  - Computes optimal weights for antenna arrays
  - **Use cases**: Spatial filtering, interference rejection

- [ ] **adaptive_filter_weights** - LMS, RLS filter coefficients
  - Adaptive algorithms for filter coefficient updates
  - **Use cases**: Echo cancellation, equalization

- [ ] **toeplitz_matrix** - For convolution operations
  - Special matrix structure for convolution
  - **Use cases**: FIR filtering, circular convolution

## **Utility Functions** (extend existing utils.h)

### Already Implemented ✅

- [x] **product** - Product of vector elements
- [x] **sum** - Sum of vector elements
- [x] **mean** - Mean of vector elements
- [x] **variance** - Variance of vector elements
- [x] **std_dev** - Standard deviation
- [x] **linspace** - Linearly spaced vectors
- [x] **zeros** - Zero-filled vectors
- [x] **ones** - One-filled vectors
- [x] **eye** - Identity matrix

### To Be Implemented

- [ ] **arange** - Range of values with step
  - Similar to Python's arange: start, stop, step
  - **Use cases**: Index generation, sampling

- [ ] **logspace** - Logarithmically spaced vectors
  - Values spaced logarithmically between limits
  - **Use cases**: Frequency sweeps, logarithmic scales

- [ ] **meshgrid** - Create coordinate matrices
  - Creates coordinate matrices from coordinate vectors
  - **Use cases**: 2D/3D function evaluation

- [ ] **reshape** - Reshape vectors/matrices
  - Changes dimensions while preserving elements
  - **Use cases**: Data reorganization, matrix operations

- [ ] **flatten** - Flatten matrices to vectors
  - Converts matrix to 1D vector
  - **Use cases**: Vectorization, data serialization

- [ ] **min_element** - Find minimum element in vector
  - Returns minimum value and optionally its index
  - **Use cases**: Peak detection, threshold finding

- [ ] **max_element** - Find maximum element in vector
  - Returns maximum value and optionally its index
  - **Use cases**: Peak detection, maximum likelihood

- [ ] **argmin** - Index of minimum element
  - Returns index of minimum element
  - **Use cases**: Decision making, optimization

- [ ] **argmax** - Index of maximum element
  - Returns index of maximum element
  - **Use cases**: Classification, peak finding

## **Advanced Operations**

- [ ] **tensor_product** - Kronecker/tensor products
  - Generalization of outer product to matrices
  - **Use cases**: MIMO systems, multidimensional processing

- [ ] **matrix_exponential** - Matrix exponential function
  - Computes e^A using series expansion or eigendecomposition
  - **Use cases**: System dynamics, differential equations

- [ ] **matrix_logarithm** - Matrix logarithm
  - Inverse of matrix exponential
  - **Use cases**: Matrix analysis, logarithmic scales

- [ ] **pseudoinverse** - Moore-Penrose pseudoinverse
  - Generalized inverse for non-square matrices
  - **Use cases**: Least squares, overdetermined systems

- [ ] **condition_number** - Matrix conditioning assessment
  - Ratio of largest to smallest singular values
  - **Use cases**: Numerical stability assessment

- [ ] **matrix_power** - Matrix raised to integer power
  - Computes A^n efficiently
  - **Use cases**: Iterative systems, Markov chains

## **Complex Number Support**

- [ ] **complex_conjugate** - Complex conjugate of matrices/vectors
  - Element-wise complex conjugation
  - **Use cases**: Hermitian operations, complex signal processing

- [ ] **complex_to_real** - Convert complex to real representation
  - Converts complex matrix to real equivalent representation
  - **Use cases**: Real-valued algorithms on complex data

- [ ] **real_to_complex** - Convert real to complex
  - Converts real matrix to complex representation
  - **Use cases**: Complex signal generation

- [ ] **hermitian_transpose** - Conjugate transpose
  - Combines transpose and complex conjugate
  - **Use cases**: Complex matrix operations, MIMO

- [ ] **complex_magnitude** - Magnitude of complex vectors
  - Computes |z| for complex elements
  - **Use cases**: Signal power, amplitude processing

- [ ] **complex_phase** - Phase of complex vectors
  - Computes arg(z) for complex elements
  - **Use cases**: Phase processing, frequency estimation

## **GNU Radio Specific Utilities**

- [ ] **block_toeplitz** - For convolution matrices
  - Special structured matrices for block processing
  - **Use cases**: Block-based filtering, circular convolution

- [ ] **circulant_matrix** - Circular convolution matrices
  - Matrices for circular convolution operations
  - **Use cases**: FFT-based filtering, periodic signals

- [ ] **dft_matrix** - Discrete Fourier Transform matrix
  - Matrix representation of DFT operation
  - **Use cases**: Spectral analysis, frequency domain processing

- [ ] **window_functions** - Hamming, Hanning, etc.
  - Common windowing functions for spectral analysis
  - **Use cases**: Spectral estimation, filtering

- [ ] **filter_matrix** - FIR filter as matrix operation
  - Represents FIR filtering as matrix multiplication
  - **Use cases**: Block processing, parallel filtering

- [ ] **steering_vector** - Array steering vectors
  - Vectors for steering antenna arrays
  - **Use cases**: Beamforming, direction finding

- [ ] **spatial_smoothing** - For DOA estimation
  - Spatial smoothing for coherent signal processing
  - **Use cases**: Direction of arrival estimation

## **Statistical Operations**

- [ ] **covariance** - Covariance between two vectors
  - Measures linear relationship between variables
  - **Use cases**: Signal correlation, statistical analysis

- [ ] **correlation_coefficient** - Pearson correlation
  - Normalized covariance (-1 to 1 range)
  - **Use cases**: Signal similarity, dependence measurement

- [ ] **median** - Median of vector elements
  - Middle value when sorted
  - **Use cases**: Robust statistics, outlier rejection

- [ ] **percentile** - Percentile calculation
  - Value below which a percentage of data falls
  - **Use cases**: Statistical analysis, threshold setting

- [ ] **histogram** - Histogram computation
  - Frequency distribution of values
  - **Use cases**: Probability density estimation, data analysis

## **Implementation Priority**

### **Tier 1 (High Priority)** - Foundation operations

Essential for basic linear algebra functionality:

- `matrix_multiply`, `matrix_transpose`, `vector_norm`
- `matrix_inverse`, `solve_linear_system`
- `vector_add`, `vector_subtract`, `vector_scale`

### **Tier 2 (Medium Priority)** - Common signal processing

Important for typical GNU Radio applications:

- `covariance_matrix`, `correlation_matrix`
- `svd`, `eigen_decomposition`
- `matrix_determinant`, `matrix_trace`
- `least_squares`, `pseudoinverse`

### **Tier 3 (Low Priority)** - Advanced/specialized

Useful for specialized applications:

- Sparse operations, optimization functions
- Advanced decompositions, tensor operations
- Statistical functions, complex utilities

## **Development Guidelines**

1. **Use Eigen library** for efficient implementations
2. **Template functions** for type flexibility (float, double, complex)
3. **Error checking** for dimension mismatches and invalid operations
4. **Unit tests** for each operation
5. **Documentation** with usage examples
6. **GNU Radio block wrappers** for stream processing applications

This roadmap provides a comprehensive foundation for developing a robust linear algebra toolkit specifically designed for GNU Radio signal processing applications.
