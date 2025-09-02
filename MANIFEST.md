title: The LINALG OOT Module
brief: The `gr-linalg` package is a comprehensive Out-of-Tree (OOT) module for GNU Radio that provides linear algebra operations for digital signal processing applications. Built on the Eigen C++ library, it offers both streaming (sync) and message-based (PDU) execution models to support various GNU Radio workflows.
tags:
  - sdr
  - linear-algebra
  - linalg
  - matrix
  - vector
  - decomposition
  - math
  - solver

author:
  - Wylie Standage-Beier <thewyliestcoyote@gmail.com>
copyright_owner:
  - Wylie Standage-Beier
license:
gr_supported_version: 3.10 tested 
#repo: https://github.com/TheWylieStCoyote/gr-linalg
---
# GNU Radio Linear Algebra (gr-linalg) Package

The `gr-linalg` package is a comprehensive Out-of-Tree (OOT) module for GNU Radio that provides linear algebra operations for digital signal processing applications. Built on the Eigen C++ library, it offers both streaming (sync) and message-based (PDU) execution models to support various GNU Radio workflows.

## Overview

This package bridges the gap between GNU Radio's signal processing capabilities and advanced linear algebra operations commonly needed in communications, radar, beamforming, and adaptive signal processing applications. It provides a unified interface for matrix and vector operations that integrates seamlessly with GNU Radio's block-based processing paradigm.

## Key Features

### Dual Execution Models
- **Synchronous Streaming Blocks**: For continuous signal processing pipelines using `gr::sync_block`
- **PDU Message Blocks**: For discrete matrix/vector processing using GNU Radio's message passing system

### Comprehensive Operation Set
- **Matrix Operations**: Addition, subtraction, multiplication, transpose, inverse, pseudo-inverse, determinant, trace, rank, various norms
- **Vector Operations**: Dot product, cross product, norms, normalization, projection, correlation, outer product
- **Matrix Decompositions**: SVD, Eigenvalue/Eigenvector, LU, QR, Cholesky, Schur decomposition
- **Linear Solvers**: Multiple solver methods (LU, QR, SVD, Cholesky) with automatic method selection
- **Specialized Operations**: Kronecker products, matrix reshaping, condition number calculation

### Data Type Support
- Single precision floating point (`float`)
- Double precision floating point (`double`) 
- Complex single precision (`std::complex<float>`)
- Complex double precision (`std::complex<double>`)
- Integer support for selected operations

### Broadcasting and Shape Handling
- Intelligent shape validation and propagation
- Multiple broadcasting policies (NONE, VECTOR, MATRIX, CUSTOM)
- Automatic output shape computation
- Runtime shape checking with informative error messages

## Architecture

### C++ Core Implementation
- Template-based design for type safety and performance
- Eigen library backend for optimized linear algebra computations
- Hierarchical base classes:
  - `linalg_base<Scalar>`: Common functionality and interface
  - `linalg_base_sync<Scalar>`: Streaming block infrastructure  
  - `linalg_base_pdu<Scalar>`: Message-based block infrastructure

### Python Integration
- Complete Python bindings using pybind11
- GNU Radio Companion (GRC) block definitions
- NumPy-based fallback implementations for testing
- Factory functions following GNU Radio naming conventions

### GRC Integration
- Visual block definitions with appropriate parameter controls
- Type-specific block variants (e.g., `matrix_add_ff`, `matrix_add_dd`)
- Automatic port configuration based on matrix/vector shapes
- Documentation strings and parameter validation

## Use Cases

### Communications Systems
- Channel matrix operations for MIMO systems
- Beamforming weight computation and application
- Equalizer coefficient calculation
- Correlation and covariance matrix processing

### Radar and Sonar
- Direction-of-arrival estimation using eigendecomposition
- Adaptive beamforming using matrix inversions
- Clutter covariance matrix operations
- Space-time adaptive processing (STAP)

### Adaptive Signal Processing
- Least-squares filter adaptation
- Principal component analysis (PCA) via SVD
- Whitening transformations using matrix decompositions
- Subspace-based signal separation

### Machine Learning Integration
- Feature transformation matrices
- Dimensionality reduction operations
- Kernel matrix computations
- Linear discriminant analysis

## Performance Characteristics

- **Optimized Backend**: Leverages Eigen's highly optimized BLAS/LAPACK implementations
- **Memory Efficient**: Zero-copy operations where possible using Eigen's expression templates
- **SIMD Acceleration**: Automatic vectorization on supported architectures
- **Numerical Stability**: Robust algorithms with appropriate tolerance handling
- **Error Handling**: Graceful handling of singular matrices and numerical edge cases

## Package Structure

```
gr-linalg/
├── include/gnuradio/linalg/     # C++ headers
├── lib/                         # C++ implementation
├── python/gnuradio/linalg/      # Python bindings and utilities
│   └── numpy/                   # NumPy fallback implementations
├── grc/                         # GNU Radio Companion block definitions
├── apps/                        # Example applications
├── docs/                        # Documentation
└── examples/                    # Usage examples and integration tests
```

## Dependencies

- **GNU Radio** 3.10+ (3.8+ may work with minor modifications)
- **Eigen3** 3.3+ (header-only C++ linear algebra library)
- **CMake** 3.8+ (build system)
- **Boost** (for unit testing and some utilities)
- **Python** 3.6+ (for bindings and GRC)
- **NumPy** (for Python fallbacks and testing)
- **pybind11** (for Python-C++ bindings)

## Installation

The package uses standard CMake build procedures:

```bash
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
make -j$(nproc)
sudo make install
sudo ldconfig  # Update linker cache
```

## Quality Assurance

- **Comprehensive Test Suite**: Unit tests for all operations using Boost.Test
- **Numerical Validation**: Cross-validation with NumPy and SciPy implementations
- **Edge Case Testing**: Singular matrices, empty inputs, numerical corner cases
- **Integration Tests**: End-to-end GNU Radio flowgraph testing
- **Performance Benchmarks**: Timing comparisons with reference implementations

## Future Roadmap

- GPU acceleration via CUDA/OpenCL backends
- Sparse matrix support for large-scale problems
- Extended decomposition algorithms (polar, Jordan normal form)
- Advanced iterative solvers for large systems
- Integration with GNU Radio's scheduler for optimal threading

This package represents a significant enhancement to GNU Radio's mathematical capabilities, enabling sophisticated linear algebra workflows within the familiar GNU Radio ecosystem while maintaining the performance and reliability required for real-time signal processing applications.

