# Matrix Norm Block Documentation

## Overview

The `matrix_norm` block computes various matrix norms for input matrices in GNU Radio flowgraphs. It supports multiple norm types and data types, making it useful for signal processing applications that require matrix analysis, condition number estimation, and signal power monitoring.

## Features

- **Multiple Norm Types**: L1, L2 (spectral), Frobenius, and Max norms
- **Multiple Data Types**: float, double, complex float, complex double
- **High Performance**: Optimized using Eigen library for matrix operations
- **Streaming Support**: Processes continuous streams of matrices
- **GNU Radio Integration**: Seamlessly integrates with other GNU Radio blocks

## Block Variants

### Sync Blocks
- `matrix_norm_sync_f` - Float (32-bit) matrices
- `matrix_norm_sync_d` - Double (64-bit) matrices  
- `matrix_norm_sync_c` - Complex float matrices
- `matrix_norm_sync_cd` - Complex double matrices

### Factory Functions
- `matrix_norm_ff(shape, norm=None)` - Float convenience function
- `matrix_norm_dd(shape, norm=None)` - Double convenience function
- `matrix_norm_cc(shape, norm=None)` - Complex float convenience function
- `matrix_norm_zz(shape, norm=None)` - Complex double convenience function
- `matrix_norm(shape, norm=None)` - Generic function (defaults to double)

## Parameters

### Constructor Parameters

1. **shape** (required): `[rows, cols]`
   - Input matrix dimensions
   - Example: `[3, 3]` for 3×3 matrices

2. **norm** (optional): `norm_type` enum
   - `norm_type.L1` - L1 norm (maximum column sum)
   - `norm_type.L2` - L2 norm (spectral norm, largest singular value)
   - `norm_type.Frobenius` - Frobenius norm (Euclidean norm)
   - `norm_type.Max` - Max norm (maximum absolute value)
   - Default: `norm_type.Frobenius`

## Input/Output

### Input
- **Type**: Matrix data as flattened vectors
- **Size**: `rows × cols` elements per matrix
- **Format**: Row-major order (C-style)

### Output  
- **Type**: Scalar values (same precision as input)
- **Size**: 1 element per input matrix
- **Content**: Computed norm value

## Norm Types Explained

### L1 Norm (`norm_type.L1`)
- **Definition**: Maximum column sum
- **Formula**: `max_j(Σ_i |a_ij|)`
- **Use Cases**: Matrix condition estimation, optimization bounds

### L2 Norm (`norm_type.L2`) 
- **Definition**: Spectral norm (largest singular value)
- **Formula**: `σ_max(A)` where σ_max is the largest singular value
- **Use Cases**: System stability analysis, control theory

### Frobenius Norm (`norm_type.Frobenius`)
- **Definition**: Euclidean norm of matrix elements
- **Formula**: `sqrt(Σ_i Σ_j |a_ij|²)`
- **Use Cases**: Signal power measurement, error quantification

### Max Norm (`norm_type.Max`)
- **Definition**: Maximum absolute value of any element
- **Formula**: `max_ij(|a_ij|)`
- **Use Cases**: Peak detection, clipping analysis

## Usage Examples

### Basic Usage

```python
from gnuradio import gr, blocks
from gnuradio.linalg import matrix_norm_sync_d, norm_type

# Create flowgraph
tb = gr.top_block()

# Generate test matrix data
matrix_data = [1.0, 2.0, 3.0, 4.0]  # 2x2 matrix: [[1,2],[3,4]]

# Create blocks
src = blocks.vector_source_d(matrix_data, False, 4)
norm_block = matrix_norm_sync_d([2, 2], norm_type.Frobenius)
sink = blocks.vector_sink_d(1)

# Connect and run
tb.connect(src, norm_block, sink)
tb.run()

# Get result
result = sink.data()[0]
print(f"Frobenius norm: {result}")  # Should be ~5.477
```

### Signal Processing Application

```python
# Covariance matrix analysis for beamforming
covariance_matrix = [
    4.0, 1.5, 0.8,
    1.5, 3.2, 1.1, 
    0.8, 1.1, 2.5
]

# Analyze matrix conditioning
l1_norm_block = matrix_norm_sync_f([3, 3], norm_type.L1)
frobenius_norm_block = matrix_norm_sync_f([3, 3], norm_type.Frobenius)

# Use norms to estimate condition number and signal quality
```

### Multiple Matrices Stream

```python
# Process stream of channel matrices
channel_data = [
    # Matrix 1: Strong channel
    2.0, 0.5, 0.3, 1.8,
    # Matrix 2: Weak channel  
    0.8, 0.2, 0.1, 0.6,
    # Matrix 3: Moderate channel
    1.2, 0.4, 0.6, 1.0
]

src = blocks.vector_source_f(channel_data, False, 4)  # 4 elements per 2x2 matrix
norm_block = matrix_norm_sync_f([2, 2], norm_type.Frobenius)
sink = blocks.vector_sink_f(1)

# Will output 3 norm values, one for each matrix
```

## Performance Characteristics

### Computational Complexity
- **L1 Norm**: O(mn) - Linear in matrix size
- **Frobenius Norm**: O(mn) - Linear in matrix size  
- **Max Norm**: O(mn) - Linear in matrix size
- **L2 Norm**: O(min(m,n)³) - Cubic due to SVD computation

### Memory Requirements
- Input buffer: `rows × cols × sizeof(datatype)`
- Output buffer: `sizeof(datatype)` 
- Working memory: Minimal (in-place operations where possible)

### Throughput
- Typical performance: >1000 matrices/second for 3×3 float matrices
- Performance scales with matrix size and norm type
- L2 norm is most computationally expensive due to SVD

## Applications

### 1. Signal Power Monitoring
```python
# Monitor MIMO channel matrix condition
channel_norm = matrix_norm_sync_f([4, 4], norm_type.Frobenius)
# High norm indicates strong channel, low norm indicates weak channel
```

### 2. Condition Number Estimation
```python
# Estimate matrix conditioning for numerical stability
l1_norm = matrix_norm_sync_d([n, n], norm_type.L1)
# Condition number ≈ ||A||₁ × ||A⁻¹||₁
```

### 3. Beamforming Weight Analysis
```python
# Analyze beamforming weight vectors
weight_norm = matrix_norm_sync_f([num_antennas, 1], norm_type.Frobenius)
# Monitor beam power and steering effectiveness
```

### 4. Adaptive Filtering
```python
# Monitor filter coefficient evolution
coeff_norm = matrix_norm_sync_d([filter_length, 1], norm_type.L2)
# Detect filter convergence and stability
```

## Error Handling

### Invalid Inputs
- Zero or negative matrix dimensions will cause construction failure
- Non-finite input values (NaN, Inf) may produce undefined results
- Mismatched vector lengths will cause runtime errors

### Numerical Considerations
- Very large matrices may cause overflow in norm computations
- Very small values may underflow to zero
- Complex matrices: norm is always real-valued

## Integration with Other Blocks

### Compatible Source Blocks
- `vector_source_*` - For test data generation
- `matrix_multiply_sync_*` - Chained matrix operations
- `matrix_add_sync_*` - Matrix preprocessing
- Custom signal sources producing matrix data

### Compatible Sink Blocks  
- `vector_sink_*` - For result collection
- `file_sink` - For logging norm values
- `probe_signal_*` - For real-time monitoring
- Custom analysis blocks

### Flowgraph Patterns
```python
# Pattern 1: Matrix processing chain
source → matrix_add → matrix_norm → sink

# Pattern 2: Parallel norm analysis  
source → ┬─ matrix_norm(L1) → sink1
         ├─ matrix_norm(L2) → sink2  
         └─ matrix_norm(Frobenius) → sink3

# Pattern 3: Condition monitoring
matrix_source → matrix_norm → threshold → alarm
```

## Troubleshooting

### Common Issues

1. **"Block creation failed"**
   - Check matrix dimensions are positive integers
   - Verify norm_type is valid enum value
   - Ensure C++ bindings are properly compiled

2. **"Unexpected norm values"**
   - Verify input data is in row-major order
   - Check matrix dimensions match actual data size
   - Consider numerical precision limitations

3. **"Performance issues"**
   - L2 norm is computationally expensive for large matrices
   - Consider using Frobenius norm for power monitoring
   - Profile with smaller matrices first

### Debug Tips

```python
# Enable verbose output
import logging
logging.basicConfig(level=logging.DEBUG)

# Test with known matrices
identity = [1,0,0,1]  # 2x2 identity, Frobenius norm = sqrt(2)
ones = [1,1,1,1]      # 2x2 ones, Frobenius norm = 2

# Verify against NumPy
import numpy as np
test_matrix = np.array([[1,2],[3,4]])
expected_frobenius = np.linalg.norm(test_matrix, 'fro')
```

## See Also

- [Matrix Operations Documentation](matrix_operations.md)
- [Linear Algebra Block Reference](linalg_reference.md)
- [GNU Radio Block Development Guide](https://wiki.gnuradio.org/index.php/OutOfTreeModules)
- [Eigen Library Documentation](https://eigen.tuxfamily.org/)

## Version History

- **v1.0** - Initial implementation with basic norm types
- **v1.1** - Added complex data type support
- **v1.2** - Performance optimizations and comprehensive testing
- **Current** - Full integration with GNU Radio linalg module