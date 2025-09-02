# GNU Radio Linear Algebra Performance Profiler CLI

A comprehensive command-line tool for performance testing and benchmarking linear algebra blocks in GNU Radio.

## Features

- **Automated Performance Testing**: Test all linear algebra block categories with configurable parameters
- **Multiple Test Modes**: Quick, comprehensive, stress testing, and custom configurations  
- **Memory Profiling**: Monitor memory usage during block execution
- **Multi-precision Support**: Test float, double, complex float, and complex double precision
- **Detailed Reporting**: Console, JSON, CSV, and HTML output formats with visualizations
- **Statistical Analysis**: Min/max/average execution times and throughput calculations

## Installation

1. Ensure GNU Radio and gr-linalg are properly built and installed
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start

```bash
# Run quick performance test with console output
python linalg_profiler_cli.py --test quick

# Run comprehensive test with JSON output
python linalg_profiler_cli.py --test comprehensive --format json --verbose

# Test specific block with custom sizes
python linalg_profiler_cli.py --block matrix_multiply --sizes 32x32,64x64 --precision f,d
```

### Test Configurations

- **quick**: Basic blocks, small sizes, 5 iterations (~30 seconds)
- **comprehensive**: All blocks, multiple sizes, 10 iterations (~5 minutes)
- **stress**: Large matrices, advanced blocks, 20 iterations (~10 minutes)
- **generators**: Matrix generator blocks only (~2 minutes)
- **custom**: User-defined configuration

## Examples

### Basic Usage

```bash
# List all available blocks
python linalg_profiler_cli.py --list-blocks

# Quick test with verbose output
python linalg_profiler_cli.py --test quick --verbose

# Test matrix operations only
python linalg_profiler_cli.py --category matrix_basic --format html
```

### Advanced Usage

```bash
# Custom test with specific parameters
python linalg_profiler_cli.py \
  --test custom \
  --block matrix_multiply \
  --sizes 16x16,32x32,64x64 \
  --precision f,d \
  --iterations 20 \
  --format json \
  --output results/
```

## Output Formats

- **Console**: Summary statistics, performance by block type, top performers
- **JSON**: Machine-readable format with complete test results
- **CSV**: Spreadsheet-compatible with one row per test result
- **HTML**: Visual report with charts and interactive results

## Block Categories

- **matrix_basic**: add, subtract, multiply, transpose, determinant, trace
- **matrix_advanced**: inverse, norm, condition_number, solve, exp, etc.
- **matrix_elementwise**: elementwise operations, diag, power
- **matrix_generators**: eye, ones, zeros, source_const
- **vector_operations**: norm, cross_product, dot_product, correlate
- **decompositions**: SVD, QR, LU, Cholesky, Schur, Hessenberg
- **solvers**: linear system solvers, least squares, iterative methods

## License

Same as GNU Radio - GPL v3 or later