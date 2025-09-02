#
# Copyright 2008,2009 Free Software Foundation, Inc.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

# The presence of this file turns this directory into a Python package

"""
This is the GNU Radio LINALG module. Place your Python package
description here (python/__init__.py).
"""
import os

# import pybind11 generated symbols into the linalg namespace
try:
    # this might fail if the module is python-only
    from .linalg_python import *

    print("Successfully imported from .linalg_python")
except (ModuleNotFoundError, ImportError) as e:
    print(f"Primary import failed: {e}")
    # Try alternate import paths
    try:
        import linalg_python
        from linalg_python import *

        print("Successfully imported linalg_python directly")
    except (ModuleNotFoundError, ImportError):
        try:
            import sys
            import os

            # Get current directory for fallback
            current_dir = os.path.dirname(
                os.path.abspath(__file__ if "__file__" in globals() else ".")
            )

            # Try various paths where the module might be
            root_dir = os.path.dirname(
                os.path.dirname(current_dir)
            )  # Go up to project root
            possible_paths = [
                current_dir,
                os.path.join(current_dir, "bindings"),
                os.path.join(root_dir, "build", "test_modules", "gnuradio", "linalg"),
                os.path.join(root_dir, "build", "python", "linalg", "bindings"),
                ".",  # Current working directory as fallback
            ]

            # Try each path individually
            linalg_python_imported = False
            for path in possible_paths:
                if path not in sys.path:
                    sys.path.insert(0, path)
                try:
                    import linalg_python
                    from linalg_python import *

                    print(f"Successfully imported linalg_python from path: {path}")
                    linalg_python_imported = True
                    break
                except ImportError:
                    continue

            if not linalg_python_imported:
                # Final attempt with all paths added
                from linalg_python import *

                print("Successfully imported linalg_python from alternate path")
        except Exception as e2:
            print(f"Warning: Could not import linalg_python: {e2}")
            # Continue without C++ bindings - only convenience functions will be available
            pass

# GNU Radio convention: Provide convenient factory functions for common data types
# This follows the pattern used in gnuradio.blocks (e.g., add_ff, add_cc, etc.)


def matrix_add_ff(shape, num_inputs=2):
    """Create a matrix addition block for float data.

    Args:
        shape: Matrix shape [rows, cols]
        num_inputs: Number of input matrices to add (default: 2)

    Returns:
        matrix_add_sync_f block instance
    """
    return matrix_add_sync_f(shape, num_inputs)


def matrix_add_dd(shape, num_inputs=2):
    """Create a matrix addition block for double data.

    Args:
        shape: Matrix shape [rows, cols]
        num_inputs: Number of input matrices to add (default: 2)

    Returns:
        matrix_add_sync_d block instance
    """
    return matrix_add_sync_d(shape, num_inputs)


def matrix_subtract_ff(shape, num_inputs=2):
    """Create a matrix subtraction block for float data."""
    return matrix_subtract_sync_f(shape, num_inputs)


def matrix_subtract_dd(shape, num_inputs=2):
    """Create a matrix subtraction block for double data."""
    return matrix_subtract_sync_d(shape, num_inputs)


def matrix_elementwise_multiply_ff(shape, num_inputs=2):
    """Create a matrix elementwise multiplication block for float data."""
    return matrix_elementwise_multiply_sync(shape, num_inputs)


def matrix_elementwise_multiply_dd(shape, num_inputs=2):
    """Create a matrix elementwise multiplication block for double data."""
    return matrix_elementwise_multiply_sync_d(shape, num_inputs)


def matrix_elementwise_multiply_cc(shape, num_inputs=2):
    """Create a matrix elementwise multiplication block for complex float data."""
    return matrix_elementwise_multiply_sync_cf(shape, num_inputs)


def matrix_elementwise_multiply_zz(shape, num_inputs=2):
    """Create a matrix elementwise multiplication block for complex double data."""
    return matrix_elementwise_multiply_sync_cd(shape, num_inputs)


def matrix_elementwise_divide_ff(shape, num_inputs=2):
    """Create a matrix elementwise division block for float data."""
    if "matrix_elementwise_divide_sync" in globals():
        return matrix_elementwise_divide_sync(shape, num_inputs)
    else:
        raise NotImplementedError("matrix_elementwise_divide_sync not available")


def matrix_elementwise_divide_dd(shape, num_inputs=2):
    """Create a matrix elementwise division block for double data."""
    if "matrix_elementwise_divide_sync_d" in globals():
        return matrix_elementwise_divide_sync_d(shape, num_inputs)
    else:
        raise NotImplementedError("matrix_elementwise_divide_sync_d not available")


def matrix_elementwise_divide_cc(shape, num_inputs=2):
    """Create a matrix elementwise division block for complex float data."""
    if "matrix_elementwise_divide_sync_cf" in globals():
        return matrix_elementwise_divide_sync_cf(shape, num_inputs)
    else:
        raise NotImplementedError("matrix_elementwise_divide_sync_cf not available")


def matrix_elementwise_divide_zz(shape, num_inputs=2):
    """Create a matrix elementwise division block for complex double data."""
    if "matrix_elementwise_divide_sync_cd" in globals():
        return matrix_elementwise_divide_sync_cd(shape, num_inputs)
    else:
        raise NotImplementedError("matrix_elementwise_divide_sync_cd not available")


def matrix_transpose_ff(shape):
    """Create a matrix transpose block for float data."""
    return matrix_transpose_sync_f(shape)


def matrix_transpose_dd(shape):
    """Create a matrix transpose block for double data."""
    return matrix_transpose_sync_d(shape)


def matrix_determinant_ff(shape):
    """Create a matrix determinant block for float data."""
    return matrix_determinant_sync_f(shape)


def matrix_determinant_dd(shape):
    """Create a matrix determinant block for double data."""
    return matrix_determinant_sync_d(shape)


# Note: matrix_power blocks have different constructor signatures
# These blocks take (matrix_mode, validate_square) not (shape, power)
def matrix_power_ff(matrix_mode=False, validate_square=True):
    """Create a matrix power block for float data.

    Args:
        matrix_mode: True for matrix power, False for element-wise power (default)
        validate_square: Validate square matrices in matrix mode (default: True)

    Returns:
        matrix_power_sync block instance
    """
    return matrix_power_sync(matrix_mode, validate_square)


def matrix_power_dd(matrix_mode=False, validate_square=True):
    """Create a matrix power block for double data.

    Args:
        matrix_mode: True for matrix power, False for element-wise power (default)
        validate_square: Validate square matrices in matrix mode (default: True)

    Returns:
        matrix_power_sync_d block instance
    """
    return matrix_power_sync_d(matrix_mode, validate_square)


def matrix_power_cc(matrix_mode=False, validate_square=True):
    """Create a matrix power block for complex float data.

    Args:
        matrix_mode: True for matrix power, False for element-wise power (default)
        validate_square: Validate square matrices in matrix mode (default: True)

    Returns:
        matrix_power_sync_cf block instance
    """
    return matrix_power_sync_cf(matrix_mode, validate_square)


def matrix_power_zz(matrix_mode=False, validate_square=True):
    """Create a matrix power block for complex double data.

    Args:
        matrix_mode: True for matrix power, False for element-wise power (default)
        validate_square: Validate square matrices in matrix mode (default: True)

    Returns:
        matrix_power_sync_cd block instance
    """
    return matrix_power_sync_cd(matrix_mode, validate_square)


def matrix_condition_number_ff(shape):
    """Create a matrix condition number block for float data.

    Args:
        shape: Input matrix shape [rows, cols]

    Returns:
        matrix_condition_number_sync_f block instance
    """
    if "matrix_condition_number_sync_f" in globals():
        return matrix_condition_number_sync_f.make(shape)
    else:
        raise RuntimeError(
            "matrix_condition_number_sync_f C++ class not available in Python bindings"
        )


def matrix_condition_number_dd(shape):
    """Create a matrix condition number block for double data.

    Args:
        shape: Input matrix shape [rows, cols]

    Returns:
        matrix_condition_number_sync_d block instance
    """
    if "matrix_condition_number_sync_d" in globals():
        return matrix_condition_number_sync_d.make(shape)
    else:
        raise RuntimeError(
            "matrix_condition_number_sync_d C++ class not available in Python bindings"
        )


def matrix_condition_number_cc(shape):
    """Create a matrix condition number block for complex float data.

    Args:
        shape: Input matrix shape [rows, cols]

    Returns:
        matrix_condition_number_sync_cf block instance
    """
    if "matrix_condition_number_sync_cf" in globals():
        return matrix_condition_number_sync_cf.make(shape)
    else:
        raise RuntimeError(
            "matrix_condition_number_sync_cf C++ class not available in Python bindings"
        )


def matrix_condition_number_zz(shape):
    """Create a matrix condition number block for complex double data.

    Args:
        shape: Input matrix shape [rows, cols]

    Returns:
        matrix_condition_number_sync_cd block instance
    """
    if "matrix_condition_number_sync_cd" in globals():
        return matrix_condition_number_sync_cd.make(shape)
    else:
        raise RuntimeError(
            "matrix_condition_number_sync_cd C++ class not available in Python bindings"
        )


# Decomposition blocks
def decomp_svd_ff(shape, algorithm=None):
    """Create an SVD decomposition block for float data.

    Args:
        shape: Matrix shape [rows, cols]
        algorithm: SVD algorithm (optional, uses default if None)

    Returns:
        decomp_svd_sync_f block instance
    """
    if algorithm is None:
        return decomp_svd_sync_f(shape)
    else:
        return decomp_svd_sync_f(shape, algorithm)


def decomp_svd_dd(shape, algorithm=None):
    """Create an SVD decomposition block for double data.

    Args:
        shape: Matrix shape [rows, cols]
        algorithm: SVD algorithm (optional, uses default if None)

    Returns:
        decomp_svd_sync_d block instance
    """
    if algorithm is None:
        return decomp_svd_sync_d(shape)
    else:
        return decomp_svd_sync_d(shape, algorithm)


def decomp_eigen_ff(shape):
    """Create an eigenvalue decomposition block for float data.

    Args:
        shape: Matrix shape [rows, cols]

    Returns:
        decomp_eigen_sync_f block instance
    """
    return decomp_eigen_sync_f(shape)


def decomp_eigen_dd(shape):
    """Create an eigenvalue decomposition block for double data.

    Args:
        shape: Matrix shape [rows, cols]

    Returns:
        decomp_eigen_sync_d block instance
    """
    return decomp_eigen_sync_d(shape)


def decomp_lu_ff(shape):
    """Create an LU decomposition block for float data.

    Args:
        shape: Matrix shape [rows, cols]

    Returns:
        decomp_lu_sync_f block instance
    """
    return decomp_lu_sync_f(shape)


def decomp_lu_dd(shape):
    """Create an LU decomposition block for double data.

    Args:
        shape: Matrix shape [rows, cols]

    Returns:
        decomp_lu_sync_d block instance
    """
    return decomp_lu_sync_d(shape)


def decomp_qr_ff(shape):
    """Create a QR decomposition block for float data.

    Args:
        shape: Matrix shape [rows, cols]

    Returns:
        decomp_qr_sync_f block instance
    """
    return decomp_qr_sync_f(shape)


def decomp_qr_dd(shape):
    """Create a QR decomposition block for double data.

    Args:
        shape: Matrix shape [rows, cols]

    Returns:
        decomp_qr_sync_d block instance
    """
    return decomp_qr_sync_d(shape)


def decomp_cholesky_ff(shape):
    """Create a Cholesky decomposition block for float data.

    Args:
        shape: Matrix shape [rows, cols]

    Returns:
        decomp_cholesky_sync_f block instance
    """
    return decomp_cholesky_sync_f(shape)


def decomp_cholesky_dd(shape):
    """Create a Cholesky decomposition block for double data.

    Args:
        shape: Matrix shape [rows, cols]

    Returns:
        decomp_cholesky_sync_d block instance
    """
    return decomp_cholesky_sync_d(shape)


# Vector operations - these blocks don't take shape parameters
def vector_outer_product_ff():
    """Create a vector outer product block for float data.

    Note: This block determines input/output shapes dynamically from connected vectors.

    Returns:
        vector_outer_product_sync block instance
    """
    if "vector_outer_product_sync" in globals():
        return vector_outer_product_sync()
    else:
        raise RuntimeError(
            "vector_outer_product_sync C++ class not available in Python bindings"
        )


def vector_outer_product_dd():
    """Create a vector outer product block for double data.

    Note: This block determines input/output shapes dynamically from connected vectors.

    Returns:
        vector_outer_product_sync_d block instance
    """
    if "vector_outer_product_sync_d" in globals():
        return vector_outer_product_sync_d()
    else:
        raise RuntimeError(
            "vector_outer_product_sync_d C++ class not available in Python bindings"
        )


# Generic factory functions that tests expect (use double precision as default)
def matrix_add(shape=[2, 2], num_inputs=2, *args, **kwargs):
    """Create a matrix addition block (defaults to double precision)."""
    if "matrix_add_sync_d" in globals():
        return matrix_add_sync_d(shape, num_inputs, *args, **kwargs)
    return None


def matrix_subtract(shape=[2, 2], num_inputs=2, *args, **kwargs):
    """Create a matrix subtraction block (defaults to double precision)."""
    if "matrix_subtract_sync_d" in globals():
        return matrix_subtract_sync_d(shape, num_inputs, *args, **kwargs)
    return None


def matrix_multiply(shape=[2, 2], num_inputs=2, *args, **kwargs):
    """Create a matrix multiplication block (defaults to double precision)."""
    if "matrix_multiply_sync_d" in globals():
        return matrix_multiply_sync_d(shape, num_inputs, *args, **kwargs)
    return None


def matrix_transpose(shape=[2, 2], *args, **kwargs):
    """Create a matrix transpose block (defaults to double precision)."""
    if "matrix_transpose_sync_d" in globals():
        return matrix_transpose_sync_d(shape, *args, **kwargs)
    return None


def matrix_determinant(shape=[2, 2], *args, **kwargs):
    """Create a matrix determinant block (defaults to float precision for GNU Radio compatibility)."""
    if "matrix_determinant_sync_f" in globals():
        return matrix_determinant_sync_f(shape, *args, **kwargs)
    return None


def matrix_trace(shape=[2, 2], *args, **kwargs):
    """Create a matrix trace block (defaults to float precision for GNU Radio compatibility)."""
    if "matrix_trace_sync_f" in globals():
        return matrix_trace_sync_f(shape, *args, **kwargs)
    return None


def matrix_rank(shape=[2, 2], *args, **kwargs):
    """Create a matrix rank block (defaults to float precision for GNU Radio compatibility)."""
    if "matrix_rank_sync_f" in globals():
        return matrix_rank_sync_f(shape, *args, **kwargs)
    return None


def matrix_inverse(*args, **kwargs):
    """Create a matrix inverse block (defaults to double precision)."""
    # Note: matrix_inverse_sync_d not yet available in C++ bindings
    return None


def matrix_norm_ff(shape, norm=None):
    """Create a matrix norm block for float data.

    Args:
        shape: Matrix shape [rows, cols]
        norm: Norm type (L1, L2, Frobenius, Max) - defaults to Frobenius

    Returns:
        matrix_norm_sync_f block instance
    """
    if norm is None:
        return matrix_norm_sync_f(shape)
    return matrix_norm_sync_f(shape, norm)


def matrix_norm_dd(shape, norm=None):
    """Create a matrix norm block for double data.

    Args:
        shape: Matrix shape [rows, cols]
        norm: Norm type (L1, L2, Frobenius, Max) - defaults to Frobenius

    Returns:
        matrix_norm_sync_d block instance
    """
    if norm is None:
        return matrix_norm_sync_d(shape)
    return matrix_norm_sync_d(shape, norm)


def matrix_norm_cc(shape, norm=None):
    """Create a matrix norm block for complex float data.

    Args:
        shape: Matrix shape [rows, cols]
        norm: Norm type (L1, L2, Frobenius, Max) - defaults to Frobenius

    Returns:
        matrix_norm_sync_c block instance
    """
    if norm is None:
        return matrix_norm_sync_c(shape)
    return matrix_norm_sync_c(shape, norm)


def matrix_norm_zz(shape, norm=None):
    """Create a matrix norm block for complex double data.

    Args:
        shape: Matrix shape [rows, cols]
        norm: Norm type (L1, L2, Frobenius, Max) - defaults to Frobenius

    Returns:
        matrix_norm_sync_cd block instance
    """
    if norm is None:
        return matrix_norm_sync_cd(shape)
    return matrix_norm_sync_cd(shape, norm)


def matrix_norm(*args, **kwargs):
    """Create a matrix norm block (defaults to double precision)."""
    if "matrix_norm_sync_d" in globals():
        return matrix_norm_sync_d(*args, **kwargs)
    return None


def matrix_reshape_ff(input_shape, output_shape):
    """Create a matrix reshape block for float data.

    Args:
        input_shape: Input matrix shape [rows, cols]
        output_shape: Output matrix shape [rows, cols]

    Returns:
        matrix_reshape_sync_f block instance
    """
    if "matrix_reshape_sync_f" in globals():
        return matrix_reshape_sync_f.make(input_shape, output_shape)
    else:
        raise RuntimeError(
            "matrix_reshape_sync_f C++ class not available in Python bindings"
        )


def matrix_reshape_dd(input_shape, output_shape):
    """Create a matrix reshape block for double data.

    Args:
        input_shape: Input matrix shape [rows, cols]
        output_shape: Output matrix shape [rows, cols]

    Returns:
        matrix_reshape_sync_d block instance
    """
    if "matrix_reshape_sync_d" in globals():
        return matrix_reshape_sync_d.make(input_shape, output_shape)
    else:
        raise RuntimeError(
            "matrix_reshape_sync_d C++ class not available in Python bindings"
        )


def matrix_reshape_cc(input_shape, output_shape):
    """Create a matrix reshape block for complex float data.

    Args:
        input_shape: Input matrix shape [rows, cols]
        output_shape: Output matrix shape [rows, cols]

    Returns:
        matrix_reshape_sync_c block instance
    """
    if "matrix_reshape_sync_c" in globals():
        return matrix_reshape_sync_c.make(input_shape, output_shape)
    else:
        raise RuntimeError(
            "matrix_reshape_sync_c C++ class not available in Python bindings"
        )


def matrix_reshape_zz(input_shape, output_shape):
    """Create a matrix reshape block for complex double data.

    Args:
        input_shape: Input matrix shape [rows, cols]
        output_shape: Output matrix shape [rows, cols]

    Returns:
        matrix_reshape_sync_cd block instance
    """
    if "matrix_reshape_sync_cd" in globals():
        return matrix_reshape_sync_cd.make(input_shape, output_shape)
    else:
        raise RuntimeError(
            "matrix_reshape_sync_cd C++ class not available in Python bindings"
        )


def matrix_reshape(input_shape=[2, 2], output_shape=[4, 1], *args, **kwargs):
    """Create a matrix reshape block (defaults to float precision for GNU Radio compatibility).

    Args:
        input_shape: Input matrix shape [rows, cols]
        output_shape: Output matrix shape [rows, cols]

    Returns:
        matrix_reshape_sync_f block instance
    """
    if "matrix_reshape_sync_f" in globals():
        return matrix_reshape_sync_f.make(input_shape, output_shape)
    return None


def matrix_solve_ff(shape_a, shape_b, method=None, tolerance=1e-6):
    """Create a matrix solve block for float data.

    Args:
        shape_a: Coefficient matrix A shape [rows, cols]
        shape_b: Right-hand side b shape [rows, cols] (cols=1 for vector)
        method: Solver method (AUTO, LU, QR, SVD, CHOLESKY)
        tolerance: Tolerance for rank-deficient systems

    Returns:
        matrix_solve_sync_f block instance
    """
    if "matrix_solve_sync_f" in globals() and "solver_method" in globals():
        if method is None:
            method = solver_method.AUTO
        return matrix_solve_sync_f.make(shape_a, shape_b, method, tolerance)
    else:
        raise RuntimeError(
            "matrix_solve_sync_f C++ class not available in Python bindings"
        )


def matrix_solve_dd(shape_a, shape_b, method=None, tolerance=1e-6):
    """Create a matrix solve block for double data.

    Args:
        shape_a: Coefficient matrix A shape [rows, cols]
        shape_b: Right-hand side b shape [rows, cols] (cols=1 for vector)
        method: Solver method (AUTO, LU, QR, SVD, CHOLESKY) or int
        tolerance: Tolerance for rank-deficient systems

    Returns:
        matrix_solve_sync_d block instance
    """
    if "matrix_solve_sync_d" in globals() and "solver_method" in globals():
        if method is None:
            method_int = 0  # AUTO = 0
        elif hasattr(method, "value"):
            method_int = method.value
        else:
            method_int = int(method)

        # Use the C++ block directly with template-specific enum
        # linalg_python is already imported at module level
        return matrix_solve_sync_d.make(
            shape_a, shape_b, solver_method_d(method_int), tolerance
        )
    else:
        raise RuntimeError(
            "matrix_solve_sync_d C++ class not available in Python bindings"
        )


def matrix_solve_cc(shape_a, shape_b, method=None, tolerance=1e-6):
    """Create a matrix solve block for complex float data.

    Args:
        shape_a: Coefficient matrix A shape [rows, cols]
        shape_b: Right-hand side b shape [rows, cols] (cols=1 for vector)
        method: Solver method (AUTO, LU, QR, SVD, CHOLESKY) or int
        tolerance: Tolerance for rank-deficient systems

    Returns:
        matrix_solve_sync_c block instance
    """
    if "matrix_solve_sync_c" in globals() and "solver_method" in globals():
        if method is None:
            method_int = 0  # AUTO = 0
        elif hasattr(method, "value"):
            method_int = method.value
        else:
            method_int = int(method)

        # Use the C++ block directly with template-specific enum
        # linalg_python is already imported at module level
        return matrix_solve_sync_c.make(
            shape_a, shape_b, solver_method_c(method_int), tolerance
        )
    else:
        raise RuntimeError(
            "matrix_solve_sync_c C++ class not available in Python bindings"
        )


def matrix_solve_zz(shape_a, shape_b, method=None, tolerance=1e-6):
    """Create a matrix solve block for complex double data.

    Args:
        shape_a: Coefficient matrix A shape [rows, cols]
        shape_b: Right-hand side b shape [rows, cols] (cols=1 for vector)
        method: Solver method (AUTO, LU, QR, SVD, CHOLESKY) or int
        tolerance: Tolerance for rank-deficient systems

    Returns:
        matrix_solve_sync_cd block instance
    """
    if "matrix_solve_sync_cd" in globals() and "solver_method" in globals():
        if method is None:
            method_int = 0  # AUTO = 0
        elif hasattr(method, "value"):
            method_int = method.value
        else:
            method_int = int(method)

        # Use the C++ block directly with template-specific enum
        # linalg_python is already imported at module level
        return matrix_solve_sync_cd.make(
            shape_a, shape_b, solver_method_cd(method_int), tolerance
        )
    else:
        raise RuntimeError(
            "matrix_solve_sync_cd C++ class not available in Python bindings"
        )


def matrix_solve(
    shape_a=[3, 3], shape_b=[3, 1], method=None, tolerance=1e-6, *args, **kwargs
):
    """Create a matrix solve block (defaults to float precision for GNU Radio compatibility).

    Args:
        shape_a: Coefficient matrix A shape [rows, cols]
        shape_b: Right-hand side b shape [rows, cols] (cols=1 for vector)
        method: Solver method (AUTO, LU, QR, SVD, CHOLESKY) or int
        tolerance: Tolerance for rank-deficient systems

    Returns:
        matrix_solve_sync_f block instance
    """
    if "matrix_solve_sync_f" in globals() and "solver_method" in globals():
        if method is None:
            method_int = 0  # AUTO = 0
        elif hasattr(method, "value"):
            method_int = method.value
        else:
            method_int = int(method)

        # Use the C++ block directly, bypassing enum type issues
        # linalg_python is already imported at module level
        return matrix_solve_sync_f.make(
            shape_a, shape_b, solver_method(method_int), tolerance
        )
    return None


def matrix_inverse_ff(shape):
    """Create a matrix inverse block for float data.

    Args:
        shape: Input matrix shape [rows, cols] (must be square)

    Returns:
        matrix_inverse_sync_f block instance
    """
    if "matrix_inverse_sync_f" in globals():
        return matrix_inverse_sync_f.make(shape)
    else:
        raise RuntimeError(
            "matrix_inverse_sync_f C++ class not available in Python bindings"
        )


def matrix_inverse_dd(shape):
    """Create a matrix inverse block for double data.

    Args:
        shape: Input matrix shape [rows, cols] (must be square)

    Returns:
        matrix_inverse_sync_d block instance
    """
    if "matrix_inverse_sync_d" in globals():
        return matrix_inverse_sync_d.make(shape)
    else:
        raise RuntimeError(
            "matrix_inverse_sync_d C++ class not available in Python bindings"
        )


def matrix_inverse_cc(shape):
    """Create a matrix inverse block for complex float data.

    Args:
        shape: Input matrix shape [rows, cols] (must be square)

    Returns:
        matrix_inverse_sync_c block instance
    """
    if "matrix_inverse_sync_c" in globals():
        return matrix_inverse_sync_c.make(shape)
    else:
        raise RuntimeError(
            "matrix_inverse_sync_c C++ class not available in Python bindings"
        )


def matrix_inverse_zz(shape):
    """Create a matrix inverse block for complex double data.

    Args:
        shape: Input matrix shape [rows, cols] (must be square)

    Returns:
        matrix_inverse_sync_cd block instance
    """
    if "matrix_inverse_sync_cd" in globals():
        return matrix_inverse_sync_cd.make(shape)
    else:
        raise RuntimeError(
            "matrix_inverse_sync_cd C++ class not available in Python bindings"
        )


def matrix_inverse(shape):
    """Create a matrix inverse block (defaults to float precision).

    Args:
        shape: Input matrix shape [rows, cols] (must be square)

    Returns:
        matrix_inverse_sync_f block instance
    """
    return matrix_inverse_ff(shape)


def matrix_hermitian_ff(shape):
    """Create a matrix Hermitian transpose block for float data.

    Args:
        shape: Input matrix shape [rows, cols]

    Returns:
        matrix_hermitian_sync_f block instance
    """
    if "matrix_hermitian_sync_f" in globals():
        return matrix_hermitian_sync_f(shape)
    else:
        raise RuntimeError(
            "matrix_hermitian_sync_f C++ class not available in Python bindings"
        )


def matrix_hermitian_dd(shape):
    """Create a matrix Hermitian transpose block for double data.

    Args:
        shape: Input matrix shape [rows, cols]

    Returns:
        matrix_hermitian_sync_d block instance
    """
    if "matrix_hermitian_sync_d" in globals():
        return matrix_hermitian_sync_d(shape)
    else:
        raise RuntimeError(
            "matrix_hermitian_sync_d C++ class not available in Python bindings"
        )


def matrix_hermitian_cc(shape):
    """Create a matrix Hermitian transpose block for complex float data.

    Args:
        shape: Input matrix shape [rows, cols]

    Returns:
        matrix_hermitian_sync_cf block instance
    """
    if "matrix_hermitian_sync_cf" in globals():
        return matrix_hermitian_sync_cf(shape)
    else:
        raise RuntimeError(
            "matrix_hermitian_sync_cf C++ class not available in Python bindings"
        )


def matrix_hermitian_zz(shape):
    """Create a matrix Hermitian transpose block for complex double data.

    Args:
        shape: Input matrix shape [rows, cols]

    Returns:
        matrix_hermitian_sync_cd block instance
    """
    if "matrix_hermitian_sync_cd" in globals():
        return matrix_hermitian_sync_cd(shape)
    else:
        raise RuntimeError(
            "matrix_hermitian_sync_cd C++ class not available in Python bindings"
        )


def matrix_hermitian(shape):
    """Create a matrix Hermitian transpose block (defaults to double precision).

    Args:
        shape: Input matrix shape [rows, cols]

    Returns:
        matrix_hermitian_sync_d block instance
    """
    return matrix_hermitian_dd(shape)


def matrix_pseudo_inverse_ff(shape):
    """Create a matrix pseudo-inverse block for float data.

    Args:
        shape: Input matrix shape [rows, cols]

    Returns:
        matrix_pseudo_inverse_sync_f block instance
    """
    if "matrix_pseudo_inverse_sync_f" in globals():
        return matrix_pseudo_inverse_sync_f.make(shape)
    else:
        raise RuntimeError(
            "matrix_pseudo_inverse_sync_f C++ class not available in Python bindings"
        )


def matrix_pseudo_inverse_dd(shape):
    """Create a matrix pseudo-inverse block for double data.

    Args:
        shape: Input matrix shape [rows, cols]

    Returns:
        matrix_pseudo_inverse_sync_d block instance
    """
    if "matrix_pseudo_inverse_sync_d" in globals():
        return matrix_pseudo_inverse_sync_d.make(shape)
    else:
        raise RuntimeError(
            "matrix_pseudo_inverse_sync_d C++ class not available in Python bindings"
        )


def matrix_pseudo_inverse_cc(shape):
    """Create a matrix pseudo-inverse block for complex float data.

    Args:
        shape: Input matrix shape [rows, cols]

    Returns:
        matrix_pseudo_inverse_sync_cf block instance
    """
    if "matrix_pseudo_inverse_sync_cf" in globals():
        return matrix_pseudo_inverse_sync_cf.make(shape)
    else:
        raise RuntimeError(
            "matrix_pseudo_inverse_sync_cf C++ class not available in Python bindings"
        )


def matrix_pseudo_inverse_zz(shape):
    """Create a matrix pseudo-inverse block for complex double data.

    Args:
        shape: Input matrix shape [rows, cols]

    Returns:
        matrix_pseudo_inverse_sync_cd block instance
    """
    if "matrix_pseudo_inverse_sync_cd" in globals():
        return matrix_pseudo_inverse_sync_cd.make(shape)
    else:
        raise RuntimeError(
            "matrix_pseudo_inverse_sync_cd C++ class not available in Python bindings"
        )


def matrix_pseudo_inverse(shape):
    """Create a matrix pseudo-inverse block (defaults to double precision).

    Args:
        shape: Input matrix shape [rows, cols]

    Returns:
        matrix_pseudo_inverse_sync_d block instance
    """
    return matrix_pseudo_inverse_dd(shape)


# Matrix Source Const precision-specific functions
def matrix_source_const_sync_ff(matrix, shape):
    """Create a float precision synchronous matrix constant source block."""
    if "matrix_source_const_sync_f" in globals():
        return matrix_source_const_sync_f.make(matrix, shape)
    return None


def matrix_source_const_sync_dd(matrix, shape):
    """Create a double precision synchronous matrix constant source block."""
    if "matrix_source_const_sync_d" in globals():
        return matrix_source_const_sync_d.make(matrix, shape)
    return None


def matrix_source_const_sync_cc(matrix, shape):
    """Create a complex float precision synchronous matrix constant source block."""
    if "matrix_source_const_sync_cf" in globals():
        return matrix_source_const_sync_cf.make(matrix, shape)
    return None


def matrix_source_const_sync_zz(matrix, shape):
    """Create a complex double precision synchronous matrix constant source block."""
    if "matrix_source_const_sync_cd" in globals():
        return matrix_source_const_sync_cd.make(matrix, shape)
    return None


def matrix_source_const_pdu_ff(matrix, shape):
    """Create a float precision PDU matrix constant source block."""
    if "matrix_source_const_pdu_f" in globals():
        return matrix_source_const_pdu_f.make(matrix, shape)
    return None


def matrix_source_const_pdu_dd(matrix, shape):
    """Create a double precision PDU matrix constant source block."""
    if "matrix_source_const_pdu_d" in globals():
        return matrix_source_const_pdu_d.make(matrix, shape)
    return None


def matrix_source_const_pdu_cc(matrix, shape):
    """Create a complex float precision PDU matrix constant source block."""
    if "matrix_source_const_pdu_cf" in globals():
        return matrix_source_const_pdu_cf.make(matrix, shape)
    return None


def matrix_source_const_pdu_zz(matrix, shape):
    """Create a complex double precision PDU matrix constant source block."""
    if "matrix_source_const_pdu_cd" in globals():
        return matrix_source_const_pdu_cd.make(matrix, shape)
    return None


def matrix_source_const(matrix, shape, sync=True, *args, **kwargs):
    """Create a matrix constant source block (defaults to float precision synchronous for GNU Radio compatibility)."""
    if sync:
        return matrix_source_const_sync_ff(matrix, shape)
    else:
        return matrix_source_const_pdu_ff(matrix, shape)


def matrix_diag(shape=[2, 2], k=0, *args, **kwargs):
    """Create a matrix diagonal extraction block (defaults to float precision for GNU Radio compatibility)."""
    if "matrix_diag_sync_f" in globals():
        return matrix_diag_sync_f(shape, k, *args, **kwargs)
    return None


# Vector operations
def dot_product_ff(shape_0, shape_1):
    """Create a dot product block for float data."""
    if "dot_product_sync_f" in globals():
        return dot_product_sync_f.make(shape_0, shape_1)
    return None


def dot_product_dd(shape_0, shape_1):
    """Create a dot product block for double data."""
    if "dot_product_sync_d" in globals():
        return dot_product_sync_d.make(shape_0, shape_1)
    return None


def dot_product_factory(shape_0=[3], shape_1=[3], *args, **kwargs):
    """Create a dot product block (defaults to float precision for GNU Radio compatibility)."""
    if "dot_product_sync_f" in globals():
        return dot_product_sync_f.make(shape_0, shape_1)
    elif "dot_product_sync_d" in globals():
        return dot_product_sync_d.make(shape_0, shape_1)
    return None


# Keep the original name as an alias to avoid breaking changes
dot_product = dot_product_factory


def vector_norm_ff(order, shape):
    """Create a vector norm block for float data."""
    if "vector_norm_sync_f" in globals():
        return vector_norm_sync_f(shape, order)
    return None


def vector_norm_dd(order, shape):
    """Create a vector norm block for double data."""
    if "vector_norm_sync_d" in globals():
        return vector_norm_sync_d(shape, order)
    return None


def vector_norm_factory(shape_or_order=None, order_or_shape=None, *args, **kwargs):
    """Create a vector norm block (defaults to float precision for GNU Radio compatibility)."""
    # Handle different calling conventions:
    # vector_norm(shape) -> order=2, shape=shape
    # vector_norm(order, shape) -> order=order, shape=shape
    # vector_norm(shape=shape, order=order) -> kwargs

    if shape_or_order is None:
        # No positional args, use defaults
        shape = [3]
        order = 2
    elif order_or_shape is None:
        # One positional arg - assume it's shape, use default order
        shape = shape_or_order
        order = 2
    else:
        # Two positional args - first is order, second is shape (maintaining backward compatibility)
        order = shape_or_order
        shape = order_or_shape

    # Override with keyword arguments if provided
    shape = kwargs.get("shape", shape)
    order = kwargs.get("order", order)

    # Prefer float precision for GNU Radio compatibility
    if "vector_norm_sync_f" in globals():
        return vector_norm_sync_f(shape, order, *args)
    elif "vector_norm_sync_d" in globals():
        return vector_norm_sync_d(shape, order, *args)
    return None


# Keep the original name as an alias
vector_norm = vector_norm_factory


def vector_normalize_ff(order, shape):
    """Create a vector normalize block for float data."""
    if "vector_normalize_sync_f" in globals():
        return vector_normalize_sync_f(shape, order)
    return None


def vector_normalize_dd(order, shape):
    """Create a vector normalize block for double data."""
    if "vector_normalize_sync_d" in globals():
        return vector_normalize_sync_d(shape, order)
    return None


def vector_normalize_factory(shape_or_order=None, order_or_shape=None, *args, **kwargs):
    """Create a vector normalize block (defaults to float precision for GNU Radio compatibility)."""
    # Handle different calling conventions:
    # vector_normalize(shape) -> order=2, shape=shape
    # vector_normalize(order, shape) -> order=order, shape=shape
    # vector_normalize(shape=shape, order=order) -> kwargs

    if shape_or_order is None:
        # No positional args, use defaults
        shape = [3]
        order = 2
    elif order_or_shape is None:
        # One positional arg - assume it's shape, use default order
        shape = shape_or_order
        order = 2
    else:
        # Two positional args - first is order, second is shape (maintaining backward compatibility)
        order = shape_or_order
        shape = order_or_shape

    # Override with keyword arguments if provided
    shape = kwargs.get("shape", shape)
    order = kwargs.get("order", order)

    # Prefer float precision for GNU Radio compatibility
    if "vector_normalize_sync_f" in globals():
        return vector_normalize_sync_f(shape, order, *args)
    elif "vector_normalize_sync_d" in globals():
        return vector_normalize_sync_d(shape, order, *args)
    return None


# Keep the original name as an alias
vector_normalize = vector_normalize_factory


def vector_dot_factory(shape_a=[3], shape_b=[3], *args, **kwargs):
    """Create a vector dot product block (defaults to float precision for compatibility).

    Note: This is an alias for dot_product functionality.
    """
    # Use the working dot_product implementation instead of the incomplete vector_dot stub
    # Default to float precision to match most tests
    if "dot_product_sync_f" in globals():
        return dot_product_sync_f.make(shape_a, shape_b)
    elif "dot_product_sync_d" in globals():
        return dot_product_sync_d.make(shape_a, shape_b)
    return None


def vector_dot_ff(shape_a, shape_b):
    """Create a vector dot product block for float data (alias for dot_product_ff)."""
    return dot_product_ff(shape_a, shape_b)


def vector_dot_dd(shape_a, shape_b):
    """Create a vector dot product block for double data (alias for dot_product_dd)."""
    return dot_product_dd(shape_a, shape_b)


def vector_dot_cc(shape_a, shape_b):
    """Create a vector dot product block for complex float data (alias for dot_product)."""
    if "dot_product_sync_c" in globals():
        return dot_product_sync_c.make(shape_a, shape_b)
    return None


def vector_dot_zz(shape_a, shape_b):
    """Create a vector dot product block for complex double data (alias for dot_product)."""
    if "dot_product_sync_cd" in globals():
        return dot_product_sync_cd.make(shape_a, shape_b)
    return None


vector_dot = vector_dot_factory


def matrix_kronecker_product_factory(shape_a=[2, 2], shape_b=[2, 2], *args, **kwargs):
    """Create a matrix Kronecker product block (defaults to double precision)."""
    if "matrix_kronecker_product_sync_d" in globals():
        return globals()["matrix_kronecker_product_sync_d"].make(
            shape_a, shape_b, *args, **kwargs
        )
    elif "matrix_kronecker_product_sync" in globals():
        return globals()["matrix_kronecker_product_sync"].make(
            shape_a, shape_b, *args, **kwargs
        )
    return None


matrix_kronecker_product = matrix_kronecker_product_factory


def matrix_kronecker_product_ff(shape_a, shape_b):
    """Create a float precision matrix Kronecker product block."""
    if "matrix_kronecker_product_sync_f" in globals():
        return matrix_kronecker_product_sync_f.make(shape_a, shape_b)
    return None


def matrix_kronecker_product_dd(shape_a, shape_b):
    """Create a double precision matrix Kronecker product block."""
    if "matrix_kronecker_product_sync_d" in globals():
        return matrix_kronecker_product_sync_d.make(shape_a, shape_b)
    return None


def matrix_kronecker_product_cc(shape_a, shape_b):
    """Create a complex float precision matrix Kronecker product block."""
    if "matrix_kronecker_product_sync_c" in globals():
        return matrix_kronecker_product_sync_c.make(shape_a, shape_b)
    return None


def matrix_kronecker_product_zz(shape_a, shape_b):
    """Create a complex double precision matrix Kronecker product block."""
    if "matrix_kronecker_product_sync_cd" in globals():
        return matrix_kronecker_product_sync_cd.make(shape_a, shape_b)
    return None


def vector_cross_product_factory(validate_3d=True, *args, **kwargs):
    """Create a vector cross product block for 3D vectors (defaults to double precision)."""
    # Check for the actual C++ binding class names
    if "vector_cross_product_sync_d" in globals():
        return globals()["vector_cross_product_sync_d"](validate_3d, *args, **kwargs)
    elif "vector_cross_product_sync" in globals():
        return globals()["vector_cross_product_sync"](validate_3d, *args, **kwargs)
    return None


vector_cross_product = vector_cross_product_factory


def vector_Kronecker(*args, **kwargs):
    """Deprecated: Use matrix_kronecker_product instead."""
    # Redirect to proper implementation
    return matrix_kronecker_product(*args, **kwargs)


# Matrix generators (eye, ones, zeros) - fixed template binding issues
def eye_ff(shape):
    """Create a float precision identity matrix generator block."""
    if "eye_sync_f" in globals():
        return eye_sync_f.make(shape)
    return None


def eye_dd(shape):
    """Create a double precision identity matrix generator block."""
    if "eye_sync_d" in globals():
        return eye_sync_d.make(shape)
    return None


def eye_cc(shape):
    """Create a complex float precision identity matrix generator block."""
    if "eye_sync_cf" in globals():
        return eye_sync_cf.make(shape)
    return None


def eye_zz(shape):
    """Create a complex double precision identity matrix generator block."""
    if "eye_sync_cd" in globals():
        return eye_sync_cd.make(shape)
    return None


def eye(shape=[3, 3], *args, **kwargs):
    """Create an identity matrix generator block (defaults to double precision)."""
    return eye_dd(shape)


def zeros_ff(shape):
    """Create a float precision zeros matrix generator block."""
    if "zeros_sync_f" in globals():
        return zeros_sync_f.make(shape)
    return None


def zeros_dd(shape):
    """Create a double precision zeros matrix generator block."""
    if "zeros_sync_d" in globals():
        return zeros_sync_d.make(shape)
    return None


def zeros_cc(shape):
    """Create a complex float precision zeros matrix generator block."""
    if "zeros_sync_cf" in globals():
        return zeros_sync_cf.make(shape)
    return None


def zeros_zz(shape):
    """Create a complex double precision zeros matrix generator block."""
    if "zeros_sync_cd" in globals():
        return zeros_sync_cd.make(shape)
    return None


def zeros(shape=[3, 3], *args, **kwargs):
    """Create a zero matrix generator block (defaults to double precision)."""
    return zeros_dd(shape)


def ones_ff(shape):
    """Create a float precision ones matrix generator block."""
    if "ones_sync_f" in globals():
        return ones_sync_f.make(shape)
    return None


def ones_dd(shape):
    """Create a double precision ones matrix generator block."""
    if "ones_sync_d" in globals():
        return ones_sync_d.make(shape)
    return None


def ones_cc(shape):
    """Create a complex float precision ones matrix generator block."""
    if "ones_sync_cf" in globals():
        return ones_sync_cf.make(shape)
    return None


def ones_zz(shape):
    """Create a complex double precision ones matrix generator block."""
    if "ones_sync_cd" in globals():
        return ones_sync_cd.make(shape)
    return None


def ones(shape=[3, 3], *args, **kwargs):
    """Create a ones matrix generator block (defaults to double precision)."""
    return ones_dd(shape)


# Decomposition blocks
def decomp_svd(shape=[2, 2], algorithm=None, *args, **kwargs):
    """Create an SVD decomposition block (defaults to double precision)."""
    if "decomp_svd_sync_d" in globals():
        # The C++ constructor expects: decomp_svd_sync_d(shape, algorithm=None)
        # When algorithm is None, we can just pass the shape
        return decomp_svd_sync_d(shape, *args, **kwargs)
    return None


def decomp_eigen(shape=[2, 2], *args, **kwargs):
    """Create an eigenvalue decomposition block (defaults to float precision for GNU Radio compatibility)."""
    if "decomp_eigen_sync_f" in globals():
        return decomp_eigen_sync_f(shape, *args, **kwargs)
    return None


def decomp_lu(shape=[2, 2], *args, **kwargs):
    """Create an LU decomposition block (defaults to float precision for GNU Radio compatibility)."""
    if "decomp_lu_sync_f" in globals():
        return decomp_lu_sync_f(shape, *args, **kwargs)
    return None


def decomp_qr(shape=[2, 2], *args, **kwargs):
    """Create a QR decomposition block (defaults to float precision for GNU Radio compatibility)."""
    if "decomp_qr_sync_f" in globals():
        return decomp_qr_sync_f(shape, *args, **kwargs)
    return None


def decomp_cholesky(shape=[2, 2], *args, **kwargs):
    """Create a Cholesky decomposition block (defaults to float precision for GNU Radio compatibility)."""
    if "decomp_cholesky_sync_f" in globals():
        return decomp_cholesky_sync_f(shape, *args, **kwargs)
    return None


# Matrix Kronecker Product factory functions
def matrix_kronecker_product_sync_ff(shape_a, shape_b):
    """Create a matrix Kronecker product block for float data (sync interface).
    Args:
        shape_a: First matrix shape [rows, cols]
        shape_b: Second matrix shape [rows, cols]
    Returns:
        matrix_kronecker_product_sync_f block instance
    """
    if "matrix_kronecker_product_sync_f" in globals():
        return matrix_kronecker_product_sync_f.make(shape_a, shape_b)
    else:
        raise RuntimeError(
            "matrix_kronecker_product_sync_f C++ class not available in Python bindings"
        )


def matrix_kronecker_product_sync_dd(shape_a, shape_b):
    """Create a matrix Kronecker product block for double data (sync interface).
    Args:
        shape_a: First matrix shape [rows, cols]
        shape_b: Second matrix shape [rows, cols]
    Returns:
        matrix_kronecker_product_sync_d block instance
    """
    if "matrix_kronecker_product_sync_d" in globals():
        return matrix_kronecker_product_sync_d.make(shape_a, shape_b)
    else:
        raise RuntimeError(
            "matrix_kronecker_product_sync_d C++ class not available in Python bindings"
        )


def matrix_kronecker_product_sync_cc(shape_a, shape_b):
    """Create a matrix Kronecker product block for complex float data (sync interface).
    Args:
        shape_a: First matrix shape [rows, cols]
        shape_b: Second matrix shape [rows, cols]
    Returns:
        matrix_kronecker_product_sync_cf block instance
    """
    if "matrix_kronecker_product_sync_cf" in globals():
        return matrix_kronecker_product_sync_cf.make(shape_a, shape_b)
    else:
        raise RuntimeError(
            "matrix_kronecker_product_sync_cf C++ class not available in Python bindings"
        )


def matrix_kronecker_product_sync_zz(shape_a, shape_b):
    """Create a matrix Kronecker product block for complex double data (sync interface).
    Args:
        shape_a: First matrix shape [rows, cols]
        shape_b: Second matrix shape [rows, cols]
    Returns:
        matrix_kronecker_product_sync_cd block instance
    """
    if "matrix_kronecker_product_sync_cd" in globals():
        return matrix_kronecker_product_sync_cd.make(shape_a, shape_b)
    else:
        raise RuntimeError(
            "matrix_kronecker_product_sync_cd C++ class not available in Python bindings"
        )


# PDU variants
def matrix_kronecker_product_pdu_ff(shape_a, shape_b):
    """Create a matrix Kronecker product block for float data (PDU interface).
    Args:
        shape_a: First matrix shape [rows, cols]
        shape_b: Second matrix shape [rows, cols]
    Returns:
        matrix_kronecker_product_pdu_f block instance
    """
    if "matrix_kronecker_product_pdu_f" in globals():
        return matrix_kronecker_product_pdu_f.make(shape_a, shape_b)
    else:
        raise RuntimeError(
            "matrix_kronecker_product_pdu_f C++ class not available in Python bindings"
        )


def matrix_kronecker_product_pdu_dd(shape_a, shape_b):
    """Create a matrix Kronecker product block for double data (PDU interface).
    Args:
        shape_a: First matrix shape [rows, cols]
        shape_b: Second matrix shape [rows, cols]
    Returns:
        matrix_kronecker_product_pdu_d block instance
    """
    if "matrix_kronecker_product_pdu_d" in globals():
        return matrix_kronecker_product_pdu_d.make(shape_a, shape_b)
    else:
        raise RuntimeError(
            "matrix_kronecker_product_pdu_d C++ class not available in Python bindings"
        )


def matrix_kronecker_product_pdu_cc(shape_a, shape_b):
    """Create a matrix Kronecker product block for complex float data (PDU interface).
    Args:
        shape_a: First matrix shape [rows, cols]
        shape_b: Second matrix shape [rows, cols]
    Returns:
        matrix_kronecker_product_pdu_cf block instance
    """
    if "matrix_kronecker_product_pdu_cf" in globals():
        return matrix_kronecker_product_pdu_cf.make(shape_a, shape_b)
    else:
        raise RuntimeError(
            "matrix_kronecker_product_pdu_cf C++ class not available in Python bindings"
        )


def matrix_kronecker_product_pdu_zz(shape_a, shape_b):
    """Create a matrix Kronecker product block for complex double data (PDU interface).
    Args:
        shape_a: First matrix shape [rows, cols]
        shape_b: Second matrix shape [rows, cols]
    Returns:
        matrix_kronecker_product_pdu_cd block instance
    """
    if "matrix_kronecker_product_pdu_cd" in globals():
        return matrix_kronecker_product_pdu_cd.make(shape_a, shape_b)
    else:
        raise RuntimeError(
            "matrix_kronecker_product_pdu_cd C++ class not available in Python bindings"
        )


# Convenient aliases for vector_Kronecker compatibility
vector_kronecker_sync_ff = matrix_kronecker_product_sync_ff
vector_kronecker_sync_dd = matrix_kronecker_product_sync_dd
vector_kronecker_sync_cc = matrix_kronecker_product_sync_cc
vector_kronecker_sync_zz = matrix_kronecker_product_sync_zz

vector_kronecker_pdu_ff = matrix_kronecker_product_pdu_ff
vector_kronecker_pdu_dd = matrix_kronecker_product_pdu_dd
vector_kronecker_pdu_cc = matrix_kronecker_product_pdu_cc
vector_kronecker_pdu_zz = matrix_kronecker_product_pdu_zz


# Matrix Pseudo-Determinant factory functions
def matrix_pseudo_determinant_sync_ff(shape):
    """Create a matrix pseudo-determinant block for float data (sync interface).
    Args:
        shape: Matrix shape [rows, cols]
    Returns:
        matrix_pseudo_determinant_sync_f block instance
    """
    if "matrix_pseudo_determinant_sync_f" in globals():
        return matrix_pseudo_determinant_sync_f.make(shape)
    else:
        raise RuntimeError(
            "matrix_pseudo_determinant_sync_f C++ class not available in Python bindings"
        )


def matrix_pseudo_determinant_sync_dd(shape):
    """Create a matrix pseudo-determinant block for double data (sync interface).
    Args:
        shape: Matrix shape [rows, cols]
    Returns:
        matrix_pseudo_determinant_sync_d block instance
    """
    if "matrix_pseudo_determinant_sync_d" in globals():
        return matrix_pseudo_determinant_sync_d.make(shape)
    else:
        raise RuntimeError(
            "matrix_pseudo_determinant_sync_d C++ class not available in Python bindings"
        )


def matrix_pseudo_determinant_sync_cc(shape):
    """Create a matrix pseudo-determinant block for complex float data (sync interface).
    Args:
        shape: Matrix shape [rows, cols]
    Returns:
        matrix_pseudo_determinant_sync_c block instance
    """
    if "matrix_pseudo_determinant_sync_c" in globals():
        return matrix_pseudo_determinant_sync_c.make(shape)
    else:
        raise RuntimeError(
            "matrix_pseudo_determinant_sync_c C++ class not available in Python bindings"
        )


def matrix_pseudo_determinant_sync_zz(shape):
    """Create a matrix pseudo-determinant block for complex double data (sync interface).
    Args:
        shape: Matrix shape [rows, cols]
    Returns:
        matrix_pseudo_determinant_sync_cd block instance
    """
    if "matrix_pseudo_determinant_sync_cd" in globals():
        return matrix_pseudo_determinant_sync_cd.make(shape)
    else:
        raise RuntimeError(
            "matrix_pseudo_determinant_sync_cd C++ class not available in Python bindings"
        )


# PDU variants
def matrix_pseudo_determinant_pdu_ff(shape):
    """Create a matrix pseudo-determinant block for float data (PDU interface).
    Args:
        shape: Matrix shape [rows, cols]
    Returns:
        matrix_pseudo_determinant_pdu_f block instance
    """
    if "matrix_pseudo_determinant_pdu_f" in globals():
        return matrix_pseudo_determinant_pdu_f.make(shape)
    else:
        raise RuntimeError(
            "matrix_pseudo_determinant_pdu_f C++ class not available in Python bindings"
        )


def matrix_pseudo_determinant_pdu_dd(shape):
    """Create a matrix pseudo-determinant block for double data (PDU interface).
    Args:
        shape: Matrix shape [rows, cols]
    Returns:
        matrix_pseudo_determinant_pdu_d block instance
    """
    if "matrix_pseudo_determinant_pdu_d" in globals():
        return matrix_pseudo_determinant_pdu_d.make(shape)
    else:
        raise RuntimeError(
            "matrix_pseudo_determinant_pdu_d C++ class not available in Python bindings"
        )


def matrix_pseudo_determinant_pdu_cc(shape):
    """Create a matrix pseudo-determinant block for complex float data (PDU interface).
    Args:
        shape: Matrix shape [rows, cols]
    Returns:
        matrix_pseudo_determinant_pdu_c block instance
    """
    if "matrix_pseudo_determinant_pdu_c" in globals():
        return matrix_pseudo_determinant_pdu_c.make(shape)
    else:
        raise RuntimeError(
            "matrix_pseudo_determinant_pdu_c C++ class not available in Python bindings"
        )


def matrix_pseudo_determinant_pdu_zz(shape):
    """Create a matrix pseudo-determinant block for complex double data (PDU interface).
    Args:
        shape: Matrix shape [rows, cols]
    Returns:
        matrix_pseudo_determinant_pdu_cd block instance
    """
    if "matrix_pseudo_determinant_pdu_cd" in globals():
        return matrix_pseudo_determinant_pdu_cd.make(shape)
    else:
        raise RuntimeError(
            "matrix_pseudo_determinant_pdu_cd C++ class not available in Python bindings"
        )


# ============================================================================
# Matrix Exponential Factory Functions
# ============================================================================


def matrix_exp_sync_ff(shape):
    """Create matrix_exp sync block (float precision)."""
    if "matrix_exp_sync_f" in globals():
        return matrix_exp_sync_f.make(shape)
    else:
        raise RuntimeError("matrix_exp_sync_f C++ class not available")


def matrix_exp_sync_dd(shape):
    """Create matrix_exp sync block (double precision)."""
    if "matrix_exp_sync_d" in globals():
        return matrix_exp_sync_d.make(shape)
    else:
        raise RuntimeError("matrix_exp_sync_d C++ class not available")


def matrix_exp_sync_cc(shape):
    """Create matrix_exp sync block (complex float precision)."""
    if "matrix_exp_sync_c" in globals():
        return matrix_exp_sync_c.make(shape)
    else:
        raise RuntimeError("matrix_exp_sync_c C++ class not available")


def matrix_exp_sync_zz(shape):
    """Create matrix_exp sync block (complex double precision)."""
    if "matrix_exp_sync_cd" in globals():
        return matrix_exp_sync_cd.make(shape)
    else:
        raise RuntimeError("matrix_exp_sync_cd C++ class not available")


def matrix_exp_pdu_ff(shape):
    """Create matrix_exp PDU block (float precision)."""
    if "matrix_exp_pdu_f" in globals():
        return matrix_exp_pdu_f.make(shape)
    else:
        raise RuntimeError("matrix_exp_pdu_f C++ class not available")


def matrix_exp_pdu_dd(shape):
    """Create matrix_exp PDU block (double precision)."""
    if "matrix_exp_pdu_d" in globals():
        return matrix_exp_pdu_d.make(shape)
    else:
        raise RuntimeError("matrix_exp_pdu_d C++ class not available")


def matrix_exp_pdu_cc(shape):
    """Create matrix_exp PDU block (complex float precision)."""
    if "matrix_exp_pdu_c" in globals():
        return matrix_exp_pdu_c.make(shape)
    else:
        raise RuntimeError("matrix_exp_pdu_c C++ class not available")


def matrix_exp_pdu_zz(shape):
    """Create matrix_exp PDU block (complex double precision)."""
    if "matrix_exp_pdu_cd" in globals():
        return matrix_exp_pdu_cd.make(shape)
    else:
        raise RuntimeError("matrix_exp_pdu_cd C++ class not available")


def matrix_exp(shape, interface="sync", precision="double"):
    """Generic matrix_exp factory function.

    Args:
        shape: Matrix dimensions [N, N] (must be square)
        interface: "sync" or "pdu"
        precision: "float", "double", "complex_float", "complex_double"

    Returns:
        GNU Radio block for computing matrix exponential exp(A)
    """
    if interface == "sync":
        if precision == "float":
            return matrix_exp_sync_ff(shape)
        elif precision == "double":
            return matrix_exp_sync_dd(shape)
        elif precision == "complex_float":
            return matrix_exp_sync_cc(shape)
        elif precision == "complex_double":
            return matrix_exp_sync_zz(shape)
        else:
            raise ValueError(f"Unknown precision: {precision}")
    elif interface == "pdu":
        if precision == "float":
            return matrix_exp_pdu_ff(shape)
        elif precision == "double":
            return matrix_exp_pdu_dd(shape)
        elif precision == "complex_float":
            return matrix_exp_pdu_cc(shape)
        elif precision == "complex_double":
            return matrix_exp_pdu_zz(shape)
        else:
            raise ValueError(f"Unknown precision: {precision}")
    else:
        raise ValueError(f"Unknown interface: {interface}")


# ============================================================================
# Solve Least Squares Factory Functions
# ============================================================================


def solve_least_squares_ff(shape_a, shape_b, method=None, tolerance=1e-10):
    """Create a solve least squares block for float data.

    Args:
        shape_a: Coefficient matrix A shape [rows, cols]
        shape_b: Right-hand side b shape [rows, cols] (cols=1 for vector)
        method: Solver method (AUTO, NORMAL_EQ, QR, SVD)
        tolerance: Tolerance for numerical computations

    Returns:
        solve_least_squares_sync_f block instance
    """
    if (
        "solve_least_squares_sync_f" in globals()
        and "least_squares_method" in globals()
    ):
        if method is None:
            method = least_squares_method.AUTO
        return solve_least_squares_sync_f.make(shape_a, shape_b, method, tolerance)
    else:
        raise RuntimeError(
            "solve_least_squares_sync_f C++ class not available in Python bindings"
        )


def solve_least_squares_dd(shape_a, shape_b, method=None, tolerance=1e-10):
    """Create a solve least squares block for double data.

    Args:
        shape_a: Coefficient matrix A shape [rows, cols]
        shape_b: Right-hand side b shape [rows, cols] (cols=1 for vector)
        method: Solver method (AUTO, NORMAL_EQ, QR, SVD)
        tolerance: Tolerance for numerical computations

    Returns:
        solve_least_squares_sync_d block instance
    """
    if (
        "solve_least_squares_sync_d" in globals()
        and "least_squares_method" in globals()
    ):
        if method is None:
            method = least_squares_method.AUTO
        return solve_least_squares_sync_d.make(shape_a, shape_b, method, tolerance)
    else:
        raise RuntimeError(
            "solve_least_squares_sync_d C++ class not available in Python bindings"
        )


def solve_least_squares_cc(shape_a, shape_b, method=None, tolerance=1e-10):
    """Create a solve least squares block for complex float data.

    Args:
        shape_a: Coefficient matrix A shape [rows, cols]
        shape_b: Right-hand side b shape [rows, cols] (cols=1 for vector)
        method: Solver method (AUTO, NORMAL_EQ, QR, SVD)
        tolerance: Tolerance for numerical computations

    Returns:
        solve_least_squares_sync_c block instance
    """
    if (
        "solve_least_squares_sync_c" in globals()
        and "least_squares_method" in globals()
    ):
        if method is None:
            method = least_squares_method.AUTO
        return solve_least_squares_sync_c.make(shape_a, shape_b, method, tolerance)
    else:
        raise RuntimeError(
            "solve_least_squares_sync_c C++ class not available in Python bindings"
        )


def solve_least_squares_zz(shape_a, shape_b, method=None, tolerance=1e-10):
    """Create a solve least squares block for complex double data.

    Args:
        shape_a: Coefficient matrix A shape [rows, cols]
        shape_b: Right-hand side b shape [rows, cols] (cols=1 for vector)
        method: Solver method (AUTO, NORMAL_EQ, QR, SVD)
        tolerance: Tolerance for numerical computations

    Returns:
        solve_least_squares_sync_cd block instance
    """
    if (
        "solve_least_squares_sync_cd" in globals()
        and "least_squares_method" in globals()
    ):
        if method is None:
            method = least_squares_method.AUTO
        return solve_least_squares_sync_cd.make(shape_a, shape_b, method, tolerance)
    else:
        raise RuntimeError(
            "solve_least_squares_sync_cd C++ class not available in Python bindings"
        )


def solve_least_squares_factory(
    shape_a=None, shape_b=None, method=None, tolerance=1e-10, *args, **kwargs
):
    """Create a solve least squares block (defaults to double precision).

    Args:
        shape_a: Coefficient matrix A shape [rows, cols] - defaults to [3, 2] (overdetermined)
        shape_b: Right-hand side b shape [rows, cols] - defaults to [3, 1] (vector)
        method: Solver method (AUTO, NORMAL_EQ, QR, SVD) - defaults to AUTO
        tolerance: Tolerance for numerical computations

    Returns:
        solve_least_squares_sync_d block instance

    Usage:
    - solve_least_squares() -> defaults: A=[3,2], b=[3,1], method=AUTO (3 equations, 2 unknowns)
    - solve_least_squares([4, 3]) -> A=[4,3], b=[4,1], method=AUTO
    - solve_least_squares([3,2], [3,1]) -> specified shapes
    - solve_least_squares([5,3], [5,1], method=QR) -> specified method
    """
    # Handle different calling conventions with sensible defaults for overdetermined systems
    if shape_a is None:
        shape_a = [3, 2]  # 3 equations, 2 unknowns (overdetermined)

    if shape_b is None:
        # Auto-infer shape_b from shape_a: same number of rows, 1 column (vector)
        if len(shape_a) >= 2:
            shape_b = [shape_a[0], 1]  # Same rows as A, 1 column
        else:
            shape_b = [3, 1]  # Fallback default

    # Override with keyword arguments if provided
    shape_a = kwargs.get("shape_a", shape_a)
    shape_b = kwargs.get("shape_b", shape_b)
    method = kwargs.get("method", method)
    tolerance = kwargs.get("tolerance", tolerance)

    return solve_least_squares_dd(shape_a, shape_b, method, tolerance)


# Keep the original name as an alias
solve_least_squares = solve_least_squares_factory


# =============================================================================
# Iterative Linear System Solver Factory Functions
# =============================================================================


def solve_iterative_ff(
    shape_a,
    shape_b,
    method=None,
    max_iterations=1000,
    tolerance=1e-6,
    use_initial_guess=False,
):
    """Create an iterative linear system solver block for float data.

    Args:
        shape_a: Coefficient matrix A shape [rows, cols]
        shape_b: Right-hand side b shape [rows, cols] (cols=1 for vector)
        method: Iterative method (AUTO, CG, BICGSTAB, GMRES, LSCG)
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        use_initial_guess: Use input solution as initial guess

    Returns:
        solve_iterative_sync_f block instance
    """
    if "solve_iterative_sync_f" in globals() and "iterative_method" in globals():
        method_enum = iterative_method.AUTO if method is None else method
        return solve_iterative_sync_f.make(
            shape_a, shape_b, method_enum, max_iterations, tolerance, use_initial_guess
        )
    else:
        raise RuntimeError("solve_iterative_sync_f not available - check C++ bindings")


def solve_iterative_dd(
    shape_a,
    shape_b,
    method=None,
    max_iterations=1000,
    tolerance=1e-6,
    use_initial_guess=False,
):
    """Create an iterative linear system solver block for double data.

    Args:
        shape_a: Coefficient matrix A shape [rows, cols]
        shape_b: Right-hand side b shape [rows, cols] (cols=1 for vector)
        method: Iterative method (AUTO, CG, BICGSTAB, GMRES, LSCG)
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        use_initial_guess: Use input solution as initial guess

    Returns:
        solve_iterative_sync_d block instance
    """
    if "solve_iterative_sync_d" in globals() and "iterative_method" in globals():
        method_enum = iterative_method.AUTO if method is None else method
        return solve_iterative_sync_d.make(
            shape_a, shape_b, method_enum, max_iterations, tolerance, use_initial_guess
        )
    else:
        raise RuntimeError("solve_iterative_sync_d not available - check C++ bindings")


def solve_iterative_cc(
    shape_a,
    shape_b,
    method=None,
    max_iterations=1000,
    tolerance=1e-6,
    use_initial_guess=False,
):
    """Create an iterative linear system solver block for complex float data.

    Args:
        shape_a: Coefficient matrix A shape [rows, cols]
        shape_b: Right-hand side b shape [rows, cols] (cols=1 for vector)
        method: Iterative method (AUTO, CG, BICGSTAB, GMRES, LSCG)
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        use_initial_guess: Use input solution as initial guess

    Returns:
        solve_iterative_sync_cf block instance
    """
    if "solve_iterative_sync_cf" in globals() and "iterative_method" in globals():
        method_enum = iterative_method.AUTO if method is None else method
        return solve_iterative_sync_cf.make(
            shape_a, shape_b, method_enum, max_iterations, tolerance, use_initial_guess
        )
    else:
        raise RuntimeError("solve_iterative_sync_cf not available - check C++ bindings")


def solve_iterative_zz(
    shape_a,
    shape_b,
    method=None,
    max_iterations=1000,
    tolerance=1e-6,
    use_initial_guess=False,
):
    """Create an iterative linear system solver block for complex double data.

    Args:
        shape_a: Coefficient matrix A shape [rows, cols]
        shape_b: Right-hand side b shape [rows, cols] (cols=1 for vector)
        method: Iterative method (AUTO, CG, BICGSTAB, GMRES, LSCG)
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        use_initial_guess: Use input solution as initial guess

    Returns:
        solve_iterative_sync_cd block instance
    """
    if "solve_iterative_sync_cd" in globals() and "iterative_method" in globals():
        method_enum = iterative_method.AUTO if method is None else method
        return solve_iterative_sync_cd.make(
            shape_a, shape_b, method_enum, max_iterations, tolerance, use_initial_guess
        )
    else:
        raise RuntimeError("solve_iterative_sync_cd not available - check C++ bindings")


# =============================================================================
# Vector Projection Factory Functions
# =============================================================================


def vector_projection_ff(shape_a, shape_b):
    """Create a vector projection block for float data.

    Args:
        shape_a: Shape of vector A to be projected [length]
        shape_b: Shape of vector B (target direction) [length]

    Returns:
        vector_projection_sync_f block instance

    Mathematical operation: proj_b(a) = (a·b / b·b) × b
    """
    if "vector_projection_sync_f" in globals():
        return vector_projection_sync_f.make(shape_a, shape_b)
    else:
        raise RuntimeError(
            "vector_projection_sync_f C++ class not available in Python bindings"
        )


def vector_projection_dd(shape_a, shape_b):
    """Create a vector projection block for double data.

    Args:
        shape_a: Shape of vector A to be projected [length]
        shape_b: Shape of vector B (target direction) [length]

    Returns:
        vector_projection_sync_d block instance

    Mathematical operation: proj_b(a) = (a·b / b·b) × b
    """
    if "vector_projection_sync_d" in globals():
        return vector_projection_sync_d.make(shape_a, shape_b)
    else:
        raise RuntimeError(
            "vector_projection_sync_d C++ class not available in Python bindings"
        )


def vector_projection_cc(shape_a, shape_b):
    """Create a vector projection block for complex float data.

    Args:
        shape_a: Shape of vector A to be projected [length]
        shape_b: Shape of vector B (target direction) [length]

    Returns:
        vector_projection_sync_c block instance

    Mathematical operation: proj_b(a) = (a·b* / b·b*) × b

    WARNING: Complex vector projections currently have a known issue where
    they return zero vectors. Use real-valued variants (vector_projection_ff/dd)
    for production applications. TODO: Fix complex conjugate dot product.
    """
    if "vector_projection_sync_c" in globals():
        return vector_projection_sync_c.make(shape_a, shape_b)
    else:
        raise RuntimeError(
            "vector_projection_sync_c C++ class not available in Python bindings"
        )


def vector_projection_zz(shape_a, shape_b):
    """Create a vector projection block for complex double data.

    Args:
        shape_a: Shape of vector A to be projected [length]
        shape_b: Shape of vector B (target direction) [length]

    Returns:
        vector_projection_sync_cd block instance

    Mathematical operation: proj_b(a) = (a·b* / b·b*) × b

    WARNING: Complex vector projections currently have a known issue where
    they return zero vectors. Use real-valued variants (vector_projection_ff/dd)
    for production applications. TODO: Fix complex conjugate dot product.
    """
    if "vector_projection_sync_cd" in globals():
        return vector_projection_sync_cd.make(shape_a, shape_b)
    else:
        raise RuntimeError(
            "vector_projection_sync_cd C++ class not available in Python bindings"
        )


def vector_projection(shape_a, shape_b):
    """Create a vector projection block (defaults to double precision).

    Args:
        shape_a: Shape of vector A to be projected [length]
        shape_b: Shape of vector B (target direction) [length]

    Returns:
        vector_projection_sync_d block instance

    Mathematical operation: proj_b(a) = (a·b / b·b) × b

    The vector projection represents the component of vector A that lies in the
    direction of vector B. Geometrically, it's the "shadow" that vector A casts
    onto the line defined by vector B.
    """
    return vector_projection_dd(shape_a, shape_b)


# =============================================================================
# Solve Iterative Functions
# =============================================================================


def solve_iterative_ff(
    shape_a=[2, 2],
    shape_b=[2, 1],
    method=None,
    max_iterations=1000,
    tolerance=1e-6,
    use_initial_guess=False,
):
    """Create an iterative linear system solver block (float precision).

    Args:
        shape_a: Coefficient matrix A shape [rows, cols]
        shape_b: Right-hand side b shape [rows, cols] (cols=1 for vector)
        method: Iterative method (AUTO, CG, BICGSTAB, GMRES, LSCG) or int
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        use_initial_guess: Use input solution as initial guess

    Returns:
        solve_iterative_sync_f block instance
    """
    if "solve_iterative_sync_f" in globals() and "iterative_method" in globals():
        if method is None:
            method_enum = linalg_python.iterative_method.AUTO
        elif hasattr(method, "value"):
            method_enum = method
        else:
            method_enum = linalg_python.iterative_method(int(method))

        return linalg_python.solve_iterative_sync_f.make(
            shape_a, shape_b, method_enum, max_iterations, tolerance, use_initial_guess
        )
    else:
        raise RuntimeError(
            "solve_iterative_sync_f C++ class not available in Python bindings"
        )


def solve_iterative_dd(
    shape_a=[2, 2],
    shape_b=[2, 1],
    method=None,
    max_iterations=1000,
    tolerance=1e-12,
    use_initial_guess=False,
):
    """Create an iterative linear system solver block (double precision).

    Args:
        shape_a: Coefficient matrix A shape [rows, cols]
        shape_b: Right-hand side b shape [rows, cols] (cols=1 for vector)
        method: Iterative method (AUTO, CG, BICGSTAB, GMRES, LSCG) or int
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        use_initial_guess: Use input solution as initial guess

    Returns:
        solve_iterative_sync_d block instance
    """
    if "solve_iterative_sync_d" in globals() and "iterative_method" in globals():
        if method is None:
            method_enum = linalg_python.iterative_method.AUTO
        elif hasattr(method, "value"):
            method_enum = method
        else:
            method_enum = linalg_python.iterative_method(int(method))

        return linalg_python.solve_iterative_sync_d.make(
            shape_a, shape_b, method_enum, max_iterations, tolerance, use_initial_guess
        )
    else:
        raise RuntimeError(
            "solve_iterative_sync_d C++ class not available in Python bindings"
        )


def solve_iterative(
    shape_a=[2, 2],
    shape_b=[2, 1],
    method=None,
    max_iterations=1000,
    tolerance=1e-6,
    use_initial_guess=False,
):
    """Create an iterative linear system solver block (defaults to float precision for GNU Radio compatibility).

    Args:
        shape_a: Coefficient matrix A shape [rows, cols]
        shape_b: Right-hand side b shape [rows, cols] (cols=1 for vector)
        method: Iterative method (AUTO, CG, BICGSTAB, GMRES, LSCG) or int
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        use_initial_guess: Use input solution as initial guess

    Returns:
        solve_iterative_sync_f block instance

    Mathematical operations:
        - CG: Conjugate Gradient (for symmetric positive definite systems)
        - BICGSTAB: Bi-conjugate gradient stabilized (general systems)
        - GMRES: Generalized minimal residual (general systems)
        - LSCG: Least squares CG (overdetermined systems)
        - AUTO: Automatically select method based on matrix properties

    The iterative solver minimizes ||Ax - b|| where A is the coefficient matrix
    and b is the right-hand side vector. For square systems, it solves Ax = b.
    For overdetermined systems (more rows than columns), it finds the least
    squares solution that minimizes the residual norm.
    """
    return solve_iterative_ff(
        shape_a, shape_b, method, max_iterations, tolerance, use_initial_guess
    )


# =============================================================================
# Vector Correlate Factory Functions
# =============================================================================


def vector_correlate_ff(shape_a, shape_b, mode=None, normalize=False):
    """Create a vector correlation block for float data.

    Args:
        shape_a: Shape of first input vector [length]
        shape_b: Shape of second input vector [length]
        mode: Correlation mode (FULL, VALID, SAME) - defaults to FULL
        normalize: Normalize correlation output (default: False)

    Returns:
        vector_correlate_sync_f block instance

    Mathematical operation: Computes cross-correlation R_ab(k) between vectors a and b.
    Used for signal detection, pattern matching, and time delay estimation.
    """
    if "vector_correlate_sync_f" in globals() and "CorrelateMode" in globals():
        if mode is None:
            mode = CorrelateMode.FULL
        return vector_correlate_sync_f.make(shape_a, shape_b, mode, normalize)
    else:
        raise RuntimeError(
            "vector_correlate_sync_f C++ class not available in Python bindings"
        )


def vector_correlate_dd(shape_a, shape_b, mode=None, normalize=False):
    """Create a vector correlation block for double data.

    Args:
        shape_a: Shape of first input vector [length]
        shape_b: Shape of second input vector [length]
        mode: Correlation mode (FULL, VALID, SAME) - defaults to FULL
        normalize: Normalize correlation output (default: False)

    Returns:
        vector_correlate_sync_d block instance

    Mathematical operation: Computes cross-correlation R_ab(k) between vectors a and b.
    Used for signal detection, pattern matching, and time delay estimation.
    """
    if "vector_correlate_sync_d" in globals() and "CorrelateMode" in globals():
        if mode is None:
            mode = CorrelateMode.FULL
        return vector_correlate_sync_d.make(shape_a, shape_b, mode, normalize)
    else:
        raise RuntimeError(
            "vector_correlate_sync_d C++ class not available in Python bindings"
        )


def vector_correlate_cc(shape_a, shape_b, mode=None, normalize=False):
    """Create a vector correlation block for complex float data.

    Args:
        shape_a: Shape of first input vector [length]
        shape_b: Shape of second input vector [length]
        mode: Correlation mode (FULL, VALID, SAME) - defaults to FULL
        normalize: Normalize correlation output (default: False)

    Returns:
        vector_correlate_sync_c block instance

    Mathematical operation: Computes cross-correlation R_ab(k) between vectors a and b.
    For complex signals, uses proper conjugation: R_ab(k) = Σ a[n] * conj(b[n+k])
    """
    if "vector_correlate_sync_c" in globals() and "CorrelateMode" in globals():
        if mode is None:
            mode = CorrelateMode.FULL
        return vector_correlate_sync_c.make(shape_a, shape_b, mode, normalize)
    else:
        raise RuntimeError(
            "vector_correlate_sync_c C++ class not available in Python bindings"
        )


def vector_correlate_zz(shape_a, shape_b, mode=None, normalize=False):
    """Create a vector correlation block for complex double data.

    Args:
        shape_a: Shape of first input vector [length]
        shape_b: Shape of second input vector [length]
        mode: Correlation mode (FULL, VALID, SAME) - defaults to FULL
        normalize: Normalize correlation output (default: False)

    Returns:
        vector_correlate_sync_cd block instance

    Mathematical operation: Computes cross-correlation R_ab(k) between vectors a and b.
    For complex signals, uses proper conjugation: R_ab(k) = Σ a[n] * conj(b[n+k])
    """
    if "vector_correlate_sync_cd" in globals() and "CorrelateMode" in globals():
        if mode is None:
            mode = CorrelateMode.FULL
        return vector_correlate_sync_cd.make(shape_a, shape_b, mode, normalize)
    else:
        raise RuntimeError(
            "vector_correlate_sync_cd C++ class not available in Python bindings"
        )


def vector_correlate_factory(
    shape_a=None, shape_b=None, mode=None, normalize=False, *args, **kwargs
):
    """Create a vector correlation block (defaults to double precision).

    Args:
        shape_a: Shape of first input vector [length] - defaults to [3]
        shape_b: Shape of second input vector [length] - defaults to [3]
        mode: Correlation mode (FULL, VALID, SAME) - defaults to FULL
        normalize: Normalize correlation output (default: False)

    Returns:
        vector_correlate_sync_d block instance

    Cross-correlation is fundamental for:
    - Signal detection and synchronization
    - Pattern matching and template matching
    - Time delay estimation between signals
    - Echo and multipath analysis
    - Matched filtering in communications

    Output modes:
    - FULL: Returns correlation of length 2*N-1 with all possible lags
    - VALID: Returns correlation only where vectors fully overlap
    - SAME: Returns correlation of same size as input vectors (central portion)

    Usage:
    - vector_correlate() -> defaults: shape_a=[3], shape_b=[3], mode=FULL
    - vector_correlate([5]) -> shape_a=[5], shape_b=[5], mode=FULL
    - vector_correlate([4], [6]) -> shape_a=[4], shape_b=[6], mode=FULL
    - vector_correlate([3], [3], mode=SAME) -> specified mode
    """
    # Handle different calling conventions with sensible defaults
    if shape_a is None:
        shape_a = [3]
    if shape_b is None:
        shape_b = [3]

    # Override with keyword arguments if provided
    shape_a = kwargs.get("shape_a", shape_a)
    shape_b = kwargs.get("shape_b", shape_b)
    mode = kwargs.get("mode", mode)
    normalize = kwargs.get("normalize", normalize)

    return vector_correlate_dd(shape_a, shape_b, mode, normalize)


# Keep the original name as an alias
vector_correlate = vector_correlate_factory


# Vector to Matrix conversion precision-specific functions
def vector_to_matrix_ff(input_shape, output_shape):
    """Create a float precision vector-to-matrix conversion block.

    Args:
        input_shape: Shape of input vector [length]
        output_shape: Shape of output matrix [rows, cols]

    Returns:
        vector_to_matrix_sync_f block instance

    Converts a vector stream to a matrix stream by reshaping the data.
    Elements are arranged in column-major order (Eigen default).
    Input vector length must equal output matrix elements (rows * cols).
    """
    if "vector_to_matrix_sync_f" in globals():
        return vector_to_matrix_sync_f.make(input_shape, output_shape)
    else:
        raise RuntimeError(
            "vector_to_matrix_sync_f C++ class not available in Python bindings"
        )


def vector_to_matrix_dd(input_shape, output_shape):
    """Create a double precision vector-to-matrix conversion block.

    Args:
        input_shape: Shape of input vector [length]
        output_shape: Shape of output matrix [rows, cols]

    Returns:
        vector_to_matrix_sync_d block instance
    """
    if "vector_to_matrix_sync_d" in globals():
        return vector_to_matrix_sync_d.make(input_shape, output_shape)
    else:
        raise RuntimeError(
            "vector_to_matrix_sync_d C++ class not available in Python bindings"
        )


def vector_to_matrix_cc(input_shape, output_shape):
    """Create a complex float precision vector-to-matrix conversion block.

    Args:
        input_shape: Shape of input vector [length]
        output_shape: Shape of output matrix [rows, cols]

    Returns:
        vector_to_matrix_sync_cf block instance
    """
    if "vector_to_matrix_sync_cf" in globals():
        return vector_to_matrix_sync_cf.make(input_shape, output_shape)
    else:
        raise RuntimeError(
            "vector_to_matrix_sync_cf C++ class not available in Python bindings"
        )


def vector_to_matrix_zz(input_shape, output_shape):
    """Create a complex double precision vector-to-matrix conversion block.

    Args:
        input_shape: Shape of input vector [length]
        output_shape: Shape of output matrix [rows, cols]

    Returns:
        vector_to_matrix_sync_cd block instance
    """
    if "vector_to_matrix_sync_cd" in globals():
        return vector_to_matrix_sync_cd.make(input_shape, output_shape)
    else:
        raise RuntimeError(
            "vector_to_matrix_sync_cd C++ class not available in Python bindings"
        )


def vector_to_matrix(input_shape, output_shape):
    """Create a vector-to-matrix conversion block (defaults to double precision).

    Args:
        input_shape: Shape of input vector [length]
        output_shape: Shape of output matrix [rows, cols]

    Returns:
        vector_to_matrix_sync_d block instance

    This is a fundamental data manipulation operation for:
    - Reformatting 1D signal data into 2D matrix form
    - Preparing data for matrix operations
    - Converting streaming vectors to matrix blocks
    - Signal processing applications requiring matrix representations

    The conversion preserves all data elements and uses column-major ordering
    which is optimal for linear algebra operations with Eigen backend.
    """
    return vector_to_matrix_dd(input_shape, output_shape)


# =============================================================================
# Array Slice Factory Functions
# =============================================================================


def array_slice_ff(input_shape, slice_specs):
    """Create an array slice block for float data (sync interface).

    Args:
        input_shape: Shape of input array [dimensions]
        slice_specs: List of SliceSpec objects defining slice parameters,
                    or list of tuples (start, stop, step) to auto-create SliceSpec

    Returns:
        array_slice_sync_f block instance

    Array slicing allows extracting subarrays from input arrays using
    Python-like slice notation with start, stop, and step parameters.
    """
    if "array_slice_sync_f" in globals():
        # Convert tuples to SliceSpec objects if needed
        if slice_specs and isinstance(slice_specs[0], (tuple, list)):
            from . import linalg_python

            slice_specs = [linalg_python.SliceSpec(*spec) for spec in slice_specs]
        return array_slice_sync_f.make(input_shape, slice_specs)
    else:
        raise RuntimeError(
            "array_slice_sync_f C++ class not available in Python bindings"
        )


def array_slice_dd(input_shape, slice_specs):
    """Create an array slice block for double data (sync interface).

    Args:
        input_shape: Shape of input array [dimensions]
        slice_specs: List of SliceSpec_d objects defining slice parameters,
                    or list of tuples (start, stop, step) to auto-create SliceSpec_d

    Returns:
        array_slice_sync_d block instance
    """
    if "array_slice_sync_d" in globals():
        # Convert tuples to SliceSpec_d objects if needed
        if slice_specs and isinstance(slice_specs[0], (tuple, list)):
            from . import linalg_python

            slice_specs = [linalg_python.SliceSpec_d(*spec) for spec in slice_specs]
        return array_slice_sync_d.make(input_shape, slice_specs)
    else:
        raise RuntimeError(
            "array_slice_sync_d C++ class not available in Python bindings"
        )


def array_slice_cc(input_shape, slice_specs):
    """Create an array slice block for complex float data (sync interface).

    Args:
        input_shape: Shape of input array [dimensions]
        slice_specs: List of SliceSpec_cf objects defining slice parameters,
                    or list of tuples (start, stop, step) to auto-create SliceSpec_cf

    Returns:
        array_slice_sync_cf block instance
    """
    if "array_slice_sync_cf" in globals():
        # Convert tuples to SliceSpec_cf objects if needed
        if slice_specs and isinstance(slice_specs[0], (tuple, list)):
            from . import linalg_python

            slice_specs = [linalg_python.SliceSpec_cf(*spec) for spec in slice_specs]
        return array_slice_sync_cf.make(input_shape, slice_specs)
    else:
        raise RuntimeError(
            "array_slice_sync_cf C++ class not available in Python bindings"
        )


def array_slice_zz(input_shape, slice_specs):
    """Create an array slice block for complex double data (sync interface).

    Args:
        input_shape: Shape of input array [dimensions]
        slice_specs: List of SliceSpec_cd objects defining slice parameters,
                    or list of tuples (start, stop, step) to auto-create SliceSpec_cd

    Returns:
        array_slice_sync_cd block instance
    """
    if "array_slice_sync_cd" in globals():
        # Convert tuples to SliceSpec_cd objects if needed
        if slice_specs and isinstance(slice_specs[0], (tuple, list)):
            from . import linalg_python

            slice_specs = [linalg_python.SliceSpec_cd(*spec) for spec in slice_specs]
        return array_slice_sync_cd.make(input_shape, slice_specs)
    else:
        raise RuntimeError(
            "array_slice_sync_cd C++ class not available in Python bindings"
        )


# PDU variants for array_slice
def array_slice_pdu_ff(input_shapes, slice_specs):
    """Create an array slice block for float data (PDU interface).

    Args:
        input_shapes: List of input array shapes
        slice_specs: List of SliceSpec objects defining slice parameters,
                    or list of tuples (start, stop, step) to auto-create SliceSpec

    Returns:
        array_slice_pdu_f block instance
    """
    if "array_slice_pdu_f" in globals():
        # Convert tuples to SliceSpec objects if needed
        if slice_specs and isinstance(slice_specs[0], (tuple, list)):
            from . import linalg_python

            slice_specs = [linalg_python.SliceSpec(*spec) for spec in slice_specs]
        return array_slice_pdu_f.make(input_shapes, slice_specs)
    else:
        raise RuntimeError(
            "array_slice_pdu_f C++ class not available in Python bindings"
        )


def array_slice_pdu_dd(input_shapes, slice_specs):
    """Create an array slice block for double data (PDU interface).

    Args:
        input_shapes: List of input array shapes
        slice_specs: List of SliceSpec_d objects defining slice parameters,
                    or list of tuples (start, stop, step) to auto-create SliceSpec_d

    Returns:
        array_slice_pdu_d block instance
    """
    if "array_slice_pdu_d" in globals():
        # Convert tuples to SliceSpec_d objects if needed
        if slice_specs and isinstance(slice_specs[0], (tuple, list)):
            from . import linalg_python

            slice_specs = [linalg_python.SliceSpec_d(*spec) for spec in slice_specs]
        return array_slice_pdu_d.make(input_shapes, slice_specs)
    else:
        raise RuntimeError(
            "array_slice_pdu_d C++ class not available in Python bindings"
        )


def array_slice_pdu_cc(input_shapes, slice_specs):
    """Create an array slice block for complex float data (PDU interface).

    Args:
        input_shapes: List of input array shapes
        slice_specs: List of SliceSpec_cf objects defining slice parameters,
                    or list of tuples (start, stop, step) to auto-create SliceSpec_cf

    Returns:
        array_slice_pdu_cf block instance
    """
    if "array_slice_pdu_cf" in globals():
        # Convert tuples to SliceSpec_cf objects if needed
        if slice_specs and isinstance(slice_specs[0], (tuple, list)):
            from . import linalg_python

            slice_specs = [linalg_python.SliceSpec_cf(*spec) for spec in slice_specs]
        return array_slice_pdu_cf.make(input_shapes, slice_specs)
    else:
        raise RuntimeError(
            "array_slice_pdu_cf C++ class not available in Python bindings"
        )


def array_slice_pdu_zz(input_shapes, slice_specs):
    """Create an array slice block for complex double data (PDU interface).

    Args:
        input_shapes: List of input array shapes
        slice_specs: List of SliceSpec_cd objects defining slice parameters,
                    or list of tuples (start, stop, step) to auto-create SliceSpec_cd

    Returns:
        array_slice_pdu_cd block instance
    """
    if "array_slice_pdu_cd" in globals():
        # Convert tuples to SliceSpec_cd objects if needed
        if slice_specs and isinstance(slice_specs[0], (tuple, list)):
            from . import linalg_python

            slice_specs = [linalg_python.SliceSpec_cd(*spec) for spec in slice_specs]
        return array_slice_pdu_cd.make(input_shapes, slice_specs)
    else:
        raise RuntimeError(
            "array_slice_pdu_cd C++ class not available in Python bindings"
        )


def array_slice(input_shape, slice_specs, interface="sync", precision="float"):
    """Generic array slice factory function.

    Args:
        input_shape: Shape of input array [dimensions] (sync) or list of shapes (PDU)
        slice_specs: List of SliceSpec objects defining slice parameters
        interface: "sync" or "pdu"
        precision: "float", "double", "complex_float", "complex_double"

    Returns:
        GNU Radio block for array slicing operations

    Array slicing provides Python-like array indexing capabilities for GNU Radio
    data streams. This is useful for:
    - Extracting specific portions of matrices or vectors
    - Implementing windowing operations
    - Subarray processing in signal processing applications
    - Data selection and filtering based on indices

    Example usage:
        # Create SliceSpec for elements 2:8:2 (start=2, stop=8, step=2)
        slice_spec = SliceSpec(start=2, stop=8, step=2)
        block = array_slice([10], [slice_spec])
    """
    if interface == "sync":
        if precision == "float":
            return array_slice_ff(input_shape, slice_specs)
        elif precision == "double":
            return array_slice_dd(input_shape, slice_specs)
        elif precision == "complex_float":
            return array_slice_cc(input_shape, slice_specs)
        elif precision == "complex_double":
            return array_slice_zz(input_shape, slice_specs)
        else:
            raise ValueError(f"Unknown precision: {precision}")
    elif interface == "pdu":
        if precision == "float":
            return array_slice_pdu_ff(input_shape, slice_specs)
        elif precision == "double":
            return array_slice_pdu_dd(input_shape, slice_specs)
        elif precision == "complex_float":
            return array_slice_pdu_cc(input_shape, slice_specs)
        elif precision == "complex_double":
            return array_slice_pdu_zz(input_shape, slice_specs)
        else:
            raise ValueError(f"Unknown precision: {precision}")
    else:
        raise ValueError(f"Unknown interface: {interface}")


# Matrix eigenvalues factory functions
def matrix_eigenvalues_ff(shape, sort_by_magnitude=True):
    """Create matrix_eigenvalues block (float precision)."""
    if "matrix_eigenvalues_sync_f" in globals():
        return matrix_eigenvalues_sync_f.make(shape, sort_by_magnitude)
    else:
        raise RuntimeError("matrix_eigenvalues_sync_f C++ class not available")


def matrix_eigenvalues_dd(shape, sort_by_magnitude=True):
    """Create matrix_eigenvalues block (double precision)."""
    if "matrix_eigenvalues_sync_d" in globals():
        return matrix_eigenvalues_sync_d.make(shape, sort_by_magnitude)
    else:
        raise RuntimeError("matrix_eigenvalues_sync_d C++ class not available")


def matrix_eigenvalues_cc(shape, sort_by_magnitude=True):
    """Create matrix_eigenvalues block (complex float precision)."""
    if "matrix_eigenvalues_sync_cf" in globals():
        return matrix_eigenvalues_sync_cf.make(shape, sort_by_magnitude)
    else:
        raise RuntimeError("matrix_eigenvalues_sync_cf C++ class not available")


def matrix_eigenvalues_zz(shape, sort_by_magnitude=True):
    """Create matrix_eigenvalues block (complex double precision)."""
    if "matrix_eigenvalues_sync_cd" in globals():
        return matrix_eigenvalues_sync_cd.make(shape, sort_by_magnitude)
    else:
        raise RuntimeError("matrix_eigenvalues_sync_cd C++ class not available")


def matrix_eigenvalues(
    shape, sort_by_magnitude=True, precision="float", interface="sync"
):
    """Generic matrix_eigenvalues factory function.

    Args:
        shape: Matrix dimensions [N, N] (must be square)
        sort_by_magnitude: Sort eigenvalues by magnitude (descending)
        precision: "float", "double", "complex_float", "complex_double"
        interface: "sync" or "pdu"

    Returns:
        GNU Radio block for computing matrix eigenvalues
    """
    if interface == "sync":
        if precision == "float":
            return matrix_eigenvalues_ff(shape, sort_by_magnitude)
        elif precision == "double":
            return matrix_eigenvalues_dd(shape, sort_by_magnitude)
        elif precision == "complex_float":
            return matrix_eigenvalues_cc(shape, sort_by_magnitude)
        elif precision == "complex_double":
            return matrix_eigenvalues_zz(shape, sort_by_magnitude)
        else:
            raise ValueError(f"Unknown precision: {precision}")
    else:
        raise ValueError(f"Unknown interface: {interface}")


# solve_triangular convenience functions
def solve_triangular_ff(shape_matrix, shape_rhs, tri_type=None, unit_diagonal=False):
    """Create solve_triangular block (float precision)."""
    if "solve_triangular_sync_f" in globals():
        if tri_type is None:
            tri_type = TriangularType.LOWER
        return solve_triangular_sync_f.make(
            shape_matrix, shape_rhs, tri_type, unit_diagonal
        )
    else:
        raise RuntimeError("solve_triangular_sync_f C++ class not available")


def solve_triangular_dd(shape_matrix, shape_rhs, tri_type=None, unit_diagonal=False):
    """Create solve_triangular block (double precision)."""
    if "solve_triangular_sync_d" in globals():
        if tri_type is None:
            tri_type = TriangularType.LOWER
        return solve_triangular_sync_d.make(
            shape_matrix, shape_rhs, tri_type, unit_diagonal
        )
    else:
        raise RuntimeError("solve_triangular_sync_d C++ class not available")


def solve_triangular_cc(shape_matrix, shape_rhs, tri_type=None, unit_diagonal=False):
    """Create solve_triangular block (complex float precision)."""
    if "solve_triangular_sync_cf" in globals():
        if tri_type is None:
            tri_type = TriangularType.LOWER
        return solve_triangular_sync_cf.make(
            shape_matrix, shape_rhs, tri_type, unit_diagonal
        )
    else:
        raise RuntimeError("solve_triangular_sync_cf C++ class not available")


def solve_triangular_zz(shape_matrix, shape_rhs, tri_type=None, unit_diagonal=False):
    """Create solve_triangular block (complex double precision)."""
    if "solve_triangular_sync_cd" in globals():
        if tri_type is None:
            tri_type = TriangularType.LOWER
        return solve_triangular_sync_cd.make(
            shape_matrix, shape_rhs, tri_type, unit_diagonal
        )
    else:
        raise RuntimeError("solve_triangular_sync_cd C++ class not available")


def solve_triangular(
    shape_matrix,
    shape_rhs,
    tri_type=None,
    unit_diagonal=False,
    precision="float",
    interface="sync",
):
    """Generic solve_triangular factory function.

    Args:
        shape_matrix: Matrix dimensions [N, N] (must be square)
        shape_rhs: Right-hand side dimensions [N, K] or [N] for vector
        tri_type: Type of triangular matrix (LOWER or UPPER)
        unit_diagonal: If true, assume diagonal elements are 1
        precision: "float", "double", "complex_float", "complex_double"
        interface: "sync" or "pdu"

    Returns:
        GNU Radio block for solving triangular linear systems
    """
    if interface == "sync":
        if precision == "float":
            return solve_triangular_ff(shape_matrix, shape_rhs, tri_type, unit_diagonal)
        elif precision == "double":
            return solve_triangular_dd(shape_matrix, shape_rhs, tri_type, unit_diagonal)
        elif precision == "complex_float":
            return solve_triangular_cc(shape_matrix, shape_rhs, tri_type, unit_diagonal)
        elif precision == "complex_double":
            return solve_triangular_zz(shape_matrix, shape_rhs, tri_type, unit_diagonal)
        else:
            raise ValueError(f"Unknown precision: {precision}")
    else:
        raise ValueError(f"Unknown interface: {interface}")


# Note: We don't define __all__ to allow importing both convenience functions
# and raw C++ classes. This enables both styles:
# - from gnuradio.linalg import matrix_add_ff  (convenience function)
# - from gnuradio.linalg import matrix_add_sync_f  (raw C++ class)
# - from gnuradio.linalg import matrix_add  (generic factory function)
