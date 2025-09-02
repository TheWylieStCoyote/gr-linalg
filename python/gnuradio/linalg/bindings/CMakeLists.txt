# Copyright 2020 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

########################################################################
# Check if there is C++ code at all
########################################################################
if(NOT linalg_sources)
    message(STATUS "No C++ sources... skipping python bindings")
    return()
endif(NOT linalg_sources)

########################################################################
# Check for pygccxml
########################################################################
gr_python_check_module_raw("pygccxml" "import pygccxml" PYGCCXML_FOUND)

include(GrPybind)

########################################################################
# Python Bindings
########################################################################

list(APPEND linalg_python_files
    # Python bindings for working C++ blocks only
    
    # Core decomposition blocks
    decomp_svd_python.cc
    decomp_eigen_python.cc
    decomp_lu_python.cc
    decomp_qr_python.cc
    decomp_cholesky_python.cc
    # decomp_hessenberg_python.cc    # DISABLED: Parameter mismatch causes undefined symbol error
    decomp_schur_python.cc            # Added: Schur decomposition Python bindings
    
    # Matrix operations
    matrix_add_python.cc
    matrix_subtract_python.cc
    matrix_multiply_python.cc
    matrix_transpose_python.cc
    matrix_hermitian_python.cc
    matrix_determinant_python.cc
    matrix_trace_python.cc
    matrix_rank_python.cc
    matrix_diag_python.cc
    matrix_inverse_python.cc
    matrix_pseudo_inverse_python.cc
    matrix_pseudo_determinant_python.cc
    matrix_exp_python.cc           # Re-enabled: Python bindings fixed with proper template support
    
    # New matrix operations (recently implemented)
    matrix_elementwise_multiply_python.cc
    matrix_elementwise_divide_python.cc
    matrix_power_python.cc
    matrix_condition_number_python.cc
    matrix_norm_python.cc
    matrix_reshape_python.cc      # Phase 3: Matrix reshape operation for data manipulation
    matrix_solve_python.cc        # Phase 3: Linear system solver (Ax = b)
    matrix_source_const_python.cc # Matrix constant source with sync/PDU variants
    matrix_kronecker_product_python.cc  # Fixed: refactored to modern linalg_base framework
    solve_least_squares_python.cc # Fixed: refactored to modern linalg_base framework
    solve_triangular_python.cc    # Fixed: refactored to modern linalg_base framework
    solve_iterative_python.cc     # Iterative linear system solver (CG, BiCGStab, GMRES, LSCG)
    
    # New advanced matrix functions
    matrix_log_python.cc          # Matrix logarithm using eigenvalue decomposition
    matrix_sqrt_python.cc         # Matrix square root (different from Cholesky) 
    matrix_sin_python.cc          # Matrix sine function
    matrix_cos_python.cc          # Matrix cosine function
    matrix_tan_python.cc          # Matrix tangent function
    
    # Statistical analysis blocks (NEW)
    # matrix_covariance_python.cc   # Statistical covariance matrix computation - DISABLED: incomplete implementation
    # matrix_correlation_python.cc  # Statistical correlation matrix computation (Pearson, Spearman, Kendall) - DISABLED: incomplete implementation  
    matrix_eigenvalues_python.cc  # Matrix eigenvalues extraction (efficient eigenvalue-only computation)
    
    # Vector operations
    vector_outer_product_python.cc
    vector_norm_python.cc          # Fixed template issues - vector norm computation
    vector_normalize_python.cc     # Fixed template issues - vector normalization
    dot_product_python.cc        # Fixed: Refactored to linalg_base framework
    # vector_dot_python.cc removed - C++ implementation doesn't exist, dot_product serves same purpose
    
    # Vector operations (additional)
    vector_cross_product_python.cc
    vector_angle_python.cc
    vector_projection_python.cc
    vector_correlate_python.cc     # Fixed: Complete C++ implementation available
    vector_to_matrix_python.cc     # Fixed: refactored to modern linalg_base framework
    array_slice_python.cc          # Fixed: Python bindings for array slicing operations
    eye_python.cc          # Fixed template binding issues
    zeros_python.cc        # Fixed template binding issues
    ones_python.cc         # Fixed template binding issues
    
    # Core bindings infrastructure
    python_bindings.cc)

gr_pybind_make_oot(linalg ../../../.. gr::linalg "${linalg_python_files}")

# copy bindings extension for use in QA test module
add_custom_command(
    TARGET linalg_python
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:linalg_python>
            ${PROJECT_BINARY_DIR}/test_modules/gnuradio/linalg/)

install(
    TARGETS linalg_python
    DESTINATION ${GR_PYTHON_DIR}/gnuradio/linalg
    COMPONENT pythonapi)
