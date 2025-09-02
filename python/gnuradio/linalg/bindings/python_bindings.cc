/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include <pybind11/pybind11.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

namespace py = pybind11;

// Headers for binding functions
/**************************************/
// The following comment block is used for
// gr_modtool to insert function prototypes
// Please do not delete
/**************************************/
// BINDING_FUNCTION_PROTOTYPES(
// Core decomposition blocks
void bind_decomp_svd(py::module &m);
void bind_decomp_eigen(py::module &m);
void bind_decomp_lu(py::module &m);
void bind_decomp_qr(py::module &m);
void bind_decomp_cholesky(py::module &m);
void bind_decomp_hessenberg(py::module &m);
void bind_decomp_schur(py::module &m); // Added

// Matrix operations
void bind_matrix_add(py::module &m);
void bind_matrix_subtract(py::module &m);
void bind_matrix_multiply(py::module &m);
void bind_matrix_transpose(py::module &m);
void bind_matrix_determinant(py::module &m);
void bind_matrix_trace(py::module &m);
void bind_matrix_rank(py::module &m);
void bind_matrix_diag(py::module &m);
void bind_matrix_inverse(py::module &m);
void bind_matrix_pseudo_inverse(py::module &m);
void bind_matrix_pseudo_determinant(py::module &m);
void bind_matrix_exp(py::module &m); // Re-enabled: template binding fixed

// New matrix operations
void bind_matrix_elementwise_multiply(py::module &m);
void bind_matrix_elementwise_divide(py::module &m);
void bind_matrix_power(py::module &m);
void bind_matrix_condition_number(py::module &m);
void bind_matrix_norm(py::module &m);
void bind_matrix_reshape(py::module &m);
void bind_matrix_solve(py::module &m);
void bind_matrix_source_const(py::module &m);
void bind_matrix_kronecker_product(py::module &m);
void bind_solve_triangular(py::module &m);

// New advanced matrix functions
void bind_matrix_log(py::module &m);
void bind_matrix_sqrt(py::module &m);
void bind_matrix_sin(py::module &m);
void bind_matrix_cos(py::module &m);
void bind_matrix_tan(py::module &m);

// Statistical analysis blocks - DISABLED due to incomplete implementations
// void bind_matrix_covariance(py::module &m);  // DISABLED: incomplete
// implementation void bind_matrix_correlation(py::module &m); // DISABLED:
// incomplete implementation
void bind_matrix_eigenvalues(py::module &m);

// Vector operations
void bind_vector_outer_product(py::module &m);
void bind_vector_norm(py::module &m);
void bind_vector_normalize(py::module &m);
void bind_vector_cross_product(py::module &m);
void bind_vector_angle(py::module &m);
void bind_dot_product(py::module &m);
// void bind_vector_dot(py::module &m);  // DISABLED: vector_dot_python.cc
// removed - dot_product serves same purpose
void bind_vector_projection(py::module &m);
void bind_vector_correlate(py::module &m);
void bind_vector_to_matrix(py::module &m);

// Matrix generators
void bind_eye(py::module &m);   // Fixed template binding issues
void bind_zeros(py::module &m); // Fixed template binding issues
void bind_ones(py::module &m);  // Fixed template binding issues
void bind_matrix_hermitian(py::module &m);
// void bind_matrix_exp(py::module& m);                 // Disabled
// void bind_decomp_schur(py::module& m);               // Disabled
// void bind_decomp_hessenberg(py::module& m);          // Disabled
// void bind_vector_angle(py::module& m);               // Disabled
// void bind_vector_projection(py::module& m);          // Disabled
// void bind_vector_product_inner(py::module& m);       // Disabled
// void bind_vector_product_outer(py::module& m);      // Disabled
// void bind_solve_triangular(py::module& m);           // Disabled
void bind_solve_least_squares(py::module &m);
void bind_solve_iterative(py::module &m);
// void bind_vector_to_matrix(py::module& m);           // Disabled
void bind_array_slice(py::module &m);
// void bind_matrix_pseudo_determinant(py::module& m);  // Disabled
// void bind_linalg_base_const_sync(py::module &m);      // Disabled - unused
// class ) END BINDING_FUNCTION_PROTOTYPES

// We need this hack because import_array() returns NULL
// for newer Python versions.
// This function is also necessary because it ensures access to the C API
// and removes a warning.
void *init_numpy() {
  import_array();
  return NULL;
}

PYBIND11_MODULE(linalg_python, m) {
  // Initialize the numpy C API
  // (otherwise we will see segmentation faults)
  init_numpy();

  // Allow access to base block methods
  py::module::import("gnuradio.gr");

  /**************************************/
  // The following comment block is used for
  // gr_modtool to insert binding function calls
  // Please do not delete
  /**************************************/
  // BINDING_FUNCTION_CALLS(
  // Only bind blocks that have working C++ implementations and fixed Python
  // bindings
  bind_matrix_add(m);            // ✓ Fixed with lambda wrappers
  bind_matrix_subtract(m);       // ✓ Enable matrix subtraction binding
  bind_matrix_multiply(m);       // ✓ No default args
  bind_matrix_transpose(m);      // ✓ Enable matrix transpose binding
  bind_matrix_determinant(m);    // ✓ Enable matrix determinant binding
  bind_matrix_trace(m);          // ✓ Enable matrix trace bindings
  bind_matrix_rank(m);           // ✓ Enable matrix rank bindings
  bind_matrix_diag(m);           // ✓ Enable matrix diagonal binding
  bind_matrix_inverse(m);        // ✓ Enable matrix inverse bindings
  bind_matrix_pseudo_inverse(m); // ✓ Enable matrix pseudo inverse bindings
  // bind_matrix_pseudo_determinant(m);
  // bind_matrix_exp(m);
  bind_decomp_svd(m);      // ✓ Enable SVD decomposition bindings
  bind_decomp_eigen(m);    // ✓ Enable eigenvalue decomposition bindings
  bind_decomp_lu(m);       // ✓ Enable LU decomposition bindings
  bind_decomp_qr(m);       // ✓ Enable QR decomposition bindings
  bind_decomp_cholesky(m); // ✓ Enable Cholesky decomposition bindings
  // bind_decomp_hessenberg(m);  // DISABLED: Parameter signature mismatch with
  // C++ implementation bind_decomp_schur(m);
  // bind_matrix_elementwise_multiply(m);
  // bind_matrix_elementwise_divide(m);
  // bind_matrix_power(m);
  // bind_vector_outer_product(m);
  // bind_matrix_condition_number(m);
  bind_matrix_norm(m); // ✓ Fixed with lambda wrappers
  // bind_matrix_reshape(m);
  // bind_matrix_solve(m);
  // bind_matrix_source_const(m);

  // Vector operations
  bind_vector_norm(m);          // ✓ Fixed with lambda wrappers
  bind_vector_normalize(m);     // ✓ Fixed with lambda wrappers
  bind_vector_cross_product(m); // ✓ Enable vector cross product binding
  // bind_vector_angle(m);
  bind_dot_product(m); // ✓ Enable dot product binding
  // bind_vector_projection(m);
  bind_vector_correlate(
      m); // Fixed: already uses lambda wrapper pattern correctly
  // bind_vector_to_matrix(m);
  // bind_vector_dot(m);       // Template issues - needs refactor

  // Matrix generators
  bind_eye(m);   // ✓ Enable eye matrix generator
  bind_zeros(m); // ✓ Enable zeros matrix generator
  bind_ones(m);  // ✓ Enable ones matrix generator

  // Commented out bindings that need fixes
  bind_matrix_kronecker_product(m); // Fixed: refactored to modern framework
  bind_matrix_hermitian(m);
  bind_solve_least_squares(m); // Fixed: applied lambda wrapper pattern
  bind_solve_triangular(m);    // Fixed: refactored to modern framework
  bind_solve_iterative(
      m); // Fixed: already uses lambda wrapper pattern correctly

  // New advanced matrix functions
  bind_matrix_log(m);  // Matrix logarithm using eigenvalue decomposition
  bind_matrix_sqrt(m); // Matrix square root (different from Cholesky)
  bind_matrix_sin(
      m); // Fixed: applied lambda wrapper pattern to handle enum default args
  bind_matrix_cos(
      m); // Fixed: applied lambda wrapper pattern to handle enum default args
  bind_matrix_tan(
      m); // Fixed: applied lambda wrapper pattern to handle enum default args

  // Statistical analysis blocks - DISABLED due to incomplete implementations
  // bind_matrix_covariance(m);  // Statistical covariance matrix computation -
  // DISABLED: incomplete implementation bind_matrix_correlation(m); //
  // Statistical correlation matrix computation (Pearson, Spearman, Kendall) -
  // DISABLED: incomplete implementation
  bind_matrix_eigenvalues(m); // Matrix eigenvalues extraction (efficient
                              // eigenvalue-only computation)

  // Data manipulation operations
  // bind_array_slice(m);  // Temporarily disabled - has default args // Array
  // slicing - fixed Python bindings

  // bind_matrix_exp(m);                    // Disabled due to compilation
  // errors bind_decomp_schur(m);                  // Disabled due to
  // compilation errors bind_decomp_hessenberg(m);             // Disabled due
  // to compilation errors bind_vector_angle(m);                  // Disabled
  // due to compilation errors bind_vector_projection(m);             //
  // Disabled due to compilation errors bind_vector_product_inner(m); //
  // Disabled due to compilation errors bind_vector_product_outer(m); //
  // Disabled due to compilation errors bind_solve_triangular(m); // Disabled
  // due to compilation errors bind_solve_least_squares(m);           //
  // Disabled due to compilation errors bind_vector_to_matrix(m); // Disabled
  // due to compilation errors bind_vector_correlate(m);              //
  // Disabled due to compilation errors
  // bind_matrix_pseudo_determinant(m);     // Disabled due
  // to compilation errors
  // bind_linalg_base_const_sync(m);        // Disabled - unused class with
  // broken bindings ) END BINDING_FUNCTION_CALLS
}