/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "matrix_solve_impl.h"
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <algorithm>
#include <complex>
#include <gnuradio/gr_complex.h>
#include <gnuradio/io_signature.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/performance_profiler.h>
#include <gnuradio/linalg/types.h>
#include <gnuradio/linalg/utils.h>
#include <limits>
#include <pmt/pmt.h>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace gr {
namespace linalg {

// Base class implementation
template <typename Scalar>
matrix_solve<Scalar>::matrix_solve(const types::shape &shape_a,
                                   const types::shape &shape_b,
                                   solver_method method, double tolerance)
    : linalg_base<Scalar>("matrix_solve", {shape_a, shape_b},
                          compute_output_shapes(shape_a, shape_b),
                          array_broadcast_type::CUSTOM),
      d_shape_a(shape_a), d_shape_b(shape_b), d_method(method),
      d_tolerance(tolerance) {

  // Validate input shapes
  if (shape_a.size() != 2) {
    throw std::invalid_argument("matrix_solve requires 2D matrix A");
  }

  if (shape_b.size() != 2) {
    throw std::invalid_argument("matrix_solve requires 2D matrix/vector b");
  }

  // Check dimension compatibility
  if (shape_a[0] != shape_b[0]) {
    throw std::invalid_argument(
        "matrix_solve: A rows (" + std::to_string(shape_a[0]) +
        ") must match b rows (" + std::to_string(shape_b[0]) + ")");
  }

  // Validate method selection
  if (method == solver_method::LU && shape_a[0] != shape_a[1]) {
    throw std::invalid_argument("LU method requires square matrix A");
  }

  if (method == solver_method::CHOLESKY && shape_a[0] != shape_a[1]) {
    throw std::invalid_argument("Cholesky method requires square matrix A");
  }
}

template <typename Scalar>
matrix_solve<Scalar>::~matrix_solve() {}

template <typename Scalar>
typename matrix_solve<Scalar>::solver_method
matrix_solve<Scalar>::choose_method(int rows_a, int cols_a) const {
  if (d_method != solver_method::AUTO) {
    return d_method;
  }

  // Auto-select based on matrix properties
  if (rows_a == cols_a) {
    // Square matrix - use LU by default
    return solver_method::LU;
  } else if (rows_a > cols_a) {
    // Overdetermined - use QR for least squares
    return solver_method::QR;
  } else {
    // Underdetermined - use SVD for minimum norm solution
    return solver_method::SVD;
  }
}

template <typename Scalar>
template <typename MatrixType>
bool matrix_solve<Scalar>::solve_lu(const MatrixType &A, const MatrixType &b,
                                    MatrixType &x) const {
  try {
    // Use Eigen's PartialPivLU for numerical stability
    Eigen::PartialPivLU<MatrixType> lu(A);
    x = lu.solve(b);
    return true;
  } catch (const std::exception &e) {
    return false;
  }
}

template <typename Scalar>
template <typename MatrixType>
bool matrix_solve<Scalar>::solve_qr(const MatrixType &A, const MatrixType &b,
                                    MatrixType &x) const {
  try {
    // Use Householder QR decomposition
    Eigen::HouseholderQR<MatrixType> qr(A);
    x = qr.solve(b);
    return true;
  } catch (const std::exception &e) {
    return false;
  }
}

template <typename Scalar>
template <typename MatrixType>
bool matrix_solve<Scalar>::solve_svd(const MatrixType &A, const MatrixType &b,
                                     MatrixType &x) const {
  try {
    // Use Jacobi SVD for better accuracy with small matrices
    Eigen::JacobiSVD<MatrixType> svd(A,
                                     Eigen::ComputeThinU | Eigen::ComputeThinV);

    // Apply tolerance for rank-deficient systems (if needed in future)
    // auto singular_values = svd.singularValues();
    // double threshold = d_tolerance * singular_values[0];

    // Compute pseudoinverse solution
    x = svd.solve(b);
    return true;
  } catch (const std::exception &e) {
    return false;
  }
}

template <typename Scalar>
template <typename MatrixType>
bool matrix_solve<Scalar>::solve_cholesky(const MatrixType &A,
                                          const MatrixType &b,
                                          MatrixType &x) const {
  try {
    // Use LDLT decomposition (more stable than LLT)
    Eigen::LDLT<MatrixType> chol(A);

    // Check if decomposition succeeded
    if (chol.info() != Eigen::Success) {
      return false;
    }

    x = chol.solve(b);
    return true;
  } catch (const std::exception &e) {
    return false;
  }
}

template <typename Scalar>
OperationReturn matrix_solve<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {

  // Validate inputs
  if (input_matrices.size() != 2) {
    return OperationReturn::FAILURE;
  }
  if (output_matrices.size() != 1) {
    return OperationReturn::FAILURE;
  }

  const auto &A = *input_matrices[0];
  const auto &b = *input_matrices[1];
  auto &x = *output_matrices[0];

  // Start performance profiling
  size_t matrix_size = (A.size() + b.size()) * sizeof(Scalar);
  PROFILE_LINALG_OPERATION("matrix_solve", matrix_size);

  // Validate dimensions
  if (A.rows() != d_shape_a[0] || A.cols() != d_shape_a[1]) {
    return OperationReturn::INVALID_SHAPE;
  }

  if (b.rows() != d_shape_b[0] || b.cols() != d_shape_b[1]) {
    return OperationReturn::INVALID_SHAPE;
  }

  // Check output dimensions
  if (x.rows() != A.cols() || x.cols() != b.cols()) {
    return OperationReturn::INVALID_SHAPE;
  }

  try {
    // Create Eigen matrices from the maps
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    MatrixType A_copy = A;
    MatrixType b_copy = b;
    MatrixType x_result(A.cols(), b.cols());

    // Choose and apply solver method
    solver_method method = choose_method(A.rows(), A.cols());
    bool success = false;

    switch (method) {
    case solver_method::LU:
      success = solve_lu(A_copy, b_copy, x_result);
      break;
    case solver_method::QR:
      success = solve_qr(A_copy, b_copy, x_result);
      break;
    case solver_method::SVD:
      success = solve_svd(A_copy, b_copy, x_result);
      break;
    case solver_method::CHOLESKY:
      success = solve_cholesky(A_copy, b_copy, x_result);
      // If Cholesky fails, fall back to LU
      if (!success && A.rows() == A.cols()) {
        success = solve_lu(A_copy, b_copy, x_result);
      }
      break;
    default:
      success = solve_svd(A_copy, b_copy, x_result); // Most general
      break;
    }

    if (!success) {
      return OperationReturn::FAILURE;
    }

    // Copy result to output
    x = x_result;

    return OperationReturn::SUCCESS;
  } catch (const std::exception &e) {
    return OperationReturn::FAILURE;
  }
}

template <typename Scalar>
types::vector_shapes
matrix_solve<Scalar>::compute_output_shapes(const types::shape &shape_a,
                                            const types::shape &shape_b) {

  if (shape_a.size() != 2 || shape_b.size() != 2) {
    throw std::invalid_argument("matrix_solve requires 2D inputs");
  }

  // Output shape: [cols_a, cols_b]
  // For Ax = b, x has dimensions [cols of A, cols of b]
  types::shape output_shape = {shape_a[1], shape_b[1]};

  return {output_shape};
}

// Sync block implementation
template <typename Scalar>
matrix_solve_sync<Scalar>::matrix_solve_sync(
    const types::shape &shape_a, const types::shape &shape_b,
    typename matrix_solve<Scalar>::solver_method method, double tolerance)
    : matrix_solve<Scalar>(shape_a, shape_b, method, tolerance) {}

template <typename Scalar>
typename matrix_solve_sync<Scalar>::sptr matrix_solve_sync<Scalar>::make(
    const types::shape &shape_a, const types::shape &shape_b,
    typename matrix_solve<Scalar>::solver_method method, double tolerance) {
  return std::make_shared<matrix_solve_sync_impl<Scalar>>(shape_a, shape_b,
                                                          method, tolerance);
}

// Implementation classes
template <typename Scalar>
matrix_solve_sync_impl<Scalar>::matrix_solve_sync_impl(
    const types::shape &shape_a, const types::shape &shape_b,
    typename matrix_solve<Scalar>::solver_method method, double tolerance)
    : linalg_base<Scalar>(
          "matrix_solve_sync", {shape_a, shape_b}, {"A", "b"},
          matrix_solve<Scalar>::compute_output_shapes(shape_a, shape_b), {"x"},
          array_broadcast_type::CUSTOM, error_tag_t::NONE, error_pdu_p::NONE),
      matrix_solve<Scalar>(shape_a, shape_b, method, tolerance),
      linalg_base_sync<Scalar>(
          "matrix_solve_sync", {shape_a, shape_b}, {"A", "b"},
          matrix_solve<Scalar>::compute_output_shapes(shape_a, shape_b), {"x"},
          array_broadcast_type::CUSTOM, error_tag_t::NONE, error_pdu_p::NONE,
          gr::block::TPP_ALL_TO_ALL),
      matrix_solve_sync<Scalar>(shape_a, shape_b, method, tolerance) {}

template <typename Scalar>
matrix_solve_sync_impl<Scalar>::~matrix_solve_sync_impl() {}

// Explicit template instantiation for all supported types
template class matrix_solve<float>;
template class matrix_solve<double>;
template class matrix_solve<std::complex<float>>;
template class matrix_solve<std::complex<double>>;

template class matrix_solve_sync<float>;
template class matrix_solve_sync<double>;
template class matrix_solve_sync<std::complex<float>>;
template class matrix_solve_sync<std::complex<double>>;

template class matrix_solve_sync_impl<float>;
template class matrix_solve_sync_impl<double>;
template class matrix_solve_sync_impl<std::complex<float>>;
template class matrix_solve_sync_impl<std::complex<double>>;

} /* namespace linalg */
} /* namespace gr */