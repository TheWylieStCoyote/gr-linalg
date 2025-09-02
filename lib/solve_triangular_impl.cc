/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "solve_triangular_impl.h"
#include <Eigen/Dense>
#include <gnuradio/io_signature.h>
#include <stdexcept>

namespace gr {
namespace linalg {

// Template-based constructor
template <typename Scalar>
solve_triangular<Scalar>::solve_triangular(const types::shape &shape_matrix,
                                           const types::shape &shape_rhs,
                                           triangular_type tri_type,
                                           bool unit_diagonal)
    : d_shape_matrix(shape_matrix), d_shape_rhs(shape_rhs),
      d_tri_type(tri_type), d_unit_diagonal(unit_diagonal) {
  // Validate input shapes
  if (shape_matrix.size() != 2 || shape_matrix[0] != shape_matrix[1]) {
    throw std::invalid_argument(
        "solve_triangular: matrix must be square [n, n]");
  }
  if (shape_rhs.size() != 2 || shape_rhs[0] != shape_matrix[0]) {
    throw std::invalid_argument(
        "solve_triangular: right-hand side must be [n, k] where n matches "
        "matrix size");
  }
}

// Compute output shapes
template <typename Scalar>
types::vector_shapes solve_triangular<Scalar>::compute_output_shapes(
    const types::shape &shape_matrix, const types::shape &shape_rhs) {
  // Output has same shape as right-hand side
  return {shape_rhs};
}

// Core operation implementation
template <typename Scalar>
OperationReturn solve_triangular<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {
  if (input_matrices.size() != 2) {
    return OperationReturn::INVALID_SHAPE;
  }
  if (output_matrices.size() != 1) {
    return OperationReturn::INVALID_SHAPE;
  }

  const auto &matrix_T = *input_matrices[0]; // Triangular matrix
  const auto &matrix_B = *input_matrices[1]; // Right-hand side
  auto &matrix_X = *output_matrices[0];      // Solution

  try {
    // Validate dimensions
    if (matrix_T.rows() != matrix_T.cols()) {
      return OperationReturn::INVALID_SHAPE;
    }
    if (matrix_T.rows() != matrix_B.rows()) {
      return OperationReturn::INVALID_SHAPE;
    }
    if (matrix_X.rows() != matrix_B.rows() ||
        matrix_X.cols() != matrix_B.cols()) {
      return OperationReturn::INVALID_SHAPE;
    }

    const int n = matrix_T.rows();
    const int k = matrix_B.cols();

    // Solve for each column of B
    for (int col = 0; col < k; ++col) {
      auto b_col = matrix_B.col(col);
      auto x_col = matrix_X.col(col);

      if (d_tri_type == triangular_type::LOWER) {
        // Forward substitution for lower triangular
        for (int i = 0; i < n; ++i) {
          Scalar sum = b_col[i];

          // Subtract known terms: sum = b[i] - Σ(T[i,j] * x[j]) for j < i
          for (int j = 0; j < i; ++j) {
            sum -= matrix_T(i, j) * x_col[j];
          }

          // Solve for x[i]
          if (d_unit_diagonal) {
            x_col[i] = sum; // Diagonal element assumed to be 1
          } else {
            if (std::abs(matrix_T(i, i)) < 1e-12) {
              return OperationReturn::FAILURE; // Singular matrix
            }
            x_col[i] = sum / matrix_T(i, i);
          }
        }
      } else {
        // Backward substitution for upper triangular
        for (int i = n - 1; i >= 0; --i) {
          Scalar sum = b_col[i];

          // Subtract known terms: sum = b[i] - Σ(T[i,j] * x[j]) for j > i
          for (int j = i + 1; j < n; ++j) {
            sum -= matrix_T(i, j) * x_col[j];
          }

          // Solve for x[i]
          if (d_unit_diagonal) {
            x_col[i] = sum; // Diagonal element assumed to be 1
          } else {
            if (std::abs(matrix_T(i, i)) < 1e-12) {
              return OperationReturn::FAILURE; // Singular matrix
            }
            x_col[i] = sum / matrix_T(i, i);
          }
        }
      }
    }

    return OperationReturn::SUCCESS;
  } catch (const std::exception &) {
    return OperationReturn::FAILURE;
  }
}

// Sync block implementation
template <typename Scalar>
solve_triangular_sync<Scalar>::solve_triangular_sync(
    const types::shape &shape_matrix, const types::shape &shape_rhs,
    typename solve_triangular<Scalar>::triangular_type tri_type,
    bool unit_diagonal)
    : linalg_base<Scalar>( // MUST explicitly initialize virtual base!
          "solve_triangular_sync", {shape_matrix, shape_rhs}, {"matrix", "rhs"},
          solve_triangular<Scalar>::compute_output_shapes(shape_matrix,
                                                          shape_rhs),
          {"solution"}, array_broadcast_type::CUSTOM, error_tag_t::NONE,
          error_pdu_p::NONE),
      linalg_base_sync<Scalar>(
          "solve_triangular_sync", {shape_matrix, shape_rhs}, {"matrix", "rhs"},
          solve_triangular<Scalar>::compute_output_shapes(shape_matrix,
                                                          shape_rhs),
          {"solution"}, array_broadcast_type::CUSTOM, error_tag_t::NONE,
          error_pdu_p::NONE, gr::block::TPP_ALL_TO_ALL),
      solve_triangular<Scalar>(shape_matrix, shape_rhs, tri_type,
                               unit_diagonal) {}

template <typename Scalar>
typename solve_triangular_sync<Scalar>::sptr
solve_triangular_sync<Scalar>::make(
    const types::shape &shape_matrix, const types::shape &shape_rhs,
    typename solve_triangular<Scalar>::triangular_type tri_type,
    bool unit_diagonal) {
  return std::make_shared<solve_triangular_sync<Scalar>>(
      shape_matrix, shape_rhs, tri_type, unit_diagonal);
}

// PDU block implementation
template <typename Scalar>
solve_triangular_pdu<Scalar>::solve_triangular_pdu(
    const types::shape &shape_matrix, const types::shape &shape_rhs,
    typename solve_triangular<Scalar>::triangular_type tri_type,
    bool unit_diagonal)
    : linalg_base_pdu<Scalar>("solve_triangular_pdu", {shape_matrix, shape_rhs},
                              {"in_matrix", "in_rhs"},
                              solve_triangular<Scalar>::compute_output_shapes(
                                  shape_matrix, shape_rhs),
                              {"out"}, array_broadcast_type::NONE,
                              error_tag_t::NONE, error_pdu_p::NONE),
      solve_triangular<Scalar>(shape_matrix, shape_rhs, tri_type,
                               unit_diagonal) {}

template <typename Scalar>
typename solve_triangular_pdu<Scalar>::sptr solve_triangular_pdu<Scalar>::make(
    const types::shape &shape_matrix, const types::shape &shape_rhs,
    typename solve_triangular<Scalar>::triangular_type tri_type,
    bool unit_diagonal) {
  return std::make_shared<solve_triangular_pdu<Scalar>>(
      shape_matrix, shape_rhs, tri_type, unit_diagonal);
}

// Explicit template instantiations
template class solve_triangular<float>;
template class solve_triangular<double>;
template class solve_triangular<std::complex<float>>;
template class solve_triangular<std::complex<double>>;

template class solve_triangular_sync<float>;
template class solve_triangular_sync<double>;
template class solve_triangular_sync<std::complex<float>>;
template class solve_triangular_sync<std::complex<double>>;

template class solve_triangular_pdu<float>;
template class solve_triangular_pdu<double>;
template class solve_triangular_pdu<std::complex<float>>;
template class solve_triangular_pdu<std::complex<double>>;

} // namespace linalg
} // namespace gr
