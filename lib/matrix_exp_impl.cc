/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <complex>
#include <gnuradio/io_signature.h>
#include <gnuradio/linalg/matrix_exp.h>
#include <gnuradio/linalg/types.h>
#include <gnuradio/linalg/utils.h>
#include <unsupported/Eigen/MatrixFunctions>

using namespace gr::linalg::types::tensor_utils;

namespace gr {
namespace linalg {

// Type trait to get the underlying real type
template <typename T>
struct real_type {
  using type = T;
};

template <typename T>
struct real_type<std::complex<T>> {
  using type = T;
};

template <typename T>
using real_type_t = typename real_type<T>::type;

//==============================================================================
// Base Implementation
//==============================================================================

template <typename Scalar>
matrix_exp<Scalar>::matrix_exp() : d_use_scaling_squaring(true) {}

template <typename Scalar>
matrix_exp<Scalar>::matrix_exp(bool use_scaling_squaring)
    : d_use_scaling_squaring(use_scaling_squaring) {}

template <typename Scalar>
OperationReturn matrix_exp<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {
  // Validate inputs
  if (input_matrices.size() != 1) {
    return OperationReturn::INVALID_SHAPE;
  }
  if (output_matrices.size() != 1) {
    return OperationReturn::INVALID_SHAPE;
  }

  const auto &input_matrix = *input_matrices[0];
  auto &output_matrix = *output_matrices[0];

  // Check that input is square
  if (input_matrix.rows() != input_matrix.cols()) {
    return OperationReturn::INVALID_SHAPE;
  }

  // Check output dimensions match input
  if (output_matrix.rows() != input_matrix.rows() ||
      output_matrix.cols() != input_matrix.cols()) {
    return OperationReturn::INVALID_SHAPE;
  }

  try {
    // Compute matrix exponential using Eigen's implementation
    if (d_use_scaling_squaring) {
      // Use Eigen's matrix exponential (scaling and squaring method)
      output_matrix = input_matrix.exp();
    } else {
      // Alternative: Use eigenvalue decomposition method
      compute_matrix_exp_eigen_decomp(input_matrix, output_matrix);
    }

    return OperationReturn::SUCCESS;
  } catch (const std::exception &e) {
    return OperationReturn::FAILURE;
  }
}

template <typename Scalar>
void matrix_exp<Scalar>::compute_matrix_exp_eigen_decomp(
    const Eigen::Ref<
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &input,
    Eigen::Ref<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> output) {

  using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using RealScalar = real_type_t<Scalar>;

  const int n = input.rows();

  // For special cases, use optimized algorithms
  if (n == 1) {
    // Scalar case
    output(0, 0) = std::exp(input(0, 0));
    return;
  }

  // Check if matrix is diagonal
  bool is_diagonal = true;
  for (int i = 0; i < n && is_diagonal; ++i) {
    for (int j = 0; j < n && is_diagonal; ++j) {
      if (i != j && std::abs(input(i, j)) > 1e-12) {
        is_diagonal = false;
      }
    }
  }

  if (is_diagonal) {
    // For diagonal matrices, exponential is element-wise
    output.setZero();
    for (int i = 0; i < n; ++i) {
      output(i, i) = std::exp(input(i, i));
    }
    return;
  }

  // General case: Use eigenvalue decomposition
  using ComplexMatrixType =
      Eigen::Matrix<std::complex<RealScalar>, Eigen::Dynamic, Eigen::Dynamic>;
  ComplexMatrixType complex_input =
      input.template cast<std::complex<RealScalar>>();

  Eigen::ComplexEigenSolver<ComplexMatrixType> solver(complex_input);
  if (solver.info() != Eigen::Success) {
    // Fallback to scaling and squaring if eigenvalue decomposition fails
    output = input.exp();
    return;
  }

  auto eigenvalues = solver.eigenvalues();
  auto eigenvectors = solver.eigenvectors();

  // Compute exp(eigenvalues)
  for (int i = 0; i < eigenvalues.size(); ++i) {
    eigenvalues(i) = std::exp(eigenvalues(i));
  }

  // Reconstruct matrix: exp(A) = P * exp(D) * P^(-1)
  auto exp_diagonal = eigenvalues.asDiagonal();
  auto result_complex = eigenvectors * exp_diagonal * eigenvectors.inverse();

  // Convert back to original scalar type
  if constexpr (std::is_same_v<Scalar, std::complex<float>> ||
                std::is_same_v<Scalar, std::complex<double>>) {
    output = result_complex.template cast<Scalar>();
  } else {
    // For real types, take the real part (should be real for real input
    // matrices)
    output = result_complex.real().template cast<Scalar>();
  }
}

template <typename Scalar>
types::vector_shapes matrix_exp<Scalar>::compute_output_shapes(
    const types::vector_shapes &input_shapes) {
  if (input_shapes.size() != 1) {
    return {};
  }

  const auto &input_shape = input_shapes[0];
  if (input_shape.size() != 2) {
    return {};
  }

  // Must be square matrix
  if (input_shape[0] != input_shape[1]) {
    return {};
  }

  // Output shape is same as input shape
  return {input_shape};
}

//==============================================================================
// Sync Implementation
//==============================================================================

template <typename Scalar>
typename matrix_exp_sync<Scalar>::sptr
matrix_exp_sync<Scalar>::make(const types::vector_shapes &input_shapes,
                              bool use_scaling_squaring) {
  return std::make_shared<matrix_exp_sync<Scalar>>(input_shapes,
                                                   use_scaling_squaring);
}

template <typename Scalar>
matrix_exp_sync<Scalar>::matrix_exp_sync(
    const types::vector_shapes &input_shapes, bool use_scaling_squaring)
    : matrix_exp<Scalar>(use_scaling_squaring),
      linalg_base_sync<Scalar>(
          "matrix_exp_sync",
          input_shapes,                 // Input shapes
          {"matrix"},                   // Input names
          {input_shapes[0]},            // Output shape same as input
          {"result"},                   // Output names
          array_broadcast_type::MATRIX, // Broadcasting type
          error_tag_t::NONE,            // No error tags
          error_pdu_p::NONE             // No error PDUs
      ) {
  // Validate input shapes
  if (input_shapes.size() != 1) {
    throw std::invalid_argument(
        "matrix_exp_sync: requires exactly 1 input matrix");
  }
  const auto &input_shape = input_shapes[0];
  if (input_shape.size() != 2 || input_shape[0] != input_shape[1]) {
    throw std::invalid_argument(
        "matrix_exp_sync: input must be a square matrix");
  }
}

//==============================================================================
// PDU Implementation
//==============================================================================

template <typename Scalar>
typename matrix_exp_pdu<Scalar>::sptr
matrix_exp_pdu<Scalar>::make(const types::vector_shapes &input_shapes,
                             bool use_scaling_squaring) {
  return std::make_shared<matrix_exp_pdu<Scalar>>(input_shapes,
                                                  use_scaling_squaring);
}

template <typename Scalar>
matrix_exp_pdu<Scalar>::matrix_exp_pdu(const types::vector_shapes &input_shapes,
                                       bool use_scaling_squaring)
    : matrix_exp<Scalar>(use_scaling_squaring),
      linalg_base_pdu<Scalar>("matrix_exp_pdu", input_shapes, {"matrix"},
                              {input_shapes[0]}, {"result"},
                              array_broadcast_type::MATRIX, error_tag_t::NONE,
                              error_pdu_p::NONE, PDU_UPDATE::ANY_INPUT,
                              MESSAGE_HANDLER_MODE::DEFAULT) {}

//==============================================================================
// Explicit Template Instantiations
//==============================================================================

template class matrix_exp<float>;
template class matrix_exp<double>;
template class matrix_exp<std::complex<float>>;
template class matrix_exp<std::complex<double>>;

template class matrix_exp_sync<float>;
template class matrix_exp_sync<double>;
template class matrix_exp_sync<std::complex<float>>;
template class matrix_exp_sync<std::complex<double>>;

template class matrix_exp_pdu<float>;
template class matrix_exp_pdu<double>;
template class matrix_exp_pdu<std::complex<float>>;
template class matrix_exp_pdu<std::complex<double>>;

} // namespace linalg
} // namespace gr