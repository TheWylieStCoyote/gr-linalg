/* -*- c++ -*- */
/*
 * Copyright 2025 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "matrix_log_impl.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <complex>
#include <gnuradio/io_signature.h>
#include <gnuradio/linalg/matrix_log.h>
#include <gnuradio/linalg/types.h>
#include <gnuradio/linalg/utils.h>

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
matrix_log<Scalar>::matrix_log() : d_use_stable_algorithm(true) {}

template <typename Scalar>
matrix_log<Scalar>::matrix_log(bool use_stable_algorithm)
    : d_use_stable_algorithm(use_stable_algorithm) {}

template <typename Scalar>
OperationReturn matrix_log<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {
  // Validate inputs
  if (input_matrices.size() != 1) {
    return OperationReturn::FAILURE;
  }
  if (output_matrices.size() != 1) {
    return OperationReturn::FAILURE;
  }

  const auto &matrix = *input_matrices[0];
  auto &result = *output_matrices[0];

  // Check matrix dimensions (must be square)
  if (matrix.rows() != matrix.cols()) {
    return OperationReturn::INVALID_SHAPE;
  }

  // Check for empty matrix
  if (matrix.rows() == 0) {
    return OperationReturn::INVALID_SHAPE;
  }

  // Special case: 1x1 matrix
  if (matrix.rows() == 1) {
    Scalar value = matrix(0, 0);
    // Check for non-positive values in real case
    if constexpr (std::is_same_v<Scalar, float> ||
                  std::is_same_v<Scalar, double>) {
      if (value <= 0) {
        return OperationReturn::FAILURE; // Cannot take log of non-positive real
                                         // number
      }
      result(0, 0) = std::log(value);
    } else {
      // Complex case: log(0) is undefined
      if (std::abs(value) == 0.0) {
        return OperationReturn::FAILURE;
      }
      result(0, 0) = std::log(value);
    }
    return OperationReturn::SUCCESS;
  }

  // Use specialized algorithm for Hermitian matrices if requested
  if (d_use_stable_algorithm && is_hermitian(matrix)) {
    return compute_hermitian_log(matrix, result);
  }

  // General case: use eigenvalue decomposition
  return compute_general_log(matrix, result);
}

template <typename Scalar>
OperationReturn matrix_log<Scalar>::compute_general_log(
    const Eigen::Map<
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &matrix,
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &result) {

  try {
    const size_t n = matrix.rows();

    // Use ComplexEigenSolver for general matrices
    Eigen::ComplexEigenSolver<
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
        solver;
    solver.compute(matrix);

    if (solver.info() != Eigen::Success) {
      return OperationReturn::FAILURE;
    }

    // Get eigenvalues and eigenvectors
    auto eigenvalues = solver.eigenvalues();
    auto eigenvectors = solver.eigenvectors();

    // Check for zero eigenvalues (matrix must be non-singular)
    for (int i = 0; i < eigenvalues.size(); ++i) {
      if (std::abs(eigenvalues[i]) < 1e-12) {
        return OperationReturn::FAILURE; // Singular matrix
      }
    }

    // Compute log of eigenvalues
    Eigen::Matrix<std::complex<real_type_t<Scalar>>, Eigen::Dynamic, 1>
        log_eigenvalues(n);
    for (size_t i = 0; i < n; ++i) {
      log_eigenvalues[i] = std::log(eigenvalues[i]);
    }

    // Reconstruct matrix: log(A) = P * log(D) * P^(-1)
    Eigen::Matrix<std::complex<real_type_t<Scalar>>, Eigen::Dynamic,
                  Eigen::Dynamic>
        log_diagonal = log_eigenvalues.asDiagonal();

    auto temp_result = eigenvectors * log_diagonal * eigenvectors.inverse();

    // Convert back to original scalar type
    if constexpr (std::is_same_v<Scalar, float> ||
                  std::is_same_v<Scalar, double>) {
      // For real matrices, result should be real if all eigenvalues have
      // positive real parts
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
          if (std::abs(temp_result(i, j).imag()) > 1e-10) {
            return OperationReturn::FAILURE; // Result is not real
          }
          result(i, j) = temp_result(i, j).real();
        }
      }
    } else {
      // Complex case: direct assignment
      result = temp_result.template cast<Scalar>();
    }

    return OperationReturn::SUCCESS;

  } catch (const std::exception &) {
    return OperationReturn::FAILURE;
  }
}

template <typename Scalar>
OperationReturn matrix_log<Scalar>::compute_hermitian_log(
    const Eigen::Map<
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &matrix,
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &result) {

  try {
    // Use SelfAdjointEigenSolver for Hermitian matrices (more stable)
    Eigen::SelfAdjointEigenSolver<
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
        solver;
    solver.compute(matrix);

    if (solver.info() != Eigen::Success) {
      return OperationReturn::FAILURE;
    }

    // Get eigenvalues and eigenvectors
    auto eigenvalues = solver.eigenvalues();
    auto eigenvectors = solver.eigenvectors();

    // Check for non-positive eigenvalues
    for (int i = 0; i < eigenvalues.size(); ++i) {
      if (eigenvalues[i] <= 0) {
        return OperationReturn::FAILURE; // Matrix is not positive definite
      }
    }

    // Compute log of eigenvalues (all real and positive)
    Eigen::Matrix<real_type_t<Scalar>, Eigen::Dynamic, 1> log_eigenvalues =
        eigenvalues.array().log();

    // Reconstruct matrix: log(A) = Q * log(Λ) * Q^T
    result =
        eigenvectors * log_eigenvalues.asDiagonal() * eigenvectors.transpose();

    return OperationReturn::SUCCESS;

  } catch (const std::exception &) {
    return OperationReturn::FAILURE;
  }
}

template <typename Scalar>
bool matrix_log<Scalar>::is_hermitian(
    const Eigen::Map<
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &matrix) {

  const double tolerance = 1e-10;

  if constexpr (std::is_same_v<Scalar, float> ||
                std::is_same_v<Scalar, double>) {
    // Real case: check if matrix is symmetric
    return matrix.isApprox(matrix.transpose(), tolerance);
  } else {
    // Complex case: check if matrix is Hermitian (A = A^H)
    return matrix.isApprox(matrix.adjoint(), tolerance);
  }
}

template <typename Scalar>
void matrix_log<Scalar>::validate_shapes(
    const types::vector_shapes &input_shapes,
    const types::vector_shapes &output_shapes, const std::string &name) {
  if (input_shapes.size() != 1) {
    throw std::invalid_argument(name + ": requires exactly 1 input matrix");
  }
  if (output_shapes.size() != 1) {
    throw std::invalid_argument(name + ": requires exactly 1 output matrix");
  }

  const auto &input_shape = input_shapes[0];
  const auto &output_shape = output_shapes[0];

  // Matrix must be square
  if (input_shape.size() != 2 || input_shape[0] != input_shape[1]) {
    throw std::invalid_argument(name + ": input must be a square matrix");
  }

  // Output shape should match input shape
  if (output_shape[0] != input_shape[0] || output_shape[1] != input_shape[1]) {
    throw std::invalid_argument(name + ": output shape mismatch");
  }
}

template <typename Scalar>
std::vector<size_t>
matrix_log<Scalar>::compute_sizes(const types::vector_shapes &shapes) {
  std::vector<size_t> sizes;
  for (const auto &shape : shapes) {
    sizes.push_back(total_elements(shape));
  }
  return sizes;
}

template <typename Scalar>
types::vector_shapes matrix_log<Scalar>::compute_output_shapes(
    const types::vector_shapes &input_shapes) {
  if (input_shapes.size() != 1) {
    return {};
  }

  // Output shape is the same as input shape (square matrix)
  return {input_shapes[0]};
}

//==============================================================================
// Sync Implementation
//==============================================================================

template <typename Scalar>
typename matrix_log_sync<Scalar>::sptr
matrix_log_sync<Scalar>::make(const types::vector_shapes &input_shapes,
                              bool use_stable_algorithm) {
  return std::make_shared<matrix_log_sync<Scalar>>(input_shapes,
                                                   use_stable_algorithm);
}

template <typename Scalar>
matrix_log_sync<Scalar>::matrix_log_sync(
    const types::vector_shapes &input_shapes, bool use_stable_algorithm)
    : matrix_log<Scalar>(use_stable_algorithm),
      linalg_base_sync<Scalar>(
          "matrix_log_sync",
          input_shapes,                 // Input shapes
          {"matrix"},                   // Input names
          {input_shapes[0]},            // Output shape same as input
          {"result"},                   // Output names
          array_broadcast_type::MATRIX, // Broadcasting type
          error_tag_t::NONE,            // No error tags
          error_pdu_p::NONE             // No error PDUs
      ) {
  // Validate input shapes
  matrix_log<Scalar>::validate_shapes(input_shapes, {input_shapes[0]},
                                      "matrix_log_sync");
}

//==============================================================================
// PDU Implementation
//==============================================================================

template <typename Scalar>
typename matrix_log_pdu<Scalar>::sptr
matrix_log_pdu<Scalar>::make(const types::vector_shapes &input_shapes,
                             bool use_stable_algorithm) {
  return std::make_shared<matrix_log_pdu<Scalar>>(input_shapes,
                                                  use_stable_algorithm);
}

template <typename Scalar>
matrix_log_pdu<Scalar>::matrix_log_pdu(const types::vector_shapes &input_shapes,
                                       bool use_stable_algorithm)
    : matrix_log<Scalar>(use_stable_algorithm),
      linalg_base_pdu<Scalar>("matrix_log_pdu", input_shapes, {"matrix"},
                              {input_shapes[0]}, {"result"},
                              array_broadcast_type::MATRIX, error_tag_t::NONE,
                              error_pdu_p::NONE, PDU_UPDATE::ANY_INPUT,
                              MESSAGE_HANDLER_MODE::DEFAULT) {}

//==============================================================================
// Template Instantiations
//==============================================================================

template class matrix_log<float>;
template class matrix_log<double>;
template class matrix_log<std::complex<float>>;
template class matrix_log<std::complex<double>>;

template class matrix_log_sync<float>;
template class matrix_log_sync<double>;
template class matrix_log_sync<std::complex<float>>;
template class matrix_log_sync<std::complex<double>>;

template class matrix_log_pdu<float>;
template class matrix_log_pdu<double>;
template class matrix_log_pdu<std::complex<float>>;
template class matrix_log_pdu<std::complex<double>>;

} // namespace linalg
} // namespace gr