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

#include "matrix_sqrt_impl.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <complex>
#include <gnuradio/io_signature.h>
#include <gnuradio/linalg/matrix_sqrt.h>
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
matrix_sqrt<Scalar>::matrix_sqrt()
    : d_algorithm(Algorithm::AUTO), d_max_iterations(100), d_tolerance(1e-10) {}

template <typename Scalar>
matrix_sqrt<Scalar>::matrix_sqrt(Algorithm algorithm, int max_iterations,
                                 double tolerance)
    : d_algorithm(algorithm), d_max_iterations(max_iterations),
      d_tolerance(tolerance) {}

template <typename Scalar>
OperationReturn matrix_sqrt<Scalar>::operation(
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
    if constexpr (std::is_same_v<Scalar, float> ||
                  std::is_same_v<Scalar, double>) {
      if (value < 0) {
        return OperationReturn::FAILURE; // Cannot take sqrt of negative real
                                         // number
      }
      result(0, 0) = std::sqrt(value);
    } else {
      // Complex case: principal square root
      result(0, 0) = std::sqrt(value);
    }
    return OperationReturn::SUCCESS;
  }

  // Select algorithm if AUTO
  Algorithm algorithm = d_algorithm;
  if (algorithm == Algorithm::AUTO) {
    algorithm = select_algorithm(matrix);
  }

  // Dispatch to appropriate algorithm
  switch (algorithm) {
  case Algorithm::EIGENVALUE:
    return compute_eigenvalue_sqrt(matrix, result);
  case Algorithm::SCHUR:
    return compute_schur_sqrt(matrix, result);
  case Algorithm::DENMAN_BEAVERS:
    return compute_denman_beavers_sqrt(matrix, result);
  case Algorithm::NEWTON:
    return compute_newton_sqrt(matrix, result);
  default:
    return compute_eigenvalue_sqrt(matrix, result); // Fallback
  }
}

template <typename Scalar>
typename matrix_sqrt<Scalar>::Algorithm matrix_sqrt<Scalar>::select_algorithm(
    const Eigen::Map<
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &matrix) {

  const size_t n = matrix.rows();

  // For small matrices, use eigenvalue decomposition
  if (n <= 4) {
    return Algorithm::EIGENVALUE;
  }

  // For larger matrices, check if positive definite
  if (is_positive_definite(matrix)) {
    return Algorithm::SCHUR; // More stable for positive definite matrices
  }

  // For general large matrices, use Schur decomposition
  return Algorithm::SCHUR;
}

template <typename Scalar>
OperationReturn matrix_sqrt<Scalar>::compute_eigenvalue_sqrt(
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

    // Check for eigenvalues on negative real axis (problematic for principal
    // square root)
    for (int i = 0; i < eigenvalues.size(); ++i) {
      if (eigenvalues[i].real() < 0 &&
          std::abs(eigenvalues[i].imag()) < 1e-12) {
        return OperationReturn::FAILURE; // Negative real eigenvalue
      }
    }

    // Compute square root of eigenvalues (principal branch)
    Eigen::Matrix<std::complex<real_type_t<Scalar>>, Eigen::Dynamic, 1>
        sqrt_eigenvalues(n);
    for (size_t i = 0; i < n; ++i) {
      sqrt_eigenvalues[i] = std::sqrt(eigenvalues[i]);
    }

    // Reconstruct matrix: sqrt(A) = P * sqrt(D) * P^(-1)
    Eigen::Matrix<std::complex<real_type_t<Scalar>>, Eigen::Dynamic,
                  Eigen::Dynamic>
        sqrt_diagonal = sqrt_eigenvalues.asDiagonal();

    auto temp_result = eigenvectors * sqrt_diagonal * eigenvectors.inverse();

    // Convert back to original scalar type
    if constexpr (std::is_same_v<Scalar, float> ||
                  std::is_same_v<Scalar, double>) {
      // For real matrices, result should be real for positive definite matrices
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
OperationReturn matrix_sqrt<Scalar>::compute_schur_sqrt(
    const Eigen::Map<
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &matrix,
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &result) {

  try {
    // Use RealSchur for real matrices, ComplexSchur for complex matrices
    if constexpr (std::is_same_v<Scalar, float> ||
                  std::is_same_v<Scalar, double>) {
      Eigen::RealSchur<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
          schur;
      schur.compute(matrix);

      if (schur.info() != Eigen::Success) {
        return OperationReturn::FAILURE;
      }

      auto T = schur.matrixT(); // Upper quasi-triangular matrix
      auto U = schur.matrixU(); // Orthogonal matrix

      // Compute square root of Schur form T
      // This is complex for general matrices, so we fall back to eigenvalue
      // method
      return compute_eigenvalue_sqrt(matrix, result);

    } else {
      Eigen::ComplexSchur<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
          schur;
      schur.compute(matrix);

      if (schur.info() != Eigen::Success) {
        return OperationReturn::FAILURE;
      }

      auto T = schur.matrixT(); // Upper triangular matrix
      auto U = schur.matrixU(); // Unitary matrix

      // Compute square root of upper triangular matrix T
      const size_t n = T.rows();
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> sqrt_T = T;

      // Square root of diagonal elements
      for (size_t i = 0; i < n; ++i) {
        sqrt_T(i, i) = std::sqrt(T(i, i));
      }

      // Fill upper triangular part using recurrence relation
      for (size_t j = 1; j < n; ++j) {
        for (size_t i = j - 1; i >= 0 && i < n; --i) {
          if (i == j - 1) {
            sqrt_T(i, j) = T(i, j) / (sqrt_T(i, i) + sqrt_T(j, j));
          } else {
            Scalar sum = Scalar(0);
            for (size_t k = i + 1; k < j; ++k) {
              sum += sqrt_T(i, k) * sqrt_T(k, j);
            }
            sqrt_T(i, j) = (T(i, j) - sum) / (sqrt_T(i, i) + sqrt_T(j, j));
          }
        }
      }

      // Reconstruct: sqrt(A) = U * sqrt(T) * U^H
      result = U * sqrt_T * U.adjoint();

      return OperationReturn::SUCCESS;
    }

  } catch (const std::exception &) {
    return OperationReturn::FAILURE;
  }
}

template <typename Scalar>
OperationReturn matrix_sqrt<Scalar>::compute_denman_beavers_sqrt(
    const Eigen::Map<
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &matrix,
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &result) {

  try {
    const size_t n = matrix.rows();

    // Initialize X_0 = A, Y_0 = I
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> X = matrix;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Y =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(n, n);

    // Denman-Beavers iteration: X_{k+1} = (X_k + Y_k^(-1)) / 2, Y_{k+1} = (Y_k
    // + X_k^(-1)) / 2
    for (int iter = 0; iter < d_max_iterations; ++iter) {
      auto X_old = X;
      auto Y_old = Y;

      // Compute inverses using LU decomposition
      Eigen::FullPivLU<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
          lu_X(X_old);
      Eigen::FullPivLU<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
          lu_Y(Y_old);

      if (!lu_X.isInvertible() || !lu_Y.isInvertible()) {
        return OperationReturn::FAILURE;
      }

      auto X_inv = lu_X.inverse();
      auto Y_inv = lu_Y.inverse();

      // Update iterations
      X = 0.5 * (X_old + Y_inv);
      Y = 0.5 * (Y_old + X_inv);

      // Check convergence
      auto diff = (X - X_old).cwiseAbs().maxCoeff();
      if (diff < d_tolerance) {
        result = X;
        return OperationReturn::SUCCESS;
      }
    }

    // Did not converge
    return OperationReturn::FAILURE;

  } catch (const std::exception &) {
    return OperationReturn::FAILURE;
  }
}

template <typename Scalar>
OperationReturn matrix_sqrt<Scalar>::compute_newton_sqrt(
    const Eigen::Map<
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &matrix,
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &result) {

  try {
    const size_t n = matrix.rows();

    // Initialize X_0 = I (identity matrix is often a good starting point)
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> X =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(n, n);

    // Newton's method: X_{k+1} = (X_k + A * X_k^(-1)) / 2
    for (int iter = 0; iter < d_max_iterations; ++iter) {
      auto X_old = X;

      // Compute X_k^(-1) using LU decomposition
      Eigen::FullPivLU<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
          lu(X_old);

      if (!lu.isInvertible()) {
        return OperationReturn::FAILURE;
      }

      auto X_inv = lu.inverse();

      // Update: X_{k+1} = (X_k + A * X_k^(-1)) / 2
      X = 0.5 * (X_old + matrix * X_inv);

      // Check convergence
      auto diff = (X - X_old).cwiseAbs().maxCoeff();
      if (diff < d_tolerance) {
        result = X;
        return OperationReturn::SUCCESS;
      }
    }

    // Did not converge
    return OperationReturn::FAILURE;

  } catch (const std::exception &) {
    return OperationReturn::FAILURE;
  }
}

template <typename Scalar>
bool matrix_sqrt<Scalar>::is_positive_definite(
    const Eigen::Map<
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &matrix) {

  try {
    // Use LLT decomposition to check positive definiteness
    Eigen::LLT<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> llt;
    llt.compute(matrix);
    return llt.info() == Eigen::Success;
  } catch (...) {
    return false;
  }
}

template <typename Scalar>
void matrix_sqrt<Scalar>::validate_shapes(
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
matrix_sqrt<Scalar>::compute_sizes(const types::vector_shapes &shapes) {
  std::vector<size_t> sizes;
  for (const auto &shape : shapes) {
    sizes.push_back(total_elements(shape));
  }
  return sizes;
}

template <typename Scalar>
types::vector_shapes matrix_sqrt<Scalar>::compute_output_shapes(
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
typename matrix_sqrt_sync<Scalar>::sptr matrix_sqrt_sync<Scalar>::make(
    const types::vector_shapes &input_shapes,
    typename matrix_sqrt<Scalar>::Algorithm algorithm, int max_iterations,
    double tolerance) {
  return std::make_shared<matrix_sqrt_sync<Scalar>>(input_shapes, algorithm,
                                                    max_iterations, tolerance);
}

template <typename Scalar>
matrix_sqrt_sync<Scalar>::matrix_sqrt_sync(
    const types::vector_shapes &input_shapes,
    typename matrix_sqrt<Scalar>::Algorithm algorithm, int max_iterations,
    double tolerance)
    : matrix_sqrt<Scalar>(algorithm, max_iterations, tolerance),
      linalg_base_sync<Scalar>(
          "matrix_sqrt_sync",
          input_shapes,                 // Input shapes
          {"matrix"},                   // Input names
          {input_shapes[0]},            // Output shape same as input
          {"result"},                   // Output names
          array_broadcast_type::MATRIX, // Broadcasting type
          error_tag_t::NONE,            // No error tags
          error_pdu_p::NONE             // No error PDUs
      ) {
  // Validate input shapes
  matrix_sqrt<Scalar>::validate_shapes(input_shapes, {input_shapes[0]},
                                       "matrix_sqrt_sync");
}

//==============================================================================
// PDU Implementation
//==============================================================================

template <typename Scalar>
typename matrix_sqrt_pdu<Scalar>::sptr
matrix_sqrt_pdu<Scalar>::make(const types::vector_shapes &input_shapes,
                              typename matrix_sqrt<Scalar>::Algorithm algorithm,
                              int max_iterations, double tolerance) {
  return std::make_shared<matrix_sqrt_pdu<Scalar>>(input_shapes, algorithm,
                                                   max_iterations, tolerance);
}

template <typename Scalar>
matrix_sqrt_pdu<Scalar>::matrix_sqrt_pdu(
    const types::vector_shapes &input_shapes,
    typename matrix_sqrt<Scalar>::Algorithm algorithm, int max_iterations,
    double tolerance)
    : matrix_sqrt<Scalar>(algorithm, max_iterations, tolerance),
      linalg_base_pdu<Scalar>("matrix_sqrt_pdu", input_shapes, {"matrix"},
                              {input_shapes[0]}, {"result"},
                              array_broadcast_type::MATRIX, error_tag_t::NONE,
                              error_pdu_p::NONE, PDU_UPDATE::ANY_INPUT,
                              MESSAGE_HANDLER_MODE::DEFAULT) {}

//==============================================================================
// Template Instantiations
//==============================================================================

template class matrix_sqrt<float>;
template class matrix_sqrt<double>;
template class matrix_sqrt<std::complex<float>>;
template class matrix_sqrt<std::complex<double>>;

template class matrix_sqrt_sync<float>;
template class matrix_sqrt_sync<double>;
template class matrix_sqrt_sync<std::complex<float>>;
template class matrix_sqrt_sync<std::complex<double>>;

template class matrix_sqrt_pdu<float>;
template class matrix_sqrt_pdu<double>;
template class matrix_sqrt_pdu<std::complex<float>>;
template class matrix_sqrt_pdu<std::complex<double>>;

} // namespace linalg
} // namespace gr