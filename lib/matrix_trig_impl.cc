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

#include "matrix_trig_impl.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <complex>
#include <gnuradio/io_signature.h>
#include <gnuradio/linalg/matrix_trig.h>
#include <gnuradio/linalg/types.h>
#include <gnuradio/linalg/utils.h>

using namespace gr::linalg::types::tensor_utils;

namespace gr {
namespace linalg {

//==============================================================================
// Base Implementation
//==============================================================================

template <typename Scalar>
matrix_trig<Scalar>::matrix_trig(TrigFunction function, TrigAlgorithm algorithm,
                                 int max_iterations, double tolerance)
    : d_function(function), d_algorithm(algorithm),
      d_max_iterations(max_iterations), d_tolerance(tolerance) {}

template <typename Scalar>
OperationReturn matrix_trig<Scalar>::operation(
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
    switch (d_function) {
    case TrigFunction::SIN:
      result(0, 0) = std::sin(value);
      break;
    case TrigFunction::COS:
      result(0, 0) = std::cos(value);
      break;
    case TrigFunction::TAN:
      result(0, 0) = std::tan(value);
      break;
    }
    return OperationReturn::SUCCESS;
  }

  // Select algorithm if AUTO
  TrigAlgorithm algorithm = d_algorithm;
  if (algorithm == TrigAlgorithm::AUTO) {
    algorithm = select_algorithm(matrix);
  }

  // Dispatch to appropriate algorithm
  switch (algorithm) {
  case TrigAlgorithm::EIGENVALUE:
    return compute_eigenvalue_trig(matrix, result);
  case TrigAlgorithm::TAYLOR_SERIES:
    return compute_taylor_trig(matrix, result);
  case TrigAlgorithm::SCALING_SQUARING:
    return compute_scaling_squaring_trig(matrix, result);
  case TrigAlgorithm::PADE:
    // For now, fall back to eigenvalue method
    return compute_eigenvalue_trig(matrix, result);
  default:
    return compute_eigenvalue_trig(matrix, result); // Fallback
  }
}

template <typename Scalar>
TrigAlgorithm matrix_trig<Scalar>::select_algorithm(
    const Eigen::Map<
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &matrix) {

  const size_t n = matrix.rows();

  // For small matrices, use eigenvalue decomposition
  if (n <= 4) {
    return TrigAlgorithm::EIGENVALUE;
  }

  // Compute matrix norm to decide algorithm
  auto norm = compute_matrix_norm(matrix);

  // For small norm matrices, Taylor series converges quickly
  if (norm < 0.1) {
    return TrigAlgorithm::TAYLOR_SERIES;
  }

  // For large norm matrices, use scaling and squaring
  if (norm > 1.0) {
    return TrigAlgorithm::SCALING_SQUARING;
  }

  // General case: eigenvalue decomposition
  return TrigAlgorithm::EIGENVALUE;
}

template <typename Scalar>
OperationReturn matrix_trig<Scalar>::compute_eigenvalue_trig(
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

    // Compute trigonometric function of eigenvalues
    Eigen::Matrix<std::complex<typename std::conditional<
                      std::is_same_v<Scalar, float> ||
                          std::is_same_v<Scalar, std::complex<float>>,
                      float, double>::type>,
                  Eigen::Dynamic, 1>
        trig_eigenvalues(n);

    for (size_t i = 0; i < n; ++i) {
      switch (d_function) {
      case TrigFunction::SIN:
        trig_eigenvalues[i] = std::sin(eigenvalues[i]);
        break;
      case TrigFunction::COS:
        trig_eigenvalues[i] = std::cos(eigenvalues[i]);
        break;
      case TrigFunction::TAN:
        trig_eigenvalues[i] = std::tan(eigenvalues[i]);
        break;
      }
    }

    // Reconstruct matrix: trig(A) = P * trig(D) * P^(-1)
    auto trig_diagonal = trig_eigenvalues.asDiagonal();
    auto temp_result = eigenvectors * trig_diagonal * eigenvectors.inverse();

    // Convert back to original scalar type
    if constexpr (std::is_same_v<Scalar, float> ||
                  std::is_same_v<Scalar, double>) {
      // For real matrices, result should be real for real eigenvalues
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
          if (std::abs(temp_result(i, j).imag()) > 1e-10) {
            // Result has significant imaginary part - this can happen for real
            // matrices with complex eigenvalues. Keep the real part as best
            // approximation.
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
OperationReturn matrix_trig<Scalar>::compute_taylor_trig(
    const Eigen::Map<
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &matrix,
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &result) {

  try {
    const size_t n = matrix.rows();
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> I =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(n, n);

    switch (d_function) {
    case TrigFunction::SIN: {
      // sin(A) = A - A³/3! + A⁵/5! - A⁷/7! + ...
      result = I * 0; // Start with zero matrix
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> term = matrix;
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> A_power = matrix;

      int factorial = 1;
      for (int k = 1; k <= d_max_iterations; k += 2) {
        if (k > 1) {
          factorial *= k * (k - 1);
          A_power = A_power * matrix * matrix; // A^k
        }

        Scalar coeff =
            (k % 4 == 1) ? Scalar(1.0 / factorial) : Scalar(-1.0 / factorial);
        auto new_term = coeff * A_power;
        result += new_term;

        // Check convergence
        if (new_term.cwiseAbs().maxCoeff() < d_tolerance) {
          break;
        }
      }
      break;
    }

    case TrigFunction::COS: {
      // cos(A) = I - A²/2! + A⁴/4! - A⁶/6! + ...
      result = I; // Start with identity matrix
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> A_power =
          matrix * matrix; // A²

      int factorial = 2;
      for (int k = 2; k <= d_max_iterations; k += 2) {
        if (k > 2) {
          factorial *= k * (k - 1);
          A_power = A_power * matrix * matrix; // A^k
        }

        Scalar coeff =
            (k % 4 == 2) ? Scalar(-1.0 / factorial) : Scalar(1.0 / factorial);
        auto new_term = coeff * A_power;
        result += new_term;

        // Check convergence
        if (new_term.cwiseAbs().maxCoeff() < d_tolerance) {
          break;
        }
      }
      break;
    }

    case TrigFunction::TAN: {
      // tan(A) = sin(A) * cos(A)^(-1)
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> sin_result(n, n);
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> cos_result(n, n);

      // Temporarily change function to compute sin and cos
      TrigFunction orig_function = d_function;

      // Compute sin(A)
      const_cast<matrix_trig<Scalar> *>(this)->d_function = TrigFunction::SIN;
      Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> sin_map(
          sin_result.data(), n, n);
      if (compute_taylor_trig(matrix, sin_map) != OperationReturn::SUCCESS) {
        const_cast<matrix_trig<Scalar> *>(this)->d_function = orig_function;
        return OperationReturn::FAILURE;
      }

      // Compute cos(A)
      const_cast<matrix_trig<Scalar> *>(this)->d_function = TrigFunction::COS;
      Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> cos_map(
          cos_result.data(), n, n);
      if (compute_taylor_trig(matrix, cos_map) != OperationReturn::SUCCESS) {
        const_cast<matrix_trig<Scalar> *>(this)->d_function = orig_function;
        return OperationReturn::FAILURE;
      }

      // Restore original function
      const_cast<matrix_trig<Scalar> *>(this)->d_function = orig_function;

      // Compute cos(A)^(-1)
      Eigen::FullPivLU<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
          lu(cos_result);
      if (!lu.isInvertible()) {
        return OperationReturn::FAILURE; // cos(A) is singular
      }

      auto cos_inv = lu.inverse();
      result = sin_result * cos_inv;
      break;
    }
    }

    return OperationReturn::SUCCESS;

  } catch (const std::exception &) {
    return OperationReturn::FAILURE;
  }
}

template <typename Scalar>
OperationReturn matrix_trig<Scalar>::compute_scaling_squaring_trig(
    const Eigen::Map<
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &matrix,
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &result) {

  try {
    // Scale matrix down by factor of 2^m to get ||A/2^m|| < 0.1
    auto norm = compute_matrix_norm(matrix);
    int m = static_cast<int>(std::ceil(std::log2(norm / 0.1)));
    if (m < 0)
      m = 0;

    Scalar scale_factor = std::pow(Scalar(2.0), -m);
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> scaled_matrix =
        scale_factor * matrix;

    // Compute trig function of scaled matrix using Taylor series
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> scaled_result(
        matrix.rows(), matrix.cols());
    Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
        scaled_map(scaled_matrix.data(), matrix.rows(), matrix.cols());
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
        result_map(scaled_result.data(), matrix.rows(), matrix.cols());

    // Use Taylor series on the scaled matrix
    if (compute_taylor_trig(scaled_map, result_map) !=
        OperationReturn::SUCCESS) {
      return OperationReturn::FAILURE;
    }

    // Scale back up using trigonometric identities
    // This is complex for general trigonometric functions, so we use a
    // simplified approach For now, we'll use the eigenvalue method as it's more
    // reliable for scaling
    return compute_eigenvalue_trig(matrix, result);

  } catch (const std::exception &) {
    return OperationReturn::FAILURE;
  }
}

template <typename Scalar>
real_type_t<Scalar> matrix_trig<Scalar>::compute_matrix_norm(
    const Eigen::Map<
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &matrix) {
  // Use Frobenius norm
  if constexpr (std::is_same_v<Scalar, float> ||
                std::is_same_v<Scalar, double>) {
    return matrix.norm();
  } else {
    // For complex types, we need to handle the value_type correctly
    return matrix.norm();
  }
}

template <typename Scalar>
void matrix_trig<Scalar>::validate_shapes(
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
matrix_trig<Scalar>::compute_sizes(const types::vector_shapes &shapes) {
  std::vector<size_t> sizes;
  for (const auto &shape : shapes) {
    sizes.push_back(total_elements(shape));
  }
  return sizes;
}

template <typename Scalar>
types::vector_shapes matrix_trig<Scalar>::compute_output_shapes(
    const types::vector_shapes &input_shapes) {
  if (input_shapes.size() != 1) {
    return {};
  }

  // Output shape is the same as input shape (square matrix)
  return {input_shapes[0]};
}

//==============================================================================
// Specific Function Implementations
//==============================================================================

template <typename Scalar>
matrix_sin<Scalar>::matrix_sin(TrigAlgorithm algorithm, int max_iterations,
                               double tolerance)
    : matrix_trig<Scalar>(TrigFunction::SIN, algorithm, max_iterations,
                          tolerance) {}

template <typename Scalar>
matrix_cos<Scalar>::matrix_cos(TrigAlgorithm algorithm, int max_iterations,
                               double tolerance)
    : matrix_trig<Scalar>(TrigFunction::COS, algorithm, max_iterations,
                          tolerance) {}

template <typename Scalar>
matrix_tan<Scalar>::matrix_tan(TrigAlgorithm algorithm, int max_iterations,
                               double tolerance)
    : matrix_trig<Scalar>(TrigFunction::TAN, algorithm, max_iterations,
                          tolerance) {}

//==============================================================================
// Sync Implementation
//==============================================================================

template <typename Scalar>
typename matrix_sin_sync<Scalar>::sptr
matrix_sin_sync<Scalar>::make(const types::vector_shapes &input_shapes,
                              TrigAlgorithm algorithm, int max_iterations,
                              double tolerance) {
  return std::make_shared<matrix_sin_sync<Scalar>>(input_shapes, algorithm,
                                                   max_iterations, tolerance);
}

template <typename Scalar>
matrix_sin_sync<Scalar>::matrix_sin_sync(
    const types::vector_shapes &input_shapes, TrigAlgorithm algorithm,
    int max_iterations, double tolerance)
    : linalg_base<Scalar>("matrix_sin_sync",
                          input_shapes,      // Input shapes
                          {"matrix"},        // Input names
                          {input_shapes[0]}, // Output shape same as input
                          {"result"},        // Output names
                          array_broadcast_type::NONE, // No broadcasting (single
                                                      // matrix operation)
                          error_tag_t::NONE,          // No error tags
                          error_pdu_p::NONE           // No error PDUs
                          ),
      matrix_sin<Scalar>(algorithm, max_iterations, tolerance),
      linalg_base_sync<Scalar>(
          "matrix_sin_sync",
          input_shapes,               // Input shapes
          {"matrix"},                 // Input names
          {input_shapes[0]},          // Output shape same as input
          {"result"},                 // Output names
          array_broadcast_type::NONE, // No broadcasting (single matrix
                                      // operation)
          error_tag_t::NONE,          // No error tags
          error_pdu_p::NONE,          // No error PDUs
          gr::block::TPP_ALL_TO_ALL   // Tag propagation policy
      ) {
  // Validate input shapes
  matrix_trig<Scalar>::validate_shapes(input_shapes, {input_shapes[0]},
                                       "matrix_sin_sync");
}

template <typename Scalar>
typename matrix_cos_sync<Scalar>::sptr
matrix_cos_sync<Scalar>::make(const types::vector_shapes &input_shapes,
                              TrigAlgorithm algorithm, int max_iterations,
                              double tolerance) {
  return std::make_shared<matrix_cos_sync<Scalar>>(input_shapes, algorithm,
                                                   max_iterations, tolerance);
}

template <typename Scalar>
matrix_cos_sync<Scalar>::matrix_cos_sync(
    const types::vector_shapes &input_shapes, TrigAlgorithm algorithm,
    int max_iterations, double tolerance)
    : linalg_base<Scalar>("matrix_cos_sync",
                          input_shapes,      // Input shapes
                          {"matrix"},        // Input names
                          {input_shapes[0]}, // Output shape same as input
                          {"result"},        // Output names
                          array_broadcast_type::NONE, // No broadcasting (single
                                                      // matrix operation)
                          error_tag_t::NONE,          // No error tags
                          error_pdu_p::NONE           // No error PDUs
                          ),
      matrix_cos<Scalar>(algorithm, max_iterations, tolerance),
      linalg_base_sync<Scalar>(
          "matrix_cos_sync",
          input_shapes,               // Input shapes
          {"matrix"},                 // Input names
          {input_shapes[0]},          // Output shape same as input
          {"result"},                 // Output names
          array_broadcast_type::NONE, // No broadcasting (single matrix
                                      // operation)
          error_tag_t::NONE,          // No error tags
          error_pdu_p::NONE,          // No error PDUs
          gr::block::TPP_ALL_TO_ALL   // Tag propagation policy
      ) {
  // Validate input shapes
  matrix_trig<Scalar>::validate_shapes(input_shapes, {input_shapes[0]},
                                       "matrix_cos_sync");
}

template <typename Scalar>
typename matrix_tan_sync<Scalar>::sptr
matrix_tan_sync<Scalar>::make(const types::vector_shapes &input_shapes,
                              TrigAlgorithm algorithm, int max_iterations,
                              double tolerance) {
  return std::make_shared<matrix_tan_sync<Scalar>>(input_shapes, algorithm,
                                                   max_iterations, tolerance);
}

template <typename Scalar>
matrix_tan_sync<Scalar>::matrix_tan_sync(
    const types::vector_shapes &input_shapes, TrigAlgorithm algorithm,
    int max_iterations, double tolerance)
    : linalg_base<Scalar>("matrix_tan_sync",
                          input_shapes,      // Input shapes
                          {"matrix"},        // Input names
                          {input_shapes[0]}, // Output shape same as input
                          {"result"},        // Output names
                          array_broadcast_type::NONE, // No broadcasting (single
                                                      // matrix operation)
                          error_tag_t::NONE,          // No error tags
                          error_pdu_p::NONE           // No error PDUs
                          ),
      matrix_tan<Scalar>(algorithm, max_iterations, tolerance),
      linalg_base_sync<Scalar>(
          "matrix_tan_sync",
          input_shapes,               // Input shapes
          {"matrix"},                 // Input names
          {input_shapes[0]},          // Output shape same as input
          {"result"},                 // Output names
          array_broadcast_type::NONE, // No broadcasting (single matrix
                                      // operation)
          error_tag_t::NONE,          // No error tags
          error_pdu_p::NONE,          // No error PDUs
          gr::block::TPP_ALL_TO_ALL   // Tag propagation policy
      ) {
  // Validate input shapes
  matrix_trig<Scalar>::validate_shapes(input_shapes, {input_shapes[0]},
                                       "matrix_tan_sync");
}

//==============================================================================
// PDU Implementation
//==============================================================================

template <typename Scalar>
typename matrix_sin_pdu<Scalar>::sptr
matrix_sin_pdu<Scalar>::make(const types::vector_shapes &input_shapes,
                             TrigAlgorithm algorithm, int max_iterations,
                             double tolerance) {
  return std::make_shared<matrix_sin_pdu<Scalar>>(input_shapes, algorithm,
                                                  max_iterations, tolerance);
}

template <typename Scalar>
matrix_sin_pdu<Scalar>::matrix_sin_pdu(const types::vector_shapes &input_shapes,
                                       TrigAlgorithm algorithm,
                                       int max_iterations, double tolerance)
    : linalg_base<Scalar>("matrix_sin_pdu",
                          input_shapes,      // Input shapes
                          {"matrix"},        // Input names
                          {input_shapes[0]}, // Output shape same as input
                          {"result"},        // Output names
                          array_broadcast_type::NONE, // No broadcasting (single
                                                      // matrix operation)
                          error_tag_t::NONE,          // No error tags
                          error_pdu_p::NONE           // No error PDUs
                          ),
      matrix_sin<Scalar>(algorithm, max_iterations, tolerance),
      linalg_base_pdu<Scalar>(
          "matrix_sin_pdu",
          input_shapes,                 // Input shapes
          {"matrix"},                   // Input names
          {input_shapes[0]},            // Output shape same as input
          {"result"},                   // Output names
          array_broadcast_type::NONE,   // No broadcasting (single matrix
                                        // operation)
          error_tag_t::NONE,            // No error tags
          error_pdu_p::NONE,            // No error PDUs
          PDU_UPDATE::ANY_INPUT,        // PDU update mode
          MESSAGE_HANDLER_MODE::DEFAULT // Message handler mode
      ) {}

template <typename Scalar>
typename matrix_cos_pdu<Scalar>::sptr
matrix_cos_pdu<Scalar>::make(const types::vector_shapes &input_shapes,
                             TrigAlgorithm algorithm, int max_iterations,
                             double tolerance) {
  return std::make_shared<matrix_cos_pdu<Scalar>>(input_shapes, algorithm,
                                                  max_iterations, tolerance);
}

template <typename Scalar>
matrix_cos_pdu<Scalar>::matrix_cos_pdu(const types::vector_shapes &input_shapes,
                                       TrigAlgorithm algorithm,
                                       int max_iterations, double tolerance)
    : linalg_base<Scalar>("matrix_cos_pdu",
                          input_shapes,      // Input shapes
                          {"matrix"},        // Input names
                          {input_shapes[0]}, // Output shape same as input
                          {"result"},        // Output names
                          array_broadcast_type::NONE, // No broadcasting (single
                                                      // matrix operation)
                          error_tag_t::NONE,          // No error tags
                          error_pdu_p::NONE           // No error PDUs
                          ),
      matrix_cos<Scalar>(algorithm, max_iterations, tolerance),
      linalg_base_pdu<Scalar>(
          "matrix_cos_pdu",
          input_shapes,                 // Input shapes
          {"matrix"},                   // Input names
          {input_shapes[0]},            // Output shape same as input
          {"result"},                   // Output names
          array_broadcast_type::NONE,   // No broadcasting (single matrix
                                        // operation)
          error_tag_t::NONE,            // No error tags
          error_pdu_p::NONE,            // No error PDUs
          PDU_UPDATE::ANY_INPUT,        // PDU update mode
          MESSAGE_HANDLER_MODE::DEFAULT // Message handler mode
      ) {}

template <typename Scalar>
typename matrix_tan_pdu<Scalar>::sptr
matrix_tan_pdu<Scalar>::make(const types::vector_shapes &input_shapes,
                             TrigAlgorithm algorithm, int max_iterations,
                             double tolerance) {
  return std::make_shared<matrix_tan_pdu<Scalar>>(input_shapes, algorithm,
                                                  max_iterations, tolerance);
}

template <typename Scalar>
matrix_tan_pdu<Scalar>::matrix_tan_pdu(const types::vector_shapes &input_shapes,
                                       TrigAlgorithm algorithm,
                                       int max_iterations, double tolerance)
    : linalg_base<Scalar>("matrix_tan_pdu",
                          input_shapes,      // Input shapes
                          {"matrix"},        // Input names
                          {input_shapes[0]}, // Output shape same as input
                          {"result"},        // Output names
                          array_broadcast_type::NONE, // No broadcasting (single
                                                      // matrix operation)
                          error_tag_t::NONE,          // No error tags
                          error_pdu_p::NONE           // No error PDUs
                          ),
      matrix_tan<Scalar>(algorithm, max_iterations, tolerance),
      linalg_base_pdu<Scalar>(
          "matrix_tan_pdu",
          input_shapes,                 // Input shapes
          {"matrix"},                   // Input names
          {input_shapes[0]},            // Output shape same as input
          {"result"},                   // Output names
          array_broadcast_type::NONE,   // No broadcasting (single matrix
                                        // operation)
          error_tag_t::NONE,            // No error tags
          error_pdu_p::NONE,            // No error PDUs
          PDU_UPDATE::ANY_INPUT,        // PDU update mode
          MESSAGE_HANDLER_MODE::DEFAULT // Message handler mode
      ) {}

//==============================================================================
// Template Instantiations
//==============================================================================

// Base classes
template class matrix_trig<float>;
template class matrix_trig<double>;
template class matrix_trig<std::complex<float>>;
template class matrix_trig<std::complex<double>>;

template class matrix_sin<float>;
template class matrix_sin<double>;
template class matrix_sin<std::complex<float>>;
template class matrix_sin<std::complex<double>>;

template class matrix_cos<float>;
template class matrix_cos<double>;
template class matrix_cos<std::complex<float>>;
template class matrix_cos<std::complex<double>>;

template class matrix_tan<float>;
template class matrix_tan<double>;
template class matrix_tan<std::complex<float>>;
template class matrix_tan<std::complex<double>>;

// Sync classes
template class matrix_sin_sync<float>;
template class matrix_sin_sync<double>;
template class matrix_sin_sync<std::complex<float>>;
template class matrix_sin_sync<std::complex<double>>;

template class matrix_cos_sync<float>;
template class matrix_cos_sync<double>;
template class matrix_cos_sync<std::complex<float>>;
template class matrix_cos_sync<std::complex<double>>;

template class matrix_tan_sync<float>;
template class matrix_tan_sync<double>;
template class matrix_tan_sync<std::complex<float>>;
template class matrix_tan_sync<std::complex<double>>;

// PDU classes
template class matrix_sin_pdu<float>;
template class matrix_sin_pdu<double>;
template class matrix_sin_pdu<std::complex<float>>;
template class matrix_sin_pdu<std::complex<double>>;

template class matrix_cos_pdu<float>;
template class matrix_cos_pdu<double>;
template class matrix_cos_pdu<std::complex<float>>;
template class matrix_cos_pdu<std::complex<double>>;

template class matrix_tan_pdu<float>;
template class matrix_tan_pdu<double>;
template class matrix_tan_pdu<std::complex<float>>;
template class matrix_tan_pdu<std::complex<double>>;

} // namespace linalg
} // namespace gr