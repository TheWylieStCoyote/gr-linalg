/* -*- c++ -*- */
/*
 * Copyright 2025 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "solve_least_squares_impl.h"
#include <Eigen/Dense>
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/performance_profiler.h>
#include <gnuradio/linalg/solve_least_squares.h>

namespace gr {
namespace linalg {

static inline types::vector_shapes _dual(const types::vector_shapes &shapes) {
  return shapes;
}

// ===== SYNC IMPLEMENTATION =====

// Factory method
template <typename Scalar>
typename solve_least_squares_sync<Scalar>::sptr
solve_least_squares_sync<Scalar>::make(const types::shape &shape_a,
                                       const types::shape &shape_b,
                                       least_squares_method method,
                                       double tolerance) {
  return gnuradio::make_block_sptr<solve_least_squares_sync_impl<Scalar>>(
      shape_a, shape_b, method, tolerance);
}

// Constructor for sync interface
template <typename Scalar>
solve_least_squares_sync<Scalar>::solve_least_squares_sync(
    const types::shape &shape_a, const types::shape &shape_b,
    least_squares_method method, double tolerance)
    : solve_least_squares<Scalar>(),
      linalg_base_sync<Scalar>("solve_least_squares",
                               types::vector_shapes{shape_a, shape_b},
                               types::vector_shapes{{shape_a[1], shape_b[1]}}) {
}

// Implementation constructor
template <typename Scalar>
solve_least_squares_sync_impl<Scalar>::solve_least_squares_sync_impl(
    const types::shape &shape_a, const types::shape &shape_b,
    least_squares_method method, double tolerance)
    : linalg_base<Scalar>(
          "solve_least_squares", types::vector_shapes{shape_a, shape_b},
          std::vector<std::string>(),
          types::vector_shapes{{shape_a[1], shape_b[1]}},
          std::vector<std::string>(), array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE),
      solve_least_squares_sync<Scalar>(shape_a, shape_b, method, tolerance),
      d_method(method), d_tolerance(tolerance) {
  // Input validation
  if (shape_a.size() != 2 || shape_b.size() != 2) {
    throw std::invalid_argument(
        "solve_least_squares requires 2D matrices for both inputs");
  }
  if (shape_a[0] != shape_b[0]) {
    throw std::invalid_argument(
        "Matrix A rows must equal vector b rows for least squares problem");
  }
  if (shape_a[0] < shape_a[1]) {
    throw std::invalid_argument(
        "Matrix A must be overdetermined (rows >= columns) for least squares");
  }
  if (shape_a[0] <= 0 || shape_a[1] <= 0 || shape_b[0] <= 0 ||
      shape_b[1] <= 0) {
    throw std::invalid_argument("All matrix dimensions must be positive");
  }
  if (tolerance <= 0.0) {
    throw std::invalid_argument("Tolerance must be positive");
  }
}

// Helper function to compute condition number estimate
template <typename MatrixType>
double compute_condition_number_estimate(const MatrixType &A) {
  try {
    using Scalar = typename MatrixType::Scalar;
    Eigen::JacobiSVD<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> svd(
        A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    auto singular_values = svd.singularValues();
    if (singular_values.size() == 0)
      return 1e16;

    double max_sv = std::abs(singular_values(0));
    double min_sv = std::abs(singular_values(singular_values.size() - 1));

    if (min_sv < 1e-16)
      return 1e16; // Essentially singular
    return max_sv / min_sv;
  } catch (...) {
    return 1e16; // Assume worst case if SVD fails
  }
}

// Main operation implementation
template <typename Scalar>
OperationReturn solve_least_squares_sync_impl<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {

  // Validate inputs
  if (input_matrices.size() != 2 || output_matrices.size() != 1) {
    return OperationReturn::INVALID_SHAPE;
  }

  const auto &matrix_A = *input_matrices[0];
  const auto &matrix_b = *input_matrices[1];
  auto &solution_x = *output_matrices[0];

  // Performance profiling
  size_t total_ops =
      matrix_A.rows() * matrix_A.cols() * matrix_b.cols() * sizeof(Scalar);
  PROFILE_LINALG_OPERATION("solve_least_squares", total_ops);

  try {
    // Determine method to use
    least_squares_method method_to_use = d_method;

    if (method_to_use == least_squares_method::AUTO) {
      // Auto-select method based on condition number
      double condition_number = compute_condition_number_estimate(matrix_A);

      if (condition_number < 1e12) {
        method_to_use = least_squares_method::NORMAL_EQ;
      } else if (condition_number < 1e15) {
        method_to_use = least_squares_method::QR;
      } else {
        method_to_use = least_squares_method::SVD;
      }
    }

    // Solve using selected method
    switch (method_to_use) {
    case least_squares_method::NORMAL_EQ: {
      // Normal Equations: x = (A^T A)^(-1) A^T b
      // Most efficient but numerically unstable for ill-conditioned matrices
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> AtA =
          matrix_A.transpose() * matrix_A;
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Atb =
          matrix_A.transpose() * matrix_b;

      // Use LLT decomposition for symmetric positive definite AtA
      Eigen::LLT<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> llt(
          AtA);
      if (llt.info() == Eigen::Success) {
        solution_x = llt.solve(Atb);
      } else {
        // Fall back to LU if LLT fails
        solution_x = AtA.lu().solve(Atb);
      }
      break;
    }

    case least_squares_method::QR: {
      // QR Decomposition: A = QR, then x = R^(-1) Q^T b
      // Better numerical stability than Normal Equations
      Eigen::HouseholderQR<
          Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
          qr(matrix_A);
      solution_x = qr.solve(matrix_b);
      break;
    }

    case least_squares_method::SVD: {
      // Singular Value Decomposition: A = UΣV^T, x = V Σ^+ U^T b
      // Most robust method, handles rank-deficient matrices
      Eigen::JacobiSVD<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
          svd(matrix_A, Eigen::ComputeThinU | Eigen::ComputeThinV);

      // Use specified tolerance for rank determination
      svd.setThreshold(d_tolerance);
      solution_x = svd.solve(matrix_b);
      break;
    }

    default:
      return OperationReturn::INVALID_SHAPE;
    }

    return OperationReturn::SUCCESS;

  } catch (const std::exception &e) {
    return OperationReturn::INVALID_SHAPE;
  }
}

// ===== PDU IMPLEMENTATION =====

// Factory method
template <typename Scalar>
typename solve_least_squares_pdu<Scalar>::sptr
solve_least_squares_pdu<Scalar>::make(const types::shape &shape_a,
                                      const types::shape &shape_b,
                                      least_squares_method method,
                                      double tolerance) {
  return gnuradio::make_block_sptr<solve_least_squares_pdu_impl<Scalar>>(
      shape_a, shape_b, method, tolerance);
}

// Constructor for PDU interface
template <typename Scalar>
solve_least_squares_pdu<Scalar>::solve_least_squares_pdu(
    const types::shape &shape_a, const types::shape &shape_b,
    least_squares_method method, double tolerance)
    : solve_least_squares<Scalar>(),
      linalg_base_pdu<Scalar>(
          "solve_least_squares", types::vector_shapes{shape_a, shape_b},
          std::vector<std::string>(),
          types::vector_shapes{{shape_a[1], shape_b[1]}},
          std::vector<std::string>(), array_broadcast_type::NONE,
          error_tag_t::NONE, error_pdu_p::NONE, PDU_UPDATE::DEFAULT,
          MESSAGE_HANDLER_MODE::DEFAULT) {}

// Implementation constructor
template <typename Scalar>
solve_least_squares_pdu_impl<Scalar>::solve_least_squares_pdu_impl(
    const types::shape &shape_a, const types::shape &shape_b,
    least_squares_method method, double tolerance)
    : linalg_base<Scalar>(
          "solve_least_squares", types::vector_shapes{shape_a, shape_b},
          std::vector<std::string>(),
          types::vector_shapes{{shape_a[1], shape_b[1]}},
          std::vector<std::string>(), array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE),
      solve_least_squares_pdu<Scalar>(shape_a, shape_b, method, tolerance),
      d_method(method), d_tolerance(tolerance) {
  // Input validation (same as sync version)
  if (shape_a.size() != 2 || shape_b.size() != 2) {
    throw std::invalid_argument(
        "solve_least_squares requires 2D matrices for both inputs");
  }
  if (shape_a[0] != shape_b[0]) {
    throw std::invalid_argument(
        "Matrix A rows must equal vector b rows for least squares problem");
  }
  if (shape_a[0] < shape_a[1]) {
    throw std::invalid_argument(
        "Matrix A must be overdetermined (rows >= columns) for least squares");
  }
  if (shape_a[0] <= 0 || shape_a[1] <= 0 || shape_b[0] <= 0 ||
      shape_b[1] <= 0) {
    throw std::invalid_argument("All matrix dimensions must be positive");
  }
  if (tolerance <= 0.0) {
    throw std::invalid_argument("Tolerance must be positive");
  }
}

// Main operation implementation (identical to sync version)
template <typename Scalar>
OperationReturn solve_least_squares_pdu_impl<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {

  // Validate inputs
  if (input_matrices.size() != 2 || output_matrices.size() != 1) {
    return OperationReturn::INVALID_SHAPE;
  }

  const auto &matrix_A = *input_matrices[0];
  const auto &matrix_b = *input_matrices[1];
  auto &solution_x = *output_matrices[0];

  // Performance profiling
  size_t total_ops =
      matrix_A.rows() * matrix_A.cols() * matrix_b.cols() * sizeof(Scalar);
  PROFILE_LINALG_OPERATION("solve_least_squares", total_ops);

  try {
    // Determine method to use
    least_squares_method method_to_use = d_method;

    if (method_to_use == least_squares_method::AUTO) {
      // Auto-select method based on condition number
      double condition_number = compute_condition_number_estimate(matrix_A);

      if (condition_number < 1e12) {
        method_to_use = least_squares_method::NORMAL_EQ;
      } else if (condition_number < 1e15) {
        method_to_use = least_squares_method::QR;
      } else {
        method_to_use = least_squares_method::SVD;
      }
    }

    // Solve using selected method (same logic as sync version)
    switch (method_to_use) {
    case least_squares_method::NORMAL_EQ: {
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> AtA =
          matrix_A.transpose() * matrix_A;
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Atb =
          matrix_A.transpose() * matrix_b;

      Eigen::LLT<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> llt(
          AtA);
      if (llt.info() == Eigen::Success) {
        solution_x = llt.solve(Atb);
      } else {
        solution_x = AtA.lu().solve(Atb);
      }
      break;
    }

    case least_squares_method::QR: {
      Eigen::HouseholderQR<
          Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
          qr(matrix_A);
      solution_x = qr.solve(matrix_b);
      break;
    }

    case least_squares_method::SVD: {
      Eigen::JacobiSVD<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
          svd(matrix_A, Eigen::ComputeThinU | Eigen::ComputeThinV);
      svd.setThreshold(d_tolerance);
      solution_x = svd.solve(matrix_b);
      break;
    }

    default:
      return OperationReturn::INVALID_SHAPE;
    }

    return OperationReturn::SUCCESS;

  } catch (const std::exception &e) {
    return OperationReturn::INVALID_SHAPE;
  }
}

// Explicit template instantiations for sync implementations
template class solve_least_squares_sync<float>;
template class solve_least_squares_sync<double>;
template class solve_least_squares_sync<std::complex<float>>;
template class solve_least_squares_sync<std::complex<double>>;

template class solve_least_squares_sync_impl<float>;
template class solve_least_squares_sync_impl<double>;
template class solve_least_squares_sync_impl<std::complex<float>>;
template class solve_least_squares_sync_impl<std::complex<double>>;

// Explicit template instantiations for PDU implementations
template class solve_least_squares_pdu<float>;
template class solve_least_squares_pdu<double>;
template class solve_least_squares_pdu<std::complex<float>>;
template class solve_least_squares_pdu<std::complex<double>>;

template class solve_least_squares_pdu_impl<float>;
template class solve_least_squares_pdu_impl<double>;
template class solve_least_squares_pdu_impl<std::complex<float>>;
template class solve_least_squares_pdu_impl<std::complex<double>>;

} // namespace linalg
} // namespace gr