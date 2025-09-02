/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "decomp_hessenberg_impl.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <gnuradio/io_signature.h>
#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/types.h>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace gr {
namespace linalg {

namespace {
static inline types::vector_shapes _single_shape(const types::shape &s) {
  types::vector_shapes v;
  v.push_back(s);
  return v;
}

static inline types::vector_shapes _hessenberg_out(const types::shape &shape,
                                                   bool compute_q) {
  if (shape.size() != 2)
    throw std::invalid_argument("decomp_hessenberg requires 2D shape");
  const int n = shape[0];
  if (shape[1] != n)
    throw std::invalid_argument("decomp_hessenberg requires square matrix");

  types::vector_shapes out;
  if (compute_q) {
    out.push_back({n, n}); // Q matrix
  }
  out.push_back({n, n}); // H matrix
  return out;
}
} // namespace

// Algorithms - Standard Implementation

template <typename Scalar>
void eigen_hessenberg<Scalar>::decompose(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &Q,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &H) {

  using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  Eigen::HessenbergDecomposition<Mat> hess(input);
  Q = hess.matrixQ();
  H = hess.matrixH();
}

// Advanced Algorithms with Enhanced Features

template <typename Scalar>
void eigen_hessenberg_advanced<Scalar>::decompose(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &Q,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &H) {
  // Default implementation for legacy compatibility
  hessenberg_result<Scalar> result = decompose_advanced(input);
  Q = result.Q;
  H = result.H;
}

template <typename Scalar>
hessenberg_result<Scalar> eigen_hessenberg_advanced<Scalar>::decompose_advanced(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input,
    const hessenberg_options &options) {

  hessenberg_result<Scalar> result;
  using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  // Make a working copy of input matrix
  Mat working_matrix = input;

  // Step 1: Balance matrix if requested
  if (options.balance_matrix) {
    result.balanced = balance_matrix(working_matrix, result.permutation,
                                     result.scale, options.balance_tolerance);
  }

  // Step 2: Perform Hessenberg decomposition
  Eigen::HessenbergDecomposition<Mat> hess_decomp(working_matrix);
  result.Q = hess_decomp.matrixQ();
  result.H = hess_decomp.matrixH();

  // Step 3: Extract Householder reflectors if requested
  if (options.extract_reflectors) {
    result.reflectors = extract_householder_reflectors(hess_decomp);
  }

  // Step 4: Estimate condition number if requested
  if (options.compute_condition) {
    result.condition_estimate = estimate_condition_number(result.H);
  }

  // Step 5: Apply packed storage format if requested
  if (options.packed_storage) {
    // For Hessenberg matrices, packed storage saves the upper triangle + first
    // subdiagonal This is an optimization for storage and certain algorithms
    const int n = result.H.rows();
    Mat packed_H = Mat::Zero(n, n);

    // Copy upper Hessenberg part (upper triangle + first subdiagonal)
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i <= std::min(j + 1, n - 1); ++i) {
        packed_H(i, j) = result.H(i, j);
      }
    }
    result.H = packed_H;
  }

  return result;
}

template <typename Scalar>
bool eigen_hessenberg_advanced<Scalar>::balance_matrix(
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &matrix,
    Eigen::VectorXd &permutation, Eigen::VectorXd &scale,
    double tolerance) const {

  const int n = matrix.rows();
  permutation.resize(n);
  scale.resize(n);

  // Initialize permutation and scale
  for (int i = 0; i < n; ++i) {
    permutation(i) = i;
    scale(i) = 1.0;
  }

  bool balanced = false;
  const int max_iterations = 50;

  // Simple balancing algorithm - scale rows and columns to reduce condition
  // number
  for (int iter = 0; iter < max_iterations; ++iter) {
    bool changed_this_iter = false;

    for (int i = 0; i < n; ++i) {
      // Calculate row and column norms
      typename Eigen::NumTraits<Scalar>::Real row_norm = 0.0;
      typename Eigen::NumTraits<Scalar>::Real col_norm = 0.0;

      for (int j = 0; j < n; ++j) {
        if (i != j) {
          row_norm += std::abs(matrix(i, j));
          col_norm += std::abs(matrix(j, i));
        }
      }

      if (row_norm > tolerance && col_norm > tolerance) {
        // Calculate scaling factor
        typename Eigen::NumTraits<Scalar>::Real scaling_factor =
            std::sqrt(col_norm / row_norm);

        if (std::abs(scaling_factor - 1.0) > tolerance) {
          // Apply scaling
          matrix.row(i) *= scaling_factor;
          matrix.col(i) /= scaling_factor;
          scale(i) *= scaling_factor;
          changed_this_iter = true;
          balanced = true;
        }
      }
    }

    if (!changed_this_iter)
      break;
  }

  return balanced;
}

template <typename Scalar>
std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>
eigen_hessenberg_advanced<Scalar>::extract_householder_reflectors(
    const Eigen::HessenbergDecomposition<
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &hess_decomp)
    const {

  // Extract Householder vectors from the Hessenberg decomposition
  // The vectors are stored in the lower triangular part of the matrix
  const auto &packed_matrix = hess_decomp.packedMatrix();
  const int n = packed_matrix.rows();

  std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> reflectors;
  reflectors.reserve(n - 1);

  for (int k = 0; k < n - 1; ++k) {
    // Extract k-th Householder vector
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> reflector(n - k - 1);

    for (int i = 0; i < n - k - 1; ++i) {
      reflector(i) = packed_matrix(k + 1 + i, k);
    }

    reflectors.push_back(reflector);
  }

  return reflectors;
}

template <typename Scalar>
typename Eigen::NumTraits<Scalar>::Real
eigen_hessenberg_advanced<Scalar>::estimate_condition_number(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &H) const {

  // Estimate condition number using the ratio of largest to smallest diagonal
  // element This is a simple but effective heuristic for Hessenberg matrices
  using Real = typename Eigen::NumTraits<Scalar>::Real;

  if (H.rows() == 0)
    return Real(1.0);

  Real min_diag = std::numeric_limits<Real>::max();
  Real max_diag = Real(0.0);

  for (int i = 0; i < H.rows(); ++i) {
    Real abs_diag = std::abs(H(i, i));
    if (abs_diag > Real(0.0)) {
      min_diag = std::min(min_diag, abs_diag);
      max_diag = std::max(max_diag, abs_diag);
    }
  }

  if (min_diag == std::numeric_limits<Real>::max() || min_diag == Real(0.0)) {
    return std::numeric_limits<Real>::infinity();
  }

  // Add contribution from off-diagonal elements for better estimate
  Real off_diagonal_norm = Real(0.0);
  for (int i = 0; i < H.rows(); ++i) {
    for (int j = 0; j < H.cols(); ++j) {
      if (i != j) {
        off_diagonal_norm += std::abs(H(i, j));
      }
    }
  }

  // Combine diagonal and off-diagonal contributions
  Real condition_estimate = max_diag / min_diag;
  if (off_diagonal_norm > Real(0.0)) {
    condition_estimate *= (Real(1.0) + off_diagonal_norm / max_diag);
  }

  return condition_estimate;
}

// Base

template <typename Scalar>
decomp_hessenberg<Scalar>::decomp_hessenberg(
    const types::shape &shape, bool compute_q,
    std::shared_ptr<hessenberg_algorithm<Scalar>> algorithm,
    const hessenberg_options &options)
    : linalg_base<Scalar>(
          "decomp_hessenberg", _single_shape(shape), std::vector<std::string>{},
          _hessenberg_out(shape, compute_q), std::vector<std::string>{},
          array_broadcast_type::CUSTOM, error_tag_t::NONE, error_pdu_p::NONE),
      compute_q_(compute_q), options_(options) {
  if (shape.size() != 2 || shape[0] != shape[1])
    throw std::invalid_argument("decomp_hessenberg requires square 2D shape");

  // Auto-select algorithm if none provided - prefer advanced version
  if (!algorithm) {
    algorithm_ = std::make_shared<eigen_hessenberg_advanced<Scalar>>();
  } else {
    algorithm_ = algorithm;
  }
}

template <typename Scalar>
OperationReturn decomp_hessenberg<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &ins,
    types::vector_matrix_map<Scalar> &outs) {
  if (ins.size() != 1)
    return OperationReturn::INVALID_SHAPE;

  const int expected_outputs = compute_q_ ? 2 : 1;
  if (static_cast<int>(outs.size()) != expected_outputs)
    return OperationReturn::INVALID_SHAPE;

  try {
    const auto &A = *ins[0];
    const int n = A.rows();

    if (A.cols() != n)
      return OperationReturn::INVALID_SHAPE;

    // Try advanced decomposition if any advanced options are enabled
    bool use_advanced = options_.balance_matrix ||
                        options_.extract_reflectors ||
                        options_.packed_storage || options_.compute_condition;

    if (use_advanced) {
      // Use advanced decomposition and cache results
      auto advanced_algo =
          std::dynamic_pointer_cast<eigen_hessenberg_advanced<Scalar>>(
              algorithm_);
      if (advanced_algo) {
        last_result_ = advanced_algo->decompose_advanced(A, options_);
      } else {
        // Fallback: create temporary advanced algorithm
        eigen_hessenberg_advanced<Scalar> temp_algo;
        last_result_ = temp_algo.decompose_advanced(A, options_);
      }

      // Copy results to outputs
      if (compute_q_) {
        auto &Q_out = *outs[0];
        auto &H_out = *outs[1];

        if (Q_out.rows() != n || Q_out.cols() != n || H_out.rows() != n ||
            H_out.cols() != n)
          return OperationReturn::INVALID_SHAPE;

        Q_out = last_result_.Q;
        H_out = last_result_.H;
      } else {
        auto &H_out = *outs[0];

        if (H_out.rows() != n || H_out.cols() != n)
          return OperationReturn::INVALID_SHAPE;

        H_out = last_result_.H;
      }
    } else {
      // Use standard decomposition for better performance
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Q, H;
      algorithm_->decompose(A, Q, H);

      // Update cache with basic result
      last_result_.Q = Q;
      last_result_.H = H;
      last_result_.balanced = false;
      last_result_.condition_estimate =
          typename Eigen::NumTraits<Scalar>::Real(0.0);

      // Copy results to outputs
      if (compute_q_) {
        auto &Q_out = *outs[0];
        auto &H_out = *outs[1];

        if (Q_out.rows() != n || Q_out.cols() != n || H_out.rows() != n ||
            H_out.cols() != n)
          return OperationReturn::INVALID_SHAPE;

        Q_out = Q;
        H_out = H;
      } else {
        auto &H_out = *outs[0];

        if (H_out.rows() != n || H_out.cols() != n)
          return OperationReturn::INVALID_SHAPE;

        H_out = H;
      }
    }

    return OperationReturn::SUCCESS;
  } catch (...) {
    return OperationReturn::FAILURE;
  }
}

template <typename Scalar>
void decomp_hessenberg<Scalar>::set_algorithm(
    std::shared_ptr<hessenberg_algorithm<Scalar>> algo) {
  if (algo)
    algorithm_ = algo;
}

template <typename Scalar>
void decomp_hessenberg<Scalar>::validate_shape(const types::vector_shapes &in,
                                               const types::vector_shapes &out,
                                               const std::string &) {
  linalg_base<Scalar>::validate_shapes(in, out, "decomp_hessenberg",
                                       array_broadcast_type::CUSTOM);
  if (in.size() != 1 || in[0].size() != 2)
    throw std::invalid_argument("decomp_hessenberg expects one 2D input shape");
  if (in[0][0] != in[0][1])
    throw std::invalid_argument("decomp_hessenberg requires square matrix");
}

template <typename Scalar>
types::vector_shapes decomp_hessenberg<Scalar>::compute_output_shapes(
    const types::vector_shapes &in) {
  if (in.size() != 1)
    return {};
  // Note: We need to determine compute_q from context - for static method,
  // assume true by default
  return _hessenberg_out(in[0], true);
}

// Sync impl

template <typename Scalar>
decomp_hessenberg_sync_impl<Scalar>::decomp_hessenberg_sync_impl(
    const types::shape &shape, bool compute_q,
    std::shared_ptr<hessenberg_algorithm<Scalar>> algorithm,
    const hessenberg_options &options)
    : linalg_base<Scalar>( // MUST explicitly initialize virtual base!
          "decomp_hessenberg_sync", _single_shape(shape),
          std::vector<std::string>{}, _hessenberg_out(shape, compute_q),
          std::vector<std::string>{}, array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE),
      decomp_hessenberg<Scalar>(shape, compute_q, algorithm, options),
      linalg_base_sync<Scalar>(
          "decomp_hessenberg_sync", _single_shape(shape),
          std::vector<std::string>{}, _hessenberg_out(shape, compute_q),
          std::vector<std::string>{}, array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE, gr::block::TPP_ALL_TO_ALL),
      decomp_hessenberg_sync<Scalar>(shape, compute_q, algorithm, options) {
  if (!this->algorithm_) {
    this->algorithm_ = std::make_shared<eigen_hessenberg_advanced<Scalar>>();
  }
}

// Sync class constructor implementations

template <typename Scalar>
decomp_hessenberg_sync<Scalar>::decomp_hessenberg_sync(
    const types::shape &shape, bool compute_q,
    std::shared_ptr<hessenberg_algorithm<Scalar>> algorithm,
    const hessenberg_options &options)
    : decomp_hessenberg<Scalar>(shape, compute_q, algorithm, options) {}

// PDU class constructor implementations

template <typename Scalar>
decomp_hessenberg_pdu<Scalar>::decomp_hessenberg_pdu(
    const types::shape &shape, bool compute_q,
    std::shared_ptr<hessenberg_algorithm<Scalar>> algorithm,
    const hessenberg_options &options)
    : linalg_base_pdu<Scalar>("decomp_hessenberg_pdu", {}, {}, {}, {},
                              array_broadcast_type::NONE, error_tag_t::NONE,
                              error_pdu_p::NONE, PDU_UPDATE::DEFAULT,
                              MESSAGE_HANDLER_MODE::DEFAULT),
      decomp_hessenberg<Scalar>(shape, compute_q, algorithm, options) {}

// Factory

template <typename Scalar>
typename decomp_hessenberg_sync<Scalar>::sptr
decomp_hessenberg_sync<Scalar>::make(
    const types::shape &shape, bool compute_q,
    std::shared_ptr<hessenberg_algorithm<Scalar>> algorithm,
    const hessenberg_options &options) {
  return gnuradio::make_block_sptr<decomp_hessenberg_sync_impl<Scalar>>(
      shape, compute_q, algorithm, options);
}

template <typename Scalar>
typename decomp_hessenberg_pdu<Scalar>::sptr
decomp_hessenberg_pdu<Scalar>::make(
    const types::shape &shape, bool compute_q,
    std::shared_ptr<hessenberg_algorithm<Scalar>> algorithm,
    const hessenberg_options &options) {
  return gnuradio::make_block_sptr<decomp_hessenberg_pdu<Scalar>>(
      shape, compute_q, algorithm, options);
}

// Explicit instantiations

template class eigen_hessenberg<float>;
template class eigen_hessenberg<double>;
template class eigen_hessenberg<std::complex<float>>;
template class eigen_hessenberg<std::complex<double>>;

template class eigen_hessenberg_advanced<float>;
template class eigen_hessenberg_advanced<double>;
template class eigen_hessenberg_advanced<std::complex<float>>;
template class eigen_hessenberg_advanced<std::complex<double>>;

template class decomp_hessenberg<float>;
template class decomp_hessenberg<double>;
template class decomp_hessenberg<std::complex<float>>;
template class decomp_hessenberg<std::complex<double>>;

template class decomp_hessenberg_sync<float>;
template class decomp_hessenberg_sync<double>;
template class decomp_hessenberg_sync<std::complex<float>>;
template class decomp_hessenberg_sync<std::complex<double>>;

template class decomp_hessenberg_pdu<float>;
template class decomp_hessenberg_pdu<double>;
template class decomp_hessenberg_pdu<std::complex<float>>;
template class decomp_hessenberg_pdu<std::complex<double>>;

template class decomp_hessenberg_sync_impl<float>;
template class decomp_hessenberg_sync_impl<double>;
template class decomp_hessenberg_sync_impl<std::complex<float>>;
template class decomp_hessenberg_sync_impl<std::complex<double>>;

} // namespace linalg
} // namespace gr