/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "matrix_pseudo_determinant_impl.h"
#include <algorithm>
#include <cmath>
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/performance_profiler.h>
#include <pmt/pmt.h>
#include <stdexcept>
#include <vector>

namespace gr {
namespace linalg {

static inline types::vector_shapes _single(const types::shape &s) {
  return types::vector_shapes{s};
}

// ===== SYNC IMPLEMENTATION =====

// Factory method
template <typename Scalar>
typename matrix_pseudo_determinant_sync<Scalar>::sptr
matrix_pseudo_determinant_sync<Scalar>::make(const types::shape &shape) {
  return gnuradio::make_block_sptr<matrix_pseudo_determinant_sync_impl<Scalar>>(
      shape);
}

// Constructor for sync interface
template <typename Scalar>
matrix_pseudo_determinant_sync<Scalar>::matrix_pseudo_determinant_sync(
    const types::shape &shape)
    : matrix_pseudo_determinant<Scalar>(),
      linalg_base_sync<Scalar>("matrix_pseudo_determinant", _single(shape),
                               _single(types::shape{1, 1})) {}

// Implementation constructor
template <typename Scalar>
matrix_pseudo_determinant_sync_impl<Scalar>::
    matrix_pseudo_determinant_sync_impl(
        const typename matrix_pseudo_determinant_sync_impl<Scalar>::shape
            &shape)
    : linalg_base<Scalar>(
          "matrix_pseudo_determinant", _single(shape),
          std::vector<std::string>(), _single(types::shape{1, 1}),
          std::vector<std::string>(), array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE),
      matrix_pseudo_determinant_sync<Scalar>(shape) {
  // Input validation: matrix must be non-empty
  if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
    throw std::invalid_argument("Matrix shape must be non-empty [rows, cols] "
                                "with rows > 0 and cols > 0");
  }
}

// Core pseudo-determinant operation for sync
template <typename Scalar>
OperationReturn matrix_pseudo_determinant_sync_impl<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {

  if (input_matrices.size() != 1 || output_matrices.size() != 1)
    return OperationReturn::INVALID_SHAPE;

  auto &input_matrix = *input_matrices[0];
  auto &output_matrix = *output_matrices[0];

  PROFILE_LINALG_OPERATION("matrix_pseudo_determinant",
                           input_matrix.rows() * input_matrix.cols() *
                               sizeof(Scalar));

  // Compute SVD to get singular values
  Eigen::JacobiSVD<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> svd(
      input_matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
  auto singular_values = svd.singularValues();

  // Compute pseudo-determinant as product of non-zero singular values
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  RealScalar pseudo_det = 1.0;

  // Set tolerance for considering singular values as zero
  const RealScalar tolerance =
      std::max(input_matrix.rows(), input_matrix.cols()) *
      singular_values.maxCoeff() * std::numeric_limits<RealScalar>::epsilon();

  bool has_nonzero_sv = false;
  for (int i = 0; i < singular_values.size(); ++i) {
    if (singular_values[i] > tolerance) {
      pseudo_det *= singular_values[i];
      has_nonzero_sv = true;
    }
  }

  // If no non-zero singular values, pseudo-determinant is 0
  if (!has_nonzero_sv) {
    pseudo_det = 0.0;
  }

  // For complex types, the result is still real-valued
  if constexpr (std::is_same_v<Scalar, std::complex<float>> ||
                std::is_same_v<Scalar, std::complex<double>>) {
    output_matrix(0, 0) =
        Scalar(pseudo_det, 0.0); // Real result with zero imaginary part
  } else {
    output_matrix(0, 0) = static_cast<Scalar>(pseudo_det);
  }

  return OperationReturn::SUCCESS;
}

// ===== PDU IMPLEMENTATION =====

// Factory method for PDU
template <typename Scalar>
typename matrix_pseudo_determinant_pdu<Scalar>::sptr
matrix_pseudo_determinant_pdu<Scalar>::make(const types::shape &shape) {
  return gnuradio::make_block_sptr<matrix_pseudo_determinant_pdu_impl<Scalar>>(
      shape);
}

// PDU implementation constructor
template <typename Scalar>
matrix_pseudo_determinant_pdu_impl<Scalar>::matrix_pseudo_determinant_pdu_impl(
    const shape &shape)
    : linalg_base<Scalar>(
          "matrix_pseudo_determinant_pdu", _single(shape),
          std::vector<std::string>{}, _single(types::shape{1, 1}),
          std::vector<std::string>{}, array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE),
      linalg_base_pdu<Scalar>(
          "matrix_pseudo_determinant_pdu", _single(shape),
          std::vector<std::string>{}, _single(types::shape{1, 1}),
          std::vector<std::string>{}, array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE, PDU_UPDATE::ANY_INPUT,
          MESSAGE_HANDLER_MODE::DEFAULT) {
  // Input validation: matrix must be non-empty
  if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
    throw std::invalid_argument("Matrix shape must be non-empty [rows, cols] "
                                "with rows > 0 and cols > 0");
  }
}

// Core pseudo-determinant operation for PDU (same algorithm)
template <typename Scalar>
OperationReturn matrix_pseudo_determinant_pdu_impl<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {

  if (input_matrices.size() != 1 || output_matrices.size() != 1)
    return OperationReturn::INVALID_SHAPE;

  auto &input_matrix = *input_matrices[0];
  auto &output_matrix = *output_matrices[0];

  PROFILE_LINALG_OPERATION("matrix_pseudo_determinant_pdu",
                           input_matrix.rows() * input_matrix.cols() *
                               sizeof(Scalar));

  // Compute SVD to get singular values
  Eigen::JacobiSVD<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> svd(
      input_matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
  auto singular_values = svd.singularValues();

  // Compute pseudo-determinant as product of non-zero singular values
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  RealScalar pseudo_det = 1.0;

  // Set tolerance for considering singular values as zero
  const RealScalar tolerance =
      std::max(input_matrix.rows(), input_matrix.cols()) *
      singular_values.maxCoeff() * std::numeric_limits<RealScalar>::epsilon();

  bool has_nonzero_sv = false;
  for (int i = 0; i < singular_values.size(); ++i) {
    if (singular_values[i] > tolerance) {
      pseudo_det *= singular_values[i];
      has_nonzero_sv = true;
    }
  }

  // If no non-zero singular values, pseudo-determinant is 0
  if (!has_nonzero_sv) {
    pseudo_det = 0.0;
  }

  // For complex types, the result is still real-valued
  if constexpr (std::is_same_v<Scalar, std::complex<float>> ||
                std::is_same_v<Scalar, std::complex<double>>) {
    output_matrix(0, 0) =
        Scalar(pseudo_det, 0.0); // Real result with zero imaginary part
  } else {
    output_matrix(0, 0) = static_cast<Scalar>(pseudo_det);
  }

  return OperationReturn::SUCCESS;
}

// Explicit template instantiations for all supported precision types

// Sync implementations
template class matrix_pseudo_determinant_sync_impl<float>;
template class matrix_pseudo_determinant_sync_impl<double>;
template class matrix_pseudo_determinant_sync_impl<std::complex<float>>;
template class matrix_pseudo_determinant_sync_impl<std::complex<double>>;

// Sync constructors
template matrix_pseudo_determinant_sync<float>::matrix_pseudo_determinant_sync(
    const types::shape &);
template matrix_pseudo_determinant_sync<double>::matrix_pseudo_determinant_sync(
    const types::shape &);
template matrix_pseudo_determinant_sync<
    std::complex<float>>::matrix_pseudo_determinant_sync(const types::shape &);
template matrix_pseudo_determinant_sync<
    std::complex<double>>::matrix_pseudo_determinant_sync(const types::shape &);

// Sync make functions
template typename matrix_pseudo_determinant_sync<float>::sptr
matrix_pseudo_determinant_sync<float>::make(const types::shape &);
template typename matrix_pseudo_determinant_sync<double>::sptr
matrix_pseudo_determinant_sync<double>::make(const types::shape &);
template typename matrix_pseudo_determinant_sync<std::complex<float>>::sptr
matrix_pseudo_determinant_sync<std::complex<float>>::make(const types::shape &);
template typename matrix_pseudo_determinant_sync<std::complex<double>>::sptr
matrix_pseudo_determinant_sync<std::complex<double>>::make(
    const types::shape &);

// PDU implementations
template class matrix_pseudo_determinant_pdu_impl<float>;
template class matrix_pseudo_determinant_pdu_impl<double>;
template class matrix_pseudo_determinant_pdu_impl<std::complex<float>>;
template class matrix_pseudo_determinant_pdu_impl<std::complex<double>>;

// PDU make functions
template typename matrix_pseudo_determinant_pdu<float>::sptr
matrix_pseudo_determinant_pdu<float>::make(const types::shape &);
template typename matrix_pseudo_determinant_pdu<double>::sptr
matrix_pseudo_determinant_pdu<double>::make(const types::shape &);
template typename matrix_pseudo_determinant_pdu<std::complex<float>>::sptr
matrix_pseudo_determinant_pdu<std::complex<float>>::make(const types::shape &);
template typename matrix_pseudo_determinant_pdu<std::complex<double>>::sptr
matrix_pseudo_determinant_pdu<std::complex<double>>::make(const types::shape &);

} /* namespace linalg */
} /* namespace gr */