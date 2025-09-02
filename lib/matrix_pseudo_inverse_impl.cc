/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "matrix_pseudo_inverse_impl.h"
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
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
matrix_pseudo_inverse<Scalar>::matrix_pseudo_inverse(const types::shape &shape)
    : linalg_base<Scalar>("matrix_pseudo_inverse", {shape},
                          compute_output_shapes({shape}),
                          array_broadcast_type::CUSTOM) {
  if (shape.size() != 2) {
    throw std::invalid_argument(
        "matrix_pseudo_inverse requires a 2D input matrix");
  }
  const int rows = utils::matrix_rows(shape);
  const int cols = utils::matrix_cols(shape);
  if (rows <= 0 || cols <= 0) {
    throw std::invalid_argument(
        "matrix_pseudo_inverse requires positive dimensions");
  }
}

template <typename Scalar>
matrix_pseudo_inverse<Scalar>::~matrix_pseudo_inverse() {}

template <typename Scalar>
OperationReturn matrix_pseudo_inverse<Scalar>::operation(
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

  // Start performance profiling
  size_t matrix_size = matrix.rows() * matrix.cols() * sizeof(Scalar);
  PROFILE_LINALG_OPERATION("matrix_pseudo_inverse", matrix_size);

  // Check for empty matrix
  if (matrix.rows() == 0 || matrix.cols() == 0) {
    return OperationReturn::INVALID_SHAPE;
  }

  // Check output dimensions match expected pseudoinverse dimensions
  if (result.rows() != matrix.cols() || result.cols() != matrix.rows()) {
    return OperationReturn::INVALID_SHAPE;
  }

  try {
    compute_pseudoinverse(matrix, result);
    return OperationReturn::SUCCESS;
  } catch (const std::exception &e) {
    return OperationReturn::FAILURE;
  }
}

template <typename Scalar>
types::vector_shapes matrix_pseudo_inverse<Scalar>::compute_output_shapes(
    const types::vector_shapes &input_shapes) {
  if (input_shapes.empty()) {
    throw std::invalid_argument(
        "matrix_pseudo_inverse requires at least one input shape");
  }

  const auto &input_shape = input_shapes[0];
  if (input_shape.size() != 2) {
    throw std::invalid_argument(
        "matrix_pseudo_inverse requires 2D input matrix");
  }

  // Output has transposed dimensions: m×n → n×m
  return {{input_shape[1], input_shape[0]}};
}

template <typename Scalar>
void matrix_pseudo_inverse<Scalar>::compute_pseudoinverse(
    const Eigen::Map<
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &matrix,
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &result)
    const {
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

  try {
    // Handle single element matrix: pinv(a) = 1/a if a != 0, 0 if a = 0
    if (matrix.rows() == 1 && matrix.cols() == 1) {
      RealScalar abs_val = extract_real_value(matrix(0, 0));
      RealScalar tolerance = std::numeric_limits<RealScalar>::epsilon();

      if (is_effectively_zero(abs_val, tolerance)) {
        result(0, 0) = Scalar(0); // Zero pseudoinverse
      } else {
        if constexpr (std::is_same_v<Scalar, std::complex<float>> ||
                      std::is_same_v<Scalar, std::complex<double>>) {
          // For complex numbers, use conjugate
          result(0, 0) = Scalar(1) / std::conj(matrix(0, 0));
        } else {
          result(0, 0) = Scalar(1) / matrix(0, 0);
        }
      }
      return;
    }

    // Use SVD for pseudoinverse: A = U * Σ * V^H
    // Pseudoinverse: A⁺ = V * Σ⁺ * U^H
    // where Σ⁺ has 1/σᵢ for σᵢ > tolerance, 0 otherwise

    Eigen::JacobiSVD<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> svd;
    {
      size_t svd_size = matrix.rows() * matrix.cols() * sizeof(Scalar);
      PROFILE_LINALG_OPERATION("svd_decomposition", svd_size);
      svd.compute(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
    }

    // Get SVD components
    auto U = svd.matrixU();
    auto V = svd.matrixV();
    auto singular_values = svd.singularValues();

    if (singular_values.size() == 0) {
      result.setZero();
      return;
    }

    // Compute tolerance: max(m,n) * max_singular_value * machine_epsilon
    RealScalar max_sv = singular_values(0); // Eigen sorts in descending order
    RealScalar tolerance =
        static_cast<RealScalar>(std::max(matrix.rows(), matrix.cols())) *
        max_sv * std::numeric_limits<RealScalar>::epsilon();

    // Create pseudoinverse of singular values
    Eigen::Matrix<RealScalar, Eigen::Dynamic, 1> sigma_pinv(
        singular_values.size());

    int rank = 0;
    for (int i = 0; i < singular_values.size(); ++i) {
      if (singular_values(i) > tolerance) {
        sigma_pinv(i) = RealScalar(1) / singular_values(i);
        rank++;
      } else {
        sigma_pinv(i) = RealScalar(0);
      }
    }

    // Compute pseudoinverse: A⁺ = V * Σ⁺ * U^H
    // For efficiency, compute V * Σ⁺ first, then multiply by U^H
    auto V_scaled = V * sigma_pinv.asDiagonal();

    if constexpr (std::is_same_v<Scalar, std::complex<float>> ||
                  std::is_same_v<Scalar, std::complex<double>>) {
      result = V_scaled *
               U.adjoint(); // Use adjoint (conjugate transpose) for complex
    } else {
      result = V_scaled * U.transpose(); // Use transpose for real
    }

  } catch (const std::exception &e) {
    // SVD failed, return zero matrix
    result.setZero();
  }
}

template <typename Scalar>
typename matrix_pseudo_inverse<Scalar>::RealScalar
matrix_pseudo_inverse<Scalar>::extract_real_value(const Scalar &value) {
  if constexpr (std::is_same_v<Scalar, std::complex<float>> ||
                std::is_same_v<Scalar, std::complex<double>>) {
    return std::abs(value);
  } else {
    return std::abs(value); // Also handles negative real values
  }
}

template <typename Scalar>
bool matrix_pseudo_inverse<Scalar>::is_effectively_zero(
    const typename matrix_pseudo_inverse<Scalar>::RealScalar &value,
    const typename matrix_pseudo_inverse<Scalar>::RealScalar &tolerance) {
  return value <= tolerance;
}

// Sync implementation
template <typename Scalar>
matrix_pseudo_inverse_sync_impl<Scalar>::matrix_pseudo_inverse_sync_impl(
    const types::shape &shape)
    : linalg_base<Scalar>(
          "matrix_pseudo_inverse_sync", {shape},
          std::vector<std::string>{"input"},
          matrix_pseudo_inverse<Scalar>::compute_output_shapes({shape}),
          std::vector<std::string>{"output"}, array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE),
      linalg_base_sync<Scalar>(
          "matrix_pseudo_inverse_sync", {shape},
          std::vector<std::string>{"input"},
          matrix_pseudo_inverse<Scalar>::compute_output_shapes({shape}),
          std::vector<std::string>{"output"}, array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE, gr::block::TPP_ALL_TO_ALL),
      matrix_pseudo_inverse<Scalar>(shape),
      matrix_pseudo_inverse_sync<Scalar>(shape) {}

template <typename Scalar>
matrix_pseudo_inverse_sync_impl<Scalar>::~matrix_pseudo_inverse_sync_impl() {}

// matrix_pseudo_inverse_sync public ctor used by factory
template <typename Scalar>
matrix_pseudo_inverse_sync<Scalar>::matrix_pseudo_inverse_sync(
    const types::shape &shape)
    : matrix_pseudo_inverse<Scalar>(shape) {}

// Factory methods
template <typename Scalar>
typename matrix_pseudo_inverse_sync<Scalar>::sptr
matrix_pseudo_inverse_sync<Scalar>::make(const types::shape &shape) {
  return gnuradio::make_block_sptr<matrix_pseudo_inverse_sync_impl<Scalar>>(
      shape);
}

template <typename Scalar>
typename matrix_pseudo_inverse_pdu<Scalar>::sptr
matrix_pseudo_inverse_pdu<Scalar>::make(const types::shape &shape) {
  return std::static_pointer_cast<matrix_pseudo_inverse_pdu<Scalar>>(
      std::make_shared<matrix_pseudo_inverse_pdu_impl<Scalar>>(shape));
}

// matrix_pseudo_inverse_pdu public ctor used by factory
template <typename Scalar>
matrix_pseudo_inverse_pdu<Scalar>::matrix_pseudo_inverse_pdu(
    const types::shape &shape)
    : linalg_base<Scalar>(
          "matrix_pseudo_inverse_pdu", {shape},
          std::vector<std::string>{"input"},
          matrix_pseudo_inverse<Scalar>::compute_output_shapes({shape}),
          std::vector<std::string>{"output"}, array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE),
      linalg_base_pdu<Scalar>(
          "matrix_pseudo_inverse_pdu", {shape},
          std::vector<std::string>{"input"},
          matrix_pseudo_inverse<Scalar>::compute_output_shapes({shape}),
          std::vector<std::string>{"output"}, array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE, PDU_UPDATE::ANY_INPUT,
          MESSAGE_HANDLER_MODE::DEFAULT),
      matrix_pseudo_inverse<Scalar>(shape) {}

// PDU implementation
template <typename Scalar>
matrix_pseudo_inverse_pdu_impl<Scalar>::matrix_pseudo_inverse_pdu_impl(
    const types::shape &shape)
    : linalg_base<Scalar>(
          "matrix_pseudo_inverse_pdu", {shape},
          std::vector<std::string>{"input"},
          matrix_pseudo_inverse<Scalar>::compute_output_shapes({shape}),
          std::vector<std::string>{"output"}, array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE),
      linalg_base_pdu<Scalar>(
          "matrix_pseudo_inverse_pdu", {shape},
          std::vector<std::string>{"input"},
          matrix_pseudo_inverse<Scalar>::compute_output_shapes({shape}),
          std::vector<std::string>{"output"}, array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE, PDU_UPDATE::ANY_INPUT,
          MESSAGE_HANDLER_MODE::DEFAULT),
      matrix_pseudo_inverse<Scalar>(shape),
      matrix_pseudo_inverse_pdu<Scalar>(shape) {}

template <typename Scalar>
matrix_pseudo_inverse_pdu_impl<Scalar>::~matrix_pseudo_inverse_pdu_impl() {}

// Explicit template instantiations
template class matrix_pseudo_inverse<float>;
template class matrix_pseudo_inverse<double>;
template class matrix_pseudo_inverse<gr_complex>;
template class matrix_pseudo_inverse<std::complex<double>>;

template class matrix_pseudo_inverse_sync_impl<float>;
template class matrix_pseudo_inverse_sync_impl<double>;
template class matrix_pseudo_inverse_sync_impl<gr_complex>;
template class matrix_pseudo_inverse_sync_impl<std::complex<double>>;

template class matrix_pseudo_inverse_pdu_impl<float>;
template class matrix_pseudo_inverse_pdu_impl<double>;
template class matrix_pseudo_inverse_pdu_impl<gr_complex>;
template class matrix_pseudo_inverse_pdu_impl<std::complex<double>>;

// Explicit instantiations for factory make functions
template typename matrix_pseudo_inverse_sync<float>::sptr
matrix_pseudo_inverse_sync<float>::make(const types::shape &);
template typename matrix_pseudo_inverse_sync<double>::sptr
matrix_pseudo_inverse_sync<double>::make(const types::shape &);
template typename matrix_pseudo_inverse_sync<gr_complex>::sptr
matrix_pseudo_inverse_sync<gr_complex>::make(const types::shape &);
template typename matrix_pseudo_inverse_sync<std::complex<double>>::sptr
matrix_pseudo_inverse_sync<std::complex<double>>::make(const types::shape &);

template typename matrix_pseudo_inverse_pdu<float>::sptr
matrix_pseudo_inverse_pdu<float>::make(const types::shape &);
template typename matrix_pseudo_inverse_pdu<double>::sptr
matrix_pseudo_inverse_pdu<double>::make(const types::shape &);
template typename matrix_pseudo_inverse_pdu<gr_complex>::sptr
matrix_pseudo_inverse_pdu<gr_complex>::make(const types::shape &);
template typename matrix_pseudo_inverse_pdu<std::complex<double>>::sptr
matrix_pseudo_inverse_pdu<std::complex<double>>::make(const types::shape &);

} // namespace linalg
} // namespace gr