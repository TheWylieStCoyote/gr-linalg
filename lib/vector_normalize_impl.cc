/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "vector_normalize_impl.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
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
vector_normalize<Scalar>::vector_normalize(const types::shape &shape, int order)
    : linalg_base<Scalar>("vector_normalize", {shape},
                          compute_output_shapes({shape}),
                          array_broadcast_type::CUSTOM),
      d_order(order) {
  if (shape.size() != 1) {
    throw std::invalid_argument("vector_normalize requires a 1D input vector");
  }
  const int size = shape[0];
  if (size <= 0) {
    throw std::invalid_argument(
        "vector_normalize requires positive vector size");
  }
  if (order < 0 && order != 0) {
    throw std::invalid_argument(
        "vector_normalize requires non-negative order (or 0 for L∞ norm)");
  }
}

template <typename Scalar>
vector_normalize<Scalar>::~vector_normalize() {}

template <typename Scalar>
void vector_normalize<Scalar>::set_order(int order) {
  d_order = order;
}

template <typename Scalar>
int vector_normalize<Scalar>::get_order() const {
  return d_order;
}

template <typename Scalar>
OperationReturn vector_normalize<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {
  // Validate inputs
  if (input_matrices.size() != 1) {
    return OperationReturn::FAILURE;
  }
  if (output_matrices.size() != 1) {
    return OperationReturn::FAILURE;
  }

  const auto &input = *input_matrices[0];
  auto &output = *output_matrices[0];

  // Start performance profiling
  size_t vector_size = input.size() * sizeof(Scalar);
  PROFILE_LINALG_OPERATION("vector_normalize", vector_size);

  // Check for empty vector
  if (input.size() == 0) {
    return OperationReturn::INVALID_SHAPE;
  }

  // Check output dimensions match input dimensions
  if (output.rows() != input.rows() || output.cols() != input.cols()) {
    return OperationReturn::INVALID_SHAPE;
  }

  try {
    // Reshape to column vector for easier computation
    Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> input_vec(
        input.data(), input.size());
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> output_vec(
        output.data(), output.size());

    normalize_vector(input_vec, output_vec);
    return OperationReturn::SUCCESS;
  } catch (const std::exception &e) {
    return OperationReturn::FAILURE;
  }
}

template <typename Scalar>
types::vector_shapes vector_normalize<Scalar>::compute_output_shapes(
    const types::vector_shapes &input_shapes) {
  if (input_shapes.empty()) {
    throw std::invalid_argument(
        "vector_normalize requires at least one input shape");
  }

  const auto &input_shape = input_shapes[0];
  if (input_shape.size() != 1) {
    throw std::invalid_argument("vector_normalize requires 1D input vector");
  }

  // Output has same dimensions as input
  return {input_shape};
}

template <typename Scalar>
void vector_normalize<Scalar>::normalize_vector(
    const Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> &input,
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> &output) const {

  // Compute the norm of the input vector
  RealScalar norm = compute_vector_norm(input);

  // Handle zero norm case - return zero vector
  RealScalar tolerance = std::numeric_limits<RealScalar>::epsilon() * 100;
  if (norm <= tolerance) {
    output.setZero();
    return;
  }

  // Normalize: output = input / norm
  output = input / Scalar(norm);
}

template <typename Scalar>
typename vector_normalize<Scalar>::RealScalar
vector_normalize<Scalar>::compute_vector_norm(
    const Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> &vector)
    const {

  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

  const int size = vector.size();
  if (size == 0) {
    return RealScalar(0);
  }

  if (d_order == 1) {
    // L1 norm: sum of absolute values
    RealScalar norm = RealScalar(0);
    for (int i = 0; i < size; ++i) {
      if constexpr (std::is_same_v<Scalar, std::complex<float>> ||
                    std::is_same_v<Scalar, std::complex<double>>) {
        norm += std::abs(vector(i));
      } else {
        norm += std::abs(vector(i));
      }
    }
    return norm;
  } else if (d_order == 2) {
    // L2 norm: Euclidean norm
    return vector.norm();
  } else if (d_order == 0) {
    // L∞ norm: maximum absolute value
    RealScalar norm = RealScalar(0);
    for (int i = 0; i < size; ++i) {
      if constexpr (std::is_same_v<Scalar, std::complex<float>> ||
                    std::is_same_v<Scalar, std::complex<double>>) {
        norm = std::max(norm, std::abs(vector(i)));
      } else {
        norm = std::max(norm, std::abs(vector(i)));
      }
    }
    return norm;
  } else {
    // General Lp norm: (sum of |x|^p)^(1/p)
    RealScalar norm = RealScalar(0);
    for (int i = 0; i < size; ++i) {
      if constexpr (std::is_same_v<Scalar, std::complex<float>> ||
                    std::is_same_v<Scalar, std::complex<double>>) {
        norm += std::pow(std::abs(vector(i)), RealScalar(d_order));
      } else {
        norm += std::pow(std::abs(vector(i)), RealScalar(d_order));
      }
    }
    return std::pow(norm, RealScalar(1) / RealScalar(d_order));
  }
}

// Sync implementation
template <typename Scalar>
vector_normalize_sync_impl<Scalar>::vector_normalize_sync_impl(
    const types::shape &shape, int order)
    // Fixed: Base class initialization order matches declaration order
    // Class hierarchy: linalg_base (virtual), vector_normalize,
    // linalg_base_sync, vector_normalize_sync
    : linalg_base<Scalar>(
          "vector_normalize_sync", {shape}, std::vector<std::string>{"input"},
          vector_normalize<Scalar>::compute_output_shapes({shape}),
          std::vector<std::string>{"output"}, array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE),
      vector_normalize<Scalar>(shape, order),
      linalg_base_sync<Scalar>(
          "vector_normalize_sync", {shape}, std::vector<std::string>{"input"},
          vector_normalize<Scalar>::compute_output_shapes({shape}),
          std::vector<std::string>{"output"}, array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE, gr::block::TPP_ALL_TO_ALL),
      vector_normalize_sync<Scalar>(shape, order) {}

template <typename Scalar>
vector_normalize_sync_impl<Scalar>::~vector_normalize_sync_impl() {}

// vector_normalize_sync public ctor used by factory
template <typename Scalar>
vector_normalize_sync<Scalar>::vector_normalize_sync(const types::shape &shape,
                                                     int order)
    : vector_normalize<Scalar>(shape, order) {}

// Factory methods
template <typename Scalar>
typename vector_normalize_sync<Scalar>::sptr
vector_normalize_sync<Scalar>::make(const types::shape &shape, int order) {
  return gnuradio::make_block_sptr<vector_normalize_sync_impl<Scalar>>(shape,
                                                                       order);
}

template <typename Scalar>
typename vector_normalize_pdu<Scalar>::sptr
vector_normalize_pdu<Scalar>::make(const types::shape &shape, int order) {
  return std::static_pointer_cast<vector_normalize_pdu<Scalar>>(
      std::make_shared<vector_normalize_pdu_impl<Scalar>>(shape, order));
}

// vector_normalize_pdu public ctor used by factory
template <typename Scalar>
vector_normalize_pdu<Scalar>::vector_normalize_pdu(const types::shape &shape,
                                                   int order)
    : linalg_base<Scalar>(
          "vector_normalize_pdu", {shape}, std::vector<std::string>{"input"},
          vector_normalize<Scalar>::compute_output_shapes({shape}),
          std::vector<std::string>{"output"}, array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE),
      vector_normalize<Scalar>(shape, order),
      linalg_base_pdu<Scalar>(
          "vector_normalize_pdu", {shape}, std::vector<std::string>{"input"},
          vector_normalize<Scalar>::compute_output_shapes({shape}),
          std::vector<std::string>{"output"}, array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE, PDU_UPDATE::ANY_INPUT,
          MESSAGE_HANDLER_MODE::DEFAULT) {}

// PDU implementation
template <typename Scalar>
vector_normalize_pdu_impl<Scalar>::vector_normalize_pdu_impl(
    const types::shape &shape, int order)
    : linalg_base<Scalar>(
          "vector_normalize_pdu", {shape}, std::vector<std::string>{"input"},
          vector_normalize<Scalar>::compute_output_shapes({shape}),
          std::vector<std::string>{"output"}, array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE),
      vector_normalize<Scalar>(shape, order),
      linalg_base_pdu<Scalar>(
          "vector_normalize_pdu", {shape}, std::vector<std::string>{"input"},
          vector_normalize<Scalar>::compute_output_shapes({shape}),
          std::vector<std::string>{"output"}, array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE, PDU_UPDATE::ANY_INPUT,
          MESSAGE_HANDLER_MODE::DEFAULT),
      vector_normalize_pdu<Scalar>(shape, order) {}

template <typename Scalar>
vector_normalize_pdu_impl<Scalar>::~vector_normalize_pdu_impl() {}

// Explicit template instantiations
template class vector_normalize<float>;
template class vector_normalize<double>;
template class vector_normalize<gr_complex>;
template class vector_normalize<std::complex<double>>;

template class vector_normalize_sync_impl<float>;
template class vector_normalize_sync_impl<double>;
template class vector_normalize_sync_impl<gr_complex>;
template class vector_normalize_sync_impl<std::complex<double>>;

template class vector_normalize_pdu_impl<float>;
template class vector_normalize_pdu_impl<double>;
template class vector_normalize_pdu_impl<gr_complex>;
template class vector_normalize_pdu_impl<std::complex<double>>;

// Explicit instantiations for factory make functions
template typename vector_normalize_sync<float>::sptr
vector_normalize_sync<float>::make(const types::shape &, int);
template typename vector_normalize_sync<double>::sptr
vector_normalize_sync<double>::make(const types::shape &, int);
template typename vector_normalize_sync<gr_complex>::sptr
vector_normalize_sync<gr_complex>::make(const types::shape &, int);
template typename vector_normalize_sync<std::complex<double>>::sptr
vector_normalize_sync<std::complex<double>>::make(const types::shape &, int);

template typename vector_normalize_pdu<float>::sptr
vector_normalize_pdu<float>::make(const types::shape &, int);
template typename vector_normalize_pdu<double>::sptr
vector_normalize_pdu<double>::make(const types::shape &, int);
template typename vector_normalize_pdu<gr_complex>::sptr
vector_normalize_pdu<gr_complex>::make(const types::shape &, int);
template typename vector_normalize_pdu<std::complex<double>>::sptr
vector_normalize_pdu<std::complex<double>>::make(const types::shape &, int);

} // namespace linalg
} // namespace gr
