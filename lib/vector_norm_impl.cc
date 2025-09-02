/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "vector_norm_impl.h"
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
vector_norm<Scalar>::vector_norm(const types::shape &shape, int order)
    : linalg_base<Scalar>("vector_norm", {shape},
                          compute_output_shapes({shape}),
                          array_broadcast_type::CUSTOM),
      d_order(order) {
  if (shape.size() != 1) {
    throw std::invalid_argument("vector_norm requires a 1D input vector");
  }
  const int size = shape[0];
  if (size <= 0) {
    throw std::invalid_argument("vector_norm requires positive vector size");
  }
  if (order < 0 && order != 0) {
    throw std::invalid_argument(
        "vector_norm requires non-negative order (or 0 for L∞ norm)");
  }
}

template <typename Scalar>
vector_norm<Scalar>::~vector_norm() {}

template <typename Scalar>
void vector_norm<Scalar>::set_order(int order) {
  d_order = order;
}

template <typename Scalar>
int vector_norm<Scalar>::get_order() const {
  return d_order;
}

template <typename Scalar>
OperationReturn vector_norm<Scalar>::operation(
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
  PROFILE_LINALG_OPERATION("vector_norm", vector_size);

  // Check for empty vector
  if (input.size() == 0) {
    return OperationReturn::INVALID_SHAPE;
  }

  // Output should be a scalar (1x1)
  if (output.rows() != 1 || output.cols() != 1) {
    return OperationReturn::INVALID_SHAPE;
  }

  try {
    // Reshape to column vector for easier computation
    Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> input_vec(
        input.data(), input.size());

    // Compute norm and store in output scalar
    RealScalar norm = compute_vector_norm(input_vec);
    output(0, 0) = Scalar(norm);

    return OperationReturn::SUCCESS;
  } catch (const std::exception &e) {
    return OperationReturn::FAILURE;
  }
}

template <typename Scalar>
types::vector_shapes vector_norm<Scalar>::compute_output_shapes(
    const types::vector_shapes &input_shapes) {
  if (input_shapes.empty()) {
    throw std::invalid_argument(
        "vector_norm requires at least one input shape");
  }

  const auto &input_shape = input_shapes[0];
  if (input_shape.size() != 1) {
    throw std::invalid_argument("vector_norm requires 1D input vector");
  }

  // Output is a scalar (1x1)
  return {{1}};
}

template <typename Scalar>
typename vector_norm<Scalar>::RealScalar
vector_norm<Scalar>::compute_vector_norm(
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
vector_norm_sync_impl<Scalar>::vector_norm_sync_impl(const types::shape &shape,
                                                     int order)
    // FIXME: Base class initialization order warning (-Wreorder)
    // Problem: Base classes are initialized in different order than declared in
    // class definition Class declaration order: linalg_base, vector_norm,
    // linalg_base_sync, vector_norm_sync Initialization order: linalg_base,
    // linalg_base_sync, vector_norm, vector_norm_sync Solution: Reorder
    // initialization to match class declaration order
    : linalg_base<Scalar>(
          "vector_norm_sync", {shape}, std::vector<std::string>{"input"},
          vector_norm<Scalar>::compute_output_shapes({shape}),
          std::vector<std::string>{"output"}, array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE),
      vector_norm<Scalar>(shape, order),
      linalg_base_sync<Scalar>(
          "vector_norm_sync", {shape}, std::vector<std::string>{"input"},
          vector_norm<Scalar>::compute_output_shapes({shape}),
          std::vector<std::string>{"output"}, array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE, gr::block::TPP_ALL_TO_ALL),
      vector_norm_sync<Scalar>(shape, order) {}

template <typename Scalar>
vector_norm_sync_impl<Scalar>::~vector_norm_sync_impl() {}

// vector_norm_sync public ctor used by factory
template <typename Scalar>
vector_norm_sync<Scalar>::vector_norm_sync(const types::shape &shape, int order)
    : vector_norm<Scalar>(shape, order) {}

// Factory methods
template <typename Scalar>
typename vector_norm_sync<Scalar>::sptr
vector_norm_sync<Scalar>::make(const types::shape &shape, int order) {
  return gnuradio::make_block_sptr<vector_norm_sync_impl<Scalar>>(shape, order);
}

template <typename Scalar>
typename vector_norm_pdu<Scalar>::sptr
vector_norm_pdu<Scalar>::make(const types::shape &shape, int order) {
  return std::static_pointer_cast<vector_norm_pdu<Scalar>>(
      std::make_shared<vector_norm_pdu_impl<Scalar>>(shape, order));
}

// vector_norm_pdu public ctor used by factory
template <typename Scalar>
vector_norm_pdu<Scalar>::vector_norm_pdu(const types::shape &shape, int order)
    // FIXME: Base class initialization order warning (-Wreorder)
    // Problem: Base classes are initialized in different order than declared in
    // class definition Class declaration order: linalg_base, vector_norm,
    // linalg_base_pdu, vector_norm_pdu Initialization order: linalg_base,
    // linalg_base_pdu, vector_norm, vector_norm_pdu Solution: Reorder
    // initialization to match class declaration order
    : linalg_base<Scalar>(
          "vector_norm_pdu", {shape}, std::vector<std::string>{"input"},
          vector_norm<Scalar>::compute_output_shapes({shape}),
          std::vector<std::string>{"output"}, array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE),
      vector_norm<Scalar>(shape, order),
      linalg_base_pdu<Scalar>(
          "vector_norm_pdu", {shape}, std::vector<std::string>{"input"},
          vector_norm<Scalar>::compute_output_shapes({shape}),
          std::vector<std::string>{"output"}, array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE, PDU_UPDATE::ANY_INPUT,
          MESSAGE_HANDLER_MODE::DEFAULT) {}

// PDU implementation
template <typename Scalar>
vector_norm_pdu_impl<Scalar>::vector_norm_pdu_impl(const types::shape &shape,
                                                   int order)
    : linalg_base<Scalar>(
          "vector_norm_pdu", {shape}, std::vector<std::string>{"input"},
          vector_norm<Scalar>::compute_output_shapes({shape}),
          std::vector<std::string>{"output"}, array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE),
      vector_norm<Scalar>(shape, order),
      linalg_base_pdu<Scalar>(
          "vector_norm_pdu", {shape}, std::vector<std::string>{"input"},
          vector_norm<Scalar>::compute_output_shapes({shape}),
          std::vector<std::string>{"output"}, array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE, PDU_UPDATE::ANY_INPUT,
          MESSAGE_HANDLER_MODE::DEFAULT),
      vector_norm_pdu<Scalar>(shape, order) {}

template <typename Scalar>
vector_norm_pdu_impl<Scalar>::~vector_norm_pdu_impl() {}

// Explicit template instantiations
template class vector_norm<float>;
template class vector_norm<double>;
template class vector_norm<gr_complex>;
template class vector_norm<std::complex<double>>;

template class vector_norm_sync_impl<float>;
template class vector_norm_sync_impl<double>;
template class vector_norm_sync_impl<gr_complex>;
template class vector_norm_sync_impl<std::complex<double>>;

template class vector_norm_pdu_impl<float>;
template class vector_norm_pdu_impl<double>;
template class vector_norm_pdu_impl<gr_complex>;
template class vector_norm_pdu_impl<std::complex<double>>;

// Explicit instantiations for factory make functions
template typename vector_norm_sync<float>::sptr
vector_norm_sync<float>::make(const types::shape &, int);
template typename vector_norm_sync<double>::sptr
vector_norm_sync<double>::make(const types::shape &, int);
template typename vector_norm_sync<gr_complex>::sptr
vector_norm_sync<gr_complex>::make(const types::shape &, int);
template typename vector_norm_sync<std::complex<double>>::sptr
vector_norm_sync<std::complex<double>>::make(const types::shape &, int);

template typename vector_norm_pdu<float>::sptr
vector_norm_pdu<float>::make(const types::shape &, int);
template typename vector_norm_pdu<double>::sptr
vector_norm_pdu<double>::make(const types::shape &, int);
template typename vector_norm_pdu<gr_complex>::sptr
vector_norm_pdu<gr_complex>::make(const types::shape &, int);
template typename vector_norm_pdu<std::complex<double>>::sptr
vector_norm_pdu<std::complex<double>>::make(const types::shape &, int);

} // namespace linalg
} // namespace gr
