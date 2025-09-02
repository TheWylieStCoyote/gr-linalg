/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "dot_product_impl.h"
#include <Eigen/Dense>
#include <algorithm>
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
dot_product<Scalar>::dot_product(const types::shape &shape_0,
                                 const types::shape &shape_1)
    : linalg_base<Scalar>("dot_product", {shape_0, shape_1},
                          compute_output_shapes({shape_0, shape_1}),
                          array_broadcast_type::CUSTOM),
      d_shape_0(shape_0), d_shape_1(shape_1) {
  // Validate both input shapes are 1D vectors
  if (shape_0.size() != 1 || shape_1.size() != 1) {
    throw std::invalid_argument("dot_product requires 1D input vectors");
  }

  const int size_0 = shape_0[0];
  const int size_1 = shape_1[0];

  if (size_0 <= 0 || size_1 <= 0) {
    throw std::invalid_argument("dot_product requires positive vector sizes");
  }

  if (size_0 != size_1) {
    throw std::invalid_argument("dot_product requires vectors of equal length");
  }
}

template <typename Scalar>
dot_product<Scalar>::~dot_product() {}

template <typename Scalar>
Scalar dot_product<Scalar>::compute_dot_product(
    const Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> &vector_a,
    const Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> &vector_b)
    const {

  // Use Eigen's built-in dot product which handles complex conjugation properly
  return vector_a.dot(vector_b);
}

template <typename Scalar>
OperationReturn dot_product<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {
  // Validate inputs
  if (input_matrices.size() != 2) {
    return OperationReturn::FAILURE;
  }
  if (output_matrices.size() != 1) {
    return OperationReturn::FAILURE;
  }

  const auto &input_a = *input_matrices[0];
  const auto &input_b = *input_matrices[1];
  auto &output = *output_matrices[0];

  // Start performance profiling
  size_t vector_size = (input_a.size() + input_b.size()) * sizeof(Scalar);
  PROFILE_LINALG_OPERATION("dot_product", vector_size);

  // Check for empty vectors
  if (input_a.size() == 0 || input_b.size() == 0) {
    return OperationReturn::INVALID_SHAPE;
  }

  // Check vector sizes match
  if (input_a.size() != input_b.size()) {
    return OperationReturn::INVALID_SHAPE;
  }

  // Output should be a scalar (1x1)
  if (output.rows() != 1 || output.cols() != 1) {
    return OperationReturn::INVALID_SHAPE;
  }

  try {
    // Reshape to column vectors for computation
    Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> vec_a(
        input_a.data(), input_a.size());
    Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> vec_b(
        input_b.data(), input_b.size());

    // Compute dot product and store in output scalar
    Scalar result = compute_dot_product(vec_a, vec_b);
    output(0, 0) = result;

    return OperationReturn::SUCCESS;
  } catch (const std::exception &e) {
    return OperationReturn::FAILURE;
  }
}

template <typename Scalar>
types::vector_shapes dot_product<Scalar>::compute_output_shapes(
    const types::vector_shapes &input_shapes) {
  if (input_shapes.size() != 2) {
    throw std::invalid_argument("dot_product requires exactly 2 input shapes");
  }

  // Validate both inputs are 1D vectors
  for (const auto &shape : input_shapes) {
    if (shape.size() != 1) {
      throw std::invalid_argument("dot_product requires 1D input vectors");
    }
    if (shape[0] <= 0) {
      throw std::invalid_argument("dot_product requires positive vector sizes");
    }
  }

  // Check vectors have same length
  if (input_shapes[0][0] != input_shapes[1][0]) {
    throw std::invalid_argument("dot_product requires vectors of equal length");
  }

  // Output is always a scalar
  return {{1}};
}

// Sync block implementation
template <typename Scalar>
dot_product_sync<Scalar>::dot_product_sync(const types::shape &shape_0,
                                           const types::shape &shape_1)
    : dot_product<Scalar>(shape_0, shape_1) {}

template <typename Scalar>
typename dot_product_sync<Scalar>::sptr
dot_product_sync<Scalar>::make(const types::shape &shape_0,
                               const types::shape &shape_1) {
  return std::make_shared<dot_product_sync_impl<Scalar>>(shape_0, shape_1);
}

// PDU block implementation temporarily disabled

// Explicit template instantiation for all supported types
template class dot_product<float>;
template class dot_product<double>;
template class dot_product<std::complex<float>>;
template class dot_product<std::complex<double>>;

template class dot_product_sync<float>;
template class dot_product_sync<double>;
template class dot_product_sync<std::complex<float>>;
template class dot_product_sync<std::complex<double>>;

// PDU classes temporarily disabled
// template class dot_product_pdu<float>;
// template class dot_product_pdu<double>;
// template class dot_product_pdu<std::complex<float>>;
// template class dot_product_pdu<std::complex<double>>;

// Implementation classes
template <typename Scalar>
dot_product_sync_impl<Scalar>::dot_product_sync_impl(
    const types::shape &shape_0, const types::shape &shape_1)
    : linalg_base<Scalar>(
          "dot_product_sync", {shape_0, shape_1}, {"input_a", "input_b"},
          dot_product<Scalar>::compute_output_shapes({shape_0, shape_1}),
          {"output"}, array_broadcast_type::CUSTOM, error_tag_t::NONE,
          error_pdu_p::NONE),
      dot_product<Scalar>(shape_0, shape_1),
      linalg_base_sync<Scalar>(
          "dot_product_sync", {shape_0, shape_1}, {"input_a", "input_b"},
          dot_product<Scalar>::compute_output_shapes({shape_0, shape_1}),
          {"output"}, array_broadcast_type::CUSTOM, error_tag_t::NONE,
          error_pdu_p::NONE, gr::block::TPP_ALL_TO_ALL),
      dot_product_sync<Scalar>(shape_0, shape_1) {}

template <typename Scalar>
dot_product_sync_impl<Scalar>::~dot_product_sync_impl() {}

// PDU implementation temporarily disabled
// template <typename Scalar>
// dot_product_pdu_impl<Scalar>::dot_product_pdu_impl(const types::shape
// &shape_0,
//                                                    const types::shape
//                                                    &shape_1)
//     : dot_product_pdu<Scalar>(shape_0, shape_1) {}
//
// template <typename Scalar>
// dot_product_pdu_impl<Scalar>::~dot_product_pdu_impl() {}

// Explicit instantiation of implementation classes
template class dot_product_sync_impl<float>;
template class dot_product_sync_impl<double>;
template class dot_product_sync_impl<std::complex<float>>;
template class dot_product_sync_impl<std::complex<double>>;

// PDU implementation classes temporarily disabled
// template class dot_product_pdu_impl<float>;
// template class dot_product_pdu_impl<double>;
// template class dot_product_pdu_impl<std::complex<float>>;
// template class dot_product_pdu_impl<std::complex<double>>;

} /* namespace linalg */
} /* namespace gr */
