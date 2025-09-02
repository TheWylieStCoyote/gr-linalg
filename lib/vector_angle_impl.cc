/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "vector_angle_impl.h"
#include <cmath>
#include <complex>
#include <gnuradio/io_signature.h>
#include <gnuradio/linalg/types.h>
#include <stdexcept>

namespace gr {
namespace linalg {

using ERROR_TAG = error_tag_t;
using ERROR_PDU = error_pdu_p;

// ============================================================================
// Helper functions for angle computation
// ============================================================================

template <typename Scalar>
static
    typename std::enable_if<std::is_floating_point<Scalar>::value, Scalar>::type
    compute_angle_between_vectors(const std::vector<Scalar> &a,
                                  const std::vector<Scalar> &b, bool degrees) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("Vector dimensions must match");
  }

  if (a.empty()) {
    return std::numeric_limits<Scalar>::quiet_NaN();
  }

  // Compute dot product: a · b
  Scalar dot_product = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    dot_product += a[i] * b[i];
  }

  // Compute magnitudes: |a| and |b|
  Scalar mag_a = 0, mag_b = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    mag_a += a[i] * a[i];
    mag_b += b[i] * b[i];
  }
  mag_a = std::sqrt(mag_a);
  mag_b = std::sqrt(mag_b);

  // Handle zero vectors
  if (mag_a == 0 || mag_b == 0) {
    return std::numeric_limits<Scalar>::quiet_NaN();
  }

  // Compute cosine of angle: cos(θ) = (a·b) / (|a||b|)
  Scalar cos_angle = dot_product / (mag_a * mag_b);

  // Clamp to [-1, 1] to handle numerical errors
  cos_angle = std::max(static_cast<Scalar>(-1.0),
                       std::min(static_cast<Scalar>(1.0), cos_angle));

  // Compute angle: θ = arccos(cos(θ))
  Scalar angle = std::acos(cos_angle);

  // Convert to degrees if requested
  if (degrees) {
    angle = angle * 180.0 / M_PI;
  }

  return angle;
}

template <typename T>
static typename std::enable_if<std::is_same<T, std::complex<float>>::value ||
                                   std::is_same<T, std::complex<double>>::value,
                               typename T::value_type>::type
compute_angle_between_vectors(const std::vector<T> &a, const std::vector<T> &b,
                              bool degrees) {
  using Scalar = typename T::value_type;

  if (a.size() != b.size()) {
    throw std::invalid_argument("Vector dimensions must match");
  }

  if (a.empty()) {
    return std::numeric_limits<Scalar>::quiet_NaN();
  }

  // Compute dot product: a · b* (conjugate dot product for complex vectors)
  T dot_product = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    dot_product += a[i] * std::conj(b[i]);
  }

  // Compute magnitudes: |a| and |b|
  Scalar mag_a = 0, mag_b = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    mag_a += std::norm(a[i]); // |z|^2
    mag_b += std::norm(b[i]);
  }
  mag_a = std::sqrt(mag_a);
  mag_b = std::sqrt(mag_b);

  // Handle zero vectors
  if (mag_a == 0 || mag_b == 0) {
    return std::numeric_limits<Scalar>::quiet_NaN();
  }

  // For complex vectors, we take the real part of the normalized dot product
  // cos(θ) = Re((a·b*) / (|a||b|))
  Scalar cos_angle = std::real(dot_product) / (mag_a * mag_b);

  // Clamp to [-1, 1] to handle numerical errors
  cos_angle = std::max(static_cast<Scalar>(-1.0),
                       std::min(static_cast<Scalar>(1.0), cos_angle));

  // Compute angle: θ = arccos(cos(θ))
  Scalar angle = std::acos(cos_angle);

  // Convert to degrees if requested
  if (degrees) {
    angle = angle * 180.0 / M_PI;
  }

  return angle;
}

// ============================================================================
// vector_angle base class implementation
// ============================================================================

template <typename Scalar>
vector_angle<Scalar>::vector_angle(const types::shape &shape, bool degrees)
    : linalg_base<Scalar>("vector_angle",
                          {shape, shape}, // Two input vectors with same shape
                          vector_angle<Scalar>::compute_output_shapes(
                              types::vector_shapes{shape, shape}),
                          array_broadcast_type::CUSTOM),
      d_degrees(degrees) {}

template <typename Scalar>
OperationReturn vector_angle<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {
  if (input_matrices.size() != 2) {
    return OperationReturn::FAILURE;
  }
  if (output_matrices.size() != 1) {
    return OperationReturn::FAILURE;
  }

  const auto &a_matrix = *input_matrices[0];
  const auto &b_matrix = *input_matrices[1];
  auto &output_matrix = *output_matrices[0];

  // Check dimensions
  if (a_matrix.rows() != b_matrix.rows() ||
      a_matrix.cols() != b_matrix.cols()) {
    return OperationReturn::INVALID_SHAPE;
  }

  // Convert matrices to vectors
  std::vector<Scalar> a_vec(a_matrix.data(), a_matrix.data() + a_matrix.size());
  std::vector<Scalar> b_vec(b_matrix.data(), b_matrix.data() + b_matrix.size());

  try {
    // Compute angle
    auto angle = compute_angle_between_vectors(a_vec, b_vec, d_degrees);

    // Store result (output is 1x1 matrix containing the scalar angle)
    output_matrix(0, 0) = angle;

    return OperationReturn::SUCCESS;
  } catch (const std::exception &) {
    return OperationReturn::FAILURE;
  }
}

template <typename Scalar>
types::vector_shapes vector_angle<Scalar>::compute_output_shapes(
    const types::vector_shapes &input_shapes) {
  if (input_shapes.size() != 2) {
    throw std::invalid_argument(
        "Vector angle requires exactly 2 input vectors");
  }

  const auto &shape_a = input_shapes[0];
  const auto &shape_b = input_shapes[1];

  // Validate that both vectors have the same shape
  if (shape_a.size() != shape_b.size()) {
    throw std::invalid_argument(
        "Input vectors must have the same dimensionality");
  }

  for (size_t i = 0; i < shape_a.size(); ++i) {
    if (shape_a[i] != shape_b[i]) {
      throw std::invalid_argument("Input vectors must have the same shape");
    }
  }

  // Output is a scalar (1x1 matrix)
  return {{1}};
}

// ============================================================================
// vector_angle_sync implementation
// ============================================================================

template <typename Scalar>
vector_angle_sync<Scalar>::vector_angle_sync(const types::shape &shape,
                                             bool degrees)
    : linalg_base<Scalar>("vector_angle", {shape, shape}, // Two input vectors
                          vector_angle<Scalar>::compute_output_shapes(
                              types::vector_shapes{shape, shape}),
                          array_broadcast_type::CUSTOM),
      linalg_base_sync<Scalar>("vector_angle", {shape, shape},
                               std::vector<std::string>{"input_a", "input_b"},
                               vector_angle<Scalar>::compute_output_shapes(
                                   types::vector_shapes{shape, shape}),
                               std::vector<std::string>{"output"},
                               array_broadcast_type::CUSTOM, error_tag_t::NONE,
                               error_pdu_p::NONE, gr::block::TPP_ALL_TO_ALL),
      vector_angle<Scalar>(shape, degrees) {}

template <typename Scalar>
typename vector_angle_sync<Scalar>::sptr
vector_angle_sync<Scalar>::make(const types::shape &shape, bool degrees) {
  return gnuradio::make_block_sptr<vector_angle_sync_impl<Scalar>>(shape,
                                                                   degrees);
}

// ============================================================================
// vector_angle_sync_impl implementation
// ============================================================================

template <typename Scalar>
vector_angle_sync_impl<Scalar>::vector_angle_sync_impl(
    const types::shape &shape, bool degrees)
    : linalg_base<Scalar>("vector_angle", {shape, shape}, // Two input vectors
                          vector_angle<Scalar>::compute_output_shapes(
                              types::vector_shapes{shape, shape}),
                          array_broadcast_type::CUSTOM),
      vector_angle<Scalar>(shape, degrees),
      linalg_base_sync<Scalar>("vector_angle", {shape, shape},
                               std::vector<std::string>{"input_a", "input_b"},
                               vector_angle<Scalar>::compute_output_shapes(
                                   types::vector_shapes{shape, shape}),
                               std::vector<std::string>{"output"},
                               array_broadcast_type::CUSTOM, error_tag_t::NONE,
                               error_pdu_p::NONE, gr::block::TPP_ALL_TO_ALL),
      vector_angle_sync<Scalar>(shape, degrees), d_degrees(degrees) {}

template <typename Scalar>
vector_angle_sync_impl<Scalar>::~vector_angle_sync_impl() {}

// ============================================================================
// Explicit template instantiations
// ============================================================================

template class vector_angle<float>;
template class vector_angle<double>;
template class vector_angle<std::complex<float>>;
template class vector_angle<std::complex<double>>;

template class vector_angle_sync<float>;
template class vector_angle_sync<double>;
template class vector_angle_sync<std::complex<float>>;
template class vector_angle_sync<std::complex<double>>;

template class vector_angle_sync_impl<float>;
template class vector_angle_sync_impl<double>;
template class vector_angle_sync_impl<std::complex<float>>;
template class vector_angle_sync_impl<std::complex<double>>;

} /* namespace linalg */
} /* namespace gr */
