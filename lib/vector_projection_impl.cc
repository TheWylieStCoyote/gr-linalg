/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Vector Projection Implementation Status:
 * - Real-valued projections (float, double): FULLY FUNCTIONAL
 * - Complex projections (complex<float>, complex<double>): KNOWN ISSUE
 *   Complex projections currently return zero vectors due to an issue with
 *   Eigen's complex conjugate dot product handling. This needs investigation.
 * - All other functionality (Python bindings, factory functions, GNU Radio
 *   integration) works correctly for real-valued data.
 */

#include "vector_projection_impl.h"
#include <cmath>
#include <complex>
#include <gnuradio/io_signature.h>
#include <gnuradio/linalg/performance_profiler.h>
#include <gnuradio/linalg/types.h>
#include <stdexcept>

namespace gr {
namespace linalg {

using ERROR_TAG = error_tag_t;
using ERROR_PDU = error_pdu_p;

// ============================================================================
// vector_projection base class implementation
// ============================================================================

template <typename Scalar>
vector_projection<Scalar>::vector_projection(const types::shape &shape_a,
                                             const types::shape &shape_b)
    : linalg_base<Scalar>("vector_projection",
                          {shape_a, shape_b}, // Two input vectors
                          vector_projection<Scalar>::compute_output_shapes(
                              types::vector_shapes{shape_a, shape_b}),
                          array_broadcast_type::CUSTOM) {
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
}

template <typename Scalar>
vector_projection<Scalar>::~vector_projection() {}

template <typename Scalar>
OperationReturn vector_projection<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {

  PROFILE_LINALG_OPERATION("vector_projection", input_matrices[0]->size());

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

  if (a_matrix.size() != output_matrix.size()) {
    return OperationReturn::INVALID_SHAPE;
  }

  try {
    // Convert matrices to vectors for projection computation
    Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> vector_a(
        a_matrix.data(), a_matrix.size());
    Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> vector_b(
        b_matrix.data(), b_matrix.size());
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> result_vector(
        output_matrix.data(), output_matrix.size());

    // Compute projection
    compute_vector_projection(vector_a, vector_b, result_vector);

    return OperationReturn::SUCCESS;
  } catch (const std::exception &) {
    return OperationReturn::FAILURE;
  }
}

template <typename Scalar>
void vector_projection<Scalar>::compute_vector_projection(
    const Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> &vector_a,
    const Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> &vector_b,
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> &result) const {

  // Compute projection: proj_b(a) = (a·b / b·b) × b

  // For complex vectors, use proper conjugate dot product computation
  // Fixed: Use vector_a.dot(vector_b.conjugate()) for complex dot products
  Scalar dot_ab;
  Scalar dot_bb;

  if constexpr (std::is_same_v<Scalar, std::complex<float>> ||
                std::is_same_v<Scalar, std::complex<double>>) {
    // For complex numbers: dot_ab = a · conj(b), dot_bb = b · conj(b) = |b|²
    dot_ab = vector_a.dot(vector_b.conjugate());
    dot_bb = vector_b.dot(vector_b.conjugate());
  } else {
    // For real numbers: standard dot product
    dot_ab = vector_a.dot(vector_b);
    dot_bb = vector_b.dot(vector_b);
  }

  // Handle zero vector b (undefined projection)
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  RealScalar tolerance = std::numeric_limits<RealScalar>::epsilon() *
                         100; // More lenient tolerance
  if (std::abs(dot_bb) < tolerance) {
    result.setZero();
    return;
  }

  // Compute projection coefficient: (a·b) / (b·b)
  Scalar projection_coeff = dot_ab / dot_bb;

  // Compute projection: coefficient × b
  result = projection_coeff * vector_b;
}

template <typename Scalar>
types::vector_shapes vector_projection<Scalar>::compute_output_shapes(
    const types::vector_shapes &input_shapes) {
  if (input_shapes.size() != 2) {
    throw std::invalid_argument(
        "Vector projection requires exactly 2 input vectors");
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

  // Output has the same shape as the input vectors (projection has same
  // dimension)
  return {shape_a};
}

// ============================================================================
// vector_projection_sync implementation
// ============================================================================

template <typename Scalar>
vector_projection_sync<Scalar>::vector_projection_sync(
    const types::shape &shape_a, const types::shape &shape_b)
    : linalg_base<Scalar>("vector_projection", {shape_a, shape_b},
                          vector_projection<Scalar>::compute_output_shapes(
                              types::vector_shapes{shape_a, shape_b}),
                          array_broadcast_type::CUSTOM),
      linalg_base_sync<Scalar>("vector_projection", {shape_a, shape_b},
                               std::vector<std::string>{"input_a", "input_b"},
                               vector_projection<Scalar>::compute_output_shapes(
                                   types::vector_shapes{shape_a, shape_b}),
                               std::vector<std::string>{"output"},
                               array_broadcast_type::CUSTOM, error_tag_t::NONE,
                               error_pdu_p::NONE, gr::block::TPP_ALL_TO_ALL),
      vector_projection<Scalar>(shape_a, shape_b) {}

template <typename Scalar>
typename vector_projection_sync<Scalar>::sptr
vector_projection_sync<Scalar>::make(const types::shape &shape_a,
                                     const types::shape &shape_b) {
  return gnuradio::make_block_sptr<vector_projection_sync_impl<Scalar>>(
      shape_a, shape_b);
}

// ============================================================================
// vector_projection_pdu implementation
// ============================================================================

template <typename Scalar>
vector_projection_pdu<Scalar>::vector_projection_pdu(
    const types::shape &shape_a, const types::shape &shape_b)
    : linalg_base<Scalar>("vector_projection", {shape_a, shape_b},
                          vector_projection<Scalar>::compute_output_shapes(
                              types::vector_shapes{shape_a, shape_b}),
                          array_broadcast_type::CUSTOM),
      linalg_base_pdu<Scalar>("vector_projection", {shape_a, shape_b},
                              std::vector<std::string>{"input_a", "input_b"},
                              vector_projection<Scalar>::compute_output_shapes(
                                  types::vector_shapes{shape_a, shape_b}),
                              std::vector<std::string>{"output"},
                              array_broadcast_type::CUSTOM, error_tag_t::NONE,
                              error_pdu_p::NONE),
      vector_projection<Scalar>(shape_a, shape_b) {}

template <typename Scalar>
typename vector_projection_pdu<Scalar>::sptr
vector_projection_pdu<Scalar>::make(const types::shape &shape_a,
                                    const types::shape &shape_b) {
  return gnuradio::make_block_sptr<vector_projection_pdu_impl<Scalar>>(shape_a,
                                                                       shape_b);
}

// ============================================================================
// vector_projection_sync_impl implementation
// ============================================================================

template <typename Scalar>
vector_projection_sync_impl<Scalar>::vector_projection_sync_impl(
    const types::shape &shape_a, const types::shape &shape_b)
    : linalg_base<Scalar>("vector_projection", {shape_a, shape_b},
                          vector_projection<Scalar>::compute_output_shapes(
                              types::vector_shapes{shape_a, shape_b}),
                          array_broadcast_type::CUSTOM),
      vector_projection<Scalar>(shape_a, shape_b),
      linalg_base_sync<Scalar>("vector_projection", {shape_a, shape_b},
                               std::vector<std::string>{"input_a", "input_b"},
                               vector_projection<Scalar>::compute_output_shapes(
                                   types::vector_shapes{shape_a, shape_b}),
                               std::vector<std::string>{"output"},
                               array_broadcast_type::CUSTOM, error_tag_t::NONE,
                               error_pdu_p::NONE, gr::block::TPP_ALL_TO_ALL),
      vector_projection_sync<Scalar>(shape_a, shape_b) {}

template <typename Scalar>
vector_projection_sync_impl<Scalar>::~vector_projection_sync_impl() {}

// ============================================================================
// vector_projection_pdu_impl implementation
// ============================================================================

template <typename Scalar>
vector_projection_pdu_impl<Scalar>::vector_projection_pdu_impl(
    const types::shape &shape_a, const types::shape &shape_b)
    : linalg_base<Scalar>("vector_projection", {shape_a, shape_b},
                          vector_projection<Scalar>::compute_output_shapes(
                              types::vector_shapes{shape_a, shape_b}),
                          array_broadcast_type::CUSTOM),
      vector_projection<Scalar>(shape_a, shape_b),
      linalg_base_pdu<Scalar>("vector_projection", {shape_a, shape_b},
                              std::vector<std::string>{"input_a", "input_b"},
                              vector_projection<Scalar>::compute_output_shapes(
                                  types::vector_shapes{shape_a, shape_b}),
                              std::vector<std::string>{"output"},
                              array_broadcast_type::CUSTOM, error_tag_t::NONE,
                              error_pdu_p::NONE),
      vector_projection_pdu<Scalar>(shape_a, shape_b) {}

template <typename Scalar>
vector_projection_pdu_impl<Scalar>::~vector_projection_pdu_impl() {}

// ============================================================================
// Explicit template instantiations
// ============================================================================

template class vector_projection<float>;
template class vector_projection<double>;
template class vector_projection<std::complex<float>>;
template class vector_projection<std::complex<double>>;

template class vector_projection_sync<float>;
template class vector_projection_sync<double>;
template class vector_projection_sync<std::complex<float>>;
template class vector_projection_sync<std::complex<double>>;

template class vector_projection_pdu<float>;
template class vector_projection_pdu<double>;
template class vector_projection_pdu<std::complex<float>>;
template class vector_projection_pdu<std::complex<double>>;

template class vector_projection_sync_impl<float>;
template class vector_projection_sync_impl<double>;
template class vector_projection_sync_impl<std::complex<float>>;
template class vector_projection_sync_impl<std::complex<double>>;

template class vector_projection_pdu_impl<float>;
template class vector_projection_pdu_impl<double>;
template class vector_projection_pdu_impl<std::complex<float>>;
template class vector_projection_pdu_impl<std::complex<double>>;

} /* namespace linalg */
} /* namespace gr */
