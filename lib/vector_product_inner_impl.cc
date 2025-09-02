/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gnuradio/io_signature.h>
#include <gnuradio/linalg/vector_product_inner.h>

namespace gr {
namespace linalg {

//==============================================================================
// vector_product_inner<Scalar> Implementation
//==============================================================================

template <typename Scalar>
vector_product_inner<Scalar>::vector_product_inner() {
  // Initialize with default shapes that will be overridden
  d_input_shape_a = types::shape{1};
  d_input_shape_b = types::shape{1};
}

template <typename Scalar>
vector_product_inner<Scalar>::vector_product_inner(
    const types::shape &input_shape_a, const types::shape &input_shape_b)
    : d_input_shape_a(input_shape_a), d_input_shape_b(input_shape_b) {}

template <typename Scalar>
OperationReturn vector_product_inner<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_vectors,
    types::vector_matrix_map<Scalar> &output_vectors) {

  // Validate inputs
  if (input_vectors.size() != 2) {
    return OperationReturn::INVALID_SHAPE;
  }
  if (output_vectors.size() != 1) {
    return OperationReturn::INVALID_SHAPE;
  }

  const auto &vector_a = *input_vectors[0];
  const auto &vector_b = *input_vectors[1];
  auto &output_scalar = *output_vectors[0];

  // Validate dimensions
  if (vector_a.cols() != 1 || vector_b.cols() != 1) {
    return OperationReturn::INVALID_SHAPE;
  }

  const int rows_a = vector_a.rows();
  const int rows_b = vector_b.rows();

  // Vectors must have the same size for inner product
  if (rows_a != rows_b) {
    return OperationReturn::INVALID_SHAPE;
  }

  // Check output scalar dimensions (should be 1x1)
  if (output_scalar.rows() != 1 || output_scalar.cols() != 1) {
    return OperationReturn::INVALID_SHAPE;
  }

  // Compute inner product (dot product): result = Σ(conj(a[i]) * b[i])
  // For real scalars, conj() is a no-op
  Scalar result = Scalar(0);
  for (int i = 0; i < rows_a; ++i) {
    // Use conjugate for complex numbers, identity for real numbers
    if constexpr (std::is_same_v<Scalar, std::complex<float>> ||
                  std::is_same_v<Scalar, std::complex<double>>) {
      result += std::conj(vector_a(i, 0)) * vector_b(i, 0);
    } else {
      result += vector_a(i, 0) * vector_b(i, 0);
    }
  }

  output_scalar(0, 0) = result;
  return OperationReturn::SUCCESS;
}

template <typename Scalar>
const types::vector_shapes vector_product_inner<Scalar>::compute_output_shapes(
    const types::vector_shapes &input_shapes) {

  if (input_shapes.size() != 2) {
    return types::vector_shapes{};
  }

  const auto &shape_a = input_shapes[0];
  const auto &shape_b = input_shapes[1];

  // Validate that inputs are vectors (Nx1 matrices)
  if (shape_a.size() != 2 || shape_b.size() != 2 || shape_a[1] != 1 ||
      shape_b[1] != 1) {
    return types::vector_shapes{};
  }

  // Validate that vectors have the same size
  if (shape_a[0] != shape_b[0]) {
    return types::vector_shapes{};
  }

  // Output is a scalar (1x1 matrix)
  types::shape output_shape = {1, 1};
  return types::vector_shapes{output_shape};
}

template <typename Scalar>
const gr::io_signature::sptr vector_product_inner<Scalar>::make_input_signature(
    const types::vector_shapes &input_shapes) {

  if (input_shapes.size() != 2) {
    return gr::io_signature::make(0, 0, 0);
  }

  std::vector<int> input_sizes;
  for (const auto &shape : input_shapes) {
    input_sizes.push_back(types::compute_size<Scalar>(shape));
  }

  return gr::io_signature::makev(2, 2, input_sizes);
}

template <typename Scalar>
const gr::io_signature::sptr
vector_product_inner<Scalar>::make_output_signature(
    const types::vector_shapes &input_shapes) {

  const auto output_shapes = compute_output_shapes(input_shapes);
  if (output_shapes.empty()) {
    return gr::io_signature::make(0, 0, 0);
  }

  const int output_size = types::compute_size<Scalar>(output_shapes[0]);
  return gr::io_signature::make(1, 1, output_size);
}

//==============================================================================
// vector_product_inner_sync<Scalar> Implementation
//==============================================================================

template <typename Scalar>
typename vector_product_inner_sync<Scalar>::sptr
vector_product_inner_sync<Scalar>::make(const types::shape &input_shape_a,
                                        const types::shape &input_shape_b) {
  return gnuradio::make_block_sptr<vector_product_inner_sync<Scalar>>(
      input_shape_a, input_shape_b);
}

template <typename Scalar>
vector_product_inner_sync<Scalar>::vector_product_inner_sync(
    const types::shape &input_shape_a, const types::shape &input_shape_b)
    : linalg_base_sync<Scalar>(
          "vector_product_inner_sync",
          types::vector_shapes{input_shape_a, input_shape_b},
          vector_product_inner<Scalar>::compute_output_shapes(
              types::vector_shapes{input_shape_a, input_shape_b})),
      vector_product_inner<Scalar>(input_shape_a, input_shape_b) {}

//==============================================================================
// vector_product_inner_pdu<Scalar> Implementation
//==============================================================================

template <typename Scalar>
typename vector_product_inner_pdu<Scalar>::sptr
vector_product_inner_pdu<Scalar>::make(
    const types::vector_shapes &input_shapes) {
  return gnuradio::make_block_sptr<vector_product_inner_pdu<Scalar>>(
      input_shapes);
}

template <typename Scalar>
vector_product_inner_pdu<Scalar>::vector_product_inner_pdu(
    const types::vector_shapes &input_shapes)
    : linalg_base_pdu<Scalar>(
          "vector_product_inner_pdu", input_shapes,
          std::vector<std::string>{"input_a", "input_b"},
          vector_product_inner<Scalar>::compute_output_shapes(input_shapes),
          std::vector<std::string>{"output"}, array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE, PDU_UPDATE::DEFAULT,
          MESSAGE_HANDLER_MODE::DEFAULT),
      vector_product_inner<Scalar>() {}

//==============================================================================
// Explicit Template Instantiations
//==============================================================================

template class vector_product_inner<float>;
template class vector_product_inner<double>;
template class vector_product_inner<std::complex<float>>;
template class vector_product_inner<std::complex<double>>;

template class vector_product_inner_sync<float>;
template class vector_product_inner_sync<double>;
template class vector_product_inner_sync<std::complex<float>>;
template class vector_product_inner_sync<std::complex<double>>;

template class vector_product_inner_pdu<float>;
template class vector_product_inner_pdu<double>;
template class vector_product_inner_pdu<std::complex<float>>;
template class vector_product_inner_pdu<std::complex<double>>;

} /* namespace linalg */
} /* namespace gr */
