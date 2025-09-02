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
#include <gnuradio/linalg/vector_product_outer.h>

namespace gr {
namespace linalg {

//==============================================================================
// vector_product_outer<Scalar> Implementation
//==============================================================================

template <typename Scalar>
vector_product_outer<Scalar>::vector_product_outer() {
  // Initialize with default shapes that will be overridden
  d_input_shape_a = types::shape{1};
  d_input_shape_b = types::shape{1};
}

template <typename Scalar>
vector_product_outer<Scalar>::vector_product_outer(
    const types::shape &input_shape_a, const types::shape &input_shape_b)
    : d_input_shape_a(input_shape_a), d_input_shape_b(input_shape_b) {}

template <typename Scalar>
OperationReturn vector_product_outer<Scalar>::operation(
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
  auto &output_matrix = *output_vectors[0];

  // Validate dimensions
  if (vector_a.cols() != 1 || vector_b.cols() != 1) {
    return OperationReturn::INVALID_SHAPE;
  }

  const int rows_a = vector_a.rows();
  const int rows_b = vector_b.rows();

  // Check output matrix dimensions
  if (output_matrix.rows() != rows_a || output_matrix.cols() != rows_b) {
    return OperationReturn::INVALID_SHAPE;
  }

  // Compute outer product: output[i,j] = vector_a[i] * vector_b[j]
  for (int i = 0; i < rows_a; ++i) {
    for (int j = 0; j < rows_b; ++j) {
      output_matrix(i, j) = vector_a(i, 0) * vector_b(j, 0);
    }
  }

  return OperationReturn::SUCCESS;
}

template <typename Scalar>
const types::vector_shapes vector_product_outer<Scalar>::compute_output_shapes(
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

  // Output is a matrix of size (rows_a x rows_b)
  types::shape output_shape = {shape_a[0], shape_b[0]};
  return types::vector_shapes{output_shape};
}

template <typename Scalar>
const gr::io_signature::sptr vector_product_outer<Scalar>::make_input_signature(
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
vector_product_outer<Scalar>::make_output_signature(
    const types::vector_shapes &input_shapes) {

  const auto output_shapes = compute_output_shapes(input_shapes);
  if (output_shapes.empty()) {
    return gr::io_signature::make(0, 0, 0);
  }

  const int output_size = types::compute_size<Scalar>(output_shapes[0]);
  return gr::io_signature::make(1, 1, output_size);
}

//==============================================================================
// vector_product_outer_sync<Scalar> Implementation
//==============================================================================

template <typename Scalar>
typename vector_product_outer_sync<Scalar>::sptr
vector_product_outer_sync<Scalar>::make(const types::shape &input_shape_a,
                                        const types::shape &input_shape_b) {
  return gnuradio::make_block_sptr<vector_product_outer_sync<Scalar>>(
      input_shape_a, input_shape_b);
}

template <typename Scalar>
vector_product_outer_sync<Scalar>::vector_product_outer_sync(
    const types::shape &input_shape_a, const types::shape &input_shape_b)
    : linalg_base_sync<Scalar>(
          "vector_product_outer_sync",
          types::vector_shapes{input_shape_a, input_shape_b},
          vector_product_outer<Scalar>::compute_output_shapes(
              types::vector_shapes{input_shape_a, input_shape_b})),
      vector_product_outer<Scalar>(input_shape_a, input_shape_b) {}

//==============================================================================
// vector_product_outer_pdu<Scalar> Implementation
//==============================================================================

template <typename Scalar>
typename vector_product_outer_pdu<Scalar>::sptr
vector_product_outer_pdu<Scalar>::make(
    const types::vector_shapes &input_shapes) {
  return gnuradio::make_block_sptr<vector_product_outer_pdu<Scalar>>(
      input_shapes);
}

template <typename Scalar>
vector_product_outer_pdu<Scalar>::vector_product_outer_pdu(
    const types::vector_shapes &input_shapes)
    : linalg_base_pdu<Scalar>(
          "vector_product_outer_pdu", input_shapes,
          std::vector<std::string>{"input_a", "input_b"},
          vector_product_outer<Scalar>::compute_output_shapes(input_shapes),
          std::vector<std::string>{"output"}, array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE, PDU_UPDATE::DEFAULT,
          MESSAGE_HANDLER_MODE::DEFAULT),
      vector_product_outer<Scalar>() {}

//==============================================================================
// Explicit Template Instantiations
//==============================================================================

template class vector_product_outer<float>;
template class vector_product_outer<double>;
template class vector_product_outer<std::complex<float>>;
template class vector_product_outer<std::complex<double>>;

template class vector_product_outer_sync<float>;
template class vector_product_outer_sync<double>;
template class vector_product_outer_sync<std::complex<float>>;
template class vector_product_outer_sync<std::complex<double>>;

template class vector_product_outer_pdu<float>;
template class vector_product_outer_pdu<double>;
template class vector_product_outer_pdu<std::complex<float>>;
template class vector_product_outer_pdu<std::complex<double>>;

} /* namespace linalg */
} /* namespace gr */
