/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <Eigen/Dense>
#include <gnuradio/io_signature.h>
#include <gnuradio/linalg/vector_to_matrix.h>
#include <stdexcept>

namespace gr {
namespace linalg {

// Template-based constructor
template <typename Scalar>
vector_to_matrix<Scalar>::vector_to_matrix(const types::shape &input_shape,
                                           const types::shape &output_shape)
    : d_input_shape(input_shape), d_output_shape(output_shape) {
  // Validate input shapes
  if (input_shape.size() != 1) {
    throw std::invalid_argument(
        "vector_to_matrix: input must be 1D vector [n]");
  }
  if (output_shape.size() != 2) {
    throw std::invalid_argument(
        "vector_to_matrix: output must be 2D matrix [rows, cols]");
  }

  // Validate element count preservation
  size_t input_elements = input_shape[0];
  size_t output_elements = output_shape[0] * output_shape[1];
  if (input_elements != output_elements) {
    throw std::invalid_argument("vector_to_matrix: input vector length (" +
                                std::to_string(input_elements) +
                                ") must equal output matrix elements (" +
                                std::to_string(output_elements) + ")");
  }
}

// Compute output shapes
template <typename Scalar>
types::vector_shapes vector_to_matrix<Scalar>::compute_output_shapes(
    const types::shape &input_shape, const types::shape &output_shape) {
  // Output has the specified output shape
  return {output_shape};
}

// Core operation implementation
template <typename Scalar>
OperationReturn vector_to_matrix<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {
  if (input_matrices.size() != 1) {
    return OperationReturn::INVALID_SHAPE;
  }
  if (output_matrices.size() != 1) {
    return OperationReturn::INVALID_SHAPE;
  }

  const auto &input_vector = *input_matrices[0]; // Input vector as matrix
  auto &output_matrix = *output_matrices[0];     // Output matrix

  try {
    // Validate total element count (more flexible than specific dimensions)
    size_t input_elements = input_vector.size();
    size_t expected_elements = d_input_shape[0];
    size_t output_elements = output_matrix.size();
    size_t expected_output_elements = d_output_shape[0] * d_output_shape[1];

    if (input_elements != expected_elements) {
      return OperationReturn::INVALID_SHAPE;
    }
    if (output_elements != expected_output_elements) {
      return OperationReturn::INVALID_SHAPE;
    }

    // Simple reshape operation: copy data from input to output matrix
    // Elements are copied in column-major order (Eigen's default)
    const int total_elements = d_input_shape[0];

    // Copy elements using linear indexing (works regardless of input matrix
    // layout)
    for (int i = 0; i < total_elements; ++i) {
      int row = i % d_output_shape[0];
      int col = i / d_output_shape[0];

      // Access input using linear indexing to handle any matrix layout
      Scalar input_value = *(input_vector.data() + i);
      output_matrix(row, col) = input_value;
    }

    return OperationReturn::SUCCESS;
  } catch (const std::exception &) {
    return OperationReturn::FAILURE;
  }
}

// Sync block implementation
template <typename Scalar>
vector_to_matrix_sync<Scalar>::vector_to_matrix_sync(
    const types::shape &input_shape, const types::shape &output_shape)
    : linalg_base<Scalar>( // MUST explicitly initialize virtual base!
          "vector_to_matrix_sync", {input_shape}, {"input_vector"},
          vector_to_matrix<Scalar>::compute_output_shapes(input_shape,
                                                          output_shape),
          {"output_matrix"}, array_broadcast_type::CUSTOM, error_tag_t::NONE,
          error_pdu_p::NONE),
      linalg_base_sync<Scalar>(
          "vector_to_matrix_sync", {input_shape}, {"input_vector"},
          vector_to_matrix<Scalar>::compute_output_shapes(input_shape,
                                                          output_shape),
          {"output_matrix"}, array_broadcast_type::CUSTOM, error_tag_t::NONE,
          error_pdu_p::NONE, gr::block::TPP_ALL_TO_ALL),
      vector_to_matrix<Scalar>(input_shape, output_shape) {}

template <typename Scalar>
typename vector_to_matrix_sync<Scalar>::sptr
vector_to_matrix_sync<Scalar>::make(const types::shape &input_shape,
                                    const types::shape &output_shape) {
  return std::make_shared<vector_to_matrix_sync<Scalar>>(input_shape,
                                                         output_shape);
}

// PDU block implementation
template <typename Scalar>
vector_to_matrix_pdu<Scalar>::vector_to_matrix_pdu(
    const types::shape &input_shape, const types::shape &output_shape)
    : linalg_base_pdu<Scalar>("vector_to_matrix_pdu", {input_shape},
                              {"in_vector"},
                              vector_to_matrix<Scalar>::compute_output_shapes(
                                  input_shape, output_shape),
                              {"out_matrix"}, array_broadcast_type::CUSTOM,
                              error_tag_t::NONE, error_pdu_p::NONE),
      vector_to_matrix<Scalar>(input_shape, output_shape) {}

template <typename Scalar>
typename vector_to_matrix_pdu<Scalar>::sptr
vector_to_matrix_pdu<Scalar>::make(const types::shape &input_shape,
                                   const types::shape &output_shape) {
  return std::make_shared<vector_to_matrix_pdu<Scalar>>(input_shape,
                                                        output_shape);
}

// Explicit template instantiations
template class vector_to_matrix<float>;
template class vector_to_matrix<double>;
template class vector_to_matrix<std::complex<float>>;
template class vector_to_matrix<std::complex<double>>;

template class vector_to_matrix_sync<float>;
template class vector_to_matrix_sync<double>;
template class vector_to_matrix_sync<std::complex<float>>;
template class vector_to_matrix_sync<std::complex<double>>;

template class vector_to_matrix_pdu<float>;
template class vector_to_matrix_pdu<double>;
template class vector_to_matrix_pdu<std::complex<float>>;
template class vector_to_matrix_pdu<std::complex<double>>;

} // namespace linalg
} // namespace gr
