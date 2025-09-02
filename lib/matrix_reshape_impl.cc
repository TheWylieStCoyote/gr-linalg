/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "matrix_reshape_impl.h"
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
matrix_reshape<Scalar>::matrix_reshape(const types::shape &input_shape,
                                       const types::shape &output_shape)
    : linalg_base<Scalar>("matrix_reshape", {input_shape},
                          compute_output_shapes({input_shape}, output_shape),
                          array_broadcast_type::CUSTOM),
      d_input_shape(input_shape), d_target_shape(output_shape) {

  // Validate input and output shapes
  if (input_shape.size() != 2) {
    throw std::invalid_argument("matrix_reshape requires 2D input matrix");
  }

  if (output_shape.size() != 2) {
    throw std::invalid_argument("matrix_reshape requires 2D output shape");
  }

  validate_reshape(input_shape, output_shape);
}

template <typename Scalar>
matrix_reshape<Scalar>::~matrix_reshape() {}

template <typename Scalar>
void matrix_reshape<Scalar>::validate_reshape(
    const types::shape &input_shape, const types::shape &output_shape) const {
  // Calculate total elements
  int input_elements = 1;
  for (int dim : input_shape) {
    if (dim <= 0) {
      throw std::invalid_argument(
          "matrix_reshape input dimensions must be positive");
    }
    input_elements *= dim;
  }

  int output_elements = 1;
  for (int dim : output_shape) {
    if (dim <= 0) {
      throw std::invalid_argument(
          "matrix_reshape output dimensions must be positive");
    }
    output_elements *= dim;
  }

  if (input_elements != output_elements) {
    throw std::invalid_argument(
        "matrix_reshape: input and output must have same total elements. "
        "Input: " +
        std::to_string(input_elements) +
        ", Output: " + std::to_string(output_elements));
  }
}

template <typename Scalar>
OperationReturn matrix_reshape<Scalar>::operation(
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
  size_t matrix_size = input.size() * sizeof(Scalar);
  PROFILE_LINALG_OPERATION("matrix_reshape", matrix_size);

  // Validate dimensions match expected shapes
  if (input.rows() != d_input_shape[0] || input.cols() != d_input_shape[1]) {
    return OperationReturn::INVALID_SHAPE;
  }

  if (output.rows() != d_target_shape[0] ||
      output.cols() != d_target_shape[1]) {
    return OperationReturn::INVALID_SHAPE;
  }

  try {
    // Reshape operation: copy data in row-major order
    // Eigen matrices are column-major by default, but we want row-major reshape
    // behavior

    const int input_rows = input.rows();
    const int input_cols = input.cols();
    const int output_cols = output.cols();

    // Copy elements in row-major order
    for (int i = 0; i < input_rows; ++i) {
      for (int j = 0; j < input_cols; ++j) {
        // Calculate linear index in row-major order
        int linear_index = i * input_cols + j;

        // Convert back to row, col in output matrix
        int out_row = linear_index / output_cols;
        int out_col = linear_index % output_cols;

        output(out_row, out_col) = input(i, j);
      }
    }

    return OperationReturn::SUCCESS;
  } catch (const std::exception &e) {
    return OperationReturn::FAILURE;
  }
}

template <typename Scalar>
types::vector_shapes matrix_reshape<Scalar>::compute_output_shapes(
    const types::vector_shapes &input_shapes,
    const types::shape &target_shape) {
  if (input_shapes.size() != 1) {
    throw std::invalid_argument(
        "matrix_reshape requires exactly 1 input shape");
  }

  const auto &input_shape = input_shapes[0];

  // Validate reshape is possible
  if (input_shape.size() != 2 || target_shape.size() != 2) {
    throw std::invalid_argument(
        "matrix_reshape requires 2D input and output shapes");
  }

  // Calculate total elements
  int input_elements = input_shape[0] * input_shape[1];
  int target_elements = target_shape[0] * target_shape[1];

  if (input_elements != target_elements) {
    throw std::invalid_argument("matrix_reshape: element count mismatch");
  }

  return {target_shape};
}

// Sync block implementation
template <typename Scalar>
matrix_reshape_sync<Scalar>::matrix_reshape_sync(
    const types::shape &input_shape, const types::shape &output_shape)
    : matrix_reshape<Scalar>(input_shape, output_shape) {}

template <typename Scalar>
typename matrix_reshape_sync<Scalar>::sptr
matrix_reshape_sync<Scalar>::make(const types::shape &input_shape,
                                  const types::shape &output_shape) {
  return std::make_shared<matrix_reshape_sync_impl<Scalar>>(input_shape,
                                                            output_shape);
}

// Implementation classes
template <typename Scalar>
matrix_reshape_sync_impl<Scalar>::matrix_reshape_sync_impl(
    const types::shape &input_shape, const types::shape &output_shape)
    : linalg_base<Scalar>("matrix_reshape_sync", {input_shape}, {"input"},
                          matrix_reshape<Scalar>::compute_output_shapes(
                              {input_shape}, output_shape),
                          {"output"}, array_broadcast_type::CUSTOM,
                          error_tag_t::NONE, error_pdu_p::NONE),
      matrix_reshape<Scalar>(input_shape, output_shape),
      linalg_base_sync<Scalar>("matrix_reshape_sync", {input_shape}, {"input"},
                               matrix_reshape<Scalar>::compute_output_shapes(
                                   {input_shape}, output_shape),
                               {"output"}, array_broadcast_type::CUSTOM,
                               error_tag_t::NONE, error_pdu_p::NONE,
                               gr::block::TPP_ALL_TO_ALL),
      matrix_reshape_sync<Scalar>(input_shape, output_shape) {}

template <typename Scalar>
matrix_reshape_sync_impl<Scalar>::~matrix_reshape_sync_impl() {}

// Explicit template instantiation for all supported types
template class matrix_reshape<float>;
template class matrix_reshape<double>;
template class matrix_reshape<std::complex<float>>;
template class matrix_reshape<std::complex<double>>;

template class matrix_reshape_sync<float>;
template class matrix_reshape_sync<double>;
template class matrix_reshape_sync<std::complex<float>>;
template class matrix_reshape_sync<std::complex<double>>;

template class matrix_reshape_sync_impl<float>;
template class matrix_reshape_sync_impl<double>;
template class matrix_reshape_sync_impl<std::complex<float>>;
template class matrix_reshape_sync_impl<std::complex<double>>;

} /* namespace linalg */
} /* namespace gr */