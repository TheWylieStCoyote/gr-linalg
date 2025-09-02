/* -*- c++ -*- */
/*
 * Copyright 2024 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "matrix_power_impl.h"
#include <Eigen/Dense>
#include <Eigen/LU>
#include <cmath>
#include <complex>
#include <gnuradio/io_signature.h>
#include <gnuradio/linalg/matrix_power.h>
#include <gnuradio/linalg/types.h>
#include <gnuradio/linalg/utils.h>

using namespace gr::linalg::types::tensor_utils;

namespace gr {
namespace linalg {

//==============================================================================
// Base Implementation
//==============================================================================

template <typename Scalar>
matrix_power<Scalar>::matrix_power()
    : d_matrix_mode(false), d_validate_square(true) {}

template <typename Scalar>
matrix_power<Scalar>::matrix_power(bool matrix_mode, bool validate_square)
    : d_matrix_mode(matrix_mode), d_validate_square(validate_square) {}

template <typename Scalar>
OperationReturn matrix_power<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {
  // Validate inputs
  if (input_matrices.size() != 2) {
    return OperationReturn::FAILURE;
  }
  if (output_matrices.size() != 1) {
    return OperationReturn::FAILURE;
  }

  const auto &matrix = *input_matrices[0];
  const auto &exponent = *input_matrices[1];
  auto &result = *output_matrices[0];

  // Check matrix dimensions
  if (d_matrix_mode) {
    // Matrix power mode: require square matrix
    if (d_validate_square && matrix.rows() != matrix.cols()) {
      return OperationReturn::INVALID_SHAPE;
    }

    // Exponent must be a scalar (1x1 matrix) and integer
    if (exponent.rows() != 1 || exponent.cols() != 1) {
      return OperationReturn::INVALID_SHAPE;
    }

    int int_exp;
    if (!matrix_power_detail::extract_integer_exponent(exponent(0, 0),
                                                       int_exp)) {
      return OperationReturn::FAILURE;
    }

    return matrix_power_computation(matrix, int_exp, result);
  } else {
    // Element-wise mode: support broadcasting
    if (!types::tensor_utils::is_broadcast_compatible(
            {static_cast<int>(matrix.rows()), static_cast<int>(matrix.cols())},
            {static_cast<int>(exponent.rows()),
             static_cast<int>(exponent.cols())})) {
      return OperationReturn::INVALID_SHAPE;
    }

    return elementwise_power(matrix, exponent, result);
  }
}

template <typename Scalar>
OperationReturn matrix_power<Scalar>::elementwise_power(
    const Eigen::Map<
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &matrix,
    const Eigen::Map<
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &exponent,
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &result) {
  const size_t rows = matrix.rows();
  const size_t cols = matrix.cols();
  const size_t exp_rows = exponent.rows();
  const size_t exp_cols = exponent.cols();

  // Element-wise power with broadcasting support
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      // Broadcast exponent indices
      size_t exp_i = (exp_rows == 1) ? 0 : i;
      size_t exp_j = (exp_cols == 1) ? 0 : j;

      result(i, j) = matrix_power_detail::elementwise_pow(
          matrix(i, j), exponent(exp_i, exp_j));
    }
  }

  return OperationReturn::SUCCESS;
}

template <typename Scalar>
OperationReturn matrix_power<Scalar>::matrix_power_computation(
    const Eigen::Map<
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &matrix,
    int exponent,
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &result) {
  const size_t n = matrix.rows();

  // Special case: zero exponent = identity matrix
  if (exponent == 0) {
    result.setZero();
    for (size_t i = 0; i < n; ++i) {
      result(i, i) = Scalar(1.0);
    }
    return OperationReturn::SUCCESS;
  }

  // Positive exponents: repeated squaring
  if (exponent > 0) {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> temp_matrix = matrix;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> temp_result(n, n);

    matrix_power_detail::matrix_power_positive(temp_matrix, exponent,
                                               temp_result);
    result = temp_result;
    return OperationReturn::SUCCESS;
  }

  // Negative exponents: compute inverse first, then positive power
  try {
    Eigen::FullPivLU<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> lu(
        matrix);

    if (!lu.isInvertible()) {
      return OperationReturn::FAILURE;
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> inverse =
        lu.inverse();
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> temp_result(n, n);

    matrix_power_detail::matrix_power_positive(inverse, -exponent, temp_result);
    result = temp_result;
    return OperationReturn::SUCCESS;
  } catch (const std::exception &) {
    return OperationReturn::FAILURE;
  }
}

template <typename Scalar>
void matrix_power<Scalar>::validate_shapes(
    const types::vector_shapes &input_shapes,
    const types::vector_shapes &output_shapes, const std::string &name,
    bool matrix_mode) {
  if (input_shapes.size() != 2) {
    throw std::invalid_argument(name + ": requires exactly 2 input matrices");
  }
  if (output_shapes.size() != 1) {
    throw std::invalid_argument(name + ": requires exactly 1 output matrix");
  }

  const auto &matrix_shape = input_shapes[0];
  const auto &exp_shape = input_shapes[1];
  const auto &result_shape = output_shapes[0];

  if (matrix_mode) {
    // Matrix power: require square matrix and scalar exponent
    if (matrix_shape[0] != matrix_shape[1]) {
      throw std::invalid_argument(name +
                                  ": matrix power requires square matrix");
    }
    if (exp_shape[0] != 1 || exp_shape[1] != 1) {
      throw std::invalid_argument(name +
                                  ": matrix power requires scalar exponent");
    }
  } else {
    // Element-wise: check broadcasting compatibility
    if (!types::tensor_utils::is_broadcast_compatible(matrix_shape,
                                                      exp_shape)) {
      throw std::invalid_argument(name +
                                  ": shapes not compatible for broadcasting");
    }
  }

  // Result shape should match matrix shape
  if (result_shape[0] != matrix_shape[0] ||
      result_shape[1] != matrix_shape[1]) {
    throw std::invalid_argument(name + ": output shape mismatch");
  }
}

template <typename Scalar>
std::vector<size_t>
matrix_power<Scalar>::compute_sizes(const types::vector_shapes &shapes) {
  std::vector<size_t> sizes;
  for (const auto &shape : shapes) {
    sizes.push_back(total_elements(shape));
  }
  return sizes;
}

template <typename Scalar>
types::vector_shapes matrix_power<Scalar>::compute_output_shapes(
    const types::vector_shapes &input_shapes) {
  if (input_shapes.size() != 2) {
    return {};
  }

  // Output shape is the same as matrix (first input) shape
  return {input_shapes[0]};
}

//==============================================================================
// Sync Implementation
//==============================================================================

template <typename Scalar>
typename matrix_power_sync<Scalar>::sptr
matrix_power_sync<Scalar>::make(const types::vector_shapes &input_shapes,
                                bool matrix_mode, bool validate_square) {
  return std::make_shared<matrix_power_sync<Scalar>>(input_shapes, matrix_mode,
                                                     validate_square);
}

template <typename Scalar>
matrix_power_sync<Scalar>::matrix_power_sync(
    const types::vector_shapes &input_shapes, bool matrix_mode,
    bool validate_square)
    : matrix_power<Scalar>(matrix_mode, validate_square),
      linalg_base_sync<Scalar>(
          "matrix_power_sync",
          input_shapes,           // Use provided input shapes
          {"matrix", "exponent"}, // Input names
          {input_shapes[0]},      // Output shape same as first input (matrix)
          {"result"},             // Output names
          array_broadcast_type::MATRIX, // Broadcasting type
          error_tag_t::NONE,            // No error tags
          error_pdu_p::NONE             // No error PDUs
      ) {}

//==============================================================================
// PDU Implementation
//==============================================================================

template <typename Scalar>
typename matrix_power_pdu<Scalar>::sptr
matrix_power_pdu<Scalar>::make(const types::vector_shapes &input_shapes,
                               bool matrix_mode, bool validate_square) {
  return std::make_shared<matrix_power_pdu<Scalar>>(input_shapes, matrix_mode,
                                                    validate_square);
}

template <typename Scalar>
matrix_power_pdu<Scalar>::matrix_power_pdu(
    const types::vector_shapes &input_shapes, bool matrix_mode,
    bool validate_square)
    : linalg_base<Scalar>("matrix_power_pdu", input_shapes,
                          {"matrix", "exponent"}, {input_shapes[0]}, {"result"},
                          array_broadcast_type::MATRIX, error_tag_t::NONE,
                          error_pdu_p::NONE),
      linalg_base_pdu<Scalar>(
          "matrix_power_pdu", input_shapes, {"matrix", "exponent"},
          {input_shapes[0]}, {"result"}, array_broadcast_type::MATRIX,
          error_tag_t::NONE, error_pdu_p::NONE, PDU_UPDATE::ANY_INPUT,
          MESSAGE_HANDLER_MODE::DEFAULT),
      matrix_power<Scalar>(matrix_mode, validate_square) {}

//==============================================================================
// Template Instantiations
//==============================================================================

template class matrix_power<float>;
template class matrix_power<double>;
template class matrix_power<std::complex<float>>;
template class matrix_power<std::complex<double>>;

template class matrix_power_sync<float>;
template class matrix_power_sync<double>;
template class matrix_power_sync<std::complex<float>>;
template class matrix_power_sync<std::complex<double>>;

template class matrix_power_pdu<float>;
template class matrix_power_pdu<double>;
template class matrix_power_pdu<std::complex<float>>;
template class matrix_power_pdu<std::complex<double>>;

} // namespace linalg
} // namespace gr