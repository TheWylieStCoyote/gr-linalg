/* -*- c++ -*- */
/*
 * Copyright 2024 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "matrix_condition_number_impl.h"
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <gnuradio/gr_complex.h>
#include <gnuradio/io_signature.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/performance_profiler.h>
#include <gnuradio/linalg/types.h>
#include <limits>
#include <pmt/pmt.h>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace gr::linalg::types::tensor_utils;

namespace gr {
namespace linalg {

//==============================================================================
// Helper Functions
//==============================================================================

static inline types::vector_shapes _single_input_shape(const types::shape &s) {
  return {s};
}

static inline types::vector_shapes _single_output_shape(const types::shape &s) {
  return {s};
}

static inline std::vector<std::string> _single_input_names() {
  return {"matrix"};
}

static inline std::vector<std::string> _single_output_names() {
  return {"condition_number"};
}

//==============================================================================
// Factory Methods
//==============================================================================

template <typename Scalar>
typename matrix_condition_number_sync<Scalar>::sptr
matrix_condition_number_sync<Scalar>::make(const types::shape &shape) {
  return gnuradio::make_block_sptr<matrix_condition_number_sync_impl<Scalar>>(
      shape);
}

template <typename Scalar>
typename matrix_condition_number_pdu<Scalar>::sptr
matrix_condition_number_pdu<Scalar>::make() {
  return gnuradio::make_block_sptr<matrix_condition_number_pdu_impl<Scalar>>();
}

//==============================================================================
// Base Class Constructors
//==============================================================================

template <typename Scalar>
matrix_condition_number<Scalar>::matrix_condition_number()
    : linalg_base<Scalar>() {}

//==============================================================================
// Sync Class Constructors
//==============================================================================

template <typename Scalar>
matrix_condition_number_sync<Scalar>::matrix_condition_number_sync()
    : matrix_condition_number<Scalar>() {}

template <typename Scalar>
matrix_condition_number_sync<Scalar>::matrix_condition_number_sync(
    const types::shape &shape)
    : matrix_condition_number<Scalar>() {}

template <typename Scalar>
matrix_condition_number_sync_impl<Scalar>::matrix_condition_number_sync_impl()
    : matrix_condition_number_sync_impl<Scalar>({2, 2}) {} // Default to 2x2

template <typename Scalar>
matrix_condition_number_sync_impl<Scalar>::matrix_condition_number_sync_impl(
    const types::shape &shape)
    : linalg_base<Scalar>(
          "matrix_condition_number_sync",
          _single_input_shape(shape), // Input shape for matrix
          _single_input_names(),
          _single_output_shape({1, 1}), // Output: scalar (1x1 matrix)
          _single_output_names(),
          array_broadcast_type::CUSTOM, // Skip strict validation
          error_tag_t::NONE, error_pdu_p::NONE),
      linalg_base_sync<Scalar>(
          "matrix_condition_number_sync",
          _single_input_shape(shape), // Input shape for matrix
          _single_input_names(), _single_output_shape({1, 1}), // Output: scalar
          _single_output_names(),
          array_broadcast_type::CUSTOM, // Skip strict validation
          error_tag_t::NONE, error_pdu_p::NONE, gr::block::TPP_ALL_TO_ALL),
      matrix_condition_number<Scalar>(),
      matrix_condition_number_sync<Scalar>(shape) {}

//==============================================================================
// PDU Class Constructors
//==============================================================================

template <typename Scalar>
matrix_condition_number_pdu<Scalar>::matrix_condition_number_pdu()
    : linalg_base<Scalar>(
          "matrix_condition_number_pdu",
          _single_input_shape({1}), // Flexible input shape
          _single_input_names(), _single_output_shape({1}), // Output: scalar
          _single_output_names(),
          array_broadcast_type::CUSTOM, // Skip strict validation
          error_tag_t::NONE, error_pdu_p::NONE),
      linalg_base_pdu<Scalar>(
          "matrix_condition_number_pdu",
          _single_input_shape({1}), // Flexible input shape
          _single_input_names(), _single_output_shape({1}), // Output: scalar
          _single_output_names(),
          array_broadcast_type::CUSTOM, // Skip strict validation
          error_tag_t::NONE, error_pdu_p::NONE, PDU_UPDATE::ANY_INPUT,
          MESSAGE_HANDLER_MODE::DEFAULT),
      matrix_condition_number<Scalar>() {}

template <typename Scalar>
matrix_condition_number_pdu_impl<Scalar>::matrix_condition_number_pdu_impl()
    : linalg_base<Scalar>(
          "matrix_condition_number_pdu",
          _single_input_shape({1}), // Flexible input shape
          _single_input_names(), _single_output_shape({1}), // Output: scalar
          _single_output_names(),
          array_broadcast_type::CUSTOM, // Skip strict validation
          error_tag_t::NONE, error_pdu_p::NONE),
      linalg_base_pdu<Scalar>(
          "matrix_condition_number_pdu",
          _single_input_shape({1}), // Flexible input shape
          _single_input_names(), _single_output_shape({1}), // Output: scalar
          _single_output_names(),
          array_broadcast_type::CUSTOM, // Skip strict validation
          error_tag_t::NONE, error_pdu_p::NONE, PDU_UPDATE::ANY_INPUT,
          MESSAGE_HANDLER_MODE::DEFAULT),
      matrix_condition_number<Scalar>(), matrix_condition_number_pdu<Scalar>() {
}

//==============================================================================
// Operation Implementation
//==============================================================================

template <typename Scalar>
OperationReturn matrix_condition_number<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {
  // Validate inputs
  if (input_matrices.size() != 1) {
    return OperationReturn::FAILURE;
  }
  if (output_matrices.size() != 1) {
    return OperationReturn::FAILURE;
  }

  const auto &matrix = *input_matrices[0];
  auto &result = *output_matrices[0];

  // Start performance profiling
  size_t matrix_size = matrix.rows() * matrix.cols() * sizeof(Scalar);
  PROFILE_LINALG_OPERATION("matrix_condition_number", matrix_size);

  // Check output is scalar (1×1)
  if (result.rows() != 1 || result.cols() != 1) {
    return OperationReturn::INVALID_SHAPE;
  }

  // Check for empty matrix
  if (matrix.rows() == 0 || matrix.cols() == 0) {
    // Empty matrix has infinite condition number
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    if constexpr (std::is_same_v<Scalar, std::complex<float>> ||
                  std::is_same_v<Scalar, std::complex<double>>) {
      result(0, 0) = Scalar(std::numeric_limits<RealScalar>::infinity(), 0.0);
    } else {
      result(0, 0) = Scalar(std::numeric_limits<RealScalar>::infinity());
    }
    return OperationReturn::SUCCESS;
  }

  return compute_condition_number(matrix, result);
}

template <typename Scalar>
OperationReturn matrix_condition_number<Scalar>::compute_condition_number(
    const Eigen::Map<
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &matrix,
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &result) {
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

  try {
    // Handle single element matrix
    if (matrix.rows() == 1 && matrix.cols() == 1) {
      RealScalar abs_val = extract_real_value(matrix(0, 0));
      if (is_effectively_zero(abs_val,
                              std::numeric_limits<RealScalar>::epsilon())) {
        if constexpr (std::is_same_v<Scalar, std::complex<float>> ||
                      std::is_same_v<Scalar, std::complex<double>>) {
          result(0, 0) =
              Scalar(std::numeric_limits<RealScalar>::infinity(), 0.0);
        } else {
          result(0, 0) = Scalar(std::numeric_limits<RealScalar>::infinity());
        }
      } else {
        if constexpr (std::is_same_v<Scalar, std::complex<float>> ||
                      std::is_same_v<Scalar, std::complex<double>>) {
          result(0, 0) = Scalar(
              1.0, 0.0); // Single non-zero element has condition number 1
        } else {
          result(0, 0) = Scalar(1.0);
        }
      }
      return OperationReturn::SUCCESS;
    }

    // Compute SVD: A = U * S * V^T
    Eigen::JacobiSVD<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> svd;
    {
      size_t svd_size = matrix.rows() * matrix.cols() * sizeof(Scalar);
      PROFILE_LINALG_OPERATION("svd_decomposition", svd_size);
      svd.compute(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
    }

    // Get singular values
    auto singular_values = svd.singularValues();

    if (singular_values.size() == 0) {
      if constexpr (std::is_same_v<Scalar, std::complex<float>> ||
                    std::is_same_v<Scalar, std::complex<double>>) {
        result(0, 0) = Scalar(std::numeric_limits<RealScalar>::infinity(), 0.0);
      } else {
        result(0, 0) = Scalar(std::numeric_limits<RealScalar>::infinity());
      }
      return OperationReturn::SUCCESS;
    }

    // Find maximum and minimum singular values
    RealScalar max_sv =
        singular_values(0); // Eigen sorts singular values in descending order
    RealScalar min_sv = singular_values(singular_values.size() - 1);

    // Set tolerance for zero detection
    RealScalar tolerance =
        std::numeric_limits<RealScalar>::epsilon() *
        static_cast<RealScalar>(std::max(matrix.rows(), matrix.cols())) *
        max_sv;

    // Check if matrix is effectively singular
    if (is_effectively_zero(min_sv, tolerance) || min_sv <= 0) {
      if constexpr (std::is_same_v<Scalar, std::complex<float>> ||
                    std::is_same_v<Scalar, std::complex<double>>) {
        result(0, 0) = Scalar(std::numeric_limits<RealScalar>::infinity(), 0.0);
      } else {
        result(0, 0) = Scalar(std::numeric_limits<RealScalar>::infinity());
      }
    } else {
      // Compute condition number: κ = σ_max / σ_min
      RealScalar condition_num = max_sv / min_sv;

      // Protect against numerical overflow
      if (condition_num > std::numeric_limits<RealScalar>::max() / 2) {
        if constexpr (std::is_same_v<Scalar, std::complex<float>> ||
                      std::is_same_v<Scalar, std::complex<double>>) {
          result(0, 0) =
              Scalar(std::numeric_limits<RealScalar>::infinity(), 0.0);
        } else {
          result(0, 0) = Scalar(std::numeric_limits<RealScalar>::infinity());
        }
      } else {
        if constexpr (std::is_same_v<Scalar, std::complex<float>> ||
                      std::is_same_v<Scalar, std::complex<double>>) {
          result(0, 0) =
              Scalar(condition_num,
                     0.0); // Real condition number with zero imaginary part
        } else {
          result(0, 0) = Scalar(condition_num);
        }
      }
    }

    return OperationReturn::SUCCESS;

  } catch (const std::exception &e) {
    // SVD failed, return infinite condition number
    if constexpr (std::is_same_v<Scalar, std::complex<float>> ||
                  std::is_same_v<Scalar, std::complex<double>>) {
      result(0, 0) = Scalar(std::numeric_limits<RealScalar>::infinity(), 0.0);
    } else {
      result(0, 0) = Scalar(std::numeric_limits<RealScalar>::infinity());
    }
    return OperationReturn::FAILURE;
  }
}

//==============================================================================
// Static Helper Methods
//==============================================================================

template <typename Scalar>
typename Eigen::NumTraits<Scalar>::Real
matrix_condition_number<Scalar>::extract_real_value(
    const Scalar &singular_value) {
  if constexpr (std::is_same_v<Scalar, std::complex<float>> ||
                std::is_same_v<Scalar, std::complex<double>>) {
    return std::abs(singular_value);
  } else {
    return std::abs(singular_value); // Also handles negative real values
  }
}

template <typename Scalar>
bool matrix_condition_number<Scalar>::is_effectively_zero(
    const typename Eigen::NumTraits<Scalar>::Real &singular_value,
    const typename Eigen::NumTraits<Scalar>::Real &tolerance) {
  return singular_value <= tolerance;
}

template <typename Scalar>
void matrix_condition_number<Scalar>::validate_shapes(
    const types::vector_shapes &input_shapes,
    const types::vector_shapes &output_shapes, const std::string &name) {
  if (input_shapes.size() != 1) {
    throw std::invalid_argument(name + ": requires exactly 1 input matrix");
  }
  if (output_shapes.size() != 1) {
    throw std::invalid_argument(name + ": requires exactly 1 output scalar");
  }

  const auto &matrix_shape = input_shapes[0];
  const auto &result_shape = output_shapes[0];

  // Check input matrix has valid dimensions
  if (matrix_shape.size() != 2 || matrix_shape[0] <= 0 ||
      matrix_shape[1] <= 0) {
    throw std::invalid_argument(name + ": input must be a valid 2D matrix");
  }

  // Check output is scalar (1×1)
  if (result_shape.size() != 2 || result_shape[0] != 1 ||
      result_shape[1] != 1) {
    throw std::invalid_argument(name + ": output must be a 1×1 scalar");
  }
}

template <typename Scalar>
std::vector<size_t> matrix_condition_number<Scalar>::compute_sizes(
    const types::vector_shapes &shapes) {
  std::vector<size_t> sizes;
  for (const auto &shape : shapes) {
    sizes.push_back(total_elements(shape));
  }
  return sizes;
}

template <typename Scalar>
types::vector_shapes matrix_condition_number<Scalar>::compute_output_shapes(
    const types::vector_shapes &input_shapes) {
  if (input_shapes.size() != 1) {
    return {};
  }

  // Output is always a 1×1 scalar regardless of input matrix size
  return {{1, 1}};
}

//==============================================================================
// Template Instantiations
//==============================================================================

// Base class constructors
template matrix_condition_number<float>::matrix_condition_number();
template matrix_condition_number<double>::matrix_condition_number();
template matrix_condition_number<
    std::complex<float>>::matrix_condition_number();
template matrix_condition_number<
    std::complex<double>>::matrix_condition_number();

// Base class operations
template OperationReturn matrix_condition_number<float>::operation(
    types::vector_const_matrix_map<float> &, types::vector_matrix_map<float> &);
template OperationReturn matrix_condition_number<double>::operation(
    types::vector_const_matrix_map<double> &,
    types::vector_matrix_map<double> &);
template OperationReturn
matrix_condition_number<std::complex<float>>::operation(
    types::vector_const_matrix_map<std::complex<float>> &,
    types::vector_matrix_map<std::complex<float>> &);
template OperationReturn
matrix_condition_number<std::complex<double>>::operation(
    types::vector_const_matrix_map<std::complex<double>> &,
    types::vector_matrix_map<std::complex<double>> &);

// Static methods
template void
matrix_condition_number<float>::validate_shapes(const types::vector_shapes &,
                                                const types::vector_shapes &,
                                                const std::string &);
template void
matrix_condition_number<double>::validate_shapes(const types::vector_shapes &,
                                                 const types::vector_shapes &,
                                                 const std::string &);
template void matrix_condition_number<std::complex<float>>::validate_shapes(
    const types::vector_shapes &, const types::vector_shapes &,
    const std::string &);
template void matrix_condition_number<std::complex<double>>::validate_shapes(
    const types::vector_shapes &, const types::vector_shapes &,
    const std::string &);

template std::vector<size_t>
matrix_condition_number<float>::compute_sizes(const types::vector_shapes &);
template std::vector<size_t>
matrix_condition_number<double>::compute_sizes(const types::vector_shapes &);
template std::vector<size_t>
matrix_condition_number<std::complex<float>>::compute_sizes(
    const types::vector_shapes &);
template std::vector<size_t>
matrix_condition_number<std::complex<double>>::compute_sizes(
    const types::vector_shapes &);

template types::vector_shapes
matrix_condition_number<float>::compute_output_shapes(
    const types::vector_shapes &);
template types::vector_shapes
matrix_condition_number<double>::compute_output_shapes(
    const types::vector_shapes &);
template types::vector_shapes
matrix_condition_number<std::complex<float>>::compute_output_shapes(
    const types::vector_shapes &);
template types::vector_shapes
matrix_condition_number<std::complex<double>>::compute_output_shapes(
    const types::vector_shapes &);

template typename Eigen::NumTraits<float>::Real
matrix_condition_number<float>::extract_real_value(const float &);
template typename Eigen::NumTraits<double>::Real
matrix_condition_number<double>::extract_real_value(const double &);
template typename Eigen::NumTraits<std::complex<float>>::Real
matrix_condition_number<std::complex<float>>::extract_real_value(
    const std::complex<float> &);
template typename Eigen::NumTraits<std::complex<double>>::Real
matrix_condition_number<std::complex<double>>::extract_real_value(
    const std::complex<double> &);

template bool matrix_condition_number<float>::is_effectively_zero(
    const typename Eigen::NumTraits<float>::Real &,
    const typename Eigen::NumTraits<float>::Real &);
template bool matrix_condition_number<double>::is_effectively_zero(
    const typename Eigen::NumTraits<double>::Real &,
    const typename Eigen::NumTraits<double>::Real &);
template bool matrix_condition_number<std::complex<float>>::is_effectively_zero(
    const typename Eigen::NumTraits<std::complex<float>>::Real &,
    const typename Eigen::NumTraits<std::complex<float>>::Real &);
template bool
matrix_condition_number<std::complex<double>>::is_effectively_zero(
    const typename Eigen::NumTraits<std::complex<double>>::Real &,
    const typename Eigen::NumTraits<std::complex<double>>::Real &);

// Sync factory methods
template typename matrix_condition_number_sync<float>::sptr
matrix_condition_number_sync<float>::make(const types::shape &);
template typename matrix_condition_number_sync<double>::sptr
matrix_condition_number_sync<double>::make(const types::shape &);
template typename matrix_condition_number_sync<std::complex<float>>::sptr
matrix_condition_number_sync<std::complex<float>>::make(const types::shape &);
template typename matrix_condition_number_sync<std::complex<double>>::sptr
matrix_condition_number_sync<std::complex<double>>::make(const types::shape &);

// PDU factory methods
template typename matrix_condition_number_pdu<float>::sptr
matrix_condition_number_pdu<float>::make();
template typename matrix_condition_number_pdu<double>::sptr
matrix_condition_number_pdu<double>::make();
template typename matrix_condition_number_pdu<std::complex<float>>::sptr
matrix_condition_number_pdu<std::complex<float>>::make();
template typename matrix_condition_number_pdu<std::complex<double>>::sptr
matrix_condition_number_pdu<std::complex<double>>::make();

// Sync constructors
template matrix_condition_number_sync<float>::matrix_condition_number_sync();
template matrix_condition_number_sync<double>::matrix_condition_number_sync();
template matrix_condition_number_sync<
    std::complex<float>>::matrix_condition_number_sync();
template matrix_condition_number_sync<
    std::complex<double>>::matrix_condition_number_sync();

template matrix_condition_number_sync<float>::matrix_condition_number_sync(
    const types::shape &);
template matrix_condition_number_sync<double>::matrix_condition_number_sync(
    const types::shape &);
template matrix_condition_number_sync<
    std::complex<float>>::matrix_condition_number_sync(const types::shape &);
template matrix_condition_number_sync<
    std::complex<double>>::matrix_condition_number_sync(const types::shape &);

// PDU constructors
template matrix_condition_number_pdu<float>::matrix_condition_number_pdu();
template matrix_condition_number_pdu<double>::matrix_condition_number_pdu();
template matrix_condition_number_pdu<
    std::complex<float>>::matrix_condition_number_pdu();
template matrix_condition_number_pdu<
    std::complex<double>>::matrix_condition_number_pdu();

// Implementation classes
template class matrix_condition_number_sync_impl<float>;
template class matrix_condition_number_sync_impl<double>;
template class matrix_condition_number_sync_impl<std::complex<float>>;
template class matrix_condition_number_sync_impl<std::complex<double>>;

template class matrix_condition_number_pdu_impl<float>;
template class matrix_condition_number_pdu_impl<double>;
template class matrix_condition_number_pdu_impl<std::complex<float>>;
template class matrix_condition_number_pdu_impl<std::complex<double>>;

} // namespace linalg
} // namespace gr