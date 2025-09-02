/* -*- c++ -*- */
/*
 * Copyright 2024 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "vector_outer_product_impl.h"
#include <algorithm>
#include <gnuradio/gr_complex.h>
#include <gnuradio/io_signature.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/types.h>
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

static inline types::vector_shapes _dual_input_shapes(const types::shape &s1,
                                                      const types::shape &s2) {
  return {s1, s2};
}

static inline types::vector_shapes _single_output_shape(const types::shape &s) {
  return {s};
}

static inline std::vector<std::string> _dual_input_names() {
  return {"vector_a", "vector_b"};
}

static inline std::vector<std::string> _single_output_names() {
  return {"result"};
}

//==============================================================================
// Factory Methods
//==============================================================================

template <typename Scalar>
typename vector_outer_product_sync<Scalar>::sptr
vector_outer_product_sync<Scalar>::make() {
  return gnuradio::make_block_sptr<vector_outer_product_sync_impl<Scalar>>();
}

template <typename Scalar>
typename vector_outer_product_pdu<Scalar>::sptr
vector_outer_product_pdu<Scalar>::make() {
  return gnuradio::make_block_sptr<vector_outer_product_pdu_impl<Scalar>>();
}

//==============================================================================
// Base Class Constructors
//==============================================================================

template <typename Scalar>
vector_outer_product<Scalar>::vector_outer_product() : linalg_base<Scalar>() {}

//==============================================================================
// Sync Class Constructors
//==============================================================================

template <typename Scalar>
vector_outer_product_sync<Scalar>::vector_outer_product_sync()
    : vector_outer_product<Scalar>() {}

template <typename Scalar>
vector_outer_product_sync_impl<Scalar>::vector_outer_product_sync_impl()
    : linalg_base<Scalar>(
          "vector_outer_product_sync",
          _dual_input_shapes({1},
                             {1}), // Flexible vector shapes - any size accepted
          _dual_input_names(),
          _single_output_shape(
              {1}), // Flexible output shape - computed dynamically
          _single_output_names(),
          array_broadcast_type::CUSTOM, // Skip strict validation
          error_tag_t::NONE, error_pdu_p::NONE),
      linalg_base_sync<Scalar>(
          "vector_outer_product_sync",
          _dual_input_shapes({1}, {1}), // Flexible vector shapes
          _dual_input_names(),
          _single_output_shape({1}), // Flexible output shape
          _single_output_names(),
          array_broadcast_type::CUSTOM, // Skip strict validation
          error_tag_t::NONE, error_pdu_p::NONE, gr::block::TPP_ALL_TO_ALL),
      vector_outer_product<Scalar>(), vector_outer_product_sync<Scalar>() {}

//==============================================================================
// PDU Class Constructors
//==============================================================================

template <typename Scalar>
vector_outer_product_pdu<Scalar>::vector_outer_product_pdu()
    : linalg_base<Scalar>(
          "vector_outer_product_pdu",
          _dual_input_shapes({1}, {1}), // Flexible vector shapes
          _dual_input_names(),
          _single_output_shape({1}), // Flexible output shape
          _single_output_names(),
          array_broadcast_type::CUSTOM, // Skip strict validation
          error_tag_t::NONE, error_pdu_p::NONE),
      linalg_base_pdu<Scalar>(
          "vector_outer_product_pdu",
          _dual_input_shapes({1}, {1}), // Flexible vector shapes
          _dual_input_names(),
          _single_output_shape({1}), // Flexible output shape
          _single_output_names(),
          array_broadcast_type::CUSTOM, // Skip strict validation
          error_tag_t::NONE, error_pdu_p::NONE, PDU_UPDATE::ANY_INPUT,
          MESSAGE_HANDLER_MODE::DEFAULT),
      vector_outer_product<Scalar>() {}

template <typename Scalar>
vector_outer_product_pdu_impl<Scalar>::vector_outer_product_pdu_impl()
    : linalg_base<Scalar>(
          "vector_outer_product_pdu",
          _dual_input_shapes({1}, {1}), // Flexible vector shapes
          _dual_input_names(),
          _single_output_shape({1}), // Flexible output shape
          _single_output_names(),
          array_broadcast_type::CUSTOM, // Skip strict validation
          error_tag_t::NONE, error_pdu_p::NONE),
      linalg_base_pdu<Scalar>(
          "vector_outer_product_pdu",
          _dual_input_shapes({1}, {1}), // Flexible vector shapes
          _dual_input_names(),
          _single_output_shape({1}), // Flexible output shape
          _single_output_names(),
          array_broadcast_type::CUSTOM, // Skip strict validation
          error_tag_t::NONE, error_pdu_p::NONE, PDU_UPDATE::ANY_INPUT,
          MESSAGE_HANDLER_MODE::DEFAULT),
      vector_outer_product<Scalar>(), vector_outer_product_pdu<Scalar>() {}

//==============================================================================
// Operation Implementation
//==============================================================================

template <typename Scalar>
OperationReturn vector_outer_product<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {
  // Validate inputs
  if (input_matrices.size() != 2) {
    return OperationReturn::FAILURE;
  }
  if (output_matrices.size() != 1) {
    return OperationReturn::FAILURE;
  }

  const auto &vector_a = *input_matrices[0];
  const auto &vector_b = *input_matrices[1];
  auto &result = *output_matrices[0];

  // Validate that inputs are vectors (1D arrays, possibly as 1×n or n×1
  // matrices)
  size_t len_a = get_vector_length(
      {static_cast<int>(vector_a.rows()), static_cast<int>(vector_a.cols())});
  size_t len_b = get_vector_length(
      {static_cast<int>(vector_b.rows()), static_cast<int>(vector_b.cols())});

  if (len_a == 0 || len_b == 0) {
    return OperationReturn::INVALID_SHAPE;
  }

  // Check output dimensions match expected m×n
  if (static_cast<size_t>(result.rows()) != len_a ||
      static_cast<size_t>(result.cols()) != len_b) {
    return OperationReturn::INVALID_SHAPE;
  }

  return compute_outer_product(vector_a, vector_b, result);
}

template <typename Scalar>
OperationReturn vector_outer_product<Scalar>::compute_outer_product(
    const Eigen::Map<
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &vector_a,
    const Eigen::Map<
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &vector_b,
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &result) {
  const size_t len_a = get_vector_length(
      {static_cast<int>(vector_a.rows()), static_cast<int>(vector_a.cols())});
  const size_t len_b = get_vector_length(
      {static_cast<int>(vector_b.rows()), static_cast<int>(vector_b.cols())});

  // Convert matrices to vectors for easier access
  Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> vec_a(
      vector_a.data(), static_cast<Eigen::Index>(len_a));
  Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> vec_b(
      vector_b.data(), static_cast<Eigen::Index>(len_b));

  // Compute outer product: result(i,j) = vec_a(i) * vec_b(j)
  for (size_t i = 0; i < len_a; ++i) {
    for (size_t j = 0; j < len_b; ++j) {
      result(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
          vec_a(static_cast<Eigen::Index>(i)) *
          vec_b(static_cast<Eigen::Index>(j));
    }
  }

  return OperationReturn::SUCCESS;
}

//==============================================================================
// Static Helper Methods
//==============================================================================

template <typename Scalar>
bool vector_outer_product<Scalar>::is_vector_shape(const types::shape &shape) {
  if (shape.size() != 2)
    return false;

  // Valid vector shapes: 1×n, n×1, or effectively 1D
  return (shape[0] == 1) || (shape[1] == 1) || (shape[0] > 0 && shape[1] > 0);
}

template <typename Scalar>
size_t
vector_outer_product<Scalar>::get_vector_length(const types::shape &shape) {
  if (shape.size() != 2)
    return 0;

  // For 1×n or n×1, return the non-unity dimension
  if (shape[0] == 1)
    return static_cast<size_t>(shape[1]);
  if (shape[1] == 1)
    return static_cast<size_t>(shape[0]);

  // For general m×n matrix, treat as m*n element vector (flatten)
  return static_cast<size_t>(shape[0] * shape[1]);
}

template <typename Scalar>
void vector_outer_product<Scalar>::validate_shapes(
    const types::vector_shapes &input_shapes,
    const types::vector_shapes &output_shapes, const std::string &name) {
  if (input_shapes.size() != 2) {
    throw std::invalid_argument(name + ": requires exactly 2 input vectors");
  }
  if (output_shapes.size() != 1) {
    throw std::invalid_argument(name + ": requires exactly 1 output matrix");
  }

  const auto &shape_a = input_shapes[0];
  const auto &shape_b = input_shapes[1];
  const auto &result_shape = output_shapes[0];

  // Validate that inputs can be treated as vectors
  if (!is_vector_shape(shape_a)) {
    throw std::invalid_argument(name + ": first input must be a vector");
  }
  if (!is_vector_shape(shape_b)) {
    throw std::invalid_argument(name + ": second input must be a vector");
  }

  size_t len_a = get_vector_length(shape_a);
  size_t len_b = get_vector_length(shape_b);

  // Output should be len_a × len_b matrix
  if (static_cast<size_t>(result_shape[0]) != len_a ||
      static_cast<size_t>(result_shape[1]) != len_b) {
    throw std::invalid_argument(name + ": output shape mismatch (expected " +
                                std::to_string(len_a) + "×" +
                                std::to_string(len_b) + ", got " +
                                std::to_string(result_shape[0]) + "×" +
                                std::to_string(result_shape[1]) + ")");
  }
}

template <typename Scalar>
std::vector<size_t> vector_outer_product<Scalar>::compute_sizes(
    const types::vector_shapes &shapes) {
  std::vector<size_t> sizes;
  for (const auto &shape : shapes) {
    sizes.push_back(total_elements(shape));
  }
  return sizes;
}

template <typename Scalar>
types::vector_shapes vector_outer_product<Scalar>::compute_output_shapes(
    const types::vector_shapes &input_shapes) {
  if (input_shapes.size() != 2) {
    return {};
  }

  size_t len_a = get_vector_length(input_shapes[0]);
  size_t len_b = get_vector_length(input_shapes[1]);

  // Output is len_a × len_b matrix
  return {{static_cast<int>(len_a), static_cast<int>(len_b)}};
}

//==============================================================================
// Template Instantiations
//==============================================================================

// Base class constructors
template vector_outer_product<float>::vector_outer_product();
template vector_outer_product<double>::vector_outer_product();
template vector_outer_product<std::complex<float>>::vector_outer_product();
template vector_outer_product<std::complex<double>>::vector_outer_product();

// Base class operations
template OperationReturn
vector_outer_product<float>::operation(types::vector_const_matrix_map<float> &,
                                       types::vector_matrix_map<float> &);
template OperationReturn vector_outer_product<double>::operation(
    types::vector_const_matrix_map<double> &,
    types::vector_matrix_map<double> &);
template OperationReturn vector_outer_product<std::complex<float>>::operation(
    types::vector_const_matrix_map<std::complex<float>> &,
    types::vector_matrix_map<std::complex<float>> &);
template OperationReturn vector_outer_product<std::complex<double>>::operation(
    types::vector_const_matrix_map<std::complex<double>> &,
    types::vector_matrix_map<std::complex<double>> &);

// Static methods
template void
vector_outer_product<float>::validate_shapes(const types::vector_shapes &,
                                             const types::vector_shapes &,
                                             const std::string &);
template void
vector_outer_product<double>::validate_shapes(const types::vector_shapes &,
                                              const types::vector_shapes &,
                                              const std::string &);
template void vector_outer_product<std::complex<float>>::validate_shapes(
    const types::vector_shapes &, const types::vector_shapes &,
    const std::string &);
template void vector_outer_product<std::complex<double>>::validate_shapes(
    const types::vector_shapes &, const types::vector_shapes &,
    const std::string &);

template std::vector<size_t>
vector_outer_product<float>::compute_sizes(const types::vector_shapes &);
template std::vector<size_t>
vector_outer_product<double>::compute_sizes(const types::vector_shapes &);
template std::vector<size_t>
vector_outer_product<std::complex<float>>::compute_sizes(
    const types::vector_shapes &);
template std::vector<size_t>
vector_outer_product<std::complex<double>>::compute_sizes(
    const types::vector_shapes &);

template types::vector_shapes
vector_outer_product<float>::compute_output_shapes(
    const types::vector_shapes &);
template types::vector_shapes
vector_outer_product<double>::compute_output_shapes(
    const types::vector_shapes &);
template types::vector_shapes
vector_outer_product<std::complex<float>>::compute_output_shapes(
    const types::vector_shapes &);
template types::vector_shapes
vector_outer_product<std::complex<double>>::compute_output_shapes(
    const types::vector_shapes &);

template bool
vector_outer_product<float>::is_vector_shape(const types::shape &);
template bool
vector_outer_product<double>::is_vector_shape(const types::shape &);
template bool vector_outer_product<std::complex<float>>::is_vector_shape(
    const types::shape &);
template bool vector_outer_product<std::complex<double>>::is_vector_shape(
    const types::shape &);

template size_t
vector_outer_product<float>::get_vector_length(const types::shape &);
template size_t
vector_outer_product<double>::get_vector_length(const types::shape &);
template size_t vector_outer_product<std::complex<float>>::get_vector_length(
    const types::shape &);
template size_t vector_outer_product<std::complex<double>>::get_vector_length(
    const types::shape &);

// Sync factory methods
template typename vector_outer_product_sync<float>::sptr
vector_outer_product_sync<float>::make();
template typename vector_outer_product_sync<double>::sptr
vector_outer_product_sync<double>::make();
template typename vector_outer_product_sync<std::complex<float>>::sptr
vector_outer_product_sync<std::complex<float>>::make();
template typename vector_outer_product_sync<std::complex<double>>::sptr
vector_outer_product_sync<std::complex<double>>::make();

// PDU factory methods
template typename vector_outer_product_pdu<float>::sptr
vector_outer_product_pdu<float>::make();
template typename vector_outer_product_pdu<double>::sptr
vector_outer_product_pdu<double>::make();
template typename vector_outer_product_pdu<std::complex<float>>::sptr
vector_outer_product_pdu<std::complex<float>>::make();
template typename vector_outer_product_pdu<std::complex<double>>::sptr
vector_outer_product_pdu<std::complex<double>>::make();

// Sync constructors
template vector_outer_product_sync<float>::vector_outer_product_sync();
template vector_outer_product_sync<double>::vector_outer_product_sync();
template vector_outer_product_sync<
    std::complex<float>>::vector_outer_product_sync();
template vector_outer_product_sync<
    std::complex<double>>::vector_outer_product_sync();

// PDU constructors
template vector_outer_product_pdu<float>::vector_outer_product_pdu();
template vector_outer_product_pdu<double>::vector_outer_product_pdu();
template vector_outer_product_pdu<
    std::complex<float>>::vector_outer_product_pdu();
template vector_outer_product_pdu<
    std::complex<double>>::vector_outer_product_pdu();

// Implementation classes
template class vector_outer_product_sync_impl<float>;
template class vector_outer_product_sync_impl<double>;
template class vector_outer_product_sync_impl<std::complex<float>>;
template class vector_outer_product_sync_impl<std::complex<double>>;

template class vector_outer_product_pdu_impl<float>;
template class vector_outer_product_pdu_impl<double>;
template class vector_outer_product_pdu_impl<std::complex<float>>;
template class vector_outer_product_pdu_impl<std::complex<double>>;

} // namespace linalg
} // namespace gr