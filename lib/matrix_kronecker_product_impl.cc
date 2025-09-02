/* -*- c++ -*- */
/*
 * Copyright 2025 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "matrix_kronecker_product_impl.h"
#include <Eigen/Dense>
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/matrix_kronecker_product.h>
#include <gnuradio/linalg/performance_profiler.h>
#include <unsupported/Eigen/KroneckerProduct>

namespace gr {
namespace linalg {

static inline types::vector_shapes _dual(const types::vector_shapes &shapes) {
  return shapes;
}

// ===== SYNC IMPLEMENTATION =====

// Factory method
template <typename Scalar>
typename matrix_kronecker_product_sync<Scalar>::sptr
matrix_kronecker_product_sync<Scalar>::make(const types::shape &shape_a,
                                            const types::shape &shape_b) {
  return gnuradio::make_block_sptr<matrix_kronecker_product_sync_impl<Scalar>>(
      shape_a, shape_b);
}

// Constructor for sync interface
template <typename Scalar>
matrix_kronecker_product_sync<Scalar>::matrix_kronecker_product_sync(
    const types::shape &shape_a, const types::shape &shape_b)
    : matrix_kronecker_product<Scalar>(),
      linalg_base_sync<Scalar>(
          "matrix_kronecker_product", types::vector_shapes{shape_a, shape_b},
          types::vector_shapes{
              {shape_a[0] * shape_b[0], shape_a[1] * shape_b[1]}}) {}

// Implementation constructor
template <typename Scalar>
matrix_kronecker_product_sync_impl<Scalar>::matrix_kronecker_product_sync_impl(
    const types::shape &shape_a, const types::shape &shape_b)
    : linalg_base<Scalar>(
          "matrix_kronecker_product", types::vector_shapes{shape_a, shape_b},
          std::vector<std::string>(),
          types::vector_shapes{
              {shape_a[0] * shape_b[0], shape_a[1] * shape_b[1]}},
          std::vector<std::string>(), array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE),
      matrix_kronecker_product_sync<Scalar>(shape_a, shape_b) {
  // Input validation: check that both shapes are 2D with positive dimensions
  if (shape_a.size() != 2 || shape_b.size() != 2) {
    throw std::invalid_argument(
        "matrix_kronecker_product requires 2D matrices");
  }
  if (shape_a[0] <= 0 || shape_a[1] <= 0 || shape_b[0] <= 0 ||
      shape_b[1] <= 0) {
    throw std::invalid_argument(
        "matrix_kronecker_product requires positive dimensions");
  }
}

// Main operation implementation
template <typename Scalar>
OperationReturn matrix_kronecker_product_sync_impl<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {

  // Validate inputs
  if (input_matrices.size() != 2 || output_matrices.size() != 1) {
    return OperationReturn::INVALID_SHAPE;
  }

  const auto &input_A = *input_matrices[0];
  const auto &input_B = *input_matrices[1];
  auto &output_matrix = *output_matrices[0];

  // Performance profiling
  size_t total_ops = input_A.rows() * input_A.cols() * input_B.rows() *
                     input_B.cols() * sizeof(Scalar);
  PROFILE_LINALG_OPERATION("matrix_kronecker_product", total_ops);

  try {
    // Compute Kronecker product using Eigen
    output_matrix = Eigen::kroneckerProduct(input_A, input_B).eval();
    return OperationReturn::SUCCESS;
  } catch (const std::exception &e) {
    return OperationReturn::INVALID_SHAPE;
  }
}

// ===== PDU IMPLEMENTATION =====

// Factory method
template <typename Scalar>
typename matrix_kronecker_product_pdu<Scalar>::sptr
matrix_kronecker_product_pdu<Scalar>::make(const types::shape &shape_a,
                                           const types::shape &shape_b) {
  return gnuradio::make_block_sptr<matrix_kronecker_product_pdu_impl<Scalar>>(
      shape_a, shape_b);
}

// Constructor for PDU interface
template <typename Scalar>
matrix_kronecker_product_pdu<Scalar>::matrix_kronecker_product_pdu(
    const types::shape &shape_a, const types::shape &shape_b)
    : matrix_kronecker_product<Scalar>(),
      linalg_base_pdu<Scalar>(
          "matrix_kronecker_product", types::vector_shapes{shape_a, shape_b},
          std::vector<std::string>(),
          types::vector_shapes{
              {shape_a[0] * shape_b[0], shape_a[1] * shape_b[1]}},
          std::vector<std::string>(), array_broadcast_type::NONE,
          error_tag_t::NONE, error_pdu_p::NONE, PDU_UPDATE::DEFAULT,
          MESSAGE_HANDLER_MODE::DEFAULT) {}

// Implementation constructor
template <typename Scalar>
matrix_kronecker_product_pdu_impl<Scalar>::matrix_kronecker_product_pdu_impl(
    const types::shape &shape_a, const types::shape &shape_b)
    : linalg_base<Scalar>(
          "matrix_kronecker_product", types::vector_shapes{shape_a, shape_b},
          std::vector<std::string>(),
          types::vector_shapes{
              {shape_a[0] * shape_b[0], shape_a[1] * shape_b[1]}},
          std::vector<std::string>(), array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE),
      matrix_kronecker_product_pdu<Scalar>(shape_a, shape_b) {
  // Input validation: check that both shapes are 2D with positive dimensions
  if (shape_a.size() != 2 || shape_b.size() != 2) {
    throw std::invalid_argument(
        "matrix_kronecker_product requires 2D matrices");
  }
  if (shape_a[0] <= 0 || shape_a[1] <= 0 || shape_b[0] <= 0 ||
      shape_b[1] <= 0) {
    throw std::invalid_argument(
        "matrix_kronecker_product requires positive dimensions");
  }
}

// Main operation implementation
template <typename Scalar>
OperationReturn matrix_kronecker_product_pdu_impl<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {

  // Validate inputs
  if (input_matrices.size() != 2 || output_matrices.size() != 1) {
    return OperationReturn::INVALID_SHAPE;
  }

  const auto &input_A = *input_matrices[0];
  const auto &input_B = *input_matrices[1];
  auto &output_matrix = *output_matrices[0];

  // Performance profiling
  size_t total_ops = input_A.rows() * input_A.cols() * input_B.rows() *
                     input_B.cols() * sizeof(Scalar);
  PROFILE_LINALG_OPERATION("matrix_kronecker_product", total_ops);

  try {
    // Compute Kronecker product using Eigen
    output_matrix = Eigen::kroneckerProduct(input_A, input_B).eval();
    return OperationReturn::SUCCESS;
  } catch (const std::exception &e) {
    return OperationReturn::INVALID_SHAPE;
  }
}

// Explicit template instantiations for sync implementations
template class matrix_kronecker_product_sync<float>;
template class matrix_kronecker_product_sync<double>;
template class matrix_kronecker_product_sync<std::complex<float>>;
template class matrix_kronecker_product_sync<std::complex<double>>;

template class matrix_kronecker_product_sync_impl<float>;
template class matrix_kronecker_product_sync_impl<double>;
template class matrix_kronecker_product_sync_impl<std::complex<float>>;
template class matrix_kronecker_product_sync_impl<std::complex<double>>;

// Explicit template instantiations for PDU implementations
template class matrix_kronecker_product_pdu<float>;
template class matrix_kronecker_product_pdu<double>;
template class matrix_kronecker_product_pdu<std::complex<float>>;
template class matrix_kronecker_product_pdu<std::complex<double>>;

template class matrix_kronecker_product_pdu_impl<float>;
template class matrix_kronecker_product_pdu_impl<double>;
template class matrix_kronecker_product_pdu_impl<std::complex<float>>;
template class matrix_kronecker_product_pdu_impl<std::complex<double>>;

} // namespace linalg
} // namespace gr