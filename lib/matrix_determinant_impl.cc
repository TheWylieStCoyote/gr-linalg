/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/* Refactored matrix_determinant sync implementation */

#include "matrix_determinant_impl.h"
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <pmt/pmt.h>
#include <stdexcept>
#include <vector>

namespace gr {
namespace linalg {

static inline types::vector_shapes _single(const types::shape &s) {
  return types::vector_shapes{s};
}

// Factory

template <typename Scalar>
typename matrix_determinant_sync<Scalar>::sptr
matrix_determinant_sync<Scalar>::make(const types::shape &shape) {
  return gnuradio::make_block_sptr<matrix_determinant_sync_impl<Scalar>>(shape);
}

template <typename Scalar>
matrix_determinant_sync<Scalar>::matrix_determinant_sync(
    const types::shape &shape)
    : matrix_determinant<Scalar>(),
      linalg_base_sync<Scalar>("matrix_determinant", _single(shape),
                               _single(types::shape{1, 1})) {}

/*
 * Impl ctor initializes virtual base linalg_base and validates square shape
 */
template <typename Scalar>
matrix_determinant_sync_impl<Scalar>::matrix_determinant_sync_impl(
    const typename matrix_determinant_sync_impl<Scalar>::shape &shape)
    : linalg_base<Scalar>(
          "matrix_determinant", _single(shape), std::vector<std::string>(),
          _single(types::shape{1, 1}), std::vector<std::string>(),
          array_broadcast_type::CUSTOM, error_tag_t::NONE, error_pdu_p::NONE),
      matrix_determinant_sync<Scalar>(shape) {
  if (shape[0] != shape[1])
    throw std::invalid_argument("Shape must be square");
}

/*
 * operation: out = det(in)
 */
template <typename Scalar>
OperationReturn matrix_determinant_sync_impl<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {
  if (input_matrices.size() != 1 || output_matrices.size() != 1)
    return OperationReturn::INVALID_SHAPE;
  auto &in = *input_matrices[0];
  auto &out = *output_matrices[0];
  if (in.rows() != in.cols())
    return OperationReturn::INVALID_SHAPE;
  out(0, 0) = in.determinant();
  return OperationReturn::SUCCESS;
}

// PDU factory and ctor

template <typename Scalar>
typename matrix_determinant_pdu<Scalar>::sptr
matrix_determinant_pdu<Scalar>::make(const types::shape &shape) {
  return gnuradio::make_block_sptr<matrix_determinant_pdu_impl<Scalar>>(shape);
}

template <typename Scalar>
matrix_determinant_pdu_impl<Scalar>::matrix_determinant_pdu_impl(
    const shape &shape)
    : linalg_base<Scalar>(
          "matrix_determinant_pdu", _single(shape), std::vector<std::string>{},
          _single(types::shape{1, 1}), std::vector<std::string>{},
          array_broadcast_type::CUSTOM, error_tag_t::NONE, error_pdu_p::NONE),
      linalg_base_pdu<Scalar>(
          "matrix_determinant_pdu", _single(shape), std::vector<std::string>{},
          _single(types::shape{1, 1}), std::vector<std::string>{},
          array_broadcast_type::CUSTOM, error_tag_t::NONE, error_pdu_p::NONE,
          PDU_UPDATE::ANY_INPUT, MESSAGE_HANDLER_MODE::DEFAULT) {
  if (shape.size() != 2 || shape[0] != shape[1]) {
    throw std::invalid_argument("Shape must be square");
  }
}

template <typename Scalar>
OperationReturn matrix_determinant_pdu_impl<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {
  if (input_matrices.size() != 1 || output_matrices.size() != 1)
    return OperationReturn::INVALID_SHAPE;
  auto &in = *input_matrices[0];
  auto &out = *output_matrices[0];
  if (in.rows() != in.cols())
    return OperationReturn::INVALID_SHAPE;
  out(0, 0) = in.determinant();
  return OperationReturn::SUCCESS;
}

// Explicit instantiations

template class matrix_determinant_sync_impl<float>;

template class matrix_determinant_sync_impl<double>;

template matrix_determinant_sync<float>::matrix_determinant_sync(
    const types::shape &);

template matrix_determinant_sync<double>::matrix_determinant_sync(
    const types::shape &);

template typename matrix_determinant_sync<float>::sptr
matrix_determinant_sync<float>::make(const types::shape &);

template typename matrix_determinant_sync<double>::sptr
matrix_determinant_sync<double>::make(const types::shape &);

// Explicit instantiations for PDU path

template matrix_determinant_pdu<float>::sptr
matrix_determinant_pdu<float>::make(const types::shape &);

template matrix_determinant_pdu<double>::sptr
matrix_determinant_pdu<double>::make(const types::shape &);

template class matrix_determinant_pdu_impl<float>;

template class matrix_determinant_pdu_impl<double>;

} /* namespace linalg */
} /* namespace gr */
