/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "matrix_transpose_impl.h"
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/types.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gr {
namespace linalg {

static inline types::vector_shapes _single_shape(const types::shape &s) {
  types::vector_shapes v;
  v.push_back(s);
  return v;
}

static inline types::shape _transpose_shape(types::shape s) {
  if (s.size() >= 2)
    std::swap(s[0], s[1]);
  return s;
}

// Factory

template <typename Scalar>
typename matrix_transpose_sync<Scalar>::sptr
matrix_transpose_sync<Scalar>::make(const types::shape &shape) {
  return gnuradio::make_block_sptr<matrix_transpose_sync_impl<Scalar>>(shape);
}

// Public sync ctor used by factory

template <typename Scalar>
matrix_transpose_sync<Scalar>::matrix_transpose_sync(const types::shape &shape)
    : matrix_transpose<Scalar>(),
      linalg_base_sync<Scalar>("matrix_transpose", _single_shape(shape),
                               _single_shape(_transpose_shape(shape))) {}

// Impl ctor: most-derived, initializes virtual base

template <typename Scalar>
matrix_transpose_sync_impl<Scalar>::matrix_transpose_sync_impl(
    const typename matrix_transpose_sync_impl<Scalar>::shape &shape)
    : linalg_base<Scalar>(
          "matrix_transpose", _single_shape(shape), std::vector<std::string>(),
          _single_shape(_transpose_shape(shape)), std::vector<std::string>(),
          array_broadcast_type::CUSTOM, error_tag_t::NONE, error_pdu_p::NONE),
      matrix_transpose_sync<Scalar>(shape) {}

// operation: out = in^T

template <typename Scalar>
OperationReturn matrix_transpose_sync_impl<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {
  if (input_matrices.size() != 1 || output_matrices.size() != 1)
    return OperationReturn::INVALID_SHAPE;
  (*output_matrices[0]) = input_matrices[0]->transpose();
  return OperationReturn::SUCCESS;
}

// Explicit instantiations

template class matrix_transpose_sync_impl<float>;

template class matrix_transpose_sync_impl<double>;

template matrix_transpose_sync<float>::matrix_transpose_sync(
    const types::shape &);

template matrix_transpose_sync<double>::matrix_transpose_sync(
    const types::shape &);

template typename matrix_transpose_sync<float>::sptr
matrix_transpose_sync<float>::make(const types::shape &);

template typename matrix_transpose_sync<double>::sptr
matrix_transpose_sync<double>::make(const types::shape &);

} // namespace linalg
} // namespace gr
