/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "matrix_inverse_impl.h"
#include <gnuradio/linalg/linalg_base_sync.h>
#include <stdexcept>
#include <vector>

namespace gr {
namespace linalg {

static inline types::vector_shapes _single(const types::shape &s) {
  return types::vector_shapes{s};
}

// Factory

template <typename Scalar>
typename matrix_inverse_sync<Scalar>::sptr
matrix_inverse_sync<Scalar>::make(const types::shape &shape) {
  return gnuradio::make_block_sptr<matrix_inverse_sync_impl<Scalar>>(shape);
}

template <typename Scalar>
matrix_inverse_sync<Scalar>::matrix_inverse_sync(const types::shape &shape)
    : matrix_inverse<Scalar>(),
      linalg_base_sync<Scalar>("matrix_inverse", _single(shape),
                               _single(shape)) {}

// Impl ctor initializes virtual base linalg_base

template <typename Scalar>
matrix_inverse_sync_impl<Scalar>::matrix_inverse_sync_impl(
    const typename matrix_inverse_sync_impl<Scalar>::shape &shape)
    : linalg_base<Scalar>(
          "matrix_inverse", _single(shape), std::vector<std::string>(),
          _single(shape), std::vector<std::string>(),
          array_broadcast_type::NONE, error_tag_t::NONE, error_pdu_p::NONE),
      matrix_inverse_sync<Scalar>(shape) {}

// operation: out = inverse(in)

template <typename Scalar>
OperationReturn matrix_inverse_sync_impl<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {
  if (input_matrices.size() != 1 || output_matrices.size() != 1)
    return OperationReturn::INVALID_SHAPE;
  auto &in = *input_matrices[0];
  auto &out = *output_matrices[0];
  if (in.rows() != in.cols())
    return OperationReturn::INVALID_SHAPE;
  out = in.inverse();
  return OperationReturn::SUCCESS;
}

// Explicit instantiations

template class matrix_inverse_sync_impl<float>;
template class matrix_inverse_sync_impl<double>;
template class matrix_inverse_sync_impl<std::complex<float>>;
template class matrix_inverse_sync_impl<std::complex<double>>;

template matrix_inverse_sync<float>::matrix_inverse_sync(const types::shape &);
template matrix_inverse_sync<double>::matrix_inverse_sync(const types::shape &);
template matrix_inverse_sync<std::complex<float>>::matrix_inverse_sync(
    const types::shape &);
template matrix_inverse_sync<std::complex<double>>::matrix_inverse_sync(
    const types::shape &);

template typename matrix_inverse_sync<float>::sptr
matrix_inverse_sync<float>::make(const types::shape &);

template typename matrix_inverse_sync<double>::sptr
matrix_inverse_sync<double>::make(const types::shape &);

template typename matrix_inverse_sync<std::complex<float>>::sptr
matrix_inverse_sync<std::complex<float>>::make(const types::shape &);

template typename matrix_inverse_sync<std::complex<double>>::sptr
matrix_inverse_sync<std::complex<double>>::make(const types::shape &);

} /* namespace linalg */
} /* namespace gr */
