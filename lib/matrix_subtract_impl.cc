/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "matrix_subtract_impl.h"
#include <gnuradio/io_signature.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/types.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace gr {
namespace linalg {

static inline types::vector_shapes _repeat_shape(const types::shape &s, int n) {
  types::vector_shapes v;
  v.reserve(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i)
    v.push_back(s);
  return v;
}
static inline types::vector_shapes _single_shape(const types::shape &s) {
  return types::vector_shapes{s};
}

// Factory definition

template <typename Scalar>
typename matrix_subtract_sync<Scalar>::sptr
matrix_subtract_sync<Scalar>::make(const types::shape &shape, int num_inputs) {
  return gnuradio::make_block_sptr<matrix_subtract_sync_impl<Scalar>>(
      shape, num_inputs);
}

// matrix_subtract_sync public ctor used by factory

template <typename Scalar>
matrix_subtract_sync<Scalar>::matrix_subtract_sync(const types::shape &shape,
                                                   int num_inputs)
    : matrix_subtract<Scalar>() {}

// Impl ctor: most-derived, initializes the virtual bases linalg_base and
// linalg_base_sync

template <typename Scalar>
matrix_subtract_sync_impl<Scalar>::matrix_subtract_sync_impl(
    const typename matrix_subtract_sync_impl<Scalar>::shape &shape,
    int num_inputs)
    : linalg_base<Scalar>("matrix_subtract", _repeat_shape(shape, num_inputs),
                          std::vector<std::string>(), _single_shape(shape),
                          std::vector<std::string>(),
                          array_broadcast_type::CUSTOM, error_tag_t::NONE,
                          error_pdu_p::NONE),
      linalg_base_sync<Scalar>(
          "matrix_subtract", _repeat_shape(shape, num_inputs),
          std::vector<std::string>(), _single_shape(shape),
          std::vector<std::string>(), array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE, gr::block::TPP_ALL_TO_ALL),
      matrix_subtract_sync<Scalar>(shape, num_inputs) {}

// Base operation implementation: out = in0 - in1 - ...

template <typename Scalar>
OperationReturn matrix_subtract<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {
  if (input_matrices.size() < 2 || output_matrices.empty())
    return OperationReturn::INVALID_SHAPE;

  auto &out = *output_matrices[0];
  out = *input_matrices[0];
  for (size_t i = 1; i < input_matrices.size(); ++i) {
    out -= *input_matrices[i];
  }
  return OperationReturn::SUCCESS;
}

// Sync impl operation forwards to base semantics (could be customized)

template <typename Scalar>
OperationReturn matrix_subtract_sync_impl<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {
  return matrix_subtract<Scalar>::operation(input_matrices, output_matrices);
}

// Explicit instantiations

template class matrix_subtract_sync_impl<float>;

template class matrix_subtract_sync_impl<double>;

template typename matrix_subtract_sync<float>::sptr
matrix_subtract_sync<float>::make(const types::shape &, int);

template typename matrix_subtract_sync<double>::sptr
matrix_subtract_sync<double>::make(const types::shape &, int);

template matrix_subtract_sync<float>::matrix_subtract_sync(const types::shape &,
                                                           int);

template matrix_subtract_sync<double>::matrix_subtract_sync(
    const types::shape &, int);

// Explicit instantiation for base operation

template OperationReturn
matrix_subtract<float>::operation(types::vector_const_matrix_map<float> &,
                                  types::vector_matrix_map<float> &);

template OperationReturn
matrix_subtract<double>::operation(types::vector_const_matrix_map<double> &,
                                   types::vector_matrix_map<double> &);

} // namespace linalg
} // namespace gr
