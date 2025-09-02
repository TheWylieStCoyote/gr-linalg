/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_MATRIX_SUBTRACT_IMPL_H
#define INCLUDED_LINALG_MATRIX_SUBTRACT_IMPL_H

#include <gnuradio/linalg/matrix_subtract.h>
#include <gnuradio/linalg/types.h>

namespace gr {
namespace linalg {

// Sync implementation for matrix_subtract
// Most-derived class initializes virtual bases and implements operation

template <typename Scalar>
class matrix_subtract_sync_impl : public matrix_subtract_sync<Scalar> {
public:
  using shape = types::shape;

  matrix_subtract_sync_impl(const shape &shape, int num_inputs);
  ~matrix_subtract_sync_impl() override = default;

  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;
};

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_SUBTRACT_IMPL_H */
