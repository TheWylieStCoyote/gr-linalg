/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_MATRIX_RESHAPE_IMPL_H
#define INCLUDED_LINALG_MATRIX_RESHAPE_IMPL_H

#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/matrix_reshape.h>
#include <gnuradio/linalg/types.h>

namespace gr {
namespace linalg {

template <typename Scalar>
class matrix_reshape_sync_impl : public matrix_reshape_sync<Scalar> {
public:
  matrix_reshape_sync_impl(const types::shape &input_shape,
                           const types::shape &output_shape);
  ~matrix_reshape_sync_impl();
};

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_RESHAPE_IMPL_H */