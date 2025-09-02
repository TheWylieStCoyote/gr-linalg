/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_MATRIX_MULTIPLY_IMPL_H
#define INCLUDED_LINALG_MATRIX_MULTIPLY_IMPL_H

#include "gnuradio/linalg/linalg_base.h"
#include <gnuradio/linalg/api.h>
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/matrix_multiply.h>
#include <gnuradio/linalg/types.h>
#include <vector>

namespace gr {
namespace linalg {

template <typename Scalar>
class LINALG_API matrix_multiply_sync_impl
    : virtual public matrix_multiply<Scalar>,
      virtual public linalg_base_sync<Scalar>,
      virtual public matrix_multiply_sync<Scalar> {
public:
  matrix_multiply_sync_impl(const types::shape &shape_0,
                            const types::shape &shape_1);
  ~matrix_multiply_sync_impl() override = default;

  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;
};

template <typename Scalar>
class LINALG_API matrix_multiply_pdu_impl
    : virtual public matrix_multiply<Scalar>,
      virtual public linalg_base_pdu<Scalar>,
      virtual public matrix_multiply_pdu<Scalar> {
public:
  matrix_multiply_pdu_impl(const types::shape &shape_0,
                           const types::shape &shape_1);
  ~matrix_multiply_pdu_impl() override = default;
};

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_MULTIPLY_IMPL_H */
