/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_MATRIX_ADD_IMPL_H
#define INCLUDED_LINALG_MATRIX_ADD_IMPL_H

#include "gnuradio/linalg/linalg_base.h"
#include <gnuradio/linalg/matrix_add.h>
#include <gnuradio/linalg/types.h>
#include <vector>

namespace gr {
namespace linalg {

template <typename Scalar>
class matrix_add_sync_impl : virtual public linalg_base<Scalar>,
                             virtual public matrix_add<Scalar>,
                             virtual public linalg_base_sync<Scalar>,
                             virtual public matrix_add_sync<Scalar> {
public:
  matrix_add_sync_impl(const types::shape &shape, const int num_inputs);
  ~matrix_add_sync_impl() override = default;

  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;
};

template <typename Scalar>
class matrix_add_pdu_impl : virtual public linalg_base<Scalar>,
                            virtual public matrix_add<Scalar>,
                            virtual public linalg_base_pdu<Scalar>,
                            virtual public matrix_add_pdu<Scalar> {
public:
  matrix_add_pdu_impl(const types::shape &shape, const int num_inputs);
  ~matrix_add_pdu_impl() override = default;
};

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_ADD_IMPL_H */
