/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_MATRIX_ELEMENTWISE_DIVIDE_IMPL_H
#define INCLUDED_LINALG_MATRIX_ELEMENTWISE_DIVIDE_IMPL_H

#include <gnuradio/linalg/matrix_elementwise_divide.h>

namespace gr {
namespace linalg {

/*!
 * \brief Implementation class for matrix element-wise division sync block
 * \ingroup linalg
 */
template <typename Scalar>
class matrix_elementwise_divide_sync_impl
    : public matrix_elementwise_divide_sync<Scalar> {
private:
  // Nothing additional needed for sync implementation

public:
  matrix_elementwise_divide_sync_impl(const types::shape &shape,
                                      const int num_inputs);
  ~matrix_elementwise_divide_sync_impl() = default;

  // Override operation for specific implementation if needed
  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;

private:
  // Note: Division by zero handling utilities (is_zero, safe_divide)
  // are inherited from base class matrix_elementwise_divide<Scalar>
};

/*!
 * \brief Implementation class for matrix element-wise division PDU block
 * \ingroup linalg
 */
template <typename Scalar>
class matrix_elementwise_divide_pdu_impl
    : public matrix_elementwise_divide_pdu<Scalar> {
public:
  matrix_elementwise_divide_pdu_impl(const types::shape &shape,
                                     const int num_inputs);
  ~matrix_elementwise_divide_pdu_impl() = default;
};

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_ELEMENTWISE_DIVIDE_IMPL_H */