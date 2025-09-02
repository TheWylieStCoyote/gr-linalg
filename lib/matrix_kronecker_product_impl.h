/* -*- c++ -*- */
/*
 * Copyright 2025 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_LINALG_MATRIX_KRONECKER_PRODUCT_IMPL_H
#define INCLUDED_LINALG_MATRIX_KRONECKER_PRODUCT_IMPL_H

#include <gnuradio/linalg/matrix_kronecker_product.h>

namespace gr {
namespace linalg {

template <typename Scalar>
class matrix_kronecker_product_sync_impl
    : public matrix_kronecker_product_sync<Scalar> {
public:
  matrix_kronecker_product_sync_impl(const types::shape &shape_a,
                                     const types::shape &shape_b);

  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;
};

template <typename Scalar>
class matrix_kronecker_product_pdu_impl
    : public matrix_kronecker_product_pdu<Scalar> {
public:
  matrix_kronecker_product_pdu_impl(const types::shape &shape_a,
                                    const types::shape &shape_b);

  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;
};

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_KRONECKER_PRODUCT_IMPL_H */