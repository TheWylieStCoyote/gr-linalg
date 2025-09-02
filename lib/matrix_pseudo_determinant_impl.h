/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_MATRIX_PSEUDO_DETERMINANT_IMPL_H
#define INCLUDED_LINALG_MATRIX_PSEUDO_DETERMINANT_IMPL_H

#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/matrix_pseudo_determinant.h>
#include <gnuradio/linalg/types.h>

namespace gr {
namespace linalg {

// Forward declarations for both sync and PDU implementations
template <typename Scalar>
class matrix_pseudo_determinant_sync_impl;

template <typename Scalar>
class matrix_pseudo_determinant_pdu_impl;

// Sync implementation
template <typename Scalar>
class matrix_pseudo_determinant_sync_impl
    : public matrix_pseudo_determinant_sync<Scalar> {
public:
  using shape = types::shape;

  matrix_pseudo_determinant_sync_impl(const shape &shape);

  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;
};

// PDU implementation
template <typename Scalar>
class matrix_pseudo_determinant_pdu_impl
    : public matrix_pseudo_determinant_pdu<Scalar> {
public:
  using shape = types::shape;

  matrix_pseudo_determinant_pdu_impl(const shape &shape);

  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;
};

// Explicit instantiations
extern template class matrix_pseudo_determinant_sync_impl<float>;
extern template class matrix_pseudo_determinant_sync_impl<double>;
extern template class matrix_pseudo_determinant_sync_impl<std::complex<float>>;
extern template class matrix_pseudo_determinant_sync_impl<std::complex<double>>;

extern template class matrix_pseudo_determinant_pdu_impl<float>;
extern template class matrix_pseudo_determinant_pdu_impl<double>;
extern template class matrix_pseudo_determinant_pdu_impl<std::complex<float>>;
extern template class matrix_pseudo_determinant_pdu_impl<std::complex<double>>;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_PSEUDO_DETERMINANT_IMPL_H */