/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_MATRIX_EIGENVALUES_IMPL_H
#define INCLUDED_LINALG_MATRIX_EIGENVALUES_IMPL_H

#include <gnuradio/linalg/matrix_eigenvalues.h>

namespace gr {
namespace linalg {

template <typename Scalar>
class matrix_eigenvalues_sync_impl
    : virtual public matrix_eigenvalues_sync<Scalar> {
private:
public:
  matrix_eigenvalues_sync_impl(const types::shape &shape,
                               bool sort_by_magnitude);
  ~matrix_eigenvalues_sync_impl();
};

template <typename Scalar>
class matrix_eigenvalues_pdu_impl
    : virtual public matrix_eigenvalues_pdu<Scalar> {
private:
public:
  matrix_eigenvalues_pdu_impl(const types::shape &shape,
                              bool sort_by_magnitude);
  ~matrix_eigenvalues_pdu_impl();
};

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_EIGENVALUES_IMPL_H */