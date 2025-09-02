/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_MATRIX_DIAG_IMPL_H
#define INCLUDED_LINALG_MATRIX_DIAG_IMPL_H

#include <gnuradio/linalg/matrix_diag.h>

namespace gr {
namespace linalg {

template <typename Scalar>
class matrix_diag_sync_impl : virtual public linalg_base<Scalar>,
                              virtual public matrix_diag<Scalar>,
                              virtual public linalg_base_sync<Scalar>,
                              virtual public matrix_diag_sync<Scalar> {
private:
  // Nothing to declare in this block.

public:
  matrix_diag_sync_impl(types::shape &shape, int k);
  ~matrix_diag_sync_impl();
};

template <typename Scalar>

class matrix_diag_pdu_impl : virtual public linalg_base<Scalar>,
                             virtual public matrix_diag<Scalar>,
                             virtual public linalg_base_pdu<Scalar>,
                             virtual public matrix_diag_pdu<Scalar> {
private:
  // Nothing to declare in this block.

public:
  matrix_diag_pdu_impl(types::shape &shape, int k);
  ~matrix_diag_pdu_impl();
};

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_DIAG_IMPL_H */
