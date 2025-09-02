/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_DECOMP_LU_IMPL_H
#define INCLUDED_LINALG_DECOMP_LU_IMPL_H

#include <gnuradio/linalg/decomp_lu.h>

namespace gr {
namespace linalg {

template <typename Scalar>
class decomp_lu_sync_impl : virtual public decomp_lu<Scalar>,
                            virtual public linalg_base_sync<Scalar>,
                            virtual public decomp_lu_sync<Scalar> {
public:
  decomp_lu_sync_impl(const types::shape &shape);
  ~decomp_lu_sync_impl() override = default;
};

// PDU implementation not yet implemented
template <typename Scalar>
class decomp_lu_pdu_impl : virtual public decomp_lu<Scalar>,
                           virtual public linalg_base_pdu<Scalar>,
                           virtual public decomp_lu_pdu<Scalar> {
public:
  decomp_lu_pdu_impl(const types::shape &shape);
  ~decomp_lu_pdu_impl() override = default;
};

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_DECOMP_LU_IMPL_H */
