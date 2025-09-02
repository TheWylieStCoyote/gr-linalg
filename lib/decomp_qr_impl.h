/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_DECOMP_QR_IMPL_H
#define INCLUDED_LINALG_DECOMP_QR_IMPL_H

#include "gnuradio/linalg/linalg_base_sync.h"
#include <gnuradio/linalg/decomp_qr.h>

namespace gr {
namespace linalg {

template <typename Scalar>
class decomp_qr_sync_impl : virtual public decomp_qr<Scalar>,
                            virtual public linalg_base_sync<Scalar>,
                            virtual public decomp_qr_sync<Scalar> {
public:
  decomp_qr_sync_impl(const types::shape &shape);
  ~decomp_qr_sync_impl() override = default;
};

// PDU implementation - commenting out for now to match decomp_lu pattern
template <typename Scalar>
class decomp_qr_pdu_impl : virtual public decomp_qr<Scalar>,
                           virtual public linalg_base_pdu<Scalar>,
                           virtual public decomp_qr_pdu<Scalar> {
public:
  decomp_qr_pdu_impl(const types::shape &shape);
  ~decomp_qr_pdu_impl() override = default;
};

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_DECOMP_QR_IMPL_H */
