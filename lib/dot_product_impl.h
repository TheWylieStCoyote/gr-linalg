/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_DOT_PRODUCT_IMPL_H
#define INCLUDED_LINALG_DOT_PRODUCT_IMPL_H

#include <gnuradio/linalg/dot_product.h>
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/types.h>

namespace gr {
namespace linalg {

template <typename Scalar>
class dot_product_sync_impl : public dot_product_sync<Scalar> {
public:
  dot_product_sync_impl(const types::shape &shape_0,
                        const types::shape &shape_1);
  ~dot_product_sync_impl();
};

// PDU implementation temporarily disabled
// template <typename Scalar>
// class dot_product_pdu_impl : public dot_product_pdu<Scalar> {
// public:
//   dot_product_pdu_impl(const types::shape &shape_0, const types::shape
//   &shape_1); ~dot_product_pdu_impl();
// };

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_DOT_PRODUCT_IMPL_H */
