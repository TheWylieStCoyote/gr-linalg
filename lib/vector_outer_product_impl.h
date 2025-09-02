/* -*- c++ -*- */
/*
 * Copyright 2024 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_LINALG_VECTOR_OUTER_PRODUCT_IMPL_H
#define INCLUDED_LINALG_VECTOR_OUTER_PRODUCT_IMPL_H

#include <gnuradio/linalg/vector_outer_product.h>

namespace gr {
namespace linalg {

template <typename Scalar>
class vector_outer_product_sync_impl
    : public vector_outer_product_sync<Scalar> {
public:
  vector_outer_product_sync_impl();
  ~vector_outer_product_sync_impl() = default;
};

template <typename Scalar>
class vector_outer_product_pdu_impl : public vector_outer_product_pdu<Scalar> {
public:
  vector_outer_product_pdu_impl();
  ~vector_outer_product_pdu_impl() = default;
};

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_VECTOR_OUTER_PRODUCT_IMPL_H */