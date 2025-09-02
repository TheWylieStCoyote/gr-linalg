/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_VECTOR_PROJECTION_IMPL_H
#define INCLUDED_LINALG_VECTOR_PROJECTION_IMPL_H

#include <gnuradio/linalg/vector_projection.h>

namespace gr {
namespace linalg {

template <typename Scalar>
class vector_projection_sync_impl : public vector_projection_sync<Scalar> {
public:
  vector_projection_sync_impl(const types::shape &shape_a,
                              const types::shape &shape_b);
  ~vector_projection_sync_impl();
};

template <typename Scalar>
class vector_projection_pdu_impl : public vector_projection_pdu<Scalar> {
public:
  vector_projection_pdu_impl(const types::shape &shape_a,
                             const types::shape &shape_b);
  ~vector_projection_pdu_impl();
};

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_VECTOR_PROJECTION_IMPL_H */
