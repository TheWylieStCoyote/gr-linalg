/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_DECOMP_EIGEN_IMPL_H
#define INCLUDED_LINALG_DECOMP_EIGEN_IMPL_H

#include <gnuradio/linalg/decomp_eigen.h>

namespace gr {
namespace linalg {

// Most-derived sync implementation: initializes linalg_base and
// linalg_base_sync

template <typename Scalar>
class decomp_eigen_sync_impl : virtual public decomp_eigen<Scalar>,
                               virtual public linalg_base_sync<Scalar>,
                               virtual public decomp_eigen_sync<Scalar> {
public:
  explicit decomp_eigen_sync_impl(const types::shape &shape);
  ~decomp_eigen_sync_impl() override = default;
};

template <typename Scalar>
class decomp_eigen_pdu_impl : virtual public decomp_eigen<Scalar>,
                              virtual public linalg_base_pdu<Scalar>,
                              virtual public decomp_eigen_pdu<Scalar> {
public:
  explicit decomp_eigen_pdu_impl(const types::shape &shape);
  ~decomp_eigen_pdu_impl() override = default;
};

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_DECOMP_EIGEN_IMPL_H */
