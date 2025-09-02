/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_DECOMP_CHOLESKY_IMPL_H
#define INCLUDED_LINALG_DECOMP_CHOLESKY_IMPL_H

#include <gnuradio/linalg/decomp_cholesky.h>

namespace gr {
namespace linalg {

// Most-derived sync implementation
// Initializes linalg_base and linalg_base_sync properly

template <typename Scalar>
class decomp_cholesky_sync_impl : virtual public decomp_cholesky<Scalar>,
                                  virtual public linalg_base_sync<Scalar>,
                                  virtual public decomp_cholesky_sync<Scalar> {
public:
  explicit decomp_cholesky_sync_impl(const types::shape &shape);
  ~decomp_cholesky_sync_impl() override = default;
};

template <typename Scalar>
class decomp_cholesky_pdu_impl : virtual public decomp_cholesky<Scalar>,
                                 virtual public linalg_base_pdu<Scalar>,
                                 virtual public decomp_cholesky_pdu<Scalar> {
public:
  explicit decomp_cholesky_pdu_impl(const types::shape &shape);
  ~decomp_cholesky_pdu_impl() override = default;
};

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_DECOMP_CHOLESKY_IMPL_H */
