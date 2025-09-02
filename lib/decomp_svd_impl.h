/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_DECOMP_SVD_IMPL_H
#define INCLUDED_LINALG_DECOMP_SVD_IMPL_H

#include <gnuradio/linalg/decomp_svd.h>

namespace gr {
namespace linalg {

// Most-derived sync implementation: initializes linalg_base and
// linalg_base_sync

template <typename Scalar>
class decomp_svd_sync_impl : virtual public decomp_svd<Scalar>,
                             virtual public linalg_base_sync<Scalar>,
                             virtual public decomp_svd_sync<Scalar> {
public:
  explicit decomp_svd_sync_impl(
      const types::shape &shape,
      std::shared_ptr<svd_algorithm<Scalar>> algorithm = nullptr);
  ~decomp_svd_sync_impl() override = default;
};

// Note: decomp_svd_pdu uses inline implementation in header, no impl class
// needed

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_DECOMP_SVD_IMPL_H */
