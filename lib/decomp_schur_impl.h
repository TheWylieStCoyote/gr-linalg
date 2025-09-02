/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_DECOMP_SCHUR_IMPL_H
#define INCLUDED_LINALG_DECOMP_SCHUR_IMPL_H

#include <gnuradio/linalg/decomp_schur.h>

namespace gr {
namespace linalg {

// Most-derived sync implementation: initializes linalg_base and
// linalg_base_sync

template <typename Scalar>
class decomp_schur_sync_impl : virtual public decomp_schur<Scalar>,
                               virtual public linalg_base_sync<Scalar>,
                               virtual public decomp_schur_sync<Scalar> {
public:
  explicit decomp_schur_sync_impl(
      const types::shape &shape, bool compute_u = true,
      std::shared_ptr<schur_algorithm<Scalar>> algorithm = nullptr,
      const schur_options &options = schur_options{});
  ~decomp_schur_sync_impl() override = default;
};

// Note: decomp_schur_pdu uses inline implementation in header, no impl class
// needed

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_DECOMP_SCHUR_IMPL_H */
