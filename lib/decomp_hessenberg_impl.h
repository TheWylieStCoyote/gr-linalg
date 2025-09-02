/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_DECOMP_HESSENBERG_IMPL_H
#define INCLUDED_LINALG_DECOMP_HESSENBERG_IMPL_H

#include <gnuradio/linalg/decomp_hessenberg.h>

namespace gr {
namespace linalg {

template <typename Scalar>
class decomp_hessenberg_sync_impl
    : virtual public decomp_hessenberg<Scalar>,
      virtual public linalg_base_sync<Scalar>,
      virtual public decomp_hessenberg_sync<Scalar> {
public:
  decomp_hessenberg_sync_impl(
      const types::shape &shape, bool compute_q = true,
      std::shared_ptr<hessenberg_algorithm<Scalar>> algorithm = nullptr,
      const hessenberg_options &options = hessenberg_options{});
  ~decomp_hessenberg_sync_impl() override = default;
};

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_DECOMP_HESSENBERG_IMPL_H */