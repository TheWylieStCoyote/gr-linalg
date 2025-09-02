/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_MATRIX_SOLVE_IMPL_H
#define INCLUDED_LINALG_MATRIX_SOLVE_IMPL_H

#include <gnuradio/linalg/matrix_solve.h>

namespace gr {
namespace linalg {

template <typename Scalar>
class matrix_solve_sync_impl : public matrix_solve_sync<Scalar> {
public:
  matrix_solve_sync_impl(const types::shape &shape_a,
                         const types::shape &shape_b,
                         typename matrix_solve<Scalar>::solver_method method,
                         double tolerance);
  ~matrix_solve_sync_impl();
};

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_SOLVE_IMPL_H */