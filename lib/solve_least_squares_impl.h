/* -*- c++ -*- */
/*
 * Copyright 2025 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_SOLVE_LEAST_SQUARES_IMPL_H
#define INCLUDED_LINALG_SOLVE_LEAST_SQUARES_IMPL_H

#include <gnuradio/linalg/solve_least_squares.h>

namespace gr {
namespace linalg {

template <typename Scalar>
class solve_least_squares_sync_impl : public solve_least_squares_sync<Scalar> {
private:
  least_squares_method d_method;
  double d_tolerance;

public:
  solve_least_squares_sync_impl(const types::shape &shape_a,
                                const types::shape &shape_b,
                                least_squares_method method, double tolerance);

  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;
};

template <typename Scalar>
class solve_least_squares_pdu_impl : public solve_least_squares_pdu<Scalar> {
private:
  least_squares_method d_method;
  double d_tolerance;

public:
  solve_least_squares_pdu_impl(const types::shape &shape_a,
                               const types::shape &shape_b,
                               least_squares_method method, double tolerance);

  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;
};

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_SOLVE_LEAST_SQUARES_IMPL_H */
