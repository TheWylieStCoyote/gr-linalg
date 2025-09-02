/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_MATRIX_COVARIANCE_IMPL_H
#define INCLUDED_LINALG_MATRIX_COVARIANCE_IMPL_H

#include <gnuradio/linalg/matrix_covariance.h>
#include <gnuradio/linalg/performance_profiler.h>

namespace gr {
namespace linalg {

template <typename Scalar>
class matrix_covariance_sync_impl : public matrix_covariance_sync<Scalar> {
public:
  matrix_covariance_sync_impl(const types::shape &shape, bool unbiased,
                              bool center);

private:
  void initialize_base_classes(const types::shape &shape, bool unbiased,
                               bool center);
};

template <typename Scalar>
class matrix_covariance_pdu_impl : public matrix_covariance_pdu<Scalar> {
public:
  matrix_covariance_pdu_impl(const types::shape &shape, bool unbiased,
                             bool center);

private:
  void initialize_base_classes(const types::shape &shape, bool unbiased,
                               bool center);
};

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_COVARIANCE_IMPL_H */
