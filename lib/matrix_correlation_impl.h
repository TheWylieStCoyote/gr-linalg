/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_MATRIX_CORRELATION_IMPL_H
#define INCLUDED_LINALG_MATRIX_CORRELATION_IMPL_H

#include <gnuradio/linalg/matrix_correlation.h>
// #include <gnuradio/linalg/profiler.h>

namespace gr {
namespace linalg {

template <typename Scalar>
class matrix_correlation_sync_impl : public matrix_correlation_sync<Scalar> {
public:
  matrix_correlation_sync_impl(const types::shape &shape,
                               CorrelationMethod method);

private:
  void initialize_base_classes(const types::shape &shape,
                               CorrelationMethod method);
};

template <typename Scalar>
class matrix_correlation_pdu_impl : public matrix_correlation_pdu<Scalar> {
public:
  matrix_correlation_pdu_impl(const types::shape &shape,
                              CorrelationMethod method);

private:
  void initialize_base_classes(const types::shape &shape,
                               CorrelationMethod method);
};

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_CORRELATION_IMPL_H */
