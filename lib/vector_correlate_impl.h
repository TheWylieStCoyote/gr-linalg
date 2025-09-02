/* -*- c++ -*- */
/*
 * Copyright 2025 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_LINALG_VECTOR_CORRELATE_IMPL_H
#define INCLUDED_LINALG_VECTOR_CORRELATE_IMPL_H

#include <Eigen/Dense>
#include <gnuradio/linalg/vector_correlate.h>

namespace gr {
namespace linalg {

/**
 * @brief Implementation helper methods for vector correlation operations
 *
 * This file contains helper methods for computing auto-correlation and
 * cross-correlation that are used by the template implementations.
 */
template <typename Scalar>
void compute_autocorrelation(
    const Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> &vec_a,
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> &output,
    typename vector_correlate<Scalar>::CorrelateMode mode);

template <typename Scalar>
void compute_crosscorrelation(
    const Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> &vec_a,
    const Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> &vec_b,
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> &output,
    typename vector_correlate<Scalar>::CorrelateMode mode);

template <typename Scalar>
void normalize_correlation(
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> &output);

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_VECTOR_CORRELATE_IMPL_H */
