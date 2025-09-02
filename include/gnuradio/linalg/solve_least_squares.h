/* -*- c++ -*- */
/*
 * Copyright 2025 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_SOLVE_LEAST_SQUARES_H
#define INCLUDED_LINALG_SOLVE_LEAST_SQUARES_H

#include <gnuradio/linalg/api.h>
#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/types.h>
#include <memory>

namespace gr {
namespace linalg {

/*!
 * \brief Solve Least Squares Problems Ax = b
 * \ingroup linalg
 *
 * This block solves the linear least squares problem:
 *
 *     minimize ||Ax - b||₂
 *
 * For an overdetermined system where A is m×n with m ≥ n.
 *
 * Mathematical Methods (automatically selected based on matrix properties):
 * 1. **Normal Equations**: x = (A^T A)^(-1) A^T b
 *    - Used for well-conditioned matrices (fastest)
 *    - Condition number < 1e12 for double precision
 *
 * 2. **QR Decomposition**: A = QR, then x = R^(-1) Q^T b
 *    - Better numerical stability for moderately ill-conditioned matrices
 *    - Used when Normal Equations would be unstable
 *
 * 3. **SVD (Singular Value Decomposition)**: A = UΣV^T, then x = V Σ^+ U^T b
 *    - Most robust method for ill-conditioned and rank-deficient matrices
 *    - Uses Moore-Penrose pseudoinverse when rank(A) < n
 *
 * Applications:
 * - Linear regression and curve fitting
 * - Parameter estimation in signal processing
 * - System identification and adaptive filtering
 * - Image reconstruction and denoising
 * - Overdetermined linear equation systems
 * - Sensor calibration and beamforming
 *
 * Input/Output:
 * - Input 0: Matrix A (m×n coefficient matrix)
 * - Input 1: Vector/Matrix b (m×k right-hand sides, k≥1)
 * - Output: Solution x (n×k)
 */
template <typename Scalar>
class LINALG_API solve_least_squares : virtual public linalg_base<Scalar> {
public:
  using sptr = std::shared_ptr<solve_least_squares<Scalar>>;
};

// Method selection enumeration
enum class least_squares_method {
  AUTO = 0,  /*!< Automatic selection based on condition number */
  NORMAL_EQ, /*!< Normal equations: (A^T A) x = A^T b */
  QR,        /*!< QR decomposition method */
  SVD        /*!< Singular Value Decomposition (most robust) */
};

// Forward declarations for implementation classes
template <typename Scalar>
class solve_least_squares_sync_impl;
template <typename Scalar>
class solve_least_squares_pdu_impl;

// Sync block type aliases with make functions
template <typename Scalar>
class LINALG_API solve_least_squares_sync : public solve_least_squares<Scalar>,
                                            public linalg_base_sync<Scalar> {
public:
  using sptr = std::shared_ptr<solve_least_squares_sync<Scalar>>;
  solve_least_squares_sync(
      const types::shape &shape_a, const types::shape &shape_b,
      least_squares_method method = least_squares_method::AUTO,
      double tolerance = 1e-10);
  static sptr make(const types::shape &shape_a, const types::shape &shape_b,
                   least_squares_method method = least_squares_method::AUTO,
                   double tolerance = 1e-10);
};

using solve_least_squares_sync_f = solve_least_squares_sync<float>;
using solve_least_squares_sync_d = solve_least_squares_sync<double>;
using solve_least_squares_sync_c =
    solve_least_squares_sync<std::complex<float>>;
using solve_least_squares_sync_cd =
    solve_least_squares_sync<std::complex<double>>;

// PDU block type aliases with make functions
template <typename Scalar>
class LINALG_API solve_least_squares_pdu : public solve_least_squares<Scalar>,
                                           public linalg_base_pdu<Scalar> {
public:
  using sptr = std::shared_ptr<solve_least_squares_pdu<Scalar>>;
  solve_least_squares_pdu(
      const types::shape &shape_a, const types::shape &shape_b,
      least_squares_method method = least_squares_method::AUTO,
      double tolerance = 1e-10);
  static sptr make(const types::shape &shape_a, const types::shape &shape_b,
                   least_squares_method method = least_squares_method::AUTO,
                   double tolerance = 1e-10);
};

using solve_least_squares_pdu_f = solve_least_squares_pdu<float>;
using solve_least_squares_pdu_d = solve_least_squares_pdu<double>;
using solve_least_squares_pdu_c = solve_least_squares_pdu<std::complex<float>>;
using solve_least_squares_pdu_cd =
    solve_least_squares_pdu<std::complex<double>>;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_SOLVE_LEAST_SQUARES_H */
