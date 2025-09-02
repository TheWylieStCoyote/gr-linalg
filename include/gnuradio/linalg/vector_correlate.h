/* -*- c++ -*- */
/*
 * Copyright 2025 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_LINALG_VECTOR_CORRELATE_H
#define INCLUDED_LINALG_VECTOR_CORRELATE_H

#include <Eigen/Dense>
#include <gnuradio/linalg/api.h>
#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <memory>

namespace gr {
namespace linalg {

/**
 * @brief Cross-correlation operations for vectors
 *
 * Computes the cross-correlation between two input vectors, which is a
 * fundamental operation in signal processing for measuring similarity and
 * finding patterns.
 *
 * **Mathematical Background:**
 * For vectors a and b of length N, the cross-correlation R_ab(k) at lag k is:
 * - R_ab(k) = Σ_{n=0}^{N-1-k} a[n] * conj(b[n+k]) for k >= 0
 * - R_ab(k) = conj(R_ba(-k)) for k < 0
 *
 * **Output Modes:**
 * - "full": Returns correlation of length 2*N-1 with all possible lags
 * - "valid": Returns correlation only where vectors fully overlap
 * - "same": Returns correlation of same size as input vectors (central portion)
 *
 * **Implementation Details:**
 * - Uses FFT-based correlation for efficiency with large vectors
 * - Supports both auto-correlation (single input) and cross-correlation (two
 * inputs)
 * - Handles complex-valued signals correctly with conjugation
 * - Output is real-valued for real inputs, complex for complex inputs
 *
 * **Applications:**
 * - Signal detection and synchronization
 * - Pattern matching and template matching
 * - Time delay estimation
 * - Echo/multipath analysis
 * - Matched filtering
 *
 * @tparam Scalar The input/output scalar type (float, double, complex<float>,
 * complex<double>)
 */
template <typename Scalar>
class LINALG_API vector_correlate : virtual public linalg_base<Scalar> {
public:
  typedef std::shared_ptr<vector_correlate<Scalar>> sptr;

  /**
   * @brief Correlation mode enumeration
   */
  enum CorrelateMode {
    FULL,  ///< Full correlation (length 2*N-1)
    VALID, ///< Valid correlation (overlap region only)
    SAME   ///< Same size as input (central portion)
  };

  // Trivial inline ctor to allow most-derived impl to construct the virtual
  // base
  vector_correlate();
  vector_correlate(const types::shape &input_shape_a,
                   const types::shape &input_shape_b, CorrelateMode mode = FULL,
                   bool normalize = false);

  /**
   * @brief Perform vector correlation operation
   *
   * @param input_vectors Input vectors (1 or 2 vectors depending on operation)
   * @param output_vectors Output correlation vector(s)
   * @return Operation status
   */
  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_vectors,
            types::vector_matrix_map<Scalar> &output_vectors);

  // Static helpers delegate to linalg_base/types
  static const types::vector_shapes
  compute_output_shapes(const types::vector_shapes &input_shapes,
                        CorrelateMode mode);

  static const int compute_output_vlen(const types::vector_shapes &input_shapes,
                                       CorrelateMode mode);

  static const gr::io_signature::sptr
  make_input_signature(const types::vector_shapes &input_shapes);

  static const gr::io_signature::sptr
  make_output_signature(const types::vector_shapes &input_shapes,
                        CorrelateMode mode);

private:
  CorrelateMode d_mode;
  bool d_normalize;
  types::shape d_input_shape_a;
  types::shape d_input_shape_b;

  // Implementation helper methods
  void compute_autocorrelation(
      const Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic,
                                           Eigen::Dynamic>> &matrix_a,
      Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
          &output_matrix);

  void compute_crosscorrelation(
      const Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic,
                                           Eigen::Dynamic>> &matrix_a,
      const Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic,
                                           Eigen::Dynamic>> &matrix_b,
      Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
          &output_matrix);

  void normalize_correlation(
      Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
          &output_matrix);
};

// Sync block implementation
template <typename Scalar>
class LINALG_API vector_correlate_sync : public linalg_base_sync<Scalar>,
                                         public vector_correlate<Scalar> {
public:
  typedef std::shared_ptr<vector_correlate_sync<Scalar>> sptr;

  static sptr make(const types::shape &input_shape_a,
                   const types::shape &input_shape_b,
                   typename vector_correlate<Scalar>::CorrelateMode mode =
                       vector_correlate<Scalar>::FULL,
                   bool normalize = false);

  vector_correlate_sync(const types::shape &input_shape_a,
                        const types::shape &input_shape_b,
                        typename vector_correlate<Scalar>::CorrelateMode mode =
                            vector_correlate<Scalar>::FULL,
                        bool normalize = false);
};

// PDU block implementation
template <typename Scalar>
class LINALG_API vector_correlate_pdu : public linalg_base_pdu<Scalar>,
                                        public vector_correlate<Scalar> {
public:
  typedef std::shared_ptr<vector_correlate_pdu<Scalar>> sptr;

  static sptr make(const types::vector_shapes &input_shapes,
                   typename vector_correlate<Scalar>::CorrelateMode mode =
                       vector_correlate<Scalar>::FULL,
                   bool normalize = false);

  vector_correlate_pdu(const types::vector_shapes &input_shapes,
                       typename vector_correlate<Scalar>::CorrelateMode mode =
                           vector_correlate<Scalar>::FULL,
                       bool normalize = false);
};

// Type aliases for common use cases
typedef vector_correlate_sync<float> vector_correlate_sync_f;
typedef vector_correlate_sync<double> vector_correlate_sync_d;
typedef vector_correlate_sync<std::complex<float>> vector_correlate_sync_c;
typedef vector_correlate_sync<std::complex<double>> vector_correlate_sync_cd;

typedef vector_correlate_pdu<float> vector_correlate_pdu_f;
typedef vector_correlate_pdu<double> vector_correlate_pdu_d;
typedef vector_correlate_pdu<std::complex<float>> vector_correlate_pdu_c;
typedef vector_correlate_pdu<std::complex<double>> vector_correlate_pdu_cd;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_VECTOR_CORRELATE_H */