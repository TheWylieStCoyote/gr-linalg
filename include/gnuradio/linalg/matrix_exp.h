/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_LINALG_MATRIX_EXP_H
#define INCLUDED_LINALG_MATRIX_EXP_H

#include <gnuradio/linalg/api.h>
#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <memory>

namespace gr {
namespace linalg {

/**
 * @brief Matrix exponential operations
 *
 * Computes the matrix exponential exp(A) for square matrices A, where exp(A) is
 * defined by the convergent power series:
 * exp(A) = I + A + A^2/2! + A^3/3! + A^4/4! + ...
 *
 * The matrix exponential is computed using Eigen's implementation with scaling
 * and squaring method for numerical stability.
 *
 * **Template Parameters:**
 * - Scalar: float, double, std::complex<float>, std::complex<double>
 *
 * **Input Requirements:**
 * - Input must be a square matrix (n × n)
 * - Matrix can be singular or non-singular
 * - Complex matrices are supported
 *
 * **Output:**
 * - Square matrix of same dimensions as input
 * - Always invertible (even if input is singular)
 */
template <typename Scalar>
class LINALG_API matrix_exp : virtual public linalg_base<Scalar> {
public:
  typedef std::shared_ptr<matrix_exp<Scalar>> sptr;

  /**
   * @brief Default constructor
   */
  matrix_exp();

  /**
   * @brief Constructor with algorithm selection
   * @param use_scaling_squaring Use scaling and squaring method
   */
  matrix_exp(bool use_scaling_squaring = true);

  /**
   * @brief Perform matrix exponential operation
   * @param input_matrices Input matrices [A]
   * @param output_matrices Output matrices [exp(A)]
   * @return Operation status
   */
  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;

  static types::vector_shapes
  compute_output_shapes(const types::vector_shapes &input_shapes);

private:
  bool d_use_scaling_squaring; ///< Use scaling and squaring method

  void compute_matrix_exp_eigen_decomp(
      const Eigen::Ref<
          const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &input,
      Eigen::Ref<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> output);
};

// Convenience typedefs for common types
typedef matrix_exp<float> matrix_exp_f;
typedef matrix_exp<double> matrix_exp_d;
typedef matrix_exp<std::complex<float>> matrix_exp_cf;
typedef matrix_exp<std::complex<double>> matrix_exp_cd;

/**
 * @brief Sync implementation of matrix exponential
 */
template <typename Scalar>
class LINALG_API matrix_exp_sync : public matrix_exp<Scalar>,
                                   public linalg_base_sync<Scalar> {
public:
  typedef std::shared_ptr<matrix_exp_sync<Scalar>> sptr;

  static sptr make(const types::vector_shapes &input_shapes,
                   bool use_scaling_squaring = true);

  matrix_exp_sync(const types::vector_shapes &input_shapes,
                  bool use_scaling_squaring = true);
};

/**
 * @brief PDU implementation of matrix exponential
 */
template <typename Scalar>
class LINALG_API matrix_exp_pdu : public matrix_exp<Scalar>,
                                  public linalg_base_pdu<Scalar> {
public:
  typedef std::shared_ptr<matrix_exp_pdu<Scalar>> sptr;

  static sptr make(const types::vector_shapes &input_shapes,
                   bool use_scaling_squaring = true);

  matrix_exp_pdu(const types::vector_shapes &input_shapes,
                 bool use_scaling_squaring = true);
};

// More convenience typedefs
typedef matrix_exp_sync<float> matrix_exp_sync_f;
typedef matrix_exp_sync<double> matrix_exp_sync_d;
typedef matrix_exp_sync<std::complex<float>> matrix_exp_sync_cf;
typedef matrix_exp_sync<std::complex<double>> matrix_exp_sync_cd;

typedef matrix_exp_pdu<float> matrix_exp_pdu_f;
typedef matrix_exp_pdu<double> matrix_exp_pdu_d;
typedef matrix_exp_pdu<std::complex<float>> matrix_exp_pdu_cf;
typedef matrix_exp_pdu<std::complex<double>> matrix_exp_pdu_cd;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_EXP_H */
