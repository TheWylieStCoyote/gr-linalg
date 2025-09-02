/* -*- c++ -*- */
/*
 * Copyright 2024 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_LINALG_MATRIX_POWER_H
#define INCLUDED_LINALG_MATRIX_POWER_H

#include <gnuradio/linalg/api.h>
#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <memory>

namespace gr {
namespace linalg {

/**
 * @brief Matrix power operations supporting both element-wise and matrix power
 *
 * This block supports two modes of power computation:
 * 1. Element-wise power: (A^p)ij = (Aij)^p for each element
 * 2. Matrix power: A^n = A * A * ... * A (n times) for square matrices with
 * integer exponents
 *
 * **Element-wise mode** (default):
 * - Applies power operation to each matrix element individually
 * - Supports any exponent type (real, complex, integer)
 * - Input: matrix A, exponent p (scalar or matrix for broadcasting)
 * - Output: matrix with same shape as A
 * - Formula: result(i,j) = A(i,j)^p
 *
 * **Matrix power mode** (matrix_mode=true):
 * - Computes true matrix power A^n = A * A * ... * A
 * - Requires square matrix input and integer exponent
 * - Uses efficient algorithms: repeated squaring for large exponents
 * - Special cases: A^0 = I, A^(-1) = inv(A), A^(-n) = inv(A)^n
 * - Input: square matrix A, integer exponent n
 * - Output: matrix with same shape as A
 *
 * **Template Parameters:**
 * - Scalar: float, double, std::complex<float>, std::complex<double>
 *
 * **Parameters:**
 * - matrix_mode: false for element-wise (default), true for matrix power
 * - validate_square: validate square matrix in matrix mode (default: true)
 *
 * **Broadcasting Support:**
 * Element-wise mode supports broadcasting when exponent is provided as matrix:
 * - A (m×n) ^ p (scalar) → result (m×n)
 * - A (m×n) ^ P (m×n) → result (m×n)
 * - A (m×n) ^ p (1×n) → result (m×n) [broadcast rows]
 * - A (m×n) ^ p (m×1) → result (m×n) [broadcast columns]
 */
template <typename Scalar>
class LINALG_API matrix_power : virtual public linalg_base<Scalar> {
public:
  typedef std::shared_ptr<matrix_power<Scalar>> sptr;

  /**
   * @brief Default constructor for element-wise power mode
   */
  matrix_power();

  /**
   * @brief Constructor with mode specification
   * @param matrix_mode true for matrix power, false for element-wise power
   * @param validate_square validate square matrices in matrix mode
   */
  matrix_power(bool matrix_mode, bool validate_square = true);

  /**
   * @brief Perform matrix power operation
   * @param input_matrices Input matrices [matrix, exponent]
   * @param output_matrices Output matrices [result]
   * @return Operation status
   */
  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;

  // Static validation and computation helpers
  static void validate_shapes(const types::vector_shapes &input_shapes,
                              const types::vector_shapes &output_shapes,
                              const std::string &name,
                              bool matrix_mode = false);

  static std::vector<size_t> compute_sizes(const types::vector_shapes &shapes);
  static types::vector_shapes
  compute_output_shapes(const types::vector_shapes &input_shapes);

private:
  bool d_matrix_mode;     ///< true for matrix power, false for element-wise
  bool d_validate_square; ///< validate square matrices in matrix mode

  /**
   * @brief Element-wise power computation with broadcasting support
   */
  OperationReturn elementwise_power(
      const Eigen::Map<
          const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &matrix,
      const Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic,
                                           Eigen::Dynamic>> &exponent,
      Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
          &result);

  /**
   * @brief Matrix power computation using repeated squaring
   */
  OperationReturn matrix_power_computation(
      const Eigen::Map<
          const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &matrix,
      int exponent,
      Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
          &result);
};

/**
 * @brief Sync implementation of matrix power
 */
template <typename Scalar>
class LINALG_API matrix_power_sync : public matrix_power<Scalar>,
                                     public linalg_base_sync<Scalar> {
public:
  typedef std::shared_ptr<matrix_power_sync<Scalar>> sptr;

  static sptr make(const types::vector_shapes &input_shapes,
                   bool matrix_mode = false, bool validate_square = true);

  matrix_power_sync(const types::vector_shapes &input_shapes,
                    bool matrix_mode = false, bool validate_square = true);
};

/**
 * @brief PDU implementation of matrix power
 */
template <typename Scalar>
class LINALG_API matrix_power_pdu : public matrix_power<Scalar>,
                                    public linalg_base_pdu<Scalar> {
public:
  typedef std::shared_ptr<matrix_power_pdu<Scalar>> sptr;

  static sptr make(const types::vector_shapes &input_shapes,
                   bool matrix_mode = false, bool validate_square = true);

  matrix_power_pdu(const types::vector_shapes &input_shapes,
                   bool matrix_mode = false, bool validate_square = true);
};

// Explicit template instantiations
typedef matrix_power<float> matrix_power_f;
typedef matrix_power<double> matrix_power_d;
typedef matrix_power<std::complex<float>> matrix_power_cf;
typedef matrix_power<std::complex<double>> matrix_power_cd;

typedef matrix_power_sync<float> matrix_power_sync_f;
typedef matrix_power_sync<double> matrix_power_sync_d;
typedef matrix_power_sync<std::complex<float>> matrix_power_sync_cf;
typedef matrix_power_sync<std::complex<double>> matrix_power_sync_cd;

typedef matrix_power_pdu<float> matrix_power_pdu_f;
typedef matrix_power_pdu<double> matrix_power_pdu_d;
typedef matrix_power_pdu<std::complex<float>> matrix_power_pdu_cf;
typedef matrix_power_pdu<std::complex<double>> matrix_power_pdu_cd;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_POWER_H */