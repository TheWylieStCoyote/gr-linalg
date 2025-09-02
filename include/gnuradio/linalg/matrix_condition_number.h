/* -*- c++ -*- */
/*
 * Copyright 2024 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_LINALG_MATRIX_CONDITION_NUMBER_H
#define INCLUDED_LINALG_MATRIX_CONDITION_NUMBER_H

#include <gnuradio/linalg/api.h>
#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <memory>

namespace gr {
namespace linalg {

/**
 * @brief Matrix condition number computation for numerical stability analysis
 * @ingroup linalg
 *
 * Computes the condition number of a matrix, which measures how sensitive
 * the solution of a linear system is to perturbations in the input data.
 *
 * **Mathematical Definition:**
 * The condition number κ(A) of a matrix A is defined as:
 * κ(A) = ||A|| × ||A⁻¹|| = σ_max(A) / σ_min(A)
 *
 * Where σ_max and σ_min are the largest and smallest singular values.
 *
 * **Key Properties:**
 * - κ(A) ≥ 1 for all invertible matrices
 * - κ(I) = 1 for identity matrix (perfectly conditioned)
 * - κ(A) = ∞ for singular matrices (rank deficient)
 * - Higher condition numbers indicate numerical instability
 *
 * **Condition Number Interpretation:**
 * - κ ≈ 1: Well-conditioned (stable numerical solutions)
 * - κ ≈ 10²-10⁶: Moderately conditioned
 * - κ ≈ 10¹²-10¹⁶: Ill-conditioned (near machine precision limits)
 * - κ = ∞: Singular matrix (no unique solution exists)
 *
 * **Algorithm:**
 * Uses Singular Value Decomposition (SVD) to compute condition number as
 * the ratio of largest to smallest singular values:
 * 1. Compute A = U Σ Vᵀ via SVD
 * 2. Extract singular values σᵢ from diagonal of Σ
 * 3. Return κ(A) = max(σᵢ) / min(σᵢ)
 *
 * **Supported Matrix Types:**
 * - Square matrices (n×n): Standard condition number
 * - Rectangular matrices (m×n): Generalized condition number
 * - Complex matrices: Uses complex SVD with magnitude-based comparison
 *
 * **Applications:**
 * - **Linear System Solving**: Assess numerical stability of Ax = b
 * - **Matrix Inversion**: Determine if matrix is safely invertible
 * - **Least Squares**: Evaluate reliability of overdetermined systems
 * - **Signal Processing**: Filter design and stability analysis
 * - **Control Systems**: System stability and robustness analysis
 * - **Optimization**: Convergence analysis and algorithm selection
 *
 * **Performance Characteristics:**
 * - Time Complexity: O(min(m²n, mn²)) for m×n matrix
 * - Space Complexity: O(mn) for SVD computation
 * - Numerical Stability: Uses robust SVD algorithms
 * - Handles edge cases: Zero matrices, rank-deficient matrices
 *
 * **Error Handling:**
 * - Empty matrices return condition number of ∞
 * - Zero matrices return condition number of ∞
 * - Single-element matrices return condition number of 1
 * - Numerical overflow protection for extreme condition numbers
 *
 * **Template Parameters:**
 * - Scalar: float, double, std::complex<float>, std::complex<double>
 *
 * **Usage Examples:**
 * ```cpp
 * // Well-conditioned matrix analysis
 * auto cond_block = matrix_condition_number_sync<double>::make();
 * // Input: 2×2 identity-like matrix
 * // Output: condition number ≈ 1.0
 *
 * // Ill-conditioned matrix detection
 * // Input: Nearly singular matrix
 * // Output: condition number > 10¹²
 * ```
 *
 * **Numerical Considerations:**
 * - Uses machine epsilon thresholds for singular value comparison
 * - Returns std::numeric_limits<T>::infinity() for singular matrices
 * - Handles floating point underflow/overflow in extreme cases
 * - Complex matrices use magnitude of singular values
 */
template <typename Scalar>
class LINALG_API matrix_condition_number : virtual public linalg_base<Scalar> {
public:
  typedef std::shared_ptr<matrix_condition_number<Scalar>> sptr;

  /**
   * @brief Default constructor
   */
  matrix_condition_number();

  /**
   * @brief Perform matrix condition number computation
   * @param input_matrices Input matrices [matrix]
   * @param output_matrices Output scalars [condition_number]
   * @return OperationReturn::SUCCESS on success, error code on failure
   */
  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;

  // Static validation and computation helpers
  static void validate_shapes(const types::vector_shapes &input_shapes,
                              const types::vector_shapes &output_shapes,
                              const std::string &name);

  static std::vector<size_t> compute_sizes(const types::vector_shapes &shapes);
  static types::vector_shapes
  compute_output_shapes(const types::vector_shapes &input_shapes);

  /**
   * @brief Extract real-valued singular value for condition number computation
   * (public for testing)
   */
  static typename Eigen::NumTraits<Scalar>::Real
  extract_real_value(const Scalar &singular_value);

  /**
   * @brief Check if a singular value should be considered zero (public for
   * testing)
   */
  static bool is_effectively_zero(
      const typename Eigen::NumTraits<Scalar>::Real &singular_value,
      const typename Eigen::NumTraits<Scalar>::Real &tolerance);

private:
  /**
   * @brief Core condition number computation using SVD
   * @param matrix Input matrix (m×n)
   * @param result Output scalar (condition number)
   * @return OperationReturn status
   */
  OperationReturn compute_condition_number(
      const Eigen::Map<
          const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &matrix,
      Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
          &result);
};

/**
 * @brief Sync implementation of matrix condition number
 */
template <typename Scalar>
class LINALG_API matrix_condition_number_sync
    : virtual public matrix_condition_number<Scalar>,
      virtual public linalg_base_sync<Scalar> {
public:
  typedef std::shared_ptr<matrix_condition_number_sync<Scalar>> sptr;

  static sptr make(const types::shape &shape);
  matrix_condition_number_sync();

protected:
  matrix_condition_number_sync(const types::shape &shape);
};

/**
 * @brief PDU implementation of matrix condition number
 */
template <typename Scalar>
class LINALG_API matrix_condition_number_pdu
    : virtual public matrix_condition_number<Scalar>,
      virtual public linalg_base_pdu<Scalar> {
public:
  typedef std::shared_ptr<matrix_condition_number_pdu<Scalar>> sptr;

  static sptr make();

protected:
  matrix_condition_number_pdu();
};

// Type aliases for convenience
typedef matrix_condition_number<float> matrix_condition_number_f;
typedef matrix_condition_number<double> matrix_condition_number_d;
typedef matrix_condition_number<std::complex<float>> matrix_condition_number_cf;
typedef matrix_condition_number<std::complex<double>>
    matrix_condition_number_cd;

typedef matrix_condition_number_sync<float> matrix_condition_number_sync_f;
typedef matrix_condition_number_sync<double> matrix_condition_number_sync_d;
typedef matrix_condition_number_sync<std::complex<float>>
    matrix_condition_number_sync_cf;
typedef matrix_condition_number_sync<std::complex<double>>
    matrix_condition_number_sync_cd;

typedef matrix_condition_number_pdu<float> matrix_condition_number_pdu_f;
typedef matrix_condition_number_pdu<double> matrix_condition_number_pdu_d;
typedef matrix_condition_number_pdu<std::complex<float>>
    matrix_condition_number_pdu_cf;
typedef matrix_condition_number_pdu<std::complex<double>>
    matrix_condition_number_pdu_cd;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_CONDITION_NUMBER_H */