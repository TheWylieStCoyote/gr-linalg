/* -*- c++ -*- */
/*
 * Copyright 2025 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_LINALG_MATRIX_LOG_H
#define INCLUDED_LINALG_MATRIX_LOG_H

#include <gnuradio/linalg/api.h>
#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <memory>

namespace gr {
namespace linalg {

/**
 * @brief Matrix logarithm operations
 *
 * Computes the matrix logarithm log(A) for square matrices A, where log(A) is
 * the inverse operation of the matrix exponential such that exp(log(A)) = A.
 *
 * The matrix logarithm is computed using eigenvalue decomposition for general
 * matrices and more efficient algorithms for special cases (Hermitian matrices,
 * triangular matrices).
 *
 * **Mathematical Background:**
 * For a square matrix A with eigenvalue decomposition A = P * D * P^(-1):
 * - log(A) = P * log(D) * P^(-1), where log(D) is element-wise logarithm of
 * eigenvalues
 * - The principal branch is chosen for complex logarithms
 * - The matrix must be invertible (non-singular) for the logarithm to exist
 *
 * **Special Cases:**
 * - Identity matrix: log(I) = 0 (zero matrix)
 * - Hermitian positive definite matrices: Use more stable Cholesky-based
 * algorithms
 * - Upper triangular matrices: Use specialized algorithms for better
 * performance
 *
 * **Template Parameters:**
 * - Scalar: float, double, std::complex<float>, std::complex<double>
 *
 * **Input Requirements:**
 * - Matrix must be square (n×n)
 * - Matrix must be invertible (det(A) ≠ 0)
 * - For real matrices, all eigenvalues must have positive real parts for real
 * logarithm
 *
 * **Error Handling:**
 * - Returns FAILURE for singular matrices
 * - Returns INVALID_SHAPE for non-square matrices
 * - Handles numerical edge cases gracefully
 */
template <typename Scalar>
class LINALG_API matrix_log : virtual public linalg_base<Scalar> {
public:
  typedef std::shared_ptr<matrix_log<Scalar>> sptr;

  /**
   * @brief Default constructor
   */
  matrix_log();

  /**
   * @brief Constructor with algorithm selection
   * @param use_stable_algorithm Use numerically stable algorithms for special
   * matrices
   */
  matrix_log(bool use_stable_algorithm = true);

  /**
   * @brief Perform matrix logarithm operation
   * @param input_matrices Input matrices [A]
   * @param output_matrices Output matrices [log(A)]
   * @return Operation status
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

private:
  bool d_use_stable_algorithm; ///< Use stable algorithms for special matrices

  /**
   * @brief General matrix logarithm using eigenvalue decomposition
   */
  OperationReturn compute_general_log(
      const Eigen::Map<
          const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &matrix,
      Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
          &result);

  /**
   * @brief Specialized algorithm for Hermitian matrices
   */
  OperationReturn compute_hermitian_log(
      const Eigen::Map<
          const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &matrix,
      Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
          &result);

  /**
   * @brief Check if matrix is Hermitian (within numerical tolerance)
   */
  bool is_hermitian(
      const Eigen::Map<
          const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &matrix);
};

/**
 * @brief Sync implementation of matrix logarithm
 */
template <typename Scalar>
class LINALG_API matrix_log_sync : public matrix_log<Scalar>,
                                   public linalg_base_sync<Scalar> {
public:
  typedef std::shared_ptr<matrix_log_sync<Scalar>> sptr;

  static sptr make(const types::vector_shapes &input_shapes,
                   bool use_stable_algorithm = true);

  matrix_log_sync(const types::vector_shapes &input_shapes,
                  bool use_stable_algorithm = true);
};

/**
 * @brief PDU implementation of matrix logarithm
 */
template <typename Scalar>
class LINALG_API matrix_log_pdu : public matrix_log<Scalar>,
                                  public linalg_base_pdu<Scalar> {
public:
  typedef std::shared_ptr<matrix_log_pdu<Scalar>> sptr;

  static sptr make(const types::vector_shapes &input_shapes,
                   bool use_stable_algorithm = true);

  matrix_log_pdu(const types::vector_shapes &input_shapes,
                 bool use_stable_algorithm = true);
};

// Explicit template instantiations
typedef matrix_log<float> matrix_log_f;
typedef matrix_log<double> matrix_log_d;
typedef matrix_log<std::complex<float>> matrix_log_cf;
typedef matrix_log<std::complex<double>> matrix_log_cd;

typedef matrix_log_sync<float> matrix_log_sync_f;
typedef matrix_log_sync<double> matrix_log_sync_d;
typedef matrix_log_sync<std::complex<float>> matrix_log_sync_cf;
typedef matrix_log_sync<std::complex<double>> matrix_log_sync_cd;

typedef matrix_log_pdu<float> matrix_log_pdu_f;
typedef matrix_log_pdu<double> matrix_log_pdu_d;
typedef matrix_log_pdu<std::complex<float>> matrix_log_pdu_cf;
typedef matrix_log_pdu<std::complex<double>> matrix_log_pdu_cd;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_LOG_H */