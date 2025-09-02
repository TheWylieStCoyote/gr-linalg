/* -*- c++ -*- */
/*
 * Copyright 2025 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_LINALG_MATRIX_SQRT_H
#define INCLUDED_LINALG_MATRIX_SQRT_H

#include <gnuradio/linalg/api.h>
#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <memory>

namespace gr {
namespace linalg {

/**
 * @brief Matrix square root operations
 *
 * Computes the principal square root of a square matrix A, denoted as A^(1/2),
 * such that (A^(1/2))^2 = A. This is different from Cholesky decomposition,
 * which applies only to positive definite matrices and gives L*L^T = A.
 *
 * The matrix square root is computed using various algorithms depending on
 * the matrix properties and desired numerical stability.
 *
 * **Mathematical Background:**
 * For a square matrix A with eigenvalue decomposition A = P * D * P^(-1):
 * - sqrt(A) = P * sqrt(D) * P^(-1), where sqrt(D) is element-wise square root
 * of eigenvalues
 * - The principal square root is chosen (eigenvalues in the right half-plane)
 * - For positive definite matrices, the result is unique and positive definite
 *
 * **Algorithm Selection:**
 * 1. **Eigenvalue Decomposition** (general case): A = P * D * P^(-1) → sqrt(A)
 * = P * sqrt(D) * P^(-1)
 * 2. **Schur Decomposition** (more stable): Uses real Schur form for numerical
 * stability
 * 3. **Denman-Beavers Iteration** (iterative): X_{k+1} = (X_k + X_k^(-1) * A) /
 * 2
 * 4. **Newton's Method** (fast convergence): X_{k+1} = (X_k + A * X_k^(-1)) / 2
 *
 * **Special Cases:**
 * - Identity matrix: sqrt(I) = I
 * - Zero matrix: sqrt(0) = 0
 * - Hermitian positive definite matrices: Use eigenvalue decomposition with
 * real eigenvalues
 * - Triangular matrices: Specialized algorithms for better performance
 *
 * **Template Parameters:**
 * - Scalar: float, double, std::complex<float>, std::complex<double>
 *
 * **Input Requirements:**
 * - Matrix must be square (n×n)
 * - Matrix should have no eigenvalues on the negative real axis for uniqueness
 * - For real matrices, complex eigenvalues should come in conjugate pairs
 *
 * **Error Handling:**
 * - Returns FAILURE for matrices with eigenvalues on negative real axis
 * - Returns INVALID_SHAPE for non-square matrices
 * - Handles numerical edge cases with appropriate tolerances
 *
 * **Note on Cholesky vs Matrix Square Root:**
 * - Cholesky: A = L*L^T (only for positive definite A, gives triangular L)
 * - Matrix sqrt: A = S*S (for general A, gives S with same structure as A)
 */
template <typename Scalar>
class LINALG_API matrix_sqrt : virtual public linalg_base<Scalar> {
public:
  typedef std::shared_ptr<matrix_sqrt<Scalar>> sptr;

  /**
   * @brief Algorithm options for computing matrix square root
   */
  enum class Algorithm {
    AUTO,           ///< Automatically select best algorithm
    EIGENVALUE,     ///< Use eigenvalue decomposition
    SCHUR,          ///< Use Schur decomposition (more stable)
    DENMAN_BEAVERS, ///< Use Denman-Beavers iteration
    NEWTON          ///< Use Newton's method
  };

  /**
   * @brief Default constructor (uses AUTO algorithm)
   */
  matrix_sqrt();

  /**
   * @brief Constructor with algorithm selection
   * @param algorithm Algorithm to use for computation
   * @param max_iterations Maximum iterations for iterative methods (default:
   * 100)
   * @param tolerance Convergence tolerance for iterative methods (default:
   * 1e-10)
   */
  matrix_sqrt(Algorithm algorithm = Algorithm::AUTO, int max_iterations = 100,
              double tolerance = 1e-10);

  /**
   * @brief Perform matrix square root operation
   * @param input_matrices Input matrices [A]
   * @param output_matrices Output matrices [sqrt(A)]
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
  Algorithm d_algorithm; ///< Selected algorithm
  int d_max_iterations;  ///< Maximum iterations for iterative methods
  double d_tolerance;    ///< Convergence tolerance

  /**
   * @brief Automatically select the best algorithm based on matrix properties
   */
  Algorithm select_algorithm(
      const Eigen::Map<
          const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &matrix);

  /**
   * @brief Compute square root using eigenvalue decomposition
   */
  OperationReturn compute_eigenvalue_sqrt(
      const Eigen::Map<
          const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &matrix,
      Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
          &result);

  /**
   * @brief Compute square root using Schur decomposition
   */
  OperationReturn compute_schur_sqrt(
      const Eigen::Map<
          const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &matrix,
      Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
          &result);

  /**
   * @brief Compute square root using Denman-Beavers iteration
   */
  OperationReturn compute_denman_beavers_sqrt(
      const Eigen::Map<
          const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &matrix,
      Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
          &result);

  /**
   * @brief Compute square root using Newton's method
   */
  OperationReturn compute_newton_sqrt(
      const Eigen::Map<
          const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &matrix,
      Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
          &result);

  /**
   * @brief Check if matrix is positive definite
   */
  bool is_positive_definite(
      const Eigen::Map<
          const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &matrix);
};

/**
 * @brief Sync implementation of matrix square root
 */
template <typename Scalar>
class LINALG_API matrix_sqrt_sync : public matrix_sqrt<Scalar>,
                                    public linalg_base_sync<Scalar> {
public:
  typedef std::shared_ptr<matrix_sqrt_sync<Scalar>> sptr;

  static sptr make(const types::vector_shapes &input_shapes,
                   typename matrix_sqrt<Scalar>::Algorithm algorithm =
                       matrix_sqrt<Scalar>::Algorithm::AUTO,
                   int max_iterations = 100, double tolerance = 1e-10);

  matrix_sqrt_sync(const types::vector_shapes &input_shapes,
                   typename matrix_sqrt<Scalar>::Algorithm algorithm =
                       matrix_sqrt<Scalar>::Algorithm::AUTO,
                   int max_iterations = 100, double tolerance = 1e-10);
};

/**
 * @brief PDU implementation of matrix square root
 */
template <typename Scalar>
class LINALG_API matrix_sqrt_pdu : public matrix_sqrt<Scalar>,
                                   public linalg_base_pdu<Scalar> {
public:
  typedef std::shared_ptr<matrix_sqrt_pdu<Scalar>> sptr;

  static sptr make(const types::vector_shapes &input_shapes,
                   typename matrix_sqrt<Scalar>::Algorithm algorithm =
                       matrix_sqrt<Scalar>::Algorithm::AUTO,
                   int max_iterations = 100, double tolerance = 1e-10);

  matrix_sqrt_pdu(const types::vector_shapes &input_shapes,
                  typename matrix_sqrt<Scalar>::Algorithm algorithm =
                      matrix_sqrt<Scalar>::Algorithm::AUTO,
                  int max_iterations = 100, double tolerance = 1e-10);
};

// Explicit template instantiations
typedef matrix_sqrt<float> matrix_sqrt_f;
typedef matrix_sqrt<double> matrix_sqrt_d;
typedef matrix_sqrt<std::complex<float>> matrix_sqrt_cf;
typedef matrix_sqrt<std::complex<double>> matrix_sqrt_cd;

typedef matrix_sqrt_sync<float> matrix_sqrt_sync_f;
typedef matrix_sqrt_sync<double> matrix_sqrt_sync_d;
typedef matrix_sqrt_sync<std::complex<float>> matrix_sqrt_sync_cf;
typedef matrix_sqrt_sync<std::complex<double>> matrix_sqrt_sync_cd;

typedef matrix_sqrt_pdu<float> matrix_sqrt_pdu_f;
typedef matrix_sqrt_pdu<double> matrix_sqrt_pdu_d;
typedef matrix_sqrt_pdu<std::complex<float>> matrix_sqrt_pdu_cf;
typedef matrix_sqrt_pdu<std::complex<double>> matrix_sqrt_pdu_cd;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_SQRT_H */