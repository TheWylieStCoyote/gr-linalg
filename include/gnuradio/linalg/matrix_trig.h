/* -*- c++ -*- */
/*
 * Copyright 2025 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_LINALG_MATRIX_TRIG_H
#define INCLUDED_LINALG_MATRIX_TRIG_H

#include <gnuradio/linalg/api.h>
#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <memory>

namespace gr {
namespace linalg {

// Type trait to get the underlying real type
template <typename T>
struct real_type {
  using type = T;
};

template <typename T>
struct real_type<std::complex<T>> {
  using type = T;
};

template <typename T>
using real_type_t = typename real_type<T>::type;

/**
 * @brief Matrix trigonometric function operations
 *
 * Computes trigonometric functions of square matrices: sin(A), cos(A), tan(A).
 * These are matrix functions computed using Taylor series, eigenvalue
 * decomposition, or more sophisticated algorithms for numerical stability.
 *
 * **Mathematical Background:**
 * For a square matrix A with eigenvalue decomposition A = P * D * P^(-1):
 * - sin(A) = P * sin(D) * P^(-1), where sin(D) is element-wise sine of
 * eigenvalues
 * - cos(A) = P * cos(D) * P^(-1), where cos(D) is element-wise cosine of
 * eigenvalues
 * - tan(A) = P * tan(D) * P^(-1), where tan(D) is element-wise tangent of
 * eigenvalues
 *
 * **Alternative Computation via Taylor Series:**
 * - sin(A) = A - A³/3! + A⁵/5! - A⁷/7! + ...
 * - cos(A) = I - A²/2! + A⁴/4! - A⁶/6! + ...
 * - tan(A) = sin(A) * cos(A)⁻¹ (where cos(A) is invertible)
 *
 * **Algorithm Selection:**
 * 1. **Eigenvalue Decomposition** (general case): Most accurate for general
 * matrices
 * 2. **Taylor Series** (small norm): Fast convergence for ||A|| << 1
 * 3. **Scaling and Squaring** (large norm): Scale matrix down, compute, then
 * scale back
 * 4. **Padé Approximation** (rational approximation): Good numerical stability
 *
 * **Template Parameters:**
 * - Scalar: float, double, std::complex<float>, std::complex<double>
 *
 * **Input Requirements:**
 * - Matrix must be square (n×n)
 * - For tan(A): cos(A) must be invertible
 *
 * **Error Handling:**
 * - Returns FAILURE for singular cos(A) in tan computation
 * - Returns INVALID_SHAPE for non-square matrices
 * - Handles numerical issues with appropriate tolerances
 */

/**
 * @brief Trigonometric function type enumeration
 */
enum class TrigFunction {
  SIN, ///< Sine function
  COS, ///< Cosine function
  TAN  ///< Tangent function
};

/**
 * @brief Algorithm options for computing matrix trigonometric functions
 */
enum class TrigAlgorithm {
  AUTO,             ///< Automatically select best algorithm
  EIGENVALUE,       ///< Use eigenvalue decomposition
  TAYLOR_SERIES,    ///< Use Taylor series expansion
  SCALING_SQUARING, ///< Use scaling and squaring method
  PADE              ///< Use Padé approximation
};

template <typename Scalar>
class LINALG_API matrix_trig : virtual public linalg_base<Scalar> {
public:
  typedef std::shared_ptr<matrix_trig<Scalar>> sptr;

  /**
   * @brief Constructor
   * @param function Trigonometric function to compute (SIN, COS, or TAN)
   * @param algorithm Algorithm to use for computation
   * @param max_iterations Maximum iterations for iterative methods
   * @param tolerance Convergence tolerance for iterative methods
   */
  matrix_trig(TrigFunction function = TrigFunction::SIN,
              TrigAlgorithm algorithm = TrigAlgorithm::AUTO,
              int max_iterations = 100, double tolerance = 1e-10);

  /**
   * @brief Perform matrix trigonometric operation
   * @param input_matrices Input matrices [A]
   * @param output_matrices Output matrices [trig(A)]
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
  TrigFunction d_function;   ///< Trigonometric function type
  TrigAlgorithm d_algorithm; ///< Selected algorithm
  int d_max_iterations;      ///< Maximum iterations for iterative methods
  double d_tolerance;        ///< Convergence tolerance

  /**
   * @brief Automatically select the best algorithm based on matrix properties
   */
  TrigAlgorithm select_algorithm(
      const Eigen::Map<
          const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &matrix);

  /**
   * @brief Compute trigonometric function using eigenvalue decomposition
   */
  OperationReturn compute_eigenvalue_trig(
      const Eigen::Map<
          const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &matrix,
      Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
          &result);

  /**
   * @brief Compute trigonometric function using Taylor series
   */
  OperationReturn compute_taylor_trig(
      const Eigen::Map<
          const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &matrix,
      Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
          &result);

  /**
   * @brief Compute trigonometric function using scaling and squaring
   */
  OperationReturn compute_scaling_squaring_trig(
      const Eigen::Map<
          const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &matrix,
      Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
          &result);

  /**
   * @brief Compute matrix norm for algorithm selection
   */
  real_type_t<Scalar> compute_matrix_norm(
      const Eigen::Map<
          const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &matrix);
};

/**
 * @brief Matrix sine function
 */
template <typename Scalar>
class LINALG_API matrix_sin : public matrix_trig<Scalar> {
public:
  typedef std::shared_ptr<matrix_sin<Scalar>> sptr;

  matrix_sin(TrigAlgorithm algorithm = TrigAlgorithm::AUTO,
             int max_iterations = 100, double tolerance = 1e-10);
};

/**
 * @brief Matrix cosine function
 */
template <typename Scalar>
class LINALG_API matrix_cos : public matrix_trig<Scalar> {
public:
  typedef std::shared_ptr<matrix_cos<Scalar>> sptr;

  matrix_cos(TrigAlgorithm algorithm = TrigAlgorithm::AUTO,
             int max_iterations = 100, double tolerance = 1e-10);
};

/**
 * @brief Matrix tangent function
 */
template <typename Scalar>
class LINALG_API matrix_tan : public matrix_trig<Scalar> {
public:
  typedef std::shared_ptr<matrix_tan<Scalar>> sptr;

  matrix_tan(TrigAlgorithm algorithm = TrigAlgorithm::AUTO,
             int max_iterations = 100, double tolerance = 1e-10);
};

//==============================================================================
// Sync implementations
//==============================================================================

template <typename Scalar>
class LINALG_API matrix_sin_sync : public matrix_sin<Scalar>,
                                   public linalg_base_sync<Scalar> {
public:
  typedef std::shared_ptr<matrix_sin_sync<Scalar>> sptr;

  static sptr make(const types::vector_shapes &input_shapes,
                   TrigAlgorithm algorithm = TrigAlgorithm::AUTO,
                   int max_iterations = 100, double tolerance = 1e-10);

  matrix_sin_sync(const types::vector_shapes &input_shapes,
                  TrigAlgorithm algorithm = TrigAlgorithm::AUTO,
                  int max_iterations = 100, double tolerance = 1e-10);
};

template <typename Scalar>
class LINALG_API matrix_cos_sync : public matrix_cos<Scalar>,
                                   public linalg_base_sync<Scalar> {
public:
  typedef std::shared_ptr<matrix_cos_sync<Scalar>> sptr;

  static sptr make(const types::vector_shapes &input_shapes,
                   TrigAlgorithm algorithm = TrigAlgorithm::AUTO,
                   int max_iterations = 100, double tolerance = 1e-10);

  matrix_cos_sync(const types::vector_shapes &input_shapes,
                  TrigAlgorithm algorithm = TrigAlgorithm::AUTO,
                  int max_iterations = 100, double tolerance = 1e-10);
};

template <typename Scalar>
class LINALG_API matrix_tan_sync : public matrix_tan<Scalar>,
                                   public linalg_base_sync<Scalar> {
public:
  typedef std::shared_ptr<matrix_tan_sync<Scalar>> sptr;

  static sptr make(const types::vector_shapes &input_shapes,
                   TrigAlgorithm algorithm = TrigAlgorithm::AUTO,
                   int max_iterations = 100, double tolerance = 1e-10);

  matrix_tan_sync(const types::vector_shapes &input_shapes,
                  TrigAlgorithm algorithm = TrigAlgorithm::AUTO,
                  int max_iterations = 100, double tolerance = 1e-10);
};

//==============================================================================
// PDU implementations
//==============================================================================

template <typename Scalar>
class LINALG_API matrix_sin_pdu : public matrix_sin<Scalar>,
                                  public linalg_base_pdu<Scalar> {
public:
  typedef std::shared_ptr<matrix_sin_pdu<Scalar>> sptr;

  static sptr make(const types::vector_shapes &input_shapes,
                   TrigAlgorithm algorithm = TrigAlgorithm::AUTO,
                   int max_iterations = 100, double tolerance = 1e-10);

  matrix_sin_pdu(const types::vector_shapes &input_shapes,
                 TrigAlgorithm algorithm = TrigAlgorithm::AUTO,
                 int max_iterations = 100, double tolerance = 1e-10);
};

template <typename Scalar>
class LINALG_API matrix_cos_pdu : public matrix_cos<Scalar>,
                                  public linalg_base_pdu<Scalar> {
public:
  typedef std::shared_ptr<matrix_cos_pdu<Scalar>> sptr;

  static sptr make(const types::vector_shapes &input_shapes,
                   TrigAlgorithm algorithm = TrigAlgorithm::AUTO,
                   int max_iterations = 100, double tolerance = 1e-10);

  matrix_cos_pdu(const types::vector_shapes &input_shapes,
                 TrigAlgorithm algorithm = TrigAlgorithm::AUTO,
                 int max_iterations = 100, double tolerance = 1e-10);
};

template <typename Scalar>
class LINALG_API matrix_tan_pdu : public matrix_tan<Scalar>,
                                  public linalg_base_pdu<Scalar> {
public:
  typedef std::shared_ptr<matrix_tan_pdu<Scalar>> sptr;

  static sptr make(const types::vector_shapes &input_shapes,
                   TrigAlgorithm algorithm = TrigAlgorithm::AUTO,
                   int max_iterations = 100, double tolerance = 1e-10);

  matrix_tan_pdu(const types::vector_shapes &input_shapes,
                 TrigAlgorithm algorithm = TrigAlgorithm::AUTO,
                 int max_iterations = 100, double tolerance = 1e-10);
};

//==============================================================================
// Type aliases
//==============================================================================

// Base classes
typedef matrix_sin<float> matrix_sin_f;
typedef matrix_sin<double> matrix_sin_d;
typedef matrix_sin<std::complex<float>> matrix_sin_cf;
typedef matrix_sin<std::complex<double>> matrix_sin_cd;

typedef matrix_cos<float> matrix_cos_f;
typedef matrix_cos<double> matrix_cos_d;
typedef matrix_cos<std::complex<float>> matrix_cos_cf;
typedef matrix_cos<std::complex<double>> matrix_cos_cd;

typedef matrix_tan<float> matrix_tan_f;
typedef matrix_tan<double> matrix_tan_d;
typedef matrix_tan<std::complex<float>> matrix_tan_cf;
typedef matrix_tan<std::complex<double>> matrix_tan_cd;

// Sync classes
typedef matrix_sin_sync<float> matrix_sin_sync_f;
typedef matrix_sin_sync<double> matrix_sin_sync_d;
typedef matrix_sin_sync<std::complex<float>> matrix_sin_sync_cf;
typedef matrix_sin_sync<std::complex<double>> matrix_sin_sync_cd;

typedef matrix_cos_sync<float> matrix_cos_sync_f;
typedef matrix_cos_sync<double> matrix_cos_sync_d;
typedef matrix_cos_sync<std::complex<float>> matrix_cos_sync_cf;
typedef matrix_cos_sync<std::complex<double>> matrix_cos_sync_cd;

typedef matrix_tan_sync<float> matrix_tan_sync_f;
typedef matrix_tan_sync<double> matrix_tan_sync_d;
typedef matrix_tan_sync<std::complex<float>> matrix_tan_sync_cf;
typedef matrix_tan_sync<std::complex<double>> matrix_tan_sync_cd;

// PDU classes
typedef matrix_sin_pdu<float> matrix_sin_pdu_f;
typedef matrix_sin_pdu<double> matrix_sin_pdu_d;
typedef matrix_sin_pdu<std::complex<float>> matrix_sin_pdu_cf;
typedef matrix_sin_pdu<std::complex<double>> matrix_sin_pdu_cd;

typedef matrix_cos_pdu<float> matrix_cos_pdu_f;
typedef matrix_cos_pdu<double> matrix_cos_pdu_d;
typedef matrix_cos_pdu<std::complex<float>> matrix_cos_pdu_cf;
typedef matrix_cos_pdu<std::complex<double>> matrix_cos_pdu_cd;

typedef matrix_tan_pdu<float> matrix_tan_pdu_f;
typedef matrix_tan_pdu<double> matrix_tan_pdu_d;
typedef matrix_tan_pdu<std::complex<float>> matrix_tan_pdu_cf;
typedef matrix_tan_pdu<std::complex<double>> matrix_tan_pdu_cd;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_TRIG_H */