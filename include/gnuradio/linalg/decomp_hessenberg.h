/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_DECOMP_HESSENBERG_H
#define INCLUDED_LINALG_DECOMP_HESSENBERG_H

#include "gnuradio/linalg/linalg_base.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <complex>
#include <gnuradio/linalg/api.h>
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/types.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace gr {
namespace linalg {

/*!
 * \brief Advanced options for Hessenberg decomposition
 */
struct hessenberg_options {
  bool balance_matrix =
      false; ///< Balance matrix before decomposition for numerical stability
  bool extract_reflectors =
      false;                   ///< Extract elementary Householder reflectors
  bool packed_storage = false; ///< Use packed storage for Hessenberg matrix
  bool compute_condition =
      false; ///< Estimate condition number during decomposition
  double balance_tolerance = 1e-8; ///< Tolerance for balancing algorithm
};

/*!
 * \brief Advanced Hessenberg decomposition results
 */
template <typename Scalar>
struct hessenberg_result {
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
      Q; ///< Orthogonal/unitary matrix
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
      H;                       ///< Hessenberg matrix
  Eigen::VectorXd permutation; ///< Permutation from balancing (if used)
  Eigen::VectorXd scale;       ///< Scaling factors from balancing (if used)
  std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>
      reflectors; ///< Householder reflectors (if requested)
  typename Eigen::NumTraits<Scalar>::Real condition_estimate =
      0.0;               ///< Estimated condition number
  bool balanced = false; ///< Whether balancing was applied
};

/*!
 * \brief Hessenberg decomposition algorithm interface with advanced features
 * \ingroup linalg
 *
 * Computes the Hessenberg decomposition A = QHQ^T where:
 * - Q is orthogonal (for real matrices) or unitary (for complex matrices)
 * - H is upper Hessenberg matrix (zero below the first subdiagonal)
 *
 * Advanced features include:
 * - Matrix balancing for improved numerical stability
 * - Householder reflector extraction
 * - Packed storage formats
 * - Condition number estimation
 */
template <typename Scalar>
class LINALG_API hessenberg_algorithm {
public:
  typedef std::shared_ptr<hessenberg_algorithm<Scalar>> sptr;
  virtual ~hessenberg_algorithm() = default;

  /*!
   * \brief Compute basic Hessenberg decomposition (legacy interface)
   * \param input Input square matrix A
   * \param Q Output orthogonal/unitary matrix Q
   * \param H Output Hessenberg matrix H (zero below first subdiagonal)
   */
  virtual void
  decompose(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input,
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &Q,
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &H) = 0;

  /*!
   * \brief Advanced Hessenberg decomposition with additional features
   * \param input Input square matrix A
   * \param options Advanced options for decomposition
   * \return Complete decomposition results with requested features
   */
  virtual hessenberg_result<Scalar> decompose_advanced(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input,
      const hessenberg_options &options = hessenberg_options{}) {
    // Default implementation for compatibility
    hessenberg_result<Scalar> result;
    decompose(input, result.Q, result.H);
    return result;
  }
};

/*!
 * \brief Advanced Hessenberg decomposition using Eigen's implementation with
 * enhancements
 */
template <typename Scalar>
class LINALG_API eigen_hessenberg_advanced
    : public hessenberg_algorithm<Scalar> {
public:
  void
  decompose(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input,
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &Q,
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &H) override;

  hessenberg_result<Scalar> decompose_advanced(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input,
      const hessenberg_options &options = hessenberg_options{}) override;

private:
  /*!
   * \brief Balance matrix using LAPACK-style balancing algorithm
   * \param matrix Input/output matrix to balance
   * \param permutation Output permutation vector
   * \param scale Output scaling factors
   * \param tolerance Balancing tolerance
   * \return true if balancing was performed
   */
  bool
  balance_matrix(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &matrix,
                 Eigen::VectorXd &permutation, Eigen::VectorXd &scale,
                 double tolerance) const;

  /*!
   * \brief Extract Householder reflectors from Hessenberg decomposition
   * \param hess_decomp Eigen Hessenberg decomposition object
   * \return Vector of Householder reflector vectors
   */
  std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>
  extract_householder_reflectors(
      const Eigen::HessenbergDecomposition<
          Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &hess_decomp)
      const;

  /*!
   * \brief Estimate condition number from Hessenberg matrix
   * \param H Hessenberg matrix
   * \return Estimated condition number
   */
  typename Eigen::NumTraits<Scalar>::Real estimate_condition_number(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &H) const;
};

/*!
 * \brief Standard Hessenberg decomposition using Eigen's implementation
 * (legacy)
 */
template <typename Scalar>
class LINALG_API eigen_hessenberg : public hessenberg_algorithm<Scalar> {
public:
  void
  decompose(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input,
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &Q,
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &H) override;
};

/*!
 * \brief Enhanced Hessenberg decomposition base class with advanced features
 */
template <typename Scalar>
class LINALG_API decomp_hessenberg : virtual public linalg_base<Scalar> {
protected:
  std::shared_ptr<hessenberg_algorithm<Scalar>> algorithm_;
  bool compute_q_;             // Whether to compute orthogonal matrix Q
  hessenberg_options options_; // Advanced decomposition options
  mutable hessenberg_result<Scalar>
      last_result_; // Cache last decomposition result

public:
  decomp_hessenberg(
      const types::shape &shape, bool compute_q = true,
      std::shared_ptr<hessenberg_algorithm<Scalar>> algorithm = nullptr,
      const hessenberg_options &options = hessenberg_options{});
  ~decomp_hessenberg() override = default;

  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;

  // Algorithm and option setters/getters
  void set_algorithm(std::shared_ptr<hessenberg_algorithm<Scalar>> algo);
  void set_compute_q(bool compute_q) { compute_q_ = compute_q; }
  void set_options(const hessenberg_options &options) { options_ = options; }

  // Advanced feature accessors
  void set_balance_matrix(bool enable) { options_.balance_matrix = enable; }
  void set_extract_reflectors(bool enable) {
    options_.extract_reflectors = enable;
  }
  void set_packed_storage(bool enable) { options_.packed_storage = enable; }
  void set_compute_condition(bool enable) {
    options_.compute_condition = enable;
  }
  void set_balance_tolerance(double tolerance) {
    options_.balance_tolerance = tolerance;
  }

  // Result accessors (available after operation() call)
  bool get_compute_q() const { return compute_q_; }
  const hessenberg_options &get_options() const { return options_; }
  const hessenberg_result<Scalar> &get_last_result() const {
    return last_result_;
  }

  // Get specific advanced results
  typename Eigen::NumTraits<Scalar>::Real get_condition_estimate() const {
    return last_result_.condition_estimate;
  }
  bool was_matrix_balanced() const { return last_result_.balanced; }
  const Eigen::VectorXd &get_permutation() const {
    return last_result_.permutation;
  }
  const Eigen::VectorXd &get_scale_factors() const {
    return last_result_.scale;
  }
  const std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> &
  get_householder_reflectors() const {
    return last_result_.reflectors;
  }

  static void validate_shape(const types::vector_shapes &input_shapes = {},
                             const types::vector_shapes &output_shapes = {},
                             const std::string &name = "decomp_hessenberg");
  static types::vector_shapes
  compute_output_shapes(const types::vector_shapes &input_shapes);
};

/*!
 * \brief Enhanced synchronous Hessenberg decomposition block with advanced
 * features
 */
template <typename Scalar>
class LINALG_API decomp_hessenberg_sync
    : virtual public decomp_hessenberg<Scalar>,
      virtual public linalg_base_sync<Scalar> {
public:
  typedef std::shared_ptr<decomp_hessenberg_sync<Scalar>> sptr;

  /*!
   * \brief Create a new enhanced Hessenberg decomposition sync block
   * \param shape Shape of the input matrix (must be square)
   * \param compute_q Whether to compute orthogonal matrix Q (default: true)
   * \param algorithm Hessenberg algorithm to use (default: auto-select
   * advanced) \param options Advanced decomposition options
   */
  static sptr
  make(const types::shape &shape, bool compute_q = true,
       std::shared_ptr<hessenberg_algorithm<Scalar>> algorithm = nullptr,
       const hessenberg_options &options = hessenberg_options{});

  decomp_hessenberg_sync(
      const types::shape &shape, bool compute_q = true,
      std::shared_ptr<hessenberg_algorithm<Scalar>> algorithm = nullptr,
      const hessenberg_options &options = hessenberg_options{});
};

/*!
 * \brief Enhanced PDU Hessenberg decomposition block with advanced features
 */
template <typename Scalar>
class LINALG_API decomp_hessenberg_pdu
    : virtual public decomp_hessenberg<Scalar>,
      virtual public linalg_base_pdu<Scalar> {
public:
  typedef std::shared_ptr<decomp_hessenberg_pdu<Scalar>> sptr;

  static sptr
  make(const types::shape &shape, bool compute_q = true,
       std::shared_ptr<hessenberg_algorithm<Scalar>> algorithm = nullptr,
       const hessenberg_options &options = hessenberg_options{});

  decomp_hessenberg_pdu(
      const types::shape &shape, bool compute_q = true,
      std::shared_ptr<hessenberg_algorithm<Scalar>> algorithm = nullptr,
      const hessenberg_options &options = hessenberg_options{});
};

// Convenient type aliases
using decomp_hessenberg_sync_f = decomp_hessenberg_sync<float>;
using decomp_hessenberg_sync_d = decomp_hessenberg_sync<double>;
using decomp_hessenberg_sync_cf = decomp_hessenberg_sync<std::complex<float>>;
using decomp_hessenberg_sync_cd = decomp_hessenberg_sync<std::complex<double>>;

using decomp_hessenberg_pdu_f = decomp_hessenberg_pdu<float>;
using decomp_hessenberg_pdu_d = decomp_hessenberg_pdu<double>;
using decomp_hessenberg_pdu_cf = decomp_hessenberg_pdu<std::complex<float>>;
using decomp_hessenberg_pdu_cd = decomp_hessenberg_pdu<std::complex<double>>;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_DECOMP_HESSENBERG_H */
