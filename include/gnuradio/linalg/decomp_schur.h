/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_DECOMP_SCHUR_H
#define INCLUDED_LINALG_DECOMP_SCHUR_H

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
 * \brief Advanced options for Schur decomposition
 */
struct schur_options {
  bool balance_matrix =
      false; ///< Balance matrix before decomposition for numerical stability
  bool extract_eigenvalues = false; ///< Extract eigenvalues from Schur form
  bool sort_eigenvalues = false; ///< Sort eigenvalues by magnitude or criteria
  bool compute_condition =
      false; ///< Estimate condition number during decomposition
  bool use_complex_schur =
      false; ///< Force complex Schur form even for real matrices
  double balance_tolerance = 1e-8; ///< Tolerance for balancing algorithm
  double sort_tolerance = 1e-12;   ///< Tolerance for eigenvalue sorting
};

/*!
 * \brief Advanced Schur decomposition results
 */
template <typename Scalar>
struct schur_result {
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
      Q; ///< Orthogonal/unitary matrix
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> T; ///< Schur matrix
  Eigen::VectorXd permutation; ///< Permutation from balancing (if used)
  Eigen::VectorXd scale;       ///< Scaling factors from balancing (if used)
  Eigen::Vector<std::complex<typename Eigen::NumTraits<Scalar>::Real>,
                Eigen::Dynamic>
      eigenvalues; ///< Extracted eigenvalues (if requested)
  typename Eigen::NumTraits<Scalar>::Real condition_estimate =
      0.0;                       ///< Estimated condition number
  bool balanced = false;         ///< Whether balancing was applied
  bool is_complex_schur = false; ///< Whether result is in complex Schur form
};

/*!
 * \brief Schur decomposition algorithm interface with advanced features
 * \ingroup linalg
 *
 * Computes the Schur decomposition A = QTQ^T where:
 * - Q is orthogonal (for real matrices) or unitary (for complex matrices)
 * - T is quasi-upper triangular (real Schur) or upper triangular (complex
 * Schur)
 *
 * Advanced features include:
 * - Matrix balancing for improved numerical stability
 * - Eigenvalue extraction and sorting
 * - Condition number estimation
 * - Forced complex Schur forms
 */
template <typename Scalar>
class LINALG_API schur_algorithm {
public:
  typedef std::shared_ptr<schur_algorithm<Scalar>> sptr;
  virtual ~schur_algorithm() = default;

  /*!
   * \brief Compute basic Schur decomposition (legacy interface)
   * \param input Input square matrix A
   * \param Q Output orthogonal/unitary matrix Q
   * \param T Output Schur matrix T (quasi-upper triangular for real, upper
   * triangular for complex)
   */
  virtual void
  decompose(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input,
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &Q,
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &T) = 0;

  /*!
   * \brief Advanced Schur decomposition with additional features
   * \param input Input square matrix A
   * \param options Advanced options for decomposition
   * \return Complete decomposition results with requested features
   */
  virtual schur_result<Scalar> decompose_advanced(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input,
      const schur_options &options = schur_options{}) {
    // Default implementation for compatibility
    schur_result<Scalar> result;
    decompose(input, result.Q, result.T);
    return result;
  }
};

/*!
 * \brief Advanced Schur decomposition using Eigen's implementation with
 * enhancements
 */
template <typename Scalar>
class LINALG_API eigen_schur_advanced : public schur_algorithm<Scalar> {
public:
  void
  decompose(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input,
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &Q,
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &T) override;

  schur_result<Scalar> decompose_advanced(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input,
      const schur_options &options = schur_options{}) override;

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
   * \brief Extract eigenvalues from Schur form matrix T
   * \param T Schur form matrix
   * \return Vector of eigenvalues
   */
  Eigen::Vector<std::complex<typename Eigen::NumTraits<Scalar>::Real>,
                Eigen::Dynamic>
  extract_eigenvalues_from_schur(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &T) const;

  /*!
   * \brief Sort eigenvalues and corresponding Schur vectors
   * \param Q Orthogonal/unitary matrix
   * \param T Schur form matrix
   * \param eigenvalues Eigenvalues to sort
   * \param tolerance Sorting tolerance
   * \return true if sorting was performed
   */
  bool sort_schur_decomposition(
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &Q,
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &T,
      Eigen::Vector<std::complex<typename Eigen::NumTraits<Scalar>::Real>,
                    Eigen::Dynamic> &eigenvalues,
      double tolerance) const;

  /*!
   * \brief Estimate condition number from Schur matrix
   * \param T Schur form matrix
   * \return Estimated condition number
   */
  typename Eigen::NumTraits<Scalar>::Real estimate_condition_number(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &T) const;
};

/*!
 * \brief Real Schur decomposition using Eigen's implementation (legacy)
 */
template <typename Scalar>
class LINALG_API eigen_real_schur : public schur_algorithm<Scalar> {
public:
  void
  decompose(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input,
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &Q,
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &T) override;
};

/*!
 * \brief Complex Schur decomposition using Eigen's implementation (legacy)
 */
template <typename Scalar>
class LINALG_API eigen_complex_schur : public schur_algorithm<Scalar> {
public:
  void
  decompose(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input,
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &Q,
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &T) override;
};

/*!
 * \brief Enhanced Schur decomposition base class with advanced features
 */
template <typename Scalar>
class LINALG_API decomp_schur : virtual public linalg_base<Scalar> {
protected:
  std::shared_ptr<schur_algorithm<Scalar>> algorithm_;
  bool compute_u_;        // Whether to compute orthogonal matrix Q
  schur_options options_; // Advanced decomposition options
  mutable schur_result<Scalar> last_result_; // Cache last decomposition result

public:
  decomp_schur(const types::shape &shape, bool compute_u = true,
               std::shared_ptr<schur_algorithm<Scalar>> algorithm = nullptr,
               const schur_options &options = schur_options{});
  ~decomp_schur() override = default;

  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;

  // Algorithm and option setters/getters
  void set_algorithm(std::shared_ptr<schur_algorithm<Scalar>> algo);
  void set_compute_u(bool compute_u) { compute_u_ = compute_u; }
  void set_options(const schur_options &options) { options_ = options; }

  // Advanced feature accessors
  void set_balance_matrix(bool enable) { options_.balance_matrix = enable; }
  void set_extract_eigenvalues(bool enable) {
    options_.extract_eigenvalues = enable;
  }
  void set_sort_eigenvalues(bool enable) { options_.sort_eigenvalues = enable; }
  void set_compute_condition(bool enable) {
    options_.compute_condition = enable;
  }
  void set_use_complex_schur(bool enable) {
    options_.use_complex_schur = enable;
  }
  void set_balance_tolerance(double tolerance) {
    options_.balance_tolerance = tolerance;
  }
  void set_sort_tolerance(double tolerance) {
    options_.sort_tolerance = tolerance;
  }

  // Result accessors (available after operation() call)
  bool get_compute_u() const { return compute_u_; }
  const schur_options &get_options() const { return options_; }
  const schur_result<Scalar> &get_last_result() const { return last_result_; }

  // Get specific advanced results
  typename Eigen::NumTraits<Scalar>::Real get_condition_estimate() const {
    return last_result_.condition_estimate;
  }
  bool was_matrix_balanced() const { return last_result_.balanced; }
  bool is_complex_schur_form() const { return last_result_.is_complex_schur; }
  const Eigen::VectorXd &get_permutation() const {
    return last_result_.permutation;
  }
  const Eigen::VectorXd &get_scale_factors() const {
    return last_result_.scale;
  }
  const Eigen::Vector<std::complex<typename Eigen::NumTraits<Scalar>::Real>,
                      Eigen::Dynamic> &
  get_eigenvalues() const {
    return last_result_.eigenvalues;
  }

  static void validate_shape(const types::vector_shapes &input_shapes = {},
                             const types::vector_shapes &output_shapes = {},
                             const std::string &name = "decomp_schur");
  static types::vector_shapes
  compute_output_shapes(const types::vector_shapes &input_shapes);
};

/*!
 * \brief Enhanced synchronous Schur decomposition block with advanced features
 */
template <typename Scalar>
class LINALG_API decomp_schur_sync : virtual public decomp_schur<Scalar>,
                                     virtual public linalg_base_sync<Scalar> {
public:
  typedef std::shared_ptr<decomp_schur_sync<Scalar>> sptr;

  /*!
   * \brief Create a new enhanced Schur decomposition sync block
   * \param shape Shape of the input matrix (must be square)
   * \param compute_u Whether to compute orthogonal matrix Q (default: true)
   * \param algorithm Schur algorithm to use (default: auto-select advanced)
   * \param options Advanced decomposition options
   */
  static sptr make(const types::shape &shape, bool compute_u = true,
                   std::shared_ptr<schur_algorithm<Scalar>> algorithm = nullptr,
                   const schur_options &options = schur_options{});

  decomp_schur_sync(
      const types::shape &shape, bool compute_u = true,
      std::shared_ptr<schur_algorithm<Scalar>> algorithm = nullptr,
      const schur_options &options = schur_options{});
};

/*!
 * \brief Enhanced PDU Schur decomposition block with advanced features
 */
template <typename Scalar>
class LINALG_API decomp_schur_pdu : virtual public decomp_schur<Scalar>,
                                    virtual public linalg_base_pdu<Scalar> {
public:
  typedef std::shared_ptr<decomp_schur_pdu<Scalar>> sptr;

  static sptr make(const types::shape &shape, bool compute_u = true,
                   std::shared_ptr<schur_algorithm<Scalar>> algorithm = nullptr,
                   const schur_options &options = schur_options{});

  decomp_schur_pdu(const types::shape &shape, bool compute_u = true,
                   std::shared_ptr<schur_algorithm<Scalar>> algorithm = nullptr,
                   const schur_options &options = schur_options{});
};

// Convenient type aliases
using decomp_schur_sync_f = decomp_schur_sync<float>;
using decomp_schur_sync_d = decomp_schur_sync<double>;
using decomp_schur_sync_cf = decomp_schur_sync<std::complex<float>>;
using decomp_schur_sync_cd = decomp_schur_sync<std::complex<double>>;

using decomp_schur_pdu_f = decomp_schur_pdu<float>;
using decomp_schur_pdu_d = decomp_schur_pdu<double>;
using decomp_schur_pdu_cf = decomp_schur_pdu<std::complex<float>>;
using decomp_schur_pdu_cd = decomp_schur_pdu<std::complex<double>>;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_DECOMP_SCHUR_H */
