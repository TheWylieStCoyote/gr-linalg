/* -*- c++ -*- */
/*
 * Copyright 2025 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_SOLVE_ITERATIVE_H
#define INCLUDED_LINALG_SOLVE_ITERATIVE_H

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <complex>
#include <gnuradio/linalg/api.h>
#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/types.h>
#include <memory>

namespace gr {
namespace linalg {

/*!
 * \brief Iterative solver methods enumeration
 * \ingroup linalg
 */
enum class iterative_method {
  AUTO = 0, //!< Automatic method selection based on matrix properties
  CG,       //!< Conjugate Gradient (for symmetric positive definite)
  BICGSTAB, //!< Biconjugate Gradient Stabilized (general matrices)
  GMRES,    //!< Generalized Minimal Residual (general matrices)
  LSCG      //!< Least Squares Conjugate Gradient (rectangular matrices)
};

// Stream operator for iterative_method (needed for Boost Test)
inline std::ostream &operator<<(std::ostream &os,
                                const iterative_method &method) {
  switch (method) {
  case iterative_method::AUTO:
    return os << "AUTO";
  case iterative_method::CG:
    return os << "CG";
  case iterative_method::BICGSTAB:
    return os << "BICGSTAB";
  case iterative_method::GMRES:
    return os << "GMRES";
  case iterative_method::LSCG:
    return os << "LSCG";
  default:
    return os << "UNKNOWN(" << static_cast<int>(method) << ")";
  }
}

/*!
 * \brief Iterative Linear System Solver
 * \ingroup linalg
 *
 * This block solves linear systems Ax = b using iterative methods, which are
 * particularly effective for large sparse systems where direct methods would
 * be computationally expensive.
 *
 * ## Mathematical Methods:
 *
 * **1. Conjugate Gradient (CG)**
 * - Requirements: A must be symmetric positive definite
 * - Convergence: O(√κ) iterations where κ is condition number
 * - Memory: O(n) storage
 * - Best for: Well-conditioned symmetric systems
 *
 * **2. Biconjugate Gradient Stabilized (BiCGStab)**
 * - Requirements: A can be any square matrix
 * - Convergence: Often faster than GMRES for certain problems
 * - Memory: O(n) storage
 * - Best for: General non-symmetric systems
 *
 * **3. Generalized Minimal Residual (GMRES)**
 * - Requirements: A can be any square matrix
 * - Convergence: Guaranteed for any invertible matrix
 * - Memory: O(mn) where m is restart parameter
 * - Best for: Systems where BiCGStab fails to converge
 *
 * **4. Least Squares Conjugate Gradient (LSCG)**
 * - Requirements: A can be rectangular (m×n)
 * - Solves: minimize ||Ax - b||₂ iteratively
 * - Memory: O(n) storage
 * - Best for: Overdetermined systems (m > n)
 *
 * ## Applications:
 * - Large sparse linear systems (finite element methods)
 * - Image reconstruction and computed tomography
 * - Signal processing on graphs and networks
 * - Optimization problems (quadratic programming)
 * - Machine learning (ridge regression, LASSO)
 *
 * ## Performance Considerations:
 * - Preconditioning can dramatically improve convergence
 * - Initial guess x₀ affects iteration count
 * - Tolerance affects accuracy vs speed tradeoff
 * - Method selection is crucial for performance
 */

/*!
 * \brief Iterative solver algorithm interface
 */
template <typename Scalar>
class LINALG_API iterative_algorithm {
public:
  typedef std::shared_ptr<iterative_algorithm<Scalar>> sptr;
  virtual ~iterative_algorithm() = default;

  /*!
   * \brief Solve Ax = b iteratively
   * \param A Coefficient matrix (can be rectangular for LSCG)
   * \param b Right-hand side vector
   * \param x Solution vector (input: initial guess, output: solution)
   * \param max_iterations Maximum number of iterations
   * \param tolerance Convergence tolerance
   * \return Number of iterations performed (-1 if failed to converge)
   */
  virtual int
  solve(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &A,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &b,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &x, int max_iterations,
        typename Eigen::NumTraits<Scalar>::Real tolerance) = 0;
};

/*!
 * \brief Conjugate Gradient solver implementation
 */
template <typename Scalar>
class LINALG_API cg_algorithm : public iterative_algorithm<Scalar> {
public:
  int solve(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &A,
            const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &b,
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &x, int max_iterations,
            typename Eigen::NumTraits<Scalar>::Real tolerance) override;
};

/*!
 * \brief BiCGStab solver implementation
 */
template <typename Scalar>
class LINALG_API bicgstab_algorithm : public iterative_algorithm<Scalar> {
public:
  int solve(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &A,
            const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &b,
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &x, int max_iterations,
            typename Eigen::NumTraits<Scalar>::Real tolerance) override;
};

/*!
 * \brief GMRES solver implementation
 */
template <typename Scalar>
class LINALG_API gmres_algorithm : public iterative_algorithm<Scalar> {
private:
  int restart_; //!< GMRES restart parameter
public:
  explicit gmres_algorithm(int restart = 30) : restart_(restart) {}

  int solve(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &A,
            const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &b,
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &x, int max_iterations,
            typename Eigen::NumTraits<Scalar>::Real tolerance) override;

  void set_restart(int restart) { restart_ = restart; }
  int get_restart() const { return restart_; }
};

/*!
 * \brief LSCG (Least Squares Conjugate Gradient) solver implementation
 */
template <typename Scalar>
class LINALG_API lscg_algorithm : public iterative_algorithm<Scalar> {
public:
  int solve(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &A,
            const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &b,
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &x, int max_iterations,
            typename Eigen::NumTraits<Scalar>::Real tolerance) override;
};

/*!
 * \brief Iterative linear system solver base class
 */
template <typename Scalar>
class LINALG_API solve_iterative : virtual public linalg_base<Scalar> {
protected:
  types::shape shape_a_;    //!< Shape of coefficient matrix A
  types::shape shape_b_;    //!< Shape of right-hand side b
  iterative_method method_; //!< Iterative method to use
  int max_iterations_;      //!< Maximum iterations
  typename Eigen::NumTraits<Scalar>::Real tolerance_; //!< Convergence tolerance
  bool use_initial_guess_; //!< Use input x as initial guess
  std::shared_ptr<iterative_algorithm<Scalar>>
      algorithm_; //!< Algorithm implementation

public:
  solve_iterative(
      const types::shape &shape_a, const types::shape &shape_b,
      iterative_method method = iterative_method::AUTO,
      int max_iterations = 1000,
      typename Eigen::NumTraits<Scalar>::Real tolerance = 1e-6,
      bool use_initial_guess = false,
      std::shared_ptr<iterative_algorithm<Scalar>> algorithm = nullptr);

  ~solve_iterative() override = default;

  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;

  // Configuration methods
  void set_method(iterative_method method);
  void set_max_iterations(int max_iterations) {
    max_iterations_ = max_iterations;
  }
  void set_tolerance(typename Eigen::NumTraits<Scalar>::Real tolerance) {
    tolerance_ = tolerance;
  }
  void set_use_initial_guess(bool use_initial_guess) {
    use_initial_guess_ = use_initial_guess;
  }
  void set_algorithm(std::shared_ptr<iterative_algorithm<Scalar>> algorithm);

  // Accessor methods
  iterative_method get_method() const { return method_; }
  int get_max_iterations() const { return max_iterations_; }
  typename Eigen::NumTraits<Scalar>::Real get_tolerance() const {
    return tolerance_;
  }
  bool get_use_initial_guess() const { return use_initial_guess_; }

  static void validate_shape(const types::vector_shapes &input_shapes = {},
                             const types::vector_shapes &output_shapes = {},
                             const std::string &name = "solve_iterative");
  static types::vector_shapes
  compute_output_shapes(const types::vector_shapes &input_shapes);
};

/*!
 * \brief Synchronous iterative solver block
 */
template <typename Scalar>
class LINALG_API solve_iterative_sync
    : virtual public solve_iterative<Scalar>,
      virtual public linalg_base_sync<Scalar> {
public:
  typedef std::shared_ptr<solve_iterative_sync<Scalar>> sptr;

  /*!
   * \brief Create iterative solver sync block
   * \param shape_a Shape of coefficient matrix A [m, n]
   * \param shape_b Shape of RHS vector b [m, 1] (or [m, k] for multiple RHS)
   * \param method Iterative method (AUTO, CG, BICGSTAB, GMRES, LSCG)
   * \param max_iterations Maximum number of iterations
   * \param tolerance Convergence tolerance
   * \param use_initial_guess Use input solution as initial guess
   * \param algorithm Custom algorithm implementation
   */
  static sptr
  make(const types::shape &shape_a, const types::shape &shape_b,
       iterative_method method = iterative_method::AUTO,
       int max_iterations = 1000,
       typename Eigen::NumTraits<Scalar>::Real tolerance = 1e-6,
       bool use_initial_guess = false,
       std::shared_ptr<iterative_algorithm<Scalar>> algorithm = nullptr);

  solve_iterative_sync(
      const types::shape &shape_a, const types::shape &shape_b,
      iterative_method method = iterative_method::AUTO,
      int max_iterations = 1000,
      typename Eigen::NumTraits<Scalar>::Real tolerance = 1e-6,
      bool use_initial_guess = false,
      std::shared_ptr<iterative_algorithm<Scalar>> algorithm = nullptr);
};

/*!
 * \brief PDU iterative solver block
 */
template <typename Scalar>
class LINALG_API solve_iterative_pdu : virtual public solve_iterative<Scalar>,
                                       virtual public linalg_base_pdu<Scalar> {
public:
  typedef std::shared_ptr<solve_iterative_pdu<Scalar>> sptr;

  static sptr
  make(const types::shape &shape_a, const types::shape &shape_b,
       iterative_method method = iterative_method::AUTO,
       int max_iterations = 1000,
       typename Eigen::NumTraits<Scalar>::Real tolerance = 1e-6,
       bool use_initial_guess = false,
       std::shared_ptr<iterative_algorithm<Scalar>> algorithm = nullptr);

  solve_iterative_pdu(
      const types::shape &shape_a, const types::shape &shape_b,
      iterative_method method = iterative_method::AUTO,
      int max_iterations = 1000,
      typename Eigen::NumTraits<Scalar>::Real tolerance = 1e-6,
      bool use_initial_guess = false,
      std::shared_ptr<iterative_algorithm<Scalar>> algorithm = nullptr);
};

// Convenient type aliases
using solve_iterative_sync_f = solve_iterative_sync<float>;
using solve_iterative_sync_d = solve_iterative_sync<double>;
using solve_iterative_sync_cf = solve_iterative_sync<std::complex<float>>;
using solve_iterative_sync_cd = solve_iterative_sync<std::complex<double>>;

using solve_iterative_pdu_f = solve_iterative_pdu<float>;
using solve_iterative_pdu_d = solve_iterative_pdu<double>;
using solve_iterative_pdu_cf = solve_iterative_pdu<std::complex<float>>;
using solve_iterative_pdu_cd = solve_iterative_pdu<std::complex<double>>;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_SOLVE_ITERATIVE_H */