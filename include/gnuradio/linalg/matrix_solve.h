/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_MATRIX_SOLVE_H
#define INCLUDED_LINALG_MATRIX_SOLVE_H

#include <complex>
#include <gnuradio/linalg/api.h>
#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/types.h>
#include <memory>

namespace gr {
namespace linalg {

/*!
 * \brief Solve linear system Ax = b
 * \ingroup linalg
 *
 * This block solves the linear system Ax = b where:
 * - Input 0: Matrix A (coefficient matrix)
 * - Input 1: Vector/Matrix b (right-hand side)
 * - Output: Solution x
 *
 * The block supports:
 * - Square systems: Uses LU decomposition with partial pivoting
 * - Overdetermined systems: Uses least squares (QR decomposition)
 * - Underdetermined systems: Uses minimum norm solution (SVD)
 *
 * For multiple right-hand sides, b can be a matrix where each column
 * is a separate system to solve.
 */
template <typename Scalar>
class LINALG_API matrix_solve : virtual public linalg_base<Scalar> {
public:
  enum class solver_method {
    AUTO = 0,    // Automatically choose based on matrix properties
    LU = 1,      // LU decomposition (square matrices only)
    QR = 2,      // QR decomposition (overdetermined systems)
    SVD = 3,     // Singular Value Decomposition (most general)
    CHOLESKY = 4 // Cholesky decomposition (positive definite only)
  };

  /*!
   * \brief Constructor
   * \param shape_a Shape of coefficient matrix A [rows, cols]
   * \param shape_b Shape of right-hand side b [rows, cols] (cols=1 for vector)
   * \param method Solver method to use (default: AUTO)
   * \param tolerance Tolerance for rank-deficient systems (default: 1e-6)
   */
  matrix_solve(const types::shape &shape_a, const types::shape &shape_b,
               solver_method method = solver_method::AUTO,
               double tolerance = 1e-6);

  virtual ~matrix_solve();

protected:
  /*!
   * \brief Perform the matrix solve operation
   */
  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;

  /*!
   * \brief Compute output shapes based on input shapes
   */
  static types::vector_shapes
  compute_output_shapes(const types::shape &shape_a,
                        const types::shape &shape_b);

private:
  types::shape d_shape_a;
  types::shape d_shape_b;
  solver_method d_method;
  double d_tolerance;

  /*!
   * \brief Choose best solver method based on matrix properties
   */
  solver_method choose_method(int rows_a, int cols_a) const;

  /*!
   * \brief Solve using LU decomposition
   */
  template <typename MatrixType>
  bool solve_lu(const MatrixType &A, const MatrixType &b, MatrixType &x) const;

  /*!
   * \brief Solve using QR decomposition
   */
  template <typename MatrixType>
  bool solve_qr(const MatrixType &A, const MatrixType &b, MatrixType &x) const;

  /*!
   * \brief Solve using SVD
   */
  template <typename MatrixType>
  bool solve_svd(const MatrixType &A, const MatrixType &b, MatrixType &x) const;

  /*!
   * \brief Solve using Cholesky decomposition
   */
  template <typename MatrixType>
  bool solve_cholesky(const MatrixType &A, const MatrixType &b,
                      MatrixType &x) const;
};

// Sync block specialization
template <typename Scalar>
class LINALG_API matrix_solve_sync : virtual public matrix_solve<Scalar>,
                                     virtual public linalg_base_sync<Scalar> {
public:
  typedef std::shared_ptr<matrix_solve_sync<Scalar>> sptr;

  /*!
   * \brief Create a matrix solve sync block
   * \param shape_a Shape of coefficient matrix A
   * \param shape_b Shape of right-hand side b
   * \param method Solver method (default: AUTO)
   * \param tolerance Tolerance for rank-deficient systems
   */
  static sptr make(const types::shape &shape_a, const types::shape &shape_b,
                   typename matrix_solve<Scalar>::solver_method method =
                       matrix_solve<Scalar>::solver_method::AUTO,
                   double tolerance = 1e-6);

protected:
  matrix_solve_sync(const types::shape &shape_a, const types::shape &shape_b,
                    typename matrix_solve<Scalar>::solver_method method =
                        matrix_solve<Scalar>::solver_method::AUTO,
                    double tolerance = 1e-6);
};

// Type definitions for common precisions
typedef matrix_solve_sync<float> matrix_solve_sync_f;
typedef matrix_solve_sync<double> matrix_solve_sync_d;
typedef matrix_solve_sync<std::complex<float>> matrix_solve_sync_cf;
typedef matrix_solve_sync<std::complex<double>> matrix_solve_sync_cd;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_SOLVE_H */