/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_SOLVE_TRIANGULAR_H
#define INCLUDED_LINALG_SOLVE_TRIANGULAR_H

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
 * \brief Solve triangular linear system Tx = b
 * \ingroup linalg
 *
 * This block efficiently solves triangular linear systems using forward or
 * backward substitution. Triangular systems arise naturally from matrix
 * decompositions (LU, QR, Cholesky) and can be solved much more efficiently
 * than general linear systems.
 *
 * **Mathematical Background:**
 * For a lower triangular matrix L and upper triangular matrix U:
 * - Forward substitution: Solve Lx = b where L[i,j] = 0 for i < j
 * - Backward substitution: Solve Ux = b where U[i,j] = 0 for i > j
 *
 * **Input/Output:**
 * - Input 0: Triangular matrix T [n×n]
 * - Input 1: Right-hand side vector b [n×1] or matrix B [n×k]
 * - Output: Solution x [n×1] or X [n×k]
 *
 * **Performance:**
 * - Time complexity: O(n²) vs O(n³) for general matrix solve
 * - Memory efficient: No pivoting or decomposition needed
 * - Numerically stable for well-conditioned triangular matrices
 *
 * **Applications:**
 * - Post-processing after LU/QR/Cholesky decomposition
 * - Iterative refinement in linear solvers
 * - Solving multiple systems with same triangular matrix
 */
template <typename Scalar>
class LINALG_API solve_triangular : virtual public linalg_base<Scalar> {
public:
  typedef std::shared_ptr<solve_triangular<Scalar>> sptr;

  /*!
   * \brief Triangular matrix type
   */
  enum class triangular_type {
    LOWER = 0, ///< Lower triangular matrix (forward substitution)
    UPPER = 1  ///< Upper triangular matrix (backward substitution)
  };

  /*!
   * \brief Constructor
   * \param shape_matrix Shape of triangular matrix T [n, n]
   * \param shape_rhs Shape of right-hand side b [n, k] (k=1 for vector)
   * \param tri_type Type of triangular matrix (LOWER or UPPER)
   * \param unit_diagonal If true, assume diagonal elements are 1
   */
  solve_triangular(const types::shape &shape_matrix,
                   const types::shape &shape_rhs,
                   triangular_type tri_type = triangular_type::LOWER,
                   bool unit_diagonal = false);

  virtual ~solve_triangular() = default;

protected:
  /*!
   * \brief Perform the triangular solve operation
   */
  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;

  /*!
   * \brief Compute output shapes based on input shapes
   */
  static types::vector_shapes
  compute_output_shapes(const types::shape &shape_matrix,
                        const types::shape &shape_rhs);

private:
  types::shape d_shape_matrix;
  types::shape d_shape_rhs;
  triangular_type d_tri_type;
  bool d_unit_diagonal;
};

/*!
 * \brief Sync block wrapper for solve_triangular
 */
template <typename Scalar>
class LINALG_API solve_triangular_sync : public linalg_base_sync<Scalar>,
                                         public solve_triangular<Scalar> {
public:
  typedef std::shared_ptr<solve_triangular_sync<Scalar>> sptr;

  static sptr make(const types::shape &shape_matrix,
                   const types::shape &shape_rhs,
                   typename solve_triangular<Scalar>::triangular_type tri_type =
                       solve_triangular<Scalar>::triangular_type::LOWER,
                   bool unit_diagonal = false);

  solve_triangular_sync(
      const types::shape &shape_matrix, const types::shape &shape_rhs,
      typename solve_triangular<Scalar>::triangular_type tri_type =
          solve_triangular<Scalar>::triangular_type::LOWER,
      bool unit_diagonal = false);
};

/*!
 * \brief PDU block wrapper for solve_triangular
 */
template <typename Scalar>
class LINALG_API solve_triangular_pdu : public linalg_base_pdu<Scalar>,
                                        public solve_triangular<Scalar> {
public:
  typedef std::shared_ptr<solve_triangular_pdu<Scalar>> sptr;

  static sptr make(const types::shape &shape_matrix,
                   const types::shape &shape_rhs,
                   typename solve_triangular<Scalar>::triangular_type tri_type =
                       solve_triangular<Scalar>::triangular_type::LOWER,
                   bool unit_diagonal = false);

  solve_triangular_pdu(
      const types::shape &shape_matrix, const types::shape &shape_rhs,
      typename solve_triangular<Scalar>::triangular_type tri_type =
          solve_triangular<Scalar>::triangular_type::LOWER,
      bool unit_diagonal = false);
};

// Type aliases for common use cases
typedef solve_triangular_sync<float> solve_triangular_sync_f;
typedef solve_triangular_sync<double> solve_triangular_sync_d;
typedef solve_triangular_sync<std::complex<float>> solve_triangular_sync_cf;
typedef solve_triangular_sync<std::complex<double>> solve_triangular_sync_cd;

typedef solve_triangular_pdu<float> solve_triangular_pdu_f;
typedef solve_triangular_pdu<double> solve_triangular_pdu_d;
typedef solve_triangular_pdu<std::complex<float>> solve_triangular_pdu_cf;
typedef solve_triangular_pdu<std::complex<double>> solve_triangular_pdu_cd;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_SOLVE_TRIANGULAR_H */
