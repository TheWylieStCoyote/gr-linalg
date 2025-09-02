/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_MATRIX_EIGENVALUES_H
#define INCLUDED_LINALG_MATRIX_EIGENVALUES_H

#include <gnuradio/linalg/api.h>
#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/types.h>
#include <memory>
#include <vector>

namespace gr {
namespace linalg {

/*!
 * \brief Matrix eigenvalues computation block base class
 * \ingroup linalg
 *
 * This class provides eigenvalues extraction from input matrices.
 * Unlike decomp_eigen which returns both eigenvalues and eigenvectors,
 * this block only computes and returns eigenvalues for efficiency.
 *
 * Mathematical Background:
 * For a square matrix A, eigenvalues λ are solutions to:
 * det(A - λI) = 0
 *
 * The eigenvalues provide information about:
 * - Matrix stability (real parts of eigenvalues)
 * - Oscillatory behavior (imaginary parts)
 * - Matrix conditioning (ratio of largest/smallest eigenvalues)
 * - Matrix determinant (product of eigenvalues)
 * - Matrix trace (sum of eigenvalues)
 *
 * Computational Notes:
 * - Uses Eigen's efficient eigenvalue solver
 * - For Hermitian matrices, eigenvalues are always real
 * - For general matrices, eigenvalues may be complex
 * - Output is sorted by magnitude (largest first)
 *
 * Applications:
 * - Stability analysis of dynamical systems
 * - Principal component analysis (PCA)
 * - Frequency domain analysis
 * - Matrix conditioning assessment
 * - Control system design
 *
 * \tparam Scalar The scalar type of the matrices (e.g., float, double)
 */
template <typename Scalar>
class LINALG_API matrix_eigenvalues : virtual public linalg_base<Scalar> {
public:
  typedef std::shared_ptr<matrix_eigenvalues<Scalar>> sptr;

  /*!
   * \brief Constructor for matrix_eigenvalues
   * \param name Name of the block
   * \param shape_inputs Vector of input shapes
   * \param shape_outputs Vector of output shapes
   * \param sort_by_magnitude Whether to sort eigenvalues by magnitude
   */
  matrix_eigenvalues(const std::string &name = "matrix_eigenvalues",
                     const types::vector_shapes &shape_inputs = {},
                     const types::vector_shapes &shape_outputs = {},
                     bool sort_by_magnitude = true);

  virtual ~matrix_eigenvalues();

  /*! \brief Set eigenvalue sorting mode */
  void set_sort_by_magnitude(bool sort);
  /*! \brief Get eigenvalue sorting mode */
  bool get_sort_by_magnitude() const;

  /*! \brief Perform the matrix eigenvalues operation
   * \param input_matrices Vector of input matrices
   * \param output_matrices Vector of output eigenvalue vectors
   * \return OperationReturn status code
   */
  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;

  /*! \brief Compute output shapes based on input shapes */
  static types::vector_shapes
  compute_output_shapes(const types::vector_shapes &input_shapes);

  /*! \brief Validate input and output shapes */
  static void validate_shape(const types::vector_shapes &input_shapes,
                             const types::vector_shapes &output_shapes = {},
                             const std::string &name = "matrix_eigenvalues");

  /*! \brief Calculate total number of elements in multiple shapes */
  static std::vector<size_t> compute_sizes(const types::vector_shapes &shapes);

private:
  bool d_sort_by_magnitude; ///< Whether to sort eigenvalues by magnitude

  /*! \brief Internal eigenvalue computation */
  template <typename EigenvalueType>
  void compute_eigenvalues_impl(
      const Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic,
                                           Eigen::Dynamic>> &input_matrix,
      Eigen::Map<Eigen::Matrix<EigenvalueType, Eigen::Dynamic, 1>>
          &eigenvalues_vector) const;
};

/*!
 * \brief Matrix eigenvalues synchronous block
 * \ingroup linalg
 *
 * This class provides a synchronous block for matrix eigenvalues computation.
 * It inherits from matrix_eigenvalues and implements the linalg_base_sync
 * interface for synchronous operation.
 *
 * \tparam Scalar The scalar type of the matrices (e.g., float, double)
 */
template <typename Scalar>
class LINALG_API matrix_eigenvalues_sync
    : virtual public matrix_eigenvalues<Scalar>,
      virtual public linalg_base_sync<Scalar> {
public:
  typedef std::shared_ptr<matrix_eigenvalues_sync<Scalar>> sptr;

  /*! \brief Return a shared_ptr to a new instance of
   * linalg::matrix_eigenvalues_sync \param shape Shape of the input square
   * matrix [n, n] \param sort_by_magnitude Whether to sort eigenvalues by
   * magnitude (default: true)
   */
  static sptr make(const types::shape &shape, bool sort_by_magnitude = true);

protected:
  matrix_eigenvalues_sync(const types::shape &shape, bool sort_by_magnitude);
};

/*!
 * \brief Matrix eigenvalues PDU block
 * \ingroup linalg
 *
 * This class provides a PDU (Protocol Data Unit) block for matrix eigenvalues
 * computation. It inherits from matrix_eigenvalues and implements the
 * linalg_base_pdu interface for PDU operation.
 *
 * \tparam Scalar The scalar type of the matrices (e.g., float, double)
 */
template <typename Scalar>
class LINALG_API matrix_eigenvalues_pdu
    : virtual public matrix_eigenvalues<Scalar>,
      virtual public linalg_base_pdu<Scalar> {
public:
  typedef std::shared_ptr<matrix_eigenvalues_pdu<Scalar>> sptr;

  /*! \brief Return a shared_ptr to a new instance of
   * linalg::matrix_eigenvalues_pdu \param shape Shape of the input square
   * matrix [n, n] \param sort_by_magnitude Whether to sort eigenvalues by
   * magnitude (default: true)
   */
  static sptr make(const types::shape &shape, bool sort_by_magnitude = true);

protected:
  matrix_eigenvalues_pdu(const types::shape &shape, bool sort_by_magnitude);
};

// Type aliases for Python bindings
using matrix_eigenvalues_sync_f = matrix_eigenvalues_sync<float>;
using matrix_eigenvalues_sync_d = matrix_eigenvalues_sync<double>;
using matrix_eigenvalues_sync_cf = matrix_eigenvalues_sync<gr_complex>;
using matrix_eigenvalues_sync_cd =
    matrix_eigenvalues_sync<std::complex<double>>;

using matrix_eigenvalues_pdu_f = matrix_eigenvalues_pdu<float>;
using matrix_eigenvalues_pdu_d = matrix_eigenvalues_pdu<double>;
using matrix_eigenvalues_pdu_cf = matrix_eigenvalues_pdu<gr_complex>;
using matrix_eigenvalues_pdu_cd = matrix_eigenvalues_pdu<std::complex<double>>;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_EIGENVALUES_H */