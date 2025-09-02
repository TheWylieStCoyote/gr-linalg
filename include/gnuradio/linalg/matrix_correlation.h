/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_MATRIX_CORRELATION_H
#define INCLUDED_LINALG_MATRIX_CORRELATION_H

#include <gnuradio/linalg/api.h>
#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/types.h>
#include <memory>
#include <vector>

namespace gr {
namespace linalg {

enum class CorrelationMethod {
  CORR_PEARSON = 0,  // Standard Pearson correlation
  CORR_SPEARMAN = 1, // Rank-based Spearman correlation
  CORR_KENDALL = 2   // Ordinal Kendall correlation
};

/*!
 * \brief Matrix correlation computation block base class
 * \ingroup linalg
 *
 * This class provides a base implementation for matrix correlation blocks.
 * It computes the correlation matrix from input data matrices.
 * The correlation matrix R is computed as:
 * R[i,j] = cov(X[i], X[j]) / (std(X[i]) * std(X[j]))
 *
 * Where X is the input matrix and cov is covariance, std is standard deviation.
 *
 * \tparam Scalar The scalar type of the matrices (e.g., float, double)
 */
template <typename Scalar>
class LINALG_API matrix_correlation : virtual public linalg_base<Scalar> {
public:
  typedef std::shared_ptr<matrix_correlation<Scalar>> sptr;

  /*!
   * \brief Constructor for matrix_correlation
   * \param name Name of the block
   * \param shape_inputs Vector of input shapes
   * \param shape_outputs Vector of output shapes
   * \param method Correlation method (Pearson, Spearman, Kendall)
   */
  matrix_correlation(
      const std::string &name = "matrix_correlation",
      const types::vector_shapes &shape_inputs = {},
      const types::vector_shapes &shape_outputs = {},
      CorrelationMethod method = CorrelationMethod::CORR_PEARSON);

  virtual ~matrix_correlation();

  /*! \brief Set correlation method */
  void set_method(CorrelationMethod method);
  /*! \brief Get correlation method */
  CorrelationMethod get_method() const;

  /*! \brief Perform the matrix correlation operation
   * \param input_matrices Vector of input matrices
   * \param output_matrices Vector of output matrices
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
                             const std::string &name = "matrix_correlation");

  /*! \brief Calculate total number of elements in multiple shapes */
  static std::vector<size_t> compute_sizes(const types::vector_shapes &shapes);

private:
  int d_method; ///< Correlation method (0=Pearson, 1=Spearman, 2=Kendall)
};

/*!
 * \brief Matrix correlation synchronous block
 * \ingroup linalg
 *
 * This class provides a synchronous block for matrix correlation computation.
 * It inherits from matrix_correlation and implements the linalg_base_sync
 * interface for synchronous operation.
 *
 * \tparam Scalar The scalar type of the matrices (e.g., float, double)
 */
template <typename Scalar>
class LINALG_API matrix_correlation_sync
    : virtual public matrix_correlation<Scalar>,
      virtual public linalg_base_sync<Scalar> {
public:
  typedef std::shared_ptr<matrix_correlation_sync<Scalar>> sptr;

  /*! \brief Return a shared_ptr to a new instance of
   * linalg::matrix_correlation_sync \param shape Shape of the input matrix
   * [samples, features] \param method Correlation method (0=Pearson,
   * 1=Spearman, 2=Kendall)
   */
  static sptr make(const types::shape &shape,
                   CorrelationMethod method = CorrelationMethod::CORR_PEARSON);

protected:
  matrix_correlation_sync(const types::shape &shape, CorrelationMethod method);
};

/*!
 * \brief Matrix correlation PDU block
 * \ingroup linalg
 *
 * This class provides a PDU (Protocol Data Unit) block for matrix correlation
 * computation. It inherits from matrix_correlation and implements the
 * linalg_base_pdu interface for PDU operation.
 *
 * \tparam Scalar The scalar type of the matrices (e.g., float, double)
 */
template <typename Scalar>
class LINALG_API matrix_correlation_pdu
    : virtual public matrix_correlation<Scalar>,
      virtual public linalg_base_pdu<Scalar> {
public:
  typedef std::shared_ptr<matrix_correlation_pdu<Scalar>> sptr;

  /*! \brief Return a shared_ptr to a new instance of
   * linalg::matrix_correlation_pdu \param shape Shape of the input matrix
   * [samples, features] \param method Correlation method (0=Pearson,
   * 1=Spearman, 2=Kendall)
   */
  static sptr make(const types::shape &shape,
                   CorrelationMethod method = CorrelationMethod::CORR_PEARSON);

protected:
  matrix_correlation_pdu(const types::shape &shape, CorrelationMethod method);
};

// Type aliases for Python bindings
using matrix_correlation_sync_f = matrix_correlation_sync<float>;
using matrix_correlation_sync_d = matrix_correlation_sync<double>;
using matrix_correlation_sync_cf = matrix_correlation_sync<gr_complex>;
using matrix_correlation_sync_cd =
    matrix_correlation_sync<std::complex<double>>;

using matrix_correlation_pdu_f = matrix_correlation_pdu<float>;
using matrix_correlation_pdu_d = matrix_correlation_pdu<double>;
using matrix_correlation_pdu_cf = matrix_correlation_pdu<gr_complex>;
using matrix_correlation_pdu_cd = matrix_correlation_pdu<std::complex<double>>;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_CORRELATION_H */
