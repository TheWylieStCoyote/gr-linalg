/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_MATRIX_COVARIANCE_H
#define INCLUDED_LINALG_MATRIX_COVARIANCE_H

#include <Eigen/Dense>
#include <gnuradio/linalg/api.h>
#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/types.h>
#include <gnuradio/sync_block.h>

namespace gr {
namespace linalg {

/*!
 * \brief Matrix covariance computation block base class
 * \ingroup linalg
 *
 * This class provides a base implementation for matrix covariance blocks.
 * It computes the sample covariance matrix from input data matrices.
 * The covariance matrix C is computed as:
 * C = (1/N) * X^T * X for unbiased=false (sample covariance)
 * C = (1/(N-1)) * X^T * X for unbiased=true (unbiased covariance)
 *
 * Where X is the input matrix (samples x features) and N is the number of
 * samples.
 *
 * \tparam Scalar The scalar type of the matrices (e.g., float, double)
 */
template <typename Scalar>
class LINALG_API matrix_covariance : virtual public linalg_base<Scalar> {
public:
  typedef std::shared_ptr<matrix_covariance<Scalar>> sptr;

  /*!
   * \brief Constructor for matrix_covariance
   * \param name Name of the block
   * \param shape_inputs Vector of input shapes
   * \param shape_outputs Vector of output shapes
   * \param unbiased Whether to use unbiased estimator (divide by N-1 instead of
   * N) \param center Whether to subtract mean before computing covariance
   */
  matrix_covariance(const std::string &name = "matrix_covariance",
                    const types::vector_shapes &shape_inputs = {},
                    const types::vector_shapes &shape_outputs = {},
                    bool unbiased = true, bool center = true);

  virtual ~matrix_covariance();

  /*! \brief Set whether to use unbiased estimator */
  void set_unbiased(bool unbiased);
  /*! \brief Get whether using unbiased estimator */
  bool get_unbiased() const;

  /*! \brief Set whether to center data (subtract mean) */
  void set_center(bool center);
  /*! \brief Get whether centering data */
  bool get_center() const;

  /*! \brief Perform the matrix covariance operation
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
                             const std::string &name = "matrix_covariance");

  /*! \brief Calculate total number of elements in multiple shapes */
  static std::vector<size_t> compute_sizes(const types::vector_shapes &shapes);

private:
  bool d_unbiased; ///< Whether to use unbiased estimator
  bool d_center;   ///< Whether to center the data
};

/*!
 * \brief Matrix covariance synchronous block
 * \ingroup linalg
 *
 * This class provides a synchronous block for matrix covariance computation.
 * It inherits from matrix_covariance and implements the linalg_base_sync
 * interface for synchronous operation.
 *
 * \tparam Scalar The scalar type of the matrices (e.g., float, double)
 */
template <typename Scalar>
class LINALG_API matrix_covariance_sync
    : virtual public matrix_covariance<Scalar>,
      virtual public linalg_base_sync<Scalar> {
public:
  typedef std::shared_ptr<matrix_covariance_sync<Scalar>> sptr;

  /*! \brief Return a shared_ptr to a new instance of
   * linalg::matrix_covariance_sync \param shape Shape of the input matrix
   * [samples, features] \param unbiased Whether to use unbiased estimator
   * (default: true) \param center Whether to center data by subtracting mean
   * (default: true)
   */
  static sptr make(const types::shape &shape, bool unbiased = true,
                   bool center = true);

protected:
  matrix_covariance_sync(const types::shape &shape, bool unbiased, bool center);
};

/*!
 * \brief Matrix covariance PDU block
 * \ingroup linalg
 *
 * This class provides a PDU (Protocol Data Unit) block for matrix covariance
 * computation. It inherits from matrix_covariance and implements the
 * linalg_base_pdu interface for PDU operation.
 *
 * \tparam Scalar The scalar type of the matrices (e.g., float, double)
 */
template <typename Scalar>
class LINALG_API matrix_covariance_pdu
    : virtual public matrix_covariance<Scalar>,
      virtual public linalg_base_pdu<Scalar> {
public:
  typedef std::shared_ptr<matrix_covariance_pdu<Scalar>> sptr;

  /*! \brief Return a shared_ptr to a new instance of
   * linalg::matrix_covariance_pdu \param shape Shape of the input matrix
   * [samples, features] \param unbiased Whether to use unbiased estimator
   * (default: true) \param center Whether to center data by subtracting mean
   * (default: true)
   */
  static sptr make(const types::shape &shape, bool unbiased = true,
                   bool center = true);

protected:
  matrix_covariance_pdu(const types::shape &shape, bool unbiased, bool center);
};

// Type aliases for Python bindings
using matrix_covariance_sync_f = matrix_covariance_sync<float>;
using matrix_covariance_sync_d = matrix_covariance_sync<double>;
using matrix_covariance_sync_cf = matrix_covariance_sync<gr_complex>;
using matrix_covariance_sync_cd = matrix_covariance_sync<std::complex<double>>;

using matrix_covariance_pdu_f = matrix_covariance_pdu<float>;
using matrix_covariance_pdu_d = matrix_covariance_pdu<double>;
using matrix_covariance_pdu_cf = matrix_covariance_pdu<gr_complex>;
using matrix_covariance_pdu_cd = matrix_covariance_pdu<std::complex<double>>;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_COVARIANCE_H */
