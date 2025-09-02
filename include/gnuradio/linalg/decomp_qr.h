/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_DECOMP_QR_H
#define INCLUDED_LINALG_DECOMP_QR_H

#include <Eigen/Dense>
#include <boost/shared_ptr.hpp>
#include <complex>
#include <gnuradio/block.h>
#include <gnuradio/linalg/api.h>
#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/types.h>
#include <gnuradio/sync_block.h>
#include <memory>
#include <string>
#include <vector>

namespace gr {
namespace linalg {

/*!
 * \brief QR decomposition algorithm interface
 * \ingroup linalg
 */
template <typename Scalar, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic>
class LINALG_API qr_algorithm {
public:
  virtual ~qr_algorithm() = default;

  /*!
   * \brief Perform the QR decomposition
   * \param input Input matrix
   * \param Q Output orthogonal matrix
   * \param R Output upper triangular matrix
   */
  virtual void decompose(const Eigen::Matrix<Scalar, Rows, Cols> &input,
                         Eigen::Matrix<Scalar, Rows, Cols> &Q,
                         Eigen::Matrix<Scalar, Rows, Cols> &R) = 0;
};

/*!
 * \brief Householder QR decomposition (fast, stable)
 */
template <typename Scalar, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic>
class LINALG_API eigen_householder_qr
    : public qr_algorithm<Scalar, Rows, Cols> {
public:
  void decompose(const Eigen::Matrix<Scalar, Rows, Cols> &input,
                 Eigen::Matrix<Scalar, Rows, Cols> &Q,
                 Eigen::Matrix<Scalar, Rows, Cols> &R) override;
};

/*!
 * \brief Column pivoting QR decomposition (more robust for rank-deficient
 * matrices)
 */
template <typename Scalar, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic>
class LINALG_API eigen_colpivhouseholder_qr
    : public qr_algorithm<Scalar, Rows, Cols> {
public:
  void decompose(const Eigen::Matrix<Scalar, Rows, Cols> &input,
                 Eigen::Matrix<Scalar, Rows, Cols> &Q,
                 Eigen::Matrix<Scalar, Rows, Cols> &R) override;
};

template <typename Scalar>
class LINALG_API decomp_qr : virtual public linalg_base<Scalar> {
public:
  typedef std::shared_ptr<decomp_qr<Scalar>> sptr;

  // Trivial inline ctor to allow most-derived impl to construct the virtual
  // base
  decomp_qr(const types::shape &shape,
            const std::shared_ptr<qr_algorithm<Scalar, Eigen::Dynamic,
                                               Eigen::Dynamic>> &algo = nullptr,
            const std::string &name = "decomp_qr");

  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices);

  // Set the algorithm for decomposition
  void set_algorithm(
      std::shared_ptr<qr_algorithm<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
          algo);

  // Static helpers delegate to linalg_base/types
  //! Validate that input_shapes → output_shapes is legal for this block
  static void validate_shape(const types::vector_shapes &input_shapes = {},
                             const types::vector_shapes &output_shapes = {},
                             const std::string &name = "decomp_qr");

  // decomp_qr.h
  static std::vector<size_t> compute_sizes(const types::vector_shapes &shapes);

  static types::vector_shapes
  compute_output_shapes(const types::vector_shapes &input_shapes);

private:
  std::shared_ptr<qr_algorithm<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
      algorithm_;
};

/*!
 * \brief QR decomposition sync block
 * \ingroup linalg
 */
template <typename Scalar>
class LINALG_API decomp_qr_sync : virtual public decomp_qr<Scalar>,
                                  virtual public linalg_base_sync<Scalar> {
public:
  typedef std::shared_ptr<decomp_qr_sync<Scalar>> sptr;

  // Factory
  static sptr make(const types::shape &shape);

protected:
  // Only implementations construct this virtual base
  decomp_qr_sync(const types::shape &shape) : decomp_qr<Scalar>(shape) {}
};

template <typename Scalar>
class LINALG_API decomp_qr_pdu : virtual public decomp_qr<Scalar>,
                                 virtual public linalg_base_pdu<Scalar> {
public:
  // Convenience typedefs
  typedef std::shared_ptr<decomp_qr_pdu<Scalar>> sptr;

  // PDU factory (not yet implemented)
  static sptr make(const types::shape &shape);

protected:
  // Only implementations construct this virtual base
  decomp_qr_pdu(const types::shape &shape) : decomp_qr<Scalar>(shape) {}
};

// Sync blocks
using decomp_qr_sync_i = decomp_qr_sync<int>;
using decomp_qr_sync_f = decomp_qr_sync<float>;
using decomp_qr_sync_d = decomp_qr_sync<double>;
using decomp_qr_sync_c = decomp_qr_sync<std::complex<float>>;
using decomp_qr_sync_cd = decomp_qr_sync<std::complex<double>>;
// PDU blocks
using decomp_qr_pdu_i = decomp_qr_pdu<int>;
using decomp_qr_pdu_f = decomp_qr_pdu<float>;
using decomp_qr_pdu_d = decomp_qr_pdu<double>;
using decomp_qr_pdu_c = decomp_qr_pdu<std::complex<float>>;
using decomp_qr_pdu_cd = decomp_qr_pdu<std::complex<double>>;

// force-instantiate the out-of-line make() for each Scalar - PDU DISABLED
/*
template decomp_qr_pdu<float>::sptr
decomp_qr_pdu<float>::make(const types::shape &);
template decomp_qr_pdu<double>::sptr
decomp_qr_pdu<double>::make(const types::shape &);
template decomp_qr_pdu<std::complex<float>>::sptr
decomp_qr_pdu<std::complex<float>>::make(const types::shape &);
template decomp_qr_pdu<std::complex<double>>::sptr
decomp_qr_pdu<std::complex<double>>::make(const types::shape &);
*/

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_DECOMP_QR_H */
