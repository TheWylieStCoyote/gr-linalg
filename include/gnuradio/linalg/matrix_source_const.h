/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_MATRIX_SOURCE_CONST_H
#define INCLUDED_LINALG_MATRIX_SOURCE_CONST_H

#include "gnuradio/linalg/types.h"
#include <Eigen/Dense>
#include <complex>
#include <gnuradio/linalg/api.h>
#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/sync_block.h>
#include <memory>
#include <string>
#include <vector>

namespace gr {
namespace linalg {

/*!
 * \brief Matrix source block base class
 * \ingroup linalg
 */
template <typename Scalar>
class LINALG_API matrix_source_const : virtual public linalg_base<Scalar> {
public:
  typedef std::shared_ptr<matrix_source_const<Scalar>> sptr;

  matrix_source_const(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &matrix,
      const types::shape &shape);
  ~matrix_source_const() override = default;

  static void validate_shape(const types::shape &shape,
                             const std::string &name = "matrix_source_const");
  static types::vector_shapes compute_output_shapes(const types::shape &shape);

  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;

protected:
  std::vector<Scalar> matrix_; // The constant matrix to output
  types::shape shape_;         // The shape of the matrix
};

/*!
 * \brief Synchronous matrix source block
 * \ingroup linalg
 */
template <typename Scalar>
class LINALG_API matrix_source_const_sync
    : virtual public matrix_source_const<Scalar>,
      virtual public linalg_base_sync<Scalar> {
public:
  typedef std::shared_ptr<matrix_source_const_sync<Scalar>> sptr;

  static sptr
  make(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &matrix,
       const types::shape &shape);

  static sptr make(const std::vector<std::vector<Scalar>> &matrix,
                   const types::shape &shape);

  static sptr make(const std::vector<Scalar> &matrix,
                   const types::shape &shape);

  matrix_source_const_sync(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &matrix,
      const types::shape &shape);
};

/*!
 * \brief PDU matrix source block
 * \ingroup linalg
 */
template <typename Scalar>
class LINALG_API matrix_source_const_pdu
    : virtual public matrix_source_const<Scalar>,
      virtual public linalg_base_pdu<Scalar> {
public:
  typedef std::shared_ptr<matrix_source_const_pdu<Scalar>> sptr;

  static sptr
  make(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &matrix,
       const types::shape &shape);

  static sptr make(const std::vector<std::vector<Scalar>> &matrix,
                   const types::shape &shape);

  static sptr make(const std::vector<Scalar> &matrix,
                   const types::shape &shape);

  matrix_source_const_pdu(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &matrix,
      const types::shape &shape);
};

// Sync typedefs
using matrix_source_const_sync_i = matrix_source_const_sync<int>;
using matrix_source_const_sync_f = matrix_source_const_sync<float>;
using matrix_source_const_sync_d = matrix_source_const_sync<double>;
using matrix_source_const_sync_c =
    matrix_source_const_sync<std::complex<float>>;
using matrix_source_const_sync_cd =
    matrix_source_const_sync<std::complex<double>>;

// PDU typedefs
using matrix_source_const_pdu_i = matrix_source_const_pdu<int>;
using matrix_source_const_pdu_f = matrix_source_const_pdu<float>;
using matrix_source_const_pdu_d = matrix_source_const_pdu<double>;
using matrix_source_const_pdu_c = matrix_source_const_pdu<std::complex<float>>;
using matrix_source_const_pdu_cd =
    matrix_source_const_pdu<std::complex<double>>;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_SOURCE_CONST_H */
