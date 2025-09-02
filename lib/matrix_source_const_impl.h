/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_MATRIX_SOURCE_CONST_IMPL_H
#define INCLUDED_LINALG_MATRIX_SOURCE_CONST_IMPL_H

#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/matrix_source_const.h>
#include <gnuradio/linalg/types.h>
#include <memory>
#include <string>
#include <vector>

namespace gr {
namespace linalg {

template <typename Scalar>
class matrix_source_const_sync_impl
    : virtual public linalg_base<Scalar>,
      virtual public matrix_source_const<Scalar>,
      virtual public linalg_base_sync<Scalar>,
      virtual public matrix_source_const_sync<Scalar> {
public:
  matrix_source_const_sync_impl(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &matrix,
      const types::shape &shape);
  ~matrix_source_const_sync_impl() override = default;

  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;
};

template <typename Scalar>
class matrix_source_const_pdu_impl
    : virtual public linalg_base<Scalar>,
      virtual public matrix_source_const<Scalar>,
      virtual public linalg_base_pdu<Scalar>,
      virtual public matrix_source_const_pdu<Scalar> {
public:
  matrix_source_const_pdu_impl(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &matrix,
      const types::shape &shape);
  ~matrix_source_const_pdu_impl() override = default;

  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;
};

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_SOURCE_CONST_IMPL_H */
