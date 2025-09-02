/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_MATRIX_ELEMENTWISE_DIVIDE_H
#define INCLUDED_LINALG_MATRIX_ELEMENTWISE_DIVIDE_H

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
#include <vector>

namespace gr {
namespace linalg {

/*!
 * \brief Matrix element-wise division operation
 * \ingroup linalg
 *
 * This block performs element-wise division of input matrices.
 * The first input is the dividend, second input is the divisor.
 * All input matrices must have the same shape, and the output matrix
 * will have the same shape as the inputs.
 *
 * Mathematical operation: (A ÷ B)ij = Aij / Bij
 *
 * Supports tensor broadcasting for compatible shapes.
 * Includes division by zero detection and handling.
 */
template <typename Scalar>
class LINALG_API matrix_elementwise_divide
    : virtual public linalg_base<Scalar> {
public:
  typedef std::shared_ptr<matrix_elementwise_divide<Scalar>> sptr;

  // Trivial inline ctor to allow most-derived impl to construct the virtual
  // base
  matrix_elementwise_divide();
  matrix_elementwise_divide(const types::shape &shape,
                            const int num_inputs = 2);

  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;

  // Static helpers delegate to linalg_base/types
  static void
  validate_shape(const types::vector_shapes &input_shapes = {},
                 const types::vector_shapes &output_shapes = {},
                 const std::string &name = "matrix_elementwise_divide");

  static std::vector<size_t> compute_sizes(const types::vector_shapes &shapes);

  static types::vector_shapes
  compute_output_shapes(const types::vector_shapes &input_shapes);

protected:
  // Division by zero handling utilities
  bool is_zero(const Scalar &val) const;
  Scalar safe_divide(const Scalar &numerator, const Scalar &denominator) const;
};

/*!
 * \brief Matrix element-wise division sync block
 * \ingroup linalg
 */
template <typename Scalar>
class LINALG_API matrix_elementwise_divide_sync
    : virtual public matrix_elementwise_divide<Scalar>,
      virtual public linalg_base_sync<Scalar> {
public:
  typedef std::shared_ptr<matrix_elementwise_divide_sync<Scalar>> sptr;

  // Constructor used by factory
  matrix_elementwise_divide_sync(const types::shape &shape,
                                 const int num_inputs = 2);

  // Factory
  static sptr make(const types::shape &shape, const int num_inputs = 2);
};

/*!
 * \brief Matrix element-wise division PDU block
 * \ingroup linalg
 */
template <typename Scalar>
class LINALG_API matrix_elementwise_divide_pdu
    : virtual public matrix_elementwise_divide<Scalar>,
      virtual public linalg_base_pdu<Scalar> {
public:
  // Convenience typedefs
  typedef std::shared_ptr<matrix_elementwise_divide_pdu<Scalar>> sptr;
  // Factory
  static sptr make(const types::shape &shape, const int num_inputs = 2);

protected:
  matrix_elementwise_divide_pdu(const types::shape &shape,
                                const int num_inputs);

private:
  int d_rows{0};
  int d_cols{0};
  int d_num_inputs{0};
  std::vector<std::vector<Scalar>> d_last_seen; // cached inputs per port
  std::vector<bool> d_seen;

  size_t element_count() const;
  void handle_pdu(int idx, const pmt::pmt_t &pdu);
  bool decode_and_store(int idx, const pmt::pmt_t &data, size_t nelem);
  pmt::pmt_t encode_vector(const std::vector<Scalar> &v);
};

// Sync blocks
using matrix_elementwise_divide_sync_i = matrix_elementwise_divide_sync<int>;
using matrix_elementwise_divide_sync_f = matrix_elementwise_divide_sync<float>;
using matrix_elementwise_divide_sync_d = matrix_elementwise_divide_sync<double>;
using matrix_elementwise_divide_sync_c =
    matrix_elementwise_divide_sync<std::complex<float>>;
using matrix_elementwise_divide_sync_cd =
    matrix_elementwise_divide_sync<std::complex<double>>;

// PDU blocks
using matrix_elementwise_divide_pdu_i = matrix_elementwise_divide_pdu<int>;
using matrix_elementwise_divide_pdu_f = matrix_elementwise_divide_pdu<float>;
using matrix_elementwise_divide_pdu_d = matrix_elementwise_divide_pdu<double>;
using matrix_elementwise_divide_pdu_c =
    matrix_elementwise_divide_pdu<std::complex<float>>;
using matrix_elementwise_divide_pdu_cd =
    matrix_elementwise_divide_pdu<std::complex<double>>;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_ELEMENTWISE_DIVIDE_H */