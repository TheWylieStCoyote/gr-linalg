/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_MATRIX_SUBTRACT_H
#define INCLUDED_LINALG_MATRIX_SUBTRACT_H

#include <complex>
#include <memory>
#include <string>
#include <vector>

#include <gnuradio/linalg/api.h>
#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/types.h>

namespace gr {
namespace linalg {

/*!
 * \brief Element-wise matrix subtraction of N inputs -> 1 output
 *
 * Semantics: out = in0 - in1 - ... - inN-1
 *
 * - Supports arbitrary dynamic shapes provided via types::shape
 * - Broadcasting is managed by linalg_base. We use CUSTOM broadcast to allow
 *   a single output shape with N input shapes (all equal in current design).
 */
template <typename Scalar>
class LINALG_API matrix_subtract : virtual public linalg_base<Scalar> {
public:
  typedef std::shared_ptr<matrix_subtract<Scalar>> sptr;

  // Thin inline ctor so most-derived impl can initialize the virtual base
  matrix_subtract() {}
  matrix_subtract(const types::shape & /*shape*/, int /*num_inputs*/ = 2) {}

  // Core operation to be used by sync and pdu variants
  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices);

  // Static helpers delegate to linalg_base/types
  static void validate_shape(const types::vector_shapes &input_shapes = {},
                             const types::vector_shapes &output_shapes = {},
                             const std::string &name = "matrix_subtract") {
    linalg_base<Scalar>::validate_shapes(input_shapes, output_shapes, name);
  }

  static std::vector<size_t> compute_sizes(const types::vector_shapes &shapes) {
    return types::compute_sizes<Scalar>(shapes);
  }

  // For subtraction, multiple inputs reduce to a single output of same shape
  static types::vector_shapes
  compute_output_shapes(const types::vector_shapes &input_shapes) {
    if (input_shapes.empty())
      return {};
    return types::vector_shapes{input_shapes.front()};
  }
};

/*!
 * \brief Synchronous matrix subtraction block
 */
template <typename Scalar>
class LINALG_API matrix_subtract_sync
    : virtual public matrix_subtract<Scalar>,
      virtual public linalg_base_sync<Scalar> {
public:
  typedef std::shared_ptr<matrix_subtract_sync<Scalar>> sptr;

  // Public ctor used by factory
  matrix_subtract_sync(const types::shape &shape, int num_inputs = 2);

  // Factory
  static sptr make(const types::shape &shape, int num_inputs = 2);
};

/*!
 * \brief PDU variant (not yet implemented)
 */
template <typename Scalar>
class LINALG_API matrix_subtract_pdu : virtual public matrix_subtract<Scalar>,
                                       virtual public linalg_base_pdu<Scalar> {
public:
  typedef std::shared_ptr<matrix_subtract_pdu<Scalar>> sptr;

  static sptr make(const types::shape &shape, int num_inputs = 2);
};

// Sync typedefs
using matrix_subtract_sync_i = matrix_subtract_sync<int>;
using matrix_subtract_sync_f = matrix_subtract_sync<float>;
using matrix_subtract_sync_d = matrix_subtract_sync<double>;
using matrix_subtract_sync_c = matrix_subtract_sync<std::complex<float>>;
using matrix_subtract_sync_cd = matrix_subtract_sync<std::complex<double>>;

// PDU typedefs
using matrix_subtract_pdu_i = matrix_subtract_pdu<int>;
using matrix_subtract_pdu_f = matrix_subtract_pdu<float>;
using matrix_subtract_pdu_d = matrix_subtract_pdu<double>;
using matrix_subtract_pdu_c = matrix_subtract_pdu<std::complex<float>>;
using matrix_subtract_pdu_cd = matrix_subtract_pdu<std::complex<double>>;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_SUBTRACT_H */
