/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_MATRIX_RESHAPE_H
#define INCLUDED_LINALG_MATRIX_RESHAPE_H

#include <complex>
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
 * \brief Matrix reshape operation block
 * \ingroup linalg
 *
 * This block reshapes input matrices to specified output dimensions while
 * preserving the total number of elements. The reshape operation reinterprets
 * the matrix data in row-major order.
 *
 * Mathematical operation: reshape(A, [m, n]) where total elements = m × n
 *
 * Requirements:
 * - Input matrix total elements must equal output matrix total elements
 * - Output shape must be specified at construction time
 * - Data is reshaped in row-major (C-style) order
 *
 * Example: reshape([1,2,3,4,5,6] as 2×3) to 3×2 gives:
 * Input:  [[1,2,3],   ->  Output: [[1,2],
 *          [4,5,6]]              [3,4],
 *                                 [5,6]]
 */

template <typename Scalar>
class LINALG_API matrix_reshape : virtual public linalg_base<Scalar> {
public:
  using sptr = std::shared_ptr<matrix_reshape<Scalar>>;

  matrix_reshape(const types::shape &input_shape,
                 const types::shape &output_shape);
  ~matrix_reshape();

  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;

  static types::vector_shapes
  compute_output_shapes(const types::vector_shapes &input_shapes,
                        const types::shape &target_shape);

  // Get the target output shape
  const types::shape &get_target_shape() const { return d_target_shape; }

private:
  types::shape d_input_shape;
  types::shape d_target_shape;

  // Validate that reshape is possible (same total elements)
  void validate_reshape(const types::shape &input_shape,
                        const types::shape &output_shape) const;
};

template <typename Scalar>
class LINALG_API matrix_reshape_sync : virtual public matrix_reshape<Scalar>,
                                       virtual public linalg_base_sync<Scalar> {
public:
  using sptr = std::shared_ptr<matrix_reshape_sync<Scalar>>;

  matrix_reshape_sync(const types::shape &input_shape,
                      const types::shape &output_shape);
  static sptr make(const types::shape &input_shape,
                   const types::shape &output_shape);
};

// PDU implementation for future extension
// template <typename Scalar>
// class LINALG_API matrix_reshape_pdu : virtual public matrix_reshape<Scalar>,
//                                       virtual public linalg_base_pdu<Scalar>
//                                       {
// public:
//   using sptr = std::shared_ptr<matrix_reshape_pdu<Scalar>>;
//
//   matrix_reshape_pdu(const types::shape &input_shape, const types::shape
//   &output_shape); static sptr make(const types::shape &input_shape, const
//   types::shape &output_shape);
// };

// Aliases
using matrix_reshape_sync_f = matrix_reshape_sync<float>;
using matrix_reshape_sync_d = matrix_reshape_sync<double>;
using matrix_reshape_sync_c = matrix_reshape_sync<std::complex<float>>;
using matrix_reshape_sync_cd = matrix_reshape_sync<std::complex<double>>;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_RESHAPE_H */