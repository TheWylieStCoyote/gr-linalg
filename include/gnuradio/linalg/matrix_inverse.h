/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_MATRIX_INVERSE_H
#define INCLUDED_LINALG_MATRIX_INVERSE_H

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

template <typename Scalar>
class LINALG_API matrix_inverse : virtual public linalg_base<Scalar> {
public:
  using sptr = std::shared_ptr<matrix_inverse<Scalar>>;
  matrix_inverse() {}

  // Output shapes identical to input shapes (per matrix inverse)
  static types::vector_shapes
  compute_output_shapes(const types::vector_shapes &input_shapes) {
    return input_shapes;
  }
};

template <typename Scalar>
class LINALG_API matrix_inverse_sync : public matrix_inverse<Scalar>,
                                       public linalg_base_sync<Scalar> {
public:
  using sptr = std::shared_ptr<matrix_inverse_sync<Scalar>>;
  explicit matrix_inverse_sync(const types::shape &shape);
  static sptr make(const types::shape &shape);
};

template <typename Scalar>
class LINALG_API matrix_inverse_pdu : public matrix_inverse<Scalar>,
                                      public linalg_base_pdu<Scalar> {
public:
  using sptr = std::shared_ptr<matrix_inverse_pdu<Scalar>>;
  static sptr make(const types::shape &shape);
};

// Aliases
using matrix_inverse_sync_f = matrix_inverse_sync<float>;
using matrix_inverse_sync_d = matrix_inverse_sync<double>;
using matrix_inverse_sync_c = matrix_inverse_sync<std::complex<float>>;
using matrix_inverse_sync_cd = matrix_inverse_sync<std::complex<double>>;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_INVERSE_H */
