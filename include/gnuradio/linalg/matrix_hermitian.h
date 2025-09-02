/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_MATRIX_HERMITIAN_H
#define INCLUDED_LINALG_MATRIX_HERMITIAN_H

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
class LINALG_API matrix_hermitian : virtual public linalg_base<Scalar> {
public:
  using sptr = std::shared_ptr<matrix_hermitian<Scalar>>;

  // Trivial inline ctor; most-derived impl initializes virtual base
  matrix_hermitian() {}

  // Optional helpers
  static types::vector_shapes
  compute_output_shapes(const types::vector_shapes &input_shapes) {
    if (input_shapes.empty())
      return {};
    types::vector_shapes out = input_shapes;
    for (auto &s : out) {
      if (s.size() >= 2)
        std::swap(s[0], s[1]);
    }
    return out;
  }
};

// Sync variant

template <typename Scalar>
class LINALG_API matrix_hermitian_sync : public matrix_hermitian<Scalar>,
                                         public linalg_base_sync<Scalar> {
public:
  using sptr = std::shared_ptr<matrix_hermitian_sync<Scalar>>;

  matrix_hermitian_sync(const types::shape &shape);
  static sptr make(const types::shape &shape);
};

// PDU variant (placeholder)

template <typename Scalar>
class LINALG_API matrix_hermitian_pdu : public matrix_hermitian<Scalar>,
                                        public linalg_base_pdu<Scalar> {
public:
  using sptr = std::shared_ptr<matrix_hermitian_pdu<Scalar>>;
  static sptr make(const types::shape &shape);
};

// Sync typedefs
using matrix_hermitian_sync_f = matrix_hermitian_sync<float>;
using matrix_hermitian_sync_d = matrix_hermitian_sync<double>;
using matrix_hermitian_sync_cf = matrix_hermitian_sync<std::complex<float>>;
using matrix_hermitian_sync_cd = matrix_hermitian_sync<std::complex<double>>;

// PDU typedefs
using matrix_hermitian_pdu_f = matrix_hermitian_pdu<float>;
using matrix_hermitian_pdu_d = matrix_hermitian_pdu<double>;
using matrix_hermitian_pdu_cf = matrix_hermitian_pdu<std::complex<float>>;
using matrix_hermitian_pdu_cd = matrix_hermitian_pdu<std::complex<double>>;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_HERMITIAN_H */