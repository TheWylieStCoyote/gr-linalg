/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_MATRIX_TRANSPOSE_H
#define INCLUDED_LINALG_MATRIX_TRANSPOSE_H

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
class LINALG_API matrix_transpose : virtual public linalg_base<Scalar> {
public:
  using sptr = std::shared_ptr<matrix_transpose<Scalar>>;

  // Trivial inline ctor; most-derived impl initializes virtual base
  matrix_transpose() {}

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
class LINALG_API matrix_transpose_sync : public matrix_transpose<Scalar>,
                                         public linalg_base_sync<Scalar> {
public:
  using sptr = std::shared_ptr<matrix_transpose_sync<Scalar>>;

  matrix_transpose_sync(const types::shape &shape);
  static sptr make(const types::shape &shape);
};

// PDU variant (placeholder)

template <typename Scalar>
class LINALG_API matrix_transpose_pdu : public matrix_transpose<Scalar>,
                                        public linalg_base_pdu<Scalar> {
public:
  using sptr = std::shared_ptr<matrix_transpose_pdu<Scalar>>;
  static sptr make(const types::shape &shape);
};

// Sync typedefs
using matrix_transpose_sync_f = matrix_transpose_sync<float>;
using matrix_transpose_sync_d = matrix_transpose_sync<double>;

// PDU typedefs
using matrix_transpose_pdu_f = matrix_transpose_pdu<float>;
using matrix_transpose_pdu_d = matrix_transpose_pdu<double>;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_TRANSPOSE_H */
