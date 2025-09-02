/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_MATRIX_TRACE_H
#define INCLUDED_LINALG_MATRIX_TRACE_H

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
class LINALG_API matrix_trace : virtual public linalg_base<Scalar> {
public:
  using sptr = std::shared_ptr<matrix_trace<Scalar>>;
  matrix_trace() {}
  // Output of a trace is a scalar (we model as 1x1 matrix) per input
  static types::vector_shapes
  compute_output_shapes(const types::vector_shapes &input_shapes) {
    types::vector_shapes out;
    out.reserve(input_shapes.size());
    for (auto const &s : input_shapes) {
      (void)s;
      out.push_back(types::shape{1, 1});
    }
    return out;
  }
};

// Sync variant

template <typename Scalar>
class LINALG_API matrix_trace_sync : public matrix_trace<Scalar>,
                                     public linalg_base_sync<Scalar> {
public:
  using sptr = std::shared_ptr<matrix_trace_sync<Scalar>>;
  explicit matrix_trace_sync(const types::shape &shape);
  static sptr make(const types::shape &shape);
};

// Placeholder PDU variant (not yet implemented)

template <typename Scalar>
class LINALG_API matrix_trace_pdu : public matrix_trace<Scalar>,
                                    public linalg_base_pdu<Scalar> {
public:
  using sptr = std::shared_ptr<matrix_trace_pdu<Scalar>>;
  static sptr make(const types::shape &shape);
};

// Convenience aliases
using matrix_trace_sync_f = matrix_trace_sync<float>;
using matrix_trace_sync_d = matrix_trace_sync<double>;
using matrix_trace_sync_c = matrix_trace_sync<std::complex<float>>;
using matrix_trace_sync_cd = matrix_trace_sync<std::complex<double>>;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_TRACE_H */
