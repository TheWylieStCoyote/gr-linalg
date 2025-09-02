/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_MATRIX_RANK_H
#define INCLUDED_LINALG_MATRIX_RANK_H

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
class LINALG_API matrix_rank : virtual public linalg_base<Scalar> {
public:
  using sptr = std::shared_ptr<matrix_rank<Scalar>>;
  matrix_rank() {}
  // Output rank is scalar 1x1 per input matrix
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

template <typename Scalar>
class LINALG_API matrix_rank_sync : public matrix_rank<Scalar>,
                                    public linalg_base_sync<Scalar> {
public:
  using sptr = std::shared_ptr<matrix_rank_sync<Scalar>>;
  explicit matrix_rank_sync(const types::shape &shape);
  static sptr make(const types::shape &shape);
};

template <typename Scalar>
class LINALG_API matrix_rank_pdu : public matrix_rank<Scalar>,
                                   public linalg_base_pdu<Scalar> {
public:
  using sptr = std::shared_ptr<matrix_rank_pdu<Scalar>>;
  static sptr make(const types::shape &shape);
};

// Aliases
using matrix_rank_sync_f = matrix_rank_sync<float>;
using matrix_rank_sync_d = matrix_rank_sync<double>;
using matrix_rank_sync_c = matrix_rank_sync<std::complex<float>>;
using matrix_rank_sync_cd = matrix_rank_sync<std::complex<double>>;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_RANK_H */
