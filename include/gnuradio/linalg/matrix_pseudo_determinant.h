/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_MATRIX_PSEUDO_DETERMINANT_H
#define INCLUDED_LINALG_MATRIX_PSEUDO_DETERMINANT_H

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
 * \brief Compute the pseudo-determinant of a matrix
 * \ingroup linalg
 *
 * The pseudo-determinant is defined as the product of all non-zero singular
 * values. For square matrices with full rank, this equals the absolute value of
 * the determinant. For non-square or rank-deficient matrices, it provides a
 * meaningful volume measure.
 *
 * Mathematical definition:
 * - pseudo_det(A) = ∏(σᵢ > 0) σᵢ   where σᵢ are singular values of A
 * - For square full-rank matrix: pseudo_det(A) = |det(A)|
 * - For rank-deficient matrix: pseudo_det(A) < |det(A)|
 * - For zero matrix: pseudo_det(A) = 0
 */
template <typename Scalar>
class LINALG_API matrix_pseudo_determinant
    : virtual public linalg_base<Scalar> {
public:
  using sptr = std::shared_ptr<matrix_pseudo_determinant<Scalar>>;
  matrix_pseudo_determinant() {}

  // Output is scalar (1x1) per matrix input
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
class LINALG_API matrix_pseudo_determinant_sync
    : public matrix_pseudo_determinant<Scalar>,
      public linalg_base_sync<Scalar> {
public:
  using sptr = std::shared_ptr<matrix_pseudo_determinant_sync<Scalar>>;
  explicit matrix_pseudo_determinant_sync(const types::shape &shape);
  static sptr make(const types::shape &shape);
};

template <typename Scalar>
class LINALG_API matrix_pseudo_determinant_pdu
    : virtual public matrix_pseudo_determinant<Scalar>,
      virtual public linalg_base_pdu<Scalar> {
public:
  using sptr = std::shared_ptr<matrix_pseudo_determinant_pdu<Scalar>>;
  static sptr make(const types::shape &shape);
};

// Precision type aliases
using matrix_pseudo_determinant_sync_f = matrix_pseudo_determinant_sync<float>;
using matrix_pseudo_determinant_sync_d = matrix_pseudo_determinant_sync<double>;
using matrix_pseudo_determinant_sync_c =
    matrix_pseudo_determinant_sync<std::complex<float>>;
using matrix_pseudo_determinant_sync_cd =
    matrix_pseudo_determinant_sync<std::complex<double>>;

using matrix_pseudo_determinant_pdu_f = matrix_pseudo_determinant_pdu<float>;
using matrix_pseudo_determinant_pdu_d = matrix_pseudo_determinant_pdu<double>;
using matrix_pseudo_determinant_pdu_c =
    matrix_pseudo_determinant_pdu<std::complex<float>>;
using matrix_pseudo_determinant_pdu_cd =
    matrix_pseudo_determinant_pdu<std::complex<double>>;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_PSEUDO_DETERMINANT_H */