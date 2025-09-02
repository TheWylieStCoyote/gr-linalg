/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_DOT_PRODUCT_H
#define INCLUDED_LINALG_DOT_PRODUCT_H

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
 * \brief Dot product computation block
 * \ingroup linalg
 *
 * This block computes the dot product (scalar product) of two input vectors.
 *
 * Mathematical operation: a · b = Σ(a_i * b_i)
 *
 * For complex vectors, this uses the standard complex dot product:
 * a · b = Σ(a_i * conj(b_i))
 *
 * Both input vectors must have the same shape, and the output is a scalar.
 */

template <typename Scalar>
class LINALG_API dot_product : virtual public linalg_base<Scalar> {
public:
  using sptr = std::shared_ptr<dot_product<Scalar>>;

  dot_product(const types::shape &shape_0, const types::shape &shape_1);
  ~dot_product();

  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;

  static types::vector_shapes
  compute_output_shapes(const types::vector_shapes &input_shapes);

private:
  types::shape d_shape_0;
  types::shape d_shape_1;

  // For complex types, get the real component type
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

  Scalar compute_dot_product(
      const Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>
          &vector_a,
      const Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>
          &vector_b) const;
};

template <typename Scalar>
class LINALG_API dot_product_sync : virtual public dot_product<Scalar>,
                                    virtual public linalg_base_sync<Scalar> {
public:
  using sptr = std::shared_ptr<dot_product_sync<Scalar>>;

  dot_product_sync(const types::shape &shape_0, const types::shape &shape_1);
  static sptr make(const types::shape &shape_0, const types::shape &shape_1);
};

// PDU implementation temporarily disabled - focus on sync block first
// template <typename Scalar>
// class LINALG_API dot_product_pdu : virtual public dot_product<Scalar>,
//                                    virtual public linalg_base_pdu<Scalar> {
// public:
//   using sptr = std::shared_ptr<dot_product_pdu<Scalar>>;
//
//   dot_product_pdu(const types::shape &shape_0, const types::shape &shape_1);
//   static sptr make(const types::shape &shape_0, const types::shape &shape_1);
// };

// Aliases
using dot_product_sync_f = dot_product_sync<float>;
using dot_product_sync_d = dot_product_sync<double>;
using dot_product_sync_c = dot_product_sync<std::complex<float>>;
using dot_product_sync_cd = dot_product_sync<std::complex<double>>;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_DOT_PRODUCT_H */
