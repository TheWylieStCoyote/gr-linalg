/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_VECTOR_NORM_H
#define INCLUDED_LINALG_VECTOR_NORM_H

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
 * \brief Vector norm computation block
 * \ingroup linalg
 *
 * This block computes the norm of input vectors.
 * Supports various norm types: L1, L2 (Euclidean), L∞ (max), and general Lp
 * norms.
 * - L1 norm: sum of absolute values
 * - L2 norm: square root of sum of squares (Euclidean norm)
 * - L∞ norm: maximum absolute value
 * - Lp norm: (sum of |x|^p)^(1/p) for general p
 */

template <typename Scalar>
class LINALG_API vector_norm : virtual public linalg_base<Scalar> {
public:
  using sptr = std::shared_ptr<vector_norm<Scalar>>;

  vector_norm(const types::shape &shape, int order = 2);
  ~vector_norm();

  void set_order(int order);
  int get_order() const;

  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;

  static types::vector_shapes
  compute_output_shapes(const types::vector_shapes &input_shapes);

private:
  int d_order;

  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

  RealScalar compute_vector_norm(
      const Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> &vector)
      const;
};

template <typename Scalar>
class LINALG_API vector_norm_sync : virtual public vector_norm<Scalar>,
                                    virtual public linalg_base_sync<Scalar> {
public:
  using sptr = std::shared_ptr<vector_norm_sync<Scalar>>;

  vector_norm_sync(const types::shape &shape, int order = 2);
  static sptr make(const types::shape &shape, int order = 2);
};

template <typename Scalar>
class LINALG_API vector_norm_pdu : virtual public vector_norm<Scalar>,
                                   virtual public linalg_base_pdu<Scalar> {
public:
  using sptr = std::shared_ptr<vector_norm_pdu<Scalar>>;

  vector_norm_pdu(const types::shape &shape, int order = 2);
  static sptr make(const types::shape &shape, int order = 2);
};

// Aliases
using vector_norm_sync_f = vector_norm_sync<float>;
using vector_norm_sync_d = vector_norm_sync<double>;
using vector_norm_sync_c = vector_norm_sync<std::complex<float>>;
using vector_norm_sync_cd = vector_norm_sync<std::complex<double>>;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_VECTOR_NORM_H */
