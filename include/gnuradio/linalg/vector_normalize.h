/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_VECTOR_NORMALIZE_H
#define INCLUDED_LINALG_VECTOR_NORMALIZE_H

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
 * \brief Vector normalization block - normalizes input vectors to unit length
 * \ingroup linalg
 *
 * This block normalizes vectors by dividing each vector by its norm.
 * For a vector v, the normalized vector is v/||v|| where ||v|| is the vector
 * norm. The norm order can be specified (L1, L2, L∞, etc).
 */

template <typename Scalar>
class LINALG_API vector_normalize : virtual public linalg_base<Scalar> {
public:
  using sptr = std::shared_ptr<vector_normalize<Scalar>>;

  vector_normalize(const types::shape &shape, int order = 2);
  ~vector_normalize();

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

  void normalize_vector(
      const Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> &input,
      Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> &output) const;

  RealScalar compute_vector_norm(
      const Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> &vector)
      const;
};

template <typename Scalar>
class LINALG_API vector_normalize_sync
    : virtual public vector_normalize<Scalar>,
      virtual public linalg_base_sync<Scalar> {
public:
  using sptr = std::shared_ptr<vector_normalize_sync<Scalar>>;

  vector_normalize_sync(const types::shape &shape, int order = 2);
  static sptr make(const types::shape &shape, int order = 2);
};

template <typename Scalar>
class LINALG_API vector_normalize_pdu : virtual public vector_normalize<Scalar>,
                                        virtual public linalg_base_pdu<Scalar> {
public:
  using sptr = std::shared_ptr<vector_normalize_pdu<Scalar>>;

  vector_normalize_pdu(const types::shape &shape, int order = 2);
  static sptr make(const types::shape &shape, int order = 2);
};

// Aliases
using vector_normalize_sync_f = vector_normalize_sync<float>;
using vector_normalize_sync_d = vector_normalize_sync<double>;
using vector_normalize_sync_c = vector_normalize_sync<std::complex<float>>;
using vector_normalize_sync_cd = vector_normalize_sync<std::complex<double>>;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_VECTOR_NORMALIZE_H */
