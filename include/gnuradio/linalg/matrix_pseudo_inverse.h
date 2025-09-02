/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_MATRIX_PSEUDO_INVERSE_H
#define INCLUDED_LINALG_MATRIX_PSEUDO_INVERSE_H

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

// TODO Add documentation for the matrix_pseudo_inverse class
// TODO Add ability to select different algorithms for pseudoinverse computation

template <typename Scalar>
class LINALG_API matrix_pseudo_inverse : virtual public linalg_base<Scalar> {
public:
  using sptr = std::shared_ptr<matrix_pseudo_inverse<Scalar>>;

  matrix_pseudo_inverse(const types::shape &shape);
  ~matrix_pseudo_inverse();

  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;

  static types::vector_shapes
  compute_output_shapes(const types::vector_shapes &input_shapes);

private:
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

  void compute_pseudoinverse(
      const Eigen::Map<
          const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &input,
      Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &output)
      const;

  static RealScalar extract_real_value(const Scalar &value);
  static bool is_effectively_zero(const RealScalar &value,
                                  const RealScalar &tolerance);
};

template <typename Scalar>
class LINALG_API matrix_pseudo_inverse_sync
    : virtual public matrix_pseudo_inverse<Scalar>,
      virtual public linalg_base_sync<Scalar> {
public:
  using sptr = std::shared_ptr<matrix_pseudo_inverse_sync<Scalar>>;
  matrix_pseudo_inverse_sync(const types::shape &shape);
  static sptr make(const types::shape &shape);
};

template <typename Scalar>
class LINALG_API matrix_pseudo_inverse_pdu
    : virtual public matrix_pseudo_inverse<Scalar>,
      virtual public linalg_base_pdu<Scalar> {
public:
  using sptr = std::shared_ptr<matrix_pseudo_inverse_pdu<Scalar>>;
  matrix_pseudo_inverse_pdu(const types::shape &shape);
  static sptr make(const types::shape &shape);
};

// Aliases
using matrix_pseudo_inverse_sync_f = matrix_pseudo_inverse_sync<float>;
using matrix_pseudo_inverse_sync_d = matrix_pseudo_inverse_sync<double>;
using matrix_pseudo_inverse_sync_c =
    matrix_pseudo_inverse_sync<std::complex<float>>;
using matrix_pseudo_inverse_sync_cd =
    matrix_pseudo_inverse_sync<std::complex<double>>;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_PSEUDO_INVERSE_H */
