/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_VECTOR_PROJECTION_H
#define INCLUDED_LINALG_VECTOR_PROJECTION_H

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
 * \brief Vector projection computation block
 * \ingroup linalg
 *
 * This block computes the projection of vector A onto vector B.
 * The mathematical operation is: proj_b(a) = (a·b / b·b) × b
 *
 * The vector projection represents the component of vector A that lies in the
 * direction of vector B. Geometrically, it's the "shadow" that vector A casts
 * onto the line defined by vector B.
 *
 * Mathematical properties:
 * - proj_b(a) is parallel to b
 * - ||proj_b(a)|| ≤ ||a|| (projection magnitude ≤ original magnitude)
 * - proj_b(a) = 0 when a ⊥ b (orthogonal vectors)
 * - proj_b(a) = a when a ∥ b (parallel vectors)
 *
 * Special cases:
 * - Zero vector B: undefined (returns zero vector)
 * - Orthogonal vectors: projection = 0
 * - Parallel vectors: projection = original vector A
 *
 * Implementation Notes:
 * - Both real and complex number projections are fully functional
 * - Complex projections use proper conjugate dot product: a·conj(b) and
 * b·conj(b)
 * - Real projections use standard dot product: a·b and b·b
 *
 * Applications:
 * - Signal decomposition into directional components
 * - Geometric calculations and computer graphics
 * - Principal component analysis preprocessing
 * - Orthogonal decomposition of vectors
 */

template <typename Scalar>
class LINALG_API vector_projection : virtual public linalg_base<Scalar> {
public:
  using sptr = std::shared_ptr<vector_projection<Scalar>>;

  vector_projection(const types::shape &shape_a, const types::shape &shape_b);
  ~vector_projection();

  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;

  static types::vector_shapes
  compute_output_shapes(const types::vector_shapes &input_shapes);

private:
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

  void compute_vector_projection(
      const Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>
          &vector_a,
      const Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>
          &vector_b,
      Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> &result) const;
};

template <typename Scalar>
class LINALG_API vector_projection_sync
    : virtual public vector_projection<Scalar>,
      virtual public linalg_base_sync<Scalar> {
public:
  using sptr = std::shared_ptr<vector_projection_sync<Scalar>>;

  vector_projection_sync(const types::shape &shape_a,
                         const types::shape &shape_b);
  static sptr make(const types::shape &shape_a, const types::shape &shape_b);
};

template <typename Scalar>
class LINALG_API vector_projection_pdu
    : virtual public vector_projection<Scalar>,
      virtual public linalg_base_pdu<Scalar> {
public:
  using sptr = std::shared_ptr<vector_projection_pdu<Scalar>>;

  vector_projection_pdu(const types::shape &shape_a,
                        const types::shape &shape_b);
  static sptr make(const types::shape &shape_a, const types::shape &shape_b);
};

// Aliases for sync blocks
using vector_projection_sync_f = vector_projection_sync<float>;
using vector_projection_sync_d = vector_projection_sync<double>;
using vector_projection_sync_c = vector_projection_sync<std::complex<float>>;
using vector_projection_sync_cd = vector_projection_sync<std::complex<double>>;

// Aliases for PDU blocks
using vector_projection_pdu_f = vector_projection_pdu<float>;
using vector_projection_pdu_d = vector_projection_pdu<double>;
using vector_projection_pdu_c = vector_projection_pdu<std::complex<float>>;
using vector_projection_pdu_cd = vector_projection_pdu<std::complex<double>>;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_VECTOR_PROJECTION_H */
