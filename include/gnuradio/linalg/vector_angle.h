/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_VECTOR_ANGLE_H
#define INCLUDED_LINALG_VECTOR_ANGLE_H

#include <gnuradio/linalg/api.h>
#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/types.h>

namespace gr {
namespace linalg {

/*!
 * \brief Compute the angle between two vectors.
 * \ingroup linalg
 *
 * This block computes the angle between two input vectors using the formula:
 * θ = arccos((a·b) / (|a||b|))
 *
 * Both input vectors must have the same dimension. The output is a scalar
 * representing the angle in either radians or degrees.
 *
 * Special cases:
 * - Parallel vectors (same direction): angle = 0
 * - Anti-parallel vectors (opposite direction): angle = π (or 180°)
 * - Orthogonal vectors: angle = π/2 (or 90°)
 * - Zero vectors: angle is undefined (returns NaN)
 */
template <typename Scalar>
class LINALG_API vector_angle : virtual public linalg_base<Scalar> {
public:
  /*!
   * \brief Create vector angle calculator.
   * \param shape Common shape of both input vectors
   * \param degrees If true, output angle in degrees; if false, output in
   * radians
   */
  vector_angle(const types::shape &shape, bool degrees = false);

  /*!
   * \brief Compute the angle between two vectors.
   * \param input_matrices Vector containing two input vectors
   * \param output_matrices Vector containing one output scalar (angle)
   * \return Operation result status
   */
  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;

  /*!
   * \brief Compute output shapes for vector angle operation.
   * \param input_shapes Vector containing shapes of two input vectors
   * \return Vector containing shape of output scalar (1x1)
   */
  static types::vector_shapes
  compute_output_shapes(const types::vector_shapes &input_shapes);

private:
  bool d_degrees; ///< Whether to output in degrees (true) or radians (false)
};

/*!
 * \brief Sync block wrapper for vector angle calculation.
 * \ingroup linalg
 */
template <typename Scalar>
class LINALG_API vector_angle_sync : virtual public linalg_base_sync<Scalar>,
                                     virtual public vector_angle<Scalar> {
public:
  typedef std::shared_ptr<vector_angle_sync<Scalar>> sptr;

  /*!
   * \brief Create vector angle sync block.
   * \param shape Common shape of both input vectors
   * \param degrees If true, output angle in degrees; if false, output in
   * radians \return Shared pointer to new vector angle sync block
   */
  static sptr make(const types::shape &shape, bool degrees = false);

  /*!
   * \brief Constructor for vector angle sync block.
   * \param shape Common shape of both input vectors
   * \param degrees If true, output angle in degrees; if false, output in
   * radians
   */
  vector_angle_sync(const types::shape &shape, bool degrees = false);
};

// Explicit template instantiations
extern template class vector_angle<float>;
extern template class vector_angle<double>;
extern template class vector_angle<std::complex<float>>;
extern template class vector_angle<std::complex<double>>;

extern template class vector_angle_sync<float>;
extern template class vector_angle_sync<double>;
extern template class vector_angle_sync<std::complex<float>>;
extern template class vector_angle_sync<std::complex<double>>;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_VECTOR_ANGLE_H */
