/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_VECTOR_ANGLE_IMPL_H
#define INCLUDED_LINALG_VECTOR_ANGLE_IMPL_H

#include <gnuradio/linalg/vector_angle.h>

namespace gr {
namespace linalg {

/*!
 * \brief Implementation class for vector angle sync block.
 */
template <typename Scalar>
class vector_angle_sync_impl : virtual public linalg_base<Scalar>,
                               virtual public vector_angle<Scalar>,
                               virtual public linalg_base_sync<Scalar>,
                               virtual public vector_angle_sync<Scalar> {
public:
  /*!
   * \brief Constructor for vector angle sync implementation.
   * \param shape Common shape of both input vectors
   * \param degrees If true, output angle in degrees; if false, output in
   * radians
   */
  vector_angle_sync_impl(const types::shape &shape, bool degrees);

  /*!
   * \brief Destructor.
   */
  ~vector_angle_sync_impl();

private:
  bool d_degrees; ///< Whether to output in degrees (true) or radians (false)
};

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_VECTOR_ANGLE_IMPL_H */
