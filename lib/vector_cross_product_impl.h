/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_VECTOR_CROSS_PRODUCT_IMPL_H
#define INCLUDED_LINALG_VECTOR_CROSS_PRODUCT_IMPL_H

#include <gnuradio/linalg/vector_cross_product.h>

namespace gr {
namespace linalg {

/*!
 * \brief Implementation class for vector cross product sync block
 * \ingroup linalg
 */
template <typename Scalar>
class vector_cross_product_sync_impl
    : virtual public linalg_base<Scalar>,
      virtual public vector_cross_product<Scalar>,
      virtual public linalg_base_sync<Scalar>,
      virtual public vector_cross_product_sync<Scalar> {
private:
  bool d_validate_3d; // Whether to enforce 3D vector constraint

public:
  vector_cross_product_sync_impl(bool validate_3d = true);
  ~vector_cross_product_sync_impl() = default;

  // Override operation for specific implementation
  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;

private:
  // Cross product computation utilities
  std::vector<Scalar> compute_cross_product_3d(const std::vector<Scalar> &a,
                                               const std::vector<Scalar> &b);
  bool validate_3d_vector(const types::const_matrix_map_dynamic<Scalar> &vec);
};

/*!
 * \brief Implementation class for vector cross product PDU block
 * \ingroup linalg
 */
template <typename Scalar>
class vector_cross_product_pdu_impl
    : virtual public linalg_base<Scalar>,
      virtual public vector_cross_product<Scalar>,
      virtual public linalg_base_pdu<Scalar>,
      virtual public vector_cross_product_pdu<Scalar> {
public:
  vector_cross_product_pdu_impl(bool validate_3d = true);
  ~vector_cross_product_pdu_impl() = default;
};

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_VECTOR_CROSS_PRODUCT_IMPL_H */