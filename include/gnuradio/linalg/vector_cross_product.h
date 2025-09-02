/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_VECTOR_CROSS_PRODUCT_H
#define INCLUDED_LINALG_VECTOR_CROSS_PRODUCT_H

#include <boost/shared_ptr.hpp>
#include <complex>
#include <gnuradio/block.h>
#include <gnuradio/linalg/api.h>
#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/types.h>
#include <gnuradio/sync_block.h>
#include <memory>
#include <vector>

namespace gr {
namespace linalg {

/*!
 * \brief Vector cross product operation for 3D vectors
 * \ingroup linalg
 *
 * This block computes the cross product of two 3D vectors.
 * Both input vectors must have exactly 3 elements, and the output
 * vector will also have 3 elements.
 *
 * Mathematical operation: a × b = (a₂b₃-a₃b₂, a₃b₁-a₁b₃, a₁b₂-a₂b₁)
 *
 * For complex vectors, the cross product is computed using the same
 * formula with complex arithmetic.
 */
template <typename Scalar>
class LINALG_API vector_cross_product : virtual public linalg_base<Scalar> {
public:
  typedef std::shared_ptr<vector_cross_product<Scalar>> sptr;

  // Trivial inline ctor to allow most-derived impl to construct the virtual
  // base
  vector_cross_product();
  vector_cross_product(bool validate_3d = true);

  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;

  // Static helpers delegate to linalg_base/types
  static void validate_shape(const types::vector_shapes &input_shapes = {},
                             const types::vector_shapes &output_shapes = {},
                             const std::string &name = "vector_cross_product");

  static std::vector<size_t> compute_sizes(const types::vector_shapes &shapes);

  static types::vector_shapes
  compute_output_shapes(const types::vector_shapes &input_shapes);

private:
  bool d_validate_3d; // Whether to enforce 3D vector constraint
};

/*!
 * \brief Vector cross product sync block
 * \ingroup linalg
 */
template <typename Scalar>
class LINALG_API vector_cross_product_sync
    : virtual public vector_cross_product<Scalar>,
      virtual public linalg_base_sync<Scalar> {
public:
  typedef std::shared_ptr<vector_cross_product_sync<Scalar>> sptr;

  // Constructor used by factory
  vector_cross_product_sync(bool validate_3d = true);

  // Factory
  static sptr make(bool validate_3d = true);
};

/*!
 * \brief Vector cross product PDU block
 * \ingroup linalg
 */
template <typename Scalar>
class LINALG_API vector_cross_product_pdu
    : virtual public vector_cross_product<Scalar>,
      virtual public linalg_base_pdu<Scalar> {
public:
  // Convenience typedefs
  typedef std::shared_ptr<vector_cross_product_pdu<Scalar>> sptr;
  // Factory
  static sptr make(bool validate_3d = true);

protected:
  vector_cross_product_pdu(bool validate_3d = true);

private:
  std::vector<std::vector<Scalar>>
      d_last_seen; // cached inputs per port (2 inputs)
  std::vector<bool> d_seen;
  bool d_validate_3d;

  void handle_pdu(int idx, const pmt::pmt_t &pdu);
  bool decode_and_store(int idx, const pmt::pmt_t &data,
                        size_t expected_len = 3);
  pmt::pmt_t encode_vector(const std::vector<Scalar> &v);

  // Cross product computation helper
  std::vector<Scalar> compute_cross_product(const std::vector<Scalar> &a,
                                            const std::vector<Scalar> &b);
};

// Sync blocks - Note: Cross product typically only meaningful for
// float/double/complex
using vector_cross_product_sync_f = vector_cross_product_sync<float>;
using vector_cross_product_sync_d = vector_cross_product_sync<double>;
using vector_cross_product_sync_c =
    vector_cross_product_sync<std::complex<float>>;
using vector_cross_product_sync_cd =
    vector_cross_product_sync<std::complex<double>>;

// PDU blocks
using vector_cross_product_pdu_f = vector_cross_product_pdu<float>;
using vector_cross_product_pdu_d = vector_cross_product_pdu<double>;
using vector_cross_product_pdu_c =
    vector_cross_product_pdu<std::complex<float>>;
using vector_cross_product_pdu_cd =
    vector_cross_product_pdu<std::complex<double>>;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_VECTOR_CROSS_PRODUCT_H */