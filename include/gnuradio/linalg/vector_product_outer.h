/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_VECTOR_PRODUCT_OUTER_H
#define INCLUDED_LINALG_VECTOR_PRODUCT_OUTER_H

#include <Eigen/Dense>
#include <gnuradio/linalg/api.h>
#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <memory>

namespace gr {
namespace linalg {

/**
 * @brief Vector outer product operations
 *
 * Computes the outer product between two input vectors, which is a
 * fundamental operation in linear algebra that produces a matrix.
 *
 * **Mathematical Background:**
 * For vectors u and v, the outer product u ⊗ v produces a matrix M where:
 * M[i,j] = u[i] * v[j]
 *
 * **Applications:**
 * - Creating rank-1 matrices from vectors
 * - Tensor products in quantum mechanics
 * - Covariance matrix construction
 * - Signal processing outer products
 *
 * @tparam Scalar The input/output scalar type (float, double, complex<float>,
 * complex<double>)
 */
template <typename Scalar>
class LINALG_API vector_product_outer : virtual public linalg_base<Scalar> {
public:
  typedef std::shared_ptr<vector_product_outer<Scalar>> sptr;

  // Trivial inline ctor to allow most-derived impl to construct the virtual
  // base
  vector_product_outer();
  vector_product_outer(const types::shape &input_shape_a,
                       const types::shape &input_shape_b);

  /**
   * @brief Perform vector outer product operation
   *
   * @param input_vectors Input vectors (2 vectors: a and b)
   * @param output_vectors Output matrix (outer product a ⊗ b)
   * @return Operation status
   */
  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_vectors,
            types::vector_matrix_map<Scalar> &output_vectors);

  // Static helpers delegate to linalg_base/types
  static const types::vector_shapes
  compute_output_shapes(const types::vector_shapes &input_shapes);

  static const gr::io_signature::sptr
  make_input_signature(const types::vector_shapes &input_shapes);

  static const gr::io_signature::sptr
  make_output_signature(const types::vector_shapes &input_shapes);

private:
  types::shape d_input_shape_a;
  types::shape d_input_shape_b;
};

// Sync block implementation
template <typename Scalar>
class LINALG_API vector_product_outer_sync
    : public linalg_base_sync<Scalar>,
      public vector_product_outer<Scalar> {
public:
  typedef std::shared_ptr<vector_product_outer_sync<Scalar>> sptr;

  static sptr make(const types::shape &input_shape_a,
                   const types::shape &input_shape_b);

  vector_product_outer_sync(const types::shape &input_shape_a,
                            const types::shape &input_shape_b);
};

// PDU block implementation
template <typename Scalar>
class LINALG_API vector_product_outer_pdu
    : public linalg_base_pdu<Scalar>,
      public vector_product_outer<Scalar> {
public:
  typedef std::shared_ptr<vector_product_outer_pdu<Scalar>> sptr;

  static sptr make(const types::vector_shapes &input_shapes);

  vector_product_outer_pdu(const types::vector_shapes &input_shapes);
};

// Type aliases for common use cases
typedef vector_product_outer_sync<float> vector_product_outer_sync_f;
typedef vector_product_outer_sync<double> vector_product_outer_sync_d;
typedef vector_product_outer_sync<std::complex<float>>
    vector_product_outer_sync_c;
typedef vector_product_outer_sync<std::complex<double>>
    vector_product_outer_sync_cd;

typedef vector_product_outer_pdu<float> vector_product_outer_pdu_f;
typedef vector_product_outer_pdu<double> vector_product_outer_pdu_d;
typedef vector_product_outer_pdu<std::complex<float>>
    vector_product_outer_pdu_c;
typedef vector_product_outer_pdu<std::complex<double>>
    vector_product_outer_pdu_cd;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_VECTOR_PRODUCT_OUTER_H */
