/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_MATRIX_NORM_H
#define INCLUDED_LINALG_MATRIX_NORM_H

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
 * \brief Matrix norm computation block
 * \ingroup linalg
 *
 * Computes various norms of input matrices including:
 * - L1 norm (maximum column sum)
 * - L2 norm (spectral norm)
 * - Frobenius norm (Euclidean norm)
 * - Max norm (maximum absolute value)
 */
enum class norm_type {
  L1 = 1,        // Maximum column sum
  L2 = 2,        // Spectral norm (largest singular value)
  Frobenius = 3, // Frobenius norm (sqrt of sum of squares)
  Max = -1,      // Maximum absolute value
};

template <typename Scalar>
class LINALG_API matrix_norm : virtual public linalg_base<Scalar> {
public:
  using sptr = std::shared_ptr<matrix_norm<Scalar>>;
  matrix_norm() : d_norm_type(norm_type::Frobenius) {}
  matrix_norm(const types::shape &shape, norm_type norm = norm_type::Frobenius)
      : d_norm_type(norm) {
    (void)shape;
  }

  // Output of a matrix norm is a scalar (we model as 1x1 matrix) per input
  static types::vector_shapes
  compute_output_shapes(const types::vector_shapes &input_shapes);

protected:
  norm_type d_norm_type{norm_type::Frobenius};
};

/*!
 * \brief Matrix norm sync block
 * \ingroup linalg
 */
template <typename Scalar>
class LINALG_API matrix_norm_sync : virtual public matrix_norm<Scalar>,
                                    virtual public linalg_base_sync<Scalar> {
public:
  typedef std::shared_ptr<matrix_norm_sync<Scalar>> sptr;

  // Constructor used by factory
  matrix_norm_sync(const types::shape &shape,
                   norm_type norm = norm_type::Frobenius);

  // Factory
  static sptr make(const types::shape &shape,
                   norm_type norm = norm_type::Frobenius);
};

template <typename Scalar>
class LINALG_API matrix_norm_pdu : virtual public matrix_norm<Scalar>,
                                   virtual public linalg_base_pdu<Scalar> {
public:
  // Convenience typedefs
  typedef std::shared_ptr<matrix_norm_pdu<Scalar>> sptr;
  // Factory
  static sptr make(const types::shape &shape,
                   norm_type norm = norm_type::Frobenius);

protected:
  matrix_norm_pdu(const types::shape &shape, norm_type norm);

private:
  int d_rows{0};
  int d_cols{0};
  norm_type d_norm_type;
  std::vector<Scalar> d_last_seen; // cached input
  bool d_seen{false};

  size_t element_count() const;
  void handle_pdu(int idx, const pmt::pmt_t &pdu);
  bool decode_and_store(int idx, const pmt::pmt_t &data, size_t nelem);
  pmt::pmt_t encode_vector(const std::vector<Scalar> &v);
};

// Sync blocks
using matrix_norm_sync_i = matrix_norm_sync<int>;
using matrix_norm_sync_f = matrix_norm_sync<float>;
using matrix_norm_sync_d = matrix_norm_sync<double>;
using matrix_norm_sync_c = matrix_norm_sync<std::complex<float>>;
using matrix_norm_sync_cd = matrix_norm_sync<std::complex<double>>;
// PDU blocks
using matrix_norm_pdu_i = matrix_norm_pdu<int>;
using matrix_norm_pdu_f = matrix_norm_pdu<float>;
using matrix_norm_pdu_d = matrix_norm_pdu<double>;
using matrix_norm_pdu_c = matrix_norm_pdu<std::complex<float>>;
using matrix_norm_pdu_cd = matrix_norm_pdu<std::complex<double>>;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_NORM_H */
