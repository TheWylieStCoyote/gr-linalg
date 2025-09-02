/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_MATRIX_DIAG_H
#define INCLUDED_LINALG_MATRIX_DIAG_H

#include <gnuradio/linalg/api.h>
#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/types.h>

namespace gr {
namespace linalg {

/*!
 * \brief <+description of block+>
 * \ingroup linalg
 *
 */

template <typename Scalar>
class LINALG_API matrix_diag : virtual public linalg_base<Scalar> {
public:
  typedef std::shared_ptr<matrix_diag<Scalar>> sptr;

  matrix_diag(types::shape &shape, int k = 0);

  ~matrix_diag();

  void set_k(int k);
  int get_k() const;

  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;

  static types::vector_shapes
  compute_output_shapes(const types::vector_shapes &input_shapes);

  static types::vector_shapes
  compute_output_shapes_with_k(const types::vector_shapes &input_shapes, int k);

private:
  int d_k;
};

template <typename Scalar>
class LINALG_API matrix_diag_sync : virtual public matrix_diag<Scalar>,
                                    virtual public linalg_base_sync<Scalar> {
public:
  typedef std::shared_ptr<matrix_diag_sync<Scalar>> sptr;

  // Constructor used by factory
  matrix_diag_sync(types::shape &shape, int k = 0);

  static sptr make(types::shape &shape, int k);
};

template <typename Scalar>
class LINALG_API matrix_diag_pdu : virtual public matrix_diag<Scalar>,
                                   virtual public linalg_base_pdu<Scalar> {
public:
  typedef std::shared_ptr<matrix_diag_pdu<Scalar>> sptr;

  // Constructor used by factory
  matrix_diag_pdu(types::shape &shape, int k = 0);

  static sptr make(types::shape &shape, int k);
};

// Type aliases for Python bindings
using matrix_diag_sync_f = matrix_diag_sync<float>;
using matrix_diag_sync_d = matrix_diag_sync<double>;
using matrix_diag_sync_cf = matrix_diag_sync<gr_complex>;
using matrix_diag_sync_cd = matrix_diag_sync<std::complex<double>>;

using matrix_diag_pdu_f = matrix_diag_pdu<float>;
using matrix_diag_pdu_d = matrix_diag_pdu<double>;
using matrix_diag_pdu_cf = matrix_diag_pdu<gr_complex>;
using matrix_diag_pdu_cd = matrix_diag_pdu<std::complex<double>>;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_DIAG_H */
