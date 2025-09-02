/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_MATRIX_ADD_H
#define INCLUDED_LINALG_MATRIX_ADD_H

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

template <typename Scalar>
class LINALG_API matrix_add : virtual public linalg_base<Scalar> {
public:
  typedef std::shared_ptr<matrix_add<Scalar>> sptr;

  // Trivial inline ctor to allow most-derived impl to construct the virtual
  // base
  matrix_add();
  matrix_add(const types::shape &shape, const int num_inputs = 2);

  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices);

  // Static helpers delegate to linalg_base/types
  static void validate_shape(const types::vector_shapes &input_shapes = {},
                             const types::vector_shapes &output_shapes = {},
                             const std::string &name = "matrix_add");

  static std::vector<size_t> compute_sizes(const types::vector_shapes &shapes);

  static types::vector_shapes
  compute_output_shapes(const types::vector_shapes &input_shapes);
};

/*!
 * \brief Matrix addition sync block
 * \ingroup linalg
 */
template <typename Scalar>
class LINALG_API matrix_add_sync : virtual public matrix_add<Scalar>,
                                   virtual public linalg_base_sync<Scalar> {
public:
  typedef std::shared_ptr<matrix_add_sync<Scalar>> sptr;

  // Factory only
  static sptr make(const types::shape &shape, const int num_inputs = 2);

protected:
  // Only implementations construct this virtual base
  matrix_add_sync(const types::shape &shape, const int num_inputs = 2);
};

template <typename Scalar>
class LINALG_API matrix_add_pdu : virtual public matrix_add<Scalar>,
                                  virtual public linalg_base_pdu<Scalar> {
public:
  // Convenience typedefs
  typedef std::shared_ptr<matrix_add_pdu<Scalar>> sptr;
  // Factory
  static sptr make(const types::shape &shape, const int num_inputs = 2);

protected:
  matrix_add_pdu(const types::shape &shape, const int num_inputs);

private:
  int d_rows{0};
  int d_cols{0};
  int d_num_inputs{0};
  std::vector<std::vector<Scalar>> d_last_seen; // cached inputs per port
  std::vector<bool> d_seen;

  size_t element_count() const;
  void handle_pdu(int idx, const pmt::pmt_t &pdu);
  bool decode_and_store(int idx, const pmt::pmt_t &data, size_t nelem);
  pmt::pmt_t encode_vector(const std::vector<Scalar> &v);
};

// Sync blocks
using matrix_add_sync_i = matrix_add_sync<int>;
using matrix_add_sync_f = matrix_add_sync<float>;
using matrix_add_sync_d = matrix_add_sync<double>;
using matrix_add_sync_c = matrix_add_sync<std::complex<float>>;
using matrix_add_sync_cd = matrix_add_sync<std::complex<double>>;
// PDU blocks
using matrix_add_pdu_i = matrix_add_pdu<int>;
using matrix_add_pdu_f = matrix_add_pdu<float>;
using matrix_add_pdu_d = matrix_add_pdu<double>;
using matrix_add_pdu_c = matrix_add_pdu<std::complex<float>>;
using matrix_add_pdu_cd = matrix_add_pdu<std::complex<double>>;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_ADD_H */
