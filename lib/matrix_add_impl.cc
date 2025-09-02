/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "matrix_add_impl.h"
#include <algorithm>
#include <gnuradio/gr_complex.h>
#include <gnuradio/io_signature.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/types.h>
#include <pmt/pmt.h>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace gr {
namespace linalg {

static inline types::vector_shapes _repeat_shape(const types::shape &s, int n) {
  types::vector_shapes v;
  v.reserve(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i)
    v.push_back(s);
  return v;
}

static inline types::vector_shapes _single_shape(const types::shape &s) {
  types::vector_shapes v;
  v.push_back(s);
  return v;
}

static inline std::vector<std::string> _make_names(const std::string &prefix,
                                                   int n) {
  std::vector<std::string> names;
  names.reserve(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i)
    names.push_back(prefix + std::to_string(i));
  return names;
}

// Factory definition

template <typename Scalar>
typename matrix_add_sync<Scalar>::sptr
matrix_add_sync<Scalar>::make(const types::shape &shape, const int num_inputs) {
  return gnuradio::make_block_sptr<matrix_add_sync_impl<Scalar>>(shape,
                                                                 num_inputs);
}

// matrix_add_sync public ctor used by factory

template <typename Scalar>
matrix_add_sync<Scalar>::matrix_add_sync(const types::shape &shape,
                                         const int num_inputs)
    : matrix_add<Scalar>(shape, num_inputs) {}

// Impl ctor: most-derived, initializes the virtual base linalg_base

template <typename Scalar>
matrix_add_sync_impl<Scalar>::matrix_add_sync_impl(const types::shape &shape,
                                                   const int num_inputs)
    : linalg_base<Scalar>("matrix_add", _repeat_shape(shape, num_inputs),
                          _make_names("input_", num_inputs),
                          _single_shape(shape), std::vector<std::string>(),
                          array_broadcast_type::CUSTOM, error_tag_t::NONE,
                          error_pdu_p::NONE),
      matrix_add<Scalar>(shape, num_inputs),
      linalg_base_sync<Scalar>("matrix_add", _repeat_shape(shape, num_inputs),
                               _make_names("input_", num_inputs),
                               _single_shape(shape), std::vector<std::string>(),
                               array_broadcast_type::CUSTOM, error_tag_t::NONE,
                               error_pdu_p::NONE, gr::block::TPP_ALL_TO_ALL),
      matrix_add_sync<Scalar>(shape, num_inputs) {
  // Initialize the base class with the shape and input/output names
}

// matrix_add base trivial ctors (kept out-of-line)

template <typename Scalar>
matrix_add<Scalar>::matrix_add() : linalg_base<Scalar>() {}

template <typename Scalar>
matrix_add<Scalar>::matrix_add(const types::shape & /*shape*/,
                               int /*num_inputs*/)
    : linalg_base<Scalar>() {}

// matrix_add base operation implementation

template <typename Scalar>
OperationReturn matrix_add<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {
  if (input_matrices.size() < 2 || output_matrices.empty())
    return OperationReturn::INVALID_SHAPE;

  // Initialize output matrix with the first input matrix
  *output_matrices[0] = *input_matrices[0];

  // Add all subsequent input matrices to the first one
  for (size_t i = 1; i < input_matrices.size(); ++i) {
    *output_matrices[0] += *input_matrices[i];
  }

  return OperationReturn::SUCCESS;
}

// Provide a trivial operation that sums inputs into first output

template <typename Scalar>
OperationReturn matrix_add_sync_impl<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {
  if (input_matrices.size() < 2 || output_matrices.empty())
    return OperationReturn::INVALID_SHAPE;
  auto &out = *output_matrices[0];
  out.setZero();
  for (auto *in : input_matrices)
    out += *in;
  return OperationReturn::SUCCESS;
}

// Static helpers moved from header

template <typename Scalar>
void matrix_add<Scalar>::validate_shape(
    const types::vector_shapes &input_shapes,
    const types::vector_shapes &output_shapes, const std::string &name) {
  // Use CUSTOM broadcast to allow N inputs -> 1 output
  linalg_base<Scalar>::validate_shapes(input_shapes, output_shapes, name,
                                       array_broadcast_type::CUSTOM);
}

template <typename Scalar>
std::vector<size_t>
matrix_add<Scalar>::compute_sizes(const types::vector_shapes &shapes) {
  return types::compute_sizes<Scalar>(shapes);
}

template <typename Scalar>
types::vector_shapes matrix_add<Scalar>::compute_output_shapes(
    const types::vector_shapes &input_shapes) {
  if (input_shapes.empty())
    return {};
  return types::vector_shapes{input_shapes.front()};
}

// ---------------- PDU implementation ----------------

// Base PDU constructor and member functions

template <typename Scalar>
matrix_add_pdu<Scalar>::matrix_add_pdu(const types::shape &shape,
                                       const int num_inputs)
    : linalg_base<Scalar>("matrix_add_pdu", _repeat_shape(shape, num_inputs),
                          _make_names("in_", num_inputs), _single_shape(shape),
                          _make_names("out_", 1), array_broadcast_type::CUSTOM,
                          error_tag_t::NONE, error_pdu_p::NONE),
      matrix_add<Scalar>(shape, num_inputs),
      linalg_base_pdu<Scalar>(
          "matrix_add_pdu", _repeat_shape(shape, num_inputs),
          _make_names("in_", num_inputs), _single_shape(shape),
          _make_names("out_", 1), array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE, PDU_UPDATE::ANY_INPUT,
          MESSAGE_HANDLER_MODE::DEFAULT) {
  d_rows = shape.size() > 0 ? shape[0] : 0;
  d_cols = shape.size() > 1 ? shape[1] : 0;
  d_num_inputs = num_inputs;
  d_last_seen.resize(static_cast<size_t>(num_inputs));
  d_seen.assign(static_cast<size_t>(num_inputs), false);

  // Install message handlers for each input port
  for (int i = 0; i < d_num_inputs; ++i) {
    auto port = pmt::mp(this->d_input_names[static_cast<size_t>(i)]);
    this->set_msg_handler(
        port, [this, i](pmt::pmt_t msg) { this->handle_pdu(i, msg); });
  }
}

template <typename Scalar>
size_t matrix_add_pdu<Scalar>::element_count() const {
  return static_cast<size_t>((d_rows <= 0 || d_cols <= 0) ? 0
                                                          : (d_rows * d_cols));
}

template <typename Scalar>
void matrix_add_pdu<Scalar>::handle_pdu(int idx, const pmt::pmt_t &pdu) {
  if (!pmt::is_pair(pdu))
    return;
  auto meta = pmt::car(pdu);
  auto data = pmt::cdr(pdu);

  const size_t nelem = element_count();
  if (nelem == 0)
    return;

  // Decode data vector by Scalar
  if (!decode_and_store(idx, data, nelem))
    return;

  // If ALL_INPUTS required, ensure all seen
  if (this->d_pdu_update == PDU_UPDATE::ALL_INPUTS) {
    for (bool s : d_seen)
      if (!s)
        return;
  }

  // Sum across inputs (missing treated as zeros)
  std::vector<Scalar> sum(nelem);
  std::fill(sum.begin(), sum.end(), Scalar{});
  for (int i = 0; i < d_num_inputs; ++i) {
    if (!d_seen[static_cast<size_t>(i)])
      continue;
    const auto &v = d_last_seen[static_cast<size_t>(i)];
    for (size_t k = 0; k < nelem; ++k)
      sum[k] += v[k];
  }

  // Publish output PDU on first output port with same metadata
  pmt::pmt_t out_vec = encode_vector(sum);
  if (out_vec == pmt::PMT_NIL)
    return;
  auto out_pdu = pmt::cons(meta, out_vec);
  this->message_port_pub(pmt::mp(this->d_output_names[0]), out_pdu);
}

// Decode helpers as before

template <typename Scalar>
bool matrix_add_pdu<Scalar>::decode_and_store(int idx, const pmt::pmt_t &data,
                                              size_t nelem) {
  auto &buf = d_last_seen[static_cast<size_t>(idx)];
  buf.resize(nelem);

  if constexpr (std::is_same<Scalar, float>::value) {
    if (!pmt::is_f32vector(data))
      return false;
    size_t len = 0;
    const float *ptr = pmt::f32vector_elements(data, len);
    if (len != nelem)
      return false;
    std::copy(ptr, ptr + len, buf.begin());
  } else if constexpr (std::is_same<Scalar, double>::value) {
    if (!pmt::is_f64vector(data))
      return false;
    size_t len = 0;
    const double *ptr = pmt::f64vector_elements(data, len);
    if (len != nelem)
      return false;
    std::copy(ptr, ptr + len, buf.begin());
  } else if constexpr (std::is_same<Scalar, std::complex<float>>::value) {
    if (!pmt::is_c32vector(data))
      return false;
    size_t len = 0;
    const gr_complex *ptr = pmt::c32vector_elements(data, len);
    if (len != nelem)
      return false;
    for (size_t i = 0; i < len; ++i)
      buf[i] = ptr[i];
  } else if constexpr (std::is_same<Scalar, std::complex<double>>::value) {
    if (!pmt::is_c64vector(data))
      return false;
    size_t len = 0;
    const gr_complexd *ptr = pmt::c64vector_elements(data, len);
    if (len != nelem)
      return false;
    for (size_t i = 0; i < len; ++i)
      buf[i] = ptr[i];
  } else {
    return false;
  }

  d_seen[static_cast<size_t>(idx)] = true;
  return true;
}

// Encode helper

template <typename Scalar>
pmt::pmt_t matrix_add_pdu<Scalar>::encode_vector(const std::vector<Scalar> &v) {
  if constexpr (std::is_same<Scalar, float>::value) {
    return pmt::init_f32vector(v.size(), v.data());
  } else if constexpr (std::is_same<Scalar, double>::value) {
    return pmt::init_f64vector(v.size(), v.data());
  } else if constexpr (std::is_same<Scalar, std::complex<float>>::value) {
    return pmt::init_c32vector(v.size(),
                               reinterpret_cast<const gr_complex *>(v.data()));
  } else if constexpr (std::is_same<Scalar, std::complex<double>>::value) {
    return pmt::init_c64vector(v.size(),
                               reinterpret_cast<const gr_complexd *>(v.data()));
  } else {
    return pmt::PMT_NIL;
  }
}

// Define matrix_add_pdu_impl constructor out-of-class

template <typename Scalar>
matrix_add_pdu_impl<Scalar>::matrix_add_pdu_impl(const types::shape &shape,
                                                 const int num_inputs)
    : linalg_base<Scalar>("matrix_add_pdu", _repeat_shape(shape, num_inputs),
                          _make_names("in_", num_inputs), _single_shape(shape),
                          _make_names("out_", 1), array_broadcast_type::CUSTOM,
                          error_tag_t::NONE, error_pdu_p::NONE),
      matrix_add<Scalar>(),
      linalg_base_pdu<Scalar>(
          "matrix_add_pdu", _repeat_shape(shape, num_inputs),
          _make_names("in_", num_inputs), _single_shape(shape),
          _make_names("out_", 1), array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE, PDU_UPDATE::ANY_INPUT,
          MESSAGE_HANDLER_MODE::DEFAULT),
      matrix_add_pdu<Scalar>(shape, num_inputs) {}

// PDU factory

template <typename Scalar>
typename matrix_add_pdu<Scalar>::sptr
matrix_add_pdu<Scalar>::make(const types::shape &shape, const int num_inputs) {
  // Use a simple derived wrapper to satisfy block_sptr requirements
  return gnuradio::make_block_sptr<matrix_add_pdu_impl<Scalar>>(shape,
                                                                num_inputs);
}

// Explicit instantiation for common scalars

// Base class: ctors
template matrix_add<float>::matrix_add();
template matrix_add<double>::matrix_add();

template matrix_add<float>::matrix_add(const types::shape &, int);
template matrix_add<double>::matrix_add(const types::shape &, int);

// Base class: operation (already instantiated below)
// Base class: validate_shape
template void matrix_add<float>::validate_shape(const types::vector_shapes &,
                                                const types::vector_shapes &,
                                                const std::string &);
template void matrix_add<double>::validate_shape(const types::vector_shapes &,
                                                 const types::vector_shapes &,
                                                 const std::string &);

// Base class: compute_sizes
template std::vector<size_t>
matrix_add<float>::compute_sizes(const types::vector_shapes &);
template std::vector<size_t>
matrix_add<double>::compute_sizes(const types::vector_shapes &);

// Base class: compute_output_shapes
template types::vector_shapes
matrix_add<float>::compute_output_shapes(const types::vector_shapes &);
template types::vector_shapes
matrix_add<double>::compute_output_shapes(const types::vector_shapes &);

// PDU factory
template matrix_add_pdu<float>::sptr
matrix_add_pdu<float>::make(const types::shape &, int);
template matrix_add_pdu<double>::sptr
matrix_add_pdu<double>::make(const types::shape &, int);
template matrix_add_pdu<std::complex<float>>::sptr
matrix_add_pdu<std::complex<float>>::make(const types::shape &, int);
template matrix_add_pdu<std::complex<double>>::sptr
matrix_add_pdu<std::complex<double>>::make(const types::shape &, int);

// Existing explicit instantiations remain
template class matrix_add_sync_impl<float>;

template class matrix_add_sync_impl<double>;

template class matrix_add_sync_impl<std::complex<float>>;

template class matrix_add_sync_impl<std::complex<double>>;

// Explicit template class instantiations for PDU impl

template class matrix_add_pdu_impl<float>;

template class matrix_add_pdu_impl<double>;

template class matrix_add_pdu_impl<std::complex<float>>;

template class matrix_add_pdu_impl<std::complex<double>>;

template typename matrix_add_sync<float>::sptr
matrix_add_sync<float>::make(const types::shape &, int);

template typename matrix_add_sync<double>::sptr
matrix_add_sync<double>::make(const types::shape &, int);

template typename matrix_add_sync<std::complex<float>>::sptr
matrix_add_sync<std::complex<float>>::make(const types::shape &, int);

template typename matrix_add_sync<std::complex<double>>::sptr
matrix_add_sync<std::complex<double>>::make(const types::shape &, int);

template matrix_add_sync<float>::matrix_add_sync(const types::shape &, int);

template matrix_add_sync<double>::matrix_add_sync(const types::shape &, int);

template matrix_add_sync<std::complex<float>>::matrix_add_sync(
    const types::shape &, int);

template matrix_add_sync<std::complex<double>>::matrix_add_sync(
    const types::shape &, int);

// Explicit instantiation for matrix_add base class operation
template OperationReturn
matrix_add<float>::operation(types::vector_const_matrix_map<float> &,
                             types::vector_matrix_map<float> &);

template OperationReturn
matrix_add<double>::operation(types::vector_const_matrix_map<double> &,
                              types::vector_matrix_map<double> &);

template OperationReturn matrix_add<std::complex<float>>::operation(
    types::vector_const_matrix_map<std::complex<float>> &,
    types::vector_matrix_map<std::complex<float>> &);

template OperationReturn matrix_add<std::complex<double>>::operation(
    types::vector_const_matrix_map<std::complex<double>> &,
    types::vector_matrix_map<std::complex<double>> &);

// Base class: ctors
template matrix_add<std::complex<float>>::matrix_add();

template matrix_add<std::complex<double>>::matrix_add();

template matrix_add<std::complex<float>>::matrix_add(const types::shape &, int);

template matrix_add<std::complex<double>>::matrix_add(const types::shape &,
                                                      int);

// Base class: validate_shape
template void
matrix_add<std::complex<float>>::validate_shape(const types::vector_shapes &,
                                                const types::vector_shapes &,
                                                const std::string &);

template void
matrix_add<std::complex<double>>::validate_shape(const types::vector_shapes &,
                                                 const types::vector_shapes &,
                                                 const std::string &);

// Base class: compute_sizes
template std::vector<size_t>
matrix_add<std::complex<float>>::compute_sizes(const types::vector_shapes &);

template std::vector<size_t>
matrix_add<std::complex<double>>::compute_sizes(const types::vector_shapes &);

// Base class: compute_output_shapes
template types::vector_shapes
matrix_add<std::complex<float>>::compute_output_shapes(
    const types::vector_shapes &);

template types::vector_shapes
matrix_add<std::complex<double>>::compute_output_shapes(
    const types::vector_shapes &);

// Explicit instantiations for matrix_add_pdu base methods
template matrix_add_pdu<float>::matrix_add_pdu(const types::shape &, int);
template matrix_add_pdu<double>::matrix_add_pdu(const types::shape &, int);
template matrix_add_pdu<std::complex<float>>::matrix_add_pdu(
    const types::shape &, int);
template matrix_add_pdu<std::complex<double>>::matrix_add_pdu(
    const types::shape &, int);

template size_t matrix_add_pdu<float>::element_count() const;
template size_t matrix_add_pdu<double>::element_count() const;
template size_t matrix_add_pdu<std::complex<float>>::element_count() const;
template size_t matrix_add_pdu<std::complex<double>>::element_count() const;

template void matrix_add_pdu<float>::handle_pdu(int, const pmt::pmt_t &);
template void matrix_add_pdu<double>::handle_pdu(int, const pmt::pmt_t &);
template void
matrix_add_pdu<std::complex<float>>::handle_pdu(int, const pmt::pmt_t &);
template void
matrix_add_pdu<std::complex<double>>::handle_pdu(int, const pmt::pmt_t &);

template bool matrix_add_pdu<float>::decode_and_store(int, const pmt::pmt_t &,
                                                      size_t);
template bool matrix_add_pdu<double>::decode_and_store(int, const pmt::pmt_t &,
                                                       size_t);
template bool
matrix_add_pdu<std::complex<float>>::decode_and_store(int, const pmt::pmt_t &,
                                                      size_t);
template bool
matrix_add_pdu<std::complex<double>>::decode_and_store(int, const pmt::pmt_t &,
                                                       size_t);

template pmt::pmt_t
matrix_add_pdu<float>::encode_vector(const std::vector<float> &);
template pmt::pmt_t
matrix_add_pdu<double>::encode_vector(const std::vector<double> &);
template pmt::pmt_t matrix_add_pdu<std::complex<float>>::encode_vector(
    const std::vector<std::complex<float>> &);
template pmt::pmt_t matrix_add_pdu<std::complex<double>>::encode_vector(
    const std::vector<std::complex<double>> &);
} // namespace linalg
} // namespace gr
