/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "matrix_elementwise_divide_impl.h"
#include <algorithm>
#include <cmath>
#include <gnuradio/gr_complex.h>
#include <gnuradio/io_signature.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/types.h>
#include <limits>
#include <pmt/pmt.h>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace gr {
namespace linalg {

// Constants for division by zero detection
template <typename T>
constexpr T DIVISION_EPSILON = std::numeric_limits<T>::epsilon() * T(100);

template <>
constexpr float DIVISION_EPSILON<float> = 1e-7f;

template <>
constexpr double DIVISION_EPSILON<double> = 1e-15;

// Helper functions
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
typename matrix_elementwise_divide_sync<Scalar>::sptr
matrix_elementwise_divide_sync<Scalar>::make(const types::shape &shape,
                                             const int num_inputs) {
  return gnuradio::make_block_sptr<matrix_elementwise_divide_sync_impl<Scalar>>(
      shape, num_inputs);
}

// matrix_elementwise_divide_sync public ctor used by factory

template <typename Scalar>
matrix_elementwise_divide_sync<Scalar>::matrix_elementwise_divide_sync(
    const types::shape &shape, const int num_inputs)
    : matrix_elementwise_divide<Scalar>() {}

// Impl ctor: most-derived, initializes the virtual base linalg_base

template <typename Scalar>
matrix_elementwise_divide_sync_impl<
    Scalar>::matrix_elementwise_divide_sync_impl(const types::shape &shape,
                                                 const int num_inputs)
    : linalg_base<Scalar>(
          "matrix_elementwise_divide", _repeat_shape(shape, num_inputs),
          _make_names("input_", num_inputs), _single_shape(shape),
          std::vector<std::string>(), array_broadcast_type::MATRIX,
          error_tag_t::NONE, error_pdu_p::NONE),
      linalg_base_sync<Scalar>(
          "matrix_elementwise_divide", _repeat_shape(shape, num_inputs),
          _make_names("input_", num_inputs), _single_shape(shape),
          std::vector<std::string>(), array_broadcast_type::MATRIX,
          error_tag_t::NONE, error_pdu_p::NONE, gr::block::TPP_ALL_TO_ALL),
      matrix_elementwise_divide<Scalar>(shape, num_inputs),
      matrix_elementwise_divide_sync<Scalar>(shape, num_inputs) {
  // Initialize the base class with the shape and input/output names
}

// matrix_elementwise_divide base trivial ctors (kept out-of-line)

template <typename Scalar>
matrix_elementwise_divide<Scalar>::matrix_elementwise_divide()
    : linalg_base<Scalar>() {}

template <typename Scalar>
matrix_elementwise_divide<Scalar>::matrix_elementwise_divide(
    const types::shape & /*shape*/, int /*num_inputs*/)
    : linalg_base<Scalar>() {}

// Division by zero handling utilities

template <typename Scalar>
bool matrix_elementwise_divide<Scalar>::is_zero(const Scalar &val) const {
  if constexpr (std::is_same<Scalar, std::complex<float>>::value ||
                std::is_same<Scalar, std::complex<double>>::value) {
    return std::abs(val) < DIVISION_EPSILON<typename Scalar::value_type>;
  } else {
    return std::abs(val) < DIVISION_EPSILON<Scalar>;
  }
}

template <typename Scalar>
Scalar matrix_elementwise_divide<Scalar>::safe_divide(
    const Scalar &numerator, const Scalar &denominator) const {
  if (is_zero(denominator)) {
    // Return infinity for non-zero numerator, NaN for 0/0
    if (is_zero(numerator)) {
      if constexpr (std::is_same<Scalar, std::complex<float>>::value ||
                    std::is_same<Scalar, std::complex<double>>::value) {
        using real_type = typename Scalar::value_type;
        real_type nan = std::numeric_limits<real_type>::quiet_NaN();
        return Scalar(nan, nan);
      } else {
        return std::numeric_limits<Scalar>::quiet_NaN();
      }
    } else {
      // Preserve sign: positive numerator -> +inf, negative numerator -> -inf
      if constexpr (std::is_same<Scalar, std::complex<float>>::value ||
                    std::is_same<Scalar, std::complex<double>>::value) {
        // For complex numbers, create infinity with proper structure
        using real_type = typename Scalar::value_type;
        real_type inf = std::numeric_limits<real_type>::infinity();
        return Scalar(
            inf,
            inf); // Return complex number with inf real and imaginary parts
      } else {
        // For real numbers, preserve sign
        return (numerator < Scalar(0))
                   ? -std::numeric_limits<Scalar>::infinity()
                   : std::numeric_limits<Scalar>::infinity();
      }
    }
  }
  return numerator / denominator;
}

// matrix_elementwise_divide base operation implementation

template <typename Scalar>
OperationReturn matrix_elementwise_divide<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {
  if (input_matrices.size() < 2 || output_matrices.empty())
    return OperationReturn::INVALID_SHAPE;

  // Initialize output matrix with the first input matrix (dividend)
  *output_matrices[0] = *input_matrices[0];

  // Element-wise divide by all subsequent input matrices (divisors)
  for (size_t i = 1; i < input_matrices.size(); ++i) {
    // Check for compatible shapes (broadcasting support)
    if (input_matrices[i]->rows() != output_matrices[0]->rows() ||
        input_matrices[i]->cols() != output_matrices[0]->cols()) {
      return OperationReturn::INVALID_SHAPE;
    }

    // Perform element-wise division using Eigen's cwiseQuotient
    *output_matrices[0] = output_matrices[0]->cwiseQuotient(*input_matrices[i]);
  }

  return OperationReturn::SUCCESS;
}

// Provide a specific operation that divides inputs element-wise with zero
// handling

template <typename Scalar>
OperationReturn matrix_elementwise_divide_sync_impl<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {
  if (input_matrices.size() < 2 || output_matrices.empty())
    return OperationReturn::INVALID_SHAPE;

  auto &out = *output_matrices[0];

  // Initialize with first input (dividend)
  out = *input_matrices[0];

  // Element-wise divide by remaining inputs (divisors)
  for (size_t i = 1; i < input_matrices.size(); ++i) {
    const auto &divisor = *input_matrices[i];

    // Check shape compatibility
    if (divisor.rows() != out.rows() || divisor.cols() != out.cols()) {
      return OperationReturn::INVALID_SHAPE;
    }

    // Safe element-wise division with zero checking
    for (int row = 0; row < out.rows(); ++row) {
      for (int col = 0; col < out.cols(); ++col) {
        out(row, col) = this->safe_divide(out(row, col), divisor(row, col));
      }
    }
  }

  return OperationReturn::SUCCESS;
}

// Static helpers moved from header

template <typename Scalar>
void matrix_elementwise_divide<Scalar>::validate_shape(
    const types::vector_shapes &input_shapes,
    const types::vector_shapes &output_shapes, const std::string &name) {
  // Use MATRIX broadcast to allow compatible shapes
  linalg_base<Scalar>::validate_shapes(input_shapes, output_shapes, name,
                                       array_broadcast_type::MATRIX);
}

template <typename Scalar>
std::vector<size_t> matrix_elementwise_divide<Scalar>::compute_sizes(
    const types::vector_shapes &shapes) {
  return types::compute_sizes<Scalar>(shapes);
}

template <typename Scalar>
types::vector_shapes matrix_elementwise_divide<Scalar>::compute_output_shapes(
    const types::vector_shapes &input_shapes) {
  if (input_shapes.empty())
    return {};
  return types::vector_shapes{input_shapes.front()};
}

// ---------------- PDU implementation ----------------

// Base PDU constructor and member functions

template <typename Scalar>
matrix_elementwise_divide_pdu<Scalar>::matrix_elementwise_divide_pdu(
    const types::shape &shape, const int num_inputs)
    : linalg_base<Scalar>("matrix_elementwise_divide_pdu",
                          _repeat_shape(shape, num_inputs),
                          _make_names("in_", num_inputs), _single_shape(shape),
                          _make_names("out_", 1), array_broadcast_type::MATRIX,
                          error_tag_t::NONE, error_pdu_p::NONE),
      linalg_base_pdu<Scalar>(
          "matrix_elementwise_divide_pdu", _repeat_shape(shape, num_inputs),
          _make_names("in_", num_inputs), _single_shape(shape),
          _make_names("out_", 1), array_broadcast_type::MATRIX,
          error_tag_t::NONE, error_pdu_p::NONE, PDU_UPDATE::ANY_INPUT,
          MESSAGE_HANDLER_MODE::DEFAULT),
      matrix_elementwise_divide<Scalar>(shape, num_inputs) {
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
size_t matrix_elementwise_divide_pdu<Scalar>::element_count() const {
  return static_cast<size_t>((d_rows <= 0 || d_cols <= 0) ? 0
                                                          : (d_rows * d_cols));
}

template <typename Scalar>
void matrix_elementwise_divide_pdu<Scalar>::handle_pdu(int idx,
                                                       const pmt::pmt_t &pdu) {
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

  // Element-wise divide across inputs (missing treated as ones for division)
  std::vector<Scalar> quotient(nelem);

  // Initialize with first input if available, otherwise ones
  if (d_seen[0]) {
    quotient = d_last_seen[0];
  } else {
    std::fill(quotient.begin(), quotient.end(), Scalar{1});
  }

  // Divide by subsequent inputs
  for (int i = 1; i < d_num_inputs; ++i) {
    if (!d_seen[static_cast<size_t>(i)])
      continue;
    const auto &divisor = d_last_seen[static_cast<size_t>(i)];
    for (size_t k = 0; k < nelem; ++k) {
      quotient[k] = this->safe_divide(quotient[k], divisor[k]);
    }
  }

  // Publish output PDU on first output port with same metadata
  pmt::pmt_t out_vec = encode_vector(quotient);
  if (out_vec == pmt::PMT_NIL)
    return;
  auto out_pdu = pmt::cons(meta, out_vec);
  this->message_port_pub(pmt::mp(this->d_output_names[0]), out_pdu);
}

// Decode helpers (same as matrix_elementwise_multiply)

template <typename Scalar>
bool matrix_elementwise_divide_pdu<Scalar>::decode_and_store(
    int idx, const pmt::pmt_t &data, size_t nelem) {
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
pmt::pmt_t matrix_elementwise_divide_pdu<Scalar>::encode_vector(
    const std::vector<Scalar> &v) {
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

// Define matrix_elementwise_divide_pdu_impl constructor out-of-class

template <typename Scalar>
matrix_elementwise_divide_pdu_impl<Scalar>::matrix_elementwise_divide_pdu_impl(
    const types::shape &shape, const int num_inputs)
    : linalg_base<Scalar>("matrix_elementwise_divide_pdu",
                          _repeat_shape(shape, num_inputs),
                          _make_names("in_", num_inputs), _single_shape(shape),
                          _make_names("out_", 1), array_broadcast_type::MATRIX,
                          error_tag_t::NONE, error_pdu_p::NONE),
      linalg_base_pdu<Scalar>(
          "matrix_elementwise_divide_pdu", _repeat_shape(shape, num_inputs),
          _make_names("in_", num_inputs), _single_shape(shape),
          _make_names("out_", 1), array_broadcast_type::MATRIX,
          error_tag_t::NONE, error_pdu_p::NONE, PDU_UPDATE::ANY_INPUT,
          MESSAGE_HANDLER_MODE::DEFAULT),
      matrix_elementwise_divide<Scalar>(),
      matrix_elementwise_divide_pdu<Scalar>(shape, num_inputs) {}

// PDU factory

template <typename Scalar>
typename matrix_elementwise_divide_pdu<Scalar>::sptr
matrix_elementwise_divide_pdu<Scalar>::make(const types::shape &shape,
                                            const int num_inputs) {
  // Use a simple derived wrapper to satisfy block_sptr requirements
  return gnuradio::make_block_sptr<matrix_elementwise_divide_pdu_impl<Scalar>>(
      shape, num_inputs);
}

// Explicit instantiation for common scalars

// Base class: ctors
template matrix_elementwise_divide<float>::matrix_elementwise_divide();
template matrix_elementwise_divide<double>::matrix_elementwise_divide();
template matrix_elementwise_divide<
    std::complex<float>>::matrix_elementwise_divide();
template matrix_elementwise_divide<
    std::complex<double>>::matrix_elementwise_divide();

template matrix_elementwise_divide<float>::matrix_elementwise_divide(
    const types::shape &, int);
template matrix_elementwise_divide<double>::matrix_elementwise_divide(
    const types::shape &, int);
template matrix_elementwise_divide<
    std::complex<float>>::matrix_elementwise_divide(const types::shape &, int);
template matrix_elementwise_divide<
    std::complex<double>>::matrix_elementwise_divide(const types::shape &, int);

// Base class: operation
template OperationReturn matrix_elementwise_divide<float>::operation(
    types::vector_const_matrix_map<float> &, types::vector_matrix_map<float> &);

template OperationReturn matrix_elementwise_divide<double>::operation(
    types::vector_const_matrix_map<double> &,
    types::vector_matrix_map<double> &);

template OperationReturn
matrix_elementwise_divide<std::complex<float>>::operation(
    types::vector_const_matrix_map<std::complex<float>> &,
    types::vector_matrix_map<std::complex<float>> &);

template OperationReturn
matrix_elementwise_divide<std::complex<double>>::operation(
    types::vector_const_matrix_map<std::complex<double>> &,
    types::vector_matrix_map<std::complex<double>> &);

// Base class: validate_shape
template void
matrix_elementwise_divide<float>::validate_shape(const types::vector_shapes &,
                                                 const types::vector_shapes &,
                                                 const std::string &);
template void
matrix_elementwise_divide<double>::validate_shape(const types::vector_shapes &,
                                                  const types::vector_shapes &,
                                                  const std::string &);
template void matrix_elementwise_divide<std::complex<float>>::validate_shape(
    const types::vector_shapes &, const types::vector_shapes &,
    const std::string &);
template void matrix_elementwise_divide<std::complex<double>>::validate_shape(
    const types::vector_shapes &, const types::vector_shapes &,
    const std::string &);

// Base class: compute_sizes
template std::vector<size_t>
matrix_elementwise_divide<float>::compute_sizes(const types::vector_shapes &);
template std::vector<size_t>
matrix_elementwise_divide<double>::compute_sizes(const types::vector_shapes &);
template std::vector<size_t>
matrix_elementwise_divide<std::complex<float>>::compute_sizes(
    const types::vector_shapes &);
template std::vector<size_t>
matrix_elementwise_divide<std::complex<double>>::compute_sizes(
    const types::vector_shapes &);

// Base class: compute_output_shapes
template types::vector_shapes
matrix_elementwise_divide<float>::compute_output_shapes(
    const types::vector_shapes &);
template types::vector_shapes
matrix_elementwise_divide<double>::compute_output_shapes(
    const types::vector_shapes &);
template types::vector_shapes
matrix_elementwise_divide<std::complex<float>>::compute_output_shapes(
    const types::vector_shapes &);
template types::vector_shapes
matrix_elementwise_divide<std::complex<double>>::compute_output_shapes(
    const types::vector_shapes &);

// Sync factory and constructor
template typename matrix_elementwise_divide_sync<float>::sptr
matrix_elementwise_divide_sync<float>::make(const types::shape &, int);

template typename matrix_elementwise_divide_sync<double>::sptr
matrix_elementwise_divide_sync<double>::make(const types::shape &, int);

template typename matrix_elementwise_divide_sync<std::complex<float>>::sptr
matrix_elementwise_divide_sync<std::complex<float>>::make(const types::shape &,
                                                          int);

template typename matrix_elementwise_divide_sync<std::complex<double>>::sptr
matrix_elementwise_divide_sync<std::complex<double>>::make(const types::shape &,
                                                           int);

template matrix_elementwise_divide_sync<float>::matrix_elementwise_divide_sync(
    const types::shape &, int);
template matrix_elementwise_divide_sync<double>::matrix_elementwise_divide_sync(
    const types::shape &, int);
template matrix_elementwise_divide_sync<
    std::complex<float>>::matrix_elementwise_divide_sync(const types::shape &,
                                                         int);
template matrix_elementwise_divide_sync<
    std::complex<double>>::matrix_elementwise_divide_sync(const types::shape &,
                                                          int);

// Explicit template class instantiations for sync impl
template class matrix_elementwise_divide_sync_impl<float>;
template class matrix_elementwise_divide_sync_impl<double>;
template class matrix_elementwise_divide_sync_impl<std::complex<float>>;
template class matrix_elementwise_divide_sync_impl<std::complex<double>>;

// PDU factory
template matrix_elementwise_divide_pdu<float>::sptr
matrix_elementwise_divide_pdu<float>::make(const types::shape &, int);
template matrix_elementwise_divide_pdu<double>::sptr
matrix_elementwise_divide_pdu<double>::make(const types::shape &, int);
template matrix_elementwise_divide_pdu<std::complex<float>>::sptr
matrix_elementwise_divide_pdu<std::complex<float>>::make(const types::shape &,
                                                         int);
template matrix_elementwise_divide_pdu<std::complex<double>>::sptr
matrix_elementwise_divide_pdu<std::complex<double>>::make(const types::shape &,
                                                          int);

// Explicit template class instantiations for PDU impl
template class matrix_elementwise_divide_pdu_impl<float>;
template class matrix_elementwise_divide_pdu_impl<double>;
template class matrix_elementwise_divide_pdu_impl<std::complex<float>>;
template class matrix_elementwise_divide_pdu_impl<std::complex<double>>;

// Explicit instantiations for matrix_elementwise_divide_pdu base methods
template matrix_elementwise_divide_pdu<float>::matrix_elementwise_divide_pdu(
    const types::shape &, int);
template matrix_elementwise_divide_pdu<double>::matrix_elementwise_divide_pdu(
    const types::shape &, int);
template matrix_elementwise_divide_pdu<
    std::complex<float>>::matrix_elementwise_divide_pdu(const types::shape &,
                                                        int);
template matrix_elementwise_divide_pdu<
    std::complex<double>>::matrix_elementwise_divide_pdu(const types::shape &,
                                                         int);

template size_t matrix_elementwise_divide_pdu<float>::element_count() const;
template size_t matrix_elementwise_divide_pdu<double>::element_count() const;
template size_t
matrix_elementwise_divide_pdu<std::complex<float>>::element_count() const;
template size_t
matrix_elementwise_divide_pdu<std::complex<double>>::element_count() const;

template void
matrix_elementwise_divide_pdu<float>::handle_pdu(int, const pmt::pmt_t &);
template void
matrix_elementwise_divide_pdu<double>::handle_pdu(int, const pmt::pmt_t &);
template void matrix_elementwise_divide_pdu<std::complex<float>>::handle_pdu(
    int, const pmt::pmt_t &);
template void matrix_elementwise_divide_pdu<std::complex<double>>::handle_pdu(
    int, const pmt::pmt_t &);

template bool
matrix_elementwise_divide_pdu<float>::decode_and_store(int, const pmt::pmt_t &,
                                                       size_t);
template bool
matrix_elementwise_divide_pdu<double>::decode_and_store(int, const pmt::pmt_t &,
                                                        size_t);
template bool
matrix_elementwise_divide_pdu<std::complex<float>>::decode_and_store(
    int, const pmt::pmt_t &, size_t);
template bool
matrix_elementwise_divide_pdu<std::complex<double>>::decode_and_store(
    int, const pmt::pmt_t &, size_t);

template pmt::pmt_t
matrix_elementwise_divide_pdu<float>::encode_vector(const std::vector<float> &);
template pmt::pmt_t matrix_elementwise_divide_pdu<double>::encode_vector(
    const std::vector<double> &);
template pmt::pmt_t
matrix_elementwise_divide_pdu<std::complex<float>>::encode_vector(
    const std::vector<std::complex<float>> &);
template pmt::pmt_t
matrix_elementwise_divide_pdu<std::complex<double>>::encode_vector(
    const std::vector<std::complex<double>> &);

// Explicit instantiations for base class helper methods
template bool matrix_elementwise_divide<float>::is_zero(const float &) const;
template bool matrix_elementwise_divide<double>::is_zero(const double &) const;
template bool matrix_elementwise_divide<std::complex<float>>::is_zero(
    const std::complex<float> &) const;
template bool matrix_elementwise_divide<std::complex<double>>::is_zero(
    const std::complex<double> &) const;

template float
matrix_elementwise_divide<float>::safe_divide(const float &,
                                              const float &) const;
template double
matrix_elementwise_divide<double>::safe_divide(const double &,
                                               const double &) const;
template std::complex<float>
matrix_elementwise_divide<std::complex<float>>::safe_divide(
    const std::complex<float> &, const std::complex<float> &) const;
template std::complex<double>
matrix_elementwise_divide<std::complex<double>>::safe_divide(
    const std::complex<double> &, const std::complex<double> &) const;

} // namespace linalg
} // namespace gr