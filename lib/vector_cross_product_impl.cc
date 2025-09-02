/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "vector_cross_product_impl.h"
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

// Helper functions
static inline types::vector_shapes _make_3d_vector_shapes(int n = 2) {
  types::vector_shapes v;
  v.reserve(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i)
    v.push_back(types::shape{3, 1}); // 3D column vectors
  return v;
}

static inline types::vector_shapes _single_3d_vector_shape() {
  types::vector_shapes v;
  v.push_back(types::shape{3, 1}); // Single 3D column vector
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
typename vector_cross_product_sync<Scalar>::sptr
vector_cross_product_sync<Scalar>::make(bool validate_3d) {
  return gnuradio::make_block_sptr<vector_cross_product_sync_impl<Scalar>>(
      validate_3d);
}

// vector_cross_product_sync public ctor used by factory

template <typename Scalar>
vector_cross_product_sync<Scalar>::vector_cross_product_sync(bool validate_3d)
    : vector_cross_product<Scalar>(validate_3d) {}

// Impl ctor: most-derived, initializes the virtual base linalg_base

template <typename Scalar>
vector_cross_product_sync_impl<Scalar>::vector_cross_product_sync_impl(
    bool validate_3d)
    : linalg_base<Scalar>("vector_cross_product", _make_3d_vector_shapes(2),
                          _make_names("input_", 2), _single_3d_vector_shape(),
                          _make_names("out_", 1), array_broadcast_type::CUSTOM,
                          error_tag_t::NONE, error_pdu_p::NONE),
      vector_cross_product<Scalar>(validate_3d),
      linalg_base_sync<Scalar>(
          "vector_cross_product", _make_3d_vector_shapes(2),
          _make_names("input_", 2), _single_3d_vector_shape(),
          _make_names("out_", 1), array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE, gr::block::TPP_ALL_TO_ALL),
      vector_cross_product_sync<Scalar>(validate_3d),
      d_validate_3d(validate_3d) {
  // Initialize the base class with 3D vector shapes
}

// Cross product computation utilities

template <typename Scalar>
bool vector_cross_product_sync_impl<Scalar>::validate_3d_vector(
    const types::const_matrix_map_dynamic<Scalar> &vec) {
  if (!d_validate_3d) {
    return true; // Skip validation if not required
  }

  // Check if vector has exactly 3 elements
  if (vec.rows() == 3 && vec.cols() == 1) {
    return true; // 3x1 column vector
  } else if (vec.rows() == 1 && vec.cols() == 3) {
    return true; // 1x3 row vector
  } else if (vec.size() == 3) {
    return true; // Flattened 3-element vector
  }

  return false;
}

template <typename Scalar>
std::vector<Scalar>
vector_cross_product_sync_impl<Scalar>::compute_cross_product_3d(
    const std::vector<Scalar> &a, const std::vector<Scalar> &b) {

  if (a.size() < 3 || b.size() < 3) {
    throw std::invalid_argument("Cross product requires 3D vectors");
  }

  // Cross product formula: a × b = (a₂b₃-a₃b₂, a₃b₁-a₁b₃, a₁b₂-a₂b₁)
  std::vector<Scalar> result(3);
  result[0] = a[1] * b[2] - a[2] * b[1]; // a₂b₃ - a₃b₂
  result[1] = a[2] * b[0] - a[0] * b[2]; // a₃b₁ - a₁b₃
  result[2] = a[0] * b[1] - a[1] * b[0]; // a₁b₂ - a₂b₁

  return result;
}

// vector_cross_product base trivial ctors (kept out-of-line)

template <typename Scalar>
vector_cross_product<Scalar>::vector_cross_product() : linalg_base<Scalar>() {}

template <typename Scalar>
vector_cross_product<Scalar>::vector_cross_product(bool validate_3d)
    : linalg_base<Scalar>(), d_validate_3d(validate_3d) {}

// vector_cross_product base operation implementation

template <typename Scalar>
OperationReturn vector_cross_product<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {
  if (input_matrices.size() != 2 || output_matrices.empty())
    return OperationReturn::INVALID_SHAPE;

  const auto &vec_a = *input_matrices[0];
  const auto &vec_b = *input_matrices[1];
  auto &result = *output_matrices[0];

  // Validate vector dimensions for 3D cross product
  if (d_validate_3d) {
    if ((vec_a.size() != 3) || (vec_b.size() != 3) || (result.size() != 3)) {
      return OperationReturn::INVALID_SHAPE;
    }
  }

  // Perform 3D cross product: a × b = (a₂b₃-a₃b₂, a₃b₁-a₁b₃, a₁b₂-a₂b₁)
  if (vec_a.size() >= 3 && vec_b.size() >= 3 && result.size() >= 3) {
    // Access elements in row-major order
    Scalar a0, a1, a2, b0, b1, b2;

    if (vec_a.rows() == 3 && vec_a.cols() == 1) {
      // Column vector
      a0 = vec_a(0, 0);
      a1 = vec_a(1, 0);
      a2 = vec_a(2, 0);
    } else if (vec_a.rows() == 1 && vec_a.cols() == 3) {
      // Row vector
      a0 = vec_a(0, 0);
      a1 = vec_a(0, 1);
      a2 = vec_a(0, 2);
    } else {
      // Flattened access
      auto data_a = vec_a.data();
      a0 = data_a[0];
      a1 = data_a[1];
      a2 = data_a[2];
    }

    if (vec_b.rows() == 3 && vec_b.cols() == 1) {
      // Column vector
      b0 = vec_b(0, 0);
      b1 = vec_b(1, 0);
      b2 = vec_b(2, 0);
    } else if (vec_b.rows() == 1 && vec_b.cols() == 3) {
      // Row vector
      b0 = vec_b(0, 0);
      b1 = vec_b(0, 1);
      b2 = vec_b(0, 2);
    } else {
      // Flattened access
      auto data_b = vec_b.data();
      b0 = data_b[0];
      b1 = data_b[1];
      b2 = data_b[2];
    }

    // Compute cross product components
    Scalar c0 = a1 * b2 - a2 * b1; // a₁b₂ - a₂b₁
    Scalar c1 = a2 * b0 - a0 * b2; // a₂b₀ - a₀b₂
    Scalar c2 = a0 * b1 - a1 * b0; // a₀b₁ - a₁b₀

    // Store result
    if (result.rows() == 3 && result.cols() == 1) {
      // Column vector result
      result(0, 0) = c0;
      result(1, 0) = c1;
      result(2, 0) = c2;
    } else if (result.rows() == 1 && result.cols() == 3) {
      // Row vector result
      result(0, 0) = c0;
      result(0, 1) = c1;
      result(0, 2) = c2;
    } else {
      // Flattened result
      auto data_result = result.data();
      data_result[0] = c0;
      data_result[1] = c1;
      data_result[2] = c2;
    }

    return OperationReturn::SUCCESS;
  }

  return OperationReturn::INVALID_SHAPE;
}

// Specific sync implementation

template <typename Scalar>
OperationReturn vector_cross_product_sync_impl<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {
  if (input_matrices.size() != 2 || output_matrices.empty())
    return OperationReturn::INVALID_SHAPE;

  const auto &vec_a = *input_matrices[0];
  const auto &vec_b = *input_matrices[1];
  auto &result = *output_matrices[0];

  // Validate 3D vectors if required
  if (!validate_3d_vector(vec_a) || !validate_3d_vector(vec_b)) {
    return OperationReturn::INVALID_SHAPE;
  }

  // Ensure output is also 3D
  if (result.size() != 3) {
    return OperationReturn::INVALID_SHAPE;
  }

  // Convert to std::vector for easier manipulation
  std::vector<Scalar> a_vec(3), b_vec(3);

  // Extract vector a elements
  if (vec_a.rows() == 3 && vec_a.cols() == 1) {
    a_vec[0] = vec_a(0, 0);
    a_vec[1] = vec_a(1, 0);
    a_vec[2] = vec_a(2, 0);
  } else if (vec_a.rows() == 1 && vec_a.cols() == 3) {
    a_vec[0] = vec_a(0, 0);
    a_vec[1] = vec_a(0, 1);
    a_vec[2] = vec_a(0, 2);
  } else {
    auto data_a = vec_a.data();
    a_vec[0] = data_a[0];
    a_vec[1] = data_a[1];
    a_vec[2] = data_a[2];
  }

  // Extract vector b elements
  if (vec_b.rows() == 3 && vec_b.cols() == 1) {
    b_vec[0] = vec_b(0, 0);
    b_vec[1] = vec_b(1, 0);
    b_vec[2] = vec_b(2, 0);
  } else if (vec_b.rows() == 1 && vec_b.cols() == 3) {
    b_vec[0] = vec_b(0, 0);
    b_vec[1] = vec_b(0, 1);
    b_vec[2] = vec_b(0, 2);
  } else {
    auto data_b = vec_b.data();
    b_vec[0] = data_b[0];
    b_vec[1] = data_b[1];
    b_vec[2] = data_b[2];
  }

  // Compute cross product
  auto cross_result = compute_cross_product_3d(a_vec, b_vec);

  // Store result
  if (result.rows() == 3 && result.cols() == 1) {
    result(0, 0) = cross_result[0];
    result(1, 0) = cross_result[1];
    result(2, 0) = cross_result[2];
  } else if (result.rows() == 1 && result.cols() == 3) {
    result(0, 0) = cross_result[0];
    result(0, 1) = cross_result[1];
    result(0, 2) = cross_result[2];
  } else {
    auto data_result = result.data();
    data_result[0] = cross_result[0];
    data_result[1] = cross_result[1];
    data_result[2] = cross_result[2];
  }

  return OperationReturn::SUCCESS;
}

// Static helpers

template <typename Scalar>
void vector_cross_product<Scalar>::validate_shape(
    const types::vector_shapes &input_shapes,
    const types::vector_shapes &output_shapes, const std::string &name) {
  // Cross product requires exactly 2 inputs and 1 output, all 3D
  if (input_shapes.size() != 2) {
    throw std::invalid_argument(name + ": requires exactly 2 input vectors");
  }

  if (output_shapes.size() != 1) {
    throw std::invalid_argument(name + ": requires exactly 1 output vector");
  }

  // Check that all shapes represent 3D vectors
  for (const auto &shape : input_shapes) {
    if (types::compute_size(shape) != 3) {
      throw std::invalid_argument(name +
                                  ": input vectors must be 3-dimensional");
    }
  }

  if (types::compute_size(output_shapes[0]) != 3) {
    throw std::invalid_argument(name + ": output vector must be 3-dimensional");
  }
}

template <typename Scalar>
std::vector<size_t> vector_cross_product<Scalar>::compute_sizes(
    const types::vector_shapes &shapes) {
  return types::compute_sizes<Scalar>(shapes);
}

template <typename Scalar>
types::vector_shapes vector_cross_product<Scalar>::compute_output_shapes(
    const types::vector_shapes &input_shapes) {
  // Cross product of two 3D vectors produces one 3D vector
  if (input_shapes.size() != 2) {
    return {};
  }
  return types::vector_shapes{types::shape{3, 1}};
}

// ---------------- PDU implementation ----------------

template <typename Scalar>
vector_cross_product_pdu<Scalar>::vector_cross_product_pdu(bool validate_3d)
    : linalg_base<Scalar>("vector_cross_product_pdu", _make_3d_vector_shapes(2),
                          _make_names("in_", 2), _single_3d_vector_shape(),
                          _make_names("out_", 1), array_broadcast_type::NONE,
                          error_tag_t::NONE, error_pdu_p::NONE),
      vector_cross_product<Scalar>(validate_3d),
      linalg_base_pdu<Scalar>("vector_cross_product_pdu",
                              _make_3d_vector_shapes(2), _make_names("in_", 2),
                              _single_3d_vector_shape(), _make_names("out_", 1),
                              array_broadcast_type::CUSTOM, error_tag_t::NONE,
                              error_pdu_p::NONE, PDU_UPDATE::ALL_INPUTS,
                              MESSAGE_HANDLER_MODE::DEFAULT),
      d_validate_3d(validate_3d) {

  d_last_seen.resize(2); // Exactly 2 inputs for cross product
  d_seen.assign(2, false);

  // Install message handlers for both input ports
  for (int i = 0; i < 2; ++i) {
    auto port = pmt::mp(this->d_input_names[static_cast<size_t>(i)]);
    this->set_msg_handler(
        port, [this, i](pmt::pmt_t msg) { this->handle_pdu(i, msg); });
  }
}

template <typename Scalar>
std::vector<Scalar> vector_cross_product_pdu<Scalar>::compute_cross_product(
    const std::vector<Scalar> &a, const std::vector<Scalar> &b) {
  if (a.size() < 3 || b.size() < 3) {
    throw std::invalid_argument("Cross product requires 3D vectors");
  }

  std::vector<Scalar> result(3);
  result[0] = a[1] * b[2] - a[2] * b[1]; // a₁b₂ - a₂b₁
  result[1] = a[2] * b[0] - a[0] * b[2]; // a₂b₀ - a₀b₂
  result[2] = a[0] * b[1] - a[1] * b[0]; // a₀b₁ - a₁b₀

  return result;
}

template <typename Scalar>
void vector_cross_product_pdu<Scalar>::handle_pdu(int idx,
                                                  const pmt::pmt_t &pdu) {
  if (!pmt::is_pair(pdu))
    return;
  auto meta = pmt::car(pdu);
  auto data = pmt::cdr(pdu);

  // Decode 3D vector
  if (!decode_and_store(idx, data, 3))
    return;

  // Wait for both inputs (cross product needs exactly 2 vectors)
  if (!d_seen[0] || !d_seen[1])
    return;

  // Compute cross product
  try {
    auto cross_result = compute_cross_product(d_last_seen[0], d_last_seen[1]);

    // Publish output PDU
    pmt::pmt_t out_vec = encode_vector(cross_result);
    if (out_vec != pmt::PMT_NIL) {
      auto out_pdu = pmt::cons(meta, out_vec);
      this->message_port_pub(pmt::mp(this->d_output_names[0]), out_pdu);
    }
  } catch (const std::exception &e) {
    // Handle cross product computation errors
  }
}

// Decode and encode helpers (similar to other PDU blocks)

template <typename Scalar>
bool vector_cross_product_pdu<Scalar>::decode_and_store(int idx,
                                                        const pmt::pmt_t &data,
                                                        size_t expected_len) {
  auto &buf = d_last_seen[static_cast<size_t>(idx)];
  buf.resize(expected_len);

  if constexpr (std::is_same<Scalar, float>::value) {
    if (!pmt::is_f32vector(data))
      return false;
    size_t len = 0;
    const float *ptr = pmt::f32vector_elements(data, len);
    if (len != expected_len)
      return false;
    std::copy(ptr, ptr + len, buf.begin());
  } else if constexpr (std::is_same<Scalar, double>::value) {
    if (!pmt::is_f64vector(data))
      return false;
    size_t len = 0;
    const double *ptr = pmt::f64vector_elements(data, len);
    if (len != expected_len)
      return false;
    std::copy(ptr, ptr + len, buf.begin());
  } else if constexpr (std::is_same<Scalar, std::complex<float>>::value) {
    if (!pmt::is_c32vector(data))
      return false;
    size_t len = 0;
    const gr_complex *ptr = pmt::c32vector_elements(data, len);
    if (len != expected_len)
      return false;
    for (size_t i = 0; i < len; ++i)
      buf[i] = ptr[i];
  } else if constexpr (std::is_same<Scalar, std::complex<double>>::value) {
    if (!pmt::is_c64vector(data))
      return false;
    size_t len = 0;
    const gr_complexd *ptr = pmt::c64vector_elements(data, len);
    if (len != expected_len)
      return false;
    for (size_t i = 0; i < len; ++i)
      buf[i] = ptr[i];
  } else {
    return false;
  }

  d_seen[static_cast<size_t>(idx)] = true;
  return true;
}

template <typename Scalar>
pmt::pmt_t
vector_cross_product_pdu<Scalar>::encode_vector(const std::vector<Scalar> &v) {
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

// PDU implementation class constructor

template <typename Scalar>
vector_cross_product_pdu_impl<Scalar>::vector_cross_product_pdu_impl(
    bool validate_3d)
    : linalg_base<Scalar>("vector_cross_product_pdu", _make_3d_vector_shapes(2),
                          _make_names("in_", 2), _single_3d_vector_shape(),
                          _make_names("out_", 1), array_broadcast_type::NONE,
                          error_tag_t::NONE, error_pdu_p::NONE),
      vector_cross_product<Scalar>(validate_3d),
      linalg_base_pdu<Scalar>("vector_cross_product_pdu",
                              _make_3d_vector_shapes(2), _make_names("in_", 2),
                              _single_3d_vector_shape(), _make_names("out_", 1),
                              array_broadcast_type::CUSTOM, error_tag_t::NONE,
                              error_pdu_p::NONE, PDU_UPDATE::ALL_INPUTS,
                              MESSAGE_HANDLER_MODE::DEFAULT),
      vector_cross_product_pdu<Scalar>(validate_3d) {}

// PDU factory

template <typename Scalar>
typename vector_cross_product_pdu<Scalar>::sptr
vector_cross_product_pdu<Scalar>::make(bool validate_3d) {
  return gnuradio::make_block_sptr<vector_cross_product_pdu_impl<Scalar>>(
      validate_3d);
}

// Explicit instantiation for common scalars

// Base class: ctors
template vector_cross_product<float>::vector_cross_product();
template vector_cross_product<double>::vector_cross_product();
template vector_cross_product<std::complex<float>>::vector_cross_product();
template vector_cross_product<std::complex<double>>::vector_cross_product();

template vector_cross_product<float>::vector_cross_product(bool);
template vector_cross_product<double>::vector_cross_product(bool);
template vector_cross_product<std::complex<float>>::vector_cross_product(bool);
template vector_cross_product<std::complex<double>>::vector_cross_product(bool);

// Base class: operation
template OperationReturn
vector_cross_product<float>::operation(types::vector_const_matrix_map<float> &,
                                       types::vector_matrix_map<float> &);

template OperationReturn vector_cross_product<double>::operation(
    types::vector_const_matrix_map<double> &,
    types::vector_matrix_map<double> &);

template OperationReturn vector_cross_product<std::complex<float>>::operation(
    types::vector_const_matrix_map<std::complex<float>> &,
    types::vector_matrix_map<std::complex<float>> &);

template OperationReturn vector_cross_product<std::complex<double>>::operation(
    types::vector_const_matrix_map<std::complex<double>> &,
    types::vector_matrix_map<std::complex<double>> &);

// Base class: static helpers
template void
vector_cross_product<float>::validate_shape(const types::vector_shapes &,
                                            const types::vector_shapes &,
                                            const std::string &);
template void
vector_cross_product<double>::validate_shape(const types::vector_shapes &,
                                             const types::vector_shapes &,
                                             const std::string &);
template void vector_cross_product<std::complex<float>>::validate_shape(
    const types::vector_shapes &, const types::vector_shapes &,
    const std::string &);
template void vector_cross_product<std::complex<double>>::validate_shape(
    const types::vector_shapes &, const types::vector_shapes &,
    const std::string &);

template std::vector<size_t>
vector_cross_product<float>::compute_sizes(const types::vector_shapes &);
template std::vector<size_t>
vector_cross_product<double>::compute_sizes(const types::vector_shapes &);
template std::vector<size_t>
vector_cross_product<std::complex<float>>::compute_sizes(
    const types::vector_shapes &);
template std::vector<size_t>
vector_cross_product<std::complex<double>>::compute_sizes(
    const types::vector_shapes &);

template types::vector_shapes
vector_cross_product<float>::compute_output_shapes(
    const types::vector_shapes &);
template types::vector_shapes
vector_cross_product<double>::compute_output_shapes(
    const types::vector_shapes &);
template types::vector_shapes
vector_cross_product<std::complex<float>>::compute_output_shapes(
    const types::vector_shapes &);
template types::vector_shapes
vector_cross_product<std::complex<double>>::compute_output_shapes(
    const types::vector_shapes &);

// Sync factory and constructor
template typename vector_cross_product_sync<float>::sptr
vector_cross_product_sync<float>::make(bool);
template typename vector_cross_product_sync<double>::sptr
vector_cross_product_sync<double>::make(bool);
template typename vector_cross_product_sync<std::complex<float>>::sptr
vector_cross_product_sync<std::complex<float>>::make(bool);
template typename vector_cross_product_sync<std::complex<double>>::sptr
vector_cross_product_sync<std::complex<double>>::make(bool);

template vector_cross_product_sync<float>::vector_cross_product_sync(bool);
template vector_cross_product_sync<double>::vector_cross_product_sync(bool);
template vector_cross_product_sync<
    std::complex<float>>::vector_cross_product_sync(bool);
template vector_cross_product_sync<
    std::complex<double>>::vector_cross_product_sync(bool);

// Explicit template class instantiations for sync impl
template class vector_cross_product_sync_impl<float>;
template class vector_cross_product_sync_impl<double>;
template class vector_cross_product_sync_impl<std::complex<float>>;
template class vector_cross_product_sync_impl<std::complex<double>>;

// PDU factory
template vector_cross_product_pdu<float>::sptr
vector_cross_product_pdu<float>::make(bool);
template vector_cross_product_pdu<double>::sptr
vector_cross_product_pdu<double>::make(bool);
template vector_cross_product_pdu<std::complex<float>>::sptr
vector_cross_product_pdu<std::complex<float>>::make(bool);
template vector_cross_product_pdu<std::complex<double>>::sptr
vector_cross_product_pdu<std::complex<double>>::make(bool);

// Explicit template class instantiations for PDU impl
template class vector_cross_product_pdu_impl<float>;
template class vector_cross_product_pdu_impl<double>;
template class vector_cross_product_pdu_impl<std::complex<float>>;
template class vector_cross_product_pdu_impl<std::complex<double>>;

// Explicit instantiations for PDU base methods
template vector_cross_product_pdu<float>::vector_cross_product_pdu(bool);
template vector_cross_product_pdu<double>::vector_cross_product_pdu(bool);
template vector_cross_product_pdu<
    std::complex<float>>::vector_cross_product_pdu(bool);
template vector_cross_product_pdu<
    std::complex<double>>::vector_cross_product_pdu(bool);

template void vector_cross_product_pdu<float>::handle_pdu(int,
                                                          const pmt::pmt_t &);
template void vector_cross_product_pdu<double>::handle_pdu(int,
                                                           const pmt::pmt_t &);
template void
vector_cross_product_pdu<std::complex<float>>::handle_pdu(int,
                                                          const pmt::pmt_t &);
template void
vector_cross_product_pdu<std::complex<double>>::handle_pdu(int,
                                                           const pmt::pmt_t &);

template bool
vector_cross_product_pdu<float>::decode_and_store(int, const pmt::pmt_t &,
                                                  size_t);
template bool
vector_cross_product_pdu<double>::decode_and_store(int, const pmt::pmt_t &,
                                                   size_t);
template bool vector_cross_product_pdu<std::complex<float>>::decode_and_store(
    int, const pmt::pmt_t &, size_t);
template bool vector_cross_product_pdu<std::complex<double>>::decode_and_store(
    int, const pmt::pmt_t &, size_t);

template pmt::pmt_t
vector_cross_product_pdu<float>::encode_vector(const std::vector<float> &);
template pmt::pmt_t
vector_cross_product_pdu<double>::encode_vector(const std::vector<double> &);
template pmt::pmt_t
vector_cross_product_pdu<std::complex<float>>::encode_vector(
    const std::vector<std::complex<float>> &);
template pmt::pmt_t
vector_cross_product_pdu<std::complex<double>>::encode_vector(
    const std::vector<std::complex<double>> &);

template std::vector<float>
vector_cross_product_pdu<float>::compute_cross_product(
    const std::vector<float> &, const std::vector<float> &);
template std::vector<double>
vector_cross_product_pdu<double>::compute_cross_product(
    const std::vector<double> &, const std::vector<double> &);
template std::vector<std::complex<float>>
vector_cross_product_pdu<std::complex<float>>::compute_cross_product(
    const std::vector<std::complex<float>> &,
    const std::vector<std::complex<float>> &);
template std::vector<std::complex<double>>
vector_cross_product_pdu<std::complex<double>>::compute_cross_product(
    const std::vector<std::complex<double>> &,
    const std::vector<std::complex<double>> &);

} // namespace linalg
} // namespace gr