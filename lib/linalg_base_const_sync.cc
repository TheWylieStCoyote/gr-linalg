/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "gnuradio/linalg/utils.h"
#include <gnuradio/io_signature.h>
#include <gnuradio/linalg/linalg_base_const_sync.h>
#include <gnuradio/linalg/types.h>
#include <iostream>
#include <stdexcept>

namespace gr {
namespace linalg {

template <class T>
std::vector<T> _tail_trim(const std::vector<T> &vec, size_t trim) {
  if (trim == 0 || trim >= vec.size())
    return trim == 0 ? vec : std::vector<T>{};
  return std::vector<T>(vec.begin(), vec.end() - trim);
}

template <class Scalar>
linalg_base_const_sync<Scalar>::linalg_base_const_sync(
    const std::string &name, // Name of the block
    size_t num_const_data,   // Number of constant data elements
    const std::vector<std::vector<Scalar>>
        &const_data, // Constant data for operations
    const types::vector_shapes &shape_inputs,
    const std::vector<std::string> &input_names,
    const types::vector_shapes &shape_outputs,
    const std::vector<std::string> &output_names, const_tag_t tag_const,
    const_pdu_t pdu_const, array_broadcast_type broadcast_type,
    error_tag_t tag_errors, error_pdu_p pdu_errors,
    gr::block::tag_propagation_policy_t tag_propagation_policy)
    : linalg_base_sync<Scalar>(
          name, // Name of the block
          _tail_trim(
              shape_inputs,
              num_const_data), // Adjust input shapes (remove const data shapes)
          _tail_trim(
              input_names,
              num_const_data), // Adjust input names (remove const data names)
          shape_outputs,       // Output shapes (unchanged - no const data)
          output_names,        // Output names (unchanged - no const data)
          broadcast_type, tag_errors, pdu_errors,
          tag_propagation_policy), // Tag propagation policy
      d_tag_const(tag_const), d_pdu_const(pdu_const),
      d_num_const_data(num_const_data) {
  if (shape_inputs.empty())
    throw std::runtime_error("Input shapes cannot be empty");
  if (num_const_data < 1)
    throw std::out_of_range(
        "Number of constant data elements must be at least 1");
  if (shape_inputs.size() <= num_const_data)
    throw std::out_of_range(
        "Number of constant data elements exceeds input shapes size");

  // Extract constant data shapes from the end of original input_shapes
  d_const_data_shapes.reserve(num_const_data);
  for (size_t i = shape_inputs.size() - num_const_data; i < shape_inputs.size();
       ++i) {
    d_const_data_shapes.push_back(shape_inputs[i]);
  }

  // Initialize d_const_data with proper size
  d_const_data.resize(num_const_data);

  // Validate and set constant data
  validate_const_data(const_data);
  d_const_data = const_data;
}

template <class Scalar>
void linalg_base_const_sync<Scalar>::validate_const_data(
    const std::vector<std::vector<Scalar>> &data) const {
  if (data.size() != d_num_const_data)
    throw std::runtime_error("const data count " + std::to_string(data.size()) +
                             " != expected " +
                             std::to_string(d_num_const_data));
  if (d_num_const_data == 0)
    return;
  if (d_const_data_shapes.size() < d_num_const_data)
    throw std::runtime_error(
        "insufficient constant data shapes for const data");
  for (size_t i = 0; i < d_num_const_data; ++i) {
    const auto &shape = d_const_data_shapes[i];
    const size_t expected = utils::compute_size(shape);
    if (data[i].size() != expected)
      throw std::runtime_error("const data[" + std::to_string(i) + "] size " +
                               std::to_string(data[i].size()) +
                               " != expected " + std::to_string(expected));
  }
}

template <class Scalar>
void linalg_base_const_sync<Scalar>::set_const_data(
    const std::vector<std::vector<Scalar>> &new_const_data) {
  // Validate the new constant data
  validate_const_data(new_const_data);

  // Copy the new constant data
  d_const_data = new_const_data;
}

template <class Scalar>
void linalg_base_const_sync<Scalar>::set_const_data(
    size_t index, const std::vector<Scalar> &data) {
  // Validate the index
  if (index >= d_num_const_data)
    throw std::out_of_range("const data index out of range");
  const size_t expected = utils::compute_size(d_const_data_shapes[index]);
  if (data.size() != expected)
    throw std::runtime_error("const data slot size mismatch (got " +
                             std::to_string(data.size()) + " expected " +
                             std::to_string(expected) + ")");
  // Set the constant data
  d_const_data[index] = data;
}

template <class Scalar>
linalg_base_const_sync<Scalar>::linalg_base_const_sync(
    const std::string &name, // Name of the block
    size_t num_const_data,   // Number of constant data elements
    const std::vector<std::vector<Scalar>>
        &const_data,                          // Constant data for operations
    const types::vector_shapes &shape_inputs, // Input shapes
    const types::vector_shapes &shape_outputs // Output shapes
    )
    : linalg_base_sync<Scalar>(name, _tail_trim(shape_inputs, num_const_data),
                               shape_outputs),
      d_tag_const(const_tag_t::TAGS), d_pdu_const(const_pdu_t::NONE),
      d_num_const_data(num_const_data) {
  if (shape_inputs.empty())
    throw std::runtime_error("Input shapes cannot be empty");
  if (num_const_data < 1)
    throw std::out_of_range(
        "Number of constant data elements must be at least 1");
  if (shape_inputs.size() <= num_const_data)
    throw std::out_of_range(
        "Number of constant data elements exceeds input shapes size");

  // Extract constant data shapes from the end of original input_shapes
  d_const_data_shapes.reserve(num_const_data);
  for (size_t i = shape_inputs.size() - num_const_data; i < shape_inputs.size();
       ++i) {
    d_const_data_shapes.push_back(shape_inputs[i]);
  }

  // Initialize constant data vector based on input shapes
  d_const_data.resize(num_const_data);

  // Validate and set constant data
  validate_const_data(const_data);
  d_const_data = const_data;
}

template <class Scalar>
linalg_base_const_sync<Scalar>::~linalg_base_const_sync() {
  // Cleanup if necessary
}

// Work function for linalg_base_const_sync
template <class Scalar>
int linalg_base_const_sync<Scalar>::work(int noutput_items,
                                         gr_vector_const_void_star &input_items,
                                         gr_vector_void_star &output_items) {
  using types::const_matrix_map_dynamic;
  using types::matrix_map_dynamic;
  using types::vector_const_matrix_map;
  using types::vector_matrix_map;

  // Validate constant data is properly initialized
  if (d_const_data.size() != d_num_const_data) {
    throw std::runtime_error("Constant data not properly initialized");
  }

  // Get shapes from parent (streaming inputs only) and add constant shapes
  const auto &streaming_shapes =
      this->shape_inputs; // From linalg_base_sync (trimmed)
  const auto &out_shapes = this->shape_outputs;

  // Build combined input shapes: streaming + constant
  types::vector_shapes combined_input_shapes;
  combined_input_shapes.reserve(streaming_shapes.size() +
                                d_const_data_shapes.size());
  combined_input_shapes.insert(combined_input_shapes.end(),
                               streaming_shapes.begin(),
                               streaming_shapes.end());
  combined_input_shapes.insert(combined_input_shapes.end(),
                               d_const_data_shapes.begin(),
                               d_const_data_shapes.end());

  const int nin_streaming = static_cast<int>(streaming_shapes.size());
  const int nin_total = static_cast<int>(combined_input_shapes.size());
  const int nout = static_cast<int>(out_shapes.size());

  // Helper to get rows/cols from a shape
  auto rows_cols_from_shape = [](const types::shape &s) -> std::pair<int, int> {
    if (s.empty())
      return {1, 0};
    if (s.size() == 1)
      return {1, s[0]};
    int rows = s[0];
    int cols = 1;
    for (size_t i = 1; i < s.size(); ++i)
      cols *= s[i];
    return {rows, cols};
  };

  // Precompute element counts for streaming inputs and outputs
  std::vector<int> streaming_elems(nin_streaming, 0), out_elems(nout, 0);
  for (int p = 0; p < nin_streaming; ++p) {
    auto rc = rows_cols_from_shape(streaming_shapes[p]);
    streaming_elems[p] = rc.first * rc.second;
  }
  for (int p = 0; p < nout; ++p) {
    auto rc = rows_cols_from_shape(out_shapes[p]);
    out_elems[p] = rc.first * rc.second;
  }

  for (int i = 0; i < noutput_items; ++i) {
    // Build maps for this item
    std::vector<const_matrix_map_dynamic<Scalar>> in_maps_storage;
    std::vector<matrix_map_dynamic<Scalar>> out_maps_storage;
    in_maps_storage.reserve(nin_total);
    out_maps_storage.reserve(nout);

    vector_const_matrix_map<Scalar> in_maps;
    vector_matrix_map<Scalar> out_maps;
    in_maps.reserve(nin_total);
    out_maps.reserve(nout);

    // Streaming inputs
    for (int p = 0; p < nin_streaming; ++p) {
      auto rc = rows_cols_from_shape(streaming_shapes[p]);
      auto ptr = static_cast<const Scalar *>(input_items[p]);
      const Scalar *offset_ptr =
          ptr + static_cast<size_t>(i) * streaming_elems[p];
      in_maps_storage.emplace_back(offset_ptr, rc.first, rc.second);
      in_maps.push_back(&in_maps_storage.back());
    }

    // Constant data inputs (no time offset, same for all items)
    for (size_t c = 0; c < d_num_const_data; ++c) {
      auto rc = rows_cols_from_shape(d_const_data_shapes[c]);
      const Scalar *const_ptr = d_const_data[c].data();
      in_maps_storage.emplace_back(const_ptr, rc.first, rc.second);
      in_maps.push_back(&in_maps_storage.back());
    }

    // Output maps
    for (int p = 0; p < nout; ++p) {
      auto rc = rows_cols_from_shape(out_shapes[p]);
      auto ptr = static_cast<Scalar *>(output_items[p]);
      Scalar *offset_ptr = ptr + static_cast<size_t>(i) * out_elems[p];
      out_maps_storage.emplace_back(offset_ptr, rc.first, rc.second);
      out_maps.push_back(&out_maps_storage.back());
    }

    // Call the derived class operation method
    auto ret = this->operation(in_maps, out_maps);
    if (ret != OperationReturn::SUCCESS) {
      // Handle errors (simplified version from parent)
      if (this->tag_errors == error_tag_t::TAGS) {
        for (int p = 0; p < nout; ++p) {
          const uint64_t abs_offset = this->nitems_written(p) + i;
          pmt::pmt_t error_dict = pmt::make_dict();
          error_dict = pmt::dict_add(
              error_dict, pmt::intern("error"),
              pmt::intern("linalg_base_const_sync operation failed"));
          error_dict = pmt::dict_add(error_dict, pmt::intern("index"),
                                     pmt::from_long(i));
          this->add_item_tag(p, abs_offset, pmt::intern("Error"), error_dict);
        }
      }
    }
  }

  return noutput_items;
}

template class linalg_base_const_sync<float>;
template class linalg_base_const_sync<double>;
template class linalg_base_const_sync<std::complex<float>>;
template class linalg_base_const_sync<std::complex<double>>;

} /* namespace linalg */
} /* namespace gr */
