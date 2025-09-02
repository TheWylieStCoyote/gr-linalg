/* -*- c++ -*- */
/*
 * Copyright 2025 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include <Eigen/Dense>
#include <algorithm>
#include <gnuradio/linalg/array_slice.h>
#include <gnuradio/logger.h>
#include <stdexcept>

namespace gr {
namespace linalg {

// Static helper for shape computation
template <typename Scalar>
const types::vector_shapes array_slice<Scalar>::compute_output_shapes(
    const types::vector_shapes &input_shapes,
    const std::vector<SliceSpec> &slice_specs) {
  if (input_shapes.empty()) {
    throw std::invalid_argument(
        "array_slice requires at least one input shape");
  }

  const auto &input_shape = input_shapes[0];

  if (slice_specs.size() > input_shape.size()) {
    throw std::invalid_argument("More slice specs than input dimensions");
  }

  types::shape output_shape;

  // Apply slice specs to each dimension
  for (size_t dim = 0; dim < input_shape.size(); ++dim) {
    int dim_size = input_shape[dim];

    if (dim < slice_specs.size()) {
      const auto &spec = slice_specs[dim];

      // Normalize indices (handle negative values)
      int start = spec.start;
      int stop = spec.stop;
      int step = spec.step;

      if (step == 0) {
        throw std::invalid_argument("Slice step cannot be zero");
      }

      // Handle negative indices
      if (start < 0)
        start += dim_size;
      if (stop < 0)
        stop += dim_size;

      // Clamp to valid range
      start = std::max(0, std::min(start, dim_size - 1));
      stop = std::max(0, std::min(stop, dim_size));

      // Calculate output size for this dimension
      int output_dim_size;
      if (step > 0) {
        output_dim_size = std::max(0, (stop - start + step - 1) / step);
      } else {
        output_dim_size = std::max(0, (start - stop - step - 1) / (-step));
      }

      output_shape.push_back(output_dim_size);
    } else {
      // No slice spec for this dimension, keep original size
      output_shape.push_back(dim_size);
    }
  }

  return {output_shape};
}

template <typename Scalar>
const int array_slice<Scalar>::compute_output_vlen(
    const types::vector_shapes &input_shapes,
    const std::vector<SliceSpec> &slice_specs) {
  auto output_shapes = compute_output_shapes(input_shapes, slice_specs);
  int total_size = 1;
  for (int dim : output_shapes[0]) {
    total_size *= dim;
  }
  return total_size;
}

// Template-based constructor implementations
template <typename Scalar>
array_slice<Scalar>::array_slice() {}

template <typename Scalar>
array_slice<Scalar>::array_slice(const types::shape &input_shape,
                                 const std::vector<SliceSpec> &slice_specs)
    : d_input_shape(input_shape), d_slice_specs(slice_specs) {
  if (slice_specs.size() > input_shape.size()) {
    throw std::invalid_argument("More slice specs than input dimensions");
  }
}

template <typename Scalar>
array_slice<Scalar>::array_slice(const types::shape &input_shape,
                                 const std::vector<int> &indices)
    : d_input_shape(input_shape) {
  if (indices.size() > input_shape.size()) {
    throw std::invalid_argument("More indices than input dimensions");
  }

  // Convert simple indices to SliceSpecs
  // For each dimension, if an index is provided, create SliceSpec(index,
  // index+1, 1) This selects a single element along that dimension
  d_slice_specs.reserve(indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    int idx = indices[i];
    if (idx < 0) {
      idx += input_shape[i]; // Handle negative indices
    }
    d_slice_specs.emplace_back(idx, idx + 1, 1);
  }
}

template <typename Scalar>
int array_slice<Scalar>::normalize_index(int index, int dimension_size) const {
  if (index < 0) {
    index += dimension_size;
  }
  return std::max(0, std::min(index, dimension_size - 1));
}

template <typename Scalar>
typename array_slice<Scalar>::SliceSpec
array_slice<Scalar>::normalize_slice(const SliceSpec &spec,
                                     int dimension_size) const {
  SliceSpec normalized = spec;

  // Handle negative indices
  if (normalized.start < 0)
    normalized.start += dimension_size;
  if (normalized.stop < 0)
    normalized.stop += dimension_size;

  // Clamp to valid range
  normalized.start =
      std::max(0, std::min(normalized.start, dimension_size - 1));
  normalized.stop = std::max(0, std::min(normalized.stop, dimension_size));

  return normalized;
}

template <typename Scalar>
OperationReturn array_slice<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_arrays,
    types::vector_matrix_map<Scalar> &output_arrays) {
  if (input_arrays.empty()) {
    return OperationReturn::INVALID_SHAPE;
  }

  if (output_arrays.empty()) {
    return OperationReturn::INVALID_SHAPE;
  }

  const auto &input = *input_arrays[0];
  auto &output = *output_arrays[0];

  try {
    // Implement array slicing based on input dimensionality
    if (d_input_shape.size() == 1) {
      // 1D vector slicing - GNU Radio provides data as contiguous arrays
      // For 1D vectors, input should be mapped as N×1 or 1×N matrix
      // Access the data directly as a vector
      Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> input_vec(
          input.data(), input.size());
      Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> output_vec(
          output.data(), output.size());

      if (d_slice_specs.empty()) {
        output_vec = input_vec;
      } else {
        const auto &spec = normalize_slice(d_slice_specs[0], input_vec.size());

        if (spec.step == 1) {
          // Simple contiguous slice
          int start = spec.start;
          int length = spec.stop - spec.start;
          output_vec = input_vec.segment(start, length);
        } else {
          // Strided slice
          int output_idx = 0;
          if (spec.step > 0) {
            for (int i = spec.start; i < spec.stop; i += spec.step) {
              output_vec[output_idx++] = input_vec[i];
            }
          } else {
            for (int i = spec.start; i > spec.stop; i += spec.step) {
              output_vec[output_idx++] = input_vec[i];
            }
          }
        }
      }
    } else if (d_input_shape.size() == 2) {
      // 2D matrix slicing
      perform_2d_slice(input, output);
    } else {
      // Higher dimensional slicing not yet implemented
      return OperationReturn::NOT_IMPLEMENTED;
    }

    return OperationReturn::SUCCESS;
  } catch (const std::exception &e) {
    return OperationReturn::FAILURE;
  }
}

template <typename Scalar>
void array_slice<Scalar>::perform_2d_slice(
    const Eigen::Map<
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &input,
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &output) {
  SliceSpec row_spec = d_slice_specs.empty()
                           ? SliceSpec(0, input.rows(), 1)
                           : normalize_slice(d_slice_specs[0], input.rows());
  SliceSpec col_spec = d_slice_specs.size() < 2
                           ? SliceSpec(0, input.cols(), 1)
                           : normalize_slice(d_slice_specs[1], input.cols());

  // Simple case: contiguous blocks
  if (row_spec.step == 1 && col_spec.step == 1) {
    int row_start = row_spec.start;
    int row_count = row_spec.stop - row_spec.start;
    int col_start = col_spec.start;
    int col_count = col_spec.stop - col_spec.start;

    output = input.block(row_start, col_start, row_count, col_count);
  } else {
    // Strided slice - need to copy element by element
    int output_row = 0;
    int row_step = row_spec.step;
    int col_step = col_spec.step;

    for (int r = row_spec.start;
         (row_step > 0) ? (r < row_spec.stop) : (r > row_spec.stop);
         r += row_step) {
      int output_col = 0;
      for (int c = col_spec.start;
           (col_step > 0) ? (c < col_spec.stop) : (c > col_spec.stop);
           c += col_step) {
        output(output_row, output_col) = input(r, c);
        output_col++;
      }
      output_row++;
    }
  }
}

// Sync block implementation
template <typename Scalar>
array_slice_sync<Scalar>::array_slice_sync(
    const types::shape &input_shape,
    const std::vector<typename array_slice<Scalar>::SliceSpec> &slice_specs)
    : linalg_base<Scalar>(
          "array_slice_sync", {input_shape}, std::vector<std::string>{"input"},
          array_slice<Scalar>::compute_output_shapes({input_shape},
                                                     slice_specs),
          std::vector<std::string>{"output"}, array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE),
      array_slice<Scalar>(input_shape, slice_specs),
      linalg_base_sync<Scalar>(
          "array_slice_sync", {input_shape}, std::vector<std::string>{"input"},
          array_slice<Scalar>::compute_output_shapes({input_shape},
                                                     slice_specs),
          std::vector<std::string>{"output"}, array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE, gr::block::TPP_ALL_TO_ALL) {}

template <typename Scalar>
typename array_slice_sync<Scalar>::sptr array_slice_sync<Scalar>::make(
    const types::shape &input_shape,
    const std::vector<typename array_slice<Scalar>::SliceSpec> &slice_specs) {
  return std::make_shared<array_slice_sync<Scalar>>(input_shape, slice_specs);
}

// PDU block implementation
template <typename Scalar>
array_slice_pdu<Scalar>::array_slice_pdu(
    const types::vector_shapes &input_shapes,
    const std::vector<typename array_slice<Scalar>::SliceSpec> &slice_specs)
    : linalg_base<Scalar>(
          "array_slice_pdu", input_shapes, {"in"},
          array_slice<Scalar>::compute_output_shapes(input_shapes, slice_specs),
          {"out"}, array_broadcast_type::CUSTOM, error_tag_t::NONE,
          error_pdu_p::NONE),
      array_slice<Scalar>(input_shapes[0], slice_specs),
      linalg_base_pdu<Scalar>(
          "array_slice_pdu", input_shapes, {"in"},
          array_slice<Scalar>::compute_output_shapes(input_shapes, slice_specs),
          {"out"}, array_broadcast_type::CUSTOM, error_tag_t::NONE,
          error_pdu_p::NONE) {}

template <typename Scalar>
typename array_slice_pdu<Scalar>::sptr array_slice_pdu<Scalar>::make(
    const types::vector_shapes &input_shapes,
    const std::vector<typename array_slice<Scalar>::SliceSpec> &slice_specs) {
  return std::make_shared<array_slice_pdu<Scalar>>(input_shapes, slice_specs);
}

// Explicit template instantiations
template class array_slice<float>;
template class array_slice<double>;
template class array_slice<std::complex<float>>;
template class array_slice<std::complex<double>>;

template class array_slice_sync<float>;
template class array_slice_sync<double>;
template class array_slice_sync<std::complex<float>>;
template class array_slice_sync<std::complex<double>>;

template class array_slice_pdu<float>;
template class array_slice_pdu<double>;
template class array_slice_pdu<std::complex<float>>;
template class array_slice_pdu<std::complex<double>>;

} // namespace linalg
} // namespace gr
