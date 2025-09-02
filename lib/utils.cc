/**
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <Eigen/Dense>
#include <algorithm>
#include <functional>
#include <gnuradio/linalg/utils.h>
#include <numeric>
#include <stdexcept>

namespace gr {
namespace linalg {
namespace utils {

bool validate_shape(const types::shape &shape, const std::string &name) {
  if (shape.empty()) {
    throw std::invalid_argument(name + " shape cannot be empty");
  }

  for (auto dim : shape) {
    if (dim <= 0) {
      throw std::invalid_argument(name + " shape dimensions must be positive");
    }
  }
  return true;
}

types::shape matrix_shape(types::shape shape) {
  if (shape.size() < 2) {
    throw std::invalid_argument("Must have atleast 2 dimensions");
  }
  if (shape.size() == 2) {
    return shape;
  }
  // If more than 2 dimensions, return the last two dimensions
  types::shape new_shape;
  new_shape.push_back(shape[shape.size() - 2]);
  new_shape.push_back(shape[shape.size() - 1]);
  return new_shape;
}

int matrix_rows(types::shape shape) {
  if (shape.size() <= 1) {
    throw std::invalid_argument("Must have atleast 2 dimensions");
  }
  return shape[shape.size() - 2];
}

int matrix_cols(types::shape shape) {
  if (shape.size() <= 1) {
    throw std::invalid_argument("Must have atleast 2 dimensions");
  }
  return shape[shape.size() - 1];
}

size_t compute_size(const types::shape &shape) {
  if (shape.empty()) {
    return 0;
  }
  return std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1),
                         std::multiplies<size_t>());
}

std::vector<size_t> compute_sizes(const types::vector_shapes &shapes) {
  std::vector<size_t> sizes;
  sizes.reserve(shapes.size());
  for (const auto &shape : shapes) {
    sizes.push_back(utils::compute_size(shape));
  }
  return sizes;
}

// Shape helpers that return vector_shapes wrappers

types::vector_shapes shapes_from_shapes(const types::shape &shape) {
  types::vector_shapes result;
  result.push_back(shape);
  return result;
}

types::vector_shapes shapes_from_shapes(const types::shape &shape_0,
                                        const types::shape &shape_1) {
  types::vector_shapes result;
  result.push_back(shape_0);
  result.push_back(shape_1);
  return result;
}

types::vector_shapes shapes_from_shapes(const types::shape &shape_0,
                                        const types::shape &shape_1,
                                        const types::shape &shape_2) {
  types::vector_shapes result;
  result.push_back(shape_0);
  result.push_back(shape_1);
  result.push_back(shape_2);
  return result;
}

types::vector_shapes shapes_from_shapes(const types::shape &shape_0,
                                        const types::shape &shape_1,
                                        const types::shape &shape_2,
                                        const types::shape &shape_3) {
  types::vector_shapes result;
  result.push_back(shape_0);
  result.push_back(shape_1);
  result.push_back(shape_2);
  result.push_back(shape_3);
  return result;
}

// Broadcasting

LINALG_API types::shape compute_broadcast_shape(const types::shape &shape_a,
                                                const types::shape &shape_b,
                                                const int n_dims) {
  return compute_broadcast(shape_a, shape_b, n_dims);
}

types::shape compute_broadcast(const types::vector_shapes &input_shapes,
                               const int n_dims) {
  const auto n_input = static_cast<int>(input_shapes.size());
  int n_dims_max = 0;
  for (const auto &input_shape : input_shapes) {
    const auto size = static_cast<int>(input_shape.size());
    if (size < n_dims) {
      throw std::invalid_argument("Input shape must have at least " +
                                  std::to_string(n_dims) + " dimensions");
    }
    if (size > n_dims_max) {
      n_dims_max = size;
    }
  }

  // Compute the output shape (in reversed order for easier indexing)
  types::shape output_shape(static_cast<size_t>(n_dims_max), 1);
  for (int i = 0; i < n_input; ++i) {
    const auto &input_shape = input_shapes[static_cast<size_t>(i)];
    types::shape input_shape_rev(input_shape.rbegin(), input_shape.rend());
    for (size_t j = 0; j < input_shape_rev.size(); ++j) {
      if (input_shape_rev[j] != 1 && output_shape[j] != 1 &&
          input_shape_rev[j] != output_shape[j]) {
        throw std::invalid_argument("Incompatible shapes for broadcasting");
      }
      output_shape[j] = std::max(output_shape[j], input_shape_rev[j]);
    }
  }

  // Reverse to original order
  std::reverse(output_shape.begin(), output_shape.end());

  // If n_dims > 0 and smaller than rank, trim to last n_dims
  if (n_dims > 0 && static_cast<size_t>(n_dims) < output_shape.size()) {
    // Keep only the last n_dims dims
    types::shape trimmed;
    trimmed.reserve(static_cast<size_t>(n_dims));
    for (size_t i = output_shape.size() - static_cast<size_t>(n_dims);
         i < output_shape.size(); ++i) {
      trimmed.push_back(output_shape[i]);
    }
    return trimmed;
  }

  return output_shape;
}

types::shape compute_broadcast(const types::shape &shape_0,
                               const types::shape &shape_1,
                               const types::shape &shape_2,
                               const types::shape &shape_3, const int n_dims) {
  types::vector_shapes input_shapes{shape_0, shape_1, shape_2, shape_3};
  return compute_broadcast(input_shapes, n_dims);
}

types::shape compute_broadcast(const types::shape &shape_0,
                               const types::shape &shape_1,
                               const types::shape &shape_2, const int n_dims) {
  types::vector_shapes input_shapes{shape_0, shape_1, shape_2};
  return compute_broadcast(input_shapes, n_dims);
}

types::shape compute_broadcast(const types::shape &shape_0,
                               const types::shape &shape_1, const int n_dims) {
  types::vector_shapes input_shapes{shape_0, shape_1};
  return compute_broadcast(input_shapes, n_dims);
}

// Template function implementations moved from header

template <class T>
T mean(const std::vector<T> &vec) {
  if (vec.empty()) {
    throw std::invalid_argument("Cannot compute mean of an empty vector");
  }
  return sum(vec) / static_cast<T>(vec.size());
}

template <class T>
std::vector<T> linspace(T start, T stop, size_t num_points) {
  std::vector<T> result(num_points);
  if (num_points == 0)
    return result;
  if (num_points == 1) {
    result[0] = start;
    return result;
  }
  T step = (stop - start) / static_cast<T>(num_points - 1);
  for (size_t i = 0; i < num_points; ++i) {
    result[i] = start + i * step;
  }
  return result;
}

template <class T>
std::vector<T> zeros(size_t size) {
  return std::vector<T>(size, static_cast<T>(0));
}

template <class T>
std::vector<T> ones(size_t size) {
  return std::vector<T>(size, static_cast<T>(1));
}

template <class T>
std::vector<T> eye(size_t n) {
  std::vector<T> result(n * n, static_cast<T>(0));
  for (size_t i = 0; i < n; ++i) {
    result[i * n + i] = static_cast<T>(1);
  }
  return result;
}

template <class T>
T variance(const std::vector<T> &vec) {
  if (vec.empty()) {
    throw std::invalid_argument("Cannot compute variance of an empty vector");
  }
  T mean_val = mean(vec);
  T sum_sq_diff = 0;
  for (const auto &val : vec) {
    T diff = val - mean_val;
    sum_sq_diff += diff * diff;
  }
  return sum_sq_diff / static_cast<T>(vec.size());
}

// Explicit template instantiations for common numeric types

template float mean<float>(const std::vector<float> &);

template double mean<double>(const std::vector<double> &);

template int mean<int>(const std::vector<int> &);

template std::vector<float> linspace<float>(float, float, size_t);

template std::vector<double> linspace<double>(double, double, size_t);

template std::vector<float> zeros<float>(size_t);

template std::vector<double> zeros<double>(size_t);

template std::vector<int> zeros<int>(size_t);

template std::vector<float> ones<float>(size_t);

template std::vector<double> ones<double>(size_t);

template std::vector<int> ones<int>(size_t);

template std::vector<float> eye<float>(size_t);

template std::vector<double> eye<double>(size_t);

template std::vector<int> eye<int>(size_t);

template float variance<float>(const std::vector<float> &);

template double variance<double>(const std::vector<double> &);

} // namespace utils

// Types namespace function implementations
namespace types {

template <class T>
std::vector<size_t> compute_sizes(const vector_shapes &shapes) {
  std::vector<size_t> sizes;
  sizes.reserve(shapes.size());
  for (const auto &shape : shapes) {
    sizes.push_back(compute_size<T>(shape));
  }
  return sizes;
}

namespace tensor_utils {

bool is_broadcast_compatible(const shape &shape1, const shape &shape2) {
  // Start from the trailing dimensions
  size_t i = shape1.size();
  size_t j = shape2.size();

  while (i > 0 && j > 0) {
    --i;
    --j;
    if (shape1[i] != shape2[j] && shape1[i] != 1 && shape2[j] != 1) {
      return false;
    }
  }
  return true;
}

shape broadcast_shapes(const shape &shape1, const shape &shape2) {
  shape result;
  size_t max_dims = std::max(shape1.size(), shape2.size());
  result.reserve(max_dims);

  for (size_t i = 0; i < max_dims; ++i) {
    dim dim1 = (i < shape1.size()) ? shape1[shape1.size() - 1 - i] : 1;
    dim dim2 = (i < shape2.size()) ? shape2[shape2.size() - 1 - i] : 1;

    if (dim1 == 1) {
      result.push_back(dim2);
    } else if (dim2 == 1) {
      result.push_back(dim1);
    } else if (dim1 == dim2) {
      result.push_back(dim1);
    } else {
      throw std::invalid_argument("Shapes not compatible for broadcasting");
    }
  }

  // Reverse to get correct order
  std::reverse(result.begin(), result.end());
  return result;
}

} // namespace tensor_utils

// Explicit template instantiations for compute_sizes
template std::vector<size_t> compute_sizes<float>(const vector_shapes &);
template std::vector<size_t> compute_sizes<double>(const vector_shapes &);
template std::vector<size_t>
compute_sizes<std::complex<float>>(const vector_shapes &);
template std::vector<size_t>
compute_sizes<std::complex<double>>(const vector_shapes &);

} // namespace types
} // namespace linalg
} // namespace gr
