/* -*- c++ -*- */
/*
 * Copyright 2025 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "vector_correlate_impl.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <gnuradio/io_signature.h>
#include <gnuradio/logger.h>

namespace gr {
namespace linalg {

// Static helpers for shape computation
template <typename Scalar>
const types::vector_shapes vector_correlate<Scalar>::compute_output_shapes(
    const types::vector_shapes &input_shapes, CorrelateMode mode) {
  if (input_shapes.empty() || input_shapes.size() > 2) {
    throw std::invalid_argument(
        "vector_correlate requires 1 or 2 input shapes");
  }

  if (input_shapes.size() == 1) {
    // Auto-correlation case
    const auto &shape_a = input_shapes[0];
    if (shape_a.size() != 1) {
      throw std::invalid_argument("vector_correlate: input must be 1D vectors");
    }

    int input_len = shape_a[0];
    int output_len;

    switch (mode) {
    case FULL:
      output_len = 2 * input_len - 1;
      break;
    case VALID:
      output_len = 1; // For auto-correlation, only one valid point
      break;
    case SAME:
      output_len = input_len;
      break;
    default:
      throw std::invalid_argument("Invalid correlation mode");
    }

    return {{output_len}};
  } else {
    // Cross-correlation case
    const auto &shape_a = input_shapes[0];
    const auto &shape_b = input_shapes[1];

    if (shape_a.size() != 1 || shape_b.size() != 1) {
      throw std::invalid_argument(
          "vector_correlate: inputs must be 1D vectors");
    }

    int len_a = shape_a[0];
    int len_b = shape_b[0];
    int output_len;

    switch (mode) {
    case FULL:
      output_len = len_a + len_b - 1;
      break;
    case VALID:
      output_len = std::abs(len_a - len_b) + 1;
      break;
    case SAME:
      output_len = std::max(len_a, len_b);
      break;
    default:
      throw std::invalid_argument("Invalid correlation mode");
    }

    return {{output_len}};
  }
}

template <typename Scalar>
const int vector_correlate<Scalar>::compute_output_vlen(
    const types::vector_shapes &input_shapes, CorrelateMode mode) {
  auto output_shapes = compute_output_shapes(input_shapes, mode);
  return output_shapes[0][0];
}

template <typename Scalar>
const gr::io_signature::sptr vector_correlate<Scalar>::make_input_signature(
    const types::vector_shapes &input_shapes) {
  std::vector<int> input_sizes;
  for (const auto &shape : input_shapes) {
    input_sizes.push_back(types::compute_size<Scalar>(shape));
  }
  return gr::io_signature::makev(input_shapes.size(), input_shapes.size(),
                                 input_sizes);
}

template <typename Scalar>
const gr::io_signature::sptr vector_correlate<Scalar>::make_output_signature(
    const types::vector_shapes &input_shapes, CorrelateMode mode) {
  auto output_shapes = compute_output_shapes(input_shapes, mode);
  std::vector<int> output_sizes;
  for (const auto &shape : output_shapes) {
    output_sizes.push_back(types::compute_size<Scalar>(shape));
  }
  return gr::io_signature::makev(output_shapes.size(), output_shapes.size(),
                                 output_sizes);
}

// Template-based constructor implementations
template <typename Scalar>
vector_correlate<Scalar>::vector_correlate() {}

template <typename Scalar>
vector_correlate<Scalar>::vector_correlate(const types::shape &input_shape_a,
                                           const types::shape &input_shape_b,
                                           CorrelateMode mode, bool normalize)
    : d_mode(mode), d_normalize(normalize), d_input_shape_a(input_shape_a),
      d_input_shape_b(input_shape_b) {
  if (input_shape_a.size() != 1 || input_shape_b.size() != 1) {
    throw std::invalid_argument("vector_correlate: inputs must be 1D vectors");
  }
}

template <typename Scalar>
OperationReturn vector_correlate<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_vectors,
    types::vector_matrix_map<Scalar> &output_vectors) {
  // Perform vector correlation operation

  if (input_vectors.empty() || input_vectors.size() > 2) {
    return OperationReturn::INVALID_SHAPE;
  }

  if (output_vectors.empty()) {
    return OperationReturn::INVALID_SHAPE;
  }

  const auto &matrix_a = *input_vectors[0];
  auto &output_matrix = *output_vectors[0];

  try {
    if (input_vectors.size() == 1) {
      // Auto-correlation
      compute_autocorrelation(matrix_a, output_matrix);
    } else {
      // Cross-correlation
      const auto &matrix_b = *input_vectors[1];
      compute_crosscorrelation(matrix_a, matrix_b, output_matrix);
    }

    if (d_normalize) {
      normalize_correlation(output_matrix);
    }

    return OperationReturn::SUCCESS;
  } catch (const std::exception &e) {
    // Log error without using d_logger
    return OperationReturn::FAILURE;
  }
}

template <typename Scalar>
void vector_correlate<Scalar>::compute_autocorrelation(
    const Eigen::Map<
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &matrix_a,
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
        &output_matrix) {
  // Extract vector from matrix - handle both row vectors (1×n) and column
  // vectors (n×1)
  Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> vec_a(
      matrix_a.data(), matrix_a.size());
  Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> output(
      output_matrix.data(), output_matrix.size());
  const int len_a = vec_a.size();
  const int output_len = output.size();

  switch (d_mode) {
  case FULL: {
    // Full auto-correlation - NumPy compatible implementation
    // For auto-correlation of [1,2,3], NumPy produces [3,8,14,8,3]
    for (int k = 0; k < output_len; ++k) {
      Scalar sum = Scalar(0);

      // NumPy formula: R[k] = Σ_n a[n] * conj(a[n+k-N+1]) for valid n
      // where N = len(a), k ranges from 0 to 2*N-2
      int shift =
          k - (len_a - 1); // shift = k - (N-1), ranges from -(N-1) to +(N-1)

      for (int n = 0; n < len_a; ++n) {
        int m = n + shift; // index in second vector
        if (m >= 0 && m < len_a) {
          Scalar term;
          if constexpr (std::is_same_v<Scalar, std::complex<float>> ||
                        std::is_same_v<Scalar, std::complex<double>>) {
            term = vec_a[n] * std::conj(vec_a[m]);
          } else {
            term = vec_a[n] * vec_a[m];
          }
          sum += term;
        }
      }
      output[k] = sum;
    }
    break;
  }
  case VALID: {
    // Only zero-lag auto-correlation
    Scalar sum = Scalar(0);
    for (int n = 0; n < len_a; ++n) {
      if constexpr (std::is_same_v<Scalar, std::complex<float>> ||
                    std::is_same_v<Scalar, std::complex<double>>) {
        sum += vec_a[n] * std::conj(vec_a[n]);
      } else {
        sum += vec_a[n] * vec_a[n];
      }
    }
    output[0] = sum;
    break;
  }
  case SAME: {
    // Same size as input (central portion) - corrected indexing
    int center = (2 * len_a - 2) / 2; // Center of full correlation
    for (int k = 0; k < output_len; ++k) {
      int full_k =
          k + center - (len_a - 1) / 2; // Map to full correlation index
      int lag = full_k - (len_a - 1);
      Scalar sum = Scalar(0);

      // For each valid index pair (i, i+lag)
      for (int i = 0; i < len_a; ++i) {
        int j = i + lag;
        if (j >= 0 && j < len_a) {
          if constexpr (std::is_same_v<Scalar, std::complex<float>> ||
                        std::is_same_v<Scalar, std::complex<double>>) {
            sum += vec_a[i] * std::conj(vec_a[j]);
          } else {
            sum += vec_a[i] * vec_a[j];
          }
        }
      }
      output[k] = sum;
    }
    break;
  }
  }
}

template <typename Scalar>
void vector_correlate<Scalar>::compute_crosscorrelation(
    const Eigen::Map<
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &matrix_a,
    const Eigen::Map<
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &matrix_b,
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
        &output_matrix) {
  // Extract vectors from matrices - handle both row vectors (1×n) and column
  // vectors (n×1)
  Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> vec_a(
      matrix_a.data(), matrix_a.size());
  Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> vec_b(
      matrix_b.data(), matrix_b.size());
  Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> output(
      output_matrix.data(), output_matrix.size());
  const int len_a = vec_a.size();
  const int len_b = vec_b.size();
  const int output_len = output.size();

  switch (d_mode) {
  case FULL: {
    // Full cross-correlation compatible with NumPy's correlate() function
    // NumPy correlate(a, v) computes: R[k] = Σ_n a[n] * conj(v[n-shift])
    // where shift = k - (len_v - 1), output length is len_a + len_v - 1
    // For cross-correlation [1,2,3] with [1,0], NumPy produces [0,1,2,3]
    for (int k = 0; k < output_len; ++k) {
      Scalar sum = Scalar(0);
      int shift = k - (len_b - 1); // shift = k - (M-1) where M = len_b

      // For each element in a, compute correlation with shifted b
      for (int n = 0; n < len_a; ++n) {
        int m = n - shift; // index in b: m = n - shift
        if (m >= 0 && m < len_b) {
          Scalar term;
          if constexpr (std::is_same_v<Scalar, std::complex<float>> ||
                        std::is_same_v<Scalar, std::complex<double>>) {
            term = vec_a[n] * std::conj(vec_b[m]);
          } else {
            term = vec_a[n] * vec_b[m];
          }
          sum += term;
        }
      }
      output[k] = sum;
    }
    break;
  }
  case VALID: {
    // Valid correlation: only where vectors fully overlap
    int start_offset = std::abs(len_a - len_b);
    for (int k = 0; k < output_len; ++k) {
      Scalar sum = Scalar(0);

      int shorter_len = std::min(len_a, len_b);
      for (int n = 0; n < shorter_len; ++n) {
        int a_idx = (len_a >= len_b) ? n + start_offset : n;
        int b_idx = (len_b > len_a) ? n + start_offset : n;

        if (a_idx < len_a && b_idx < len_b) {
          if constexpr (std::is_same_v<Scalar, std::complex<float>> ||
                        std::is_same_v<Scalar, std::complex<double>>) {
            sum += vec_a[a_idx] * std::conj(vec_b[b_idx]);
          } else {
            sum += vec_a[a_idx] * vec_b[b_idx];
          }
        }
      }
      output[k] = sum;
    }
    break;
  }
  case SAME: {
    // Same size as larger input (central portion of full correlation)
    int max_len = std::max(len_a, len_b);
    int full_len = len_a + len_b - 1;
    int start_offset = (full_len - max_len) / 2;

    for (int k = 0; k < output_len; ++k) {
      Scalar sum = Scalar(0);
      int full_k = k + start_offset;    // Index in full correlation
      int shift = full_k - (len_b - 1); // NumPy correlation shift

      // Compute as if doing full correlation with corrected formula
      for (int i = 0; i < len_a; ++i) {
        int j = i - shift; // Corresponding index in b
        if (j >= 0 && j < len_b) {
          if constexpr (std::is_same_v<Scalar, std::complex<float>> ||
                        std::is_same_v<Scalar, std::complex<double>>) {
            sum += vec_a[i] * std::conj(vec_b[j]);
          } else {
            sum += vec_a[i] * vec_b[j];
          }
        }
      }
      output[k] = sum;
    }
    break;
  }
  }
}

template <typename Scalar>
void vector_correlate<Scalar>::normalize_correlation(
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
        &output_matrix) {
  // Extract vector from matrix - handle both row vectors (1×n) and column
  // vectors (n×1)
  Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> output(
      output_matrix.data(), output_matrix.size());
  if (output.size() == 0)
    return;

  // Find maximum absolute value for normalization
  if constexpr (std::is_same_v<Scalar, std::complex<float>> ||
                std::is_same_v<Scalar, std::complex<double>>) {
    auto max_val = std::abs(output[0]);
    for (int i = 1; i < output.size(); ++i) {
      max_val = std::max(max_val, std::abs(output[i]));
    }
    if (max_val > 0) {
      output /= max_val;
    }
  } else {
    auto max_val = std::abs(output.maxCoeff());
    auto min_val = std::abs(output.minCoeff());
    auto norm_val = std::max(max_val, min_val);
    if (norm_val > 0) {
      output /= norm_val;
    }
  }
}

// Sync block implementation
template <typename Scalar>
vector_correlate_sync<Scalar>::vector_correlate_sync(
    const types::shape &input_shape_a, const types::shape &input_shape_b,
    typename vector_correlate<Scalar>::CorrelateMode mode, bool normalize)
    : linalg_base<Scalar>( // MUST explicitly initialize virtual base!
          "vector_correlate_sync", {input_shape_a, input_shape_b},
          {"input_a", "input_b"},
          vector_correlate<Scalar>::compute_output_shapes(
              {input_shape_a, input_shape_b}, mode),
          {"output"}, array_broadcast_type::CUSTOM, error_tag_t::NONE,
          error_pdu_p::NONE),
      linalg_base_sync<Scalar>(
          "vector_correlate_sync", {input_shape_a, input_shape_b},
          {"input_a", "input_b"},
          vector_correlate<Scalar>::compute_output_shapes(
              {input_shape_a, input_shape_b}, mode),
          {"output"}, array_broadcast_type::CUSTOM, error_tag_t::NONE,
          error_pdu_p::NONE, gr::block::TPP_ALL_TO_ALL),
      vector_correlate<Scalar>(input_shape_a, input_shape_b, mode, normalize) {}

template <typename Scalar>
typename vector_correlate_sync<Scalar>::sptr
vector_correlate_sync<Scalar>::make(
    const types::shape &input_shape_a, const types::shape &input_shape_b,
    typename vector_correlate<Scalar>::CorrelateMode mode, bool normalize) {
  return std::make_shared<vector_correlate_sync<Scalar>>(
      input_shape_a, input_shape_b, mode, normalize);
}

// PDU block implementation
template <typename Scalar>
vector_correlate_pdu<Scalar>::vector_correlate_pdu(
    const types::vector_shapes &input_shapes,
    typename vector_correlate<Scalar>::CorrelateMode mode, bool normalize)
    : linalg_base_pdu<Scalar>(
          "vector_correlate_pdu", input_shapes, {"in_a", "in_b"},
          vector_correlate<Scalar>::compute_output_shapes(input_shapes, mode),
          {"out"}, array_broadcast_type::NONE, error_tag_t::NONE,
          error_pdu_p::NONE),
      vector_correlate<Scalar>(input_shapes[0],
                               input_shapes.size() > 1 ? input_shapes[1]
                                                       : types::shape{},
                               mode, normalize) {}

template <typename Scalar>
typename vector_correlate_pdu<Scalar>::sptr vector_correlate_pdu<Scalar>::make(
    const types::vector_shapes &input_shapes,
    typename vector_correlate<Scalar>::CorrelateMode mode, bool normalize) {
  return std::make_shared<vector_correlate_pdu<Scalar>>(input_shapes, mode,
                                                        normalize);
}

// Explicit template instantiations
template class vector_correlate<float>;
template class vector_correlate<double>;
template class vector_correlate<std::complex<float>>;
template class vector_correlate<std::complex<double>>;

template class vector_correlate_sync<float>;
template class vector_correlate_sync<double>;
template class vector_correlate_sync<std::complex<float>>;
template class vector_correlate_sync<std::complex<double>>;

template class vector_correlate_pdu<float>;
template class vector_correlate_pdu<double>;
template class vector_correlate_pdu<std::complex<float>>;
template class vector_correlate_pdu<std::complex<double>>;

} // namespace linalg
} // namespace gr