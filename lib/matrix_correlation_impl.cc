/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "matrix_correlation_impl.h"
#include <algorithm>
#include <gnuradio/io_signature.h>
#include <numeric>
#include <stdexcept>

namespace gr {
namespace linalg {

//==============================================================================
// Base class implementation
//==============================================================================

template <typename Scalar>
matrix_correlation<Scalar>::matrix_correlation(
    const std::string &name, const types::vector_shapes &shape_inputs,
    const types::vector_shapes &shape_outputs, CorrelationMethod method)
    : linalg_base<Scalar>(name, shape_inputs, shape_outputs), d_method(method) {
}

template <typename Scalar>
matrix_correlation<Scalar>::~matrix_correlation() = default;

template <typename Scalar>
void matrix_correlation<Scalar>::set_method(CorrelationMethod method) {
  d_method = method;
}

template <typename Scalar>
CorrelationMethod matrix_correlation<Scalar>::get_method() const {
  return d_method;
}

template <typename Scalar>
OperationReturn matrix_correlation<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {
  PROFILE_LINALG_OPERATION("matrix_correlation_operation");

  // Validate inputs
  if (input_matrices.size() != 1) {
    return OperationReturn::ERROR_INVALID_INPUT;
  }
  if (output_matrices.size() != 1) {
    return OperationReturn::ERROR_INVALID_OUTPUT;
  }

  const auto &input_matrix = *input_matrices[0];
  auto &output_matrix = *output_matrices[0];

  // Input should be [samples x features], output is [features x features]
  if (input_matrix.rows() < 2) {
    // Need at least 2 samples for meaningful correlation
    return OperationReturn::ERROR_INVALID_INPUT;
  }

  if (output_matrix.rows() != input_matrix.cols() ||
      output_matrix.cols() != input_matrix.cols()) {
    return OperationReturn::ERROR_INVALID_OUTPUT;
  }

  try {
    // Compute correlation matrix based on method
    switch (d_method) {
    case 0: // Pearson correlation
      return compute_pearson_correlation(input_matrix, output_matrix);
    case 1: // Spearman rank correlation
      return compute_spearman_correlation(input_matrix, output_matrix);
    case 2: // Kendall tau correlation
      return compute_kendall_correlation(input_matrix, output_matrix);
    default:
      return OperationReturn::ERROR_INVALID_INPUT;
    }

  } catch (const std::exception &e) {
    return OperationReturn::ERROR_RUNTIME;
  }
}

template <typename Scalar>
OperationReturn matrix_correlation<Scalar>::compute_pearson_correlation(
    const Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic,
                                         Eigen::Dynamic>> &input_matrix,
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
        &output_matrix) {
  // Create working copy and center the data
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> data = input_matrix;
  Eigen::Matrix<Scalar, 1, Eigen::Dynamic> means = data.colwise().mean();
  data.rowwise() -= means;

  // Compute covariance matrix
  Scalar divisor = static_cast<Scalar>(data.rows() - 1);
  if (std::abs(divisor) < std::numeric_limits<Scalar>::epsilon()) {
    return OperationReturn::ERROR_INVALID_INPUT;
  }

  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> cov_matrix =
      (data.transpose() * data) / divisor;

  // Compute standard deviations
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> std_devs =
      cov_matrix.diagonal().array().sqrt();

  // Compute correlation matrix: R[i,j] = cov[i,j] / (std[i] * std[j])
  for (int i = 0; i < output_matrix.rows(); ++i) {
    for (int j = 0; j < output_matrix.cols(); ++j) {
      Scalar std_product = std_devs(i) * std_devs(j);
      if (std::abs(std_product) < std::numeric_limits<Scalar>::epsilon()) {
        output_matrix(i, j) = (i == j) ? Scalar(1) : Scalar(0);
      } else {
        output_matrix(i, j) = cov_matrix(i, j) / std_product;
      }
    }
  }

  return OperationReturn::SUCCESS;
}

template <typename Scalar>
OperationReturn matrix_correlation<Scalar>::compute_spearman_correlation(
    const Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic,
                                         Eigen::Dynamic>> &input_matrix,
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
        &output_matrix) {
  // Convert to ranks and then compute Pearson correlation on ranks
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> ranks(
      input_matrix.rows(), input_matrix.cols());

  // Rank each column
  for (int col = 0; col < input_matrix.cols(); ++col) {
    std::vector<std::pair<Scalar, int>> values;
    values.reserve(input_matrix.rows());

    // Create pairs of (value, original_index)
    for (int row = 0; row < input_matrix.rows(); ++row) {
      values.emplace_back(input_matrix(row, col), row);
    }

    // Sort by value
    std::sort(values.begin(), values.end());

    // Assign ranks (handle ties by averaging)
    for (size_t i = 0; i < values.size(); ++i) {
      ranks(values[i].second, col) = static_cast<Scalar>(i + 1);
    }
  }

  // Now compute Pearson correlation on the rank matrix
  return compute_pearson_correlation_from_data(ranks, output_matrix);
}

template <typename Scalar>
OperationReturn matrix_correlation<Scalar>::compute_kendall_correlation(
    const Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic,
                                         Eigen::Dynamic>> &input_matrix,
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
        &output_matrix) {
  // Kendall tau correlation is computationally expensive - simplified
  // implementation
  int n_samples = input_matrix.rows();
  int n_features = input_matrix.cols();

  for (int i = 0; i < n_features; ++i) {
    for (int j = 0; j < n_features; ++j) {
      if (i == j) {
        output_matrix(i, j) = Scalar(1);
        continue;
      }

      int concordant = 0;
      int discordant = 0;

      // Count concordant and discordant pairs
      for (int p = 0; p < n_samples - 1; ++p) {
        for (int q = p + 1; q < n_samples; ++q) {
          Scalar diff_i = input_matrix(p, i) - input_matrix(q, i);
          Scalar diff_j = input_matrix(p, j) - input_matrix(q, j);

          if (diff_i * diff_j > 0) {
            concordant++;
          } else if (diff_i * diff_j < 0) {
            discordant++;
          }
          // Ties are ignored in basic Kendall tau
        }
      }

      int total_pairs = (n_samples * (n_samples - 1)) / 2;
      if (total_pairs > 0) {
        output_matrix(i, j) = static_cast<Scalar>(concordant - discordant) /
                              static_cast<Scalar>(total_pairs);
      } else {
        output_matrix(i, j) = Scalar(0);
      }
    }
  }

  return OperationReturn::SUCCESS;
}

template <typename Scalar>
OperationReturn
matrix_correlation<Scalar>::compute_pearson_correlation_from_data(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &data,
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
        &output_matrix) {
  // Center the data
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> centered_data = data;
  Eigen::Matrix<Scalar, 1, Eigen::Dynamic> means =
      centered_data.colwise().mean();
  centered_data.rowwise() -= means;

  // Compute covariance matrix
  Scalar divisor = static_cast<Scalar>(centered_data.rows() - 1);
  if (std::abs(divisor) < std::numeric_limits<Scalar>::epsilon()) {
    return OperationReturn::ERROR_INVALID_INPUT;
  }

  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> cov_matrix =
      (centered_data.transpose() * centered_data) / divisor;

  // Compute standard deviations
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> std_devs =
      cov_matrix.diagonal().array().sqrt();

  // Compute correlation matrix
  for (int i = 0; i < output_matrix.rows(); ++i) {
    for (int j = 0; j < output_matrix.cols(); ++j) {
      Scalar std_product = std_devs(i) * std_devs(j);
      if (std::abs(std_product) < std::numeric_limits<Scalar>::epsilon()) {
        output_matrix(i, j) = (i == j) ? Scalar(1) : Scalar(0);
      } else {
        output_matrix(i, j) = cov_matrix(i, j) / std_product;
      }
    }
  }

  return OperationReturn::SUCCESS;
}

template <typename Scalar>
types::vector_shapes matrix_correlation<Scalar>::compute_output_shapes(
    const types::vector_shapes &input_shapes) {
  if (input_shapes.size() != 1) {
    throw std::invalid_argument("matrix_correlation requires exactly 1 input");
  }

  const auto &input_shape = input_shapes[0];
  if (input_shape.size() != 2) {
    throw std::invalid_argument("matrix_correlation requires 2D input matrix");
  }

  // Output is square matrix with dimensions [features x features]
  int features = input_shape[1];
  return {{features, features}};
}

template <typename Scalar>
void matrix_correlation<Scalar>::validate_shape(
    const types::vector_shapes &input_shapes,
    const types::vector_shapes &output_shapes, const std::string &name) {
  if (input_shapes.size() != 1) {
    throw std::invalid_argument(name + ": requires exactly 1 input");
  }

  const auto &input_shape = input_shapes[0];
  if (input_shape.size() != 2) {
    throw std::invalid_argument(
        name + ": input must be 2D matrix [samples x features]");
  }

  if (input_shape[0] < 2) {
    throw std::invalid_argument(name +
                                ": need at least 2 samples for correlation");
  }

  if (input_shape[1] <= 0) {
    throw std::invalid_argument(name + ": number of features must be positive");
  }

  if (!output_shapes.empty()) {
    if (output_shapes.size() != 1) {
      throw std::invalid_argument(name + ": requires exactly 1 output");
    }

    const auto &output_shape = output_shapes[0];
    if (output_shape.size() != 2) {
      throw std::invalid_argument(name + ": output must be 2D matrix");
    }

    if (output_shape[0] != input_shape[1] ||
        output_shape[1] != input_shape[1]) {
      throw std::invalid_argument(
          name + ": output shape must be [features x features]");
    }
  }
}

template <typename Scalar>
std::vector<size_t>
matrix_correlation<Scalar>::compute_sizes(const types::vector_shapes &shapes) {
  std::vector<size_t> sizes;
  sizes.reserve(shapes.size());

  for (const auto &shape : shapes) {
    size_t size = sizeof(Scalar);
    for (const auto &dim : shape) {
      size *= static_cast<size_t>(dim);
    }
    sizes.push_back(size);
  }

  return sizes;
}

//==============================================================================
// Sync implementation
//==============================================================================

template <typename Scalar>
matrix_correlation_sync_impl<Scalar>::matrix_correlation_sync_impl(
    const types::shape &shape, CorrelationMethod method)
    : matrix_correlation_sync<Scalar>(shape, method) {
  initialize_base_classes(shape, method);
}

template <typename Scalar>
void matrix_correlation_sync_impl<Scalar>::initialize_base_classes(
    const types::shape &shape, CorrelationMethod method) {
  // Compute output shape
  types::vector_shapes input_shapes = {shape};
  types::vector_shapes output_shapes =
      matrix_correlation<Scalar>::compute_output_shapes(input_shapes);

  // Initialize base classes explicitly
  linalg_base<Scalar>::initialize("matrix_correlation_sync", input_shapes,
                                  output_shapes);
  matrix_correlation<Scalar>::initialize("matrix_correlation_sync",
                                         input_shapes, output_shapes, method);
  linalg_base_sync<Scalar>::initialize("matrix_correlation_sync", input_shapes,
                                       output_shapes);
}

template <typename Scalar>
matrix_correlation_sync<Scalar>::matrix_correlation_sync(
    const types::shape &shape, CorrelationMethod method)
    : linalg_base<Scalar>(
          "matrix_correlation_sync", {shape},
          matrix_correlation<Scalar>::compute_output_shapes({shape})),
      matrix_correlation<Scalar>(
          "matrix_correlation_sync", {shape},
          matrix_correlation<Scalar>::compute_output_shapes({shape}), method),
      linalg_base_sync<Scalar>(
          "matrix_correlation_sync", {shape},
          matrix_correlation<Scalar>::compute_output_shapes({shape})) {}

template <typename Scalar>
typename matrix_correlation_sync<Scalar>::sptr
matrix_correlation_sync<Scalar>::make(const types::shape &shape,
                                      CorrelationMethod method) {
  return std::make_shared<matrix_correlation_sync_impl<Scalar>>(shape, method);
}

//==============================================================================
// PDU implementation
//==============================================================================

template <typename Scalar>
matrix_correlation_pdu_impl<Scalar>::matrix_correlation_pdu_impl(
    const types::shape &shape, CorrelationMethod method)
    : matrix_correlation_pdu<Scalar>(shape, method) {
  initialize_base_classes(shape, method);
}

template <typename Scalar>
void matrix_correlation_pdu_impl<Scalar>::initialize_base_classes(
    const types::shape &shape, CorrelationMethod method) {
  // Compute output shape
  types::vector_shapes input_shapes = {shape};
  types::vector_shapes output_shapes =
      matrix_correlation<Scalar>::compute_output_shapes(input_shapes);

  // Initialize base classes explicitly
  linalg_base<Scalar>::initialize("matrix_correlation_pdu", input_shapes,
                                  output_shapes);
  matrix_correlation<Scalar>::initialize("matrix_correlation_pdu", input_shapes,
                                         output_shapes, method);
  linalg_base_pdu<Scalar>::initialize("matrix_correlation_pdu", input_shapes,
                                      {"in"}, output_shapes, {"out"});
}

template <typename Scalar>
matrix_correlation_pdu<Scalar>::matrix_correlation_pdu(
    const types::shape &shape, CorrelationMethod method)
    : linalg_base<Scalar>(
          "matrix_correlation_pdu", {shape},
          matrix_correlation<Scalar>::compute_output_shapes({shape})),
      matrix_correlation<Scalar>(
          "matrix_correlation_pdu", {shape},
          matrix_correlation<Scalar>::compute_output_shapes({shape}), method),
      linalg_base_pdu<Scalar>(
          "matrix_correlation_pdu", {shape}, {"in"},
          matrix_correlation<Scalar>::compute_output_shapes({shape}), {"out"},
          array_broadcast_type::CUSTOM, error_tag_t::NONE, error_pdu_p::NONE) {}

template <typename Scalar>
typename matrix_correlation_pdu<Scalar>::sptr
matrix_correlation_pdu<Scalar>::make(const types::shape &shape,
                                     CorrelationMethod method) {
  return std::make_shared<matrix_correlation_pdu_impl<Scalar>>(shape, method);
}

//==============================================================================
// Explicit template instantiations
//==============================================================================

template class matrix_correlation<float>;
template class matrix_correlation<double>;
template class matrix_correlation<gr_complex>;
template class matrix_correlation<std::complex<double>>;

template class matrix_correlation_sync<float>;
template class matrix_correlation_sync<double>;
template class matrix_correlation_sync<gr_complex>;
template class matrix_correlation_sync<std::complex<double>>;

template class matrix_correlation_pdu<float>;
template class matrix_correlation_pdu<double>;
template class matrix_correlation_pdu<gr_complex>;
template class matrix_correlation_pdu<std::complex<double>>;

template class matrix_correlation_sync_impl<float>;
template class matrix_correlation_sync_impl<double>;
template class matrix_correlation_sync_impl<gr_complex>;
template class matrix_correlation_sync_impl<std::complex<double>>;

template class matrix_correlation_pdu_impl<float>;
template class matrix_correlation_pdu_impl<double>;
template class matrix_correlation_pdu_impl<gr_complex>;
template class matrix_correlation_pdu_impl<std::complex<double>>;

} /* namespace linalg */
} /* namespace gr */
