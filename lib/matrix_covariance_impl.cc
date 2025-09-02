/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "matrix_covariance_impl.h"
#include <gnuradio/io_signature.h>
#include <stdexcept>

namespace gr {
namespace linalg {

//==============================================================================
// Base class implementation
//==============================================================================

template <typename Scalar>
matrix_covariance<Scalar>::matrix_covariance(
    const std::string &name, const types::vector_shapes &shape_inputs,
    const types::vector_shapes &shape_outputs, bool unbiased, bool center)
    : linalg_base<Scalar>(name, shape_inputs, shape_outputs),
      d_unbiased(unbiased), d_center(center) {}

template <typename Scalar>
matrix_covariance<Scalar>::~matrix_covariance() = default;

template <typename Scalar>
void matrix_covariance<Scalar>::set_unbiased(bool unbiased) {
  d_unbiased = unbiased;
}

template <typename Scalar>
bool matrix_covariance<Scalar>::get_unbiased() const {
  return d_unbiased;
}

template <typename Scalar>
void matrix_covariance<Scalar>::set_center(bool center) {
  d_center = center;
}

template <typename Scalar>
bool matrix_covariance<Scalar>::get_center() const {
  return d_center;
}

template <typename Scalar>
OperationReturn matrix_covariance<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {
  PROFILE_LINALG_OPERATION("matrix_covariance_operation",
                           input_matrices[0]->rows() *
                               input_matrices[0]->cols());

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
    // Need at least 2 samples for meaningful covariance
    return OperationReturn::ERROR_INVALID_INPUT;
  }

  if (output_matrix.rows() != input_matrix.cols() ||
      output_matrix.cols() != input_matrix.cols()) {
    return OperationReturn::ERROR_INVALID_OUTPUT;
  }

  try {
    // Create a working copy of the input data
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> data = input_matrix;

    // Center the data if requested (subtract column means)
    if (d_center) {
      Eigen::Matrix<Scalar, 1, Eigen::Dynamic> means = data.colwise().mean();
      data.rowwise() -= means;
    }

    // Compute covariance matrix: C = (1/N) * X^T * X or (1/(N-1)) * X^T * X
    Scalar divisor;
    if (d_unbiased) {
      divisor = static_cast<Scalar>(data.rows() - 1);
    } else {
      divisor = static_cast<Scalar>(data.rows());
    }

    if (std::abs(divisor) < std::numeric_limits<Scalar>::epsilon()) {
      return OperationReturn::ERROR_INVALID_INPUT;
    }

    output_matrix = (data.transpose() * data) / divisor;

    return OperationReturn::SUCCESS;

  } catch (const std::exception &e) {
    return OperationReturn::ERROR_RUNTIME;
  }
}

template <typename Scalar>
types::vector_shapes matrix_covariance<Scalar>::compute_output_shapes(
    const types::vector_shapes &input_shapes) {
  if (input_shapes.size() != 1) {
    throw std::invalid_argument("matrix_covariance requires exactly 1 input");
  }

  const auto &input_shape = input_shapes[0];
  if (input_shape.size() != 2) {
    throw std::invalid_argument("matrix_covariance requires 2D input matrix");
  }

  // Output is square matrix with dimensions [features x features]
  int features = input_shape[1];
  return {{features, features}};
}

template <typename Scalar>
void matrix_covariance<Scalar>::validate_shape(
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
                                ": need at least 2 samples for covariance");
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
matrix_covariance<Scalar>::compute_sizes(const types::vector_shapes &shapes) {
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
matrix_covariance_sync_impl<Scalar>::matrix_covariance_sync_impl(
    const types::shape &shape, bool unbiased, bool center)
    : matrix_covariance_sync<Scalar>(shape, unbiased, center) {
  initialize_base_classes(shape, unbiased, center);
}

template <typename Scalar>
void matrix_covariance_sync_impl<Scalar>::initialize_base_classes(
    const types::shape &shape, bool unbiased, bool center) {
  // Compute output shape
  types::vector_shapes input_shapes = {shape};
  types::vector_shapes output_shapes =
      matrix_covariance<Scalar>::compute_output_shapes(input_shapes);

  // Initialize base classes explicitly
  linalg_base<Scalar>::initialize("matrix_covariance_sync", input_shapes,
                                  output_shapes);
  matrix_covariance<Scalar>::initialize("matrix_covariance_sync", input_shapes,
                                        output_shapes, unbiased, center);
  linalg_base_sync<Scalar>::initialize("matrix_covariance_sync", input_shapes,
                                       output_shapes);
}

template <typename Scalar>
matrix_covariance_sync<Scalar>::matrix_covariance_sync(
    const types::shape &shape, bool unbiased, bool center)
    : linalg_base<Scalar>(
          "matrix_covariance_sync", {shape},
          matrix_covariance<Scalar>::compute_output_shapes({shape})),
      matrix_covariance<Scalar>(
          "matrix_covariance_sync", {shape},
          matrix_covariance<Scalar>::compute_output_shapes({shape}), unbiased,
          center),
      linalg_base_sync<Scalar>(
          "matrix_covariance_sync", {shape},
          matrix_covariance<Scalar>::compute_output_shapes({shape})) {}

template <typename Scalar>
typename matrix_covariance_sync<Scalar>::sptr
matrix_covariance_sync<Scalar>::make(const types::shape &shape, bool unbiased,
                                     bool center) {
  return std::make_shared<matrix_covariance_sync_impl<Scalar>>(shape, unbiased,
                                                               center);
}

//==============================================================================
// PDU implementation
//==============================================================================

template <typename Scalar>
matrix_covariance_pdu_impl<Scalar>::matrix_covariance_pdu_impl(
    const types::shape &shape, bool unbiased, bool center)
    : matrix_covariance_pdu<Scalar>(shape, unbiased, center) {
  initialize_base_classes(shape, unbiased, center);
}

template <typename Scalar>
void matrix_covariance_pdu_impl<Scalar>::initialize_base_classes(
    const types::shape &shape, bool unbiased, bool center) {
  // Compute output shape
  types::vector_shapes input_shapes = {shape};
  types::vector_shapes output_shapes =
      matrix_covariance<Scalar>::compute_output_shapes(input_shapes);

  // Initialize base classes explicitly
  linalg_base<Scalar>::initialize("matrix_covariance_pdu", input_shapes,
                                  output_shapes);
  matrix_covariance<Scalar>::initialize("matrix_covariance_pdu", input_shapes,
                                        output_shapes, unbiased, center);
  linalg_base_pdu<Scalar>::initialize("matrix_covariance_pdu", input_shapes,
                                      {"in"}, output_shapes, {"out"});
}

template <typename Scalar>
matrix_covariance_pdu<Scalar>::matrix_covariance_pdu(const types::shape &shape,
                                                     bool unbiased, bool center)
    : linalg_base<Scalar>(
          "matrix_covariance_pdu", {shape},
          matrix_covariance<Scalar>::compute_output_shapes({shape})),
      matrix_covariance<Scalar>(
          "matrix_covariance_pdu", {shape},
          matrix_covariance<Scalar>::compute_output_shapes({shape}), unbiased,
          center),
      linalg_base_pdu<Scalar>(
          "matrix_covariance_pdu", {shape}, {"in"},
          matrix_covariance<Scalar>::compute_output_shapes({shape}), {"out"},
          array_broadcast_type::CUSTOM, error_tag_t::NONE, error_pdu_p::NONE) {}

template <typename Scalar>
typename matrix_covariance_pdu<Scalar>::sptr
matrix_covariance_pdu<Scalar>::make(const types::shape &shape, bool unbiased,
                                    bool center) {
  return std::make_shared<matrix_covariance_pdu_impl<Scalar>>(shape, unbiased,
                                                              center);
}

//==============================================================================
// Explicit template instantiations
//==============================================================================

template class matrix_covariance<float>;
template class matrix_covariance<double>;
template class matrix_covariance<gr_complex>;
template class matrix_covariance<std::complex<double>>;

template class matrix_covariance_sync<float>;
template class matrix_covariance_sync<double>;
template class matrix_covariance_sync<gr_complex>;
template class matrix_covariance_sync<std::complex<double>>;

template class matrix_covariance_pdu<float>;
template class matrix_covariance_pdu<double>;
template class matrix_covariance_pdu<gr_complex>;
template class matrix_covariance_pdu<std::complex<double>>;

template class matrix_covariance_sync_impl<float>;
template class matrix_covariance_sync_impl<double>;
template class matrix_covariance_sync_impl<gr_complex>;
template class matrix_covariance_sync_impl<std::complex<double>>;

template class matrix_covariance_pdu_impl<float>;
template class matrix_covariance_pdu_impl<double>;
template class matrix_covariance_pdu_impl<gr_complex>;
template class matrix_covariance_pdu_impl<std::complex<double>>;

} /* namespace linalg */
} /* namespace gr */
