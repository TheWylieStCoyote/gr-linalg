/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "matrix_source_const_impl.h"
#include "gnuradio/linalg/matrix_source_const.h"
#include "gnuradio/linalg/utils.h"
#include <gnuradio/io_signature.h>
#include <stdexcept>

namespace gr {
namespace linalg {

// Base class constructor
template <typename Scalar>
matrix_source_const<Scalar>::matrix_source_const(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &matrix,
    const types::shape &shape)
    : linalg_base<Scalar>("matrix_source_const", {}, std::vector<std::string>(),
                          {shape}, std::vector<std::string>(),
                          array_broadcast_type::CUSTOM, error_tag_t::NONE,
                          error_pdu_p::NONE),
      matrix_(matrix.data(), matrix.data() + matrix.size()), shape_(shape) {
  // Validate the shape
  if (shape.size() != 2) {
    throw std::invalid_argument("Matrix shape must be 2D");
  }

  // Verify the matrix dimensions match the shape
  if (matrix.rows() != shape[0] || matrix.cols() != shape[1]) {
    throw std::invalid_argument(
        "Matrix dimensions do not match specified shape");
  }
}

// Base class static methods
template <typename Scalar>
void matrix_source_const<Scalar>::validate_shape(const types::shape &shape,
                                                 const std::string &name) {
  if (shape.size() != 2) {
    throw std::invalid_argument(name + ": shape must be 2D");
  }
  if (shape[0] <= 0 || shape[1] <= 0) {
    throw std::invalid_argument(name + ": shape dimensions must be positive");
  }
}

template <typename Scalar>
types::vector_shapes
matrix_source_const<Scalar>::compute_output_shapes(const types::shape &shape) {
  return types::vector_shapes{shape};
}

// Base class operation
template <typename Scalar>
OperationReturn matrix_source_const<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {
  if (output_matrices.empty()) {
    return OperationReturn::INVALID_SHAPE;
  }

  // Copy the stored matrix data to the output
  // Note: matrix_ is stored in column-major order (Eigen default)
  auto &output = *output_matrices[0];
  for (int i = 0; i < shape_[0]; ++i) {
    for (int j = 0; j < shape_[1]; ++j) {
      output(i, j) = matrix_[j * shape_[0] + i]; // Column-major access
    }
  }

  return OperationReturn::SUCCESS;
}

// Interface class constructors
template <typename Scalar>
matrix_source_const_sync<Scalar>::matrix_source_const_sync(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &matrix,
    const types::shape &shape)
    : linalg_base_sync<Scalar>(
          "matrix_source_const_sync", {}, std::vector<std::string>(), {shape},
          std::vector<std::string>(), array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE),
      matrix_source_const<Scalar>(matrix, shape) {}

template <typename Scalar>
matrix_source_const_pdu<Scalar>::matrix_source_const_pdu(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &matrix,
    const types::shape &shape)
    : linalg_base_pdu<Scalar>("matrix_source_const_pdu", {}, // No inputs
                              {},                            // No input names
                              {shape}, // Single output shape
                              {"out"}, // Single output name
                              array_broadcast_type::CUSTOM, error_tag_t::NONE,
                              error_pdu_p::NONE, PDU_UPDATE::DEFAULT,
                              MESSAGE_HANDLER_MODE::DEFAULT),
      matrix_source_const<Scalar>(matrix, shape) {}

// Sync implementation constructor
template <typename Scalar>
matrix_source_const_sync_impl<Scalar>::matrix_source_const_sync_impl(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &matrix,
    const types::shape &shape)
    : linalg_base<Scalar>("matrix_source_const", {}, std::vector<std::string>(),
                          {shape}, std::vector<std::string>(),
                          array_broadcast_type::CUSTOM, error_tag_t::NONE,
                          error_pdu_p::NONE),
      linalg_base_sync<Scalar>(
          "matrix_source_const", {}, std::vector<std::string>(), {shape},
          std::vector<std::string>(), array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE),
      matrix_source_const<Scalar>(matrix, shape),
      matrix_source_const_sync<Scalar>(matrix, shape) {}

// Sync operation implementation
template <typename Scalar>
OperationReturn matrix_source_const_sync_impl<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {
  // Matrix source has no inputs, just copy the stored matrix to output
  auto &output = *output_matrices[0];

  // Copy from the flat vector to the output matrix
  // Note: matrix_ is stored in column-major order (Eigen default)
  int rows = this->shape_[0];
  int cols = this->shape_[1];
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      output(i, j) = this->matrix_[j * rows + i]; // Column-major access
    }
  }
  return OperationReturn::SUCCESS;
}

// PDU implementation constructor
template <typename Scalar>
matrix_source_const_pdu_impl<Scalar>::matrix_source_const_pdu_impl(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &matrix,
    const types::shape &shape)
    : linalg_base<Scalar>("matrix_source_const", {}, {shape}),
      linalg_base_pdu<Scalar>("matrix_source_const", {}, // No inputs
                              {},                        // No input names
                              {shape},                   // Single output shape
                              {"out"},                   // Single output name
                              array_broadcast_type::CUSTOM, error_tag_t::NONE,
                              error_pdu_p::NONE, PDU_UPDATE::DEFAULT,
                              MESSAGE_HANDLER_MODE::DEFAULT),
      matrix_source_const<Scalar>(matrix, shape),
      matrix_source_const_pdu<Scalar>(matrix, shape) {}

// PDU operation implementation
template <typename Scalar>
OperationReturn matrix_source_const_pdu_impl<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {
  // Matrix source has no inputs, just copy the stored matrix to output
  auto &output = *output_matrices[0];

  // Copy from the flat vector to the output matrix
  // Note: matrix_ is stored in column-major order (Eigen default)
  int rows = utils::matrix_rows(this->shape_);
  int cols = utils::matrix_cols(this->shape_);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      output(i, j) = this->matrix_[j * rows + i]; // Column-major access
    }
  }
  return OperationReturn::SUCCESS;
}

// Factory methods
template <typename Scalar>
typename matrix_source_const_sync<Scalar>::sptr
matrix_source_const_sync<Scalar>::make(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &matrix,
    const types::shape &shape) {
  return gnuradio::get_initial_sptr(
      new matrix_source_const_sync_impl<Scalar>(matrix, shape));
}

template <typename Scalar>
typename matrix_source_const_sync<Scalar>::sptr
matrix_source_const_sync<Scalar>::make(
    const std::vector<std::vector<Scalar>> &matrix, const types::shape &shape) {
  // Convert vector of vectors to Eigen matrix
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix(shape[0],
                                                                     shape[1]);
  for (int i = 0; i < shape[0]; ++i) {
    for (int j = 0; j < shape[1]; ++j) {
      eigen_matrix(i, j) = matrix[i][j];
    }
  }
  return make(eigen_matrix, shape);
}

template <typename Scalar>
typename matrix_source_const_sync<Scalar>::sptr
matrix_source_const_sync<Scalar>::make(const std::vector<Scalar> &matrix,
                                       const types::shape &shape) {
  // Convert flat vector to Eigen matrix
  // Assume flat vector is in row-major order, but Eigen stores column-major
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix(shape[0],
                                                                     shape[1]);
  for (int i = 0; i < shape[0]; ++i) {
    for (int j = 0; j < shape[1]; ++j) {
      eigen_matrix(i, j) =
          matrix[i * shape[1] + j]; // Row-major input to column-major Eigen
    }
  }
  return make(eigen_matrix, shape);
}

template <typename Scalar>
typename matrix_source_const_pdu<Scalar>::sptr
matrix_source_const_pdu<Scalar>::make(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &matrix,
    const types::shape &shape) {
  return gnuradio::get_initial_sptr(
      new matrix_source_const_pdu_impl<Scalar>(matrix, shape));
}

template <typename Scalar>
typename matrix_source_const_pdu<Scalar>::sptr
matrix_source_const_pdu<Scalar>::make(
    const std::vector<std::vector<Scalar>> &matrix, const types::shape &shape) {
  // Convert vector of vectors to Eigen matrix
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix(shape[0],
                                                                     shape[1]);
  for (int i = 0; i < shape[0]; ++i) {
    for (int j = 0; j < shape[1]; ++j) {
      eigen_matrix(i, j) = matrix[i][j];
    }
  }
  return make(eigen_matrix, shape);
}

template <typename Scalar>
typename matrix_source_const_pdu<Scalar>::sptr
matrix_source_const_pdu<Scalar>::make(const std::vector<Scalar> &matrix,
                                      const types::shape &shape) {
  // Convert flat vector to Eigen matrix
  // Assume flat vector is in row-major order, but Eigen stores column-major
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix(shape[0],
                                                                     shape[1]);
  for (int i = 0; i < shape[0]; ++i) {
    for (int j = 0; j < shape[1]; ++j) {
      eigen_matrix(i, j) =
          matrix[i * shape[1] + j]; // Row-major input to column-major Eigen
    }
  }
  return make(eigen_matrix, shape);
}

// Explicit template instantiations
template class matrix_source_const<float>;
template class matrix_source_const<double>;
template class matrix_source_const<std::complex<float>>;
template class matrix_source_const<std::complex<double>>;

template class matrix_source_const_sync_impl<float>;
template class matrix_source_const_sync_impl<double>;
template class matrix_source_const_sync_impl<std::complex<float>>;
template class matrix_source_const_sync_impl<std::complex<double>>;

template class matrix_source_const_pdu_impl<float>;
template class matrix_source_const_pdu_impl<double>;
template class matrix_source_const_pdu_impl<std::complex<float>>;
template class matrix_source_const_pdu_impl<std::complex<double>>;

// Explicit instantiation of factory methods
template typename matrix_source_const_sync<float>::sptr
matrix_source_const_sync<float>::make(
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &,
    const types::shape &);
template typename matrix_source_const_sync<double>::sptr
matrix_source_const_sync<double>::make(
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &,
    const types::shape &);
template typename matrix_source_const_sync<std::complex<float>>::sptr
matrix_source_const_sync<std::complex<float>>::make(
    const Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic> &,
    const types::shape &);
template typename matrix_source_const_sync<std::complex<double>>::sptr
matrix_source_const_sync<std::complex<double>>::make(
    const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> &,
    const types::shape &);

template typename matrix_source_const_pdu<float>::sptr
matrix_source_const_pdu<float>::make(
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &,
    const types::shape &);
template typename matrix_source_const_pdu<double>::sptr
matrix_source_const_pdu<double>::make(
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &,
    const types::shape &);
template typename matrix_source_const_pdu<std::complex<float>>::sptr
matrix_source_const_pdu<std::complex<float>>::make(
    const Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic> &,
    const types::shape &);
template typename matrix_source_const_pdu<std::complex<double>>::sptr
matrix_source_const_pdu<std::complex<double>>::make(
    const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> &,
    const types::shape &);

// Instantiate vector<vector<Scalar>> overloads for sync
template typename matrix_source_const_sync<float>::sptr
matrix_source_const_sync<float>::make(const std::vector<std::vector<float>> &,
                                      const types::shape &);
template typename matrix_source_const_sync<double>::sptr
matrix_source_const_sync<double>::make(const std::vector<std::vector<double>> &,
                                       const types::shape &);
template typename matrix_source_const_sync<std::complex<float>>::sptr
matrix_source_const_sync<std::complex<float>>::make(
    const std::vector<std::vector<std::complex<float>>> &,
    const types::shape &);
template typename matrix_source_const_sync<std::complex<double>>::sptr
matrix_source_const_sync<std::complex<double>>::make(
    const std::vector<std::vector<std::complex<double>>> &,
    const types::shape &);

// Instantiate vector<Scalar> overloads for sync
template typename matrix_source_const_sync<float>::sptr
matrix_source_const_sync<float>::make(const std::vector<float> &,
                                      const types::shape &);
template typename matrix_source_const_sync<double>::sptr
matrix_source_const_sync<double>::make(const std::vector<double> &,
                                       const types::shape &);
template typename matrix_source_const_sync<std::complex<float>>::sptr
matrix_source_const_sync<std::complex<float>>::make(
    const std::vector<std::complex<float>> &, const types::shape &);
template typename matrix_source_const_sync<std::complex<double>>::sptr
matrix_source_const_sync<std::complex<double>>::make(
    const std::vector<std::complex<double>> &, const types::shape &);

// Instantiate vector<vector<Scalar>> overloads for PDU
template typename matrix_source_const_pdu<float>::sptr
matrix_source_const_pdu<float>::make(const std::vector<std::vector<float>> &,
                                     const types::shape &);
template typename matrix_source_const_pdu<double>::sptr
matrix_source_const_pdu<double>::make(const std::vector<std::vector<double>> &,
                                      const types::shape &);
template typename matrix_source_const_pdu<std::complex<float>>::sptr
matrix_source_const_pdu<std::complex<float>>::make(
    const std::vector<std::vector<std::complex<float>>> &,
    const types::shape &);
template typename matrix_source_const_pdu<std::complex<double>>::sptr
matrix_source_const_pdu<std::complex<double>>::make(
    const std::vector<std::vector<std::complex<double>>> &,
    const types::shape &);

// Instantiate vector<Scalar> overloads for PDU
template typename matrix_source_const_pdu<float>::sptr
matrix_source_const_pdu<float>::make(const std::vector<float> &,
                                     const types::shape &);
template typename matrix_source_const_pdu<double>::sptr
matrix_source_const_pdu<double>::make(const std::vector<double> &,
                                      const types::shape &);
template typename matrix_source_const_pdu<std::complex<float>>::sptr
matrix_source_const_pdu<std::complex<float>>::make(
    const std::vector<std::complex<float>> &, const types::shape &);
template typename matrix_source_const_pdu<std::complex<double>>::sptr
matrix_source_const_pdu<std::complex<double>>::make(
    const std::vector<std::complex<double>> &, const types::shape &);

} /* namespace linalg */
} /* namespace gr */
