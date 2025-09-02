/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "decomp_qr_impl.h"
#include "gnuradio/linalg/utils.h"
#include <atomic>
#include <complex>
#include <gnuradio/io_signature.h>
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/types.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace gr {
namespace linalg {

// Unique name generator for QR instances
namespace {
static std::atomic<uint64_t> g_qr_sync_uid{0};
inline std::string make_unique_qr_name(const char *base) {
  uint64_t id = g_qr_sync_uid.fetch_add(1, std::memory_order_relaxed);
  return std::string(base) + "#" + std::to_string(id);
}
} // namespace

// Helper function to create single shape vector
static inline types::vector_shapes _single_shape(const types::shape &s) {
  types::vector_shapes v;
  v.push_back(s);
  return v;
}

// Helper function to compute the output shapes for QR decomposition
// For sync blocks, we output a single matrix containing Q and R concatenated
static inline types::vector_shapes
_qr_output_shapes(const types::shape &shape) {
  // Output is a single matrix with 2 times the width (Q, R side by side)
  if (shape.size() != 2) {
    throw std::invalid_argument("QR decomposition requires 2D matrix");
  }

  int rows = shape[0];
  int cols = shape[1];

  types::vector_shapes shapes;
  shapes.push_back({rows, cols * 2}); // Q, R concatenated horizontally
  return shapes;
}

// Algorithm implementations
template <typename Scalar, int Rows, int Cols>
void eigen_householder_qr<Scalar, Rows, Cols>::decompose(
    const Eigen::Matrix<Scalar, Rows, Cols> &input,
    Eigen::Matrix<Scalar, Rows, Cols> &Q,
    Eigen::Matrix<Scalar, Rows, Cols> &R) {

  // Use Eigen's HouseholderQR decomposition
  Eigen::HouseholderQR<Eigen::Matrix<Scalar, Rows, Cols>> qr_solver(input);

  // Extract Q and R matrices
  Q = qr_solver.householderQ();
  R = qr_solver.matrixQR().template triangularView<Eigen::Upper>();
}

template <typename Scalar, int Rows, int Cols>
void eigen_colpivhouseholder_qr<Scalar, Rows, Cols>::decompose(
    const Eigen::Matrix<Scalar, Rows, Cols> &input,
    Eigen::Matrix<Scalar, Rows, Cols> &Q,
    Eigen::Matrix<Scalar, Rows, Cols> &R) {

  // Use Eigen's ColPivHouseholderQR decomposition
  Eigen::ColPivHouseholderQR<Eigen::Matrix<Scalar, Rows, Cols>> qr_solver(
      input);

  // Extract Q and R matrices
  Q = qr_solver.householderQ();
  R = qr_solver.matrixR();
}

template <typename Scalar>
decomp_qr<Scalar>::decomp_qr(
    const types::shape &shape,
    const std::shared_ptr<qr_algorithm<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
        &algo,
    const std::string &name)
    : linalg_base<Scalar>(
          name, _single_shape(shape), _qr_output_shapes(shape),
          array_broadcast_type::CUSTOM) /* use convenience ctor */,
      algorithm_(algo) {
  // Validate matrix dimensions
  if (shape.size() < 2) {
    throw std::invalid_argument("QR decomposition requires 2D matrix input");
  }
}

// Base class operation implementation
template <typename Scalar>
OperationReturn decomp_qr<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {

  if (input_matrices.empty() || output_matrices.empty()) {
    return OperationReturn::FAILURE; // No input or output matrices
  }

  if (input_matrices.size() != 1) {
    throw std::runtime_error("QR decomposition expects exactly 1 input matrix");
  }

  if (output_matrices.size() != 1) {
    throw std::runtime_error("QR decomposition sync block expects exactly 1 "
                             "output matrix (concatenated Q|R)");
  }

  try {
    // Get input matrix
    const auto &input_matrix = *input_matrices[0];
    auto &output_matrix = *output_matrices[0];

    int rows = input_matrix.rows();
    int cols = input_matrix.cols();

    // Verify output matrix has correct dimensions (should be rows x (cols*2))
    if (output_matrix.rows() != rows || output_matrix.cols() != cols * 2) {
      throw std::runtime_error(
          "Output matrix dimensions incorrect for QR decomposition");
    }

    // Create temporary matrices for computation
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> temp_input =
        input_matrix;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Q(rows, rows),
        R(rows, cols);

    // Use default algorithm if none set
    if (!algorithm_) {
      algorithm_ = std::make_shared<
          eigen_householder_qr<Scalar, Eigen::Dynamic, Eigen::Dynamic>>();
    }

    // Perform decomposition
    algorithm_->decompose(temp_input, Q, R);

    // Copy results to output matrix (Q|R concatenated)
    output_matrix.leftCols(cols) = Q.leftCols(
        cols); // Thin Q: first 'cols' columns of (rows x rows) Q matrix
    output_matrix.rightCols(cols) = R; // R matrix (full R matrix)

    return OperationReturn::SUCCESS; // Successfully processed one matrix
  } catch (const std::exception &e) {
    return OperationReturn::FAILURE; // Error occurred
  }
}

// Static helper functions
template <typename Scalar>
types::vector_shapes decomp_qr<Scalar>::compute_output_shapes(
    const types::vector_shapes &input_shapes) {
  if (input_shapes.empty()) {
    return {};
  }
  return _qr_output_shapes(input_shapes[0]);
}

template <typename Scalar>
void decomp_qr<Scalar>::set_algorithm(
    std::shared_ptr<qr_algorithm<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
        algo) {
  if (!algo) {
    throw std::invalid_argument("Algorithm cannot be null");
  }
  algorithm_ = algo;
}

template <typename Scalar>
void decomp_qr<Scalar>::validate_shape(
    const types::vector_shapes &input_shapes,
    const types::vector_shapes &output_shapes, const std::string &name) {}

template <typename Scalar>
std::vector<size_t>
decomp_qr<Scalar>::compute_sizes(const types::vector_shapes &shapes) {
  std::vector<size_t> sizes;
  return sizes;
}

// Sync block implementation
template <typename Scalar>
decomp_qr_sync_impl<Scalar>::decomp_qr_sync_impl(const types::shape &shape)
    : linalg_base<Scalar>(
          "decomp_qr_sync", _single_shape(shape), std::vector<std::string>{},
          _qr_output_shapes(shape), std::vector<std::string>{},
          array_broadcast_type::CUSTOM, error_tag_t::NONE, error_pdu_p::NONE),
      decomp_qr<Scalar>(shape) /* initialize virtual base */,
      linalg_base_sync<Scalar>(
          make_unique_qr_name("decomp_qr_sync"), _single_shape(shape),
          std::vector<std::string>(), _qr_output_shapes(shape),
          std::vector<std::string>(), array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE, gr::block::TPP_ALL_TO_ALL),
      decomp_qr_sync<Scalar>(shape) /* select the non-default ctor */ {
  // Validate matrix dimensions
  if (shape.size() < 2) {
    throw std::invalid_argument("QR decomposition requires 2D matrix input");
  }

  // Default to Householder QR
  this->set_algorithm(
      std::make_shared<
          eigen_householder_qr<Scalar, Eigen::Dynamic, Eigen::Dynamic>>());
}

template <typename Scalar>
typename decomp_qr_sync<Scalar>::sptr
decomp_qr_sync<Scalar>::make(const types::shape &shape) {
  return gnuradio::make_block_sptr<decomp_qr_sync_impl<Scalar>>(shape);
}

// PDU block factory methods - basic stub to avoid undefined references
template <typename Scalar>
typename decomp_qr_pdu<Scalar>::sptr
decomp_qr_pdu<Scalar>::make(const types::shape &shape) {
  throw std::runtime_error(
      "decomp_qr_pdu blocks are not yet fully implemented");
}

// Explicit instantiations

template class eigen_householder_qr<float>;
template class eigen_householder_qr<double>;
template class eigen_householder_qr<std::complex<float>>;
template class eigen_householder_qr<std::complex<double>>;

template class eigen_colpivhouseholder_qr<float>;
template class eigen_colpivhouseholder_qr<double>;
template class eigen_colpivhouseholder_qr<std::complex<float>>;
template class eigen_colpivhouseholder_qr<std::complex<double>>;

template class decomp_qr<float>;
template class decomp_qr<double>;
template class decomp_qr<std::complex<float>>;
template class decomp_qr<std::complex<double>>;

template class decomp_qr_sync<float>;
template class decomp_qr_sync<double>;
template class decomp_qr_sync<std::complex<float>>;
template class decomp_qr_sync<std::complex<double>>;

template class decomp_qr_sync_impl<float>;
template class decomp_qr_sync_impl<double>;
template class decomp_qr_sync_impl<std::complex<float>>;
template class decomp_qr_sync_impl<std::complex<double>>;

// Do not instantiate decomp_qr_pdu classes here; PDU path not implemented

} /* namespace linalg */
} /* namespace gr */
