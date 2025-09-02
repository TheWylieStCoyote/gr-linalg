/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "decomp_lu_impl.h"
#include <Eigen/Dense>
#include <atomic>
#include <complex>
#include <gnuradio/io_signature.h>
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/types.h>
#include <gnuradio/linalg/utils.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace gr {
namespace linalg {

// Unique name generator for block instances
namespace {
static std::atomic<uint64_t> g_lu_sync_uid{0};
inline std::string make_unique_lu_name(const char *base) {
  uint64_t id = g_lu_sync_uid.fetch_add(1, std::memory_order_relaxed);
  return std::string(base) + "#" + std::to_string(id);
}
} // namespace

// Helper function to create single shape vector
static inline types::vector_shapes _single_shape(const types::shape &s) {
  types::vector_shapes v;
  v.push_back(s);
  return v;
}

// Helper function to compute the output shapes for LU decomposition
// For sync blocks, we output a single matrix containing L, U, P concatenated
static inline types::vector_shapes
_lu_output_shapes(const types::shape &shape) {
  // Output is a single matrix with 3 times the width (L, U, P side by side)
  if (shape.size() != 2) {
    throw std::invalid_argument("LU decomposition requires 2D square matrix");
  }

  int rows = shape[0];
  int cols = shape[1];

  types::vector_shapes shapes;
  shapes.push_back({rows, cols * 3}); // L, U, P concatenated horizontally
  return shapes;
}

// Algorithm implementations
template <typename Scalar, int Rows, int Cols>
void eigen_partial_pivlu<Scalar, Rows, Cols>::decompose(
    const Eigen::Matrix<Scalar, Rows, Cols> &input,
    Eigen::Matrix<Scalar, Rows, Cols> &L, Eigen::Matrix<Scalar, Rows, Cols> &U,
    Eigen::Matrix<Scalar, Rows, Cols> &P) {

  Eigen::PartialPivLU<Eigen::Matrix<Scalar, Rows, Cols>> solver(input);
  auto LU_matrix = solver.matrixLU();

  // Extract L and U from the LU decomposition
  L = LU_matrix.template triangularView<Eigen::Lower>();
  U = LU_matrix.template triangularView<Eigen::Upper>();

  // Get permutation matrix
  P = solver.permutationP().toDenseMatrix().template cast<Scalar>();

  // Set diagonal of L to 1 (standard LU decomposition form)
  for (int i = 0; i < L.rows() && i < L.cols(); ++i) {
    L(i, i) = Scalar(1.0);
  }
}

template <typename Scalar, int Rows, int Cols>
void eigen_full_pivlu<Scalar, Rows, Cols>::decompose(
    const Eigen::Matrix<Scalar, Rows, Cols> &input,
    Eigen::Matrix<Scalar, Rows, Cols> &L, Eigen::Matrix<Scalar, Rows, Cols> &U,
    Eigen::Matrix<Scalar, Rows, Cols> &P) {

  Eigen::FullPivLU<Eigen::Matrix<Scalar, Rows, Cols>> solver(input);
  auto LU_matrix = solver.matrixLU();

  // Extract L and U from the LU decomposition
  L = LU_matrix.template triangularView<Eigen::Lower>();
  U = LU_matrix.template triangularView<Eigen::Upper>();

  // Get permutation matrix (only row permutation for FullPivLU)
  P = solver.permutationP().toDenseMatrix().template cast<Scalar>();

  // Set diagonal of L to 1
  for (int i = 0; i < L.rows() && i < L.cols(); ++i) {
    L(i, i) = Scalar(1.0);
  }
}

// decomp_lu base constructor definition
template <typename Scalar>
decomp_lu<Scalar>::decomp_lu(const types::shape &shape, const std::string &name)
    : linalg_base<Scalar>(
          name, _single_shape(shape), std::vector<std::string>{},
          _lu_output_shapes(shape), std::vector<std::string>{},
          array_broadcast_type::CUSTOM, error_tag_t::NONE, error_pdu_p::NONE) {
  if (shape.size() != 2 || shape[0] != shape[1]) {
    throw std::invalid_argument("LU decomposition requires square matrices");
  }
}

// decomp_lu base operation implementation
template <typename Scalar>
OperationReturn decomp_lu<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {

  if (input_matrices.empty() || output_matrices.empty()) {
    return OperationReturn::INVALID_SHAPE;
  }

  try {
    // Work directly with mapped matrices
    const auto &input_matrix = *input_matrices[0];
    auto &concatenated_output =
        *output_matrices[0]; // Single output with L, U, P concatenated

    // Create temporary matrices for algorithm interface
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> temp_input =
        input_matrix;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> temp_L, temp_U,
        temp_P;

    temp_L.resize(temp_input.rows(), temp_input.cols());
    temp_U.resize(temp_input.rows(), temp_input.cols());
    temp_P.resize(temp_input.rows(), temp_input.cols());

    // Use default partial pivoting algorithm if none set
    if (!algorithm_) {
      algorithm_ = std::make_shared<
          eigen_partial_pivlu<Scalar, Eigen::Dynamic, Eigen::Dynamic>>();
    }

    algorithm_->decompose(temp_input, temp_L, temp_U, temp_P);

    // Concatenate L, U, P horizontally into output matrix
    int rows = temp_input.rows();
    int cols = temp_input.cols();

    // Copy L, U, P into concatenated output matrix
    concatenated_output.block(0, 0, rows, cols) = temp_L;    // L in first cols
    concatenated_output.block(0, cols, rows, cols) = temp_U; // U in middle cols
    concatenated_output.block(0, cols * 2, rows, cols) =
        temp_P; // P in last cols

    return OperationReturn::SUCCESS;
  } catch (const std::exception &e) {
    return OperationReturn::FAILURE;
  }
}

template <typename Scalar>
void decomp_lu<Scalar>::set_algorithm(
    std::shared_ptr<lu_algorithm<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
        algo) {
  if (!algo) {
    throw std::invalid_argument("Algorithm cannot be null");
  }
  algorithm_ = algo;
}

template <typename Scalar>
types::vector_shapes decomp_lu<Scalar>::compute_output_shapes(
    const types::vector_shapes &input_shapes) {
  if (input_shapes.empty()) {
    return {};
  }
  return _lu_output_shapes(input_shapes[0]);
}

template <typename Scalar>
void decomp_lu<Scalar>::validate_shape(
    const types::vector_shapes &input_shapes,
    const types::vector_shapes &output_shapes, const std::string &name) {
  if (input_shapes.empty()) {
    throw std::invalid_argument(name + ": No input shapes provided");
  }

  const auto &shape = input_shapes[0];
  if (shape.size() != 2 || shape[0] != shape[1]) {
    throw std::invalid_argument(name +
                                ": LU decomposition requires square matrices");
  }
}

template <typename Scalar>
std::vector<size_t>
decomp_lu<Scalar>::compute_sizes(const types::vector_shapes &shapes) {
  return types::compute_sizes<Scalar>(shapes);
}

// Factory definition for sync blocks
template <typename Scalar>
typename decomp_lu_sync<Scalar>::sptr
decomp_lu_sync<Scalar>::make(const types::shape &shape) {
  return gnuradio::make_block_sptr<decomp_lu_sync_impl<Scalar>>(shape);
}

// decomp_lu_sync_impl constructor: most-derived, initializes in declared order

template <typename Scalar>
decomp_lu_sync_impl<Scalar>::decomp_lu_sync_impl(const types::shape &shape)
    : linalg_base<Scalar>(
          "decomp_lu_sync", _single_shape(shape), std::vector<std::string>{},
          _lu_output_shapes(shape), std::vector<std::string>{},
          array_broadcast_type::CUSTOM, error_tag_t::NONE, error_pdu_p::NONE),
      decomp_lu<Scalar>(shape),
      linalg_base_sync<Scalar>(
          make_unique_lu_name("decomp_lu_sync"), _single_shape(shape),
          std::vector<std::string>(), _lu_output_shapes(shape),
          std::vector<std::string>(), array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE, gr::block::TPP_ALL_TO_ALL),
      decomp_lu_sync<Scalar>(shape) {
  // Validate square matrix
  if (shape.size() != 2 || shape[0] != shape[1]) {
    throw std::invalid_argument("LU decomposition requires square matrices");
  }

  // Default to partial pivot LU
  this->set_algorithm(
      std::make_shared<
          eigen_partial_pivlu<Scalar, Eigen::Dynamic, Eigen::Dynamic>>());
}

// Explicit template instantiations for common types
// Algorithm instantiations
template class eigen_partial_pivlu<float>;
template class eigen_partial_pivlu<double>;
template class eigen_partial_pivlu<std::complex<float>>;
template class eigen_partial_pivlu<std::complex<double>>;

template class eigen_full_pivlu<float>;
template class eigen_full_pivlu<double>;
template class eigen_full_pivlu<std::complex<float>>;
template class eigen_full_pivlu<std::complex<double>>;

// Base class instantiations
template class decomp_lu<float>;
template class decomp_lu<double>;
template class decomp_lu<std::complex<float>>;
template class decomp_lu<std::complex<double>>;

// Sync block instantiations
template class decomp_lu_sync<float>;
template class decomp_lu_sync<double>;
template class decomp_lu_sync<std::complex<float>>;
template class decomp_lu_sync<std::complex<double>>;

template class decomp_lu_sync_impl<float>;
template class decomp_lu_sync_impl<double>;
template class decomp_lu_sync_impl<std::complex<float>>;
template class decomp_lu_sync_impl<std::complex<double>>;

// PDU blocks not yet implemented

} /* namespace linalg */
} /* namespace gr */
