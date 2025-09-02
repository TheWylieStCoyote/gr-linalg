/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "decomp_eigen_impl.h"
#include <Eigen/Eigenvalues>
#include <atomic>
#include <gnuradio/io_signature.h>

namespace gr {
namespace linalg {

// Unique name generator for Eigen instances
namespace {
static std::atomic<uint64_t> g_eigen_sync_uid{0};
inline std::string make_unique_eigen_name(const char *base) {
  uint64_t id = g_eigen_sync_uid.fetch_add(1, std::memory_order_relaxed);
  return std::string(base) + "#" + std::to_string(id);
}

static inline types::vector_shapes _single_shape(const types::shape &s) {
  types::vector_shapes v;
  v.push_back(s);
  return v;
}

static inline types::vector_shapes
_eigen_output_shapes(const types::shape &shape) {
  // Outputs: [eigenvalues Nx1, eigenvectors NxN]
  if (shape.size() != 2) {
    throw std::invalid_argument(
        "Eigen decomposition requires 2D square matrix");
  }
  int n = shape[0];
  if (shape[1] != n) {
    throw std::invalid_argument(
        "Eigen decomposition requires square matrix (NxN)");
  }
  types::vector_shapes out;
  out.push_back({n, 1});
  out.push_back({n, n});
  return out;
}
} // namespace

// Algorithm implementations

template <typename Scalar>
void eigen_general_solver<Scalar>::decompose(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input,
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &eigenvalues,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &eigenvectors) {
  using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  Eigen::EigenSolver<Mat> solver(input, /* computeEigenvectors = */ true);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error("EigenSolver failed");
  }
  // For real Scalar, EigenSolver returns complex; cast real parts
  if constexpr (std::is_same_v<Scalar, float> ||
                std::is_same_v<Scalar, double>) {
    eigenvalues = solver.eigenvalues().real();
    eigenvectors = solver.eigenvectors().real();
  } else {
    eigenvalues = solver.eigenvalues();
    eigenvectors = solver.eigenvectors();
  }
}

template <typename Scalar>
void eigen_selfadjoint_solver<Scalar>::decompose(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input,
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &eigenvalues,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &eigenvectors) {
  using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  Eigen::SelfAdjointEigenSolver<Mat> solver(input);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error("SelfAdjointEigenSolver failed");
  }
  eigenvalues = solver.eigenvalues();
  eigenvectors = solver.eigenvectors();
}

// decomp_eigen base

template <typename Scalar>
static inline types::vector_shapes
_compute_eigen_outputs_from_inputs(const types::vector_shapes &in_shapes) {
  if (in_shapes.size() != 1)
    return {};
  return _eigen_output_shapes(in_shapes[0]);
}

template <typename Scalar>
decomp_eigen<Scalar>::decomp_eigen(const types::shape &shape,
                                   const std::string &name)
    : linalg_base<Scalar>(name, _single_shape(shape),
                          _eigen_output_shapes(shape),
                          array_broadcast_type::CUSTOM),
      algorithm_(std::make_shared<eigen_selfadjoint_solver<Scalar>>()) {
  if (shape.size() != 2 || shape[0] != shape[1]) {
    throw std::invalid_argument("decomp_eigen requires square shape NxN");
  }
}

template <typename Scalar>
OperationReturn decomp_eigen<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {
  if (input_matrices.size() != 1 || output_matrices.size() != 2) {
    return OperationReturn::INVALID_SHAPE;
  }
  try {
    const auto &A = *input_matrices[0];
    auto &eigvals = *output_matrices[0];
    auto &eigvecs = *output_matrices[1];

    if (A.rows() != A.cols())
      return OperationReturn::INVALID_SHAPE;
    const int n = A.rows();
    if (eigvals.rows() != n || eigvals.cols() != 1)
      return OperationReturn::INVALID_SHAPE;
    if (eigvecs.rows() != n || eigvecs.cols() != n)
      return OperationReturn::INVALID_SHAPE;

    // Execute algorithm
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> temp_vals(n);
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> temp_vecs(n, n);
    algorithm_->decompose(A, temp_vals, temp_vecs);

    eigvals = temp_vals;
    eigvecs = temp_vecs;
    return OperationReturn::SUCCESS;
  } catch (...) {
    return OperationReturn::FAILURE;
  }
}

template <typename Scalar>
types::vector_shapes decomp_eigen<Scalar>::compute_output_shapes(
    const types::vector_shapes &input_shapes) {
  return _compute_eigen_outputs_from_inputs<Scalar>(input_shapes);
}

template <typename Scalar>
void decomp_eigen<Scalar>::validate_shape(
    const types::vector_shapes &input_shapes,
    const types::vector_shapes &output_shapes, const std::string &name) {
  (void)name;
  // Skip strict base validation due to multi-output using CUSTOM broadcast
  linalg_base<Scalar>::validate_shapes(input_shapes, output_shapes,
                                       "decomp_eigen",
                                       array_broadcast_type::CUSTOM);
  // Additional basic checks
  if (input_shapes.size() != 1) {
    throw std::invalid_argument("decomp_eigen expects exactly one input shape");
  }
  const auto &s = input_shapes[0];
  if (s.size() != 2 || s[0] != s[1]) {
    throw std::invalid_argument("decomp_eigen requires square input shape");
  }
}

template <typename Scalar>
std::vector<size_t>
decomp_eigen<Scalar>::compute_sizes(const types::vector_shapes &shapes) {
  return types::compute_sizes<Scalar>(shapes);
}

// Sync impl

template <typename Scalar>
decomp_eigen_sync_impl<Scalar>::decomp_eigen_sync_impl(
    const types::shape &shape)
    : linalg_base<Scalar>(
          "decomp_eigen_sync", _single_shape(shape), std::vector<std::string>{},
          _eigen_output_shapes(shape), std::vector<std::string>{},
          array_broadcast_type::CUSTOM, error_tag_t::NONE, error_pdu_p::NONE),
      decomp_eigen<Scalar>(shape),
      linalg_base_sync<Scalar>(
          make_unique_eigen_name("decomp_eigen_sync"), _single_shape(shape),
          std::vector<std::string>{}, _eigen_output_shapes(shape),
          std::vector<std::string>{}, array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE, gr::block::TPP_ALL_TO_ALL),
      decomp_eigen_sync<Scalar>(shape) {}

template <typename Scalar>
typename decomp_eigen_sync<Scalar>::sptr
decomp_eigen_sync<Scalar>::make(const types::shape &shape) {
  return gnuradio::make_block_sptr<decomp_eigen_sync_impl<Scalar>>(shape);
}

// Explicit instantiations

template class eigen_general_solver<float>;
template class eigen_general_solver<double>;

template class eigen_selfadjoint_solver<float>;
template class eigen_selfadjoint_solver<double>;

template class decomp_eigen<float>;
template class decomp_eigen<double>;

template class decomp_eigen_sync<float>;
template class decomp_eigen_sync<double>;

template class decomp_eigen_sync_impl<float>;
template class decomp_eigen_sync_impl<double>;

} // namespace linalg
} // namespace gr
