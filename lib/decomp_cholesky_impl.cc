/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <atomic>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

#include <gnuradio/io_signature.h>
#include <gnuradio/linalg/decomp_cholesky.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/types.h>
#include <gnuradio/linalg/utils.h>

#include "decomp_cholesky_impl.h"

namespace gr {
namespace linalg {

namespace {
static std::atomic<uint64_t> g_chol_sync_uid{0};
inline std::string make_unique_chol_name(const char *base) {
  uint64_t id = g_chol_sync_uid.fetch_add(1, std::memory_order_relaxed);
  return std::string(base) + "#" + std::to_string(id);
}

static inline types::vector_shapes _single_shape(const types::shape &s) {
  types::vector_shapes v;
  v.push_back(s);
  return v;
}
} // namespace

// Algorithms

template <typename Scalar>
void eigen_llt_cholesky<Scalar>::decompose(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &L) {
  Eigen::LLT<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> solver(
      input);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error("Cholesky LLT failed: matrix not SPD");
  }
  L = solver.matrixL();
}

template <typename Scalar>
void eigen_ldlt_cholesky<Scalar>::decompose(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &L) {
  Eigen::LDLT<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> solver(
      input);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error("Cholesky LDLT failed: matrix not SPD");
  }
  // Convert LDL^T to L' such that A = L' L'^T
  // LDLT gives A = L D L^T; take D^{1/2} into L
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Lin = solver.matrixL();
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> D = solver.vectorD();
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Dhalf =
      D.array().sqrt().matrix().asDiagonal();
  L = Lin * Dhalf;
}

// Base

template <typename Scalar>
decomp_cholesky<Scalar>::decomp_cholesky(const types::shape &shape,
                                         const std::string &name)
    : linalg_base<Scalar>(name, _single_shape(shape), _single_shape(shape),
                          array_broadcast_type::NONE),
      algorithm_(std::make_shared<eigen_llt_cholesky<Scalar>>()) {
  validate_shape(_single_shape(shape), _single_shape(shape), name);
}

template <typename Scalar>
OperationReturn decomp_cholesky<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {
  if (input_matrices.empty() || output_matrices.empty()) {
    return OperationReturn::INVALID_SHAPE;
  }
  try {
    const auto &Amap = *input_matrices[0];
    auto &Lmap = *output_matrices[0];

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> A = Amap;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> L;
    if (!algorithm_) {
      algorithm_ = std::make_shared<eigen_llt_cholesky<Scalar>>();
    }
    algorithm_->decompose(A, L);
    Lmap = L;
    return OperationReturn::SUCCESS;
  } catch (...) {
    // Fallback: try Eigen LLT directly
    try {
      const auto &Amap = *input_matrices[0];
      auto &Lmap = *output_matrices[0];
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> A = Amap;
      Eigen::LLT<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> solver(
          A);
      auto L = solver.matrixL();
      Lmap = L;
      return OperationReturn::SUCCESS;
    } catch (...) {
      return OperationReturn::FAILURE;
    }
  }
}

template <typename Scalar>
void decomp_cholesky<Scalar>::set_algorithm(
    std::shared_ptr<cholesky_algorithm<Scalar>> algo) {
  if (!algo)
    throw std::invalid_argument("Algorithm cannot be null");
  algorithm_ = std::move(algo);
}

template <typename Scalar>
types::vector_shapes decomp_cholesky<Scalar>::compute_output_shapes(
    const types::vector_shapes &input_shapes) {
  if (input_shapes.empty())
    return {};
  return _single_shape(input_shapes[0]);
}

template <typename Scalar>
void decomp_cholesky<Scalar>::validate_shape(
    const types::vector_shapes &input_shapes,
    const types::vector_shapes &output_shapes, const std::string &name) {
  if (input_shapes.empty()) {
    throw std::invalid_argument(name + ": No input shapes provided");
  }
  const auto &s = input_shapes[0];
  if (s.size() != 2 || s[0] != s[1]) {
    throw std::invalid_argument(name + ": Cholesky requires square matrices");
  }
  if (!output_shapes.empty() && output_shapes[0] != s) {
    throw std::invalid_argument(name + ": Output shape must match input shape");
  }
}

template <typename Scalar>
std::vector<size_t>
decomp_cholesky<Scalar>::compute_sizes(const types::vector_shapes &shapes) {
  return types::compute_sizes<Scalar>(shapes);
}

// Sync impl ctor

template <typename Scalar>
decomp_cholesky_sync_impl<Scalar>::decomp_cholesky_sync_impl(
    const types::shape &shape)
    : linalg_base<Scalar>("decomp_cholesky_sync", _single_shape(shape),
                          std::vector<std::string>{}, _single_shape(shape),
                          std::vector<std::string>{},
                          array_broadcast_type::NONE, error_tag_t::NONE,
                          error_pdu_p::NONE),
      decomp_cholesky<Scalar>(shape),
      linalg_base_sync<Scalar>("decomp_cholesky_sync", _single_shape(shape),
                               std::vector<std::string>{}, _single_shape(shape),
                               std::vector<std::string>{},
                               array_broadcast_type::NONE, error_tag_t::NONE,
                               error_pdu_p::NONE, gr::block::TPP_ALL_TO_ALL),
      decomp_cholesky_sync<Scalar>(shape) {
  // Default to LLT
  this->set_algorithm(std::make_shared<eigen_llt_cholesky<Scalar>>());
}

// Factory

template <typename Scalar>
typename decomp_cholesky_sync<Scalar>::sptr
decomp_cholesky_sync<Scalar>::make(const types::shape &shape) {
  return gnuradio::make_block_sptr<decomp_cholesky_sync_impl<Scalar>>(shape);
}

// Explicit instantiations

template class eigen_llt_cholesky<float>;
template class eigen_llt_cholesky<double>;
template class eigen_llt_cholesky<std::complex<float>>;
template class eigen_llt_cholesky<std::complex<double>>;

template class eigen_ldlt_cholesky<float>;
template class eigen_ldlt_cholesky<double>;
template class eigen_ldlt_cholesky<std::complex<float>>;
template class eigen_ldlt_cholesky<std::complex<double>>;

template class decomp_cholesky<float>;
template class decomp_cholesky<double>;
template class decomp_cholesky<std::complex<float>>;
template class decomp_cholesky<std::complex<double>>;

template class decomp_cholesky_sync<float>;
template class decomp_cholesky_sync<double>;
template class decomp_cholesky_sync<std::complex<float>>;
template class decomp_cholesky_sync<std::complex<double>>;

template class decomp_cholesky_sync_impl<float>;
template class decomp_cholesky_sync_impl<double>;
template class decomp_cholesky_sync_impl<std::complex<float>>;
template class decomp_cholesky_sync_impl<std::complex<double>>;

} /* namespace linalg */
} /* namespace gr */
