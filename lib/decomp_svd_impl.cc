/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "decomp_svd_impl.h"
#include <Eigen/Dense>
#include <algorithm>
#include <gnuradio/io_signature.h>
#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/types.h>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace gr {
namespace linalg {

namespace {
static inline types::vector_shapes _single_shape(const types::shape &s) {
  types::vector_shapes v;
  v.push_back(s);
  return v;
}
static inline types::vector_shapes _svd_out(const types::shape &shape) {
  if (shape.size() != 2)
    throw std::invalid_argument("decomp_svd requires 2D shape");
  const int m = shape[0];
  const int n = shape[1];
  types::vector_shapes out;
  out.push_back({m, m});
  out.push_back({std::min(m, n), 1});
  out.push_back({n, n});
  return out;
}
} // namespace

// Algorithms

template <typename Scalar>
void eigen_jacobi_svd<Scalar>::decompose(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &U,
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &S,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &V) {
  using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  Eigen::JacobiSVD<Mat> solver(input,
                               Eigen::ComputeFullU | Eigen::ComputeFullV);
  if (solver.info() != Eigen::Success)
    throw std::runtime_error("JacobiSVD failed");
  U = solver.matrixU();
  S = solver.singularValues();
  V = solver.matrixV().transpose();
}

template <typename Scalar>
void eigen_bdc_svd<Scalar>::decompose(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &U,
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &S,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &V) {
  using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  Eigen::BDCSVD<Mat> solver(input, Eigen::ComputeFullU | Eigen::ComputeFullV);
  if (solver.info() != Eigen::Success)
    throw std::runtime_error("BDCSVD failed");
  U = solver.matrixU();
  S = solver.singularValues();
  V = solver.matrixV().transpose();
}

// Base

template <typename Scalar>
decomp_svd<Scalar>::decomp_svd(const types::shape &shape,
                               std::shared_ptr<svd_algorithm<Scalar>> algorithm)
    : linalg_base<Scalar>(
          "decomp_svd", _single_shape(shape), std::vector<std::string>{},
          _svd_out(shape), std::vector<std::string>{},
          array_broadcast_type::CUSTOM, error_tag_t::NONE, error_pdu_p::NONE),
      algorithm_(algorithm ? algorithm
                           : std::shared_ptr<svd_algorithm<Scalar>>(
                                 new eigen_jacobi_svd<Scalar>())) {
  if (shape.size() != 2)
    throw std::invalid_argument("decomp_svd requires 2D shape");
}

template <typename Scalar>
OperationReturn
decomp_svd<Scalar>::operation(types::vector_const_matrix_map<Scalar> &ins,
                              types::vector_matrix_map<Scalar> &outs) {
  if (ins.size() != 1 || outs.size() != 3)
    return OperationReturn::INVALID_SHAPE;
  try {
    const auto &A = *ins[0];
    auto &U = *outs[0];
    auto &S = *outs[1];
    auto &V = *outs[2];
    const int m = A.rows();
    const int n = A.cols();
    if (U.rows() != m || U.cols() != m)
      return OperationReturn::INVALID_SHAPE;
    if (S.rows() != std::min(m, n) || S.cols() != 1)
      return OperationReturn::INVALID_SHAPE;
    if (V.rows() != n || V.cols() != n)
      return OperationReturn::INVALID_SHAPE;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Utmp, Vtmp;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Stmp;
    algorithm_->decompose(A, Utmp, Stmp, Vtmp);
    U = Utmp;
    S = Stmp;
    V = Vtmp;
    return OperationReturn::SUCCESS;
  } catch (...) {
    return OperationReturn::FAILURE;
  }
}

template <typename Scalar>
void decomp_svd<Scalar>::set_algorithm(
    std::shared_ptr<svd_algorithm<Scalar>> algo) {
  if (algo)
    algorithm_ = algo;
}

template <typename Scalar>
void decomp_svd<Scalar>::validate_shape(const types::vector_shapes &in,
                                        const types::vector_shapes &out,
                                        const std::string &) {
  linalg_base<Scalar>::validate_shapes(in, out, "decomp_svd",
                                       array_broadcast_type::CUSTOM);
  if (in.size() != 1 || in[0].size() != 2)
    throw std::invalid_argument("decomp_svd expects one 2D input shape");
}

template <typename Scalar>
types::vector_shapes
decomp_svd<Scalar>::compute_output_shapes(const types::vector_shapes &in) {
  if (in.size() != 1)
    return {};
  return _svd_out(in[0]);
}

// Sync impl

template <typename Scalar>
decomp_svd_sync_impl<Scalar>::decomp_svd_sync_impl(
    const types::shape &shape, std::shared_ptr<svd_algorithm<Scalar>> algorithm)
    : linalg_base<Scalar>(
          "decomp_svd_sync", _single_shape(shape), std::vector<std::string>{},
          _svd_out(shape), std::vector<std::string>{},
          array_broadcast_type::CUSTOM, error_tag_t::NONE, error_pdu_p::NONE),
      decomp_svd<Scalar>(shape, algorithm),
      linalg_base_sync<Scalar>("decomp_svd_sync", _single_shape(shape),
                               std::vector<std::string>{}, _svd_out(shape),
                               std::vector<std::string>{},
                               array_broadcast_type::CUSTOM, error_tag_t::NONE,
                               error_pdu_p::NONE, gr::block::TPP_ALL_TO_ALL),
      decomp_svd_sync<Scalar>(shape, algorithm) {
  if (!this->algorithm_) {
    this->algorithm_ = std::make_shared<eigen_jacobi_svd<Scalar>>();
  }
}

// Factory

template <typename Scalar>
typename decomp_svd_sync<Scalar>::sptr decomp_svd_sync<Scalar>::make(
    const types::shape &shape,
    std::shared_ptr<svd_algorithm<Scalar>> algorithm) {
  return gnuradio::make_block_sptr<decomp_svd_sync_impl<Scalar>>(shape,
                                                                 algorithm);
}

// PDU impl - commented out since header uses minimal stub pattern
// The current header defines decomp_svd_pdu as a simple stub that doesn't
// require decomp_svd_pdu_impl

// Explicit instantiations for PDU impl if needed in future
// template class decomp_svd_pdu_impl<float>;
// template class decomp_svd_pdu_impl<double>;

// Explicit instantiations

template class eigen_jacobi_svd<float>;
template class eigen_jacobi_svd<double>;

template class eigen_bdc_svd<float>;
template class eigen_bdc_svd<double>;

template class decomp_svd<float>;
template class decomp_svd<double>;

template class decomp_svd_sync<float>;
template class decomp_svd_sync<double>;

template class decomp_svd_pdu<float>;
template class decomp_svd_pdu<double>;

template class decomp_svd_sync_impl<float>;
template class decomp_svd_sync_impl<double>;

} // namespace linalg
} // namespace gr
