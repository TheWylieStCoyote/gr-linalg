/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "matrix_rank_impl.h"
#include <Eigen/Dense>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace gr {
namespace linalg {

static inline types::vector_shapes _single(const types::shape &s) {
  return types::vector_shapes{s};
}

// Factory
template <typename Scalar>
typename matrix_rank_sync<Scalar>::sptr
matrix_rank_sync<Scalar>::make(const types::shape &shape) {
  return gnuradio::make_block_sptr<matrix_rank_sync_impl<Scalar>>(shape);
}

// matrix_rank_sync ctor bridging to linalg_base_sync
template <typename Scalar>
matrix_rank_sync<Scalar>::matrix_rank_sync(const types::shape &shape)
    : matrix_rank<Scalar>(),
      linalg_base_sync<Scalar>("matrix_rank", _single(shape),
                               _single(types::shape{1, 1})) {}

// Impl constructor initializes virtual base linalg_base and validates shape
template <typename Scalar>
matrix_rank_sync_impl<Scalar>::matrix_rank_sync_impl(
    const typename matrix_rank_sync_impl<Scalar>::shape &shape)
    : linalg_base<Scalar>(
          "matrix_rank", _single(shape), std::vector<std::string>(),
          _single(types::shape{1, 1}), std::vector<std::string>(),
          array_broadcast_type::CUSTOM, error_tag_t::NONE, error_pdu_p::NONE),
      matrix_rank_sync<Scalar>(shape) {
  if (shape.size() != 2) {
    throw std::invalid_argument("matrix_rank expects a 2D shape");
  }
}

// operation: out = rank(in)
template <typename Scalar>
OperationReturn matrix_rank_sync_impl<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {
  if (input_matrices.size() != 1 || output_matrices.size() != 1)
    return OperationReturn::INVALID_SHAPE;
  auto &in = *input_matrices[0];
  auto &out = *output_matrices[0];
  if (in.rows() == 0 || in.cols() == 0)
    return OperationReturn::INVALID_SHAPE;

  using Real = typename Eigen::NumTraits<Scalar>::Real;
  Eigen::JacobiSVD<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> svd(
      in, Eigen::ComputeThinU | Eigen::ComputeThinV);
  auto sing = svd.singularValues();
  if (sing.size() == 0) {
    out(0, 0) = Scalar(0);
    return OperationReturn::SUCCESS;
  }
  Real tol = Real(std::numeric_limits<Real>::epsilon()) *
             std::max(in.rows(), in.cols()) * sing(0);
  int r = 0;
  for (int i = 0; i < sing.size(); ++i) {
    if (sing(i) > tol)
      ++r;
  }
  out(0, 0) = static_cast<Scalar>(r);
  return OperationReturn::SUCCESS;
}

// Explicit instantiations
template class matrix_rank_sync_impl<float>;
template class matrix_rank_sync_impl<double>;
template class matrix_rank_sync_impl<std::complex<float>>;
template class matrix_rank_sync_impl<std::complex<double>>;

template matrix_rank_sync<float>::matrix_rank_sync(const types::shape &);
template matrix_rank_sync<double>::matrix_rank_sync(const types::shape &);
template matrix_rank_sync<std::complex<float>>::matrix_rank_sync(
    const types::shape &);
template matrix_rank_sync<std::complex<double>>::matrix_rank_sync(
    const types::shape &);

template typename matrix_rank_sync<float>::sptr
matrix_rank_sync<float>::make(const types::shape &);
template typename matrix_rank_sync<double>::sptr
matrix_rank_sync<double>::make(const types::shape &);
template typename matrix_rank_sync<std::complex<float>>::sptr
matrix_rank_sync<std::complex<float>>::make(const types::shape &);
template typename matrix_rank_sync<std::complex<double>>::sptr
matrix_rank_sync<std::complex<double>>::make(const types::shape &);

} // namespace linalg
} // namespace gr
