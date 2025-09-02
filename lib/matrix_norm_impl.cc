/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "matrix_norm_impl.h"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <cmath>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <stdexcept>
#include <vector>

namespace gr {
namespace linalg {

static inline types::vector_shapes _single(const types::shape &s) {
  return types::vector_shapes{s};
}

// Static method implementation for matrix_norm
template <typename Scalar>
types::vector_shapes matrix_norm<Scalar>::compute_output_shapes(
    const types::vector_shapes &input_shapes) {
  types::vector_shapes out;
  out.reserve(input_shapes.size());
  for (auto const &s : input_shapes) {
    (void)s;
    out.push_back(types::shape{1, 1});
  }
  return out;
}

// Factory
template <typename Scalar>
typename matrix_norm_sync<Scalar>::sptr
matrix_norm_sync<Scalar>::make(const types::shape &shape, norm_type norm) {
  return gnuradio::make_block_sptr<matrix_norm_sync_impl<Scalar>>(shape, norm);
}

template <typename Scalar>
matrix_norm_sync<Scalar>::matrix_norm_sync(const types::shape &shape,
                                           norm_type norm)
    : matrix_norm<Scalar>(),
      linalg_base_sync<Scalar>("matrix_norm", _single(shape),
                               _single(types::shape{1, 1})) {
  this->d_norm_type = norm;
}

/*
 * Impl ctor initializes virtual base linalg_base
 */
template <typename Scalar>
matrix_norm_sync_impl<Scalar>::matrix_norm_sync_impl(const shape &shape,
                                                     norm_type norm)
    : linalg_base<Scalar>(
          "matrix_norm", _single(shape), std::vector<std::string>(),
          _single(types::shape{1, 1}), std::vector<std::string>(),
          array_broadcast_type::CUSTOM, error_tag_t::NONE, error_pdu_p::NONE),
      linalg_base_sync<Scalar>(
          "matrix_norm", _single(shape), std::vector<std::string>(),
          _single(types::shape{1, 1}), std::vector<std::string>(),
          array_broadcast_type::CUSTOM, error_tag_t::NONE, error_pdu_p::NONE,
          gr::block::TPP_ALL_TO_ALL),
      matrix_norm_sync<Scalar>(shape, norm) {}

/*
 * operation: out = norm(in)
 */
template <typename Scalar>
OperationReturn matrix_norm_sync_impl<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {

  if (input_matrices.size() != 1 || output_matrices.size() != 1) {
    return OperationReturn::INVALID_SHAPE;
  }

  const auto &input_matrix = *input_matrices[0];
  auto &output_matrix = *output_matrices[0];

  if (output_matrix.rows() != 1 || output_matrix.cols() != 1) {
    return OperationReturn::INVALID_SHAPE;
  }

  typename Eigen::NumTraits<Scalar>::Real norm_value;

  switch (this->d_norm_type) {
  case norm_type::L1:
    // L1 norm: maximum column sum
    norm_value = input_matrix.colwise().template lpNorm<1>().maxCoeff();
    break;

  case norm_type::L2:
    // L2 norm: spectral norm (largest singular value)
    if constexpr (std::is_same_v<Scalar, float> ||
                  std::is_same_v<Scalar, double>) {
      Eigen::JacobiSVD<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
          svd(input_matrix);
      norm_value = svd.singularValues()(0);
    } else {
      // For complex types, use Frobenius norm as approximation
      norm_value = input_matrix.norm();
    }
    break;

  case norm_type::Frobenius:
    // Frobenius norm: sqrt of sum of squares
    norm_value = input_matrix.norm();
    break;

  case norm_type::Max:
    // Max norm: maximum absolute value
    norm_value = input_matrix.cwiseAbs().maxCoeff();
    break;

  default:
    return OperationReturn::INVALID_SHAPE;
  }

  // Cast the real norm value to the output scalar type
  if constexpr (std::is_same_v<Scalar, std::complex<float>> ||
                std::is_same_v<Scalar, std::complex<double>>) {
    output_matrix(0, 0) =
        Scalar(norm_value, 0); // Real part only for complex output
  } else {
    output_matrix(0, 0) = static_cast<Scalar>(norm_value);
  }

  return OperationReturn::SUCCESS;
}

// Explicit instantiations

// Base template class instantiations (needed for virtual inheritance)
template class matrix_norm<float>;
template class matrix_norm<double>;
template class matrix_norm<std::complex<float>>;
template class matrix_norm<std::complex<double>>;

template class matrix_norm_sync<float>;
template class matrix_norm_sync<double>;
template class matrix_norm_sync<std::complex<float>>;
template class matrix_norm_sync<std::complex<double>>;

template class matrix_norm_sync_impl<float>;
template class matrix_norm_sync_impl<double>;
template class matrix_norm_sync_impl<std::complex<float>>;
template class matrix_norm_sync_impl<std::complex<double>>;

// Note: Static method instantiations are not needed when we instantiate
// the full template class above, as they are automatically instantiated

} /* namespace linalg */
} /* namespace gr */