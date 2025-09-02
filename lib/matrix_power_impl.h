/* -*- c++ -*- */
/*
 * Copyright 2024 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_LINALG_MATRIX_POWER_IMPL_H
#define INCLUDED_LINALG_MATRIX_POWER_IMPL_H

#include <Eigen/Dense>
#include <Eigen/LU>
#include <cmath>
#include <complex>
#include <gnuradio/linalg/matrix_power.h>

namespace gr {
namespace linalg {

namespace matrix_power_detail {

/**
 * @brief Helper function for safe integer exponent extraction
 */
template <typename Scalar>
bool extract_integer_exponent(const Scalar &value, int &result) {
  if constexpr (std::is_same_v<Scalar, float> ||
                std::is_same_v<Scalar, double>) {
    // For real scalars, check if it's close to an integer
    double rounded = std::round(static_cast<double>(value));
    if (std::abs(static_cast<double>(value) - rounded) < 1e-10) {
      result = static_cast<int>(rounded);
      return true;
    }
  } else if constexpr (std::is_same_v<Scalar, std::complex<float>> ||
                       std::is_same_v<Scalar, std::complex<double>>) {
    // For complex scalars, must be real and integer
    if (std::abs(value.imag()) < 1e-10) {
      double rounded = std::round(value.real());
      if (std::abs(value.real() - rounded) < 1e-10) {
        result = static_cast<int>(rounded);
        return true;
      }
    }
  }
  return false;
}

/**
 * @brief Compute matrix power using repeated squaring for positive integer
 * exponents
 */
template <typename Scalar>
void matrix_power_positive(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &A, int n,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &result) {

  if (n == 0) {
    result = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(
        A.rows(), A.cols());
    return;
  }

  if (n == 1) {
    result = A;
    return;
  }

  // Repeated squaring algorithm: A^n = A^(n/2) * A^(n/2) * A^(n%2)
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> base = A;
  result = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(
      A.rows(), A.cols());

  int exp = n;
  while (exp > 0) {
    if (exp % 2 == 1) {
      result *= base;
    }
    base *= base;
    exp /= 2;
  }
}

/**
 * @brief Compute element-wise power with proper complex number handling
 */
template <typename Scalar>
Scalar elementwise_pow(const Scalar &base, const Scalar &exp) {
  if constexpr (std::is_same_v<Scalar, float> ||
                std::is_same_v<Scalar, double>) {
    return std::pow(base, exp);
  } else {
    // Complex power: z^w = exp(w * log(z))
    // Handle special cases for better numerical stability
    if (std::abs(base) == 0.0) {
      if (exp.real() > 0)
        return Scalar(0.0, 0.0);
      if (exp.real() == 0 && exp.imag() == 0)
        return Scalar(1.0, 0.0);
      // For negative real part of exponent with zero base, return infinity
      return Scalar(
          std::numeric_limits<typename Scalar::value_type>::infinity(), 0.0);
    }
    return std::pow(base, exp);
  }
}

} // namespace matrix_power_detail

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_POWER_IMPL_H */