/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_UTILS_FIXED_H
#define INCLUDED_LINALG_UTILS_FIXED_H

#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <gnuradio/linalg/api.h>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace gr {
namespace linalg {
namespace utils {

/*!
 * \brief Computes the product of elements in a vector.
 * \param vec The input vector.
 * \return The product of the elements in the vector.
 */
template <class T>
T product(const std::vector<T> &vec) {
  return std::accumulate(vec.begin(), vec.end(), T(1), std::multiplies<T>());
}

/*!
 * \brief Computes the sum of elements in a vector.
 * \param vec The input vector.
 * \return The sum of the elements in the vector.
 */
template <class T>
T sum(const std::vector<T> &vec) {
  return std::accumulate(vec.begin(), vec.end(), T(0), std::plus<T>());
}

/*!
 * \brief Computes the mean of elements in a vector.
 * \param vec The input vector.
 * \return The mean of the elements in the vector.
 * \throws std::invalid_argument if the vector is empty.
 */
template <class T>
T mean(const std::vector<T> &vec) {
  if (vec.empty()) {
    throw std::invalid_argument("Cannot compute mean of an empty vector");
  }
  return sum(vec) / static_cast<T>(vec.size());
}

/*!
 * \brief Generates a linearly spaced vector.
 * \param start The starting value.
 * \param stop The ending value.
 * \param num_points The number of points in the vector.
 * \return A vector containing linearly spaced values.
 */
template <class T>
std::vector<T> linspace(T start, T stop, size_t num_points);

/*!
 * \brief Generates a vector of zeros.
 * \param size The size of the vector.
 * \return A vector filled with zeros.
 */
template <class T>
std::vector<T> zeros(size_t size) {
  return std::vector<T>(size, static_cast<T>(0));
}

/*!
 * \brief Generates a vector of ones.
 * \param size The size of the vector.
 * \return A vector filled with ones.
 */
template <class T>
std::vector<T> ones(size_t size) {
  return std::vector<T>(size, static_cast<T>(1));
}

/*!
 * \brief Generates an identity matrix as a vector.
 * \param n The size of the identity matrix (n x n).
 * \return A vector representing the identity matrix.
 */
template <class T>
std::vector<T> eye(size_t n);

/*!
 * \brief Computes the variance of elements in a vector.
 * \param vec The input vector.
 * \return The variance of the elements in the vector.
 * \throws std::invalid_argument if the vector is empty.
 */
template <class T>
T variance(const std::vector<T> &vec) {
  if (vec.empty()) {
    throw std::invalid_argument("Cannot compute variance of an empty vector");
  }
  T mean_val = mean(vec);
  T sum_sq_diff = 0;
  for (const auto &val : vec) {
    T diff = val - mean_val;
    sum_sq_diff += diff * diff;
  }
  return sum_sq_diff / static_cast<T>(vec.size());
}

/*!
 * \brief Computes the standard deviation of elements in a vector.
 * \param vec The input vector.
 * \return The standard deviation of the elements in the vector.
 * \throws std::invalid_argument if the vector is empty.
 */
template <class T>
T std_dev(const std::vector<T> &vec) {
  return std::sqrt(variance(vec));
}

} // namespace utils
} // namespace linalg
} // namespace gr

#endif // INCLUDED_LINALG_UTILS_FIXED_H
