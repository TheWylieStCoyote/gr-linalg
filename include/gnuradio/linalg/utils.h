/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_UTILS_H
#define INCLUDED_LINALG_UTILS_H

#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <gnuradio/linalg/api.h>
#include <gnuradio/linalg/types.h>
#include <numeric>
#include <pmt/pmt.h>
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

// Helpful functions for working with shapes
LINALG_API bool validate_shape(const types::shape &shape,
                               const std::string &name);

LINALG_API types::shape matrix_shape(types::shape shape);

LINALG_API int matrix_rows(types::shape shape);

LINALG_API int matrix_cols(types::shape shape);

/*!
 * \brief Computes the mean of elements in a vector.
 * \param vec The input vector.
 * \return The mean of the elements in the vector.
 * \throws std::invalid_argument if the vector is empty.
 */
// template <class T> T mean(const std::vector<T> &vec) {
//   if (vec.empty()) {
//     throw std::invalid_argument("Cannot compute mean of an empty vector");
//   }
//   return sum(vec) / static_cast<T>(vec.size());
// }

/*!
 * \brief Computes the size of a shape.
 * \param shape The input shape.
 * \return The size of the shape.
 * This function computes the total number of elements in a shape,
 * which is the product of its dimensions.
 * If the shape is empty, it returns 0.
 */
LINALG_API size_t compute_size(const types::shape &shape);

/*!
 * \brief Computes the sizes of multiple shapes.
 * \param shapes The input vector of shapes.
 * \return A vector containing the sizes of each shape.
 * This function computes the total number of elements for each shape
 * in the input vector.
 */
LINALG_API std::vector<size_t>
compute_sizes(const types::vector_shapes &shapes);

LINALG_API types::shape
compute_broadcast_shape(const types::shape &shape_a,
                        const types::shape &shape_b,
                        const int /*broadcast_type*/ = 0);

LINALG_API types::vector_shapes shapes_from_shapes(const types::shape &shape);

LINALG_API types::vector_shapes shapes_from_shapes(const types::shape &shape_0,
                                                   const types::shape &shape_1);

LINALG_API types::vector_shapes shapes_from_shapes(const types::shape &shape_0,
                                                   const types::shape &shape_1,
                                                   const types::shape &shape_2);

LINALG_API types::vector_shapes shapes_from_shapes(const types::shape &shape_0,
                                                   const types::shape &shape_1,
                                                   const types::shape &shape_2,
                                                   const types::shape &shape_3);

LINALG_API types::shape shapes_from_shapes(const types::shape &shape_0,
                                           const types::shape &shape_1,
                                           const types::shape &shape_2,
                                           const types::shape &shape_3,
                                           const types::shape &shape_4);

LINALG_API types::shape compute_broadcast(const types::shape &shape_0,
                                          const types::shape &shape_1,
                                          const int n_dims);

LINALG_API types::shape compute_broadcast(const types::shape &shape_0,
                                          const types::shape &shape_1,
                                          const types::shape &shape_2,
                                          const int n_dims);

LINALG_API types::shape compute_broadcast(const types::shape &shape_0,
                                          const types::shape &shape_1,
                                          const types::shape &shape_2,
                                          const types::shape &shape_3,
                                          const int n_dims);

/*!
 * \brief Computes the broadcast shape for a vector of shapes.
 * \param input_shapes The input vector of shapes.
 * \param n_dims The number of dimensions that is not consider for broadcasting.
 * \return A vector of shapes representing the broadcasted output shapes.
 */
LINALG_API types::shape
compute_broadcast(const types::vector_shapes &input_shapes, const int n_dims);

/*!
 * \brief Computes the mean of elements in a vector.
 * \param vec The input vector.
 * \return The mean of the elements in the vector.
 * \throws std::invalid_argument if the vector is empty.
 */
template <class T>
T mean(const std::vector<T> &vec);

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
std::vector<T> zeros(size_t size);

/*!
 * \brief Generates a vector of ones.
 * \param size The size of the vector.
 * \return A vector filled with ones.
 */
template <class T>
std::vector<T> ones(size_t size);

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
T variance(const std::vector<T> &vec);

/*!
 * \brief Computes the standard deviation of elements in a vector.
 * \param vec The input vector.
 * \return The standard deviation of the elements in the vector.
 * \throws std::invalid_argument if the vector is empty.
 */
// template <class T> T std_dev(const std::vector<T> &vec) {
//   return std::sqrt(variance(vec));
// }

template <class Scalar>
std::vector<Scalar> pmt_to_std_vector(const pmt::pmt_t &vec) {
  if (!pmt::is_vector(vec)) {
    throw std::invalid_argument("Input PMT is not a vector");
  }
  size_t size = pmt::length(vec);
  std::vector<Scalar> result(size);
  for (size_t i = 0; i < size; ++i) {
    auto element = pmt::vector_ref(vec, i);
    if constexpr (std::is_same<Scalar, double>::value) {
      result[i] = pmt::to_double(element);
    } else if constexpr (std::is_same<Scalar, float>::value) {
      result[i] = static_cast<float>(pmt::to_double(element));
    } else if constexpr (std::is_same<Scalar, int>::value) {
      result[i] = static_cast<int>(pmt::to_long(element));
    } else if constexpr (std::is_same<Scalar, long>::value) {
      result[i] = pmt::to_long(element);
    } else if constexpr (std::is_same<Scalar, std::complex<float>>::value) {
      result[i] = std::complex<float>(
          static_cast<float>(pmt::to_double(pmt::car(element))),
          static_cast<float>(pmt::to_double(pmt::cdr(element))));
    } else if constexpr (std::is_same<Scalar, std::complex<double>>::value) {
      result[i] = std::complex<double>(pmt::to_double(pmt::car(element)),
                                       pmt::to_double(pmt::cdr(element)));
    } else {
      static_assert(sizeof(Scalar) == 0,
                    "Unsupported type for pmt_to_std_vector");
    }
  }
  return result;
}

/*!
 * \brief Converts a vector to PMT format.
 * \param vec The input vector.
 * \return A PMT vector containing the elements of the input vector.
 */
template <class Scalar>
pmt::pmt_t std_vector_to_pmt(const std::vector<Scalar> &vec) {
  pmt::pmt_t result = pmt::make_vector(vec.size());
  for (size_t i = 0; i < vec.size(); ++i) {
    pmt::vector_set(result, i, pmt::from_double(vec[i]));
  }
  return result;
}

} // namespace utils
} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_UTILS_H */
