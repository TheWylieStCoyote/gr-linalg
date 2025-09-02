/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_TYPES_H
#define INCLUDED_LINALG_TYPES_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <complex>
#include <cstddef> // for size_t
#include <functional>
#include <gnuradio/linalg/api.h>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace gr {
namespace linalg {
namespace types {

/*!
 * \brief A type alias for a dimension, which is an integer.
 *
 * This type is used to represent the size of each dimension in a shape.
 */
typedef int dim;

/*!
 * \brief A type alias for a shape, which is a vector of dimensions.
 *
 * This type is used to represent the dimensions of matrices or tensors in
 * linear algebra operations. For tensors:
 * - 1D: [n] (vector of length n)
 * - 2D: [m, n] (matrix of size m×n)
 * - 3D: [k, m, n] (tensor of size k×m×n)
 * - Higher dimensions supported for tensor operations
 */
typedef std::vector<dim> shape;

template <class T>
inline size_t compute_size(const shape &shape) {
  if (shape.empty()) {
    return 0;
  }
  return sizeof(T) *
         std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<dim>());
}

inline size_t compute_size(const shape &shape) {
  if (shape.empty()) {
    return 0;
  }
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<dim>());
}

/*!
 * \brief A type alias for a shared pointer to a shape.
 */
typedef std::shared_ptr<shape> sptr_shape;

/*!
 * \brief A type alias for a constant shared pointer to a shape.
 */
typedef std::shared_ptr<const shape> shape_csptr;

/*!
 * \brief A type alias for a vector of shapes.
 *
 * This type is used to represent multiple shapes, such as input and output
 * shapes in linear algebra operations.
 */
typedef std::vector<shape> vector_shapes;

/*!
 * \brief Calculate sizes for multiple shapes (returns vector for consistency
 * with some APIs)
 * \param shapes The vector of shapes to compute sizes for
 * \return A vector containing the sizes of each shape in bytes.
 * This function computes the total number of elements for each shape
 * in the input vector.
 * If the shape is empty, it returns 0.
 * This function is useful for determining the memory requirements for
 * storing arrays or matrices of a given shape.
 */
template <class T>
std::vector<size_t> compute_sizes(const vector_shapes &shapes);

typedef std::vector<void *> linalg_vector_void_star;
typedef std::vector<const void *> linalg_vector_const_void_star;

/*!
 * \brief Template alias for fixed-size matrix map
 */
template <typename Scalar, int Rows, int Cols>
using matrix_map_fixed = Eigen::Map<const Eigen::Matrix<Scalar, Rows, Cols>>;

/*!
 * \brief Template alias for dynamic matrix map
 */
template <typename Scalar, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic>
using matrix_map_dyn = Eigen::Map<const Eigen::Matrix<Scalar, Rows, Cols>>;

/*!
 * \brief Template alias for shared pointer to fixed-size matrix map
 */
template <typename Scalar, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic>
using matrix_map_ptr = std::shared_ptr<matrix_map_fixed<Scalar, Rows, Cols>>;

/*!
 * \brief Template alias for vector of fixed-size matrix map pointers
 */
template <typename Scalar, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic>
using vector_matrix_map_fixed =
    std::vector<matrix_map_fixed<Scalar, Rows, Cols> *>;

/*!
 * \brief A type alias for a matrix of floats.
 */
typedef Eigen::MatrixXf matrix_float;

/*!
 * \brief A type alias for a matrix of doubles.
 */
typedef Eigen::MatrixXd matrix_double;

/*!
 * \brief A type alias for a matrix of integers.
 */
typedef Eigen::MatrixXi matrix_int;

/*!
 * \brief A type alias for a matrix of complex floats.
 */
typedef Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic>
    matrix_complex_float;

/*!
 * \brief A type alias for a matrix of complex doubles.
 */
typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>
    matrix_complex_double;

typedef matrix_complex_double complex_double_matrix;

/*!
 * \brief A type alias for a map of a matrix of floats.
 */
typedef Eigen::Map<Eigen::MatrixXf> matrix_map_float;

/*!
 * \brief A type alias for a map of a matrix of doubles.
 */
typedef Eigen::Map<Eigen::MatrixXd> matrix_map_double;

/*!
 * \brief A type alias for a map of a matrix of integers.
 */
typedef Eigen::Map<Eigen::MatrixXi> matrix_map_int;

/*!
 * \brief A type alias for a map of a matrix of complex floats.
 */
typedef Eigen::Map<
    Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic>>
    matrix_map_complex_float;

/*!
 * \brief A type alias for a map of a matrix of complex doubles.
 */
typedef Eigen::Map<
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>>
    matrix_map_complex_double;

/*!
 * \brief Template alias for a dynamic matrix map of any type
 */
template <class T>
using matrix_map_dynamic =
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;

/*!
 * \brief Template alias for a const dynamic matrix map of any type
 */
template <class T>
using const_matrix_map_dynamic =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;

/*!
 * \brief Template alias for a vector of matrix map pointers
 */
// Removed duplicate definition

/*!
 * \brief Template alias for a vector of vector matrix map pointers
 */
// template <class T>
// using vector_vector_matrix_map = std::vector<vector_matrix_map<T> *>;

template <class T>
using vector_const_matrix_map = std::vector<const_matrix_map_dynamic<T> *>;

template <class T>
using vector_matrix_map = std::vector<matrix_map_dynamic<T> *>;

/*!
 * \brief Template alias for tensor map (higher-dimensional arrays)
 */
template <class T>
using tensor_map = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;

template <class T>
using const_tensor_map =
    Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;

template <class T>
using vector_tensor_map = std::vector<tensor_map<T> *>;

template <class T>
using vector_const_tensor_map = std::vector<const_tensor_map<T> *>;

/*!
 * \brief Tensor broadcasting and shape utilities
 */
namespace tensor_utils {

/*!
 * \brief Check if two shapes are compatible for broadcasting
 * \param shape1 First shape to compare
 * \param shape2 Second shape to compare
 * \return true if shapes are broadcast compatible
 */
bool is_broadcast_compatible(const shape &shape1, const shape &shape2);

/*!
 * \brief Compute the broadcast shape of two shapes
 * \param shape1 First input shape
 * \param shape2 Second input shape
 * \return Broadcast result shape
 */
shape broadcast_shapes(const shape &shape1, const shape &shape2);

/*!
 * \brief Get the number of elements in a tensor shape
 * \param s The shape to compute size for
 * \return Number of elements
 */
inline size_t total_elements(const shape &s) {
  if (s.empty())
    return 0;
  return std::accumulate(s.begin(), s.end(), 1, std::multiplies<dim>());
}

/*!
 * \brief Check if shape represents a vector (1D)
 * \param s Shape to check
 * \return true if 1D vector
 */
inline bool is_vector(const shape &s) { return s.size() == 1; }

/*!
 * \brief Check if shape represents a matrix (2D)
 * \param s Shape to check
 * \return true if 2D matrix
 */
inline bool is_matrix(const shape &s) { return s.size() == 2; }

/*!
 * \brief Check if shape represents a tensor (3D or higher)
 * \param s Shape to check
 * \return true if 3D or higher tensor
 */
inline bool is_tensor(const shape &s) { return s.size() > 2; }

/*!
 * \brief Reshape a flattened array to tensor shape
 * \param flat_size Size of flattened array
 * \param target_shape Desired tensor shape
 * \return true if reshape is possible
 */
inline bool can_reshape(size_t flat_size, const shape &target_shape) {
  return flat_size == total_elements(target_shape);
}

/*!
 * \brief Convert multi-dimensional index to flattened index
 * \param indices Multi-dimensional indices
 * \param shape Shape of the tensor
 * \return Flattened index
 */
inline size_t ravel_index(const std::vector<dim> &indices, const shape &shape) {
  if (indices.size() != shape.size()) {
    throw std::invalid_argument("Index dimension mismatch");
  }

  size_t flat_index = 0;
  size_t stride = 1;

  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    flat_index += static_cast<size_t>(indices[static_cast<size_t>(i)]) * stride;
    stride *= static_cast<size_t>(shape[static_cast<size_t>(i)]);
  }

  return flat_index;
}

/*!
 * \brief Convert flattened index to multi-dimensional index
 * \param flat_index Flattened index
 * \param shape Shape of the tensor
 * \return Multi-dimensional indices
 */
inline std::vector<dim> unravel_index(size_t flat_index, const shape &shape) {
  std::vector<dim> indices(shape.size());

  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    indices[static_cast<size_t>(i)] = static_cast<dim>(
        flat_index % static_cast<size_t>(shape[static_cast<size_t>(i)]));
    flat_index /= static_cast<size_t>(shape[static_cast<size_t>(i)]);
  }

  return indices;
}

} // namespace tensor_utils

} // namespace types
} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_TYPES_H */
