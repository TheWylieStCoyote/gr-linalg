/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "matrix_eigenvalues_impl.h"
#include <algorithm>
#include <gnuradio/io_signature.h>
#include <gnuradio/linalg/performance_profiler.h>
#include <gnuradio/linalg/types.h>
#include <stdexcept>

namespace gr {
namespace linalg {

using ERROR_TAG = error_tag_t;
using ERROR_PDU = error_pdu_p;

// ============================================================================
// matrix_eigenvalues base class implementation
// ============================================================================

template <typename Scalar>
matrix_eigenvalues<Scalar>::matrix_eigenvalues(
    const std::string &name, const types::vector_shapes &shape_inputs,
    const types::vector_shapes &shape_outputs, bool sort_by_magnitude)
    : linalg_base<Scalar>(name, shape_inputs, shape_outputs,
                          array_broadcast_type::NONE),
      d_sort_by_magnitude(sort_by_magnitude) {}

template <typename Scalar>
matrix_eigenvalues<Scalar>::~matrix_eigenvalues() {}

template <typename Scalar>
void matrix_eigenvalues<Scalar>::set_sort_by_magnitude(bool sort) {
  d_sort_by_magnitude = sort;
}

template <typename Scalar>
bool matrix_eigenvalues<Scalar>::get_sort_by_magnitude() const {
  return d_sort_by_magnitude;
}

template <typename Scalar>
OperationReturn matrix_eigenvalues<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {

  PROFILE_LINALG_OPERATION("matrix_eigenvalues", input_matrices[0]->size());

  if (input_matrices.size() != 1) {
    return OperationReturn::FAILURE;
  }
  if (output_matrices.size() != 1) {
    return OperationReturn::FAILURE;
  }

  const auto &input_matrix = *input_matrices[0];
  auto &output_matrix = *output_matrices[0];

  // Validate square matrix
  if (input_matrix.rows() != input_matrix.cols()) {
    return OperationReturn::INVALID_SHAPE;
  }

  // Validate output dimensions
  if (output_matrix.rows() != input_matrix.rows() ||
      output_matrix.cols() != 1) {
    return OperationReturn::INVALID_SHAPE;
  }

  try {
    // Map input matrix
    Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
        eigen_input(input_matrix.data(), input_matrix.rows(),
                    input_matrix.cols());

    // Create eigenvalue output vector
    // For all matrices (real and complex), eigenvalues are complex in the
    // general case
    if constexpr (std::is_same_v<Scalar, float>) {
      using ComplexEigenvalueType = std::complex<float>;
      Eigen::Map<Eigen::Matrix<ComplexEigenvalueType, Eigen::Dynamic, 1>>
          eigenvalues_vector(
              reinterpret_cast<ComplexEigenvalueType *>(output_matrix.data()),
              output_matrix.rows());
      compute_eigenvalues_impl<ComplexEigenvalueType>(eigen_input,
                                                      eigenvalues_vector);
    } else if constexpr (std::is_same_v<Scalar, double>) {
      using ComplexEigenvalueType = std::complex<double>;
      Eigen::Map<Eigen::Matrix<ComplexEigenvalueType, Eigen::Dynamic, 1>>
          eigenvalues_vector(
              reinterpret_cast<ComplexEigenvalueType *>(output_matrix.data()),
              output_matrix.rows());
      compute_eigenvalues_impl<ComplexEigenvalueType>(eigen_input,
                                                      eigenvalues_vector);
    } else {
      // For complex matrices, eigenvalues are already complex
      Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> eigenvalues_vector(
          output_matrix.data(), output_matrix.rows());
      compute_eigenvalues_impl<Scalar>(eigen_input, eigenvalues_vector);
    }

    return OperationReturn::SUCCESS;
  } catch (const std::exception &) {
    return OperationReturn::FAILURE;
  }
}

template <typename Scalar>
template <typename EigenvalueType>
void matrix_eigenvalues<Scalar>::compute_eigenvalues_impl(
    const Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic,
                                         Eigen::Dynamic>> &input_matrix,
    Eigen::Map<Eigen::Matrix<EigenvalueType, Eigen::Dynamic, 1>>
        &eigenvalues_vector) const {

  // Use appropriate solver based on scalar type
  if constexpr (std::is_same_v<Scalar, std::complex<float>> ||
                std::is_same_v<Scalar, std::complex<double>>) {
    // For complex matrices, use ComplexEigenSolver
    Eigen::ComplexEigenSolver<
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
        solver;
    solver.compute(input_matrix);

    if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Complex eigenvalue computation failed");
    }

    // Get eigenvalues (already complex)
    auto eigenvalues = solver.eigenvalues();

    // Copy eigenvalues to output
    for (Eigen::Index i = 0; i < eigenvalues.size(); ++i) {
      eigenvalues_vector(i) = static_cast<EigenvalueType>(eigenvalues(i));
    }

    // Sort by magnitude if requested
    if (d_sort_by_magnitude) {
      // Create vector of indices for sorting
      std::vector<Eigen::Index> indices(eigenvalues.size());
      std::iota(indices.begin(), indices.end(), 0);

      // Sort indices by eigenvalue magnitude (largest first)
      std::sort(indices.begin(), indices.end(),
                [&eigenvalues_vector](Eigen::Index i, Eigen::Index j) {
                  return std::abs(eigenvalues_vector(i)) >
                         std::abs(eigenvalues_vector(j));
                });

      // Reorder eigenvalues
      Eigen::Matrix<EigenvalueType, Eigen::Dynamic, 1> sorted_eigenvalues(
          eigenvalues.size());
      for (size_t k = 0; k < indices.size(); ++k) {
        sorted_eigenvalues(k) = eigenvalues_vector(indices[k]);
      }
      eigenvalues_vector = sorted_eigenvalues;
    }
  } else {
    // For real matrices, use EigenSolver (eigenvalues may be complex)
    Eigen::EigenSolver<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
        solver;
    solver.compute(input_matrix);

    if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Real eigenvalue computation failed");
    }

    // Get eigenvalues
    auto eigenvalues = solver.eigenvalues();

    // Copy eigenvalues to output
    for (Eigen::Index i = 0; i < eigenvalues.size(); ++i) {
      eigenvalues_vector(i) = static_cast<EigenvalueType>(eigenvalues(i));
    }

    // Sort by magnitude if requested
    if (d_sort_by_magnitude) {
      // Create vector of indices for sorting
      std::vector<Eigen::Index> indices(eigenvalues.size());
      std::iota(indices.begin(), indices.end(), 0);

      // Sort indices by eigenvalue magnitude (largest first)
      std::sort(indices.begin(), indices.end(),
                [&eigenvalues_vector](Eigen::Index i, Eigen::Index j) {
                  return std::abs(eigenvalues_vector(i)) >
                         std::abs(eigenvalues_vector(j));
                });

      // Reorder eigenvalues
      Eigen::Matrix<EigenvalueType, Eigen::Dynamic, 1> sorted_eigenvalues(
          eigenvalues.size());
      for (size_t k = 0; k < indices.size(); ++k) {
        sorted_eigenvalues(k) = eigenvalues_vector(indices[k]);
      }
      eigenvalues_vector = sorted_eigenvalues;
    }
  }
}

template <typename Scalar>
types::vector_shapes matrix_eigenvalues<Scalar>::compute_output_shapes(
    const types::vector_shapes &input_shapes) {
  if (input_shapes.size() != 1) {
    throw std::invalid_argument(
        "Matrix eigenvalues requires exactly 1 input matrix");
  }

  const auto &matrix_shape = input_shapes[0];
  if (matrix_shape.size() != 2) {
    throw std::invalid_argument("Input must be a 2D matrix");
  }

  if (matrix_shape[0] != matrix_shape[1]) {
    throw std::invalid_argument("Input matrix must be square");
  }

  // Output is a column vector with same number of rows as input matrix
  return {{matrix_shape[0], 1}};
}

template <typename Scalar>
void matrix_eigenvalues<Scalar>::validate_shape(
    const types::vector_shapes &input_shapes,
    const types::vector_shapes &output_shapes, const std::string &name) {

  if (input_shapes.size() != 1) {
    throw std::invalid_argument(name + " requires exactly 1 input matrix");
  }

  const auto &matrix_shape = input_shapes[0];
  if (matrix_shape.size() != 2) {
    throw std::invalid_argument(name + " input must be a 2D matrix");
  }

  if (matrix_shape[0] != matrix_shape[1]) {
    throw std::invalid_argument(name + " input matrix must be square");
  }

  if (!output_shapes.empty()) {
    if (output_shapes.size() != 1) {
      throw std::invalid_argument(name + " must have exactly 1 output");
    }

    const auto &output_shape = output_shapes[0];
    if (output_shape.size() != 2 || output_shape[1] != 1) {
      throw std::invalid_argument(name + " output must be a column vector");
    }

    if (output_shape[0] != matrix_shape[0]) {
      throw std::invalid_argument(name + " output dimension mismatch");
    }
  }
}

template <typename Scalar>
std::vector<size_t>
matrix_eigenvalues<Scalar>::compute_sizes(const types::vector_shapes &shapes) {
  std::vector<size_t> sizes;
  sizes.reserve(shapes.size());

  for (const auto &shape : shapes) {
    size_t size = 1;
    for (size_t dim : shape) {
      size *= dim;
    }
    sizes.push_back(size);
  }

  return sizes;
}

// ============================================================================
// matrix_eigenvalues_sync implementation
// ============================================================================

template <typename Scalar>
matrix_eigenvalues_sync<Scalar>::matrix_eigenvalues_sync(
    const types::shape &shape, bool sort_by_magnitude)
    : linalg_base<Scalar>(
          "matrix_eigenvalues", {shape},
          matrix_eigenvalues<Scalar>::compute_output_shapes({shape}),
          array_broadcast_type::NONE),
      linalg_base_sync<Scalar>(
          "matrix_eigenvalues", {shape}, std::vector<std::string>{"input"},
          matrix_eigenvalues<Scalar>::compute_output_shapes({shape}),
          std::vector<std::string>{"eigenvalues"}, array_broadcast_type::NONE,
          error_tag_t::NONE, error_pdu_p::NONE, gr::block::TPP_ALL_TO_ALL),
      matrix_eigenvalues<Scalar>(
          "matrix_eigenvalues", {shape},
          matrix_eigenvalues<Scalar>::compute_output_shapes({shape}),
          sort_by_magnitude) {}

template <typename Scalar>
typename matrix_eigenvalues_sync<Scalar>::sptr
matrix_eigenvalues_sync<Scalar>::make(const types::shape &shape,
                                      bool sort_by_magnitude) {
  return gnuradio::make_block_sptr<matrix_eigenvalues_sync_impl<Scalar>>(
      shape, sort_by_magnitude);
}

// ============================================================================
// matrix_eigenvalues_pdu implementation
// ============================================================================

template <typename Scalar>
matrix_eigenvalues_pdu<Scalar>::matrix_eigenvalues_pdu(
    const types::shape &shape, bool sort_by_magnitude)
    : linalg_base<Scalar>(
          "matrix_eigenvalues", {shape},
          matrix_eigenvalues<Scalar>::compute_output_shapes({shape}),
          array_broadcast_type::NONE),
      linalg_base_pdu<Scalar>(
          "matrix_eigenvalues", {shape}, std::vector<std::string>{"input"},
          matrix_eigenvalues<Scalar>::compute_output_shapes({shape}),
          std::vector<std::string>{"eigenvalues"}, array_broadcast_type::NONE,
          error_tag_t::NONE, error_pdu_p::NONE),
      matrix_eigenvalues<Scalar>(
          "matrix_eigenvalues", {shape},
          matrix_eigenvalues<Scalar>::compute_output_shapes({shape}),
          sort_by_magnitude) {}

template <typename Scalar>
typename matrix_eigenvalues_pdu<Scalar>::sptr
matrix_eigenvalues_pdu<Scalar>::make(const types::shape &shape,
                                     bool sort_by_magnitude) {
  return gnuradio::make_block_sptr<matrix_eigenvalues_pdu_impl<Scalar>>(
      shape, sort_by_magnitude);
}

// ============================================================================
// matrix_eigenvalues_sync_impl implementation
// ============================================================================

template <typename Scalar>
matrix_eigenvalues_sync_impl<Scalar>::matrix_eigenvalues_sync_impl(
    const types::shape &shape, bool sort_by_magnitude)
    : linalg_base<Scalar>(
          "matrix_eigenvalues", {shape},
          matrix_eigenvalues<Scalar>::compute_output_shapes({shape}),
          array_broadcast_type::NONE),
      matrix_eigenvalues<Scalar>(
          "matrix_eigenvalues", {shape},
          matrix_eigenvalues<Scalar>::compute_output_shapes({shape}),
          sort_by_magnitude),
      linalg_base_sync<Scalar>(
          "matrix_eigenvalues", {shape}, std::vector<std::string>{"input"},
          matrix_eigenvalues<Scalar>::compute_output_shapes({shape}),
          std::vector<std::string>{"eigenvalues"}, array_broadcast_type::NONE,
          error_tag_t::NONE, error_pdu_p::NONE, gr::block::TPP_ALL_TO_ALL),
      matrix_eigenvalues_sync<Scalar>(shape, sort_by_magnitude) {}

template <typename Scalar>
matrix_eigenvalues_sync_impl<Scalar>::~matrix_eigenvalues_sync_impl() {}

// ============================================================================
// matrix_eigenvalues_pdu_impl implementation
// ============================================================================

template <typename Scalar>
matrix_eigenvalues_pdu_impl<Scalar>::matrix_eigenvalues_pdu_impl(
    const types::shape &shape, bool sort_by_magnitude)
    : linalg_base<Scalar>(
          "matrix_eigenvalues", {shape},
          matrix_eigenvalues<Scalar>::compute_output_shapes({shape}),
          array_broadcast_type::NONE),
      matrix_eigenvalues<Scalar>(
          "matrix_eigenvalues", {shape},
          matrix_eigenvalues<Scalar>::compute_output_shapes({shape}),
          sort_by_magnitude),
      linalg_base_pdu<Scalar>(
          "matrix_eigenvalues", {shape}, std::vector<std::string>{"input"},
          matrix_eigenvalues<Scalar>::compute_output_shapes({shape}),
          std::vector<std::string>{"eigenvalues"}, array_broadcast_type::NONE,
          error_tag_t::NONE, error_pdu_p::NONE),
      matrix_eigenvalues_pdu<Scalar>(shape, sort_by_magnitude) {}

template <typename Scalar>
matrix_eigenvalues_pdu_impl<Scalar>::~matrix_eigenvalues_pdu_impl() {}

// ============================================================================
// Explicit template instantiations
// ============================================================================

template class matrix_eigenvalues<float>;
template class matrix_eigenvalues<double>;
template class matrix_eigenvalues<gr_complex>;
template class matrix_eigenvalues<std::complex<double>>;

template class matrix_eigenvalues_sync<float>;
template class matrix_eigenvalues_sync<double>;
template class matrix_eigenvalues_sync<gr_complex>;
template class matrix_eigenvalues_sync<std::complex<double>>;

template class matrix_eigenvalues_pdu<float>;
template class matrix_eigenvalues_pdu<double>;
template class matrix_eigenvalues_pdu<gr_complex>;
template class matrix_eigenvalues_pdu<std::complex<double>>;

template class matrix_eigenvalues_sync_impl<float>;
template class matrix_eigenvalues_sync_impl<double>;
template class matrix_eigenvalues_sync_impl<gr_complex>;
template class matrix_eigenvalues_sync_impl<std::complex<double>>;

template class matrix_eigenvalues_pdu_impl<float>;
template class matrix_eigenvalues_pdu_impl<double>;
template class matrix_eigenvalues_pdu_impl<gr_complex>;
template class matrix_eigenvalues_pdu_impl<std::complex<double>>;

} /* namespace linalg */
} /* namespace gr */