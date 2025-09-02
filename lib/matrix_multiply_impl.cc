/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "matrix_multiply_impl.h"
#include <gnuradio/gr_complex.h>
#include <gnuradio/io_signature.h>
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/types.h>
#include <gnuradio/linalg/utils.h>
#include <gnuradio/sptr_magic.h>
#include <stdexcept>

using gr::linalg::array_broadcast_type;
using gr::linalg::error_pdu_p;
using gr::linalg::error_tag_t;

namespace gr {
namespace linalg {

// Static method implementations for matrix_multiply
template <typename Scalar, int A_Rows, int A_Cols, int B_Rows, int B_Cols>
void matrix_multiply<Scalar, A_Rows, A_Cols, B_Rows, B_Cols>::validate_shape(
    const types::vector_shapes &input_shapes,
    const types::vector_shapes &output_shapes, const std::string &name) {
  if (input_shapes.size() != 2) {
    throw std::invalid_argument(
        name + ": Matrix multiplication requires exactly 2 input matrices");
  }

  for (const auto &shape : input_shapes) {
    if (shape.size() != 2) {
      throw std::invalid_argument(
          name + ": Matrix multiplication requires 2D matrices");
    }
    for (int dim : shape) {
      if (dim <= 0) {
        throw std::invalid_argument(name +
                                    ": Matrix dimensions must be positive");
      }
    }
  }

  if (input_shapes[0][1] != input_shapes[1][0]) {
    throw std::invalid_argument(
        name + ": Matrix multiplication: columns of A must equal rows of B");
  }

  if (!output_shapes.empty()) {
    auto expected_output = compute_output_shapes(input_shapes);
    if (output_shapes != expected_output) {
      throw std::invalid_argument(
          name + ": Output shapes do not match expected shapes");
    }
  }
}

template <typename Scalar, int A_Rows, int A_Cols, int B_Rows, int B_Cols>
std::vector<size_t>
matrix_multiply<Scalar, A_Rows, A_Cols, B_Rows, B_Cols>::compute_sizes(
    const types::vector_shapes &shapes) {
  std::vector<size_t> sizes;
  for (const auto &shape : shapes) {
    size_t total_size = 1;
    for (int dim : shape) {
      total_size *= static_cast<size_t>(dim);
    }
    sizes.push_back(total_size);
  }
  return sizes;
}

template <typename Scalar, int A_Rows, int A_Cols, int B_Rows, int B_Cols>
types::vector_shapes
matrix_multiply<Scalar, A_Rows, A_Cols, B_Rows, B_Cols>::compute_output_shapes(
    const types::vector_shapes &input_shapes) {
  if (input_shapes.size() != 2) {
    throw std::invalid_argument(
        "Matrix multiplication requires exactly 2 input matrices");
  }

  const auto &shape_a = input_shapes[0];
  const auto &shape_b = input_shapes[1];

  if (shape_a.size() != 2 || shape_b.size() != 2) {
    throw std::invalid_argument("Matrix multiplication requires 2D matrices");
  }

  if (shape_a[1] != shape_b[0]) {
    throw std::invalid_argument(
        "Matrix multiplication: columns of A must equal rows of B");
  }

  // Output matrix has shape [rows_A, cols_B]
  types::vector_shapes output_shapes;
  output_shapes.push_back({shape_a[0], shape_b[1]});
  return output_shapes;
}

// #pragma message("set the following appropriately and remove this warning")
// using input_type = float;
// #pragma message("set the following appropriately and remove this warning")
// using output_type = float;

// matrix_multiply_sync_impl constructors
template <typename Scalar>
matrix_multiply_sync_impl<Scalar>::matrix_multiply_sync_impl(
    const types::shape &shape_0, const types::shape &shape_1)
    : linalg_base<Scalar>(
          "matrix_multiply_sync", {shape_0, shape_1},
          std::vector<std::string>{}, {{shape_0[0], shape_1[1]}},
          std::vector<std::string>{}, array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE),
      matrix_multiply<Scalar>("matrix_multiply_sync", {shape_0, shape_1},
                              {{shape_0[0], shape_1[1]}}),
      linalg_base_sync<Scalar>("matrix_multiply_sync", {shape_0, shape_1},
                               {{shape_0[0], shape_1[1]}}),
      matrix_multiply_sync<Scalar>(shape_0, shape_1) {}

// matrix_multiply_pdu_impl constructors
template <typename Scalar>
matrix_multiply_pdu_impl<Scalar>::matrix_multiply_pdu_impl(
    const types::shape &shape_0, const types::shape &shape_1)
    : matrix_multiply<Scalar>("matrix_multiply_pdu", {shape_0, shape_1},
                              {{shape_0[0], shape_1[1]}}),
      linalg_base_pdu<Scalar>(
          "matrix_multiply_pdu", {shape_0, shape_1}, // Input shapes
          {"in0", "in1"},                            // Input names
          {{shape_0[0], shape_1[1]}},                // Output shapes
          {"out"},                                   // Output names
          array_broadcast_type::CUSTOM,              // Skip strict validation
          error_tag_t::NONE, error_pdu_p::NONE),
      matrix_multiply_pdu<Scalar>(shape_0, shape_1) {}

// Base class operation implementation
template <typename Scalar, int A_Rows, int A_Cols, int B_Rows, int B_Cols>
OperationReturn
matrix_multiply<Scalar, A_Rows, A_Cols, B_Rows, B_Cols>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {
  // Validate input with detailed error messages for debugging
  if (input_matrices.size() != 2) {
    std::string msg =
        "Matrix multiplication requires exactly 2 input matrices, got " +
        std::to_string(input_matrices.size());
    throw std::invalid_argument(msg);
  }
  if (output_matrices.size() != 1) {
    std::string msg =
        "Matrix multiplication produces exactly 1 output matrix, got " +
        std::to_string(output_matrices.size());
    throw std::invalid_argument(msg);
  }

  // Get input matrices
  const auto &A = *input_matrices[0];
  const auto &B = *input_matrices[1];
  auto &C = *output_matrices[0];

  // Check dimensions
  if (A.cols() != B.rows()) {
    throw std::invalid_argument(
        "Matrix dimensions incompatible for multiplication");
  }
  if (C.rows() != A.rows() || C.cols() != B.cols()) {
    throw std::invalid_argument("Output matrix has incorrect dimensions");
  }

  // Perform multiplication: C = A * B
  C = A * B;

  return gr::linalg::OperationReturn::SUCCESS;
}

// operation implementation for sync variant
template <typename Scalar>
OperationReturn matrix_multiply_sync_impl<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {
  return matrix_multiply<Scalar>::operation(input_matrices, output_matrices);
}

// make() method implementations
template <class Scalar, int A_Rows, int A_Cols, int B_Rows, int B_Cols>
typename matrix_multiply_sync<Scalar, A_Rows, A_Cols, B_Rows, B_Cols>::sptr
matrix_multiply_sync<Scalar, A_Rows, A_Cols, B_Rows, B_Cols>::make(
    const types::shape &shape_0, const types::shape &shape_1) {
  return std::static_pointer_cast<
      matrix_multiply_sync<Scalar, A_Rows, A_Cols, B_Rows, B_Cols>>(
      gnuradio::make_block_sptr<matrix_multiply_sync_impl<Scalar>>(shape_0,
                                                                   shape_1));
}

template <class Scalar, int A_Rows, int A_Cols, int B_Rows, int B_Cols>
typename matrix_multiply_pdu<Scalar, A_Rows, A_Cols, B_Rows, B_Cols>::sptr
matrix_multiply_pdu<Scalar, A_Rows, A_Cols, B_Rows, B_Cols>::make(
    const types::shape &shape_0, const types::shape &shape_1) {
  return std::static_pointer_cast<
      matrix_multiply_pdu<Scalar, A_Rows, A_Cols, B_Rows, B_Cols>>(
      gnuradio::make_block_sptr<matrix_multiply_pdu_impl<Scalar>>(shape_0,
                                                                  shape_1));
}

// Explicit template instantiations for impl class constructors
template matrix_multiply_sync_impl<float>::matrix_multiply_sync_impl(
    const types::shape &, const types::shape &);
template matrix_multiply_sync_impl<double>::matrix_multiply_sync_impl(
    const types::shape &, const types::shape &);
template matrix_multiply_sync_impl<
    std::complex<float>>::matrix_multiply_sync_impl(const types::shape &,
                                                    const types::shape &);
template matrix_multiply_sync_impl<
    std::complex<double>>::matrix_multiply_sync_impl(const types::shape &,
                                                     const types::shape &);

template matrix_multiply_pdu_impl<float>::matrix_multiply_pdu_impl(
    const types::shape &, const types::shape &);
template matrix_multiply_pdu_impl<double>::matrix_multiply_pdu_impl(
    const types::shape &, const types::shape &);
template matrix_multiply_pdu_impl<
    std::complex<float>>::matrix_multiply_pdu_impl(const types::shape &,
                                                   const types::shape &);
template matrix_multiply_pdu_impl<
    std::complex<double>>::matrix_multiply_pdu_impl(const types::shape &,
                                                    const types::shape &);

// Explicit template instantiations for impl classes
template class matrix_multiply_sync_impl<float>;
template class matrix_multiply_sync_impl<double>;
template class matrix_multiply_sync_impl<std::complex<float>>;
template class matrix_multiply_sync_impl<std::complex<double>>;

template class matrix_multiply_pdu_impl<float>;
template class matrix_multiply_pdu_impl<double>;
template class matrix_multiply_pdu_impl<std::complex<float>>;
template class matrix_multiply_pdu_impl<std::complex<double>>;

// Explicit template instantiations for base classes with fixed dimensions (used
// by tests)
template class matrix_multiply<float, 2, 2, 2, 2>;
template class matrix_multiply<double, 2, 2, 2, 2>;
template class matrix_multiply<std::complex<float>, 2, 2, 2, 2>;
template class matrix_multiply<std::complex<double>, 2, 2, 2, 2>;

// Explicit template instantiations for static methods with full template
// parameters Using Eigen::Dynamic as default for all dimension parameters
template void
matrix_multiply<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic,
                Eigen::Dynamic>::validate_shape(const types::vector_shapes &,
                                                const types::vector_shapes &,
                                                const std::string &);
template void
matrix_multiply<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic,
                Eigen::Dynamic>::validate_shape(const types::vector_shapes &,
                                                const types::vector_shapes &,
                                                const std::string &);
template void
matrix_multiply<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic,
                Eigen::Dynamic,
                Eigen::Dynamic>::validate_shape(const types::vector_shapes &,
                                                const types::vector_shapes &,
                                                const std::string &);
template void
matrix_multiply<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic,
                Eigen::Dynamic,
                Eigen::Dynamic>::validate_shape(const types::vector_shapes &,
                                                const types::vector_shapes &,
                                                const std::string &);

template std::vector<size_t>
matrix_multiply<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic,
                Eigen::Dynamic>::compute_sizes(const types::vector_shapes &);
template std::vector<size_t>
matrix_multiply<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic,
                Eigen::Dynamic>::compute_sizes(const types::vector_shapes &);
template std::vector<size_t>
matrix_multiply<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic,
                Eigen::Dynamic,
                Eigen::Dynamic>::compute_sizes(const types::vector_shapes &);
template std::vector<size_t>
matrix_multiply<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic,
                Eigen::Dynamic,
                Eigen::Dynamic>::compute_sizes(const types::vector_shapes &);

template types::vector_shapes matrix_multiply<
    float, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic,
    Eigen::Dynamic>::compute_output_shapes(const types::vector_shapes &);
template types::vector_shapes matrix_multiply<
    double, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic,
    Eigen::Dynamic>::compute_output_shapes(const types::vector_shapes &);
template types::vector_shapes matrix_multiply<
    std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic,
    Eigen::Dynamic>::compute_output_shapes(const types::vector_shapes &);
template types::vector_shapes matrix_multiply<
    std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic,
    Eigen::Dynamic>::compute_output_shapes(const types::vector_shapes &);

// Explicit template instantiations for operation method - Dynamic dimensions
template OperationReturn
matrix_multiply<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic,
                Eigen::Dynamic>::operation(types::vector_const_matrix_map<float>
                                               &,
                                           types::vector_matrix_map<float> &);
template OperationReturn matrix_multiply<
    double, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic,
    Eigen::Dynamic>::operation(types::vector_const_matrix_map<double> &,
                               types::vector_matrix_map<double> &);
template OperationReturn
matrix_multiply<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic,
                Eigen::Dynamic, Eigen::Dynamic>::
    operation(types::vector_const_matrix_map<std::complex<float>> &,
              types::vector_matrix_map<std::complex<float>> &);
template OperationReturn
matrix_multiply<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic,
                Eigen::Dynamic, Eigen::Dynamic>::
    operation(types::vector_const_matrix_map<std::complex<double>> &,
              types::vector_matrix_map<std::complex<double>> &);

// Explicit template instantiations for make() methods
template typename matrix_multiply_sync<float, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::Dynamic, Eigen::Dynamic>::sptr
matrix_multiply_sync<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic,
                     Eigen::Dynamic>::make(const types::shape &,
                                           const types::shape &);
template typename matrix_multiply_sync<double, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::Dynamic, Eigen::Dynamic>::sptr
matrix_multiply_sync<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic,
                     Eigen::Dynamic>::make(const types::shape &,
                                           const types::shape &);
template typename matrix_multiply_sync<std::complex<float>, Eigen::Dynamic,
                                       Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::Dynamic>::sptr
matrix_multiply_sync<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic,
                     Eigen::Dynamic, Eigen::Dynamic>::make(const types::shape &,
                                                           const types::shape
                                                               &);
template typename matrix_multiply_sync<std::complex<double>, Eigen::Dynamic,
                                       Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::Dynamic>::sptr
matrix_multiply_sync<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic,
                     Eigen::Dynamic, Eigen::Dynamic>::make(const types::shape &,
                                                           const types::shape
                                                               &);

template typename matrix_multiply_pdu<float, Eigen::Dynamic, Eigen::Dynamic,
                                      Eigen::Dynamic, Eigen::Dynamic>::sptr
matrix_multiply_pdu<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic,
                    Eigen::Dynamic>::make(const types::shape &,
                                          const types::shape &);
template typename matrix_multiply_pdu<double, Eigen::Dynamic, Eigen::Dynamic,
                                      Eigen::Dynamic, Eigen::Dynamic>::sptr
matrix_multiply_pdu<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic,
                    Eigen::Dynamic>::make(const types::shape &,
                                          const types::shape &);
template typename matrix_multiply_pdu<std::complex<float>, Eigen::Dynamic,
                                      Eigen::Dynamic, Eigen::Dynamic,
                                      Eigen::Dynamic>::sptr
matrix_multiply_pdu<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic,
                    Eigen::Dynamic, Eigen::Dynamic>::make(const types::shape &,
                                                          const types::shape &);
template typename matrix_multiply_pdu<std::complex<double>, Eigen::Dynamic,
                                      Eigen::Dynamic, Eigen::Dynamic,
                                      Eigen::Dynamic>::sptr
matrix_multiply_pdu<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic,
                    Eigen::Dynamic, Eigen::Dynamic>::make(const types::shape &,
                                                          const types::shape &);

} /* namespace linalg */
} /* namespace gr */
