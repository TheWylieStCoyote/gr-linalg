/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "matrix_diag_impl.h"
#include <gnuradio/gr_complex.h>
#include <gnuradio/io_signature.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/types.h>
#include <gnuradio/linalg/utils.h>
#include <pmt/pmt.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace gr {
namespace linalg {

// Base class implementation
template <typename Scalar>
matrix_diag<Scalar>::matrix_diag(types::shape &shape, int k)
    : linalg_base<Scalar>("matrix_diag", {shape},
                          compute_output_shapes_with_k({shape}, k),
                          array_broadcast_type::CUSTOM),
      d_k(k) {
  if (shape.size() < 2) {
    throw std::invalid_argument("matrix_diag requires a 2D input matrix");
  }
  const int rows = utils::matrix_rows(shape);
  const int cols = utils::matrix_cols(shape);
  // Check that the rows and columns are within limits
  if (rows <= 0 || cols <= 0) {
    throw std::invalid_argument("matrix_diag requires positive dimensions");
  }
  // Check that the diagonal is within the size of the matrix
  if (k < -rows + 1 || k > cols - 1) {
    throw std::invalid_argument("k must be in the range [-rows + 1, cols - 1]");
  }
}
template <typename Scalar>
matrix_diag<Scalar>::~matrix_diag() {}

template <typename Scalar>
void matrix_diag<Scalar>::set_k(int k) {
  const auto &input_shape = this->shape_inputs[0];
  const int rows = utils::matrix_rows(input_shape);
  const int cols = utils::matrix_cols(input_shape);
  if (k < -rows + 1 || k > cols - 1) {
    throw std::invalid_argument("k must be in the range [-rows + 1, cols - 1]");
  }
  d_k = k;
}

template <typename Scalar>
int matrix_diag<Scalar>::get_k() const {
  return d_k;
}

template <typename Scalar>
OperationReturn matrix_diag<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {
  if (input_matrices.empty() || output_matrices.empty()) {
    return OperationReturn::INVALID_SHAPE;
  }

  const auto &input_matrix = *input_matrices[0];
  auto &output_vector = *output_matrices[0];

  // Extract diagonal elements based on offset k
  int rows = static_cast<int>(input_matrix.rows());
  int cols = static_cast<int>(input_matrix.cols());
  int diag_size = std::min(rows, cols);

  // Adjust diagonal size based on offset k
  if (d_k > 0) {
    // Super-diagonal: reduce size if k goes beyond matrix bounds
    diag_size = std::min(diag_size, cols - d_k);
  } else if (d_k < 0) {
    // Sub-diagonal: reduce size if k goes beyond matrix bounds
    diag_size = std::min(diag_size, rows + d_k);
  }

  if (diag_size <= 0) {
    return OperationReturn::INVALID_SHAPE;
  }

  // Extract diagonal elements
  for (int i = 0; i < diag_size; ++i) {
    int row_idx = (d_k < 0) ? i - d_k : i;
    int col_idx = (d_k > 0) ? i + d_k : i;

    if (row_idx >= 0 && row_idx < rows && col_idx >= 0 && col_idx < cols) {
      output_vector(i, 0) = input_matrix(row_idx, col_idx);
    }
  }

  return OperationReturn::SUCCESS;
}

template <typename Scalar>
types::vector_shapes matrix_diag<Scalar>::compute_output_shapes(
    const types::vector_shapes &input_shapes) {
  if (input_shapes.empty()) {
    throw std::invalid_argument(
        "matrix_diag requires at least one input shape");
  }

  const auto &input_shape = input_shapes[0];
  if (input_shape.size() != 2) {
    throw std::invalid_argument("matrix_diag requires 2D input matrix");
  }

  // Extract diagonal: output is a vector with min(rows, cols) elements
  // Note: This computes for k=0 (main diagonal). For non-zero k, the actual
  // length will be computed dynamically in the operation() method.
  int rows = input_shape[0];
  int cols = input_shape[1];
  int diag_length = std::min(rows, cols);

  return {{diag_length, 1}}; // Column vector
}

// Helper method to compute output shapes with k parameter
template <typename Scalar>
types::vector_shapes matrix_diag<Scalar>::compute_output_shapes_with_k(
    const types::vector_shapes &input_shapes, int k) {
  if (input_shapes.empty()) {
    throw std::invalid_argument(
        "matrix_diag requires at least one input shape");
  }

  const auto &input_shape = input_shapes[0];
  if (input_shape.size() != 2) {
    throw std::invalid_argument("matrix_diag requires 2D input matrix");
  }

  int rows = input_shape[0];
  int cols = input_shape[1];
  int diag_length = std::min(rows, cols);

  // Adjust diagonal size based on offset k
  if (k > 0) {
    // Super-diagonal: reduce size if k goes beyond matrix bounds
    diag_length = std::min(diag_length, cols - k);
  } else if (k < 0) {
    // Sub-diagonal: reduce size if k goes beyond matrix bounds
    diag_length = std::min(diag_length, rows + k);
  }

  if (diag_length <= 0) {
    diag_length = 1; // Minimum size to avoid zero-sized outputs
  }

  return {{diag_length, 1}}; // Column vector
}

// Sync implementation
template <typename Scalar>
matrix_diag_sync_impl<Scalar>::matrix_diag_sync_impl(types::shape &shape, int k)
    : linalg_base<Scalar>(
          "matrix_diag_sync", {shape}, std::vector<std::string>{"input"},
          matrix_diag<Scalar>::compute_output_shapes_with_k({shape}, k),
          std::vector<std::string>{"output"}, array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE),
      linalg_base_sync<Scalar>(
          "matrix_diag_sync", {shape}, std::vector<std::string>{"input"},
          matrix_diag<Scalar>::compute_output_shapes_with_k({shape}, k),
          std::vector<std::string>{"output"}, array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE, gr::block::TPP_ALL_TO_ALL),
      matrix_diag<Scalar>(shape, k), matrix_diag_sync<Scalar>(shape, k) {}

template <typename Scalar>
matrix_diag_sync_impl<Scalar>::~matrix_diag_sync_impl() {}

// matrix_diag_sync public ctor used by factory
template <typename Scalar>
matrix_diag_sync<Scalar>::matrix_diag_sync(types::shape &shape, int k)
    : matrix_diag<Scalar>(shape, k) {}

// Factory methods
template <typename Scalar>
typename matrix_diag_sync<Scalar>::sptr
matrix_diag_sync<Scalar>::make(types::shape &shape, int k) {
  return gnuradio::make_block_sptr<matrix_diag_sync_impl<Scalar>>(shape, k);
}

template <typename Scalar>
typename matrix_diag_pdu<Scalar>::sptr
matrix_diag_pdu<Scalar>::make(types::shape &shape, int k) {
  return std::static_pointer_cast<matrix_diag_pdu<Scalar>>(
      std::make_shared<matrix_diag_pdu_impl<Scalar>>(shape, k));
}

// matrix_diag_pdu public ctor used by factory
template <typename Scalar>
matrix_diag_pdu<Scalar>::matrix_diag_pdu(types::shape &shape, int k)
    : linalg_base<Scalar>(
          "matrix_diag_pdu", {shape}, std::vector<std::string>{"input"},
          matrix_diag<Scalar>::compute_output_shapes_with_k({shape}, k),
          std::vector<std::string>{"output"}, array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE),
      linalg_base_pdu<Scalar>(
          "matrix_diag_pdu", {shape}, std::vector<std::string>{"input"},
          matrix_diag<Scalar>::compute_output_shapes_with_k({shape}, k),
          std::vector<std::string>{"output"}, array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE, PDU_UPDATE::ANY_INPUT,
          MESSAGE_HANDLER_MODE::DEFAULT),
      matrix_diag<Scalar>(shape, k) {}

// PDU implementation
template <typename Scalar>
matrix_diag_pdu_impl<Scalar>::matrix_diag_pdu_impl(types::shape &shape, int k)
    : linalg_base<Scalar>(
          "matrix_diag_pdu", {shape}, std::vector<std::string>{"input"},
          matrix_diag<Scalar>::compute_output_shapes_with_k({shape}, k),
          std::vector<std::string>{"output"}, array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE),
      linalg_base_pdu<Scalar>(
          "matrix_diag_pdu", {shape}, std::vector<std::string>{"input"},
          matrix_diag<Scalar>::compute_output_shapes_with_k({shape}, k),
          std::vector<std::string>{"output"}, array_broadcast_type::CUSTOM,
          error_tag_t::NONE, error_pdu_p::NONE, PDU_UPDATE::ANY_INPUT,
          MESSAGE_HANDLER_MODE::DEFAULT),
      matrix_diag<Scalar>(shape, k), matrix_diag_pdu<Scalar>(shape, k) {}

template <typename Scalar>
matrix_diag_pdu_impl<Scalar>::~matrix_diag_pdu_impl() {}

// Explicit template instantiations
template class matrix_diag<float>;
template class matrix_diag<double>;
template class matrix_diag<gr_complex>;
template class matrix_diag<std::complex<double>>;

template class matrix_diag_sync_impl<float>;
template class matrix_diag_sync_impl<double>;
template class matrix_diag_sync_impl<gr_complex>;
template class matrix_diag_sync_impl<std::complex<double>>;

template class matrix_diag_pdu_impl<float>;
template class matrix_diag_pdu_impl<double>;
template class matrix_diag_pdu_impl<gr_complex>;
template class matrix_diag_pdu_impl<std::complex<double>>;

// Explicit instantiations for factory make functions
template typename matrix_diag_sync<float>::sptr
matrix_diag_sync<float>::make(types::shape &, int);
template typename matrix_diag_sync<double>::sptr
matrix_diag_sync<double>::make(types::shape &, int);
template typename matrix_diag_sync<gr_complex>::sptr
matrix_diag_sync<gr_complex>::make(types::shape &, int);
template typename matrix_diag_sync<std::complex<double>>::sptr
matrix_diag_sync<std::complex<double>>::make(types::shape &, int);

template typename matrix_diag_pdu<float>::sptr
matrix_diag_pdu<float>::make(types::shape &, int);
template typename matrix_diag_pdu<double>::sptr
matrix_diag_pdu<double>::make(types::shape &, int);
template typename matrix_diag_pdu<gr_complex>::sptr
matrix_diag_pdu<gr_complex>::make(types::shape &, int);
template typename matrix_diag_pdu<std::complex<double>>::sptr
matrix_diag_pdu<std::complex<double>>::make(types::shape &, int);

// Explicit instantiations for public constructors
template matrix_diag_sync<float>::matrix_diag_sync(types::shape &, int);
template matrix_diag_sync<double>::matrix_diag_sync(types::shape &, int);
template matrix_diag_sync<gr_complex>::matrix_diag_sync(types::shape &, int);
template matrix_diag_sync<std::complex<double>>::matrix_diag_sync(
    types::shape &, int);

template matrix_diag_pdu<float>::matrix_diag_pdu(types::shape &, int);
template matrix_diag_pdu<double>::matrix_diag_pdu(types::shape &, int);
template matrix_diag_pdu<gr_complex>::matrix_diag_pdu(types::shape &, int);
template matrix_diag_pdu<std::complex<double>>::matrix_diag_pdu(types::shape &,
                                                                int);

} /* namespace linalg */
} /* namespace gr */