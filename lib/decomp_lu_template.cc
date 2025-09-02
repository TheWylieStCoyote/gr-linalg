/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <complex>
#include <gnuradio/io_signature.h>
#include <gnuradio/linalg/decomp_lu.h>

namespace gr {
namespace linalg {

// Algorithm implementations
template <typename Scalar, int Rows, int Cols>
void eigen_partial_pivlu<Scalar, Rows, Cols>::decompose(
    const Eigen::Matrix<Scalar, Rows, Cols> &input,
    Eigen::Matrix<Scalar, Rows, Cols> &L, Eigen::Matrix<Scalar, Rows, Cols> &U,
    Eigen::Matrix<Scalar, Rows, Cols> &P) {
  Eigen::PartialPivLU<Eigen::Matrix<Scalar, Rows, Cols>> solver(input);
  L = solver.matrixLU().template triangularView<Eigen::Lower>();
  U = solver.matrixLU().template triangularView<Eigen::Upper>();
  P = solver.permutationP().toDenseMatrix().template cast<Scalar>();

  // Set diagonal of L to 1
  for (int i = 0; i < L.rows() && i < L.cols(); ++i) {
    L(i, i) = Scalar(1.0);
  }
}

template <typename Scalar, int Rows, int Cols>
void eigen_full_pivlu<Scalar, Rows, Cols>::decompose(
    const Eigen::Matrix<Scalar, Rows, Cols> &input,
    Eigen::Matrix<Scalar, Rows, Cols> &L, Eigen::Matrix<Scalar, Rows, Cols> &U,
    Eigen::Matrix<Scalar, Rows, Cols> &P) {
  Eigen::FullPivLU<Eigen::Matrix<Scalar, Rows, Cols>> solver(input);
  // Full pivot LU is more complex to extract, using reconstructed approach
  auto LU = solver.matrixLU();
  L = LU.template triangularView<Eigen::Lower>();
  U = LU.template triangularView<Eigen::Upper>();
  P = solver.permutationP().toDenseMatrix().template cast<Scalar>();

  // Set diagonal of L to 1
  for (int i = 0; i < L.rows() && i < L.cols(); ++i) {
    L(i, i) = Scalar(1.0);
  }
}

// decomp_lu_sync implementations
template <typename Scalar, int Rows, int Cols, class LUAlgorithm,
          class LUBackend, class LUBackendOptions>
decomp_lu_sync<Scalar, Rows, Cols, LUAlgorithm, LUBackend,
               LUBackendOptions>::decomp_lu_sync(const types::shape &shape)
    : linalg_base_sync<Scalar>("decomp_lu_sync", {shape}, {shape}),
      algorithm_(std::make_shared<eigen_partial_pivlu<Scalar>>()) {

  // Validate square matrix
  if (shape.size() != 2 || shape[0] != shape[1]) {
    throw std::invalid_argument("LU decomposition requires square matrices");
  }
}

template <typename Scalar, int Rows, int Cols, class LUAlgorithm,
          class LUBackend, class LUBackendOptions>
OperationReturn
decomp_lu_sync<Scalar, Rows, Cols, LUAlgorithm, LUBackend, LUBackendOptions>::
    operation(types::vector_const_matrix_map<Scalar> &input_matrices,
              types::vector_matrix_map<Scalar> &output_matrices) {

  if (input_matrices.empty() || output_matrices.size() < 3) {
    return OperationReturn::FAILURE; // Need at least 3 outputs: L, U, P
  }

  try {
    // Work directly with mapped matrices to avoid unnecessary copies
    const auto &input_matrix = *input_matrices[0];
    auto &L_output = *output_matrices[0]; // L matrix
    auto &U_output = *output_matrices[1]; // U matrix
    auto &P_output = *output_matrices[2]; // P matrix

    // Create temporary matrices for algorithm interface
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> temp_input =
        input_matrix;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> temp_L, temp_U,
        temp_P;

    algorithm_->decompose(temp_input, temp_L, temp_U, temp_P);

    L_output = temp_L;
    U_output = temp_U;
    P_output = temp_P;

    return OperationReturn::SUCCESS; // Successfully processed one matrix
  } catch (const std::exception &e) {
    // Log error and return failure
    return OperationReturn::FAILURE;
  }
}

template <typename Scalar, int Rows, int Cols, class LUAlgorithm,
          class LUBackend, class LUBackendOptions>
void decomp_lu_sync<Scalar, Rows, Cols, LUAlgorithm, LUBackend,
                    LUBackendOptions>::
    set_algorithm(std::shared_ptr<lu_algorithm<Scalar>> algo) {
  if (!algo) {
    throw std::invalid_argument("Algorithm cannot be null");
  }
  algorithm_ = algo;
}

// decomp_lu_pdu implementations
template <typename Scalar, int Rows, int Cols, class LUAlgorithm,
          class LUBackend, class LUBackendOptions>
decomp_lu_pdu<Scalar, Rows, Cols, LUAlgorithm, LUBackend,
              LUBackendOptions>::decomp_lu_pdu()
    : linalg_base_pdu<Scalar>("decomp_lu_pdu", {}, {}),
      algorithm_(std::make_shared<eigen_partial_pivlu<Scalar>>()) {}

template <typename Scalar, int Rows, int Cols, class LUAlgorithm,
          class LUBackend, class LUBackendOptions>
OperationReturn
decomp_lu_pdu<Scalar, Rows, Cols, LUAlgorithm, LUBackend, LUBackendOptions>::
    operation(types::vector_const_matrix_map<Scalar> &input_matrices,
              types::vector_matrix_map<Scalar> &output_matrices) {

  if (input_matrices.empty() || output_matrices.size() < 3) {
    return OperationReturn::FAILURE; // Need at least 3 outputs: L, U, P
  }

  try {
    // Work directly with mapped matrices to avoid unnecessary copies
    const auto &input_matrix = *input_matrices[0];
    auto &L_output = *output_matrices[0]; // L matrix
    auto &U_output = *output_matrices[1]; // U matrix
    auto &P_output = *output_matrices[2]; // P matrix

    // Create temporary matrices for algorithm interface
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> temp_input =
        input_matrix;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> temp_L, temp_U,
        temp_P;

    algorithm_->decompose(temp_input, temp_L, temp_U, temp_P);

    L_output = temp_L;
    U_output = temp_U;
    P_output = temp_P;

    return OperationReturn::SUCCESS;
  } catch (const std::exception &e) {
    // Log error and return failure
    return OperationReturn::FAILURE;
  }
}

template <typename Scalar, int Rows, int Cols, class LUAlgorithm,
          class LUBackend, class LUBackendOptions>
void decomp_lu_pdu<Scalar, Rows, Cols, LUAlgorithm, LUBackend,
                   LUBackendOptions>::
    set_algorithm(std::shared_ptr<lu_algorithm<Scalar>> algo) {
  if (!algo) {
    throw std::invalid_argument("Algorithm cannot be null");
  }
  algorithm_ = algo;
}

// Explicit template instantiations for common types
// Algorithm instantiations
template class eigen_partial_pivlu<float>;
template class eigen_partial_pivlu<double>;
template class eigen_partial_pivlu<std::complex<float>>;
template class eigen_partial_pivlu<std::complex<double>>;

template class eigen_full_pivlu<float>;
template class eigen_full_pivlu<double>;
template class eigen_full_pivlu<std::complex<float>>;
template class eigen_full_pivlu<std::complex<double>>;

// Sync block instantiations
template class decomp_lu_sync<float>;
template class decomp_lu_sync<double>;
template class decomp_lu_sync<std::complex<float>>;
template class decomp_lu_sync<std::complex<double>>;

// PDU block instantiations
template class decomp_lu_pdu<float>;
template class decomp_lu_pdu<double>;
template class decomp_lu_pdu<std::complex<float>>;
template class decomp_lu_pdu<std::complex<double>>;

} // namespace linalg
} // namespace gr
