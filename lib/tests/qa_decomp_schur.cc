/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <Eigen/Dense>
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <complex>
#include <gnuradio/attributes.h>
#include <gnuradio/linalg/decomp_schur.h>
#include <gnuradio/linalg/types.h>

namespace gr {
namespace linalg {

BOOST_AUTO_TEST_CASE(test_real_schur_decomposition) {
  // Test real Schur decomposition with a 2x2 matrix
  types::shape input_shape = {2, 2};
  auto schur_block = decomp_schur_sync_f::make(input_shape, true);

  // Create input matrix A = [[1, 2], [3, 4]]
  Eigen::Matrix<float, 2, 2> input_matrix;
  input_matrix << 1.0f, 2.0f, 3.0f, 4.0f;

  // Prepare input/output containers
  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      input_map(input_matrix.data(), 2, 2);
  inputs.push_back(&input_map);

  Eigen::Matrix<float, 2, 2> Q_output, T_output;
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> Q_map(
      Q_output.data(), 2, 2);
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> T_map(
      T_output.data(), 2, 2);
  outputs.push_back(&Q_map);
  outputs.push_back(&T_map);

  // Perform decomposition
  auto result = schur_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Verify A = Q * T * Q^T
  Eigen::Matrix<float, 2, 2> reconstructed =
      Q_output * T_output * Q_output.transpose();

  float tolerance = 1e-4f;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      BOOST_CHECK_CLOSE(reconstructed(i, j), input_matrix(i, j), tolerance);
    }
  }

  // Verify Q is orthogonal (Q^T * Q = I)
  Eigen::Matrix<float, 2, 2> identity_check = Q_output.transpose() * Q_output;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      float expected = (i == j) ? 1.0f : 0.0f;
      BOOST_CHECK_SMALL(identity_check(i, j) - expected, tolerance);
    }
  }

  // Verify T is quasi-upper triangular (for real Schur form)
  // T(1,0) should be small (not necessarily zero for 2x2 case with complex
  // eigenvalues)
  BOOST_CHECK(std::abs(T_output(1, 0)) <= std::abs(T_output(0, 1)) + tolerance);
}

BOOST_AUTO_TEST_CASE(test_complex_schur_decomposition) {
  // Test complex Schur decomposition
  types::shape input_shape = {2, 2};
  auto schur_block = decomp_schur_sync_cf::make(input_shape, true);

  // Create complex input matrix
  Eigen::Matrix<std::complex<float>, 2, 2> input_matrix;
  input_matrix << std::complex<float>(1.0f, 0.5f),
      std::complex<float>(2.0f, 0.0f), std::complex<float>(0.0f, 1.0f),
      std::complex<float>(3.0f, -0.5f);

  // Prepare input/output containers
  types::vector_const_matrix_map<std::complex<float>> inputs;
  types::vector_matrix_map<std::complex<float>> outputs;

  Eigen::Map<
      const Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic>>
      input_map(input_matrix.data(), 2, 2);
  inputs.push_back(&input_map);

  Eigen::Matrix<std::complex<float>, 2, 2> Q_output, T_output;
  Eigen::Map<Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic>>
      Q_map(Q_output.data(), 2, 2);
  Eigen::Map<Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic>>
      T_map(T_output.data(), 2, 2);
  outputs.push_back(&Q_map);
  outputs.push_back(&T_map);

  // Perform decomposition
  auto result = schur_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Verify A = Q * T * Q^H (conjugate transpose)
  Eigen::Matrix<std::complex<float>, 2, 2> reconstructed =
      Q_output * T_output * Q_output.adjoint();

  float tolerance = 1e-4f;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      BOOST_CHECK_SMALL(std::abs(reconstructed(i, j) - input_matrix(i, j)),
                        tolerance);
    }
  }

  // Verify T is upper triangular for complex Schur
  BOOST_CHECK_SMALL(std::abs(T_output(1, 0)), tolerance);
}

BOOST_AUTO_TEST_CASE(test_schur_without_u_matrix) {
  // Test Schur decomposition returning only T matrix
  types::shape input_shape = {2, 2};
  auto schur_block =
      decomp_schur_sync_f::make(input_shape, false); // compute_u = false

  // Create input matrix
  Eigen::Matrix<float, 2, 2> input_matrix;
  input_matrix << 2.0f, 1.0f, 0.0f, 3.0f; // Upper triangular matrix

  // Prepare input/output containers
  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      input_map(input_matrix.data(), 2, 2);
  inputs.push_back(&input_map);

  Eigen::Matrix<float, 2, 2> T_output;
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> T_map(
      T_output.data(), 2, 2);
  outputs.push_back(&T_map); // Only T matrix output

  // Perform decomposition
  auto result = schur_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // For an already upper triangular matrix, T should be very similar to input
  float tolerance = 1e-4f;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      if (i <= j) { // Upper triangular part
        BOOST_CHECK_SMALL(
            std::abs(T_output(i, j)) - std::abs(input_matrix(i, j)), tolerance);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(test_schur_error_handling) {
  // Test error handling for invalid input shapes
  types::shape input_shape = {2, 2};
  auto schur_block = decomp_schur_sync_f::make(input_shape, true);

  // Test with wrong input size
  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  // Empty inputs should fail
  auto result = schur_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::INVALID_SHAPE);

  // Wrong number of outputs should fail
  Eigen::Matrix<float, 2, 2> input_matrix =
      Eigen::Matrix<float, 2, 2>::Identity();
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      input_map(input_matrix.data(), 2, 2);
  inputs.push_back(&input_map);

  // Should expect 2 outputs for compute_u=true, but provide only 1
  Eigen::Matrix<float, 2, 2> output_matrix;
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> output_map(
      output_matrix.data(), 2, 2);
  outputs.push_back(&output_map);

  result = schur_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::INVALID_SHAPE);
}

} /* namespace linalg */
} /* namespace gr */
