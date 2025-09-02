/* -*- c++ -*- */
/*
 * Copyright 2025 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <complex>
#include <gnuradio/attributes.h>
#include <gnuradio/linalg/matrix_trig.h>
#include <iostream>

namespace gr {
namespace linalg {

BOOST_AUTO_TEST_SUITE(qa_matrix_cos)

// Helper function to create test matrices
template <typename Scalar>
std::vector<Scalar> create_zero_matrix_2x2() {
  // Zero matrix: [[0, 0], [0, 0]]
  // Column-major: [0, 0, 0, 0]
  return {Scalar(0), Scalar(0), Scalar(0), Scalar(0)};
}

template <typename Scalar>
std::vector<Scalar> create_small_matrix_2x2() {
  // Small matrix: [[0.1, 0], [0, 0.2]]
  // Column-major: [0.1, 0, 0, 0.2]
  return {Scalar(0.1), Scalar(0), Scalar(0), Scalar(0.2)};
}

template <typename Scalar>
std::vector<Scalar> create_pi_matrix_2x2() {
  // Matrix with pi values: [[pi/2, 0], [0, pi]]
  // Column-major: [pi/2, 0, 0, pi]
  const auto pi = static_cast<Scalar>(M_PI);
  return {pi / Scalar(2), Scalar(0), Scalar(0), pi};
}

// Test matrix cosine of zero matrix (should be identity)
BOOST_AUTO_TEST_CASE(test_matrix_cos_zero_float) {
  auto cos_block =
      matrix_cos_sync_f::make({{2, 2}}, TrigAlgorithm::AUTO, 50, 1e-6);
  BOOST_REQUIRE(cos_block);

  auto zero_matrix = create_zero_matrix_2x2<float>();
  std::vector<float> output(4, 0.0f);

  // Simulate processing
  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  Eigen::Map<const Eigen::MatrixXf> input_map(zero_matrix.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXf> output_map(output.data(), 2, 2);

  inputs.push_back(&input_map);
  outputs.push_back(&output_map);

  auto result = cos_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // cos(0) should be I (identity matrix)
  BOOST_CHECK_CLOSE(output[0], 1.0f, 1e-5f); // [0,0]
  BOOST_CHECK_SMALL(output[1], 1e-6f);       // [1,0]
  BOOST_CHECK_SMALL(output[2], 1e-6f);       // [0,1]
  BOOST_CHECK_CLOSE(output[3], 1.0f, 1e-5f); // [1,1]
}

// Test matrix cosine of diagonal matrix with known values
BOOST_AUTO_TEST_CASE(test_matrix_cos_diagonal_float) {
  auto cos_block =
      matrix_cos_sync_f::make({{2, 2}}, TrigAlgorithm::AUTO, 50, 1e-6);
  BOOST_REQUIRE(cos_block);

  auto pi_matrix = create_pi_matrix_2x2<float>();
  std::vector<float> output(4, 0.0f);

  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  Eigen::Map<const Eigen::MatrixXf> input_map(pi_matrix.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXf> output_map(output.data(), 2, 2);

  inputs.push_back(&input_map);
  outputs.push_back(&output_map);

  auto result = cos_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // For diagonal matrix [[pi/2, 0], [0, pi]], cos should be [[0, 0], [0, -1]]
  // Column-major: [0, 0, 0, -1]
  BOOST_CHECK_SMALL(output[0], 1e-4f);        // cos(pi/2) ≈ 0
  BOOST_CHECK_SMALL(output[1], 1e-5f);        // 0
  BOOST_CHECK_SMALL(output[2], 1e-5f);        // 0
  BOOST_CHECK_CLOSE(output[3], -1.0f, 1e-4f); // cos(pi) = -1
}

// Test Taylor series algorithm specifically
BOOST_AUTO_TEST_CASE(test_matrix_cos_taylor_series_float) {
  auto cos_block =
      matrix_cos_sync_f::make({{2, 2}}, TrigAlgorithm::TAYLOR_SERIES, 50, 1e-6);
  BOOST_REQUIRE(cos_block);

  auto small_matrix = create_small_matrix_2x2<float>();
  std::vector<float> output(4, 0.0f);

  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  Eigen::Map<const Eigen::MatrixXf> input_map(small_matrix.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXf> output_map(output.data(), 2, 2);

  inputs.push_back(&input_map);
  outputs.push_back(&output_map);

  auto result = cos_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // For small diagonal matrix, cos should be close to standard cosine values
  // cos(0.1) ≈ 0.9950, cos(0.2) ≈ 0.9801
  BOOST_CHECK_CLOSE(output[0], std::cos(0.1f), 1e-3f);
  BOOST_CHECK_CLOSE(output[3], std::cos(0.2f), 1e-3f);
  BOOST_CHECK_SMALL(output[1], 1e-6f);
  BOOST_CHECK_SMALL(output[2], 1e-6f);
}

// Test eigenvalue algorithm
BOOST_AUTO_TEST_CASE(test_matrix_cos_eigenvalue_algorithm_float) {
  auto cos_block =
      matrix_cos_sync_f::make({{2, 2}}, TrigAlgorithm::EIGENVALUE, 50, 1e-6);
  BOOST_REQUIRE(cos_block);

  auto small_matrix = create_small_matrix_2x2<float>();
  std::vector<float> output(4, 0.0f);

  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  Eigen::Map<const Eigen::MatrixXf> input_map(small_matrix.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXf> output_map(output.data(), 2, 2);

  inputs.push_back(&input_map);
  outputs.push_back(&output_map);

  auto result = cos_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Should give same result as Taylor series for diagonal case
  BOOST_CHECK_CLOSE(output[0], std::cos(0.1f), 1e-3f);
  BOOST_CHECK_CLOSE(output[3], std::cos(0.2f), 1e-3f);
}

// Test trigonometric identity: sin²(A) + cos²(A) = I (for diagonal matrices)
BOOST_AUTO_TEST_CASE(test_matrix_cos_sin_identity_float) {
  auto cos_block =
      matrix_cos_sync_f::make({{2, 2}}, TrigAlgorithm::AUTO, 50, 1e-6);
  auto sin_block =
      matrix_sin_sync_f::make({{2, 2}}, TrigAlgorithm::AUTO, 50, 1e-6);
  BOOST_REQUIRE(cos_block);
  BOOST_REQUIRE(sin_block);

  auto small_matrix = create_small_matrix_2x2<float>();
  std::vector<float> cos_output(4, 0.0f);
  std::vector<float> sin_output(4, 0.0f);

  // Compute cos(A)
  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> cos_outputs, sin_outputs;

  Eigen::Map<const Eigen::MatrixXf> input_map(small_matrix.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXf> cos_output_map(cos_output.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXf> sin_output_map(sin_output.data(), 2, 2);

  inputs.push_back(&input_map);
  cos_outputs.push_back(&cos_output_map);
  sin_outputs.push_back(&sin_output_map);

  auto cos_result = cos_block->operation(inputs, cos_outputs);
  auto sin_result = sin_block->operation(inputs, sin_outputs);

  BOOST_CHECK_EQUAL(cos_result, OperationReturn::SUCCESS);
  BOOST_CHECK_EQUAL(sin_result, OperationReturn::SUCCESS);

  // Compute sin²(A) + cos²(A) using Eigen
  Eigen::MatrixXf cos_mat = cos_output_map;
  Eigen::MatrixXf sin_mat = sin_output_map;
  Eigen::MatrixXf identity_check =
      sin_mat.cwiseProduct(sin_mat) + cos_mat.cwiseProduct(cos_mat);

  // For diagonal matrices, this should be identity
  float tolerance = 1e-3f;
  BOOST_CHECK_CLOSE(identity_check(0, 0), 1.0f, tolerance);
  BOOST_CHECK_CLOSE(identity_check(1, 1), 1.0f, tolerance);
  BOOST_CHECK_SMALL(identity_check(0, 1), 1e-6f);
  BOOST_CHECK_SMALL(identity_check(1, 0), 1e-6f);
}

// Test complex matrix cosine
BOOST_AUTO_TEST_CASE(test_matrix_cos_complex_float) {
  auto cos_block =
      matrix_cos_sync_cf::make({{2, 2}}, TrigAlgorithm::AUTO, 50, 1e-6);
  BOOST_REQUIRE(cos_block);

  // Create complex matrix with imaginary components
  std::vector<std::complex<float>> input = {
      std::complex<float>(0.1f, 0.1f), // (0.1+0.1i)
      std::complex<float>(0, 0),       // 0
      std::complex<float>(0, 0),       // 0
      std::complex<float>(0.2f, -0.1f) // (0.2-0.1i)
  };

  std::vector<std::complex<float>> output(4, std::complex<float>(0, 0));

  types::vector_const_matrix_map<std::complex<float>> inputs;
  types::vector_matrix_map<std::complex<float>> outputs;

  Eigen::Map<const Eigen::MatrixXcf> input_map(input.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXcf> output_map(output.data(), 2, 2);

  inputs.push_back(&input_map);
  outputs.push_back(&output_map);

  auto result = cos_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // For diagonal complex matrix, verify with standard complex cos
  auto expected_cos_1 = std::cos(std::complex<float>(0.1f, 0.1f));
  auto expected_cos_2 = std::cos(std::complex<float>(0.2f, -0.1f));

  BOOST_CHECK_CLOSE(output[0].real(), expected_cos_1.real(), 1e-2f);
  BOOST_CHECK_CLOSE(output[0].imag(), expected_cos_1.imag(), 1e-2f);
  BOOST_CHECK_CLOSE(output[3].real(), expected_cos_2.real(), 1e-2f);
  BOOST_CHECK_CLOSE(output[3].imag(), expected_cos_2.imag(), 1e-2f);
}

// Test double precision
BOOST_AUTO_TEST_CASE(test_matrix_cos_double_precision) {
  auto cos_block =
      matrix_cos_sync_d::make({{2, 2}}, TrigAlgorithm::AUTO, 50, 1e-10);
  BOOST_REQUIRE(cos_block);

  auto small_matrix = create_small_matrix_2x2<double>();
  std::vector<double> output(4, 0.0);

  types::vector_const_matrix_map<double> inputs;
  types::vector_matrix_map<double> outputs;

  Eigen::Map<const Eigen::MatrixXd> input_map(small_matrix.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXd> output_map(output.data(), 2, 2);

  inputs.push_back(&input_map);
  outputs.push_back(&output_map);

  auto result = cos_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Higher precision verification
  BOOST_CHECK_CLOSE(output[0], std::cos(0.1), 1e-8);
  BOOST_CHECK_CLOSE(output[3], std::cos(0.2), 1e-8);
}

// Test error handling for non-square matrix
BOOST_AUTO_TEST_CASE(test_matrix_cos_error_handling_non_square) {
  BOOST_CHECK_THROW(
      matrix_cos_sync_f::make({{2, 3}}, TrigAlgorithm::AUTO, 50, 1e-6),
      std::invalid_argument);
}

// Test 3x3 matrix cosine
BOOST_AUTO_TEST_CASE(test_matrix_cos_3x3_float) {
  auto cos_block =
      matrix_cos_sync_f::make({{3, 3}}, TrigAlgorithm::AUTO, 50, 1e-6);
  BOOST_REQUIRE(cos_block);

  // Create 3x3 diagonal matrix with small values
  std::vector<float> input_3x3 = {
      0.1f, 0.0f, 0.0f, // First column
      0.0f, 0.2f, 0.0f, // Second column
      0.0f, 0.0f, 0.3f  // Third column
  };
  std::vector<float> output(9, 0.0f);

  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  Eigen::Map<const Eigen::MatrixXf> input_map(input_3x3.data(), 3, 3);
  Eigen::Map<Eigen::MatrixXf> output_map(output.data(), 3, 3);

  inputs.push_back(&input_map);
  outputs.push_back(&output_map);

  auto result = cos_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Check diagonal elements
  BOOST_CHECK_CLOSE(output[0], std::cos(0.1f), 1e-4f); // [0,0]
  BOOST_CHECK_CLOSE(output[4], std::cos(0.2f), 1e-4f); // [1,1]
  BOOST_CHECK_CLOSE(output[8], std::cos(0.3f), 1e-4f); // [2,2]

  // Check off-diagonal elements are zero
  for (int i : {1, 2, 3, 5, 6, 7}) {
    BOOST_CHECK_SMALL(output[i], 1e-6f);
  }
}

BOOST_AUTO_TEST_SUITE_END()

} // namespace linalg
} // namespace gr