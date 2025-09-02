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

BOOST_AUTO_TEST_SUITE(qa_matrix_sin)

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

// Test matrix sine of zero matrix (should be zero)
BOOST_AUTO_TEST_CASE(test_matrix_sin_zero_float) {
  auto sin_block =
      matrix_sin_sync_f::make({{2, 2}}, TrigAlgorithm::AUTO, 50, 1e-6);
  BOOST_REQUIRE(sin_block);

  auto zero_matrix = create_zero_matrix_2x2<float>();
  std::vector<float> output(4, 0.0f);

  // Simulate processing
  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  Eigen::Map<const Eigen::MatrixXf> input_map(zero_matrix.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXf> output_map(output.data(), 2, 2);

  inputs.push_back(&input_map);
  outputs.push_back(&output_map);

  auto result = sin_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // sin(0) should be 0
  for (size_t i = 0; i < 4; ++i) {
    BOOST_CHECK_SMALL(output[i], 1e-6f);
  }
}

// Test matrix sine of diagonal matrix with known values
BOOST_AUTO_TEST_CASE(test_matrix_sin_diagonal_float) {
  auto sin_block =
      matrix_sin_sync_f::make({{2, 2}}, TrigAlgorithm::AUTO, 50, 1e-6);
  BOOST_REQUIRE(sin_block);

  auto pi_matrix = create_pi_matrix_2x2<float>();
  std::vector<float> output(4, 0.0f);

  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  Eigen::Map<const Eigen::MatrixXf> input_map(pi_matrix.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXf> output_map(output.data(), 2, 2);

  inputs.push_back(&input_map);
  outputs.push_back(&output_map);

  auto result = sin_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // For diagonal matrix [[pi/2, 0], [0, pi]], sin should be [[1, 0], [0, 0]]
  // Column-major: [1, 0, 0, 0]
  BOOST_CHECK_CLOSE(output[0], 1.0f, 1e-4f); // sin(pi/2) = 1
  BOOST_CHECK_SMALL(output[1], 1e-5f);       // 0
  BOOST_CHECK_SMALL(output[2], 1e-5f);       // 0
  BOOST_CHECK_SMALL(output[3], 1e-4f);       // sin(pi) ≈ 0
}

// Test Taylor series algorithm specifically
BOOST_AUTO_TEST_CASE(test_matrix_sin_taylor_series_float) {
  auto sin_block =
      matrix_sin_sync_f::make({{2, 2}}, TrigAlgorithm::TAYLOR_SERIES, 50, 1e-6);
  BOOST_REQUIRE(sin_block);

  auto small_matrix = create_small_matrix_2x2<float>();
  std::vector<float> output(4, 0.0f);

  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  Eigen::Map<const Eigen::MatrixXf> input_map(small_matrix.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXf> output_map(output.data(), 2, 2);

  inputs.push_back(&input_map);
  outputs.push_back(&output_map);

  auto result = sin_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // For small diagonal matrix, sin should be approximately the matrix itself
  // sin(0.1) ≈ 0.0998, sin(0.2) ≈ 0.1987
  BOOST_CHECK_CLOSE(output[0], std::sin(0.1f), 1e-3f);
  BOOST_CHECK_CLOSE(output[3], std::sin(0.2f), 1e-3f);
  BOOST_CHECK_SMALL(output[1], 1e-6f);
  BOOST_CHECK_SMALL(output[2], 1e-6f);
}

// Test eigenvalue algorithm
BOOST_AUTO_TEST_CASE(test_matrix_sin_eigenvalue_algorithm_float) {
  auto sin_block =
      matrix_sin_sync_f::make({{2, 2}}, TrigAlgorithm::EIGENVALUE, 50, 1e-6);
  BOOST_REQUIRE(sin_block);

  auto small_matrix = create_small_matrix_2x2<float>();
  std::vector<float> output(4, 0.0f);

  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  Eigen::Map<const Eigen::MatrixXf> input_map(small_matrix.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXf> output_map(output.data(), 2, 2);

  inputs.push_back(&input_map);
  outputs.push_back(&output_map);

  auto result = sin_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Should give same result as Taylor series for diagonal case
  BOOST_CHECK_CLOSE(output[0], std::sin(0.1f), 1e-3f);
  BOOST_CHECK_CLOSE(output[3], std::sin(0.2f), 1e-3f);
}

// Test complex matrix sine
BOOST_AUTO_TEST_CASE(test_matrix_sin_complex_float) {
  auto sin_block =
      matrix_sin_sync_cf::make({{2, 2}}, TrigAlgorithm::AUTO, 50, 1e-6);
  BOOST_REQUIRE(sin_block);

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

  auto result = sin_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // For diagonal complex matrix, verify with standard complex sin
  auto expected_sin_1 = std::sin(std::complex<float>(0.1f, 0.1f));
  auto expected_sin_2 = std::sin(std::complex<float>(0.2f, -0.1f));

  BOOST_CHECK_CLOSE(output[0].real(), expected_sin_1.real(), 1e-2f);
  BOOST_CHECK_CLOSE(output[0].imag(), expected_sin_1.imag(), 1e-2f);
  BOOST_CHECK_CLOSE(output[3].real(), expected_sin_2.real(), 1e-2f);
  BOOST_CHECK_CLOSE(output[3].imag(), expected_sin_2.imag(), 1e-2f);
}

// Test double precision
BOOST_AUTO_TEST_CASE(test_matrix_sin_double_precision) {
  auto sin_block =
      matrix_sin_sync_d::make({{2, 2}}, TrigAlgorithm::AUTO, 50, 1e-10);
  BOOST_REQUIRE(sin_block);

  auto small_matrix = create_small_matrix_2x2<double>();
  std::vector<double> output(4, 0.0);

  types::vector_const_matrix_map<double> inputs;
  types::vector_matrix_map<double> outputs;

  Eigen::Map<const Eigen::MatrixXd> input_map(small_matrix.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXd> output_map(output.data(), 2, 2);

  inputs.push_back(&input_map);
  outputs.push_back(&output_map);

  auto result = sin_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Higher precision verification
  BOOST_CHECK_CLOSE(output[0], std::sin(0.1), 1e-8);
  BOOST_CHECK_CLOSE(output[3], std::sin(0.2), 1e-8);
}

// Test error handling for non-square matrix
BOOST_AUTO_TEST_CASE(test_matrix_sin_error_handling_non_square) {
  BOOST_CHECK_THROW(
      matrix_sin_sync_f::make({{2, 3}}, TrigAlgorithm::AUTO, 50, 1e-6),
      std::invalid_argument);
}

// Test convergence properties
BOOST_AUTO_TEST_CASE(test_matrix_sin_convergence_float) {
  auto sin_block =
      matrix_sin_sync_f::make({{2, 2}}, TrigAlgorithm::TAYLOR_SERIES, 10, 1e-3);
  BOOST_REQUIRE(sin_block);

  auto small_matrix = create_small_matrix_2x2<float>();
  std::vector<float> output(4, 0.0f);

  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  Eigen::Map<const Eigen::MatrixXf> input_map(small_matrix.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXf> output_map(output.data(), 2, 2);

  inputs.push_back(&input_map);
  outputs.push_back(&output_map);

  auto result = sin_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Should still converge reasonably with fewer iterations for small matrices
  BOOST_CHECK_CLOSE(output[0], std::sin(0.1f), 1e-2f);
  BOOST_CHECK_CLOSE(output[3], std::sin(0.2f), 1e-2f);
}

// Test 3x3 matrix sine
BOOST_AUTO_TEST_CASE(test_matrix_sin_3x3_float) {
  auto sin_block =
      matrix_sin_sync_f::make({{3, 3}}, TrigAlgorithm::AUTO, 50, 1e-6);
  BOOST_REQUIRE(sin_block);

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

  auto result = sin_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Check diagonal elements
  BOOST_CHECK_CLOSE(output[0], std::sin(0.1f), 1e-4f); // [0,0]
  BOOST_CHECK_CLOSE(output[4], std::sin(0.2f), 1e-4f); // [1,1]
  BOOST_CHECK_CLOSE(output[8], std::sin(0.3f), 1e-4f); // [2,2]

  // Check off-diagonal elements are zero
  for (int i : {1, 2, 3, 5, 6, 7}) {
    BOOST_CHECK_SMALL(output[i], 1e-6f);
  }
}

BOOST_AUTO_TEST_SUITE_END()

} // namespace linalg
} // namespace gr