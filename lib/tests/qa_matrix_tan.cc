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

BOOST_AUTO_TEST_SUITE(qa_matrix_tan)

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
std::vector<Scalar> create_pi4_matrix_2x2() {
  // Matrix with pi/4 values: [[pi/4, 0], [0, pi/6]]
  // Column-major: [pi/4, 0, 0, pi/6]
  const auto pi = static_cast<Scalar>(M_PI);
  return {pi / Scalar(4), Scalar(0), Scalar(0), pi / Scalar(6)};
}

// Test matrix tangent of zero matrix (should be zero)
BOOST_AUTO_TEST_CASE(test_matrix_tan_zero_float) {
  auto tan_block =
      matrix_tan_sync_f::make({{2, 2}}, TrigAlgorithm::AUTO, 50, 1e-6);
  BOOST_REQUIRE(tan_block);

  auto zero_matrix = create_zero_matrix_2x2<float>();
  std::vector<float> output(4, 0.0f);

  // Simulate processing
  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  Eigen::Map<const Eigen::MatrixXf> input_map(zero_matrix.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXf> output_map(output.data(), 2, 2);

  inputs.push_back(&input_map);
  outputs.push_back(&output_map);

  auto result = tan_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // tan(0) should be 0
  for (size_t i = 0; i < 4; ++i) {
    BOOST_CHECK_SMALL(output[i], 1e-6f);
  }
}

// Test matrix tangent of diagonal matrix with known values
BOOST_AUTO_TEST_CASE(test_matrix_tan_diagonal_float) {
  auto tan_block =
      matrix_tan_sync_f::make({{2, 2}}, TrigAlgorithm::AUTO, 50, 1e-6);
  BOOST_REQUIRE(tan_block);

  auto pi4_matrix = create_pi4_matrix_2x2<float>();
  std::vector<float> output(4, 0.0f);

  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  Eigen::Map<const Eigen::MatrixXf> input_map(pi4_matrix.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXf> output_map(output.data(), 2, 2);

  inputs.push_back(&input_map);
  outputs.push_back(&output_map);

  auto result = tan_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // For diagonal matrix [[pi/4, 0], [0, pi/6]], tan should be [[1, 0], [0,
  // tan(pi/6)]] tan(pi/4) = 1, tan(pi/6) = 1/sqrt(3) ≈ 0.5774 Column-major: [1,
  // 0, 0, tan(pi/6)]
  BOOST_CHECK_CLOSE(output[0], 1.0f, 1e-4f);               // tan(pi/4) = 1
  BOOST_CHECK_SMALL(output[1], 1e-5f);                     // 0
  BOOST_CHECK_SMALL(output[2], 1e-5f);                     // 0
  BOOST_CHECK_CLOSE(output[3], std::tan(M_PI / 6), 1e-4f); // tan(pi/6)
}

// Test Taylor series algorithm specifically for small values
BOOST_AUTO_TEST_CASE(test_matrix_tan_taylor_series_float) {
  auto tan_block =
      matrix_tan_sync_f::make({{2, 2}}, TrigAlgorithm::TAYLOR_SERIES, 50, 1e-6);
  BOOST_REQUIRE(tan_block);

  auto small_matrix = create_small_matrix_2x2<float>();
  std::vector<float> output(4, 0.0f);

  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  Eigen::Map<const Eigen::MatrixXf> input_map(small_matrix.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXf> output_map(output.data(), 2, 2);

  inputs.push_back(&input_map);
  outputs.push_back(&output_map);

  auto result = tan_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // For small diagonal matrix, tan should be close to standard tangent values
  // tan(0.1) ≈ 0.1003, tan(0.2) ≈ 0.2027
  BOOST_CHECK_CLOSE(output[0], std::tan(0.1f), 1e-3f);
  BOOST_CHECK_CLOSE(output[3], std::tan(0.2f), 1e-3f);
  BOOST_CHECK_SMALL(output[1], 1e-6f);
  BOOST_CHECK_SMALL(output[2], 1e-6f);
}

// Test eigenvalue algorithm
BOOST_AUTO_TEST_CASE(test_matrix_tan_eigenvalue_algorithm_float) {
  auto tan_block =
      matrix_tan_sync_f::make({{2, 2}}, TrigAlgorithm::EIGENVALUE, 50, 1e-6);
  BOOST_REQUIRE(tan_block);

  auto small_matrix = create_small_matrix_2x2<float>();
  std::vector<float> output(4, 0.0f);

  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  Eigen::Map<const Eigen::MatrixXf> input_map(small_matrix.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXf> output_map(output.data(), 2, 2);

  inputs.push_back(&input_map);
  outputs.push_back(&output_map);

  auto result = tan_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Should give same result as Taylor series for diagonal case
  BOOST_CHECK_CLOSE(output[0], std::tan(0.1f), 1e-3f);
  BOOST_CHECK_CLOSE(output[3], std::tan(0.2f), 1e-3f);
}

// Test trigonometric identity: tan(A) = sin(A)/cos(A) (for diagonal matrices)
BOOST_AUTO_TEST_CASE(test_matrix_tan_identity_float) {
  auto tan_block =
      matrix_tan_sync_f::make({{2, 2}}, TrigAlgorithm::AUTO, 50, 1e-6);
  auto sin_block =
      matrix_sin_sync_f::make({{2, 2}}, TrigAlgorithm::AUTO, 50, 1e-6);
  auto cos_block =
      matrix_cos_sync_f::make({{2, 2}}, TrigAlgorithm::AUTO, 50, 1e-6);
  BOOST_REQUIRE(tan_block);
  BOOST_REQUIRE(sin_block);
  BOOST_REQUIRE(cos_block);

  auto small_matrix = create_small_matrix_2x2<float>();
  std::vector<float> tan_output(4, 0.0f);
  std::vector<float> sin_output(4, 0.0f);
  std::vector<float> cos_output(4, 0.0f);

  // Compute tan(A), sin(A), and cos(A)
  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> tan_outputs, sin_outputs, cos_outputs;

  Eigen::Map<const Eigen::MatrixXf> input_map(small_matrix.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXf> tan_output_map(tan_output.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXf> sin_output_map(sin_output.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXf> cos_output_map(cos_output.data(), 2, 2);

  inputs.push_back(&input_map);
  tan_outputs.push_back(&tan_output_map);
  sin_outputs.push_back(&sin_output_map);
  cos_outputs.push_back(&cos_output_map);

  auto tan_result = tan_block->operation(inputs, tan_outputs);
  auto sin_result = sin_block->operation(inputs, sin_outputs);
  auto cos_result = cos_block->operation(inputs, cos_outputs);

  BOOST_CHECK_EQUAL(tan_result, OperationReturn::SUCCESS);
  BOOST_CHECK_EQUAL(sin_result, OperationReturn::SUCCESS);
  BOOST_CHECK_EQUAL(cos_result, OperationReturn::SUCCESS);

  // Verify tan(A) ≈ sin(A) * cos(A)^(-1) for diagonal elements
  float tolerance = 1e-3f;
  BOOST_CHECK_CLOSE(tan_output[0], sin_output[0] / cos_output[0],
                    tolerance); // [0,0]
  BOOST_CHECK_CLOSE(tan_output[3], sin_output[3] / cos_output[3],
                    tolerance); // [1,1]
}

// Test complex matrix tangent
BOOST_AUTO_TEST_CASE(test_matrix_tan_complex_float) {
  auto tan_block =
      matrix_tan_sync_cf::make({{2, 2}}, TrigAlgorithm::AUTO, 50, 1e-6);
  BOOST_REQUIRE(tan_block);

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

  auto result = tan_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // For diagonal complex matrix, verify with standard complex tan
  auto expected_tan_1 = std::tan(std::complex<float>(0.1f, 0.1f));
  auto expected_tan_2 = std::tan(std::complex<float>(0.2f, -0.1f));

  BOOST_CHECK_CLOSE(output[0].real(), expected_tan_1.real(), 1e-2f);
  BOOST_CHECK_CLOSE(output[0].imag(), expected_tan_1.imag(), 1e-2f);
  BOOST_CHECK_CLOSE(output[3].real(), expected_tan_2.real(), 1e-2f);
  BOOST_CHECK_CLOSE(output[3].imag(), expected_tan_2.imag(), 1e-2f);
}

// Test double precision
BOOST_AUTO_TEST_CASE(test_matrix_tan_double_precision) {
  auto tan_block =
      matrix_tan_sync_d::make({{2, 2}}, TrigAlgorithm::AUTO, 50, 1e-10);
  BOOST_REQUIRE(tan_block);

  auto small_matrix = create_small_matrix_2x2<double>();
  std::vector<double> output(4, 0.0);

  types::vector_const_matrix_map<double> inputs;
  types::vector_matrix_map<double> outputs;

  Eigen::Map<const Eigen::MatrixXd> input_map(small_matrix.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXd> output_map(output.data(), 2, 2);

  inputs.push_back(&input_map);
  outputs.push_back(&output_map);

  auto result = tan_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Higher precision verification
  BOOST_CHECK_CLOSE(output[0], std::tan(0.1), 1e-8);
  BOOST_CHECK_CLOSE(output[3], std::tan(0.2), 1e-8);
}

// Test error handling for non-square matrix
BOOST_AUTO_TEST_CASE(test_matrix_tan_error_handling_non_square) {
  BOOST_CHECK_THROW(
      matrix_tan_sync_f::make({{2, 3}}, TrigAlgorithm::AUTO, 50, 1e-6),
      std::invalid_argument);
}

// Test handling near singularities (where cosine approaches zero)
BOOST_AUTO_TEST_CASE(test_matrix_tan_near_singularity) {
  auto tan_block =
      matrix_tan_sync_f::make({{2, 2}}, TrigAlgorithm::EIGENVALUE, 50, 1e-6);
  BOOST_REQUIRE(tan_block);

  // Create matrix close to pi/2 (where tan becomes large)
  float near_pi_2 = static_cast<float>(M_PI / 2 - 0.01);
  std::vector<float> near_singular = {near_pi_2, 0.0f, 0.0f, 0.1f};
  std::vector<float> output(4, 0.0f);

  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  Eigen::Map<const Eigen::MatrixXf> input_map(near_singular.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXf> output_map(output.data(), 2, 2);

  inputs.push_back(&input_map);
  outputs.push_back(&output_map);

  auto result = tan_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Should handle gracefully even if tan is large
  BOOST_CHECK_CLOSE(output[0], std::tan(near_pi_2),
                    5e-2f); // Allow larger tolerance
  BOOST_CHECK_CLOSE(output[3], std::tan(0.1f), 1e-3f);
}

// Test 3x3 matrix tangent
BOOST_AUTO_TEST_CASE(test_matrix_tan_3x3_float) {
  auto tan_block =
      matrix_tan_sync_f::make({{3, 3}}, TrigAlgorithm::AUTO, 50, 1e-6);
  BOOST_REQUIRE(tan_block);

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

  auto result = tan_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Check diagonal elements
  BOOST_CHECK_CLOSE(output[0], std::tan(0.1f), 1e-4f); // [0,0]
  BOOST_CHECK_CLOSE(output[4], std::tan(0.2f), 1e-4f); // [1,1]
  BOOST_CHECK_CLOSE(output[8], std::tan(0.3f), 1e-4f); // [2,2]

  // Check off-diagonal elements are zero
  for (int i : {1, 2, 3, 5, 6, 7}) {
    BOOST_CHECK_SMALL(output[i], 1e-6f);
  }
}

BOOST_AUTO_TEST_SUITE_END()

} // namespace linalg
} // namespace gr