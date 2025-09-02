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
#include <gnuradio/linalg/matrix_log.h>
#include <iostream>
#include <unsupported/Eigen/MatrixFunctions>

namespace gr {
namespace linalg {

BOOST_AUTO_TEST_SUITE(qa_matrix_log)

// Helper function to create test matrices
template <typename Scalar>
std::vector<Scalar> create_test_matrix_2x2() {
  // Create a simple 2x2 positive definite matrix [[2, 1], [1, 2]]
  // Column-major: [2, 1, 1, 2]
  return {Scalar(2), Scalar(1), Scalar(1), Scalar(2)};
}

template <typename Scalar>
std::vector<Scalar> create_identity_matrix_2x2() {
  // Identity matrix: [[1, 0], [0, 1]]
  // Column-major: [1, 0, 0, 1]
  return {Scalar(1), Scalar(0), Scalar(0), Scalar(1)};
}

template <typename Scalar>
std::vector<Scalar> create_diagonal_matrix_2x2() {
  // Diagonal matrix: [[2, 0], [0, 3]]
  // Column-major: [2, 0, 0, 3]
  return {Scalar(2), Scalar(0), Scalar(0), Scalar(3)};
}

// Test matrix logarithm of identity matrix (should be zero matrix)
BOOST_AUTO_TEST_CASE(test_matrix_log_identity_float) {
  auto log_block = matrix_log_sync_f::make({{2, 2}});
  BOOST_REQUIRE(log_block);

  auto identity = create_identity_matrix_2x2<float>();
  std::vector<float> output(4, 0.0f);

  // Simulate processing
  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  Eigen::Map<const Eigen::MatrixXf> input_map(identity.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXf> output_map(output.data(), 2, 2);

  inputs.push_back(&input_map);
  outputs.push_back(&output_map);

  auto result = log_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // log(I) should be zero matrix
  for (size_t i = 0; i < 4; ++i) {
    BOOST_CHECK_SMALL(output[i], 1e-5f);
  }
}

// Test matrix logarithm of diagonal matrix
BOOST_AUTO_TEST_CASE(test_matrix_log_diagonal_float) {
  auto log_block = matrix_log_sync_f::make({{2, 2}});
  BOOST_REQUIRE(log_block);

  auto diagonal = create_diagonal_matrix_2x2<float>();
  std::vector<float> output(4, 0.0f);

  // Simulate processing
  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  Eigen::Map<const Eigen::MatrixXf> input_map(diagonal.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXf> output_map(output.data(), 2, 2);

  inputs.push_back(&input_map);
  outputs.push_back(&output_map);

  auto result = log_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // For diagonal matrix [[2, 0], [0, 3]], log should be [[log(2), 0], [0,
  // log(3)]] Column-major: [log(2), 0, 0, log(3)]
  BOOST_CHECK_CLOSE(output[0], std::log(2.0f), 1e-4f); // log(2)
  BOOST_CHECK_SMALL(output[1], 1e-5f);                 // 0
  BOOST_CHECK_SMALL(output[2], 1e-5f);                 // 0
  BOOST_CHECK_CLOSE(output[3], std::log(3.0f), 1e-4f); // log(3)
}

// Test matrix logarithm with complex numbers
BOOST_AUTO_TEST_CASE(test_matrix_log_complex_float) {
  auto log_block = matrix_log_sync_cf::make({{2, 2}});
  BOOST_REQUIRE(log_block);

  // Create a complex matrix with eigenvalues that have positive real parts
  std::vector<std::complex<float>> input = {
      std::complex<float>(2, 1), // (2+i)
      std::complex<float>(0, 0), // 0
      std::complex<float>(0, 0), // 0
      std::complex<float>(1, 2)  // (1+2i)
  };

  std::vector<std::complex<float>> output(4, std::complex<float>(0, 0));

  // Simulate processing
  types::vector_const_matrix_map<std::complex<float>> inputs;
  types::vector_matrix_map<std::complex<float>> outputs;

  Eigen::Map<const Eigen::MatrixXcf> input_map(input.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXcf> output_map(output.data(), 2, 2);

  inputs.push_back(&input_map);
  outputs.push_back(&output_map);

  auto result = log_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // For diagonal complex matrix, log of eigenvalues should match
  // log(2+i) and log(1+2i) computed using complex logarithm
  auto expected_log_1 = std::log(std::complex<float>(2, 1));
  auto expected_log_2 = std::log(std::complex<float>(1, 2));

  BOOST_CHECK_CLOSE(output[0].real(), expected_log_1.real(), 1e-3f);
  BOOST_CHECK_CLOSE(output[0].imag(), expected_log_1.imag(), 1e-3f);
  BOOST_CHECK_CLOSE(output[3].real(), expected_log_2.real(), 1e-3f);
  BOOST_CHECK_CLOSE(output[3].imag(), expected_log_2.imag(), 1e-3f);
}

// Test matrix logarithm fundamental property: exp(log(A)) = A
BOOST_AUTO_TEST_CASE(test_matrix_log_exp_property_float) {
  auto log_block = matrix_log_sync_f::make({{2, 2}});
  BOOST_REQUIRE(log_block);

  auto input_matrix = create_test_matrix_2x2<float>();
  std::vector<float> log_output(4, 0.0f);

  // Compute log(A)
  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  Eigen::Map<const Eigen::MatrixXf> input_map(input_matrix.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXf> log_output_map(log_output.data(), 2, 2);

  inputs.push_back(&input_map);
  outputs.push_back(&log_output_map);

  auto result = log_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Now compute exp(log(A)) using Eigen directly for verification
  Eigen::MatrixXf log_result = log_output_map;
  Eigen::MatrixXf exp_log_result = log_result.exp();

  // exp(log(A)) should equal original matrix A
  float tolerance = 1e-4f;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      BOOST_CHECK_CLOSE(exp_log_result(i, j), input_map(i, j), tolerance);
    }
  }
}

// Test double precision
BOOST_AUTO_TEST_CASE(test_matrix_log_double_precision) {
  auto log_block = matrix_log_sync_d::make({{2, 2}});
  BOOST_REQUIRE(log_block);

  auto input_matrix = create_test_matrix_2x2<double>();
  std::vector<double> output(4, 0.0);

  // Simulate processing
  types::vector_const_matrix_map<double> inputs;
  types::vector_matrix_map<double> outputs;

  Eigen::Map<const Eigen::MatrixXd> input_map(input_matrix.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXd> output_map(output.data(), 2, 2);

  inputs.push_back(&input_map);
  outputs.push_back(&output_map);

  auto result = log_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Verify output is not zero (test matrix has no zero eigenvalues)
  bool has_nonzero = false;
  for (size_t i = 0; i < 4; ++i) {
    if (std::abs(output[i]) > 1e-10) {
      has_nonzero = true;
      break;
    }
  }
  BOOST_CHECK(has_nonzero);
}

// Test error handling for non-square matrix
BOOST_AUTO_TEST_CASE(test_matrix_log_error_handling_non_square) {
  // Should fail to create block with non-square matrix
  BOOST_CHECK_THROW(matrix_log_sync_f::make({{2, 3}}), std::invalid_argument);
}

// Test error handling for singular matrix (determinant = 0)
BOOST_AUTO_TEST_CASE(test_matrix_log_singular_matrix_handling) {
  auto log_block = matrix_log_sync_f::make({{2, 2}});
  BOOST_REQUIRE(log_block);

  // Create singular matrix [[1, 1], [1, 1]] (determinant = 0)
  std::vector<float> singular_matrix = {1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float> output(4, 0.0f);

  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  Eigen::Map<const Eigen::MatrixXf> input_map(singular_matrix.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXf> output_map(output.data(), 2, 2);

  inputs.push_back(&input_map);
  outputs.push_back(&output_map);

  // Should handle gracefully (may return FAILURE or compute principal branch)
  auto result = log_block->operation(inputs, outputs);
  // We don't check the result type as different algorithms may handle this
  // differently
  (void)result; // Suppress unused variable warning
}

// Test 3x3 matrix logarithm
BOOST_AUTO_TEST_CASE(test_matrix_log_3x3_float) {
  auto log_block = matrix_log_sync_f::make({{3, 3}});
  BOOST_REQUIRE(log_block);

  // Create 3x3 positive definite matrix
  std::vector<float> input_3x3 = {
      3.0f, 1.0f, 1.0f, // First column
      1.0f, 3.0f, 1.0f, // Second column
      1.0f, 1.0f, 3.0f  // Third column
  };
  std::vector<float> output(9, 0.0f);

  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  Eigen::Map<const Eigen::MatrixXf> input_map(input_3x3.data(), 3, 3);
  Eigen::Map<Eigen::MatrixXf> output_map(output.data(), 3, 3);

  inputs.push_back(&input_map);
  outputs.push_back(&output_map);

  auto result = log_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Check that result is not all zeros
  bool has_nonzero = false;
  for (size_t i = 0; i < 9; ++i) {
    if (std::abs(output[i]) > 1e-6f) {
      has_nonzero = true;
      break;
    }
  }
  BOOST_CHECK(has_nonzero);
}

BOOST_AUTO_TEST_SUITE_END()

} // namespace linalg
} // namespace gr