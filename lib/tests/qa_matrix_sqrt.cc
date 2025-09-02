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
#include <gnuradio/linalg/matrix_sqrt.h>
#include <iostream>

namespace gr {
namespace linalg {

BOOST_AUTO_TEST_SUITE(qa_matrix_sqrt)

// Helper function to create test matrices
template <typename Scalar>
std::vector<Scalar> create_identity_matrix_2x2() {
  // Identity matrix: [[1, 0], [0, 1]]
  // Column-major: [1, 0, 0, 1]
  return {Scalar(1), Scalar(0), Scalar(0), Scalar(1)};
}

template <typename Scalar>
std::vector<Scalar> create_diagonal_matrix_2x2() {
  // Diagonal matrix: [[4, 0], [0, 9]]
  // Column-major: [4, 0, 0, 9]
  return {Scalar(4), Scalar(0), Scalar(0), Scalar(9)};
}

template <typename Scalar>
std::vector<Scalar> create_positive_definite_2x2() {
  // Positive definite matrix: [[2, 1], [1, 2]]
  // Column-major: [2, 1, 1, 2]
  return {Scalar(2), Scalar(1), Scalar(1), Scalar(2)};
}

// Test matrix square root of identity matrix (should be identity)
BOOST_AUTO_TEST_CASE(test_matrix_sqrt_identity_float) {
  auto sqrt_block = matrix_sqrt_sync_f::make(
      {{2, 2}}, matrix_sqrt<float>::Algorithm::AUTO, 100, 1e-6);
  BOOST_REQUIRE(sqrt_block);

  auto identity = create_identity_matrix_2x2<float>();
  std::vector<float> output(4, 0.0f);

  // Simulate processing
  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  Eigen::Map<const Eigen::MatrixXf> input_map(identity.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXf> output_map(output.data(), 2, 2);

  inputs.push_back(&input_map);
  outputs.push_back(&output_map);

  auto result = sqrt_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // sqrt(I) should be I
  BOOST_CHECK_CLOSE(output[0], 1.0f, 1e-4f); // [0,0]
  BOOST_CHECK_SMALL(output[1], 1e-5f);       // [1,0]
  BOOST_CHECK_SMALL(output[2], 1e-5f);       // [0,1]
  BOOST_CHECK_CLOSE(output[3], 1.0f, 1e-4f); // [1,1]
}

// Test matrix square root of diagonal matrix
BOOST_AUTO_TEST_CASE(test_matrix_sqrt_diagonal_float) {
  auto sqrt_block = matrix_sqrt_sync_f::make(
      {{2, 2}}, matrix_sqrt<float>::Algorithm::AUTO, 100, 1e-6);
  BOOST_REQUIRE(sqrt_block);

  auto diagonal = create_diagonal_matrix_2x2<float>();
  std::vector<float> output(4, 0.0f);

  // Simulate processing
  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  Eigen::Map<const Eigen::MatrixXf> input_map(diagonal.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXf> output_map(output.data(), 2, 2);

  inputs.push_back(&input_map);
  outputs.push_back(&output_map);

  auto result = sqrt_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // For diagonal matrix [[4, 0], [0, 9]], sqrt should be [[2, 0], [0, 3]]
  // Column-major: [2, 0, 0, 3]
  BOOST_CHECK_CLOSE(output[0], 2.0f, 1e-4f); // sqrt(4)
  BOOST_CHECK_SMALL(output[1], 1e-5f);       // 0
  BOOST_CHECK_SMALL(output[2], 1e-5f);       // 0
  BOOST_CHECK_CLOSE(output[3], 3.0f, 1e-4f); // sqrt(9)
}

// Test fundamental property: (sqrt(A))^2 = A
BOOST_AUTO_TEST_CASE(test_matrix_sqrt_square_property_float) {
  auto sqrt_block = matrix_sqrt_sync_f::make(
      {{2, 2}}, matrix_sqrt<float>::Algorithm::AUTO, 100, 1e-6);
  BOOST_REQUIRE(sqrt_block);

  auto input_matrix = create_positive_definite_2x2<float>();
  std::vector<float> sqrt_output(4, 0.0f);

  // Compute sqrt(A)
  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  Eigen::Map<const Eigen::MatrixXf> input_map(input_matrix.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXf> sqrt_output_map(sqrt_output.data(), 2, 2);

  inputs.push_back(&input_map);
  outputs.push_back(&sqrt_output_map);

  auto result = sqrt_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Now compute (sqrt(A))^2 using Eigen directly
  Eigen::MatrixXf sqrt_result = sqrt_output_map;
  Eigen::MatrixXf squared_result = sqrt_result * sqrt_result;

  // (sqrt(A))^2 should equal original matrix A
  float tolerance = 1e-3f;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      BOOST_CHECK_CLOSE(squared_result(i, j), input_map(i, j), tolerance);
    }
  }
}

// Test different algorithms
BOOST_AUTO_TEST_CASE(test_matrix_sqrt_eigenvalue_algorithm_float) {
  auto sqrt_block = matrix_sqrt_sync_f::make(
      {{2, 2}}, matrix_sqrt<float>::Algorithm::EIGENVALUE, 100, 1e-6);
  BOOST_REQUIRE(sqrt_block);

  auto input_matrix = create_diagonal_matrix_2x2<float>();
  std::vector<float> output(4, 0.0f);

  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  Eigen::Map<const Eigen::MatrixXf> input_map(input_matrix.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXf> output_map(output.data(), 2, 2);

  inputs.push_back(&input_map);
  outputs.push_back(&output_map);

  auto result = sqrt_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Should give same result as AUTO for this diagonal case
  BOOST_CHECK_CLOSE(output[0], 2.0f, 1e-4f);
  BOOST_CHECK_CLOSE(output[3], 3.0f, 1e-4f);
}

BOOST_AUTO_TEST_CASE(test_matrix_sqrt_newton_algorithm_float) {
  auto sqrt_block = matrix_sqrt_sync_f::make(
      {{2, 2}}, matrix_sqrt<float>::Algorithm::NEWTON, 100, 1e-6);
  BOOST_REQUIRE(sqrt_block);

  auto input_matrix = create_positive_definite_2x2<float>();
  std::vector<float> output(4, 0.0f);

  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  Eigen::Map<const Eigen::MatrixXf> input_map(input_matrix.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXf> output_map(output.data(), 2, 2);

  inputs.push_back(&input_map);
  outputs.push_back(&output_map);

  auto result = sqrt_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Verify the square property
  Eigen::MatrixXf sqrt_result = output_map;
  Eigen::MatrixXf squared = sqrt_result * sqrt_result;

  float tolerance = 1e-3f;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      BOOST_CHECK_CLOSE(squared(i, j), input_map(i, j), tolerance);
    }
  }
}

// Test complex matrix square root
BOOST_AUTO_TEST_CASE(test_matrix_sqrt_complex_float) {
  auto sqrt_block = matrix_sqrt_sync_cf::make(
      {{2, 2}}, matrix_sqrt<std::complex<float>>::Algorithm::AUTO, 100, 1e-6);
  BOOST_REQUIRE(sqrt_block);

  // Create complex matrix with known square root
  std::vector<std::complex<float>> input = {
      std::complex<float>(1, 1), // (1+i)
      std::complex<float>(0, 0), // 0
      std::complex<float>(0, 0), // 0
      std::complex<float>(2, 2)  // (2+2i)
  };

  std::vector<std::complex<float>> output(4, std::complex<float>(0, 0));

  types::vector_const_matrix_map<std::complex<float>> inputs;
  types::vector_matrix_map<std::complex<float>> outputs;

  Eigen::Map<const Eigen::MatrixXcf> input_map(input.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXcf> output_map(output.data(), 2, 2);

  inputs.push_back(&input_map);
  outputs.push_back(&output_map);

  auto result = sqrt_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Verify by squaring the result
  Eigen::MatrixXcf sqrt_result = output_map;
  Eigen::MatrixXcf squared = sqrt_result * sqrt_result;

  float tolerance = 1e-3f;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      BOOST_CHECK_CLOSE(squared(i, j).real(), input_map(i, j).real(),
                        tolerance);
      BOOST_CHECK_CLOSE(squared(i, j).imag(), input_map(i, j).imag(),
                        tolerance);
    }
  }
}

// Test double precision
BOOST_AUTO_TEST_CASE(test_matrix_sqrt_double_precision) {
  auto sqrt_block = matrix_sqrt_sync_d::make(
      {{2, 2}}, matrix_sqrt<double>::Algorithm::AUTO, 100, 1e-10);
  BOOST_REQUIRE(sqrt_block);

  auto input_matrix = create_positive_definite_2x2<double>();
  std::vector<double> output(4, 0.0);

  types::vector_const_matrix_map<double> inputs;
  types::vector_matrix_map<double> outputs;

  Eigen::Map<const Eigen::MatrixXd> input_map(input_matrix.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXd> output_map(output.data(), 2, 2);

  inputs.push_back(&input_map);
  outputs.push_back(&output_map);

  auto result = sqrt_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Higher precision verification
  Eigen::MatrixXd sqrt_result = output_map;
  Eigen::MatrixXd squared = sqrt_result * sqrt_result;

  double tolerance = 1e-8;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      BOOST_CHECK_CLOSE(squared(i, j), input_map(i, j), tolerance);
    }
  }
}

// Test error handling for non-square matrix
BOOST_AUTO_TEST_CASE(test_matrix_sqrt_error_handling_non_square) {
  BOOST_CHECK_THROW(
      matrix_sqrt_sync_f::make({{2, 3}}, matrix_sqrt<float>::Algorithm::AUTO,
                               100, 1e-6),
      std::invalid_argument);
}

// Test convergence with limited iterations
BOOST_AUTO_TEST_CASE(test_matrix_sqrt_convergence_limit) {
  // Very few iterations should still work for well-conditioned matrices
  auto sqrt_block = matrix_sqrt_sync_f::make(
      {{2, 2}}, matrix_sqrt<float>::Algorithm::NEWTON, 5, 1e-6);
  BOOST_REQUIRE(sqrt_block);

  auto input_matrix = create_identity_matrix_2x2<float>();
  std::vector<float> output(4, 0.0f);

  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  Eigen::Map<const Eigen::MatrixXf> input_map(input_matrix.data(), 2, 2);
  Eigen::Map<Eigen::MatrixXf> output_map(output.data(), 2, 2);

  inputs.push_back(&input_map);
  outputs.push_back(&output_map);

  auto result = sqrt_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Identity should converge quickly
  BOOST_CHECK_CLOSE(output[0], 1.0f, 1e-3f);
  BOOST_CHECK_CLOSE(output[3], 1.0f, 1e-3f);
}

// Test 3x3 matrix square root
BOOST_AUTO_TEST_CASE(test_matrix_sqrt_3x3_float) {
  auto sqrt_block = matrix_sqrt_sync_f::make(
      {{3, 3}}, matrix_sqrt<float>::Algorithm::AUTO, 100, 1e-6);
  BOOST_REQUIRE(sqrt_block);

  // Create 3x3 positive definite matrix
  std::vector<float> input_3x3 = {
      4.0f, 1.0f, 1.0f, // First column
      1.0f, 4.0f, 1.0f, // Second column
      1.0f, 1.0f, 4.0f  // Third column
  };
  std::vector<float> output(9, 0.0f);

  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  Eigen::Map<const Eigen::MatrixXf> input_map(input_3x3.data(), 3, 3);
  Eigen::Map<Eigen::MatrixXf> output_map(output.data(), 3, 3);

  inputs.push_back(&input_map);
  outputs.push_back(&output_map);

  auto result = sqrt_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Verify the square property for 3x3 case
  Eigen::MatrixXf sqrt_result = output_map;
  Eigen::MatrixXf squared = sqrt_result * sqrt_result;

  float tolerance = 1e-3f;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      BOOST_CHECK_CLOSE(squared(i, j), input_map(i, j), tolerance);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()

} // namespace linalg
} // namespace gr