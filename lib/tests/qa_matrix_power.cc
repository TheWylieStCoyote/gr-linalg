/* -*- c++ -*- */
/*
 * Copyright 2024 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include <Eigen/Dense>
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <complex>
#include <gnuradio/linalg/matrix_power.h>
#include <gnuradio/linalg/types.h>

using namespace gr::linalg;

BOOST_AUTO_TEST_SUITE(test_matrix_power)

// Test basic element-wise power operations
BOOST_AUTO_TEST_CASE(test_elementwise_power_basic) {
  types::shape shape{2, 2};
  const int num_inputs = 2;

  auto power_op =
      std::make_shared<matrix_power<float>>(false, true); // element-wise mode

  // Create input data
  std::vector<float> input_matrix{2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<float> input_exponent{2.0f};
  std::vector<float> output(4);

  // Create matrix maps using the correct types
  types::matrix_map_dynamic<float> matrix_map(input_matrix.data(), 2, 2);
  types::matrix_map_dynamic<float> exp_map(input_exponent.data(), 1, 1);
  types::matrix_map_dynamic<float> result_map(output.data(), 2, 2);

  types::vector_const_matrix_map<float> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&matrix_map),
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&exp_map)};
  types::vector_matrix_map<float> outputs{&result_map};

  BOOST_CHECK_EQUAL(power_op->operation(inputs, outputs),
                    OperationReturn::SUCCESS);

  // Check results: each element should be squared
  BOOST_CHECK_CLOSE(output[0], 4.0f, 1e-5);  // 2^2 = 4
  BOOST_CHECK_CLOSE(output[1], 9.0f, 1e-5);  // 3^2 = 9
  BOOST_CHECK_CLOSE(output[2], 16.0f, 1e-5); // 4^2 = 16
  BOOST_CHECK_CLOSE(output[3], 25.0f, 1e-5); // 5^2 = 25
}

// Test matrix power with positive exponent
BOOST_AUTO_TEST_CASE(test_matrix_power_positive) {
  auto power_op =
      std::make_shared<matrix_power<double>>(true, true); // matrix power mode

  // Create 2x2 test matrix: [[1, 1], [0, 1]]
  std::vector<double> input_matrix{1.0, 0.0, 1.0, 1.0}; // Column-major
  std::vector<double> input_exponent{3.0};
  std::vector<double> output(4);

  types::matrix_map_dynamic<double> matrix_map(input_matrix.data(), 2, 2);
  types::matrix_map_dynamic<double> exp_map(input_exponent.data(), 1, 1);
  types::matrix_map_dynamic<double> result_map(output.data(), 2, 2);

  types::vector_const_matrix_map<double> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<double> *>(&matrix_map),
      reinterpret_cast<types::const_matrix_map_dynamic<double> *>(&exp_map)};
  types::vector_matrix_map<double> outputs{&result_map};

  BOOST_CHECK_EQUAL(power_op->operation(inputs, outputs),
                    OperationReturn::SUCCESS);

  // For matrix [[1, 1], [0, 1]]^3 = [[1, 3], [0, 1]]
  BOOST_CHECK_CLOSE(output[0], 1.0, 1e-10); // (0,0)
  BOOST_CHECK_CLOSE(output[1], 0.0, 1e-10); // (1,0)
  BOOST_CHECK_CLOSE(output[2], 3.0, 1e-10); // (0,1)
  BOOST_CHECK_CLOSE(output[3], 1.0, 1e-10); // (1,1)
}

// Test error handling
BOOST_AUTO_TEST_CASE(test_error_handling) {
  auto power_op = std::make_shared<matrix_power<float>>(true, true);

  // Non-square matrix should fail in matrix mode
  std::vector<float> input_matrix(6, 1.0f); // 2x3 matrix
  std::vector<float> input_exponent{2.0f};
  std::vector<float> output(6);

  types::matrix_map_dynamic<float> matrix_map(input_matrix.data(), 2, 3);
  types::matrix_map_dynamic<float> exp_map(input_exponent.data(), 1, 1);
  types::matrix_map_dynamic<float> result_map(output.data(), 2, 3);

  types::vector_const_matrix_map<float> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&matrix_map),
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&exp_map)};
  types::vector_matrix_map<float> outputs{&result_map};

  // Should fail because matrix is not square
  BOOST_CHECK_EQUAL(power_op->operation(inputs, outputs),
                    OperationReturn::INVALID_SHAPE);
}

BOOST_AUTO_TEST_SUITE_END()