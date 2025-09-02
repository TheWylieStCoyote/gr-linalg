/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <Eigen/Dense>
#include <boost/test/unit_test.hpp>
#include <chrono>
#include <cmath>
#include <complex>
#include <gnuradio/blocks/vector_sink.h>
#include <gnuradio/blocks/vector_source.h>
#include <gnuradio/linalg/matrix_pseudo_inverse.h>
#include <gnuradio/linalg/types.h>
#include <gnuradio/top_block.h>
#include <limits>

using namespace gr::linalg;

BOOST_AUTO_TEST_SUITE(test_matrix_pseudo_inverse)

// Constants for numerical comparisons
constexpr float FLOAT_TOLERANCE = 1e-4f;
constexpr double DOUBLE_TOLERANCE = 1e-10;

// Test identity matrix (should return itself)
BOOST_AUTO_TEST_CASE(test_pseudo_inverse_identity) {
  auto pinv_op = std::make_shared<matrix_pseudo_inverse<double>>();

  // Create 3×3 identity matrix
  std::vector<double> input_matrix{
      1.0, 0.0, 0.0, // Column 0
      0.0, 1.0, 0.0, // Column 1
      0.0, 0.0, 1.0  // Column 2
  };
  std::vector<double> output(9); // 3x3 output matrix

  types::matrix_map_dynamic<double> matrix_map(input_matrix.data(), 3, 3);
  types::matrix_map_dynamic<double> result_map(output.data(), 3, 3);

  types::vector_const_matrix_map<double> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<double> *>(&matrix_map)};
  types::vector_matrix_map<double> outputs{&result_map};

  BOOST_CHECK_EQUAL(pinv_op->operation(inputs, outputs),
                    OperationReturn::SUCCESS);

  // Identity matrix pseudoinverse should be identity
  std::vector<double> expected{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_CHECK_CLOSE(output[i], expected[i], DOUBLE_TOLERANCE * 1000);
  }
}

// Test diagonal matrix
BOOST_AUTO_TEST_CASE(test_pseudo_inverse_diagonal) {
  auto pinv_op = std::make_shared<matrix_pseudo_inverse<float>>();

  // Create diagonal matrix with values [2, 3, 4]
  // Pseudoinverse should have [1/2, 1/3, 1/4]
  std::vector<float> input_matrix{
      2.0f, 0.0f, 0.0f, // Column 0
      0.0f, 3.0f, 0.0f, // Column 1
      0.0f, 0.0f, 4.0f  // Column 2
  };
  std::vector<float> output(9);

  types::matrix_map_dynamic<float> matrix_map(input_matrix.data(), 3, 3);
  types::matrix_map_dynamic<float> result_map(output.data(), 3, 3);

  types::vector_const_matrix_map<float> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&matrix_map)};
  types::vector_matrix_map<float> outputs{&result_map};

  BOOST_CHECK_EQUAL(pinv_op->operation(inputs, outputs),
                    OperationReturn::SUCCESS);

  // Check diagonal elements: should be reciprocals
  BOOST_CHECK_CLOSE(output[0], 0.5f, FLOAT_TOLERANCE);        // 1/2
  BOOST_CHECK_CLOSE(output[4], 1.0f / 3.0f, FLOAT_TOLERANCE); // 1/3
  BOOST_CHECK_CLOSE(output[8], 0.25f, FLOAT_TOLERANCE);       // 1/4

  // Check off-diagonal elements: should be zero
  BOOST_CHECK_SMALL(output[1], FLOAT_TOLERANCE);
  BOOST_CHECK_SMALL(output[2], FLOAT_TOLERANCE);
  BOOST_CHECK_SMALL(output[3], FLOAT_TOLERANCE);
  BOOST_CHECK_SMALL(output[5], FLOAT_TOLERANCE);
  BOOST_CHECK_SMALL(output[6], FLOAT_TOLERANCE);
  BOOST_CHECK_SMALL(output[7], FLOAT_TOLERANCE);
}

// Test rank-deficient matrix
BOOST_AUTO_TEST_CASE(test_pseudo_inverse_rank_deficient) {
  auto pinv_op = std::make_shared<matrix_pseudo_inverse<double>>();

  // Create rank-1 matrix: [[1, 2], [2, 4]] (second column = 2 * first column)
  std::vector<double> input_matrix{
      1.0, 2.0, // Column 0: [1, 2]
      2.0, 4.0  // Column 1: [2, 4]
  };
  std::vector<double> output(4); // 2x2 output

  types::matrix_map_dynamic<double> matrix_map(input_matrix.data(), 2, 2);
  types::matrix_map_dynamic<double> result_map(output.data(), 2, 2);

  types::vector_const_matrix_map<double> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<double> *>(&matrix_map)};
  types::vector_matrix_map<double> outputs{&result_map};

  BOOST_CHECK_EQUAL(pinv_op->operation(inputs, outputs),
                    OperationReturn::SUCCESS);

  // For rank-deficient matrices, we verify A * A+ * A = A
  Eigen::Matrix2d A = Eigen::Map<Eigen::Matrix2d>(input_matrix.data());
  Eigen::Matrix2d A_pinv = Eigen::Map<Eigen::Matrix2d>(output.data());
  Eigen::Matrix2d result = A * A_pinv * A;

  for (int i = 0; i < 4; ++i) {
    BOOST_CHECK_CLOSE(result.data()[i], A.data()[i], DOUBLE_TOLERANCE * 1000);
  }
}

// Test rectangular matrix (tall)
BOOST_AUTO_TEST_CASE(test_pseudo_inverse_rectangular_tall) {
  auto pinv_op = std::make_shared<matrix_pseudo_inverse<float>>();

  // Create 3×2 matrix (overdetermined system)
  std::vector<float> input_matrix{
      1.0f, 0.0f, 1.0f, // Column 0: [1, 0, 1]
      0.0f, 1.0f, 1.0f  // Column 1: [0, 1, 1]
  };
  std::vector<float> output(6); // 2×3 output (transposed dimensions)

  types::matrix_map_dynamic<float> matrix_map(input_matrix.data(), 3, 2);
  types::matrix_map_dynamic<float> result_map(output.data(), 2, 3);

  types::vector_const_matrix_map<float> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&matrix_map)};
  types::vector_matrix_map<float> outputs{&result_map};

  BOOST_CHECK_EQUAL(pinv_op->operation(inputs, outputs),
                    OperationReturn::SUCCESS);

  // Verify dimensions: input was 3×2, output should be 2×3
  // For overdetermined systems: A+ = (A^T * A)^(-1) * A^T
  Eigen::MatrixXf A = Eigen::Map<Eigen::MatrixXf>(input_matrix.data(), 3, 2);
  Eigen::MatrixXf A_pinv = Eigen::Map<Eigen::MatrixXf>(output.data(), 2, 3);

  // Test pseudoinverse property: A+ * A should be identity
  Eigen::MatrixXf should_be_identity = A_pinv * A;
  BOOST_CHECK_CLOSE(should_be_identity(0, 0), 1.0f, FLOAT_TOLERANCE);
  BOOST_CHECK_CLOSE(should_be_identity(1, 1), 1.0f, FLOAT_TOLERANCE);
  BOOST_CHECK_SMALL(should_be_identity(0, 1), FLOAT_TOLERANCE);
  BOOST_CHECK_SMALL(should_be_identity(1, 0), FLOAT_TOLERANCE);
}

// Test rectangular matrix (wide)
BOOST_AUTO_TEST_CASE(test_pseudo_inverse_rectangular_wide) {
  auto pinv_op = std::make_shared<matrix_pseudo_inverse<double>>();

  // Create 2×3 matrix (underdetermined system)
  std::vector<double> input_matrix{
      1.0, 0.0, // Column 0: [1, 0]
      0.0, 1.0, // Column 1: [0, 1]
      1.0, 1.0  // Column 2: [1, 1]
  };
  std::vector<double> output(6); // 3×2 output (transposed dimensions)

  types::matrix_map_dynamic<double> matrix_map(input_matrix.data(), 2, 3);
  types::matrix_map_dynamic<double> result_map(output.data(), 3, 2);

  types::vector_const_matrix_map<double> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<double> *>(&matrix_map)};
  types::vector_matrix_map<double> outputs{&result_map};

  BOOST_CHECK_EQUAL(pinv_op->operation(inputs, outputs),
                    OperationReturn::SUCCESS);

  // For underdetermined systems: A+ = A^T * (A * A^T)^(-1)
  Eigen::MatrixXd A = Eigen::Map<Eigen::MatrixXd>(input_matrix.data(), 2, 3);
  Eigen::MatrixXd A_pinv = Eigen::Map<Eigen::MatrixXd>(output.data(), 3, 2);

  // Test pseudoinverse property: A * A+ should be identity
  Eigen::MatrixXd should_be_identity = A * A_pinv;
  BOOST_CHECK_CLOSE(should_be_identity(0, 0), 1.0, DOUBLE_TOLERANCE * 100);
  BOOST_CHECK_CLOSE(should_be_identity(1, 1), 1.0, DOUBLE_TOLERANCE * 100);
  BOOST_CHECK_SMALL(should_be_identity(0, 1), DOUBLE_TOLERANCE * 100);
  BOOST_CHECK_SMALL(should_be_identity(1, 0), DOUBLE_TOLERANCE * 100);
}

// Test zero matrix
BOOST_AUTO_TEST_CASE(test_pseudo_inverse_zero_matrix) {
  auto pinv_op = std::make_shared<matrix_pseudo_inverse<float>>();

  // Create zero matrix
  std::vector<float> input_matrix(6, 0.0f); // 2×3 zero matrix
  std::vector<float> output(6);             // 3×2 output

  types::matrix_map_dynamic<float> matrix_map(input_matrix.data(), 2, 3);
  types::matrix_map_dynamic<float> result_map(output.data(), 3, 2);

  types::vector_const_matrix_map<float> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&matrix_map)};
  types::vector_matrix_map<float> outputs{&result_map};

  BOOST_CHECK_EQUAL(pinv_op->operation(inputs, outputs),
                    OperationReturn::SUCCESS);

  // Zero matrix pseudoinverse should be zero matrix of transposed dimensions
  for (float val : output) {
    BOOST_CHECK_SMALL(val, FLOAT_TOLERANCE);
  }
}

// Test single element matrix
BOOST_AUTO_TEST_CASE(test_pseudo_inverse_single_element) {
  auto pinv_op = std::make_shared<matrix_pseudo_inverse<double>>();

  // Single non-zero element
  std::vector<double> input_matrix{5.0};
  std::vector<double> output(1);

  types::matrix_map_dynamic<double> matrix_map(input_matrix.data(), 1, 1);
  types::matrix_map_dynamic<double> result_map(output.data(), 1, 1);

  types::vector_const_matrix_map<double> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<double> *>(&matrix_map)};
  types::vector_matrix_map<double> outputs{&result_map};

  BOOST_CHECK_EQUAL(pinv_op->operation(inputs, outputs),
                    OperationReturn::SUCCESS);

  // Single element pseudoinverse should be reciprocal
  BOOST_CHECK_CLOSE(output[0], 0.2, DOUBLE_TOLERANCE * 100); // 1/5
}

// Test single zero element
BOOST_AUTO_TEST_CASE(test_pseudo_inverse_single_zero) {
  auto pinv_op = std::make_shared<matrix_pseudo_inverse<double>>();

  // Single zero element
  std::vector<double> input_matrix{0.0};
  std::vector<double> output(1);

  types::matrix_map_dynamic<double> matrix_map(input_matrix.data(), 1, 1);
  types::matrix_map_dynamic<double> result_map(output.data(), 1, 1);

  types::vector_const_matrix_map<double> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<double> *>(&matrix_map)};
  types::vector_matrix_map<double> outputs{&result_map};

  BOOST_CHECK_EQUAL(pinv_op->operation(inputs, outputs),
                    OperationReturn::SUCCESS);

  // Zero element pseudoinverse should be zero
  BOOST_CHECK_SMALL(output[0], DOUBLE_TOLERANCE);
}

// Test complex matrix
BOOST_AUTO_TEST_CASE(test_pseudo_inverse_complex) {
  auto pinv_op =
      std::make_shared<matrix_pseudo_inverse<std::complex<double>>>();

  // Create 2×2 complex matrix
  std::vector<std::complex<double>> input_matrix{
      std::complex<double>(1.0, 1.0), // (0,0) = 1+i
      std::complex<double>(0.0, 0.0), // (1,0) = 0
      std::complex<double>(0.0, 0.0), // (0,1) = 0
      std::complex<double>(1.0, -1.0) // (1,1) = 1-i
  };
  std::vector<std::complex<double>> output(4);

  types::matrix_map_dynamic<std::complex<double>> matrix_map(
      input_matrix.data(), 2, 2);
  types::matrix_map_dynamic<std::complex<double>> result_map(output.data(), 2,
                                                             2);

  types::vector_const_matrix_map<std::complex<double>> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<std::complex<double>> *>(
          &matrix_map)};
  types::vector_matrix_map<std::complex<double>> outputs{&result_map};

  BOOST_CHECK_EQUAL(pinv_op->operation(inputs, outputs),
                    OperationReturn::SUCCESS);

  // For diagonal complex matrix, pseudoinverse should be conjugate reciprocals
  // (1+i)^+ = 1/(1+i) = (1-i)/|1+i|^2 = (1-i)/2
  // (1-i)^+ = 1/(1-i) = (1+i)/|1-i|^2 = (1+i)/2
  BOOST_CHECK_CLOSE(output[0].real(), 0.5,
                    DOUBLE_TOLERANCE * 1000); // Real part of (1-i)/2
  BOOST_CHECK_CLOSE(output[0].imag(), -0.5,
                    DOUBLE_TOLERANCE * 1000); // Imag part of (1-i)/2
  BOOST_CHECK_CLOSE(output[3].real(), 0.5,
                    DOUBLE_TOLERANCE * 1000); // Real part of (1+i)/2
  BOOST_CHECK_CLOSE(output[3].imag(), 0.5,
                    DOUBLE_TOLERANCE * 1000); // Imag part of (1+i)/2
}

// Test error handling
BOOST_AUTO_TEST_CASE(test_pseudo_inverse_error_handling) {
  auto pinv_op = std::make_shared<matrix_pseudo_inverse<float>>();

  std::vector<float> input_matrix{1.0f, 2.0f};
  std::vector<float> output(2);

  types::matrix_map_dynamic<float> matrix_map(input_matrix.data(), 1, 2);
  types::matrix_map_dynamic<float> result_map(output.data(), 2, 1);

  // Test insufficient inputs
  types::vector_const_matrix_map<float> empty_inputs{};
  types::vector_matrix_map<float> outputs{&result_map};

  BOOST_CHECK_EQUAL(pinv_op->operation(empty_inputs, outputs),
                    OperationReturn::FAILURE);

  // Test insufficient outputs
  types::vector_const_matrix_map<float> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&matrix_map)};
  types::vector_matrix_map<float> empty_outputs{};

  BOOST_CHECK_EQUAL(pinv_op->operation(inputs, empty_outputs),
                    OperationReturn::FAILURE);
}

// Test output shape validation
BOOST_AUTO_TEST_CASE(test_pseudo_inverse_output_shape) {
  auto pinv_op = std::make_shared<matrix_pseudo_inverse<double>>();

  std::vector<double> input_matrix{1.0, 2.0, 3.0, 4.0}; // 2×2 matrix
  std::vector<double> output(4); // Wrong size: should match input dimensions

  types::matrix_map_dynamic<double> matrix_map(input_matrix.data(), 2, 2);
  types::matrix_map_dynamic<double> result_map(output.data(),
                                               3, // Wrong dimensions
                                               2);

  types::vector_const_matrix_map<double> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<double> *>(&matrix_map)};
  types::vector_matrix_map<double> outputs{&result_map};

  // Should fail due to output dimension mismatch
  BOOST_CHECK_EQUAL(pinv_op->operation(inputs, outputs),
                    OperationReturn::INVALID_SHAPE);
}

// Test static helper functions
BOOST_AUTO_TEST_CASE(test_pseudo_inverse_static_methods) {
  // Test compute_output_shapes
  types::vector_shapes input_shapes = {{3, 2}}; // 3×2 matrix

  auto output_shapes =
      matrix_pseudo_inverse<float>::compute_output_shapes(input_shapes);

  BOOST_REQUIRE_EQUAL(output_shapes.size(), 1);
  BOOST_CHECK_EQUAL(output_shapes[0][0], 2); // Transposed: 2 rows
  BOOST_CHECK_EQUAL(output_shapes[0][1], 3); // Transposed: 3 columns

  // Test extract_real_value for complex numbers
  std::complex<double> complex_val(3.0, 4.0); // Magnitude = 5.0
  auto real_val =
      matrix_pseudo_inverse<std::complex<double>>::extract_real_value(
          complex_val);
  BOOST_CHECK_CLOSE(real_val, 5.0, DOUBLE_TOLERANCE * 100);

  // Test is_effectively_zero
  BOOST_CHECK(matrix_pseudo_inverse<double>::is_effectively_zero(1e-16, 1e-15));
  BOOST_CHECK(
      !matrix_pseudo_inverse<double>::is_effectively_zero(1e-14, 1e-15));
}

// Test Penrose properties (A*A+*A = A)
BOOST_AUTO_TEST_CASE(test_pseudo_inverse_penrose_property_1) {
  auto pinv_op = std::make_shared<matrix_pseudo_inverse<double>>();

  // Test with rectangular matrix
  std::vector<double> input_matrix{
      1.0, 2.0, // Column 0: [1, 2]
      3.0, 4.0, // Column 1: [3, 4]
      5.0, 6.0  // Column 2: [5, 6]
  };
  std::vector<double> output(6); // 3×2 output

  types::matrix_map_dynamic<double> matrix_map(input_matrix.data(), 2, 3);
  types::matrix_map_dynamic<double> result_map(output.data(), 3, 2);

  types::vector_const_matrix_map<double> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<double> *>(&matrix_map)};
  types::vector_matrix_map<double> outputs{&result_map};

  BOOST_CHECK_EQUAL(pinv_op->operation(inputs, outputs),
                    OperationReturn::SUCCESS);

  // Test Penrose property 1: A * A+ * A = A
  Eigen::MatrixXd A = Eigen::Map<Eigen::MatrixXd>(input_matrix.data(), 2, 3);
  Eigen::MatrixXd A_pinv = Eigen::Map<Eigen::MatrixXd>(output.data(), 3, 2);
  Eigen::MatrixXd result = A * A_pinv * A;

  for (int i = 0; i < A.size(); ++i) {
    BOOST_CHECK_CLOSE(result.data()[i], A.data()[i], DOUBLE_TOLERANCE * 1000);
  }
}

// GNU Radio Flowgraph Tests
BOOST_AUTO_TEST_CASE(test_pseudo_inverse_sync_flowgraph) {
  BOOST_TEST_MESSAGE(
      "Testing matrix pseudo-inverse sync block in GNU Radio flowgraph...");

  auto tb = gr::make_top_block("pseudo_inverse_test");

  // Create 2×2 diagonal matrix: [[2, 0], [0, 3]]
  // Pseudoinverse should be [[0.5, 0], [0, 1/3]]
  std::vector<float> input_data = {2.0f, 0.0f, 0.0f, 3.0f};

  auto vector_source = gr::blocks::vector_source_f::make(input_data, false, 4);
  auto pinv_block = matrix_pseudo_inverse_sync<float>::make({2, 2});
  auto output_sink = gr::blocks::vector_sink_f::make(4); // 2×2 output

  tb->connect(vector_source, 0, pinv_block, 0);
  tb->connect(pinv_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 4);

  // Check diagonal elements
  BOOST_CHECK_CLOSE(output_data[0], 0.5f, FLOAT_TOLERANCE);        // 1/2
  BOOST_CHECK_CLOSE(output_data[3], 1.0f / 3.0f, FLOAT_TOLERANCE); // 1/3

  // Check off-diagonal elements
  BOOST_CHECK_SMALL(output_data[1], FLOAT_TOLERANCE);
  BOOST_CHECK_SMALL(output_data[2], FLOAT_TOLERANCE);

  BOOST_TEST_MESSAGE("Matrix pseudo-inverse sync flowgraph test passed.");
}

BOOST_AUTO_TEST_CASE(test_pseudo_inverse_sync_flowgraph_rectangular) {
  BOOST_TEST_MESSAGE("Testing pseudo-inverse sync block with rectangular "
                     "matrix in flowgraph...");

  auto tb = gr::make_top_block("pseudo_inverse_rect_test");

  // Create 3×2 matrix: [[1, 0], [0, 1], [1, 1]]
  std::vector<float> input_data = {
      1.0f, 0.0f, 1.0f, // Column 0: [1, 0, 1]
      0.0f, 1.0f, 1.0f  // Column 1: [0, 1, 1]
  };

  auto vector_source = gr::blocks::vector_source_f::make(input_data, false, 6);
  auto pinv_block = matrix_pseudo_inverse_sync<float>::make({3, 2});
  auto output_sink = gr::blocks::vector_sink_f::make(6); // 2×3 output

  tb->connect(vector_source, 0, pinv_block, 0);
  tb->connect(pinv_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 6);

  // Verify that all output values are finite
  for (float val : output_data) {
    BOOST_CHECK(std::isfinite(val));
  }

  BOOST_TEST_MESSAGE(
      "Rectangular matrix pseudo-inverse flowgraph test passed.");
}

BOOST_AUTO_TEST_CASE(test_pseudo_inverse_sync_flowgraph_rank_deficient) {
  BOOST_TEST_MESSAGE("Testing pseudo-inverse sync block with rank-deficient "
                     "matrix in flowgraph...");

  auto tb = gr::make_top_block("pseudo_inverse_rank_test");

  // Create rank-1 matrix: [[1, 2], [2, 4]] (rank deficient)
  std::vector<float> input_data = {1.0f, 2.0f, 2.0f, 4.0f};

  auto vector_source = gr::blocks::vector_source_f::make(input_data, false, 4);
  auto pinv_block = matrix_pseudo_inverse_sync<float>::make({2, 2});
  auto output_sink = gr::blocks::vector_sink_f::make(4);

  tb->connect(vector_source, 0, pinv_block, 0);
  tb->connect(pinv_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 4);

  // Verify that all output values are finite
  for (float val : output_data) {
    BOOST_CHECK(std::isfinite(val));
  }

  BOOST_TEST_MESSAGE(
      "Rank-deficient matrix pseudo-inverse flowgraph test passed.");
}

BOOST_AUTO_TEST_CASE(test_pseudo_inverse_sync_flowgraph_identity) {
  BOOST_TEST_MESSAGE(
      "Testing pseudo-inverse sync block with identity matrix in flowgraph...");

  auto tb = gr::make_top_block("pseudo_inverse_identity_test");

  // Create 3×3 identity matrix
  std::vector<float> input_data = {
      1.0f, 0.0f, 0.0f, // Column 0: [1, 0, 0]
      0.0f, 1.0f, 0.0f, // Column 1: [0, 1, 0]
      0.0f, 0.0f, 1.0f  // Column 2: [0, 0, 1]
  };

  auto vector_source = gr::blocks::vector_source_f::make(input_data, false, 9);
  auto pinv_block = matrix_pseudo_inverse_sync<float>::make({3, 3});
  auto output_sink = gr::blocks::vector_sink_f::make(9);

  tb->connect(vector_source, 0, pinv_block, 0);
  tb->connect(pinv_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 9);

  // Identity matrix pseudoinverse should be identity
  std::vector<float> expected = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                                 0.0f, 0.0f, 0.0f, 1.0f};
  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_CHECK_CLOSE(output_data[i], expected[i], FLOAT_TOLERANCE);
  }

  BOOST_TEST_MESSAGE("Identity matrix pseudo-inverse flowgraph test passed.");
}

BOOST_AUTO_TEST_CASE(test_pseudo_inverse_sync_flowgraph_zero_matrix) {
  BOOST_TEST_MESSAGE(
      "Testing pseudo-inverse sync block with zero matrix in flowgraph...");

  auto tb = gr::make_top_block("pseudo_inverse_zero_test");

  // Create 2×2 zero matrix
  std::vector<float> input_data = {0.0f, 0.0f, 0.0f, 0.0f};

  auto vector_source = gr::blocks::vector_source_f::make(input_data, false, 4);
  auto pinv_block = matrix_pseudo_inverse_sync<float>::make({2, 2});
  auto output_sink = gr::blocks::vector_sink_f::make(4);

  tb->connect(vector_source, 0, pinv_block, 0);
  tb->connect(pinv_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 4);

  // Zero matrix pseudoinverse should be zero
  for (float val : output_data) {
    BOOST_CHECK_SMALL(val, FLOAT_TOLERANCE);
  }

  BOOST_TEST_MESSAGE("Zero matrix pseudo-inverse flowgraph test passed.");
}

BOOST_AUTO_TEST_CASE(test_pseudo_inverse_sync_flowgraph_performance) {
  BOOST_TEST_MESSAGE(
      "Testing pseudo-inverse sync block performance in flowgraph...");

  auto tb = gr::make_top_block("pseudo_inverse_perf_test");

  // Create many 2×2 diagonal matrices for performance testing
  int num_matrices = 100;
  std::vector<float> input_data;
  input_data.reserve(num_matrices * 4);

  for (int i = 0; i < num_matrices; ++i) {
    float diag_val = 1.0f + 0.1f * i; // Varying diagonal values
    input_data.insert(input_data.end(), {diag_val, 0.0f, 0.0f, diag_val});
  }

  auto start_time = std::chrono::high_resolution_clock::now();

  auto vector_source = gr::blocks::vector_source_f::make(input_data, false, 4);
  auto pinv_block = matrix_pseudo_inverse_sync<float>::make({2, 2});
  auto output_sink = gr::blocks::vector_sink_f::make(4);

  tb->connect(vector_source, 0, pinv_block, 0);
  tb->connect(pinv_block, 0, output_sink, 0);

  tb->run();

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), num_matrices * 4);

  // Verify first matrix results
  BOOST_CHECK_CLOSE(output_data[0], 1.0f, FLOAT_TOLERANCE); // 1/1.0
  BOOST_CHECK_CLOSE(output_data[3], 1.0f, FLOAT_TOLERANCE); // 1/1.0
  BOOST_CHECK_SMALL(output_data[1], FLOAT_TOLERANCE);
  BOOST_CHECK_SMALL(output_data[2], FLOAT_TOLERANCE);

  // All should be finite
  for (size_t i = 0; i < output_data.size(); ++i) {
    BOOST_CHECK(std::isfinite(output_data[i]));
  }

  BOOST_TEST_MESSAGE("Processed " << num_matrices << " matrices in "
                                  << duration.count() << " ms");
  BOOST_TEST_MESSAGE("Pseudo-inverse performance flowgraph test passed.");
}

BOOST_AUTO_TEST_CASE(test_pseudo_inverse_sync_flowgraph_single_element) {
  BOOST_TEST_MESSAGE(
      "Testing pseudo-inverse sync block with single element in flowgraph...");

  auto tb = gr::make_top_block("pseudo_inverse_single_test");

  // Create 1×1 matrix with non-zero element
  std::vector<float> input_data = {4.0f};

  auto vector_source = gr::blocks::vector_source_f::make(input_data, false, 1);
  auto pinv_block = matrix_pseudo_inverse_sync<float>::make({1, 1});
  auto output_sink = gr::blocks::vector_sink_f::make(1);

  tb->connect(vector_source, 0, pinv_block, 0);
  tb->connect(pinv_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 1);

  // Single element pseudoinverse should be reciprocal
  BOOST_CHECK_CLOSE(output_data[0], 0.25f, FLOAT_TOLERANCE); // 1/4

  BOOST_TEST_MESSAGE("Single element pseudo-inverse flowgraph test passed.");
}

BOOST_AUTO_TEST_SUITE_END()

} /* namespace linalg */
} /* namespace gr */
