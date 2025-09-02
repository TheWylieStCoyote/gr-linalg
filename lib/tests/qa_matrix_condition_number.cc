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
#include <chrono>
#include <cmath>
#include <complex>
#include <gnuradio/blocks/vector_sink.h>
#include <gnuradio/blocks/vector_source.h>
#include <gnuradio/linalg/matrix_condition_number.h>
#include <gnuradio/linalg/types.h>
#include <gnuradio/top_block.h>
#include <limits>

using namespace gr::linalg;

BOOST_AUTO_TEST_SUITE(test_matrix_condition_number)

// Constants for numerical comparisons
constexpr float FLOAT_TOLERANCE = 1e-5f;
constexpr double DOUBLE_TOLERANCE = 1e-12;

// Test identity matrix (perfectly conditioned)
BOOST_AUTO_TEST_CASE(test_condition_number_identity) {
  auto cond_op = std::make_shared<matrix_condition_number<double>>();

  // Create 3×3 identity matrix
  std::vector<double> input_matrix{
      1.0, 0.0, 0.0, // Column 0
      0.0, 1.0, 0.0, // Column 1
      0.0, 0.0, 1.0  // Column 2
  };
  std::vector<double> output(1); // Scalar output

  types::matrix_map_dynamic<double> matrix_map(input_matrix.data(), 3, 3);
  types::matrix_map_dynamic<double> result_map(output.data(), 1, 1);

  types::vector_const_matrix_map<double> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<double> *>(&matrix_map)};
  types::vector_matrix_map<double> outputs{&result_map};

  BOOST_CHECK_EQUAL(cond_op->operation(inputs, outputs),
                    OperationReturn::SUCCESS);

  // Identity matrix should have condition number ≈ 1
  BOOST_CHECK_CLOSE(output[0], 1.0, DOUBLE_TOLERANCE * 100);
}

// Test diagonal matrix with known condition number
BOOST_AUTO_TEST_CASE(test_condition_number_diagonal) {
  auto cond_op = std::make_shared<matrix_condition_number<float>>();

  // Create diagonal matrix with eigenvalues [10, 5, 1]
  // Condition number should be 10/1 = 10
  std::vector<float> input_matrix{
      10.0f, 0.0f, 0.0f, // Column 0
      0.0f,  5.0f, 0.0f, // Column 1
      0.0f,  0.0f, 1.0f  // Column 2
  };
  std::vector<float> output(1);

  types::matrix_map_dynamic<float> matrix_map(input_matrix.data(), 3, 3);
  types::matrix_map_dynamic<float> result_map(output.data(), 1, 1);

  types::vector_const_matrix_map<float> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&matrix_map)};
  types::vector_matrix_map<float> outputs{&result_map};

  BOOST_CHECK_EQUAL(cond_op->operation(inputs, outputs),
                    OperationReturn::SUCCESS);

  // Condition number should be 10/1 = 10
  BOOST_CHECK_CLOSE(output[0], 10.0f, FLOAT_TOLERANCE);
}

// Test singular matrix (infinite condition number)
BOOST_AUTO_TEST_CASE(test_condition_number_singular) {
  auto cond_op = std::make_shared<matrix_condition_number<double>>();

  // Create singular matrix (rank deficient)
  // Row 2 = Row 1, so matrix is not full rank
  std::vector<double> input_matrix{
      1.0, 2.0, 3.0, // Column 0
      4.0, 5.0, 6.0, // Column 1
      1.0, 2.0, 3.0  // Column 2 (same as column 0)
  };
  std::vector<double> output(1);

  types::matrix_map_dynamic<double> matrix_map(input_matrix.data(), 3, 3);
  types::matrix_map_dynamic<double> result_map(output.data(), 1, 1);

  types::vector_const_matrix_map<double> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<double> *>(&matrix_map)};
  types::vector_matrix_map<double> outputs{&result_map};

  BOOST_CHECK_EQUAL(cond_op->operation(inputs, outputs),
                    OperationReturn::SUCCESS);

  // Singular matrix should have infinite condition number
  BOOST_CHECK(std::isinf(output[0]));
}

// Test zero matrix
BOOST_AUTO_TEST_CASE(test_condition_number_zero_matrix) {
  auto cond_op = std::make_shared<matrix_condition_number<float>>();

  // Create zero matrix
  std::vector<float> input_matrix(4, 0.0f); // 2×2 zero matrix
  std::vector<float> output(1);

  types::matrix_map_dynamic<float> matrix_map(input_matrix.data(), 2, 2);
  types::matrix_map_dynamic<float> result_map(output.data(), 1, 1);

  types::vector_const_matrix_map<float> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&matrix_map)};
  types::vector_matrix_map<float> outputs{&result_map};

  BOOST_CHECK_EQUAL(cond_op->operation(inputs, outputs),
                    OperationReturn::SUCCESS);

  // Zero matrix should have infinite condition number
  BOOST_CHECK(std::isinf(output[0]));
}

// Test single element matrix
BOOST_AUTO_TEST_CASE(test_condition_number_single_element) {
  auto cond_op = std::make_shared<matrix_condition_number<double>>();

  // Single non-zero element
  std::vector<double> input_matrix{5.0};
  std::vector<double> output(1);

  types::matrix_map_dynamic<double> matrix_map(input_matrix.data(), 1, 1);
  types::matrix_map_dynamic<double> result_map(output.data(), 1, 1);

  types::vector_const_matrix_map<double> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<double> *>(&matrix_map)};
  types::vector_matrix_map<double> outputs{&result_map};

  BOOST_CHECK_EQUAL(cond_op->operation(inputs, outputs),
                    OperationReturn::SUCCESS);

  // Single non-zero element should have condition number 1
  BOOST_CHECK_CLOSE(output[0], 1.0, DOUBLE_TOLERANCE * 100);
}

// Test single zero element
BOOST_AUTO_TEST_CASE(test_condition_number_single_zero) {
  auto cond_op = std::make_shared<matrix_condition_number<double>>();

  // Single zero element
  std::vector<double> input_matrix{0.0};
  std::vector<double> output(1);

  types::matrix_map_dynamic<double> matrix_map(input_matrix.data(), 1, 1);
  types::matrix_map_dynamic<double> result_map(output.data(), 1, 1);

  types::vector_const_matrix_map<double> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<double> *>(&matrix_map)};
  types::vector_matrix_map<double> outputs{&result_map};

  BOOST_CHECK_EQUAL(cond_op->operation(inputs, outputs),
                    OperationReturn::SUCCESS);

  // Single zero element should have infinite condition number
  BOOST_CHECK(std::isinf(output[0]));
}

// Test rectangular matrix
BOOST_AUTO_TEST_CASE(test_condition_number_rectangular) {
  auto cond_op = std::make_shared<matrix_condition_number<float>>();

  // Create 2×3 matrix with full rank
  std::vector<float> input_matrix{
      1.0f, 0.0f, // Column 0: [1, 0]
      0.0f, 2.0f, // Column 1: [0, 2]
      1.0f, 1.0f  // Column 2: [1, 1]
  };
  std::vector<float> output(1);

  types::matrix_map_dynamic<float> matrix_map(input_matrix.data(), 2, 3);
  types::matrix_map_dynamic<float> result_map(output.data(), 1, 1);

  types::vector_const_matrix_map<float> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&matrix_map)};
  types::vector_matrix_map<float> outputs{&result_map};

  BOOST_CHECK_EQUAL(cond_op->operation(inputs, outputs),
                    OperationReturn::SUCCESS);

  // This matrix should have a finite condition number
  BOOST_CHECK(std::isfinite(output[0]));
  BOOST_CHECK_GT(output[0], 1.0f); // Should be > 1
}

// Test complex matrix
BOOST_AUTO_TEST_CASE(test_condition_number_complex) {
  auto cond_op =
      std::make_shared<matrix_condition_number<std::complex<double>>>();

  // Create 2×2 complex identity matrix (well-conditioned)
  std::vector<std::complex<double>> input_matrix{
      std::complex<double>(1.0, 0.0), // (0,0) = 1+0i
      std::complex<double>(0.0, 0.0), // (1,0) = 0+0i
      std::complex<double>(0.0, 0.0), // (0,1) = 0+0i
      std::complex<double>(1.0, 0.0)  // (1,1) = 1+0i
  };
  std::vector<std::complex<double>> output(1);

  types::matrix_map_dynamic<std::complex<double>> matrix_map(
      input_matrix.data(), 2, 2);
  types::matrix_map_dynamic<std::complex<double>> result_map(output.data(), 1,
                                                             1);

  types::vector_const_matrix_map<std::complex<double>> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<std::complex<double>> *>(
          &matrix_map)};
  types::vector_matrix_map<std::complex<double>> outputs{&result_map};

  BOOST_CHECK_EQUAL(cond_op->operation(inputs, outputs),
                    OperationReturn::SUCCESS);

  // Complex identity matrix should have condition number ≈ 1
  // Output should be real (complex condition numbers are converted to real)
  BOOST_CHECK_CLOSE(output[0].real(), 1.0, DOUBLE_TOLERANCE * 100);
  BOOST_CHECK_SMALL(output[0].imag(), DOUBLE_TOLERANCE);
}

// Test ill-conditioned matrix
BOOST_AUTO_TEST_CASE(test_condition_number_ill_conditioned) {
  auto cond_op = std::make_shared<matrix_condition_number<double>>();

  // Create Hilbert matrix (known to be ill-conditioned)
  // H(i,j) = 1/(i+j+1) for i,j = 0,1,2
  std::vector<double> input_matrix{
      1.0,       1.0 / 2.0, 1.0 / 3.0, // Column 0: [1, 1/2, 1/3]
      1.0 / 2.0, 1.0 / 3.0, 1.0 / 4.0, // Column 1: [1/2, 1/3, 1/4]
      1.0 / 3.0, 1.0 / 4.0, 1.0 / 5.0  // Column 2: [1/3, 1/4, 1/5]
  };
  std::vector<double> output(1);

  types::matrix_map_dynamic<double> matrix_map(input_matrix.data(), 3, 3);
  types::matrix_map_dynamic<double> result_map(output.data(), 1, 1);

  types::vector_const_matrix_map<double> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<double> *>(&matrix_map)};
  types::vector_matrix_map<double> outputs{&result_map};

  BOOST_CHECK_EQUAL(cond_op->operation(inputs, outputs),
                    OperationReturn::SUCCESS);

  // 3×3 Hilbert matrix has known condition number around 524
  BOOST_CHECK_GT(output[0], 500.0);  // Should be > 500
  BOOST_CHECK_LT(output[0], 1000.0); // Should be < 1000
}

// Test error handling
BOOST_AUTO_TEST_CASE(test_condition_number_error_handling) {
  auto cond_op = std::make_shared<matrix_condition_number<float>>();

  std::vector<float> input_matrix{1.0f, 2.0f};
  std::vector<float> output(1);

  types::matrix_map_dynamic<float> matrix_map(input_matrix.data(), 1, 2);
  types::matrix_map_dynamic<float> result_map(output.data(), 1, 1);

  // Test insufficient inputs
  types::vector_const_matrix_map<float> empty_inputs{};
  types::vector_matrix_map<float> outputs{&result_map};

  BOOST_CHECK_EQUAL(cond_op->operation(empty_inputs, outputs),
                    OperationReturn::FAILURE);

  // Test insufficient outputs
  types::vector_const_matrix_map<float> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&matrix_map)};
  types::vector_matrix_map<float> empty_outputs{};

  BOOST_CHECK_EQUAL(cond_op->operation(inputs, empty_outputs),
                    OperationReturn::FAILURE);
}

// Test output shape validation
BOOST_AUTO_TEST_CASE(test_condition_number_output_shape) {
  auto cond_op = std::make_shared<matrix_condition_number<double>>();

  std::vector<double> input_matrix{1.0, 2.0, 3.0, 4.0}; // 2×2 matrix
  std::vector<double> output(4); // Wrong size: 2×2 instead of 1×1

  types::matrix_map_dynamic<double> matrix_map(input_matrix.data(), 2, 2);
  types::matrix_map_dynamic<double> result_map(output.data(), 2,
                                               2); // Wrong dimensions

  types::vector_const_matrix_map<double> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<double> *>(&matrix_map)};
  types::vector_matrix_map<double> outputs{&result_map};

  // Should fail due to output dimension mismatch
  BOOST_CHECK_EQUAL(cond_op->operation(inputs, outputs),
                    OperationReturn::INVALID_SHAPE);
}

// Test static helper functions
BOOST_AUTO_TEST_CASE(test_condition_number_static_methods) {
  // Test compute_output_shapes
  types::vector_shapes input_shapes = {{3, 3}}; // 3×3 matrix

  auto output_shapes =
      matrix_condition_number<float>::compute_output_shapes(input_shapes);

  BOOST_REQUIRE_EQUAL(output_shapes.size(), 1);
  BOOST_CHECK_EQUAL(output_shapes[0][0], 1); // 1 row
  BOOST_CHECK_EQUAL(output_shapes[0][1], 1); // 1 column

  // Test extract_real_value for complex numbers
  std::complex<double> complex_val(3.0, 4.0); // Magnitude = 5.0
  auto real_val =
      matrix_condition_number<std::complex<double>>::extract_real_value(
          complex_val);
  BOOST_CHECK_CLOSE(real_val, 5.0, DOUBLE_TOLERANCE);

  // Test is_effectively_zero
  BOOST_CHECK(
      matrix_condition_number<double>::is_effectively_zero(1e-16, 1e-15));
  BOOST_CHECK(
      !matrix_condition_number<double>::is_effectively_zero(1e-14, 1e-15));
}

// Test empty matrix handling
BOOST_AUTO_TEST_CASE(test_condition_number_empty_matrix) {
  auto cond_op = std::make_shared<matrix_condition_number<double>>();

  // Create empty matrix (0×0)
  std::vector<double> input_matrix{};
  std::vector<double> output(1);

  types::matrix_map_dynamic<double> matrix_map(nullptr, 0, 0);
  types::matrix_map_dynamic<double> result_map(output.data(), 1, 1);

  types::vector_const_matrix_map<double> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<double> *>(&matrix_map)};
  types::vector_matrix_map<double> outputs{&result_map};

  BOOST_CHECK_EQUAL(cond_op->operation(inputs, outputs),
                    OperationReturn::SUCCESS);

  // Empty matrix should have infinite condition number
  BOOST_CHECK(std::isinf(output[0]));
}

// GNU Radio Flowgraph Tests
BOOST_AUTO_TEST_CASE(test_condition_number_sync_flowgraph) {
  BOOST_TEST_MESSAGE(
      "Testing matrix condition number sync block in GNU Radio flowgraph...");

  auto tb = gr::make_top_block("condition_number_test");

  // Create well-conditioned matrix: [[3, 1], [1, 3]] (condition number ≈ 2)
  // In column-major format: [3, 1, 1, 3]
  std::vector<float> input_data = {3.0f, 1.0f, 1.0f, 3.0f};

  auto vector_source = gr::blocks::vector_source_f::make(input_data, false, 4);
  auto condition_block = matrix_condition_number_sync<float>::make({2, 2});
  auto output_sink = gr::blocks::vector_sink_f::make(1); // Scalar output

  tb->connect(vector_source, 0, condition_block, 0);
  tb->connect(condition_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 1);

  // Condition number should be approximately 2 for this symmetric matrix
  BOOST_CHECK_GT(output_data[0], 1.5f);
  BOOST_CHECK_LT(output_data[0], 2.5f);
  BOOST_CHECK(std::isfinite(output_data[0]));

  BOOST_TEST_MESSAGE("Matrix condition number sync flowgraph test passed.");
}

BOOST_AUTO_TEST_CASE(test_condition_number_sync_flowgraph_identity) {
  BOOST_TEST_MESSAGE("Testing condition number sync block with identity matrix "
                     "in flowgraph...");

  auto tb = gr::make_top_block("condition_identity_test");

  // Create 3x3 identity matrix in column-major format
  std::vector<float> input_data = {
      1.0f, 0.0f, 0.0f, // Column 0: [1, 0, 0]
      0.0f, 1.0f, 0.0f, // Column 1: [0, 1, 0]
      0.0f, 0.0f, 1.0f  // Column 2: [0, 0, 1]
  };

  auto vector_source = gr::blocks::vector_source_f::make(input_data, false, 9);
  auto condition_block = matrix_condition_number_sync<float>::make({3, 3});
  auto output_sink = gr::blocks::vector_sink_f::make(1);

  tb->connect(vector_source, 0, condition_block, 0);
  tb->connect(condition_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 1);

  // Identity matrix should have condition number = 1
  BOOST_CHECK_CLOSE(output_data[0], 1.0f, 1e-4f);

  BOOST_TEST_MESSAGE("Identity matrix condition number flowgraph test passed.");
}

BOOST_AUTO_TEST_CASE(test_condition_number_sync_flowgraph_diagonal) {
  BOOST_TEST_MESSAGE("Testing condition number sync block with diagonal matrix "
                     "in flowgraph...");

  auto tb = gr::make_top_block("condition_diagonal_test");

  // Create diagonal matrix with eigenvalues [10, 5, 1] => condition number = 10
  std::vector<float> input_data = {
      10.0f, 0.0f, 0.0f, // Column 0: [10, 0, 0]
      0.0f,  5.0f, 0.0f, // Column 1: [0, 5, 0]
      0.0f,  0.0f, 1.0f  // Column 2: [0, 0, 1]
  };

  auto vector_source = gr::blocks::vector_source_f::make(input_data, false, 9);
  auto condition_block = matrix_condition_number_sync<float>::make({3, 3});
  auto output_sink = gr::blocks::vector_sink_f::make(1);

  tb->connect(vector_source, 0, condition_block, 0);
  tb->connect(condition_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 1);

  // Condition number should be 10/1 = 10
  BOOST_CHECK_CLOSE(output_data[0], 10.0f, 1e-3f);

  BOOST_TEST_MESSAGE("Diagonal matrix condition number flowgraph test passed.");
}

BOOST_AUTO_TEST_CASE(test_condition_number_sync_flowgraph_singular) {
  BOOST_TEST_MESSAGE("Testing condition number sync block with singular matrix "
                     "in flowgraph...");

  auto tb = gr::make_top_block("condition_singular_test");

  // Create singular matrix: [[1, 2], [2, 4]] (rank 1, infinite condition
  // number)
  std::vector<float> input_data = {1.0f, 2.0f, 2.0f, 4.0f};

  auto vector_source = gr::blocks::vector_source_f::make(input_data, false, 4);
  auto condition_block = matrix_condition_number_sync<float>::make({2, 2});
  auto output_sink = gr::blocks::vector_sink_f::make(1);

  tb->connect(vector_source, 0, condition_block, 0);
  tb->connect(condition_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 1);

  // Singular matrix should have infinite condition number
  BOOST_CHECK(std::isinf(output_data[0]));

  BOOST_TEST_MESSAGE("Singular matrix condition number flowgraph test passed.");
}

BOOST_AUTO_TEST_CASE(test_condition_number_sync_flowgraph_multiple_matrices) {
  BOOST_TEST_MESSAGE("Testing condition number sync block with multiple "
                     "matrices in flowgraph...");

  auto tb = gr::make_top_block("condition_multi_test");

  // Create multiple 2x2 matrices with different condition numbers
  std::vector<float> input_data = {
      // Matrix 1: Identity [[1, 0], [0, 1]] => cond = 1
      1.0f, 0.0f, 0.0f, 1.0f,
      // Matrix 2: Diagonal [[2, 0], [0, 1]] => cond = 2
      2.0f, 0.0f, 0.0f, 1.0f,
      // Matrix 3: Well-conditioned [[3, 1], [1, 3]] => cond ≈ 2
      3.0f, 1.0f, 1.0f, 3.0f};

  auto vector_source = gr::blocks::vector_source_f::make(input_data, false, 4);
  auto condition_block = matrix_condition_number_sync<float>::make({2, 2});
  auto output_sink = gr::blocks::vector_sink_f::make(1);

  tb->connect(vector_source, 0, condition_block, 0);
  tb->connect(condition_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 3); // 3 matrices processed

  // Verify condition numbers
  BOOST_CHECK_CLOSE(output_data[0], 1.0f, 1e-3f); // Identity: cond = 1
  BOOST_CHECK_CLOSE(output_data[1], 2.0f, 1e-3f); // Diagonal: cond = 2
  BOOST_CHECK_GT(output_data[2], 1.5f);           // Symmetric: cond ≈ 2
  BOOST_CHECK_LT(output_data[2], 2.5f);

  BOOST_TEST_MESSAGE(
      "Multiple matrices condition number flowgraph test passed.");
}

BOOST_AUTO_TEST_CASE(test_condition_number_sync_flowgraph_rectangular) {
  BOOST_TEST_MESSAGE("Testing condition number sync block with rectangular "
                     "matrix in flowgraph...");

  auto tb = gr::make_top_block("condition_rect_test");

  // Create 2x3 rectangular matrix: [[1, 0, 1], [0, 2, 1]]
  std::vector<float> input_data = {
      1.0f, 0.0f, // Column 0: [1, 0]
      0.0f, 2.0f, // Column 1: [0, 2]
      1.0f, 1.0f  // Column 2: [1, 1]
  };

  auto vector_source = gr::blocks::vector_source_f::make(input_data, false, 6);
  auto condition_block = matrix_condition_number_sync<float>::make({2, 3});
  auto output_sink = gr::blocks::vector_sink_f::make(1);

  tb->connect(vector_source, 0, condition_block, 0);
  tb->connect(condition_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 1);

  // Rectangular matrix should have finite condition number
  BOOST_CHECK(std::isfinite(output_data[0]));
  BOOST_CHECK_GT(output_data[0], 1.0f);   // Should be > 1
  BOOST_CHECK_LT(output_data[0], 100.0f); // Should be reasonable

  BOOST_TEST_MESSAGE(
      "Rectangular matrix condition number flowgraph test passed.");
}

BOOST_AUTO_TEST_CASE(test_condition_number_sync_flowgraph_ill_conditioned) {
  BOOST_TEST_MESSAGE("Testing condition number sync block with ill-conditioned "
                     "matrix in flowgraph...");

  auto tb = gr::make_top_block("condition_ill_test");

  // Create 3x3 Hilbert matrix (known to be ill-conditioned)
  // H(i,j) = 1/(i+j+1)
  std::vector<float> input_data = {
      1.0f,        1.0f / 2.0f, 1.0f / 3.0f, // Column 0: [1, 1/2, 1/3]
      1.0f / 2.0f, 1.0f / 3.0f, 1.0f / 4.0f, // Column 1: [1/2, 1/3, 1/4]
      1.0f / 3.0f, 1.0f / 4.0f, 1.0f / 5.0f  // Column 2: [1/3, 1/4, 1/5]
  };

  auto vector_source = gr::blocks::vector_source_f::make(input_data, false, 9);
  auto condition_block = matrix_condition_number_sync<float>::make({3, 3});
  auto output_sink = gr::blocks::vector_sink_f::make(1);

  tb->connect(vector_source, 0, condition_block, 0);
  tb->connect(condition_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 1);

  // 3x3 Hilbert matrix has condition number around 524
  BOOST_CHECK_GT(output_data[0], 400.0f);  // Should be > 400
  BOOST_CHECK_LT(output_data[0], 1000.0f); // Should be < 1000
  BOOST_CHECK(std::isfinite(output_data[0]));

  BOOST_TEST_MESSAGE(
      "Ill-conditioned matrix condition number flowgraph test passed.");
}

BOOST_AUTO_TEST_CASE(test_condition_number_sync_flowgraph_performance) {
  BOOST_TEST_MESSAGE(
      "Testing condition number sync block performance in flowgraph...");

  auto tb = gr::make_top_block("condition_perf_test");

  // Create many 2x2 identity matrices for performance testing
  int num_matrices = 1000;
  std::vector<float> input_data;
  input_data.reserve(num_matrices * 4);

  for (int i = 0; i < num_matrices; ++i) {
    // Identity matrix with slight perturbation
    float eps = 0.001f * i;
    input_data.insert(input_data.end(), {1.0f + eps, 0.0f, 0.0f, 1.0f + eps});
  }

  auto start_time = std::chrono::high_resolution_clock::now();

  auto vector_source = gr::blocks::vector_source_f::make(input_data, false, 4);
  auto condition_block = matrix_condition_number_sync<float>::make({2, 2});
  auto output_sink = gr::blocks::vector_sink_f::make(1);

  tb->connect(vector_source, 0, condition_block, 0);
  tb->connect(condition_block, 0, output_sink, 0);

  tb->run();

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), num_matrices);

  // Verify first and last condition numbers
  BOOST_CHECK_CLOSE(output_data[0], 1.0f, 1e-3f);                // First matrix
  BOOST_CHECK_CLOSE(output_data[num_matrices - 1], 1.0f, 1e-2f); // Last matrix

  // All should be finite and close to 1
  for (int i = 0; i < num_matrices; ++i) {
    BOOST_CHECK(std::isfinite(output_data[i]));
    BOOST_CHECK_GT(output_data[i], 0.9f);
    BOOST_CHECK_LT(output_data[i], 1.1f);
  }

  BOOST_TEST_MESSAGE("Processed " << num_matrices << " matrices in "
                                  << duration.count() << " ms");
  BOOST_TEST_MESSAGE("Condition number performance flowgraph test passed.");
}

BOOST_AUTO_TEST_CASE(test_condition_number_sync_flowgraph_single_element) {
  BOOST_TEST_MESSAGE("Testing condition number sync block with single element "
                     "in flowgraph...");

  auto tb = gr::make_top_block("condition_single_test");

  // Create 1x1 matrix with non-zero element
  std::vector<float> input_data = {5.0f};

  auto vector_source = gr::blocks::vector_source_f::make(input_data, false, 1);
  auto condition_block = matrix_condition_number_sync<float>::make({1, 1});
  auto output_sink = gr::blocks::vector_sink_f::make(1);

  tb->connect(vector_source, 0, condition_block, 0);
  tb->connect(condition_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 1);

  // Single non-zero element should have condition number = 1
  BOOST_CHECK_CLOSE(output_data[0], 1.0f, 1e-5f);

  BOOST_TEST_MESSAGE("Single element condition number flowgraph test passed.");
}

BOOST_AUTO_TEST_SUITE_END()