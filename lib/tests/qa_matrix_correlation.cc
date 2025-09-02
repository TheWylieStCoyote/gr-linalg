/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <boost/test/unit_test.hpp>
#include <cmath>
#include <complex>
#include <gnuradio/attributes.h>
#include <gnuradio/blocks/vector_sink.h>
#include <gnuradio/blocks/vector_source.h>
#include <gnuradio/linalg/matrix_correlation.h>
#include <gnuradio/top_block.h>
#include <vector>

namespace gr {
namespace linalg {

//==============================================================================
// Static method tests
//==============================================================================

BOOST_AUTO_TEST_CASE(test_matrix_correlation_validate_shape_valid) {
  // Valid configuration: 3 samples, 2 features -> 2x2 correlation matrix
  types::vector_shapes input_shapes = {{3, 2}};
  types::vector_shapes output_shapes = {{2, 2}};

  BOOST_CHECK_NO_THROW(
      matrix_correlation<float>::validate_shape(input_shapes, output_shapes));
}

BOOST_AUTO_TEST_CASE(test_matrix_correlation_validate_shape_invalid_inputs) {
  // Test various invalid input configurations

  // No inputs
  types::vector_shapes empty_inputs = {};
  types::vector_shapes valid_outputs = {{2, 2}};
  BOOST_CHECK_THROW(
      matrix_correlation<float>::validate_shape(empty_inputs, valid_outputs),
      std::invalid_argument);

  // Multiple inputs
  types::vector_shapes multiple_inputs = {{3, 2}, {3, 2}};
  BOOST_CHECK_THROW(
      matrix_correlation<float>::validate_shape(multiple_inputs, valid_outputs),
      std::invalid_argument);

  // 1D input matrix
  types::vector_shapes input_1d = {{6}};
  BOOST_CHECK_THROW(
      matrix_correlation<float>::validate_shape(input_1d, valid_outputs),
      std::invalid_argument);

  // Too few samples (need at least 2 for correlation)
  types::vector_shapes input_few_samples = {{1, 2}};
  BOOST_CHECK_THROW(matrix_correlation<float>::validate_shape(input_few_samples,
                                                              valid_outputs),
                    std::invalid_argument);

  // Zero features
  types::vector_shapes input_zero_features = {{3, 0}};
  BOOST_CHECK_THROW(matrix_correlation<float>::validate_shape(
                        input_zero_features, valid_outputs),
                    std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_matrix_correlation_validate_shape_invalid_outputs) {
  types::vector_shapes valid_inputs = {{3, 2}};

  // Multiple outputs
  types::vector_shapes multiple_outputs = {{2, 2}, {2, 2}};
  BOOST_CHECK_THROW(
      matrix_correlation<float>::validate_shape(valid_inputs, multiple_outputs),
      std::invalid_argument);

  // 1D output
  types::vector_shapes output_1d = {{4}};
  BOOST_CHECK_THROW(
      matrix_correlation<float>::validate_shape(valid_inputs, output_1d),
      std::invalid_argument);

  // Wrong output dimensions
  types::vector_shapes output_wrong_size = {{3, 3}};
  BOOST_CHECK_THROW(matrix_correlation<float>::validate_shape(
                        valid_inputs, output_wrong_size),
                    std::invalid_argument);

  // Non-square output
  types::vector_shapes output_nonsquare = {{2, 3}};
  BOOST_CHECK_THROW(
      matrix_correlation<float>::validate_shape(valid_inputs, output_nonsquare),
      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_matrix_correlation_compute_output_shapes) {
  // Test with 4 samples, 3 features -> should produce 3x3 correlation matrix
  types::vector_shapes input_shapes = {{4, 3}};
  types::vector_shapes output_shapes =
      matrix_correlation<float>::compute_output_shapes(input_shapes);

  BOOST_REQUIRE_EQUAL(output_shapes.size(), 1);
  BOOST_REQUIRE_EQUAL(output_shapes[0].size(), 2);
  BOOST_CHECK_EQUAL(output_shapes[0][0], 3);
  BOOST_CHECK_EQUAL(output_shapes[0][1], 3);

  // Test with different dimensions
  input_shapes = {{10, 5}};
  output_shapes =
      matrix_correlation<float>::compute_output_shapes(input_shapes);

  BOOST_REQUIRE_EQUAL(output_shapes.size(), 1);
  BOOST_REQUIRE_EQUAL(output_shapes[0].size(), 2);
  BOOST_CHECK_EQUAL(output_shapes[0][0], 5);
  BOOST_CHECK_EQUAL(output_shapes[0][1], 5);
}

BOOST_AUTO_TEST_CASE(test_matrix_correlation_compute_output_shapes_invalid) {
  // Test with no inputs
  types::vector_shapes empty_inputs = {};
  BOOST_CHECK_THROW(
      matrix_correlation<float>::compute_output_shapes(empty_inputs),
      std::invalid_argument);

  // Test with 1D input
  types::vector_shapes input_1d = {{6}};
  BOOST_CHECK_THROW(matrix_correlation<float>::compute_output_shapes(input_1d),
                    std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_matrix_correlation_compute_sizes) {
  types::vector_shapes shapes = {{2, 3}, {3, 3}};
  std::vector<size_t> sizes = matrix_correlation<float>::compute_sizes(shapes);

  BOOST_REQUIRE_EQUAL(sizes.size(), 2);
  BOOST_CHECK_EQUAL(sizes[0], 2 * 3 * sizeof(float)); // 2x3 matrix
  BOOST_CHECK_EQUAL(sizes[1], 3 * 3 * sizeof(float)); // 3x3 matrix
}

//==============================================================================
// Operation method tests
//==============================================================================

BOOST_AUTO_TEST_CASE(test_matrix_correlation_pearson_identity) {
  // Test Pearson correlation with perfectly correlated data
  // Input: [[1, 2], [2, 4], [3, 6]] - second column is 2x first column
  types::vector_shapes input_shapes = {{3, 2}};
  types::vector_shapes output_shapes = {{2, 2}};

  auto correlation_block =
      matrix_correlation_sync<float>::make({3, 2}, 0); // Pearson

  // Create input data in column-major order: [1, 2, 3, 2, 4, 6]
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 2.0f, 4.0f, 6.0f};

  // Create input and output matrices
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      input_matrix(input_data.data(), 3, 2);
  std::vector<float> output_data(4, 0.0f);
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      output_matrix(output_data.data(), 2, 2);

  types::vector_const_matrix_map<float> input_matrices = {&input_matrix};
  types::vector_matrix_map<float> output_matrices = {&output_matrix};

  OperationReturn result =
      correlation_block->operation(input_matrices, output_matrices);

  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Check diagonal elements (should be 1.0)
  BOOST_CHECK_CLOSE(output_matrix(0, 0), 1.0f, 1e-5);
  BOOST_CHECK_CLOSE(output_matrix(1, 1), 1.0f, 1e-5);

  // Check off-diagonal elements (should be 1.0 for perfect correlation)
  BOOST_CHECK_CLOSE(output_matrix(0, 1), 1.0f, 1e-5);
  BOOST_CHECK_CLOSE(output_matrix(1, 0), 1.0f, 1e-5);
}

BOOST_AUTO_TEST_CASE(test_matrix_correlation_pearson_anticorrelated) {
  // Test with anti-correlated data
  // Input: [[1, 3], [2, 2], [3, 1]] - second column decreases as first
  // increases
  types::vector_shapes input_shapes = {{3, 2}};
  types::vector_shapes output_shapes = {{2, 2}};

  auto correlation_block =
      matrix_correlation_sync<float>::make({3, 2}, 0); // Pearson

  // Create input data in column-major order: [1, 2, 3, 3, 2, 1]
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 3.0f, 2.0f, 1.0f};

  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      input_matrix(input_data.data(), 3, 2);
  std::vector<float> output_data(4, 0.0f);
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      output_matrix(output_data.data(), 2, 2);

  types::vector_const_matrix_map<float> input_matrices = {&input_matrix};
  types::vector_matrix_map<float> output_matrices = {&output_matrix};

  OperationReturn result =
      correlation_block->operation(input_matrices, output_matrices);

  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Check diagonal elements (should be 1.0)
  BOOST_CHECK_CLOSE(output_matrix(0, 0), 1.0f, 1e-5);
  BOOST_CHECK_CLOSE(output_matrix(1, 1), 1.0f, 1e-5);

  // Check off-diagonal elements (should be -1.0 for perfect anti-correlation)
  BOOST_CHECK_CLOSE(output_matrix(0, 1), -1.0f, 1e-5);
  BOOST_CHECK_CLOSE(output_matrix(1, 0), -1.0f, 1e-5);
}

BOOST_AUTO_TEST_CASE(test_matrix_correlation_spearman) {
  // Test Spearman rank correlation
  // Use data where Pearson and Spearman differ
  types::vector_shapes input_shapes = {{4, 2}};

  auto correlation_block =
      matrix_correlation_sync<float>::make({4, 2}, 1); // Spearman

  // Create input data: [[1, 1], [2, 4], [3, 9], [4, 16]]
  // First column: linear, second column: quadratic
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                   1.0f, 4.0f, 9.0f, 16.0f};

  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      input_matrix(input_data.data(), 4, 2);
  std::vector<float> output_data(4, 0.0f);
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      output_matrix(output_data.data(), 2, 2);

  types::vector_const_matrix_map<float> input_matrices = {&input_matrix};
  types::vector_matrix_map<float> output_matrices = {&output_matrix};

  OperationReturn result =
      correlation_block->operation(input_matrices, output_matrices);

  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Check diagonal elements (should be 1.0)
  BOOST_CHECK_CLOSE(output_matrix(0, 0), 1.0f, 1e-5);
  BOOST_CHECK_CLOSE(output_matrix(1, 1), 1.0f, 1e-5);

  // For monotonic data, Spearman correlation should be 1.0
  BOOST_CHECK_CLOSE(output_matrix(0, 1), 1.0f, 1e-5);
  BOOST_CHECK_CLOSE(output_matrix(1, 0), 1.0f, 1e-5);
}

BOOST_AUTO_TEST_CASE(test_matrix_correlation_kendall) {
  // Test Kendall tau correlation
  types::vector_shapes input_shapes = {{4, 2}};

  auto correlation_block =
      matrix_correlation_sync<float>::make({4, 2}, 2); // Kendall

  // Create simple monotonic data
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                   1.0f, 2.0f, 3.0f, 4.0f};

  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      input_matrix(input_data.data(), 4, 2);
  std::vector<float> output_data(4, 0.0f);
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      output_matrix(output_data.data(), 2, 2);

  types::vector_const_matrix_map<float> input_matrices = {&input_matrix};
  types::vector_matrix_map<float> output_matrices = {&output_matrix};

  OperationReturn result =
      correlation_block->operation(input_matrices, output_matrices);

  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Check diagonal elements (should be 1.0)
  BOOST_CHECK_CLOSE(output_matrix(0, 0), 1.0f, 1e-5);
  BOOST_CHECK_CLOSE(output_matrix(1, 1), 1.0f, 1e-5);

  // For identical data, Kendall tau should be 1.0
  BOOST_CHECK_CLOSE(output_matrix(0, 1), 1.0f, 1e-5);
  BOOST_CHECK_CLOSE(output_matrix(1, 0), 1.0f, 1e-5);
}

//==============================================================================
// Error condition tests
//==============================================================================

BOOST_AUTO_TEST_CASE(test_matrix_correlation_invalid_inputs) {
  auto correlation_block = matrix_correlation_sync<float>::make({3, 2}, 0);

  // Test with no input matrices
  types::vector_const_matrix_map<float> empty_inputs = {};
  std::vector<float> output_data(4, 0.0f);
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      output_matrix(output_data.data(), 2, 2);
  types::vector_matrix_map<float> output_matrices = {&output_matrix};

  OperationReturn result =
      correlation_block->operation(empty_inputs, output_matrices);
  BOOST_CHECK_EQUAL(result, OperationReturn::ERROR_INVALID_INPUT);

  // Test with no output matrices
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 2.0f, 4.0f, 6.0f};
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      input_matrix(input_data.data(), 3, 2);
  types::vector_const_matrix_map<float> input_matrices = {&input_matrix};
  types::vector_matrix_map<float> empty_outputs = {};

  result = correlation_block->operation(input_matrices, empty_outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::ERROR_INVALID_OUTPUT);
}

BOOST_AUTO_TEST_CASE(test_matrix_correlation_insufficient_samples) {
  auto correlation_block = matrix_correlation_sync<float>::make({2, 2}, 0);

  // Test with only 1 sample (need at least 2 for correlation)
  std::vector<float> input_data = {1.0f, 2.0f};
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      input_matrix(input_data.data(), 1, 2);
  std::vector<float> output_data(4, 0.0f);
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      output_matrix(output_data.data(), 2, 2);

  types::vector_const_matrix_map<float> input_matrices = {&input_matrix};
  types::vector_matrix_map<float> output_matrices = {&output_matrix};

  OperationReturn result =
      correlation_block->operation(input_matrices, output_matrices);
  BOOST_CHECK_EQUAL(result, OperationReturn::ERROR_INVALID_INPUT);
}

BOOST_AUTO_TEST_CASE(test_matrix_correlation_invalid_method) {
  auto correlation_block =
      matrix_correlation_sync<float>::make({3, 2}, 99); // Invalid method

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 2.0f, 4.0f, 6.0f};
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      input_matrix(input_data.data(), 3, 2);
  std::vector<float> output_data(4, 0.0f);
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      output_matrix(output_data.data(), 2, 2);

  types::vector_const_matrix_map<float> input_matrices = {&input_matrix};
  types::vector_matrix_map<float> output_matrices = {&output_matrix};

  OperationReturn result =
      correlation_block->operation(input_matrices, output_matrices);
  BOOST_CHECK_EQUAL(result, OperationReturn::ERROR_INVALID_INPUT);
}

//==============================================================================
// Method setter/getter tests
//==============================================================================

BOOST_AUTO_TEST_CASE(test_matrix_correlation_method_operations) {
  auto correlation_block = matrix_correlation_sync<float>::make({3, 2}, 0);

  // Test initial method
  BOOST_CHECK_EQUAL(correlation_block->get_method(), 0);

  // Test method setter
  correlation_block->set_method(1);
  BOOST_CHECK_EQUAL(correlation_block->get_method(), 1);

  correlation_block->set_method(2);
  BOOST_CHECK_EQUAL(correlation_block->get_method(), 2);
}

//==============================================================================
// Complex number support tests
//==============================================================================

BOOST_AUTO_TEST_CASE(test_matrix_correlation_complex_float) {
  typedef std::complex<float> complex_t;

  auto correlation_block = matrix_correlation_sync<complex_t>::make({3, 2}, 0);

  // Create complex input data
  std::vector<complex_t> input_data = {
      complex_t(1.0f, 0.0f), complex_t(2.0f, 0.0f), complex_t(3.0f, 0.0f),
      complex_t(1.0f, 1.0f), complex_t(2.0f, 2.0f), complex_t(3.0f, 3.0f)};

  Eigen::Map<const Eigen::Matrix<complex_t, Eigen::Dynamic, Eigen::Dynamic>>
      input_matrix(input_data.data(), 3, 2);
  std::vector<complex_t> output_data(4, complex_t(0.0f, 0.0f));
  Eigen::Map<Eigen::Matrix<complex_t, Eigen::Dynamic, Eigen::Dynamic>>
      output_matrix(output_data.data(), 2, 2);

  types::vector_const_matrix_map<complex_t> input_matrices = {&input_matrix};
  types::vector_matrix_map<complex_t> output_matrices = {&output_matrix};

  OperationReturn result =
      correlation_block->operation(input_matrices, output_matrices);

  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Check that diagonal elements are close to 1.0
  BOOST_CHECK(std::abs(output_matrix(0, 0) - complex_t(1.0f, 0.0f)) < 1e-5);
  BOOST_CHECK(std::abs(output_matrix(1, 1) - complex_t(1.0f, 0.0f)) < 1e-5);
}

BOOST_AUTO_TEST_CASE(test_matrix_correlation_complex_double) {
  typedef std::complex<double> complex_t;

  auto correlation_block = matrix_correlation_sync<complex_t>::make({3, 2}, 0);

  // Create complex input data
  std::vector<complex_t> input_data = {
      complex_t(1.0, 1.0), complex_t(2.0, 2.0), complex_t(3.0, 3.0),
      complex_t(2.0, 0.0), complex_t(4.0, 0.0), complex_t(6.0, 0.0)};

  Eigen::Map<const Eigen::Matrix<complex_t, Eigen::Dynamic, Eigen::Dynamic>>
      input_matrix(input_data.data(), 3, 2);
  std::vector<complex_t> output_data(4, complex_t(0.0, 0.0));
  Eigen::Map<Eigen::Matrix<complex_t, Eigen::Dynamic, Eigen::Dynamic>>
      output_matrix(output_data.data(), 2, 2);

  types::vector_const_matrix_map<complex_t> input_matrices = {&input_matrix};
  types::vector_matrix_map<complex_t> output_matrices = {&output_matrix};

  OperationReturn result =
      correlation_block->operation(input_matrices, output_matrices);

  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Check that diagonal elements are close to 1.0
  BOOST_CHECK(std::abs(output_matrix(0, 0) - complex_t(1.0, 0.0)) < 1e-10);
  BOOST_CHECK(std::abs(output_matrix(1, 1) - complex_t(1.0, 0.0)) < 1e-10);
}

//==============================================================================
// GNU Radio flowgraph integration tests
//==============================================================================

BOOST_AUTO_TEST_CASE(test_matrix_correlation_flowgraph_integration) {
  auto tb = gr::make_top_block("test_matrix_correlation_flowgraph");

  // Create test data: 4 samples, 2 features
  // Input matrix: [[1, 2], [2, 4], [3, 6], [4, 8]]
  // Column-major format: [1, 2, 3, 4, 2, 4, 6, 8]
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                   2.0f, 4.0f, 6.0f, 8.0f};

  auto src = gr::blocks::vector_source_f::make(input_data, false, 8);
  auto correlation_block =
      matrix_correlation_sync_f::make({4, 2}, 0); // Pearson
  auto sink = gr::blocks::vector_sink_f::make(4); // 2x2 = 4 elements

  tb->connect(src, 0, correlation_block, 0);
  tb->connect(correlation_block, 0, sink, 0);

  tb->run();

  std::vector<float> result = sink->data();

  BOOST_REQUIRE_EQUAL(result.size(), 4);

  // Check diagonal elements (should be 1.0)
  BOOST_CHECK_CLOSE(result[0], 1.0f, 1e-5); // (0,0)
  BOOST_CHECK_CLOSE(result[3], 1.0f, 1e-5); // (1,1) in column-major

  // Check off-diagonal elements (should be 1.0 for perfect correlation)
  BOOST_CHECK_CLOSE(result[1], 1.0f, 1e-5); // (1,0) in column-major
  BOOST_CHECK_CLOSE(result[2], 1.0f, 1e-5); // (0,1) in column-major
}

BOOST_AUTO_TEST_CASE(test_matrix_correlation_pdu_integration) {
  auto correlation_pdu = matrix_correlation_pdu_f::make({3, 2}, 0);

  // Test that PDU block can be created and has correct interface
  BOOST_CHECK(correlation_pdu != nullptr);

  // Check that it has proper GNU Radio block interface
  auto basic_block =
      std::dynamic_pointer_cast<gr::basic_block>(correlation_pdu);
  BOOST_CHECK(basic_block != nullptr);
}

//==============================================================================
// Double precision tests
//==============================================================================

BOOST_AUTO_TEST_CASE(test_matrix_correlation_double_precision) {
  auto correlation_block = matrix_correlation_sync<double>::make({3, 2}, 0);

  // Create input data with higher precision
  std::vector<double> input_data = {1.123456789, 2.123456789, 3.123456789,
                                    2.246913578, 4.246913578, 6.246913578};

  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
      input_matrix(input_data.data(), 3, 2);
  std::vector<double> output_data(4, 0.0);
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
      output_matrix(output_data.data(), 2, 2);

  types::vector_const_matrix_map<double> input_matrices = {&input_matrix};
  types::vector_matrix_map<double> output_matrices = {&output_matrix};

  OperationReturn result =
      correlation_block->operation(input_matrices, output_matrices);

  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Check diagonal elements with double precision
  BOOST_CHECK_CLOSE(output_matrix(0, 0), 1.0, 1e-10);
  BOOST_CHECK_CLOSE(output_matrix(1, 1), 1.0, 1e-10);

  // Check perfect correlation
  BOOST_CHECK_CLOSE(output_matrix(0, 1), 1.0, 1e-10);
  BOOST_CHECK_CLOSE(output_matrix(1, 0), 1.0, 1e-10);
}

} /* namespace linalg */
} /* namespace gr */
