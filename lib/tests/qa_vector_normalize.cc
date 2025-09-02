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
#include <gnuradio/blocks/vector_sink.h>
#include <gnuradio/blocks/vector_source.h>
#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/vector_normalize.h>
#include <gnuradio/top_block.h>

namespace gr {
namespace linalg {

BOOST_AUTO_TEST_SUITE(qa_vector_normalize)

BOOST_AUTO_TEST_CASE(test_vector_normalize_sync_basic_constructor) {
  // Test basic constructor functionality
  types::shape shape = {3};
  auto normalize_block = vector_normalize_sync_f::make(shape, 2);
  BOOST_CHECK(normalize_block != nullptr);

  // Check parameter access
  BOOST_CHECK_EQUAL(normalize_block->get_order(), 2);
  normalize_block->set_order(1);
  BOOST_CHECK_EQUAL(normalize_block->get_order(), 1);
}

BOOST_AUTO_TEST_CASE(test_vector_normalize_output_shapes) {
  // Test output shape computation
  types::vector_shapes input_shapes = {{4}};
  auto output_shapes =
      vector_normalize<float>::compute_output_shapes(input_shapes);

  BOOST_REQUIRE_EQUAL(output_shapes.size(), 1);
  BOOST_REQUIRE_EQUAL(output_shapes[0].size(), 1);
  BOOST_CHECK_EQUAL(output_shapes[0][0], 4); // Output has same size as input
}

BOOST_AUTO_TEST_CASE(test_vector_normalize_l2_operation) {
  // Test L2 normalization operation directly
  types::shape shape = {2};
  auto normalize_block = vector_normalize_sync_f::make(shape, 2);

  // Create test input: [3, 4] -> L2 normalized = [0.6, 0.8]
  Eigen::Matrix<float, 2, 1> input_matrix;
  input_matrix << 3.0f, 4.0f;

  Eigen::Matrix<float, 2, 1> output_matrix;

  // Create matrix maps
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      input_map(input_matrix.data(), 2, 1);
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> output_map(
      output_matrix.data(), 2, 1);

  // Create vectors of pointers using correct type aliases
  types::vector_const_matrix_map<float> input_matrices;
  types::vector_matrix_map<float> output_matrices;

  input_matrices.push_back(&input_map);
  output_matrices.push_back(&output_map);

  // Perform operation
  OperationReturn result =
      normalize_block->operation(input_matrices, output_matrices);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Check result: L2 normalized [3, 4] should be [0.6, 0.8]
  BOOST_CHECK_CLOSE(output_matrix(0, 0), 0.6f, 1e-5);
  BOOST_CHECK_CLOSE(output_matrix(1, 0), 0.8f, 1e-5);

  // Verify the result has unit norm
  float norm = std::sqrt(output_matrix(0, 0) * output_matrix(0, 0) +
                         output_matrix(1, 0) * output_matrix(1, 0));
  BOOST_CHECK_CLOSE(norm, 1.0f, 1e-5);
}

BOOST_AUTO_TEST_CASE(test_vector_normalize_l1_operation) {
  // Test L1 normalization operation
  types::shape shape = {3};
  auto normalize_block = vector_normalize_sync_f::make(shape, 1);

  // Create test input: [1, 2, 3] -> L1 norm = 6, normalized = [1/6, 2/6, 3/6]
  Eigen::Matrix<float, 3, 1> input_matrix;
  input_matrix << 1.0f, 2.0f, 3.0f;

  Eigen::Matrix<float, 3, 1> output_matrix;

  // Create matrix maps
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      input_map(input_matrix.data(), 3, 1);
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> output_map(
      output_matrix.data(), 3, 1);

  // Create vectors of pointers using correct type aliases
  types::vector_const_matrix_map<float> input_matrices;
  types::vector_matrix_map<float> output_matrices;

  input_matrices.push_back(&input_map);
  output_matrices.push_back(&output_map);

  // Perform operation
  OperationReturn result =
      normalize_block->operation(input_matrices, output_matrices);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Check result: L1 normalized [1, 2, 3] should be [1/6, 2/6, 3/6]
  BOOST_CHECK_CLOSE(output_matrix(0, 0), 1.0f / 6.0f, 1e-5);
  BOOST_CHECK_CLOSE(output_matrix(1, 0), 2.0f / 6.0f, 1e-5);
  BOOST_CHECK_CLOSE(output_matrix(2, 0), 3.0f / 6.0f, 1e-5);

  // Verify the result has unit L1 norm
  float l1_norm = std::abs(output_matrix(0, 0)) +
                  std::abs(output_matrix(1, 0)) + std::abs(output_matrix(2, 0));
  BOOST_CHECK_CLOSE(l1_norm, 1.0f, 1e-5);
}

BOOST_AUTO_TEST_CASE(test_vector_normalize_sync_flowgraph) {
  // Test full GNU Radio flowgraph
  auto tb = gr::make_top_block("test_vector_normalize");

  // Create test data: [1, 1] -> L2 normalized = [1/√2, 1/√2]
  std::vector<float> input_data = {1.0f, 1.0f};

  auto src = gr::blocks::vector_source_f::make(input_data, false, 2);
  auto normalize_block = vector_normalize_sync_f::make({2}, 2);
  auto sink = gr::blocks::vector_sink_f::make(2);

  tb->connect(src, 0, normalize_block, 0);
  tb->connect(normalize_block, 0, sink, 0);

  tb->run();

  auto result = sink->data();
  BOOST_REQUIRE_EQUAL(result.size(), 2);

  float expected = 1.0f / std::sqrt(2.0f);
  BOOST_CHECK_CLOSE(result[0], expected, 1e-5);
  BOOST_CHECK_CLOSE(result[1], expected, 1e-5);

  // Verify unit norm
  float norm = std::sqrt(result[0] * result[0] + result[1] * result[1]);
  BOOST_CHECK_CLOSE(norm, 1.0f, 1e-5);
}

BOOST_AUTO_TEST_CASE(test_vector_normalize_complex_data) {
  // Test with complex data
  types::shape shape = {2};
  auto normalize_block = vector_normalize_sync_c::make(shape, 2);

  // Create test input: [3+4i, 0] -> norm = 5, normalized = [0.6+0.8i, 0]
  Eigen::Matrix<std::complex<float>, 2, 1> input_matrix;
  input_matrix << std::complex<float>(3.0f, 4.0f),
      std::complex<float>(0.0f, 0.0f);

  Eigen::Matrix<std::complex<float>, 2, 1> output_matrix;

  // Create matrix maps
  Eigen::Map<
      const Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic>>
      input_map(input_matrix.data(), 2, 1);
  Eigen::Map<Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic>>
      output_map(output_matrix.data(), 2, 1);

  // Create vectors of pointers using correct type aliases
  types::vector_const_matrix_map<std::complex<float>> input_matrices;
  types::vector_matrix_map<std::complex<float>> output_matrices;

  input_matrices.push_back(&input_map);
  output_matrices.push_back(&output_map);

  // Perform operation
  OperationReturn result =
      normalize_block->operation(input_matrices, output_matrices);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Check result: normalized [3+4i, 0] should be [0.6+0.8i, 0]
  BOOST_CHECK_CLOSE(std::real(output_matrix(0, 0)), 0.6f, 1e-5);
  BOOST_CHECK_CLOSE(std::imag(output_matrix(0, 0)), 0.8f, 1e-5);
  BOOST_CHECK_CLOSE(std::real(output_matrix(1, 0)), 0.0f, 1e-5);
  BOOST_CHECK_CLOSE(std::imag(output_matrix(1, 0)), 0.0f, 1e-5);

  // Verify unit norm
  float norm = std::abs(output_matrix(0, 0)) * std::abs(output_matrix(0, 0)) +
               std::abs(output_matrix(1, 0)) * std::abs(output_matrix(1, 0));
  BOOST_CHECK_CLOSE(std::sqrt(norm), 1.0f, 1e-5);
}

BOOST_AUTO_TEST_CASE(test_vector_normalize_zero_vector) {
  // Test with zero vector - should return zero vector
  types::shape shape = {3};
  auto normalize_block = vector_normalize_sync_f::make(shape, 2);

  // Create test input: [0, 0, 0] -> normalized = [0, 0, 0]
  Eigen::Matrix<float, 3, 1> input_matrix;
  input_matrix << 0.0f, 0.0f, 0.0f;

  Eigen::Matrix<float, 3, 1> output_matrix;

  // Create matrix maps
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      input_map(input_matrix.data(), 3, 1);
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> output_map(
      output_matrix.data(), 3, 1);

  // Create vectors of pointers using correct type aliases
  types::vector_const_matrix_map<float> input_matrices;
  types::vector_matrix_map<float> output_matrices;

  input_matrices.push_back(&input_map);
  output_matrices.push_back(&output_map);

  // Perform operation
  OperationReturn result =
      normalize_block->operation(input_matrices, output_matrices);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Check result: normalized zero vector should be zero vector
  BOOST_CHECK_CLOSE(output_matrix(0, 0), 0.0f, 1e-5);
  BOOST_CHECK_CLOSE(output_matrix(1, 0), 0.0f, 1e-5);
  BOOST_CHECK_CLOSE(output_matrix(2, 0), 0.0f, 1e-5);
}

BOOST_AUTO_TEST_CASE(test_vector_normalize_unit_vector) {
  // Test with already unit vector - should remain unchanged
  types::shape shape = {2};
  auto normalize_block = vector_normalize_sync_f::make(shape, 2);

  // Create test input: [0.6, 0.8] (already unit vector)
  Eigen::Matrix<float, 2, 1> input_matrix;
  input_matrix << 0.6f, 0.8f;

  Eigen::Matrix<float, 2, 1> output_matrix;

  // Create matrix maps
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      input_map(input_matrix.data(), 2, 1);
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> output_map(
      output_matrix.data(), 2, 1);

  // Create vectors of pointers using correct type aliases
  types::vector_const_matrix_map<float> input_matrices;
  types::vector_matrix_map<float> output_matrices;

  input_matrices.push_back(&input_map);
  output_matrices.push_back(&output_map);

  // Perform operation
  OperationReturn result =
      normalize_block->operation(input_matrices, output_matrices);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Check result: unit vector should remain unchanged
  BOOST_CHECK_CLOSE(output_matrix(0, 0), 0.6f, 1e-4);
  BOOST_CHECK_CLOSE(output_matrix(1, 0), 0.8f, 1e-4);

  // Verify still unit norm
  float norm = std::sqrt(output_matrix(0, 0) * output_matrix(0, 0) +
                         output_matrix(1, 0) * output_matrix(1, 0));
  BOOST_CHECK_CLOSE(norm, 1.0f, 1e-5);
}

BOOST_AUTO_TEST_CASE(test_vector_normalize_error_conditions) {
  // Test invalid input shapes
  BOOST_CHECK_THROW(vector_normalize_sync_f::make({2, 2}, 2),
                    std::invalid_argument);
  BOOST_CHECK_THROW(vector_normalize_sync_f::make({0}, 2),
                    std::invalid_argument);
  BOOST_CHECK_THROW(vector_normalize_sync_f::make({-1}, 2),
                    std::invalid_argument);
  BOOST_CHECK_THROW(vector_normalize_sync_f::make({3}, -1),
                    std::invalid_argument);
}

BOOST_AUTO_TEST_SUITE_END()

} /* namespace linalg */
} /* namespace gr */
