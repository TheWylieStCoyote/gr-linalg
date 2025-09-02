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
#include <gnuradio/linalg/vector_norm.h>
#include <gnuradio/top_block.h>

namespace gr {
namespace linalg {

BOOST_AUTO_TEST_SUITE(qa_vector_norm)

BOOST_AUTO_TEST_CASE(test_vector_norm_sync_basic_constructor) {
  // Test basic constructor functionality
  types::shape shape = {3};
  auto norm_block = vector_norm_sync_f::make(shape, 2);
  BOOST_CHECK(norm_block != nullptr);

  // Check parameter access
  BOOST_CHECK_EQUAL(norm_block->get_order(), 2);
  norm_block->set_order(1);
  BOOST_CHECK_EQUAL(norm_block->get_order(), 1);
}

BOOST_AUTO_TEST_CASE(test_vector_norm_output_shapes) {
  // Test output shape computation
  types::vector_shapes input_shapes = {{4}};
  auto output_shapes = vector_norm<float>::compute_output_shapes(input_shapes);

  BOOST_REQUIRE_EQUAL(output_shapes.size(), 1);
  BOOST_REQUIRE_EQUAL(output_shapes[0].size(), 1);
  BOOST_CHECK_EQUAL(output_shapes[0][0], 1); // Output is scalar
}

BOOST_AUTO_TEST_CASE(test_vector_norm_l2_operation) {
  // Test L2 norm operation directly
  types::shape shape = {3};
  auto norm_block = vector_norm_sync_f::make(shape, 2);

  // Create test input: [3, 4, 0] -> L2 norm = 5
  Eigen::Matrix<float, 3, 1> input_matrix;
  input_matrix << 3.0f, 4.0f, 0.0f;

  Eigen::Matrix<float, 1, 1> output_matrix;

  // Create matrix maps
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      input_map(input_matrix.data(), 3, 1);
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> output_map(
      output_matrix.data(), 1, 1);

  // Create vectors of pointers using correct type aliases
  types::vector_const_matrix_map<float> input_matrices;
  types::vector_matrix_map<float> output_matrices;

  input_matrices.push_back(&input_map);
  output_matrices.push_back(&output_map);

  // Perform operation
  OperationReturn result =
      norm_block->operation(input_matrices, output_matrices);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Check result: L2 norm of [3, 4, 0] should be 5
  BOOST_CHECK_CLOSE(output_matrix(0, 0), 5.0f, 1e-5);
}

BOOST_AUTO_TEST_CASE(test_vector_norm_l1_operation) {
  // Test L1 norm operation
  types::shape shape = {3};
  auto norm_block = vector_norm_sync_f::make(shape, 1);

  // Create test input: [1, -2, 3] -> L1 norm = 6
  Eigen::Matrix<float, 3, 1> input_matrix;
  input_matrix << 1.0f, -2.0f, 3.0f;

  Eigen::Matrix<float, 1, 1> output_matrix;

  // Create matrix maps
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      input_map(input_matrix.data(), 3, 1);
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> output_map(
      output_matrix.data(), 1, 1);

  // Create vectors of pointers using correct type aliases
  types::vector_const_matrix_map<float> input_matrices;
  types::vector_matrix_map<float> output_matrices;

  input_matrices.push_back(&input_map);
  output_matrices.push_back(&output_map);

  // Perform operation
  OperationReturn result =
      norm_block->operation(input_matrices, output_matrices);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Check result: L1 norm of [1, -2, 3] should be 6
  BOOST_CHECK_CLOSE(output_matrix(0, 0), 6.0f, 1e-5);
}

BOOST_AUTO_TEST_CASE(test_vector_norm_linf_operation) {
  // Test L∞ norm operation
  types::shape shape = {4};
  auto norm_block = vector_norm_sync_f::make(shape, 0);

  // Create test input: [1, -5, 3, 2] -> L∞ norm = 5
  Eigen::Matrix<float, 4, 1> input_matrix;
  input_matrix << 1.0f, -5.0f, 3.0f, 2.0f;

  Eigen::Matrix<float, 1, 1> output_matrix;

  // Create matrix maps
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      input_map(input_matrix.data(), 4, 1);
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> output_map(
      output_matrix.data(), 1, 1);

  // Create vectors of pointers using correct type aliases
  types::vector_const_matrix_map<float> input_matrices;
  types::vector_matrix_map<float> output_matrices;

  input_matrices.push_back(&input_map);
  output_matrices.push_back(&output_map);

  // Perform operation
  OperationReturn result =
      norm_block->operation(input_matrices, output_matrices);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Check result: L∞ norm of [1, -5, 3, 2] should be 5
  BOOST_CHECK_CLOSE(output_matrix(0, 0), 5.0f, 1e-5);
}

BOOST_AUTO_TEST_CASE(test_vector_norm_sync_flowgraph) {
  // Test full GNU Radio flowgraph
  auto tb = gr::make_top_block("test_vector_norm");

  // Create test data: [1, 1, 1] -> L2 norm = sqrt(3) ≈ 1.732
  std::vector<float> input_data = {1.0f, 1.0f, 1.0f};

  auto src = gr::blocks::vector_source_f::make(input_data, false, 3);
  auto norm_block = vector_norm_sync_f::make({3}, 2);
  auto sink = gr::blocks::vector_sink_f::make(1);

  tb->connect(src, 0, norm_block, 0);
  tb->connect(norm_block, 0, sink, 0);

  tb->run();

  auto result = sink->data();
  BOOST_REQUIRE_EQUAL(result.size(), 1);
  BOOST_CHECK_CLOSE(result[0], std::sqrt(3.0f), 1e-5);
}

BOOST_AUTO_TEST_CASE(test_vector_norm_complex_data) {
  // Test with complex data
  types::shape shape = {2};
  auto norm_block = vector_norm_sync_c::make(shape, 2);

  // Create test input: [3+4i, 0+0i] -> L2 norm = 5
  Eigen::Matrix<std::complex<float>, 2, 1> input_matrix;
  input_matrix << std::complex<float>(3.0f, 4.0f),
      std::complex<float>(0.0f, 0.0f);

  Eigen::Matrix<std::complex<float>, 1, 1> output_matrix;

  // Create matrix maps
  Eigen::Map<
      const Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic>>
      input_map(input_matrix.data(), 2, 1);
  Eigen::Map<Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic>>
      output_map(output_matrix.data(), 1, 1);

  // Create vectors of pointers using correct type aliases
  types::vector_const_matrix_map<std::complex<float>> input_matrices;
  types::vector_matrix_map<std::complex<float>> output_matrices;

  input_matrices.push_back(&input_map);
  output_matrices.push_back(&output_map);

  // Perform operation
  OperationReturn result =
      norm_block->operation(input_matrices, output_matrices);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Check result: L2 norm of [3+4i, 0] should be 5
  BOOST_CHECK_CLOSE(std::real(output_matrix(0, 0)), 5.0f, 1e-5);
  BOOST_CHECK_CLOSE(std::imag(output_matrix(0, 0)), 0.0f, 1e-5);
}

BOOST_AUTO_TEST_CASE(test_vector_norm_zero_vector) {
  // Test with zero vector
  types::shape shape = {3};
  auto norm_block = vector_norm_sync_f::make(shape, 2);

  // Create test input: [0, 0, 0] -> L2 norm = 0
  Eigen::Matrix<float, 3, 1> input_matrix;
  input_matrix << 0.0f, 0.0f, 0.0f;

  Eigen::Matrix<float, 1, 1> output_matrix;

  // Create matrix maps
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      input_map(input_matrix.data(), 3, 1);
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> output_map(
      output_matrix.data(), 1, 1);

  // Create vectors of pointers using correct type aliases
  types::vector_const_matrix_map<float> input_matrices;
  types::vector_matrix_map<float> output_matrices;

  input_matrices.push_back(&input_map);
  output_matrices.push_back(&output_map);

  // Perform operation
  OperationReturn result =
      norm_block->operation(input_matrices, output_matrices);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Check result: norm of zero vector should be 0
  BOOST_CHECK_CLOSE(output_matrix(0, 0), 0.0f, 1e-5);
}

BOOST_AUTO_TEST_CASE(test_vector_norm_error_conditions) {
  // Test invalid input shapes
  BOOST_CHECK_THROW(vector_norm_sync_f::make({2, 2}, 2), std::invalid_argument);
  BOOST_CHECK_THROW(vector_norm_sync_f::make({0}, 2), std::invalid_argument);
  BOOST_CHECK_THROW(vector_norm_sync_f::make({-1}, 2), std::invalid_argument);
  BOOST_CHECK_THROW(vector_norm_sync_f::make({3}, -1), std::invalid_argument);
}

BOOST_AUTO_TEST_SUITE_END()

} /* namespace linalg */
} /* namespace gr */
