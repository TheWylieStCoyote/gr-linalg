/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <boost/test/unit_test.hpp>
#include <chrono>
#include <cmath>
#include <complex>
#include <gnuradio/attributes.h>
#include <gnuradio/blocks/vector_sink.h>
#include <gnuradio/blocks/vector_source.h>
#include <gnuradio/linalg/matrix_elementwise_divide.h>
#include <gnuradio/top_block.h>
#include <limits>
#include <vector>

namespace gr {
namespace linalg {

// Test constants
constexpr float FLOAT_TOLERANCE = 1e-6f;
constexpr double DOUBLE_TOLERANCE = 1e-12;

/*!
 * \brief Test matrix element-wise division operation with float matrices
 *
 * Tests basic 2x2 element-wise division:
 * A = [[4, 6], [8, 12]]
 * B = [[2, 3], [4, 4]]
 * Expected result: A ÷ B = [[2, 2], [2, 3]]
 */
BOOST_AUTO_TEST_CASE(test_matrix_elementwise_divide_float_basic) {
  types::shape shape{2, 2};
  const int num_inputs = 2;

  auto div_op = matrix_elementwise_divide_sync<float>::make(shape, num_inputs);

  // Create input matrices A (dividend) and B (divisor)
  std::vector<float> input_a{4.0f, 6.0f, 8.0f, 12.0f};
  std::vector<float> input_b{2.0f, 3.0f, 4.0f, 4.0f};
  std::vector<float> output(4);

  // Create matrix maps
  types::matrix_map_dynamic<float> map_a(input_a.data(), 2, 2);
  types::matrix_map_dynamic<float> map_b(input_b.data(), 2, 2);
  types::matrix_map_dynamic<float> map_out(output.data(), 2, 2);

  types::vector_const_matrix_map<float> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&map_a),
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&map_b)};
  types::vector_matrix_map<float> outputs{&map_out};

  // Perform operation
  OperationReturn result = div_op->operation(inputs, outputs);

  BOOST_REQUIRE_EQUAL(result, OperationReturn::SUCCESS);

  // Check results: [4/2, 6/3, 8/4, 12/4] = [2, 2, 2, 3]
  std::vector<float> expected{2.0f, 2.0f, 2.0f, 3.0f};
  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_CHECK_CLOSE(output[i], expected[i], FLOAT_TOLERANCE);
  }
}

/*!
 * \brief Test matrix element-wise division with multiple divisors
 *
 * Tests chained division: A ÷ B ÷ C
 * A = [[8, 12], [16, 24]]
 * B = [[2, 3], [4, 6]]
 * C = [[2, 2], [2, 2]]
 * Expected result: ((A ÷ B) ÷ C) = [[2, 2], [2, 2]]
 */
BOOST_AUTO_TEST_CASE(test_matrix_elementwise_divide_multiple_divisors) {
  types::shape shape{2, 2};
  const int num_inputs = 3;

  auto div_op = matrix_elementwise_divide_sync<float>::make(shape, num_inputs);

  // Create input matrices A, B, and C
  std::vector<float> input_a{8.0f, 12.0f, 16.0f, 24.0f};
  std::vector<float> input_b{2.0f, 3.0f, 4.0f, 6.0f};
  std::vector<float> input_c{2.0f, 2.0f, 2.0f, 2.0f};
  std::vector<float> output(4);

  // Create matrix maps
  types::matrix_map_dynamic<float> map_a(input_a.data(), 2, 2);
  types::matrix_map_dynamic<float> map_b(input_b.data(), 2, 2);
  types::matrix_map_dynamic<float> map_c(input_c.data(), 2, 2);
  types::matrix_map_dynamic<float> map_out(output.data(), 2, 2);

  types::vector_const_matrix_map<float> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&map_a),
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&map_b),
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&map_c)};
  types::vector_matrix_map<float> outputs{&map_out};

  // Perform operation
  OperationReturn result = div_op->operation(inputs, outputs);

  BOOST_REQUIRE_EQUAL(result, OperationReturn::SUCCESS);

  // Check results: ((8/2)/2, (12/3)/2, (16/4)/2, (24/6)/2) = (2, 2, 2, 2)
  std::vector<float> expected{2.0f, 2.0f, 2.0f, 2.0f};
  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_CHECK_CLOSE(output[i], expected[i], FLOAT_TOLERANCE);
  }
}

/*!
 * \brief Test matrix element-wise division with complex numbers
 *
 * Tests element-wise division with complex float matrices:
 * A = [[2+2j, 4+0j], [0+4j, 6+6j]]
 * B = [[1+1j, 2+0j], [0+2j, 3+3j]]
 * Expected result computed element-wise
 */
BOOST_AUTO_TEST_CASE(test_matrix_elementwise_divide_complex) {
  using cf = std::complex<float>;
  types::shape shape{2, 2};
  const int num_inputs = 2;

  auto div_op = matrix_elementwise_divide_sync<cf>::make(shape, num_inputs);

  // Create complex input matrices
  std::vector<cf> input_a{cf(2, 2), cf(4, 0), cf(0, 4), cf(6, 6)};
  std::vector<cf> input_b{cf(1, 1), cf(2, 0), cf(0, 2), cf(3, 3)};
  std::vector<cf> output(4);

  // Create matrix maps
  types::matrix_map_dynamic<cf> map_a(input_a.data(), 2, 2);
  types::matrix_map_dynamic<cf> map_b(input_b.data(), 2, 2);
  types::matrix_map_dynamic<cf> map_out(output.data(), 2, 2);

  types::vector_const_matrix_map<cf> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<cf> *>(&map_a),
      reinterpret_cast<types::const_matrix_map_dynamic<cf> *>(&map_b)};
  types::vector_matrix_map<cf> outputs{&map_out};

  // Perform operation
  OperationReturn result = div_op->operation(inputs, outputs);

  BOOST_REQUIRE_EQUAL(result, OperationReturn::SUCCESS);

  // Expected results (computed manually):
  // (2+2j)/(1+1j) = ((2+2j)*(1-1j))/((1+1j)*(1-1j)) = (4+0j)/2 = 2+0j
  // (4+0j)/(2+0j) = 2+0j
  // (0+4j)/(0+2j) = ((0+4j)*(0-2j))/((0+2j)*(0-2j)) = (8+0j)/4 = 2+0j
  // (6+6j)/(3+3j) = ((6+6j)*(3-3j))/((3+3j)*(3-3j)) = (36+0j)/18 = 2+0j
  std::vector<cf> expected{cf(2, 0), cf(2, 0), cf(2, 0), cf(2, 0)};

  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_CHECK_CLOSE(output[i].real(), expected[i].real(), FLOAT_TOLERANCE);
    BOOST_CHECK_CLOSE(output[i].imag(), expected[i].imag(), FLOAT_TOLERANCE);
  }
}

/*!
 * \brief Test matrix element-wise division with double precision
 *
 * Tests higher precision computation with double matrices
 */
BOOST_AUTO_TEST_CASE(test_matrix_elementwise_divide_double_precision) {
  types::shape shape{3, 3};
  const int num_inputs = 2;

  auto div_op = matrix_elementwise_divide_sync<double>::make(shape, num_inputs);

  // Create double precision input matrices (3x3)
  std::vector<double> input_a{2.123456789,  4.987654321,  6.141592654,
                              8.271828183,  10.000000001, 12.999999999,
                              14.123123123, 16.456456456, 18.789789789};

  std::vector<double> input_b{1.123456789, 1.987654321, 2.141592654,
                              2.271828183, 5.000000001, 4.999999999,
                              7.123123123, 8.456456456, 9.789789789};
  std::vector<double> output(9);

  // Create matrix maps
  types::matrix_map_dynamic<double> map_a(input_a.data(), 3, 3);
  types::matrix_map_dynamic<double> map_b(input_b.data(), 3, 3);
  types::matrix_map_dynamic<double> map_out(output.data(), 3, 3);

  types::vector_const_matrix_map<double> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<double> *>(&map_a),
      reinterpret_cast<types::const_matrix_map_dynamic<double> *>(&map_b)};
  types::vector_matrix_map<double> outputs{&map_out};

  // Perform operation
  OperationReturn result = div_op->operation(inputs, outputs);

  BOOST_REQUIRE_EQUAL(result, OperationReturn::SUCCESS);

  // Verify element-wise division with high precision
  for (size_t i = 0; i < input_a.size(); ++i) {
    double expected = input_a[i] / input_b[i];
    BOOST_CHECK_CLOSE(output[i], expected, DOUBLE_TOLERANCE);
  }
}

/*!
 * \brief Test division by zero handling
 *
 * Tests that the operation handles division by zero gracefully,
 * returning infinity with appropriate sign
 */
BOOST_AUTO_TEST_CASE(test_matrix_elementwise_divide_zero_handling) {
  types::shape shape{2, 2};
  const int num_inputs = 2;

  auto div_op = matrix_elementwise_divide_sync<float>::make(shape, num_inputs);

  // Create matrices with zero divisors
  std::vector<float> input_a{4.0f, -6.0f, 8.0f,
                             0.0f}; // Some negative for sign testing
  std::vector<float> input_b{0.0f, 0.0f, 2.0f, 0.0f}; // Zeros and non-zero
  std::vector<float> output(4);

  // Create matrix maps
  types::matrix_map_dynamic<float> map_a(input_a.data(), 2, 2);
  types::matrix_map_dynamic<float> map_b(input_b.data(), 2, 2);
  types::matrix_map_dynamic<float> map_out(output.data(), 2, 2);

  types::vector_const_matrix_map<float> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&map_a),
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&map_b)};
  types::vector_matrix_map<float> outputs{&map_out};

  // Perform operation
  OperationReturn result = div_op->operation(inputs, outputs);

  BOOST_REQUIRE_EQUAL(result, OperationReturn::SUCCESS);

  // Check results: [+inf, -inf, 4, nan/inf]
  BOOST_CHECK(std::isinf(output[0]) && output[0] > 0); // 4/0 = +inf
  BOOST_CHECK(std::isinf(output[1]) && output[1] < 0); // -6/0 = -inf
  BOOST_CHECK_CLOSE(output[2], 4.0f, FLOAT_TOLERANCE); // 8/2 = 4
  BOOST_CHECK(std::isnan(output[3]) ||
              std::isinf(output[3])); // 0/0 = nan or inf
}

/*!
 * \brief Test division by very small numbers
 *
 * Tests behavior when dividing by numbers close to zero
 */
BOOST_AUTO_TEST_CASE(test_matrix_elementwise_divide_small_numbers) {
  types::shape shape{2, 2};
  const int num_inputs = 2;

  auto div_op = matrix_elementwise_divide_sync<double>::make(shape, num_inputs);

  // Create matrices with very small divisors
  std::vector<double> input_a{1.0, 2.0, 3.0, 4.0};
  std::vector<double> input_b{1e-10, 1e-15, 1e-14, 1e-12}; // Very small numbers
  std::vector<double> output(4);

  // Create matrix maps
  types::matrix_map_dynamic<double> map_a(input_a.data(), 2, 2);
  types::matrix_map_dynamic<double> map_b(input_b.data(), 2, 2);
  types::matrix_map_dynamic<double> map_out(output.data(), 2, 2);

  types::vector_const_matrix_map<double> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<double> *>(&map_a),
      reinterpret_cast<types::const_matrix_map_dynamic<double> *>(&map_b)};
  types::vector_matrix_map<double> outputs{&map_out};

  // Perform operation
  OperationReturn result = div_op->operation(inputs, outputs);

  BOOST_REQUIRE_EQUAL(result, OperationReturn::SUCCESS);

  // Results should be very large but finite numbers
  for (size_t i = 0; i < output.size(); ++i) {
    double expected = input_a[i] / input_b[i];
    BOOST_CHECK_CLOSE(output[i], expected, 1e-10); // Allow some numerical error
    BOOST_CHECK(std::isfinite(output[i]));         // Should be finite, not inf
  }
}

/*!
 * \brief Test error handling for mismatched shapes
 *
 * Tests that the operation returns appropriate error codes when
 * input matrix shapes don't match
 */
BOOST_AUTO_TEST_CASE(test_matrix_elementwise_divide_shape_mismatch) {
  types::shape shape{2, 2};
  const int num_inputs = 2;

  auto div_op = matrix_elementwise_divide_sync<float>::make(shape, num_inputs);

  // Create matrices with different shapes
  std::vector<float> input_a{1.0f, 2.0f, 3.0f, 4.0f}; // 2x2
  std::vector<float> input_b{1.0f, 2.0f, 3.0f,
                             4.0f, 5.0f, 6.0f}; // 2x3 (mismatched)
  std::vector<float> output(4);

  // Create matrix maps with different shapes
  types::matrix_map_dynamic<float> map_a(input_a.data(), 2, 2);
  types::matrix_map_dynamic<float> map_b(input_b.data(), 2,
                                         3); // Different shape
  types::matrix_map_dynamic<float> map_out(output.data(), 2, 2);

  types::vector_const_matrix_map<float> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&map_a),
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&map_b)};
  types::vector_matrix_map<float> outputs{&map_out};

  // Perform operation - should fail due to shape mismatch
  OperationReturn result = div_op->operation(inputs, outputs);

  BOOST_REQUIRE_EQUAL(result, OperationReturn::INVALID_SHAPE);
}

/*!
 * \brief Test insufficient inputs error handling
 *
 * Tests error handling when fewer than 2 inputs are provided
 */
BOOST_AUTO_TEST_CASE(test_matrix_elementwise_divide_insufficient_inputs) {
  types::shape shape{2, 2};
  const int num_inputs = 2;

  auto div_op = matrix_elementwise_divide_sync<float>::make(shape, num_inputs);

  // Create only one input (insufficient for division)
  std::vector<float> input_a{1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> output(4);

  types::matrix_map_dynamic<float> map_a(input_a.data(), 2, 2);
  types::matrix_map_dynamic<float> map_out(output.data(), 2, 2);

  // Provide only one input (should require at least 2 for division)
  types::vector_const_matrix_map<float> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&map_a)};
  types::vector_matrix_map<float> outputs{&map_out};

  // Perform operation - should fail due to insufficient inputs
  OperationReturn result = div_op->operation(inputs, outputs);

  BOOST_REQUIRE_EQUAL(result, OperationReturn::INVALID_SHAPE);
}

/*!
 * \brief Test vector (1D) element-wise division
 *
 * Tests element-wise division of vectors:
 * a = [8, 6, 12, 20]
 * b = [2, 3, 4, 5]
 * Expected result: a ÷ b = [4, 2, 3, 4]
 */
BOOST_AUTO_TEST_CASE(test_matrix_elementwise_divide_vector) {
  types::shape shape{4, 1}; // Column vector
  const int num_inputs = 2;

  auto div_op = matrix_elementwise_divide_sync<float>::make(shape, num_inputs);

  // Create input vectors as column matrices
  std::vector<float> input_a{8.0f, 6.0f, 12.0f, 20.0f};
  std::vector<float> input_b{2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<float> output(4);

  // Create matrix maps (column vectors)
  types::matrix_map_dynamic<float> map_a(input_a.data(), 4, 1);
  types::matrix_map_dynamic<float> map_b(input_b.data(), 4, 1);
  types::matrix_map_dynamic<float> map_out(output.data(), 4, 1);

  types::vector_const_matrix_map<float> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&map_a),
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&map_b)};
  types::vector_matrix_map<float> outputs{&map_out};

  // Perform operation
  OperationReturn result = div_op->operation(inputs, outputs);

  BOOST_REQUIRE_EQUAL(result, OperationReturn::SUCCESS);

  // Check results: [8/2, 6/3, 12/4, 20/5] = [4, 2, 3, 4]
  std::vector<float> expected{4.0f, 2.0f, 3.0f, 4.0f};
  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_CHECK_CLOSE(output[i], expected[i], FLOAT_TOLERANCE);
  }
}

/*!
 * \brief Test complex division by zero
 *
 * Tests division by zero for complex numbers
 */
BOOST_AUTO_TEST_CASE(test_matrix_elementwise_divide_complex_zero) {
  using cf = std::complex<float>;
  types::shape shape{2, 2};
  const int num_inputs = 2;

  auto div_op = matrix_elementwise_divide_sync<cf>::make(shape, num_inputs);

  // Create complex matrices with zero divisor
  std::vector<cf> input_a{cf(1, 1), cf(2, 2), cf(3, 3), cf(4, 4)};
  std::vector<cf> input_b{cf(0, 0), cf(1, 1), cf(0, 0), cf(2, 2)};
  std::vector<cf> output(4);

  // Create matrix maps
  types::matrix_map_dynamic<cf> map_a(input_a.data(), 2, 2);
  types::matrix_map_dynamic<cf> map_b(input_b.data(), 2, 2);
  types::matrix_map_dynamic<cf> map_out(output.data(), 2, 2);

  types::vector_const_matrix_map<cf> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<cf> *>(&map_a),
      reinterpret_cast<types::const_matrix_map_dynamic<cf> *>(&map_b)};
  types::vector_matrix_map<cf> outputs{&map_out};

  // Perform operation
  OperationReturn result = div_op->operation(inputs, outputs);

  BOOST_REQUIRE_EQUAL(result, OperationReturn::SUCCESS);

  // Check that division by zero results in complex infinity
  BOOST_CHECK(std::isinf(output[0].real()) ||
              std::isinf(output[0].imag())); // (1+1j)/(0+0j)
  // Normal division: (2+2j)/(1+1j) = 2+0j
  BOOST_CHECK_CLOSE(output[1].real(), 2.0f, FLOAT_TOLERANCE);
  BOOST_CHECK_SMALL(output[1].imag(), FLOAT_TOLERANCE);
  BOOST_CHECK(std::isinf(output[2].real()) ||
              std::isinf(output[2].imag())); // (3+3j)/(0+0j)
}

//==============================================================================
// GNU Radio Flowgraph Tests
//==============================================================================

/*!
 * \brief Test matrix elementwise divide sync block in GNU Radio flowgraph
 *
 * Tests basic 2×2 element-wise division through actual signal processing
 * pipeline: Input A: [[4, 8], [6, 12]] (column-major: [4, 6, 8, 12]) Input B:
 * [[2, 4], [3, 4]] (column-major: [2, 3, 4, 4]) Expected result: A ÷ B = [[2,
 * 2], [2, 3]] (column-major: [2, 2, 2, 3])
 */
BOOST_AUTO_TEST_CASE(test_matrix_elementwise_divide_sync_flowgraph) {
  BOOST_TEST_MESSAGE(
      "Testing matrix elementwise divide sync block in GNU Radio flowgraph...");

  auto tb = gr::make_top_block("elementwise_divide_test");

  // Create input matrices: A = [[4, 8], [6, 12]], B = [[2, 4], [3, 4]]
  std::vector<float> input_a = {4.0f, 6.0f, 8.0f, 12.0f}; // Dividend matrix
  std::vector<float> input_b = {2.0f, 3.0f, 4.0f, 4.0f};  // Divisor matrix

  auto vector_source_a = gr::blocks::vector_source_f::make(input_a, false, 4);
  auto vector_source_b = gr::blocks::vector_source_f::make(input_b, false, 4);
  auto divide_block = matrix_elementwise_divide_sync<float>::make({2, 2}, 2);
  auto output_sink = gr::blocks::vector_sink_f::make(4);

  tb->connect(vector_source_a, 0, divide_block, 0);
  tb->connect(vector_source_b, 0, divide_block, 1);
  tb->connect(divide_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 4);

  // Expected result: [4/2, 6/3, 8/4, 12/4] = [2, 2, 2, 3]
  std::vector<float> expected = {2.0f, 2.0f, 2.0f, 3.0f};
  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_CHECK_CLOSE(output_data[i], expected[i], FLOAT_TOLERANCE);
  }

  BOOST_TEST_MESSAGE("Matrix elementwise divide sync flowgraph test passed.");
}

/*!
 * \brief Test elementwise divide with multiple inputs in flowgraph
 *
 * Tests chained division A ÷ B ÷ C through GNU Radio pipeline:
 * A = [[8, 16], [12, 24]]
 * B = [[2, 4], [3, 6]]
 * C = [[2, 2], [2, 2]]
 * Expected: ((A ÷ B) ÷ C) = [[2, 2], [2, 2]]
 */
BOOST_AUTO_TEST_CASE(
    test_matrix_elementwise_divide_sync_flowgraph_multiple_inputs) {
  BOOST_TEST_MESSAGE("Testing elementwise divide sync block with multiple "
                     "inputs in flowgraph...");

  auto tb = gr::make_top_block("elementwise_divide_multi_test");

  // Create input matrices for chained division
  std::vector<float> input_a = {8.0f, 12.0f, 16.0f, 24.0f}; // Dividend
  std::vector<float> input_b = {2.0f, 3.0f, 4.0f, 6.0f};    // First divisor
  std::vector<float> input_c = {2.0f, 2.0f, 2.0f, 2.0f};    // Second divisor

  auto vector_source_a = gr::blocks::vector_source_f::make(input_a, false, 4);
  auto vector_source_b = gr::blocks::vector_source_f::make(input_b, false, 4);
  auto vector_source_c = gr::blocks::vector_source_f::make(input_c, false, 4);
  auto divide_block = matrix_elementwise_divide_sync<float>::make({2, 2}, 3);
  auto output_sink = gr::blocks::vector_sink_f::make(4);

  tb->connect(vector_source_a, 0, divide_block, 0);
  tb->connect(vector_source_b, 0, divide_block, 1);
  tb->connect(vector_source_c, 0, divide_block, 2);
  tb->connect(divide_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 4);

  // Expected result: ((8/2)/2, (12/3)/2, (16/4)/2, (24/6)/2) = (2, 2, 2, 2)
  std::vector<float> expected = {2.0f, 2.0f, 2.0f, 2.0f};
  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_CHECK_CLOSE(output_data[i], expected[i], FLOAT_TOLERANCE);
  }

  BOOST_TEST_MESSAGE(
      "Multiple inputs elementwise divide flowgraph test passed.");
}

/*!
 * \brief Test elementwise divide with complex numbers in flowgraph
 *
 * Tests complex element-wise division through GNU Radio pipeline:
 * A = [[2+2j, 4+0j], [0+4j, 6+6j]]
 * B = [[1+1j, 2+0j], [0+2j, 3+3j]]
 * Expected computed element-wise
 */
BOOST_AUTO_TEST_CASE(test_matrix_elementwise_divide_sync_flowgraph_complex) {
  BOOST_TEST_MESSAGE("Testing elementwise divide sync block with complex "
                     "numbers in flowgraph...");

  using cf = std::complex<float>;
  auto tb = gr::make_top_block("elementwise_divide_complex_test");

  // Create complex input matrices
  std::vector<cf> input_a = {cf(2, 2), cf(0, 4), cf(4, 0), cf(6, 6)};
  std::vector<cf> input_b = {cf(1, 1), cf(0, 2), cf(2, 0), cf(3, 3)};

  auto vector_source_a = gr::blocks::vector_source_c::make(input_a, false, 4);
  auto vector_source_b = gr::blocks::vector_source_c::make(input_b, false, 4);
  auto divide_block = matrix_elementwise_divide_sync<cf>::make({2, 2}, 2);
  auto output_sink = gr::blocks::vector_sink_c::make(4);

  tb->connect(vector_source_a, 0, divide_block, 0);
  tb->connect(vector_source_b, 0, divide_block, 1);
  tb->connect(divide_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 4);

  // Expected results (computed manually):
  // (2+2j)/(1+1j) = 2+0j, (0+4j)/(0+2j) = 2+0j
  // (4+0j)/(2+0j) = 2+0j, (6+6j)/(3+3j) = 2+0j
  std::vector<cf> expected = {cf(2, 0), cf(2, 0), cf(2, 0), cf(2, 0)};

  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_CHECK_CLOSE(output_data[i].real(), expected[i].real(),
                      FLOAT_TOLERANCE);
    BOOST_CHECK_CLOSE(output_data[i].imag(), expected[i].imag(),
                      FLOAT_TOLERANCE);
  }

  BOOST_TEST_MESSAGE("Complex elementwise divide flowgraph test passed.");
}

/*!
 * \brief Test elementwise divide with double precision in flowgraph
 *
 * Tests high precision element-wise division through GNU Radio pipeline
 */
BOOST_AUTO_TEST_CASE(test_matrix_elementwise_divide_sync_flowgraph_double) {
  BOOST_TEST_MESSAGE("Testing elementwise divide sync block with double "
                     "precision in flowgraph...");

  auto tb = gr::make_top_block("elementwise_divide_double_test");

  // Create double precision matrices (3×3)
  std::vector<double> input_a = {2.123456789, 8.271828183,  14.123123123,
                                 4.987654321, 10.000000001, 16.456456456,
                                 6.141592654, 12.999999999, 18.789789789};

  std::vector<double> input_b = {1.123456789, 2.271828183, 7.123123123,
                                 1.987654321, 5.000000001, 8.456456456,
                                 2.141592654, 4.999999999, 9.789789789};

  // Convert double to float for compatibility with available GNU Radio blocks
  std::vector<float> input_a_f(input_a.begin(), input_a.end());
  std::vector<float> input_b_f(input_b.begin(), input_b.end());

  auto vector_source_a = gr::blocks::vector_source_f::make(input_a_f, false, 9);
  auto vector_source_b = gr::blocks::vector_source_f::make(input_b_f, false, 9);
  auto divide_block = matrix_elementwise_divide_sync<float>::make({3, 3}, 2);
  auto output_sink = gr::blocks::vector_sink_f::make(9);

  tb->connect(vector_source_a, 0, divide_block, 0);
  tb->connect(vector_source_b, 0, divide_block, 1);
  tb->connect(divide_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 9);

  // Verify element-wise division (using relaxed tolerance for float precision)
  for (size_t i = 0; i < input_a.size(); ++i) {
    float expected = static_cast<float>(input_a[i] / input_b[i]);
    BOOST_CHECK_CLOSE(output_data[i], expected,
                      1e-4f); // More relaxed tolerance for float precision
  }

  BOOST_TEST_MESSAGE(
      "Double precision elementwise divide flowgraph test passed.");
}

/*!
 * \brief Test elementwise divide with division by zero in flowgraph
 *
 * Tests division by zero handling through GNU Radio pipeline
 */
BOOST_AUTO_TEST_CASE(
    test_matrix_elementwise_divide_sync_flowgraph_zero_division) {
  BOOST_TEST_MESSAGE("Testing elementwise divide sync block with division by "
                     "zero in flowgraph...");

  auto tb = gr::make_top_block("elementwise_divide_zero_test");

  // Create matrices with zero divisors
  std::vector<float> input_a = {4.0f, -6.0f, 8.0f, 0.0f};
  std::vector<float> input_b = {0.0f, 0.0f, 2.0f, 0.0f}; // Contains zeros

  auto vector_source_a = gr::blocks::vector_source_f::make(input_a, false, 4);
  auto vector_source_b = gr::blocks::vector_source_f::make(input_b, false, 4);
  auto divide_block = matrix_elementwise_divide_sync<float>::make({2, 2}, 2);
  auto output_sink = gr::blocks::vector_sink_f::make(4);

  tb->connect(vector_source_a, 0, divide_block, 0);
  tb->connect(vector_source_b, 0, divide_block, 1);
  tb->connect(divide_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 4);

  // Check division by zero results: [+inf, -inf, 4, nan/inf]
  BOOST_CHECK(std::isinf(output_data[0]) && output_data[0] > 0); // 4/0 = +inf
  BOOST_CHECK(std::isinf(output_data[1]) && output_data[1] < 0); // -6/0 = -inf
  BOOST_CHECK_CLOSE(output_data[2], 4.0f, FLOAT_TOLERANCE);      // 8/2 = 4
  BOOST_CHECK(std::isnan(output_data[3]) || std::isinf(output_data[3])); // 0/0

  BOOST_TEST_MESSAGE(
      "Division by zero elementwise divide flowgraph test passed.");
}

/*!
 * \brief Test elementwise divide with vector data in flowgraph
 *
 * Tests element-wise division of vectors through GNU Radio pipeline:
 * a = [8, 6, 12, 20] (column vector)
 * b = [2, 3, 4, 5] (column vector)
 * Expected: a ÷ b = [4, 2, 3, 4]
 */
BOOST_AUTO_TEST_CASE(test_matrix_elementwise_divide_sync_flowgraph_vector) {
  BOOST_TEST_MESSAGE(
      "Testing elementwise divide sync block with vectors in flowgraph...");

  auto tb = gr::make_top_block("elementwise_divide_vector_test");

  // Create input vectors as column matrices
  std::vector<float> input_a = {8.0f, 6.0f, 12.0f, 20.0f};
  std::vector<float> input_b = {2.0f, 3.0f, 4.0f, 5.0f};

  auto vector_source_a = gr::blocks::vector_source_f::make(input_a, false, 4);
  auto vector_source_b = gr::blocks::vector_source_f::make(input_b, false, 4);
  auto divide_block =
      matrix_elementwise_divide_sync<float>::make({4, 1}, 2); // Column vector
  auto output_sink = gr::blocks::vector_sink_f::make(4);

  tb->connect(vector_source_a, 0, divide_block, 0);
  tb->connect(vector_source_b, 0, divide_block, 1);
  tb->connect(divide_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 4);

  // Expected result: [8/2, 6/3, 12/4, 20/5] = [4, 2, 3, 4]
  std::vector<float> expected = {4.0f, 2.0f, 3.0f, 4.0f};
  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_CHECK_CLOSE(output_data[i], expected[i], FLOAT_TOLERANCE);
  }

  BOOST_TEST_MESSAGE("Vector elementwise divide flowgraph test passed.");
}

/*!
 * \brief Test elementwise divide with small numbers in flowgraph
 *
 * Tests division by very small numbers through GNU Radio pipeline
 */
BOOST_AUTO_TEST_CASE(
    test_matrix_elementwise_divide_sync_flowgraph_small_numbers) {
  BOOST_TEST_MESSAGE("Testing elementwise divide sync block with small numbers "
                     "in flowgraph...");

  auto tb = gr::make_top_block("elementwise_divide_small_test");

  // Create matrices with small but not extreme divisors (to avoid float
  // overflow)
  std::vector<double> input_a = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> input_b = {
      1e-6, 1e-7, 1e-6, 1e-7}; // Small but manageable numbers for float

  // Convert double to float for compatibility with available GNU Radio blocks
  std::vector<float> input_a_f(input_a.begin(), input_a.end());
  std::vector<float> input_b_f(input_b.begin(), input_b.end());

  auto vector_source_a = gr::blocks::vector_source_f::make(input_a_f, false, 4);
  auto vector_source_b = gr::blocks::vector_source_f::make(input_b_f, false, 4);
  auto divide_block = matrix_elementwise_divide_sync<float>::make({2, 2}, 2);
  auto output_sink = gr::blocks::vector_sink_f::make(4);

  tb->connect(vector_source_a, 0, divide_block, 0);
  tb->connect(vector_source_b, 0, divide_block, 1);
  tb->connect(divide_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 4);

  // Results should be very large but finite numbers
  for (size_t i = 0; i < output_data.size(); ++i) {
    float expected = static_cast<float>(input_a[i] / input_b[i]);
    BOOST_CHECK_CLOSE(output_data[i], expected,
                      1e-4f); // Allow some numerical error for float precision
    BOOST_CHECK(std::isfinite(output_data[i])); // Should be finite, not inf
  }

  BOOST_TEST_MESSAGE("Small numbers elementwise divide flowgraph test passed.");
}

/*!
 * \brief Test elementwise divide with multiple matrices in flowgraph
 *
 * Tests processing of multiple consecutive matrices through GNU Radio pipeline
 */
BOOST_AUTO_TEST_CASE(
    test_matrix_elementwise_divide_sync_flowgraph_multiple_matrices) {
  BOOST_TEST_MESSAGE("Testing elementwise divide sync block with multiple "
                     "matrices in flowgraph...");

  auto tb = gr::make_top_block("elementwise_divide_multi_matrices_test");

  // Create multiple 2x2 matrices for consecutive processing
  std::vector<float> input_a = {// Matrix 1: [[4, 8], [6, 12]]
                                4.0f, 6.0f, 8.0f, 12.0f,
                                // Matrix 2: [[8, 16], [12, 24]]
                                8.0f, 12.0f, 16.0f, 24.0f,
                                // Matrix 3: [[12, 24], [18, 36]]
                                12.0f, 18.0f, 24.0f, 36.0f};

  std::vector<float> input_b = {// Matrix 1: [[2, 4], [3, 4]]
                                2.0f, 3.0f, 4.0f, 4.0f,
                                // Matrix 2: [[4, 8], [6, 8]]
                                4.0f, 6.0f, 8.0f, 8.0f,
                                // Matrix 3: [[6, 12], [9, 12]]
                                6.0f, 9.0f, 12.0f, 12.0f};

  auto vector_source_a = gr::blocks::vector_source_f::make(input_a, false, 4);
  auto vector_source_b = gr::blocks::vector_source_f::make(input_b, false, 4);
  auto divide_block = matrix_elementwise_divide_sync<float>::make({2, 2}, 2);
  auto output_sink = gr::blocks::vector_sink_f::make(4);

  tb->connect(vector_source_a, 0, divide_block, 0);
  tb->connect(vector_source_b, 0, divide_block, 1);
  tb->connect(divide_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 12); // 3 matrices × 4 elements each

  // Expected results for all three matrices: [2, 2, 2, 3]
  std::vector<float> expected_per_matrix = {2.0f, 2.0f, 2.0f, 3.0f};

  for (int matrix = 0; matrix < 3; ++matrix) {
    for (int i = 0; i < 4; ++i) {
      size_t idx = matrix * 4 + i;
      BOOST_CHECK_CLOSE(output_data[idx], expected_per_matrix[i],
                        FLOAT_TOLERANCE);
    }
  }

  BOOST_TEST_MESSAGE(
      "Multiple matrices elementwise divide flowgraph test passed.");
}

/*!
 * \brief Test elementwise divide performance in flowgraph
 *
 * Tests performance with a large number of matrices through GNU Radio pipeline
 */
BOOST_AUTO_TEST_CASE(
    test_matrix_elementwise_divide_sync_flowgraph_performance) {
  BOOST_TEST_MESSAGE(
      "Testing elementwise divide sync block performance in flowgraph...");

  auto tb = gr::make_top_block("elementwise_divide_perf_test");

  // Create many 2x2 matrices for performance testing
  int num_matrices = 1000;
  std::vector<float> input_a, input_b;
  input_a.reserve(num_matrices * 4);
  input_b.reserve(num_matrices * 4);

  for (int i = 0; i < num_matrices; ++i) {
    // Generate matrices with varying values for performance test
    float base = 1.0f + 0.001f * i;
    input_a.insert(input_a.end(),
                   {4.0f * base, 6.0f * base, 8.0f * base, 12.0f * base});
    input_b.insert(input_b.end(),
                   {2.0f * base, 3.0f * base, 4.0f * base, 4.0f * base});
  }

  auto start_time = std::chrono::high_resolution_clock::now();

  auto vector_source_a = gr::blocks::vector_source_f::make(input_a, false, 4);
  auto vector_source_b = gr::blocks::vector_source_f::make(input_b, false, 4);
  auto divide_block = matrix_elementwise_divide_sync<float>::make({2, 2}, 2);
  auto output_sink = gr::blocks::vector_sink_f::make(4);

  tb->connect(vector_source_a, 0, divide_block, 0);
  tb->connect(vector_source_b, 0, divide_block, 1);
  tb->connect(divide_block, 0, output_sink, 0);

  tb->run();

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), num_matrices * 4);

  // Verify first and last matrix results
  BOOST_CHECK_CLOSE(output_data[0], 2.0f,
                    FLOAT_TOLERANCE); // First matrix first element
  BOOST_CHECK_CLOSE(output_data[1], 2.0f,
                    FLOAT_TOLERANCE); // First matrix second element

  size_t last_matrix_idx = (num_matrices - 1) * 4;
  BOOST_CHECK_CLOSE(output_data[last_matrix_idx], 2.0f,
                    1e-2f); // Last matrix first element
  BOOST_CHECK_CLOSE(output_data[last_matrix_idx + 1], 2.0f,
                    1e-2f); // Last matrix second element

  // All results should be finite
  for (size_t i = 0; i < output_data.size(); ++i) {
    BOOST_CHECK(std::isfinite(output_data[i]));
  }

  BOOST_TEST_MESSAGE("Processed " << num_matrices << " matrices in "
                                  << duration.count() << " ms");
  BOOST_TEST_MESSAGE("Elementwise divide performance flowgraph test passed.");
}

} // namespace linalg
} // namespace gr