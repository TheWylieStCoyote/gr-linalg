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
#include <gnuradio/linalg/matrix_elementwise_multiply.h>
#include <gnuradio/top_block.h>
#include <vector>

namespace gr {
namespace linalg {

// Test constants
constexpr float FLOAT_TOLERANCE = 1e-6f;
constexpr double DOUBLE_TOLERANCE = 1e-12;

/*!
 * \brief Test matrix element-wise multiplication operation with float matrices
 *
 * Tests basic 2x2 element-wise multiplication:
 * A = [[1, 2], [3, 4]]
 * B = [[2, 3], [4, 5]]
 * Expected result: A ⊙ B = [[2, 6], [12, 20]]
 */
BOOST_AUTO_TEST_CASE(test_matrix_elementwise_multiply_float_basic) {
  types::shape shape{2, 2};
  const int num_inputs = 2;

  auto mult_op =
      matrix_elementwise_multiply_sync<float>::make(shape, num_inputs);

  // Create input matrices A and B
  std::vector<float> input_a{1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> input_b{2.0f, 3.0f, 4.0f, 5.0f};
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
  OperationReturn result = mult_op->operation(inputs, outputs);

  BOOST_REQUIRE_EQUAL(result, OperationReturn::SUCCESS);

  // Check results: [1*2, 2*3, 3*4, 4*5] = [2, 6, 12, 20]
  std::vector<float> expected{2.0f, 6.0f, 12.0f, 20.0f};
  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_CHECK_CLOSE(output[i], expected[i], 1e-6);
  }
}

/*!
 * \brief Test matrix element-wise multiplication with multiple inputs
 *
 * Tests 3-way element-wise multiplication:
 * A = [[1, 2], [3, 4]]
 * B = [[2, 1], [1, 2]]
 * C = [[3, 2], [2, 1]]
 * Expected result: A ⊙ B ⊙ C = [[6, 4], [6, 8]]
 */
BOOST_AUTO_TEST_CASE(test_matrix_elementwise_multiply_multiple_inputs) {
  types::shape shape{2, 2};
  const int num_inputs = 3;

  auto mult_op =
      matrix_elementwise_multiply_sync<float>::make(shape, num_inputs);

  // Create input matrices A, B, and C
  std::vector<float> input_a{1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> input_b{2.0f, 1.0f, 1.0f, 2.0f};
  std::vector<float> input_c{3.0f, 2.0f, 2.0f, 1.0f};
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
  OperationReturn result = mult_op->operation(inputs, outputs);

  BOOST_REQUIRE_EQUAL(result, OperationReturn::SUCCESS);

  // Check results: [1*2*3, 2*1*2, 3*1*2, 4*2*1] = [6, 4, 6, 8]
  std::vector<float> expected{6.0f, 4.0f, 6.0f, 8.0f};
  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_CHECK_CLOSE(output[i], expected[i], 1e-6);
  }
}

/*!
 * \brief Test matrix element-wise multiplication with complex numbers
 *
 * Tests element-wise multiplication with complex float matrices:
 * A = [[1+1j, 2+0j], [0+2j, 1+1j]]
 * B = [[2+0j, 1+1j], [1-1j, 2+0j]]
 * Expected result computed element-wise
 */
BOOST_AUTO_TEST_CASE(test_matrix_elementwise_multiply_complex) {
  using cf = std::complex<float>;
  types::shape shape{2, 2};
  const int num_inputs = 2;

  auto mult_op = matrix_elementwise_multiply_sync<cf>::make(shape, num_inputs);

  // Create complex input matrices
  std::vector<cf> input_a{cf(1, 1), cf(2, 0), cf(0, 2), cf(1, 1)};
  std::vector<cf> input_b{cf(2, 0), cf(1, 1), cf(1, -1), cf(2, 0)};
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
  OperationReturn result = mult_op->operation(inputs, outputs);

  BOOST_REQUIRE_EQUAL(result, OperationReturn::SUCCESS);

  // Expected results:
  // (1+1j)*(2+0j) = 2+2j
  // (2+0j)*(1+1j) = 2+2j
  // (0+2j)*(1-1j) = 2+2j
  // (1+1j)*(2+0j) = 2+2j
  std::vector<cf> expected{cf(2, 2), cf(2, 2), cf(2, 2), cf(2, 2)};

  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_CHECK_CLOSE(output[i].real(), expected[i].real(), 1e-6);
    BOOST_CHECK_CLOSE(output[i].imag(), expected[i].imag(), 1e-6);
  }
}

/*!
 * \brief Test matrix element-wise multiplication with double precision
 *
 * Tests higher precision computation with double matrices
 */
BOOST_AUTO_TEST_CASE(test_matrix_elementwise_multiply_double_precision) {
  types::shape shape{3, 3};
  const int num_inputs = 2;

  auto mult_op =
      matrix_elementwise_multiply_sync<double>::make(shape, num_inputs);

  // Create double precision input matrices (3x3)
  std::vector<double> input_a{1.123456789, 2.987654321, 3.141592654,
                              4.271828183, 5.000000001, 6.999999999,
                              7.123123123, 8.456456456, 9.789789789};

  std::vector<double> input_b{0.123456789, 0.987654321, 1.141592654,
                              1.271828183, 2.000000001, 2.999999999,
                              3.123123123, 3.456456456, 3.789789789};
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
  OperationReturn result = mult_op->operation(inputs, outputs);

  BOOST_REQUIRE_EQUAL(result, OperationReturn::SUCCESS);

  // Verify element-wise multiplication with high precision
  for (size_t i = 0; i < input_a.size(); ++i) {
    double expected = input_a[i] * input_b[i];
    BOOST_CHECK_CLOSE(output[i], expected, 1e-12); // Higher precision check
  }
}

/*!
 * \brief Test error handling for mismatched shapes
 *
 * Tests that the operation returns appropriate error codes when
 * input matrix shapes don't match
 */
BOOST_AUTO_TEST_CASE(test_matrix_elementwise_multiply_shape_mismatch) {
  types::shape shape{2, 2};
  const int num_inputs = 2;

  auto mult_op =
      matrix_elementwise_multiply_sync<float>::make(shape, num_inputs);

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
  OperationReturn result = mult_op->operation(inputs, outputs);

  BOOST_REQUIRE_EQUAL(result, OperationReturn::INVALID_SHAPE);
}

/*!
 * \brief Test insufficient inputs error handling
 *
 * Tests error handling when fewer than 2 inputs are provided
 */
BOOST_AUTO_TEST_CASE(test_matrix_elementwise_multiply_insufficient_inputs) {
  types::shape shape{2, 2};
  const int num_inputs = 2;

  auto mult_op =
      matrix_elementwise_multiply_sync<float>::make(shape, num_inputs);

  // Create only one input (insufficient)
  std::vector<float> input_a{1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> output(4);

  types::matrix_map_dynamic<float> map_a(input_a.data(), 2, 2);
  types::matrix_map_dynamic<float> map_out(output.data(), 2, 2);

  // Provide only one input (should require at least 2)
  types::vector_const_matrix_map<float> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&map_a)};
  types::vector_matrix_map<float> outputs{&map_out};

  // Perform operation - should fail due to insufficient inputs
  OperationReturn result = mult_op->operation(inputs, outputs);

  BOOST_REQUIRE_EQUAL(result, OperationReturn::INVALID_SHAPE);
}

/*!
 * \brief Test vector (1D) element-wise multiplication
 *
 * Tests element-wise multiplication of vectors:
 * a = [1, 2, 3, 4]
 * b = [2, 3, 4, 5]
 * Expected result: a ⊙ b = [2, 6, 12, 20]
 */
BOOST_AUTO_TEST_CASE(test_matrix_elementwise_multiply_vector) {
  types::shape shape{4, 1}; // Column vector
  const int num_inputs = 2;

  auto mult_op =
      matrix_elementwise_multiply_sync<float>::make(shape, num_inputs);

  // Create input vectors as column matrices
  std::vector<float> input_a{1.0f, 2.0f, 3.0f, 4.0f};
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
  OperationReturn result = mult_op->operation(inputs, outputs);

  BOOST_REQUIRE_EQUAL(result, OperationReturn::SUCCESS);

  // Check results: [1*2, 2*3, 3*4, 4*5] = [2, 6, 12, 20]
  std::vector<float> expected{2.0f, 6.0f, 12.0f, 20.0f};
  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_CHECK_CLOSE(output[i], expected[i], 1e-6);
  }
}

//==============================================================================
// GNU Radio Flowgraph Tests
//==============================================================================

/*!
 * \brief Test matrix elementwise multiply sync block in GNU Radio flowgraph
 *
 * Tests basic 2×2 element-wise multiplication through actual signal processing
 * pipeline: Input A: [[2, 4], [3, 6]] (column-major: [2, 3, 4, 6]) Input B:
 * [[3, 2], [4, 5]] (column-major: [3, 4, 2, 5]) Expected result: A ⊙ B = [[6,
 * 8], [12, 30]] (column-major: [6, 12, 8, 30])
 */
BOOST_AUTO_TEST_CASE(test_matrix_elementwise_multiply_sync_flowgraph) {
  BOOST_TEST_MESSAGE("Testing matrix elementwise multiply sync block in GNU "
                     "Radio flowgraph...");

  auto tb = gr::make_top_block("elementwise_multiply_test");

  // Create input matrices: A = [[2, 4], [3, 6]], B = [[3, 2], [4, 5]]
  std::vector<float> input_a = {2.0f, 3.0f, 4.0f, 6.0f}; // Matrix A
  std::vector<float> input_b = {3.0f, 4.0f, 2.0f, 5.0f}; // Matrix B

  auto vector_source_a = gr::blocks::vector_source_f::make(input_a, false, 4);
  auto vector_source_b = gr::blocks::vector_source_f::make(input_b, false, 4);
  auto multiply_block =
      matrix_elementwise_multiply_sync<float>::make({2, 2}, 2);
  auto output_sink = gr::blocks::vector_sink_f::make(4);

  tb->connect(vector_source_a, 0, multiply_block, 0);
  tb->connect(vector_source_b, 0, multiply_block, 1);
  tb->connect(multiply_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 4);

  // Expected result: [2*3, 3*4, 4*2, 6*5] = [6, 12, 8, 30]
  std::vector<float> expected = {6.0f, 12.0f, 8.0f, 30.0f};
  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_CHECK_CLOSE(output_data[i], expected[i], FLOAT_TOLERANCE);
  }

  BOOST_TEST_MESSAGE("Matrix elementwise multiply sync flowgraph test passed.");
}

/*!
 * \brief Test elementwise multiply with multiple inputs in flowgraph
 *
 * Tests chained multiplication A ⊙ B ⊙ C through GNU Radio pipeline:
 * A = [[2, 4], [3, 6]]
 * B = [[2, 2], [3, 2]]
 * C = [[3, 2], [2, 1]]
 * Expected: ((A ⊙ B) ⊙ C) = [[12, 16], [18, 12]]
 */
BOOST_AUTO_TEST_CASE(
    test_matrix_elementwise_multiply_sync_flowgraph_multiple_inputs) {
  BOOST_TEST_MESSAGE("Testing elementwise multiply sync block with multiple "
                     "inputs in flowgraph...");

  auto tb = gr::make_top_block("elementwise_multiply_multi_test");

  // Create input matrices for chained multiplication
  std::vector<float> input_a = {2.0f, 3.0f, 4.0f, 6.0f}; // Matrix A
  std::vector<float> input_b = {2.0f, 3.0f, 2.0f, 2.0f}; // Matrix B
  std::vector<float> input_c = {3.0f, 2.0f, 2.0f, 1.0f}; // Matrix C

  auto vector_source_a = gr::blocks::vector_source_f::make(input_a, false, 4);
  auto vector_source_b = gr::blocks::vector_source_f::make(input_b, false, 4);
  auto vector_source_c = gr::blocks::vector_source_f::make(input_c, false, 4);
  auto multiply_block =
      matrix_elementwise_multiply_sync<float>::make({2, 2}, 3);
  auto output_sink = gr::blocks::vector_sink_f::make(4);

  tb->connect(vector_source_a, 0, multiply_block, 0);
  tb->connect(vector_source_b, 0, multiply_block, 1);
  tb->connect(vector_source_c, 0, multiply_block, 2);
  tb->connect(multiply_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 4);

  // Expected result: ((2*2*3, 3*3*2, 4*2*2, 6*2*1) = (12, 18, 16, 12)
  std::vector<float> expected = {12.0f, 18.0f, 16.0f, 12.0f};
  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_CHECK_CLOSE(output_data[i], expected[i], FLOAT_TOLERANCE);
  }

  BOOST_TEST_MESSAGE(
      "Multiple inputs elementwise multiply flowgraph test passed.");
}

/*!
 * \brief Test elementwise multiply with complex numbers in flowgraph
 *
 * Tests complex element-wise multiplication through GNU Radio pipeline:
 * A = [[1+1j, 2+0j], [0+2j, 1+1j]]
 * B = [[2+0j, 1+1j], [1-1j, 2+0j]]
 * Expected computed element-wise
 */
BOOST_AUTO_TEST_CASE(test_matrix_elementwise_multiply_sync_flowgraph_complex) {
  BOOST_TEST_MESSAGE("Testing elementwise multiply sync block with complex "
                     "numbers in flowgraph...");

  using cf = std::complex<float>;
  auto tb = gr::make_top_block("elementwise_multiply_complex_test");

  // Create complex input matrices
  std::vector<cf> input_a = {cf(1, 1), cf(0, 2), cf(2, 0), cf(1, 1)};
  std::vector<cf> input_b = {cf(2, 0), cf(1, -1), cf(1, 1), cf(2, 0)};

  auto vector_source_a = gr::blocks::vector_source_c::make(input_a, false, 4);
  auto vector_source_b = gr::blocks::vector_source_c::make(input_b, false, 4);
  auto multiply_block = matrix_elementwise_multiply_sync<cf>::make({2, 2}, 2);
  auto output_sink = gr::blocks::vector_sink_c::make(4);

  tb->connect(vector_source_a, 0, multiply_block, 0);
  tb->connect(vector_source_b, 0, multiply_block, 1);
  tb->connect(multiply_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 4);

  // Expected results (computed manually):
  // (1+1j)*(2+0j) = 2+2j, (0+2j)*(1-1j) = 2+2j
  // (2+0j)*(1+1j) = 2+2j, (1+1j)*(2+0j) = 2+2j
  std::vector<cf> expected = {cf(2, 2), cf(2, 2), cf(2, 2), cf(2, 2)};

  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_CHECK_CLOSE(output_data[i].real(), expected[i].real(),
                      FLOAT_TOLERANCE);
    BOOST_CHECK_CLOSE(output_data[i].imag(), expected[i].imag(),
                      FLOAT_TOLERANCE);
  }

  BOOST_TEST_MESSAGE("Complex elementwise multiply flowgraph test passed.");
}

/*!
 * \brief Test elementwise multiply with double precision in flowgraph
 *
 * Tests high precision element-wise multiplication through GNU Radio pipeline
 */
BOOST_AUTO_TEST_CASE(test_matrix_elementwise_multiply_sync_flowgraph_double) {
  BOOST_TEST_MESSAGE("Testing elementwise multiply sync block with double "
                     "precision in flowgraph...");

  auto tb = gr::make_top_block("elementwise_multiply_double_test");

  // Create double precision matrices (3×3)
  std::vector<double> input_a = {1.123456789, 2.271828183, 3.141592654,
                                 4.987654321, 5.000000001, 6.456456456,
                                 7.999999999, 8.123123123, 9.789789789};

  std::vector<double> input_b = {2.123456789, 1.271828183, 2.141592654,
                                 1.987654321, 3.000000001, 1.456456456,
                                 2.999999999, 3.123123123, 1.789789789};

  // Convert double to float for compatibility with available GNU Radio blocks
  std::vector<float> input_a_f(input_a.begin(), input_a.end());
  std::vector<float> input_b_f(input_b.begin(), input_b.end());

  auto vector_source_a = gr::blocks::vector_source_f::make(input_a_f, false, 9);
  auto vector_source_b = gr::blocks::vector_source_f::make(input_b_f, false, 9);
  auto multiply_block =
      matrix_elementwise_multiply_sync<float>::make({3, 3}, 2);
  auto output_sink = gr::blocks::vector_sink_f::make(9);

  tb->connect(vector_source_a, 0, multiply_block, 0);
  tb->connect(vector_source_b, 0, multiply_block, 1);
  tb->connect(multiply_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 9);

  // Verify element-wise multiplication (using relaxed tolerance for float
  // precision)
  for (size_t i = 0; i < input_a.size(); ++i) {
    float expected = static_cast<float>(input_a[i] * input_b[i]);
    BOOST_CHECK_CLOSE(output_data[i], expected,
                      1e-4f); // More relaxed tolerance for float precision
  }

  BOOST_TEST_MESSAGE(
      "Double precision elementwise multiply flowgraph test passed.");
}

/*!
 * \brief Test elementwise multiply with zero elements in flowgraph
 *
 * Tests multiplication with zero elements through GNU Radio pipeline
 */
BOOST_AUTO_TEST_CASE(
    test_matrix_elementwise_multiply_sync_flowgraph_zero_handling) {
  BOOST_TEST_MESSAGE("Testing elementwise multiply sync block with zero "
                     "elements in flowgraph...");

  auto tb = gr::make_top_block("elementwise_multiply_zero_test");

  // Create matrices with zero elements
  std::vector<float> input_a = {4.0f, 0.0f, -3.0f, 2.0f};
  std::vector<float> input_b = {0.0f, 5.0f, 2.0f, 0.0f}; // Contains zeros

  auto vector_source_a = gr::blocks::vector_source_f::make(input_a, false, 4);
  auto vector_source_b = gr::blocks::vector_source_f::make(input_b, false, 4);
  auto multiply_block =
      matrix_elementwise_multiply_sync<float>::make({2, 2}, 2);
  auto output_sink = gr::blocks::vector_sink_f::make(4);

  tb->connect(vector_source_a, 0, multiply_block, 0);
  tb->connect(vector_source_b, 0, multiply_block, 1);
  tb->connect(multiply_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 4);

  // Expected results: [4*0, 0*5, -3*2, 2*0] = [0, 0, -6, 0]
  std::vector<float> expected = {0.0f, 0.0f, -6.0f, 0.0f};
  for (size_t i = 0; i < expected.size(); ++i) {
    if (expected[i] == 0.0f) {
      BOOST_CHECK_SMALL(output_data[i], FLOAT_TOLERANCE);
    } else {
      BOOST_CHECK_CLOSE(output_data[i], expected[i], FLOAT_TOLERANCE);
    }
  }

  BOOST_TEST_MESSAGE(
      "Zero handling elementwise multiply flowgraph test passed.");
}

/*!
 * \brief Test elementwise multiply with vector data in flowgraph
 *
 * Tests element-wise multiplication of vectors through GNU Radio pipeline:
 * a = [2, 3, 4, 5] (column vector)
 * b = [3, 2, 3, 2] (column vector)
 * Expected: a ⊙ b = [6, 6, 12, 10]
 */
BOOST_AUTO_TEST_CASE(test_matrix_elementwise_multiply_sync_flowgraph_vector) {
  BOOST_TEST_MESSAGE(
      "Testing elementwise multiply sync block with vectors in flowgraph...");

  auto tb = gr::make_top_block("elementwise_multiply_vector_test");

  // Create input vectors as column matrices
  std::vector<float> input_a = {2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<float> input_b = {3.0f, 2.0f, 3.0f, 2.0f};

  auto vector_source_a = gr::blocks::vector_source_f::make(input_a, false, 4);
  auto vector_source_b = gr::blocks::vector_source_f::make(input_b, false, 4);
  auto multiply_block =
      matrix_elementwise_multiply_sync<float>::make({4, 1}, 2); // Column vector
  auto output_sink = gr::blocks::vector_sink_f::make(4);

  tb->connect(vector_source_a, 0, multiply_block, 0);
  tb->connect(vector_source_b, 0, multiply_block, 1);
  tb->connect(multiply_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 4);

  // Expected result: [2*3, 3*2, 4*3, 5*2] = [6, 6, 12, 10]
  std::vector<float> expected = {6.0f, 6.0f, 12.0f, 10.0f};
  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_CHECK_CLOSE(output_data[i], expected[i], FLOAT_TOLERANCE);
  }

  BOOST_TEST_MESSAGE("Vector elementwise multiply flowgraph test passed.");
}

/*!
 * \brief Test elementwise multiply with large numbers in flowgraph
 *
 * Tests multiplication of large numbers through GNU Radio pipeline
 */
BOOST_AUTO_TEST_CASE(
    test_matrix_elementwise_multiply_sync_flowgraph_large_numbers) {
  BOOST_TEST_MESSAGE("Testing elementwise multiply sync block with large "
                     "numbers in flowgraph...");

  auto tb = gr::make_top_block("elementwise_multiply_large_test");

  // Create matrices with large numbers
  std::vector<float> input_a = {1e6f, 2e6f, 3e6f, 4e6f};
  std::vector<float> input_b = {2e3f, 3e3f, 4e3f,
                                5e3f}; // Large but manageable numbers

  auto vector_source_a = gr::blocks::vector_source_f::make(input_a, false, 4);
  auto vector_source_b = gr::blocks::vector_source_f::make(input_b, false, 4);
  auto multiply_block =
      matrix_elementwise_multiply_sync<float>::make({2, 2}, 2);
  auto output_sink = gr::blocks::vector_sink_f::make(4);

  tb->connect(vector_source_a, 0, multiply_block, 0);
  tb->connect(vector_source_b, 0, multiply_block, 1);
  tb->connect(multiply_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 4);

  // Results should be large but finite numbers
  for (size_t i = 0; i < output_data.size(); ++i) {
    float expected = input_a[i] * input_b[i];
    BOOST_CHECK_CLOSE(output_data[i], expected,
                      1e-4f); // Allow some numerical error for large numbers
    BOOST_CHECK(std::isfinite(output_data[i])); // Should be finite, not inf
  }

  BOOST_TEST_MESSAGE(
      "Large numbers elementwise multiply flowgraph test passed.");
}

/*!
 * \brief Test elementwise multiply with multiple matrices in flowgraph
 *
 * Tests processing of multiple consecutive matrices through GNU Radio pipeline
 */
BOOST_AUTO_TEST_CASE(
    test_matrix_elementwise_multiply_sync_flowgraph_multiple_matrices) {
  BOOST_TEST_MESSAGE("Testing elementwise multiply sync block with multiple "
                     "matrices in flowgraph...");

  auto tb = gr::make_top_block("elementwise_multiply_multi_matrices_test");

  // Create multiple 2x2 matrices for consecutive processing
  std::vector<float> input_a = {// Matrix 1: [[2, 4], [3, 6]]
                                2.0f, 3.0f, 4.0f, 6.0f,
                                // Matrix 2: [[1, 3], [2, 4]]
                                1.0f, 2.0f, 3.0f, 4.0f,
                                // Matrix 3: [[4, 2], [5, 3]]
                                4.0f, 5.0f, 2.0f, 3.0f};

  std::vector<float> input_b = {// Matrix 1: [[3, 2], [4, 5]]
                                3.0f, 4.0f, 2.0f, 5.0f,
                                // Matrix 2: [[2, 1], [3, 2]]
                                2.0f, 3.0f, 1.0f, 2.0f,
                                // Matrix 3: [[1, 2], [2, 3]]
                                1.0f, 2.0f, 2.0f, 3.0f};

  auto vector_source_a = gr::blocks::vector_source_f::make(input_a, false, 4);
  auto vector_source_b = gr::blocks::vector_source_f::make(input_b, false, 4);
  auto multiply_block =
      matrix_elementwise_multiply_sync<float>::make({2, 2}, 2);
  auto output_sink = gr::blocks::vector_sink_f::make(4);

  tb->connect(vector_source_a, 0, multiply_block, 0);
  tb->connect(vector_source_b, 0, multiply_block, 1);
  tb->connect(multiply_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 12); // 3 matrices × 4 elements each

  // Expected results for each matrix
  std::vector<std::vector<float>> expected_matrices = {
      {6.0f, 12.0f, 8.0f, 30.0f}, // Matrix 1: [2*3, 3*4, 4*2, 6*5]
      {2.0f, 6.0f, 3.0f, 8.0f},   // Matrix 2: [1*2, 2*3, 3*1, 4*2]
      {4.0f, 10.0f, 4.0f, 9.0f}   // Matrix 3: [4*1, 5*2, 2*2, 3*3]
  };

  for (int matrix = 0; matrix < 3; ++matrix) {
    for (int i = 0; i < 4; ++i) {
      size_t idx = matrix * 4 + i;
      BOOST_CHECK_CLOSE(output_data[idx], expected_matrices[matrix][i],
                        FLOAT_TOLERANCE);
    }
  }

  BOOST_TEST_MESSAGE(
      "Multiple matrices elementwise multiply flowgraph test passed.");
}

/*!
 * \brief Test elementwise multiply performance in flowgraph
 *
 * Tests performance with a large number of matrices through GNU Radio pipeline
 */
BOOST_AUTO_TEST_CASE(
    test_matrix_elementwise_multiply_sync_flowgraph_performance) {
  BOOST_TEST_MESSAGE(
      "Testing elementwise multiply sync block performance in flowgraph...");

  auto tb = gr::make_top_block("elementwise_multiply_perf_test");

  // Create many 2x2 matrices for performance testing
  int num_matrices = 1000;
  std::vector<float> input_a, input_b;
  input_a.reserve(num_matrices * 4);
  input_b.reserve(num_matrices * 4);

  for (int i = 0; i < num_matrices; ++i) {
    // Generate matrices with varying values for performance test
    float base = 1.0f + 0.001f * i;
    input_a.insert(input_a.end(),
                   {2.0f * base, 3.0f * base, 4.0f * base, 6.0f * base});
    input_b.insert(input_b.end(),
                   {3.0f * base, 4.0f * base, 2.0f * base, 5.0f * base});
  }

  auto start_time = std::chrono::high_resolution_clock::now();

  auto vector_source_a = gr::blocks::vector_source_f::make(input_a, false, 4);
  auto vector_source_b = gr::blocks::vector_source_f::make(input_b, false, 4);
  auto multiply_block =
      matrix_elementwise_multiply_sync<float>::make({2, 2}, 2);
  auto output_sink = gr::blocks::vector_sink_f::make(4);

  tb->connect(vector_source_a, 0, multiply_block, 0);
  tb->connect(vector_source_b, 0, multiply_block, 1);
  tb->connect(multiply_block, 0, output_sink, 0);

  tb->run();

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), num_matrices * 4);

  // Verify first and last matrix results (accounting for scaling factor)
  BOOST_CHECK_CLOSE(output_data[0], 6.0f,
                    FLOAT_TOLERANCE); // First matrix first element: 2*3
  BOOST_CHECK_CLOSE(output_data[1], 12.0f,
                    FLOAT_TOLERANCE); // First matrix second element: 3*4

  size_t last_matrix_idx = (num_matrices - 1) * 4;
  // For the last matrix, the scaling factor is base = 1.0f + 0.001f *
  // (num_matrices-1) = 1.999f
  float last_base = 1.0f + 0.001f * (num_matrices - 1);
  float expected_last_0 =
      (2.0f * last_base) * (3.0f * last_base); // Should be ~24
  float expected_last_1 =
      (3.0f * last_base) * (4.0f * last_base); // Should be ~48
  BOOST_CHECK_CLOSE(output_data[last_matrix_idx], expected_last_0,
                    1e-2f); // Last matrix first element
  BOOST_CHECK_CLOSE(output_data[last_matrix_idx + 1], expected_last_1,
                    1e-2f); // Last matrix second element

  // All results should be finite
  for (size_t i = 0; i < output_data.size(); ++i) {
    BOOST_CHECK(std::isfinite(output_data[i]));
  }

  BOOST_TEST_MESSAGE("Processed " << num_matrices << " matrices in "
                                  << duration.count() << " ms");
  BOOST_TEST_MESSAGE("Elementwise multiply performance flowgraph test passed.");
}

} // namespace linalg
} // namespace gr