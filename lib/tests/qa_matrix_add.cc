/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "../matrix_add_impl.h"
#include <boost/test/unit_test.hpp>
#include <cstddef> // For std::size_t
#include <gnuradio/attributes.h>
#include <gnuradio/block.h>
#include <gnuradio/blocks/vector_sink.h>
#include <gnuradio/blocks/vector_source.h>
#include <gnuradio/io_signature.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/matrix_add.h>
#include <gnuradio/linalg/types.h>
#include <gnuradio/top_block.h>

// Backward compatibility alias expected by legacy test name
using matrix_add_f = gr::linalg::matrix_add_sync_f;

namespace gr {
namespace linalg {

BOOST_AUTO_TEST_CASE(test_matrix_add_sync_block) {
  BOOST_TEST_MESSAGE(
      "Testing matrix add sync block creation and basic functionality...");

  auto shape = types::shape({3, 3}); // 3x3 matrix shape
  auto num_inputs = 2;               // Number of input matrices

  // Create the matrix add sync block
  auto block_matrix_add = matrix_add_f::make(shape, num_inputs);
  BOOST_REQUIRE(block_matrix_add);
  BOOST_TEST_MESSAGE("Matrix add sync block created successfully.");

  // Test that the block has correct input/output signature
  auto input_sig = block_matrix_add->input_signature();
  auto output_sig = block_matrix_add->output_signature();

  BOOST_CHECK_EQUAL(input_sig->max_streams(), num_inputs);
  BOOST_CHECK_EQUAL(output_sig->max_streams(), 1);

  BOOST_TEST_MESSAGE("Matrix add sync block basic functionality test passed.");
}

BOOST_AUTO_TEST_CASE(test_matrix_add_flowgraph_integration) {
  BOOST_TEST_MESSAGE(
      "Testing matrix add sync block GNU Radio flowgraph integration...");

  // NOTE: This test demonstrates the approach for full GNU Radio integration
  // testing The actual flowgraph execution may require resolving pointer type
  // compatibility between boost::shared_ptr (linalg blocks) and std::shared_ptr
  // (GNU Radio)

  auto shape = types::shape({2, 2}); // Use smaller 2x2 for easier verification
  auto num_inputs = 2;

  // Create test matrices: A + B = C
  std::vector<float> matrix_A = {1.0f, 2.0f, 3.0f, 4.0f};     // [[1,2],[3,4]]
  std::vector<float> matrix_B = {5.0f, 6.0f, 7.0f, 8.0f};     // [[5,6],[7,8]]
  std::vector<float> expected_C = {6.0f, 8.0f, 10.0f, 12.0f}; // [[6,8],[10,12]]

  // Create GNU Radio blocks
  auto matrix_add_block = matrix_add_f::make(shape, num_inputs);
  BOOST_REQUIRE(matrix_add_block);

  // Test the work method directly (this tests actual GNU Radio sync block
  // processing)
  BOOST_TEST_MESSAGE("Testing matrix addition work() method directly...");

  // Create input and output buffers (simulating GNU Radio's buffer system)
  std::vector<const void *> input_items(2); // 2 input matrices
  std::vector<void *> output_items(1);      // 1 output matrix
  std::vector<float> output_buffer(4);      // 2x2 output matrix = 4 elements

  input_items[0] = matrix_A.data();
  input_items[1] = matrix_B.data();
  output_items[0] = output_buffer.data();

  // Simulate calling work() method (this tests the actual GNU Radio sync block
  // processing)
  int noutput_items = 1; // Process 1 matrix
  int result = matrix_add_block->work(noutput_items, input_items, output_items);

  // Verify work() processed the data
  BOOST_CHECK_EQUAL(result, 1);

  // Verify the matrix addition result
  for (size_t i = 0; i < expected_C.size(); ++i) {
    BOOST_CHECK_CLOSE(output_buffer[i], expected_C[i], 1e-5f);
  }

  BOOST_TEST_MESSAGE("Matrix add work() method functionality test passed.");

  // TODO: Full GNU Radio flowgraph test would look like this:
  // (Currently commented due to pointer type compatibility issues)
  /*
  auto tb = gr::make_top_block("test_matrix_add");
  auto source_A = gr::blocks::vector_source_f::make(matrix_A, false, 4);
  auto source_B = gr::blocks::vector_source_f::make(matrix_B, false, 4);
  auto sink = gr::blocks::vector_sink_f::make(4);

  // This would require resolving boost::shared_ptr vs std::shared_ptr
  compatibility tb->connect(source_A, 0, matrix_add_block, 0);
  tb->connect(source_B, 0, matrix_add_block, 1);
  tb->connect(matrix_add_block, 0, sink, 0);

  tb->run();
  auto output = sink->data();

  for (size_t i = 0; i < expected_C.size(); ++i) {
    BOOST_CHECK_CLOSE(output[i], expected_C[i], 1e-5f);
  }
  */
}

BOOST_AUTO_TEST_CASE(test_matrix_add_complex_float_work) {
  auto shape = types::shape({2, 2});
  auto num_inputs = 2;
  auto block = matrix_add_sync<std::complex<float>>::make(shape, num_inputs);
  BOOST_REQUIRE(block);

  std::vector<std::complex<float>> A = {
      {1.f, 2.f}, {3.f, 4.f}, {5.f, 6.f}, {7.f, 8.f}};
  std::vector<std::complex<float>> B = {
      {-1.f, 1.f}, {-2.f, 2.f}, {-3.f, 3.f}, {-4.f, 4.f}};
  std::vector<std::complex<float>> out(4);

  std::vector<const void *> in(2);
  std::vector<void *> outv(1);
  in[0] = A.data();
  in[1] = B.data();
  outv[0] = out.data();

  int r = block->work(1, in, outv);
  BOOST_CHECK_EQUAL(r, 1);
  for (size_t i = 0; i < out.size(); ++i) {
    auto expected = A[i] + B[i];
    BOOST_CHECK_SMALL(std::abs(out[i] - expected), 1e-5f);
  }
}

BOOST_AUTO_TEST_CASE(test_matrix_add_complex_double_work) {
  auto shape = types::shape({2, 2});
  auto num_inputs = 2;
  auto block = matrix_add_sync<std::complex<double>>::make(shape, num_inputs);
  BOOST_REQUIRE(block);

  std::vector<std::complex<double>> A = {
      {1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}};
  std::vector<std::complex<double>> B = {
      {-1.0, 1.0}, {-2.0, 2.0}, {-3.0, 3.0}, {-4.0, 4.0}};
  std::vector<std::complex<double>> out(4);

  std::vector<const void *> in(2);
  std::vector<void *> outv(1);
  in[0] = A.data();
  in[1] = B.data();
  outv[0] = out.data();

  int r = block->work(1, in, outv);
  BOOST_CHECK_EQUAL(r, 1);
  for (size_t i = 0; i < out.size(); ++i) {
    auto expected = A[i] + B[i];
    BOOST_CHECK_SMALL(std::abs(out[i] - expected), 1e-10);
  }
}

} /* namespace linalg */
} /* namespace gr */
