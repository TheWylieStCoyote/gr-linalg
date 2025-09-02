/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <gnuradio/attributes.h>
#include <gnuradio/blocks/vector_sink.h>
#include <gnuradio/blocks/vector_source.h>
#include <gnuradio/linalg/matrix_transpose.h>
#include <gnuradio/top_block.h>
#include <vector>

namespace gr {
namespace linalg {

BOOST_AUTO_TEST_CASE(test_matrix_transpose_2x3_float) {
  auto tb = gr::make_top_block("test");

  // Create a 2x3 input matrix [1 2 3; 4 5 6]
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> expected_output = {1.0f, 4.0f, 2.0f,
                                        5.0f, 3.0f, 6.0f}; // 3x2 transposed

  auto src = gr::blocks::vector_source_f::make(input_data, false, 6);
  auto transpose_block = matrix_transpose<float>::make({2, 3});
  auto sink = gr::blocks::vector_sink_f::make(6);

  tb->connect(src, 0, transpose_block, 0);
  tb->connect(transpose_block, 0, sink, 0);

  tb->run();

  auto output_data = sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), expected_output.size());

  for (size_t i = 0; i < expected_output.size(); ++i) {
    BOOST_CHECK_CLOSE(output_data[i], expected_output[i], 1e-6);
  }
}

BOOST_AUTO_TEST_CASE(test_matrix_transpose_3x2_double) {
  auto tb = gr::make_top_block("test");

  // Create a 3x2 input matrix [1 2; 3 4; 5 6]
  std::vector<double> input_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  std::vector<double> expected_output = {1.0, 3.0, 5.0,
                                         2.0, 4.0, 6.0}; // 2x3 transposed

  auto src = gr::blocks::vector_source_d::make(input_data, false, 6);
  auto transpose_block = matrix_transpose<double>::make({3, 2});
  auto sink = gr::blocks::vector_sink_d::make(6);

  tb->connect(src, 0, transpose_block, 0);
  tb->connect(transpose_block, 0, sink, 0);

  tb->run();

  auto output_data = sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), expected_output.size());

  for (size_t i = 0; i < expected_output.size(); ++i) {
    BOOST_CHECK_CLOSE(output_data[i], expected_output[i], 1e-12);
  }
}

BOOST_AUTO_TEST_CASE(test_matrix_transpose_square_matrix) {
  auto tb = gr::make_top_block("test");

  // Create a 3x3 identity-like matrix
  std::vector<float> input_data = {1.0f, 0.0f, 0.0f, 0.0f, 2.0f,
                                   0.0f, 0.0f, 0.0f, 3.0f};
  std::vector<float> expected_output = {
      1.0f, 0.0f, 0.0f, 0.0f, 2.0f,
      0.0f, 0.0f, 0.0f, 3.0f}; // Same for diagonal matrix

  auto src = gr::blocks::vector_source_f::make(input_data, false, 9);
  auto transpose_block = matrix_transpose<float>::make({3, 3});
  auto sink = gr::blocks::vector_sink_f::make(9);

  tb->connect(src, 0, transpose_block, 0);
  tb->connect(transpose_block, 0, sink, 0);

  tb->run();

  auto output_data = sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), expected_output.size());

  for (size_t i = 0; i < expected_output.size(); ++i) {
    BOOST_CHECK_CLOSE(output_data[i], expected_output[i], 1e-6);
  }
}

BOOST_AUTO_TEST_CASE(test_matrix_transpose_multiple_matrices) {
  auto tb = gr::make_top_block("test");

  // Create two 2x2 matrices: [1 2; 3 4] and [5 6; 7 8]
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f,  // First matrix
                                   5.0f, 6.0f, 7.0f, 8.0f}; // Second matrix
  std::vector<float> expected_output = {
      1.0f, 3.0f, 2.0f, 4.0f,  // First transposed
      5.0f, 7.0f, 6.0f, 8.0f}; // Second transposed

  auto src = gr::blocks::vector_source_f::make(input_data, false, 4);
  auto transpose_block = matrix_transpose<float>::make({2, 2});
  auto sink = gr::blocks::vector_sink_f::make(4);

  tb->connect(src, 0, transpose_block, 0);
  tb->connect(transpose_block, 0, sink, 0);

  tb->run();

  auto output_data = sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), expected_output.size());

  for (size_t i = 0; i < expected_output.size(); ++i) {
    BOOST_CHECK_CLOSE(output_data[i], expected_output[i], 1e-6);
  }
}

} /* namespace linalg */
} /* namespace gr */
