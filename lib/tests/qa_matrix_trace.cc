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
#include <gnuradio/linalg/matrix_trace.h>
#include <gnuradio/top_block.h>
#include <vector>

namespace gr {
namespace linalg {

BOOST_AUTO_TEST_CASE(test_matrix_trace_2x2_identity) {
  auto tb = gr::make_top_block("test");

  // 2x2 identity matrix [1 0; 0 1] - trace should be 1+1 = 2
  std::vector<float> input_data = {1.0f, 0.0f, 0.0f, 1.0f};
  float expected_trace = 2.0f;

  auto src = gr::blocks::vector_source_f::make(input_data, false, 4);
  auto trace_block = matrix_trace<float>::make({2, 2});
  auto sink = gr::blocks::vector_sink_f::make(1);

  tb->connect(src, 0, trace_block, 0);
  tb->connect(trace_block, 0, sink, 0);

  tb->run();

  auto output_data = sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 1u);
  BOOST_CHECK_CLOSE(output_data[0], expected_trace, 1e-6);
}

BOOST_AUTO_TEST_CASE(test_matrix_trace_3x3_diagonal) {
  auto tb = gr::make_top_block("test");

  // 3x3 diagonal matrix [2 0 0; 0 3 0; 0 0 4] - trace should be 2+3+4 = 9
  double input_data[] = {2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0};
  double expected_trace = 9.0;

  auto src = gr::blocks::vector_source_d::make(input_data, 9, false, 9);
  auto trace_block = matrix_trace<double>::make({3, 3});
  auto sink = gr::blocks::vector_sink_d::make(1);

  tb->connect(src, 0, trace_block, 0);
  tb->connect(trace_block, 0, sink, 0);

  tb->run();

  auto output_data = sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 1);
  BOOST_CHECK_CLOSE(output_data[0], expected_trace, 1e-12);
}

BOOST_AUTO_TEST_CASE(test_matrix_trace_2x2_general) {
  auto tb = gr::make_top_block("test");

  // 2x2 matrix [5 3; 7 2] - trace should be 5+2 = 7
  float input_data[] = {5.0f, 3.0f, 7.0f, 2.0f};
  float expected_trace = 7.0f;

  auto src = gr::blocks::vector_source_f::make(input_data, 4, false, 4);
  auto trace_block = matrix_trace<float>::make({2, 2});
  auto sink = gr::blocks::vector_sink_f::make(1);

  tb->connect(src, 0, trace_block, 0);
  tb->connect(trace_block, 0, sink, 0);

  tb->run();

  auto output_data = sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 1);
  BOOST_CHECK_CLOSE(output_data[0], expected_trace, 1e-6);
}

BOOST_AUTO_TEST_CASE(test_matrix_trace_4x4_zeros) {
  auto tb = gr::make_top_block("test");

  // 4x4 zero matrix - trace should be 0
  float input_data[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float expected_trace = 0.0f;

  auto src = gr::blocks::vector_source_f::make(input_data, 16, false, 16);
  auto trace_block = matrix_trace<float>::make({4, 4});
  auto sink = gr::blocks::vector_sink_f::make(1);

  tb->connect(src, 0, trace_block, 0);
  tb->connect(trace_block, 0, sink, 0);

  tb->run();

  auto output_data = sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 1);
  BOOST_CHECK_SMALL(output_data[0], 1e-6f);
}

BOOST_AUTO_TEST_CASE(test_matrix_trace_multiple_matrices) {
  auto tb = gr::make_top_block("test");

  // Two 2x2 matrices: [1 2; 3 4] (trace=5) and [6 7; 8 9] (trace=15)
  float input_data[] = {1.0f, 2.0f, 3.0f, 4.0f,  // First matrix
                        6.0f, 7.0f, 8.0f, 9.0f}; // Second matrix
  float expected_traces[] = {5.0f, 15.0f};

  auto src = gr::blocks::vector_source_f::make(input_data, 8, false, 4);
  auto trace_block = matrix_trace<float>::make({2, 2});
  auto sink = gr::blocks::vector_sink_f::make(1);

  tb->connect(src, 0, trace_block, 0);
  tb->connect(trace_block, 0, sink, 0);

  tb->run();

  auto output_data = sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 2);
  for (int i = 0; i < 2; ++i) {
    BOOST_CHECK_CLOSE(output_data[i], expected_traces[i], 1e-6);
  }
}

} /* namespace linalg */
} /* namespace gr */
