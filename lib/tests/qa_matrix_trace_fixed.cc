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

  // 2x2 identity matrix [1 0; 0 1] - trace should be 2
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

  // 3x3 diagonal matrix [1 0 0; 0 2 0; 0 0 3] - trace should be 6
  std::vector<float> input_data = {1.0f, 0.0f, 0.0f, 0.0f, 2.0f,
                                   0.0f, 0.0f, 0.0f, 3.0f};
  float expected_trace = 6.0f;

  auto src = gr::blocks::vector_source_f::make(input_data, false, 9);
  auto trace_block = matrix_trace<float>::make({3, 3});
  auto sink = gr::blocks::vector_sink_f::make(1);

  tb->connect(src, 0, trace_block, 0);
  tb->connect(trace_block, 0, sink, 0);

  tb->run();

  auto output_data = sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 1u);
  BOOST_CHECK_CLOSE(output_data[0], expected_trace, 1e-6);
}

BOOST_AUTO_TEST_CASE(test_matrix_trace_2x2_general) {
  auto tb = gr::make_top_block("test");

  // 2x2 matrix [4 7; 2 6] - trace should be 4+6 = 10
  std::vector<float> input_data = {4.0f, 7.0f, 2.0f, 6.0f};
  float expected_trace = 10.0f;

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

} /* namespace linalg */
} /* namespace gr */
