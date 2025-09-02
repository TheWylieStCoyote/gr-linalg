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
#include <gnuradio/linalg/matrix_determinant.h>
#include <gnuradio/linalg/types.h>
#include <gnuradio/top_block.h>
#include <vector>

namespace gr {
namespace linalg {

BOOST_AUTO_TEST_CASE(test_matrix_determinant_2x2_identity) {
  auto tb = gr::make_top_block("test");

  // 2x2 identity matrix [1 0; 0 1] - determinant should be 1
  std::vector<float> input_data = {1.0f, 0.0f, 0.0f, 1.0f};

  auto src = gr::blocks::vector_source_f::make(input_data, false, 4);
  auto det_block = matrix_determinant<float>::make({2, 2});
  auto sink = gr::blocks::vector_sink_f::make(1);

  tb->connect(src, 0, det_block, 0);
  tb->connect(det_block, 0, sink, 0);

  tb->run();

  auto output_data = sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 1);
  BOOST_CHECK_CLOSE(output_data[0], 1.0f, 1e-6);
}

BOOST_AUTO_TEST_CASE(test_matrix_determinant_2x2_general) {
  auto tb = gr::make_top_block("test");

  // 2x2 matrix [3 2; 1 4] - determinant should be 3*4 - 2*1 = 10
  std::vector<float> input_data = {3.0f, 2.0f, 1.0f, 4.0f};

  auto src = gr::blocks::vector_source_f::make(input_data, false, 4);
  auto det_block = matrix_determinant<float>::make({2, 2});
  auto sink = gr::blocks::vector_sink_f::make(1);

  tb->connect(src, 0, det_block, 0);
  tb->connect(det_block, 0, sink, 0);

  tb->run();

  auto output_data = sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 1);
  BOOST_CHECK_CLOSE(output_data[0], 10.0f, 1e-6);
}

BOOST_AUTO_TEST_CASE(test_matrix_determinant_singular_matrix) {
  auto tb = gr::make_top_block("test");

  // Singular 2x2 matrix [1 2; 2 4] - determinant should be 0
  std::vector<float> input_data = {1.0f, 2.0f, 2.0f, 4.0f};

  auto src = gr::blocks::vector_source_f::make(input_data, false, 4);
  auto det_block = matrix_determinant<float>::make({2, 2});
  auto sink = gr::blocks::vector_sink_f::make(1);

  tb->connect(src, 0, det_block, 0);
  tb->connect(det_block, 0, sink, 0);

  tb->run();

  auto output_data = sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 1);
  BOOST_CHECK_SMALL(output_data[0], 1e-6f);
}

} /* namespace linalg */
} /* namespace gr */
