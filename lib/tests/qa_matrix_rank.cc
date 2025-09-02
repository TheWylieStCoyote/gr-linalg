/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <boost/test/unit_test.hpp>
#include <gnuradio/attributes.h>
#include <gnuradio/blocks/vector_sink.h>
#include <gnuradio/blocks/vector_source.h>
#include <gnuradio/linalg/matrix_rank.h>
#include <gnuradio/top_block.h>

namespace gr {
namespace linalg {

BOOST_AUTO_TEST_CASE(test_matrix_rank_2x2_identity) {
  auto tb = gr::make_top_block("test");

  // 2x2 identity matrix [1 0; 0 1] - rank should be 2
  float input_data[] = {1.0f, 0.0f, 0.0f, 1.0f};
  float expected_rank = 2.0f;

  auto src = gr::blocks::vector_source_f::make(input_data, 4, false, 4);
  auto rank_block = matrix_rank<float>::make({2, 2});
  auto sink = gr::blocks::vector_sink_f::make(1);

  tb->connect(src, 0, rank_block, 0);
  tb->connect(rank_block, 0, sink, 0);

  tb->run();

  auto output_data = sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 1);
  BOOST_CHECK_CLOSE(output_data[0], expected_rank, 1e-6);
}

BOOST_AUTO_TEST_CASE(test_matrix_rank_3x3_full_rank) {
  auto tb = gr::make_top_block("test");

  // 3x3 full rank matrix [1 2 3; 0 1 4; 5 6 0] - rank should be 3
  double input_data[] = {1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0};
  double expected_rank = 3.0;

  auto src = gr::blocks::vector_source_d::make(input_data, 9, false, 9);
  auto rank_block = matrix_rank<double>::make({3, 3});
  auto sink = gr::blocks::vector_sink_d::make(1);

  tb->connect(src, 0, rank_block, 0);
  tb->connect(rank_block, 0, sink, 0);

  tb->run();

  auto output_data = sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 1);
  BOOST_CHECK_CLOSE(output_data[0], expected_rank, 1e-12);
}

BOOST_AUTO_TEST_CASE(test_matrix_rank_singular_matrix) {
  auto tb = gr::make_top_block("test");

  // Singular 2x2 matrix [1 2; 2 4] - rank should be 1
  float input_data[] = {1.0f, 2.0f, 2.0f, 4.0f};
  float expected_rank = 1.0f;

  auto src = gr::blocks::vector_source_f::make(input_data, 4, false, 4);
  auto rank_block = matrix_rank<float>::make({2, 2});
  auto sink = gr::blocks::vector_sink_f::make(1);

  tb->connect(src, 0, rank_block, 0);
  tb->connect(rank_block, 0, sink, 0);

  tb->run();

  auto output_data = sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 1);
  BOOST_CHECK_CLOSE(output_data[0], expected_rank, 1e-6);
}

BOOST_AUTO_TEST_CASE(test_matrix_rank_zero_matrix) {
  auto tb = gr::make_top_block("test");

  // 3x3 zero matrix - rank should be 0
  float input_data[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float expected_rank = 0.0f;

  auto src = gr::blocks::vector_source_f::make(input_data, 9, false, 9);
  auto rank_block = matrix_rank<float>::make({3, 3});
  auto sink = gr::blocks::vector_sink_f::make(1);

  tb->connect(src, 0, rank_block, 0);
  tb->connect(rank_block, 0, sink, 0);

  tb->run();

  auto output_data = sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 1);
  BOOST_CHECK_CLOSE(output_data[0], expected_rank, 1e-6);
}

BOOST_AUTO_TEST_CASE(test_matrix_rank_rectangular_matrix) {
  auto tb = gr::make_top_block("test");

  // 3x2 matrix [1 2; 3 4; 5 6] - rank should be 2
  double input_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  double expected_rank = 2.0;

  auto src = gr::blocks::vector_source_d::make(input_data, 6, false, 6);
  auto rank_block = matrix_rank<double>::make({3, 2});
  auto sink = gr::blocks::vector_sink_d::make(1);

  tb->connect(src, 0, rank_block, 0);
  tb->connect(rank_block, 0, sink, 0);

  tb->run();

  auto output_data = sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 1);
  BOOST_CHECK_CLOSE(output_data[0], expected_rank, 1e-12);
}

BOOST_AUTO_TEST_CASE(test_matrix_rank_rank_deficient_3x3) {
  auto tb = gr::make_top_block("test");

  // 3x3 rank-2 matrix [1 2 3; 4 5 6; 7 8 9] - third row is sum of first two
  // scaled Actually [1 2 3; 4 5 6; 5 7 9] where third row = first + second
  float input_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 5.0f, 7.0f, 9.0f};
  float expected_rank = 2.0f;

  auto src = gr::blocks::vector_source_f::make(input_data, 9, false, 9);
  auto rank_block = matrix_rank<float>::make({3, 3});
  auto sink = gr::blocks::vector_sink_f::make(1);

  tb->connect(src, 0, rank_block, 0);
  tb->connect(rank_block, 0, sink, 0);

  tb->run();

  auto output_data = sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 1);
  BOOST_CHECK_CLOSE(output_data[0], expected_rank, 1e-6);
}

} /* namespace linalg */
} /* namespace gr */
