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
#include <gnuradio/linalg/matrix_inverse.h>
#include <gnuradio/top_block.h>
#include <vector>

namespace gr {
namespace linalg {

BOOST_AUTO_TEST_CASE(test_matrix_inverse_2x2_identity) {
  auto tb = gr::make_top_block("test");

  // 2x2 identity matrix [1 0; 0 1] - inverse should be itself
  std::vector<float> input_data = {1.0f, 0.0f, 0.0f, 1.0f};
  std::vector<float> expected_inv = {1.0f, 0.0f, 0.0f, 1.0f};

  auto src = gr::blocks::vector_source_f::make(input_data, false, 4);
  auto inv_block = matrix_inverse<float>::make({2, 2});
  auto sink = gr::blocks::vector_sink_f::make(4);

  tb->connect(src, 0, inv_block, 0);
  tb->connect(inv_block, 0, sink, 0);

  tb->run();

  auto output_data = sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 4u);
  for (size_t i = 0; i < 4; ++i) {
    BOOST_CHECK_CLOSE(output_data[i], expected_inv[i], 1e-6);
  }
}

BOOST_AUTO_TEST_CASE(test_matrix_inverse_2x2_general) {
  auto tb = gr::make_top_block("test");

  // 2x2 matrix [2 1; 1 1] - inverse should be [1 -1; -1 2]
  std::vector<float> input_data = {2.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float> expected_inv = {1.0f, -1.0f, -1.0f, 2.0f};

  auto src = gr::blocks::vector_source_f::make(input_data, false, 4);
  auto inv_block = matrix_inverse<float>::make({2, 2});
  auto sink = gr::blocks::vector_sink_f::make(4);

  tb->connect(src, 0, inv_block, 0);
  tb->connect(inv_block, 0, sink, 0);

  tb->run();

  auto output_data = sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 4u);
  for (size_t i = 0; i < 4; ++i) {
    BOOST_CHECK_CLOSE(output_data[i], expected_inv[i], 1e-5);
  }
}

BOOST_AUTO_TEST_CASE(test_matrix_inverse_3x3_diagonal) {
  auto tb = gr::make_top_block("test");

  // 3x3 diagonal matrix [2 0 0; 0 3 0; 0 0 4] - inverse should be [0.5 0 0; 0
  // 1/3 0; 0 0 0.25]
  std::vector<float> input_data = {2.0f, 0.0f, 0.0f, 0.0f, 3.0f,
                                   0.0f, 0.0f, 0.0f, 4.0f};
  std::vector<float> expected_inv = {0.5f, 0.0f, 0.0f, 0.0f, 1.0f / 3.0f,
                                     0.0f, 0.0f, 0.0f, 0.25f};

  auto src = gr::blocks::vector_source_f::make(input_data, false, 9);
  auto inv_block = matrix_inverse<float>::make({3, 3});
  auto sink = gr::blocks::vector_sink_f::make(9);

  tb->connect(src, 0, inv_block, 0);
  tb->connect(inv_block, 0, sink, 0);

  tb->run();

  auto output_data = sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 9u);
  for (size_t i = 0; i < 9; ++i) {
    BOOST_CHECK_CLOSE(output_data[i], expected_inv[i], 1e-5);
  }
}

BOOST_AUTO_TEST_CASE(test_matrix_inverse_orthogonal_matrix) {
  auto tb = gr::make_top_block("test");

  // Simple rotation matrix [cos(π/4) -sin(π/4); sin(π/4) cos(π/4)]
  // For 45 degrees: cos(π/4) = sin(π/4) = √2/2 ≈ 0.7071
  float cos45 = 0.7071067812f;
  std::vector<float> input_data = {cos45, -cos45, cos45, cos45};
  std::vector<float> expected_inv = {cos45, cos45, -cos45,
                                     cos45}; // Transpose for orthogonal matrix

  auto src = gr::blocks::vector_source_f::make(input_data, false, 4);
  auto inv_block = matrix_inverse<float>::make({2, 2});
  auto sink = gr::blocks::vector_sink_f::make(4);

  tb->connect(src, 0, inv_block, 0);
  tb->connect(inv_block, 0, sink, 0);

  tb->run();

  auto output_data = sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 4u);
  for (size_t i = 0; i < 4; ++i) {
    BOOST_CHECK_CLOSE(output_data[i], expected_inv[i], 1e-5);
  }
}

} /* namespace linalg */
} /* namespace gr */
