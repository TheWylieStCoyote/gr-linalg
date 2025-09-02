/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <boost/test/tools/old/interface.hpp>
#include <boost/test/unit_test.hpp>
#include <gnuradio/linalg/utils.h>

// Backward compatibility alias expected by legacy test name
// using matrix_add_f = gr::linalg::matrix_add_sync_f;

namespace gr {
namespace linalg {

BOOST_AUTO_TEST_CASE(test_utils_shaps_product) {
  // Test product function with a simple vector
  std::vector<int> vec = {1, 2, 3, 4};
  int result = utils::product(vec);
  BOOST_CHECK_EQUAL(result, 24); // 1 * 2 * 3 * 4 = 24
}

BOOST_AUTO_TEST_CASE(test_utils_shaps_sum) {
  // Test sum function with a simple vector
  std::vector<int> vec = {1, 2, 3, 4};
  int result = utils::sum(vec);
  BOOST_CHECK_EQUAL(result, 10); // 1 + 2 + 3 + 4 = 10
}

BOOST_AUTO_TEST_CASE(test_utils_shapes_validate_shape) {
  // Test shape manipulation functions
  types::shape shape_1 = {3, 4, 5};
  BOOST_CHECK(utils::validate_shape(shape_1, "test_shape"));

  types::shape shape_2 = {3, 4, 5};
  BOOST_CHECK(utils::validate_shape(shape_2, "test_shape"));

  types::shape shape_3 = {};
  BOOST_CHECK_THROW(utils::validate_shape(shape_3, "test_shape"),
                    std::invalid_argument);

  types::shape shape_4 = {3, 0, 5};
  BOOST_CHECK_THROW(utils::validate_shape(shape_4, "test_shape"),
                    std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_utils_shape_validate_matrix_shape) {
  // Test matrix shape validation, no exceptions expected
  types::shape shape_1 = {3, 4, 5};
  BOOST_CHECK_NO_THROW(utils::matrix_shape(shape_1));
  // Test matrix shape validation with 2D shape, no exceptions expected
  types::shape shape_2 = {3, 4};
  BOOST_CHECK_NO_THROW(utils::matrix_shape(shape_2));
  // Test matrix shape validation with 1D shape, should throw
  types::shape shape_3 = {3};
  BOOST_CHECK_THROW(utils::matrix_shape(shape_3), std::invalid_argument);
  // Test matrix shape validation with empty shape, should throw
  // This is a change from the original code, which allowed empty shapes
  types::shape shape_4 = {};
  BOOST_CHECK_THROW(utils::matrix_shape(shape_4), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_utils_shape_matrix_rows) {
  // Test matrix rows extraction
  types::shape shape_1 = {3, 4, 5};
  BOOST_CHECK_EQUAL(utils::matrix_rows(shape_1), 4); // Last dimension is rows

  types::shape shape_2 = {3, 4};
  BOOST_CHECK_EQUAL(utils::matrix_rows(shape_2), 3); // Last dimension is rows

  types::shape shape_3 = {3};
  BOOST_CHECK_THROW(utils::matrix_rows(shape_3),
                    std::invalid_argument); // Should throw for 1D

  types::shape shape_4 = {};
  BOOST_CHECK_THROW(utils::matrix_rows(shape_4),
                    std::invalid_argument); // Should throw for empty
}

BOOST_AUTO_TEST_CASE(test_utils_shape_matrix_cols) {
  // Test matrix columns extraction
  types::shape shape_1 = {3, 4, 5};
  BOOST_CHECK_EQUAL(utils::matrix_cols(shape_1), 5); // Last dimension is cols

  types::shape shape_2 = {3, 4};
  BOOST_CHECK_EQUAL(utils::matrix_cols(shape_2), 4); // Last dimension is cols

  types::shape shape_3 = {3};
  BOOST_CHECK_THROW(utils::matrix_cols(shape_3),
                    std::invalid_argument); // Should throw for 1D

  types::shape shape_4 = {};
  BOOST_CHECK_THROW(utils::matrix_cols(shape_4),
                    std::invalid_argument); // Should throw for empty
}

BOOST_AUTO_TEST_CASE(test_utils_shape_compute_size) {
  // Test compute size function
  types::shape shape_1 = {3, 4, 5};
  BOOST_CHECK_EQUAL(utils::compute_size(shape_1), 60); // 3 * 4 * 5 = 60

  types::shape shape_2 = {3, 4};
  BOOST_CHECK_EQUAL(utils::compute_size(shape_2), 12); // 3 * 4 = 12

  types::shape shape_3 = {3};
  BOOST_CHECK_EQUAL(utils::compute_size(shape_3), 3); // Single dimension

  types::shape shape_4 = {};
  BOOST_CHECK_EQUAL(utils::compute_size(shape_4), 0); // Empty shape
}

BOOST_AUTO_TEST_CASE(test_utils_shape_compute_sizes) {
  // Test compute sizes function
  types::vector_shapes shapes = {{3, 4, 5}, {2, 3}};
  auto sizes = utils::compute_sizes(shapes);
  BOOST_CHECK_EQUAL(sizes.size(), 2);
  BOOST_CHECK_EQUAL(sizes[0], 60); // 3 * 4 * 5 = 60
  BOOST_CHECK_EQUAL(sizes[1], 6);  // 2 * 3 = 6

  types::vector_shapes empty_shapes;
  sizes = utils::compute_sizes(empty_shapes);
  BOOST_CHECK_EQUAL(sizes.size(), 0); // Empty input should return empty output
  types::vector_shapes single_shape = {{3}};
  sizes = utils::compute_sizes(single_shape);
  BOOST_CHECK_EQUAL(sizes.size(), 1);
  BOOST_CHECK_EQUAL(sizes[0],
                    3); // Single dimension shape should return its size

  types::vector_shapes single_shape_2 = {{4}};
  sizes = utils::compute_sizes(single_shape_2);
  BOOST_CHECK_EQUAL(sizes.size(), 1);
  BOOST_CHECK_EQUAL(sizes[0],
                    4); // Single dimension shape should return its size
}

BOOST_AUTO_TEST_CASE(test_utils_shape_compute_broadcast_shape) {
  // Test compute broadcast shape function - use compatible broadcasting shapes
  types::shape shape_1 = {3, 4, 5};
  types::shape shape_2 = {3, 4, 5}; // Same shape is always compatible
  auto broadcast_shape = utils::compute_broadcast_shape(shape_1, shape_2);
  BOOST_CHECK_EQUAL(broadcast_shape.size(), 3);
  BOOST_CHECK_EQUAL(broadcast_shape[0], 3);
  BOOST_CHECK_EQUAL(broadcast_shape[1], 4);
  BOOST_CHECK_EQUAL(broadcast_shape[2], 5);

  types::shape shape_3 = {3, 4};
  types::shape shape_4 = {3, 4}; // Same shape is always compatible
  broadcast_shape = utils::compute_broadcast_shape(shape_3, shape_4);
  BOOST_CHECK_EQUAL(broadcast_shape.size(), 2);
  BOOST_CHECK_EQUAL(broadcast_shape[0], 3);
  BOOST_CHECK_EQUAL(broadcast_shape[1], 4);

  types::shape shape_5 = {3};
  types::shape shape_6 = {3}; // Same shape is compatible for broadcast
  broadcast_shape = utils::compute_broadcast_shape(shape_5, shape_6);
  BOOST_CHECK_EQUAL(broadcast_shape.size(), 1);
  BOOST_CHECK_EQUAL(broadcast_shape[0], 3);
}

BOOST_AUTO_TEST_CASE(test_utils_shape_compute_broadcast_shape_empty) {
  // Test compute broadcast shape with empty shapes
  types::shape shape_1 = {3, 4, 5};
  types::shape shape_2 = {};
  auto broadcast_shape = utils::compute_broadcast_shape(shape_1, shape_2);
  BOOST_CHECK_EQUAL(broadcast_shape.size(), 3);
  BOOST_CHECK_EQUAL(broadcast_shape[0], 3); // Max of 3 and 1
  BOOST_CHECK_EQUAL(broadcast_shape[1], 4); // Max of 4 and 1
  BOOST_CHECK_EQUAL(broadcast_shape[2], 5); // Max of 5 and 1

  types::shape shape_3 = {};
  types::shape shape_4 = {};
  broadcast_shape = utils::compute_broadcast_shape(shape_3, shape_4);
  BOOST_CHECK_EQUAL(broadcast_shape.size(),
                    0); // Both empty should return empty
}

} /* namespace linalg */
} /* namespace gr */
