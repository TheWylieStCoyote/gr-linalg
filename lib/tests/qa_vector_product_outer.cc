/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <boost/test/unit_test.hpp>
#include <complex>
#include <gnuradio/attributes.h>
#include <gnuradio/linalg/types.h>
#include <gnuradio/linalg/vector_product_outer.h>

namespace gr {
namespace linalg {

BOOST_AUTO_TEST_CASE(test_vector_product_outer_basic_functionality) {
  // Test vector outer product: [1, 2] ⊗ [3, 4] = [[3, 4], [6, 8]]
  const types::shape shape_a{2, 1}; // 2x1 vector
  const types::shape shape_b{2, 1}; // 2x1 vector

  auto block = vector_product_outer_sync<float>::make(shape_a, shape_b);
  BOOST_REQUIRE(block);

  // Test shape computation
  const auto output_shapes =
      vector_product_outer<float>::compute_output_shapes({shape_a, shape_b});
  BOOST_REQUIRE_EQUAL(output_shapes.size(), 1);
  BOOST_REQUIRE_EQUAL(output_shapes[0].size(), 2);
  BOOST_REQUIRE_EQUAL(output_shapes[0][0], 2); // 2 rows
  BOOST_REQUIRE_EQUAL(output_shapes[0][1], 2); // 2 columns
}

BOOST_AUTO_TEST_CASE(test_vector_product_outer_operation) {
  // Test the actual mathematical operation
  const types::shape shape_a{3, 1}; // 3x1 vector
  const types::shape shape_b{2, 1}; // 2x1 vector

  // Create test vectors
  Eigen::Matrix<float, 3, 1> vector_a;
  vector_a << 1.0f, 2.0f, 3.0f;

  Eigen::Matrix<float, 2, 1> vector_b;
  vector_b << 4.0f, 5.0f;

  // Create output matrix
  Eigen::Matrix<float, 3, 2> output_matrix;

  // Create maps using proper types
  auto input_a_map =
      types::const_matrix_map_dynamic<float>(vector_a.data(), 3, 1);
  auto input_b_map =
      types::const_matrix_map_dynamic<float>(vector_b.data(), 2, 1);
  auto output_map =
      types::matrix_map_dynamic<float>(output_matrix.data(), 3, 2);

  types::vector_const_matrix_map<float> input_vectors = {&input_a_map,
                                                         &input_b_map};
  types::vector_matrix_map<float> output_vectors = {&output_map};

  // Create operation instance and test
  vector_product_outer<float> op(shape_a, shape_b);
  auto result = op.operation(input_vectors, output_vectors);

  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Verify results: outer product should be:
  // [1*4, 1*5]   [4, 5]
  // [2*4, 2*5] = [8, 10]
  // [3*4, 3*5]   [12, 15]
  BOOST_CHECK_CLOSE(output_matrix(0, 0), 4.0f, 1e-6);
  BOOST_CHECK_CLOSE(output_matrix(0, 1), 5.0f, 1e-6);
  BOOST_CHECK_CLOSE(output_matrix(1, 0), 8.0f, 1e-6);
  BOOST_CHECK_CLOSE(output_matrix(1, 1), 10.0f, 1e-6);
  BOOST_CHECK_CLOSE(output_matrix(2, 0), 12.0f, 1e-6);
  BOOST_CHECK_CLOSE(output_matrix(2, 1), 15.0f, 1e-6);
}

BOOST_AUTO_TEST_CASE(test_vector_product_outer_complex) {
  // Test with complex vectors
  const types::shape shape_a{2, 1}; // 2x1 vector
  const types::shape shape_b{2, 1}; // 2x1 vector

  // Create test complex vectors
  Eigen::Matrix<std::complex<float>, 2, 1> vector_a;
  vector_a << std::complex<float>(1.0f, 1.0f), std::complex<float>(2.0f, 0.0f);

  Eigen::Matrix<std::complex<float>, 2, 1> vector_b;
  vector_b << std::complex<float>(1.0f, -1.0f), std::complex<float>(0.0f, 1.0f);

  // Create output matrix
  Eigen::Matrix<std::complex<float>, 2, 2> output_matrix;

  // Create maps using proper types
  auto input_a_map = types::const_matrix_map_dynamic<std::complex<float>>(
      vector_a.data(), 2, 1);
  auto input_b_map = types::const_matrix_map_dynamic<std::complex<float>>(
      vector_b.data(), 2, 1);
  auto output_map = types::matrix_map_dynamic<std::complex<float>>(
      output_matrix.data(), 2, 2);

  types::vector_const_matrix_map<std::complex<float>> input_vectors = {
      &input_a_map, &input_b_map};
  types::vector_matrix_map<std::complex<float>> output_vectors = {&output_map};

  // Create operation instance and test
  vector_product_outer<std::complex<float>> op(shape_a, shape_b);
  auto result = op.operation(input_vectors, output_vectors);

  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Verify complex multiplication results
  // (1+i) * (1-i) = 1 + i - i + 1 = 2
  // (1+i) * (i) = i - 1 = -1+i
  BOOST_CHECK_CLOSE(output_matrix(0, 0).real(), 2.0f, 1e-6);
  BOOST_CHECK_CLOSE(output_matrix(0, 0).imag(), 0.0f, 1e-6);
  BOOST_CHECK_CLOSE(output_matrix(0, 1).real(), -1.0f, 1e-6);
  BOOST_CHECK_CLOSE(output_matrix(0, 1).imag(), 1.0f, 1e-6);
}

} /* namespace linalg */
} /* namespace gr */
