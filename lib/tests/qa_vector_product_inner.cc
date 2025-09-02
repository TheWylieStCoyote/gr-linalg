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
#include <gnuradio/linalg/vector_product_inner.h>

namespace gr {
namespace linalg {

BOOST_AUTO_TEST_CASE(test_vector_product_inner_basic_functionality) {
  // Test vector inner product: [1, 2, 3] · [4, 5, 6] = 32
  const types::shape shape_a{3, 1}; // 3x1 vector
  const types::shape shape_b{3, 1}; // 3x1 vector

  auto block = vector_product_inner_sync<float>::make(shape_a, shape_b);
  BOOST_REQUIRE(block);

  // Test shape computation
  const auto output_shapes =
      vector_product_inner<float>::compute_output_shapes({shape_a, shape_b});
  BOOST_REQUIRE_EQUAL(output_shapes.size(), 1);
  BOOST_REQUIRE_EQUAL(output_shapes[0].size(), 2);
  BOOST_REQUIRE_EQUAL(output_shapes[0][0], 1); // 1 row (scalar)
  BOOST_REQUIRE_EQUAL(output_shapes[0][1], 1); // 1 column (scalar)
}

BOOST_AUTO_TEST_CASE(test_vector_product_inner_operation) {
  // Test the actual mathematical operation
  const types::shape shape_a{3, 1}; // 3x1 vector
  const types::shape shape_b{3, 1}; // 3x1 vector

  // Create test vectors
  Eigen::Matrix<float, 3, 1> vector_a;
  vector_a << 1.0f, 2.0f, 3.0f;

  Eigen::Matrix<float, 3, 1> vector_b;
  vector_b << 4.0f, 5.0f, 6.0f;

  // Create output scalar
  Eigen::Matrix<float, 1, 1> output_scalar;

  // Create maps using proper types
  auto input_a_map =
      types::const_matrix_map_dynamic<float>(vector_a.data(), 3, 1);
  auto input_b_map =
      types::const_matrix_map_dynamic<float>(vector_b.data(), 3, 1);
  auto output_map =
      types::matrix_map_dynamic<float>(output_scalar.data(), 1, 1);

  types::vector_const_matrix_map<float> input_vectors = {&input_a_map,
                                                         &input_b_map};
  types::vector_matrix_map<float> output_vectors = {&output_map};

  // Create operation instance and test
  vector_product_inner<float> op(shape_a, shape_b);
  auto result = op.operation(input_vectors, output_vectors);

  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Verify result: inner product should be 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
  BOOST_CHECK_CLOSE(output_scalar(0, 0), 32.0f, 1e-6);
}

BOOST_AUTO_TEST_CASE(test_vector_product_inner_complex) {
  // Test with complex vectors using conjugate
  const types::shape shape_a{2, 1}; // 2x1 vector
  const types::shape shape_b{2, 1}; // 2x1 vector

  // Create test complex vectors
  Eigen::Matrix<std::complex<float>, 2, 1> vector_a;
  vector_a << std::complex<float>(1.0f, 2.0f), std::complex<float>(3.0f, 0.0f);

  Eigen::Matrix<std::complex<float>, 2, 1> vector_b;
  vector_b << std::complex<float>(2.0f, 1.0f), std::complex<float>(1.0f, 1.0f);

  // Create output scalar
  Eigen::Matrix<std::complex<float>, 1, 1> output_scalar;

  // Create maps using proper types
  auto input_a_map = types::const_matrix_map_dynamic<std::complex<float>>(
      vector_a.data(), 2, 1);
  auto input_b_map = types::const_matrix_map_dynamic<std::complex<float>>(
      vector_b.data(), 2, 1);
  auto output_map = types::matrix_map_dynamic<std::complex<float>>(
      output_scalar.data(), 1, 1);

  types::vector_const_matrix_map<std::complex<float>> input_vectors = {
      &input_a_map, &input_b_map};
  types::vector_matrix_map<std::complex<float>> output_vectors = {&output_map};

  // Create operation instance and test
  vector_product_inner<std::complex<float>> op(shape_a, shape_b);
  auto result = op.operation(input_vectors, output_vectors);

  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Verify complex inner product with conjugate:
  // conj(1+2i) * (2+i) + conj(3+0i) * (1+i)
  // = (1-2i) * (2+i) + (3-0i) * (1+i)
  // = (2 + i - 4i + 2) + (3 + 3i)
  // = (4 - 3i) + (3 + 3i) = 7 + 0i
  BOOST_CHECK_CLOSE(output_scalar(0, 0).real(), 7.0f, 1e-6);
  BOOST_CHECK_CLOSE(output_scalar(0, 0).imag(), 0.0f, 1e-6);
}

} /* namespace linalg */
} /* namespace gr */
