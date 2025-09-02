/* -*- c++ -*- */
/*
 * Copyright 2024 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include <Eigen/Dense>
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <complex>
#include <gnuradio/linalg/types.h>
#include <gnuradio/linalg/vector_outer_product.h>

using namespace gr::linalg;

BOOST_AUTO_TEST_SUITE(test_vector_outer_product)

// Constants for numerical comparisons
constexpr float FLOAT_TOLERANCE = 1e-6f;
constexpr double DOUBLE_TOLERANCE = 1e-12;

// Test basic outer product functionality
BOOST_AUTO_TEST_CASE(test_outer_product_basic) {
  auto outer_op = std::make_shared<vector_outer_product<float>>();

  // Create test vectors: a = [1, 2, 3]^T, b = [4, 5]^T
  std::vector<float> vector_a{1.0f, 2.0f, 3.0f}; // 3×1 vector
  std::vector<float> vector_b{4.0f, 5.0f};       // 2×1 vector
  std::vector<float> output(6);                  // 3×2 result matrix

  // Create matrix maps - vectors as column vectors
  types::matrix_map_dynamic<float> vec_a_map(vector_a.data(), 3, 1);
  types::matrix_map_dynamic<float> vec_b_map(vector_b.data(), 2, 1);
  types::matrix_map_dynamic<float> result_map(output.data(), 3, 2);

  types::vector_const_matrix_map<float> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&vec_a_map),
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&vec_b_map)};
  types::vector_matrix_map<float> outputs{&result_map};

  BOOST_CHECK_EQUAL(outer_op->operation(inputs, outputs),
                    OperationReturn::SUCCESS);

  // Expected result: [1*4  1*5]   = [4   5 ]
  //                  [2*4  2*5]     [8   10]
  //                  [3*4  3*5]     [12  15]

  // Check results (matrix stored in column-major order)
  BOOST_CHECK_CLOSE(output[0], 4.0f, FLOAT_TOLERANCE);  // (0,0) = 1*4
  BOOST_CHECK_CLOSE(output[1], 8.0f, FLOAT_TOLERANCE);  // (1,0) = 2*4
  BOOST_CHECK_CLOSE(output[2], 12.0f, FLOAT_TOLERANCE); // (2,0) = 3*4
  BOOST_CHECK_CLOSE(output[3], 5.0f, FLOAT_TOLERANCE);  // (0,1) = 1*5
  BOOST_CHECK_CLOSE(output[4], 10.0f, FLOAT_TOLERANCE); // (1,1) = 2*5
  BOOST_CHECK_CLOSE(output[5], 15.0f, FLOAT_TOLERANCE); // (2,1) = 3*5
}

// Test row vector inputs (1×n matrices treated as vectors)
BOOST_AUTO_TEST_CASE(test_outer_product_row_vectors) {
  auto outer_op = std::make_shared<vector_outer_product<double>>();

  // Create test vectors as row vectors: a = [2, 3] (1×2), b = [1, 4, 5] (1×3)
  std::vector<double> vector_a{2.0, 3.0};      // 1×2 row vector
  std::vector<double> vector_b{1.0, 4.0, 5.0}; // 1×3 row vector
  std::vector<double> output(6);               // 2×3 result matrix

  types::matrix_map_dynamic<double> vec_a_map(vector_a.data(), 1, 2);
  types::matrix_map_dynamic<double> vec_b_map(vector_b.data(), 1, 3);
  types::matrix_map_dynamic<double> result_map(output.data(), 2, 3);

  types::vector_const_matrix_map<double> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<double> *>(&vec_a_map),
      reinterpret_cast<types::const_matrix_map_dynamic<double> *>(&vec_b_map)};
  types::vector_matrix_map<double> outputs{&result_map};

  BOOST_CHECK_EQUAL(outer_op->operation(inputs, outputs),
                    OperationReturn::SUCCESS);

  // Expected result: [2*1  2*4  2*5]   = [2   8   10]
  //                  [3*1  3*4  3*5]     [3   12  15]

  BOOST_CHECK_CLOSE(output[0], 2.0, DOUBLE_TOLERANCE);  // (0,0) = 2*1
  BOOST_CHECK_CLOSE(output[1], 3.0, DOUBLE_TOLERANCE);  // (1,0) = 3*1
  BOOST_CHECK_CLOSE(output[2], 8.0, DOUBLE_TOLERANCE);  // (0,1) = 2*4
  BOOST_CHECK_CLOSE(output[3], 12.0, DOUBLE_TOLERANCE); // (1,1) = 3*4
  BOOST_CHECK_CLOSE(output[4], 10.0, DOUBLE_TOLERANCE); // (0,2) = 2*5
  BOOST_CHECK_CLOSE(output[5], 15.0, DOUBLE_TOLERANCE); // (1,2) = 3*5
}

// Test complex number outer products
BOOST_AUTO_TEST_CASE(test_outer_product_complex) {
  auto outer_op = std::make_shared<vector_outer_product<std::complex<float>>>();

  // Create complex test vectors
  std::vector<std::complex<float>> vector_a{
      std::complex<float>(1.0f, 1.0f), // 1+i
      std::complex<float>(2.0f, 0.0f)  // 2+0i
  };
  std::vector<std::complex<float>> vector_b{
      std::complex<float>(1.0f, -1.0f), // 1-i
      std::complex<float>(0.0f, 2.0f)   // 0+2i
  };
  std::vector<std::complex<float>> output(4); // 2×2 result

  types::matrix_map_dynamic<std::complex<float>> vec_a_map(vector_a.data(), 2,
                                                           1);
  types::matrix_map_dynamic<std::complex<float>> vec_b_map(vector_b.data(), 2,
                                                           1);
  types::matrix_map_dynamic<std::complex<float>> result_map(output.data(), 2,
                                                            2);

  types::vector_const_matrix_map<std::complex<float>> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<std::complex<float>> *>(
          &vec_a_map),
      reinterpret_cast<types::const_matrix_map_dynamic<std::complex<float>> *>(
          &vec_b_map)};
  types::vector_matrix_map<std::complex<float>> outputs{&result_map};

  BOOST_CHECK_EQUAL(outer_op->operation(inputs, outputs),
                    OperationReturn::SUCCESS);

  // Expected results:
  // (1+i)*(1-i) = 1 - i + i - i^2 = 1 - (-1) = 2
  // (1+i)*(0+2i) = 0 + 2i + 0 + 2i^2 = 2i - 2 = -2+2i
  // (2+0i)*(1-i) = 2 - 2i
  // (2+0i)*(0+2i) = 0 + 4i

  BOOST_CHECK_CLOSE(output[0].real(), 2.0f, FLOAT_TOLERANCE);
  BOOST_CHECK_CLOSE(output[0].imag(), 0.0f, FLOAT_TOLERANCE);

  BOOST_CHECK_CLOSE(output[1].real(), 2.0f, FLOAT_TOLERANCE);
  BOOST_CHECK_CLOSE(output[1].imag(), -2.0f, FLOAT_TOLERANCE);

  BOOST_CHECK_CLOSE(output[2].real(), -2.0f, FLOAT_TOLERANCE);
  BOOST_CHECK_CLOSE(output[2].imag(), 2.0f, FLOAT_TOLERANCE);

  BOOST_CHECK_CLOSE(output[3].real(), 0.0f, FLOAT_TOLERANCE);
  BOOST_CHECK_CLOSE(output[3].imag(), 4.0f, FLOAT_TOLERANCE);
}

// Test single-element vectors (scalars)
BOOST_AUTO_TEST_CASE(test_outer_product_scalars) {
  auto outer_op = std::make_shared<vector_outer_product<float>>();

  // Scalar outer product: 3 ⊗ 4 = [[12]]
  std::vector<float> vector_a{3.0f};
  std::vector<float> vector_b{4.0f};
  std::vector<float> output(1); // 1×1 result

  types::matrix_map_dynamic<float> vec_a_map(vector_a.data(), 1, 1);
  types::matrix_map_dynamic<float> vec_b_map(vector_b.data(), 1, 1);
  types::matrix_map_dynamic<float> result_map(output.data(), 1, 1);

  types::vector_const_matrix_map<float> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&vec_a_map),
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&vec_b_map)};
  types::vector_matrix_map<float> outputs{&result_map};

  BOOST_CHECK_EQUAL(outer_op->operation(inputs, outputs),
                    OperationReturn::SUCCESS);
  BOOST_CHECK_CLOSE(output[0], 12.0f, FLOAT_TOLERANCE);
}

// Test different length vectors
BOOST_AUTO_TEST_CASE(test_outer_product_different_lengths) {
  auto outer_op = std::make_shared<vector_outer_product<double>>();

  // Vectors of different lengths: a = [1, 2, 3, 4]^T (4×1), b = [5, 6, 7]^T
  // (3×1)
  std::vector<double> vector_a{1.0, 2.0, 3.0, 4.0}; // 4×1
  std::vector<double> vector_b{5.0, 6.0, 7.0};      // 3×1
  std::vector<double> output(12);                   // 4×3 result

  types::matrix_map_dynamic<double> vec_a_map(vector_a.data(), 4, 1);
  types::matrix_map_dynamic<double> vec_b_map(vector_b.data(), 3, 1);
  types::matrix_map_dynamic<double> result_map(output.data(), 4, 3);

  types::vector_const_matrix_map<double> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<double> *>(&vec_a_map),
      reinterpret_cast<types::const_matrix_map_dynamic<double> *>(&vec_b_map)};
  types::vector_matrix_map<double> outputs{&result_map};

  BOOST_CHECK_EQUAL(outer_op->operation(inputs, outputs),
                    OperationReturn::SUCCESS);

  // Verify a few key elements
  BOOST_CHECK_CLOSE(output[0], 5.0, DOUBLE_TOLERANCE);   // (0,0) = 1*5
  BOOST_CHECK_CLOSE(output[1], 10.0, DOUBLE_TOLERANCE);  // (1,0) = 2*5
  BOOST_CHECK_CLOSE(output[4], 6.0, DOUBLE_TOLERANCE);   // (0,1) = 1*6
  BOOST_CHECK_CLOSE(output[11], 28.0, DOUBLE_TOLERANCE); // (3,2) = 4*7
}

// Test error handling
BOOST_AUTO_TEST_CASE(test_outer_product_error_handling) {
  auto outer_op = std::make_shared<vector_outer_product<float>>();

  // Test insufficient inputs
  std::vector<float> vector_a{1.0f, 2.0f};
  std::vector<float> output(4);

  types::matrix_map_dynamic<float> vec_a_map(vector_a.data(), 2, 1);
  types::matrix_map_dynamic<float> result_map(output.data(), 2, 2);

  // Only provide one input instead of two
  types::vector_const_matrix_map<float> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&vec_a_map)};
  types::vector_matrix_map<float> outputs{&result_map};

  BOOST_CHECK_EQUAL(outer_op->operation(inputs, outputs),
                    OperationReturn::FAILURE);
}

// Test output shape validation
BOOST_AUTO_TEST_CASE(test_outer_product_shape_validation) {
  auto outer_op = std::make_shared<vector_outer_product<float>>();

  std::vector<float> vector_a{1.0f, 2.0f, 3.0f}; // 3×1
  std::vector<float> vector_b{4.0f, 5.0f};       // 2×1
  std::vector<float> output(4); // Wrong size: 2×2 instead of expected 3×2

  types::matrix_map_dynamic<float> vec_a_map(vector_a.data(), 3, 1);
  types::matrix_map_dynamic<float> vec_b_map(vector_b.data(), 2, 1);
  types::matrix_map_dynamic<float> result_map(output.data(), 2,
                                              2); // Wrong dimensions

  types::vector_const_matrix_map<float> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&vec_a_map),
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&vec_b_map)};
  types::vector_matrix_map<float> outputs{&result_map};

  // Should fail due to output dimension mismatch
  BOOST_CHECK_EQUAL(outer_op->operation(inputs, outputs),
                    OperationReturn::INVALID_SHAPE);
}

// Test vector shape validation helper functions
BOOST_AUTO_TEST_CASE(test_vector_shape_utilities) {
  // Test is_vector_shape
  BOOST_CHECK(
      vector_outer_product<float>::is_vector_shape({1, 5})); // 1×5 row vector
  BOOST_CHECK(vector_outer_product<float>::is_vector_shape(
      {5, 1})); // 5×1 column vector
  BOOST_CHECK(vector_outer_product<float>::is_vector_shape(
      {3, 3})); // 3×3 matrix (acceptable)
  BOOST_CHECK(
      !vector_outer_product<float>::is_vector_shape({1, 2, 3})); // 3D tensor

  // Test get_vector_length
  BOOST_CHECK_EQUAL(vector_outer_product<float>::get_vector_length({1, 5}), 5);
  BOOST_CHECK_EQUAL(vector_outer_product<float>::get_vector_length({7, 1}), 7);
  BOOST_CHECK_EQUAL(vector_outer_product<float>::get_vector_length({3, 2}),
                    6); // Flattened
}

// Test compute_output_shapes static method
BOOST_AUTO_TEST_CASE(test_compute_output_shapes) {
  types::vector_shapes input_shapes = {{3, 1}, {2, 1}}; // 3×1 and 2×1 vectors

  auto output_shapes =
      vector_outer_product<float>::compute_output_shapes(input_shapes);

  BOOST_REQUIRE_EQUAL(output_shapes.size(), 1);
  BOOST_CHECK_EQUAL(output_shapes[0][0], 3); // 3 rows
  BOOST_CHECK_EQUAL(output_shapes[0][1], 2); // 2 columns
}

// Test zero vectors
BOOST_AUTO_TEST_CASE(test_outer_product_zero_vectors) {
  auto outer_op = std::make_shared<vector_outer_product<float>>();

  // Zero vectors should produce zero matrix
  std::vector<float> vector_a{0.0f, 0.0f};
  std::vector<float> vector_b{0.0f, 0.0f, 0.0f};
  std::vector<float> output(6); // 2×3 result

  types::matrix_map_dynamic<float> vec_a_map(vector_a.data(), 2, 1);
  types::matrix_map_dynamic<float> vec_b_map(vector_b.data(), 3, 1);
  types::matrix_map_dynamic<float> result_map(output.data(), 2, 3);

  types::vector_const_matrix_map<float> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&vec_a_map),
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&vec_b_map)};
  types::vector_matrix_map<float> outputs{&result_map};

  BOOST_CHECK_EQUAL(outer_op->operation(inputs, outputs),
                    OperationReturn::SUCCESS);

  // All elements should be zero
  for (size_t i = 0; i < output.size(); ++i) {
    BOOST_CHECK_SMALL(output[i], FLOAT_TOLERANCE);
  }
}

BOOST_AUTO_TEST_SUITE_END()