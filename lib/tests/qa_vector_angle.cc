/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "vector_angle_impl.h"
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <complex>
#include <gnuradio/attributes.h>
#include <gnuradio/linalg/vector_angle.h>

namespace gr {
namespace linalg {

BOOST_AUTO_TEST_CASE(test_vector_angle_sync_basic_constructor) {
  // Test basic constructor functionality
  types::shape shape = {3};
  bool degrees = false;

  auto angle_block = vector_angle_sync<float>::make(shape, degrees);
  BOOST_REQUIRE(angle_block != nullptr);

  // Test with degrees = true
  auto angle_block_deg = vector_angle_sync<float>::make(shape, true);
  BOOST_REQUIRE(angle_block_deg != nullptr);
}

BOOST_AUTO_TEST_CASE(test_vector_angle_compute_output_shapes) {
  // Test output shape computation
  types::shapes input_shapes = {{3}, {3}};
  auto output_shapes = vector_angle<float>::compute_output_shapes(input_shapes);

  BOOST_REQUIRE_EQUAL(output_shapes.size(), 1);
  BOOST_REQUIRE_EQUAL(output_shapes[0].size(), 1);
  BOOST_REQUIRE_EQUAL(output_shapes[0][0], 1);
}

BOOST_AUTO_TEST_CASE(test_vector_angle_compute_output_shapes_invalid) {
  // Test error handling for mismatched shapes
  types::shapes mismatched_shapes = {{3}, {4}};

  BOOST_CHECK_THROW(
      vector_angle<float>::compute_output_shapes(mismatched_shapes),
      std::invalid_argument);

  // Test error handling for wrong number of inputs
  types::shapes single_shape = {{3}};

  BOOST_CHECK_THROW(vector_angle<float>::compute_output_shapes(single_shape),
                    std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_vector_angle_orthogonal_vectors_radians) {
  // Test angle between orthogonal vectors (should be π/2 radians)
  types::shape shape = {2};

  vector_angle<float> angle_op(shape, false); // radians

  // Create orthogonal vectors: [1, 0] and [0, 1]
  types::matrix<float> a(2, 1);
  types::matrix<float> b(2, 1);
  types::matrix<float> result(1, 1);

  a(0, 0) = 1.0f; // [1, 0]
  a(1, 0) = 0.0f;
  b(0, 0) = 0.0f; // [0, 1]
  b(1, 0) = 1.0f;

  types::vector_const_matrix_map<float> inputs = {a, b};
  types::vector_matrix_map<float> outputs = {result};

  auto op_result = angle_op.operation(inputs, outputs);
  BOOST_REQUIRE_EQUAL(op_result, OperationReturn::SUCCESS);

  // Should be π/2 ≈ 1.5708 radians
  BOOST_CHECK_CLOSE(result(0, 0), M_PI / 2, 1e-5);
}

BOOST_AUTO_TEST_CASE(test_vector_angle_orthogonal_vectors_degrees) {
  // Test angle between orthogonal vectors (should be 90 degrees)
  types::shape shape = {2};

  vector_angle<float> angle_op(shape, true); // degrees

  // Create orthogonal vectors: [1, 0] and [0, 1]
  types::matrix<float> a(2, 1);
  types::matrix<float> b(2, 1);
  types::matrix<float> result(1, 1);

  a(0, 0) = 1.0f; // [1, 0]
  a(1, 0) = 0.0f;
  b(0, 0) = 0.0f; // [0, 1]
  b(1, 0) = 1.0f;

  types::vector_const_matrix_map<float> inputs = {a, b};
  types::vector_matrix_map<float> outputs = {result};

  auto op_result = angle_op.operation(inputs, outputs);
  BOOST_REQUIRE_EQUAL(op_result, OperationReturn::SUCCESS);

  // Should be 90 degrees
  BOOST_CHECK_CLOSE(result(0, 0), 90.0f, 1e-5);
}

BOOST_AUTO_TEST_CASE(test_vector_angle_parallel_vectors) {
  // Test angle between parallel vectors (should be 0)
  types::shape shape = {3};

  vector_angle<float> angle_op(shape, false); // radians

  // Create parallel vectors: [1, 2, 3] and [2, 4, 6]
  types::matrix<float> a(3, 1);
  types::matrix<float> b(3, 1);
  types::matrix<float> result(1, 1);

  a(0, 0) = 1.0f;
  a(1, 0) = 2.0f;
  a(2, 0) = 3.0f;
  b(0, 0) = 2.0f; // b = 2*a
  b(1, 0) = 4.0f;
  b(2, 0) = 6.0f;

  types::vector_const_matrix_map<float> inputs = {a, b};
  types::vector_matrix_map<float> outputs = {result};

  auto op_result = angle_op.operation(inputs, outputs);
  BOOST_REQUIRE_EQUAL(op_result, OperationReturn::SUCCESS);

  // Should be 0 radians
  BOOST_CHECK_SMALL(result(0, 0), 1e-5f);
}

BOOST_AUTO_TEST_CASE(test_vector_angle_antiparallel_vectors) {
  // Test angle between anti-parallel vectors (should be π radians)
  types::shape shape = {3};

  vector_angle<float> angle_op(shape, false); // radians

  // Create anti-parallel vectors: [1, 2, 3] and [-1, -2, -3]
  types::matrix<float> a(3, 1);
  types::matrix<float> b(3, 1);
  types::matrix<float> result(1, 1);

  a(0, 0) = 1.0f;
  a(1, 0) = 2.0f;
  a(2, 0) = 3.0f;
  b(0, 0) = -1.0f; // b = -a
  b(1, 0) = -2.0f;
  b(2, 0) = -3.0f;

  types::vector_const_matrix_map<float> inputs = {a, b};
  types::vector_matrix_map<float> outputs = {result};

  auto op_result = angle_op.operation(inputs, outputs);
  BOOST_REQUIRE_EQUAL(op_result, OperationReturn::SUCCESS);

  // Should be π radians
  BOOST_CHECK_CLOSE(result(0, 0), M_PI, 1e-5);
}

BOOST_AUTO_TEST_CASE(test_vector_angle_known_angle) {
  // Test angle with known result: vectors at 60 degrees
  types::shape shape = {2};

  vector_angle<float> angle_op(shape, true); // degrees

  // Create vectors at 60-degree angle
  // Vector a: [1, 0] (along x-axis)
  // Vector b: [cos(60°), sin(60°)] = [0.5, √3/2]
  types::matrix<float> a(2, 1);
  types::matrix<float> b(2, 1);
  types::matrix<float> result(1, 1);

  a(0, 0) = 1.0f;
  a(1, 0) = 0.0f;
  b(0, 0) = 0.5f;                // cos(60°)
  b(1, 0) = std::sqrt(3.0f) / 2; // sin(60°)

  types::vector_const_matrix_map<float> inputs = {a, b};
  types::vector_matrix_map<float> outputs = {result};

  auto op_result = angle_op.operation(inputs, outputs);
  BOOST_REQUIRE_EQUAL(op_result, OperationReturn::SUCCESS);

  // Should be 60 degrees
  BOOST_CHECK_CLOSE(result(0, 0), 60.0f, 1e-3);
}

BOOST_AUTO_TEST_CASE(test_vector_angle_zero_vector_handling) {
  // Test handling of zero vectors (should return NaN)
  types::shape shape = {3};

  vector_angle<float> angle_op(shape, false);

  // Create zero vector and normal vector
  types::matrix<float> a(3, 1);
  types::matrix<float> b(3, 1);
  types::matrix<float> result(1, 1);

  a(0, 0) = 0.0f; // Zero vector
  a(1, 0) = 0.0f;
  a(2, 0) = 0.0f;
  b(0, 0) = 1.0f; // Normal vector
  b(1, 0) = 0.0f;
  b(2, 0) = 0.0f;

  types::vector_const_matrix_map<float> inputs = {a, b};
  types::vector_matrix_map<float> outputs = {result};

  auto op_result = angle_op.operation(inputs, outputs);
  BOOST_REQUIRE_EQUAL(op_result, OperationReturn::SUCCESS);

  // Should be NaN for zero vector
  BOOST_CHECK(std::isnan(result(0, 0)));
}

BOOST_AUTO_TEST_CASE(test_vector_angle_double_precision) {
  // Test with double precision
  types::shape shape = {2};

  vector_angle<double> angle_op(shape, false); // radians

  // Create orthogonal vectors
  types::matrix<double> a(2, 1);
  types::matrix<double> b(2, 1);
  types::matrix<double> result(1, 1);

  a(0, 0) = 1.0;
  a(1, 0) = 0.0;
  b(0, 0) = 0.0;
  b(1, 0) = 1.0;

  types::vector_const_matrix_map<double> inputs = {a, b};
  types::vector_matrix_map<double> outputs = {result};

  auto op_result = angle_op.operation(inputs, outputs);
  BOOST_REQUIRE_EQUAL(op_result, OperationReturn::SUCCESS);

  // Should be π/2 radians with double precision
  BOOST_CHECK_CLOSE(result(0, 0), M_PI / 2, 1e-10);
}

BOOST_AUTO_TEST_CASE(test_vector_angle_complex_vectors) {
  // Test angle between complex vectors
  types::shape shape = {2};

  vector_angle<std::complex<float>> angle_op(shape, false); // radians

  // Create complex vectors
  types::matrix<std::complex<float>> a(2, 1);
  types::matrix<std::complex<float>> b(2, 1);
  types::matrix<std::complex<float>> result(1, 1);

  a(0, 0) = std::complex<float>(1.0f, 0.0f); // [1+0j, 0+0j]
  a(1, 0) = std::complex<float>(0.0f, 0.0f);
  b(0, 0) = std::complex<float>(0.0f, 0.0f); // [0+0j, 1+0j]
  b(1, 0) = std::complex<float>(1.0f, 0.0f);

  types::vector_const_matrix_map<std::complex<float>> inputs = {a, b};
  types::vector_matrix_map<std::complex<float>> outputs = {result};

  auto op_result = angle_op.operation(inputs, outputs);
  BOOST_REQUIRE_EQUAL(op_result, OperationReturn::SUCCESS);

  // For complex vectors, we expect the angle to be real
  // These are orthogonal complex vectors, so angle should be π/2
  BOOST_CHECK_CLOSE(std::real(result(0, 0)), M_PI / 2, 1e-5);
  BOOST_CHECK_SMALL(std::imag(result(0, 0)),
                    1e-10f); // Imaginary part should be zero
}

BOOST_AUTO_TEST_CASE(test_vector_angle_high_dimensional) {
  // Test with higher dimensional vectors
  types::shape shape = {5};

  vector_angle<float> angle_op(shape, true); // degrees

  // Create two vectors with known angle
  types::matrix<float> a(5, 1);
  types::matrix<float> b(5, 1);
  types::matrix<float> result(1, 1);

  // Vector a: [1, 0, 0, 0, 0]
  a(0, 0) = 1.0f;
  a(1, 0) = 0.0f;
  a(2, 0) = 0.0f;
  a(3, 0) = 0.0f;
  a(4, 0) = 0.0f;

  // Vector b: [0, 1, 0, 0, 0] (orthogonal to a)
  b(0, 0) = 0.0f;
  b(1, 0) = 1.0f;
  b(2, 0) = 0.0f;
  b(3, 0) = 0.0f;
  b(4, 0) = 0.0f;

  types::vector_const_matrix_map<float> inputs = {a, b};
  types::vector_matrix_map<float> outputs = {result};

  auto op_result = angle_op.operation(inputs, outputs);
  BOOST_REQUIRE_EQUAL(op_result, OperationReturn::SUCCESS);

  // Should be 90 degrees
  BOOST_CHECK_CLOSE(result(0, 0), 90.0f, 1e-5);
}

BOOST_AUTO_TEST_CASE(test_vector_angle_error_handling) {
  // Test error handling for invalid inputs
  types::shape shape = {3};

  vector_angle<float> angle_op(shape, false);

  types::matrix<float> a(3, 1);
  types::matrix<float> result(1, 1);

  // Test with wrong number of inputs
  types::vector_const_matrix_map<float> single_input = {a};
  types::vector_matrix_map<float> outputs = {result};

  auto op_result = angle_op.operation(single_input, outputs);
  BOOST_REQUIRE_EQUAL(op_result, OperationReturn::INVALID_INPUT_SIZE);

  // Test with wrong number of outputs
  types::vector_const_matrix_map<float> inputs = {a, a};
  types::vector_matrix_map<float> empty_outputs = {};

  op_result = angle_op.operation(inputs, empty_outputs);
  BOOST_REQUIRE_EQUAL(op_result, OperationReturn::INVALID_OUTPUT_SIZE);
}

} /* namespace linalg */
} /* namespace gr */
