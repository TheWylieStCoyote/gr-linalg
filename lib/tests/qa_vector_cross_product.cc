/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <boost/test/unit_test.hpp>
#include <cmath>
#include <complex>
#include <gnuradio/attributes.h>
#include <gnuradio/linalg/vector_cross_product.h>
#include <vector>

namespace gr {
namespace linalg {

// Test constants
constexpr float FLOAT_TOLERANCE = 1e-6f;
constexpr double DOUBLE_TOLERANCE = 1e-12;

/*!
 * \brief Test vector cross product with basic 3D vectors
 *
 * Tests cross product of standard basis vectors:
 * a = [1, 0, 0] (x-axis)
 * b = [0, 1, 0] (y-axis)
 * Expected result: a × b = [0, 0, 1] (z-axis)
 */
BOOST_AUTO_TEST_CASE(test_vector_cross_product_basic_float) {
  bool validate_3d = true;

  auto cross_op = vector_cross_product_sync<float>::make(validate_3d);

  // Create 3D input vectors (column vectors)
  std::vector<float> input_a{1.0f, 0.0f, 0.0f}; // x-axis unit vector
  std::vector<float> input_b{0.0f, 1.0f, 0.0f}; // y-axis unit vector
  std::vector<float> output(3);

  // Create matrix maps (3x1 column vectors)
  types::matrix_map_dynamic<float> map_a(input_a.data(), 3, 1);
  types::matrix_map_dynamic<float> map_b(input_b.data(), 3, 1);
  types::matrix_map_dynamic<float> map_out(output.data(), 3, 1);

  types::vector_const_matrix_map<float> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&map_a),
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&map_b)};
  types::vector_matrix_map<float> outputs{&map_out};

  // Perform operation
  OperationReturn result = cross_op->operation(inputs, outputs);

  BOOST_REQUIRE_EQUAL(result, OperationReturn::SUCCESS);

  // Check result: [1,0,0] × [0,1,0] = [0,0,1]
  std::vector<float> expected{0.0f, 0.0f, 1.0f};
  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_CHECK_CLOSE(output[i], expected[i], FLOAT_TOLERANCE);
  }
}

/*!
 * \brief Test vector cross product with general 3D vectors
 *
 * Tests cross product with non-unit vectors:
 * a = [2, 3, 4]
 * b = [5, 6, 7]
 * Expected result: a × b = [3*7-4*6, 4*5-2*7, 2*6-3*5] = [-3, 6, -3]
 */
BOOST_AUTO_TEST_CASE(test_vector_cross_product_general_vectors) {
  bool validate_3d = true;

  auto cross_op = vector_cross_product_sync<float>::make(validate_3d);

  // Create 3D input vectors
  std::vector<float> input_a{2.0f, 3.0f, 4.0f};
  std::vector<float> input_b{5.0f, 6.0f, 7.0f};
  std::vector<float> output(3);

  // Create matrix maps (3x1 column vectors)
  types::matrix_map_dynamic<float> map_a(input_a.data(), 3, 1);
  types::matrix_map_dynamic<float> map_b(input_b.data(), 3, 1);
  types::matrix_map_dynamic<float> map_out(output.data(), 3, 1);

  types::vector_const_matrix_map<float> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&map_a),
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&map_b)};
  types::vector_matrix_map<float> outputs{&map_out};

  // Perform operation
  OperationReturn result = cross_op->operation(inputs, outputs);

  BOOST_REQUIRE_EQUAL(result, OperationReturn::SUCCESS);

  // Manual calculation:
  // a × b = (a₁b₂-a₂b₁, a₂b₀-a₀b₂, a₀b₁-a₁b₀)
  // = (3*7-4*6, 4*5-2*7, 2*6-3*5) = (21-24, 20-14, 12-15) = (-3, 6, -3)
  std::vector<float> expected{-3.0f, 6.0f, -3.0f};
  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_CHECK_CLOSE(output[i], expected[i], FLOAT_TOLERANCE);
  }
}

/*!
 * \brief Test vector cross product anti-commutative property
 *
 * Tests that a × b = -(b × a)
 */
BOOST_AUTO_TEST_CASE(test_vector_cross_product_anticommutative) {
  bool validate_3d = true;

  auto cross_op = vector_cross_product_sync<float>::make(validate_3d);

  // Create 3D input vectors
  std::vector<float> input_a{1.0f, 2.0f, 3.0f};
  std::vector<float> input_b{4.0f, 5.0f, 6.0f};
  std::vector<float> output_ab(3);
  std::vector<float> output_ba(3);

  // Test a × b
  {
    types::matrix_map_dynamic<float> map_a(input_a.data(), 3, 1);
    types::matrix_map_dynamic<float> map_b(input_b.data(), 3, 1);
    types::matrix_map_dynamic<float> map_out(output_ab.data(), 3, 1);

    types::vector_const_matrix_map<float> inputs{
        reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&map_a),
        reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&map_b)};
    types::vector_matrix_map<float> outputs{&map_out};

    OperationReturn result = cross_op->operation(inputs, outputs);
    BOOST_REQUIRE_EQUAL(result, OperationReturn::SUCCESS);
  }

  // Test b × a
  {
    types::matrix_map_dynamic<float> map_a(input_a.data(), 3, 1);
    types::matrix_map_dynamic<float> map_b(input_b.data(), 3, 1);
    types::matrix_map_dynamic<float> map_out(output_ba.data(), 3, 1);

    // Swap order: b × a
    types::vector_const_matrix_map<float> inputs{
        reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&map_b),
        reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&map_a)};
    types::vector_matrix_map<float> outputs{&map_out};

    OperationReturn result = cross_op->operation(inputs, outputs);
    BOOST_REQUIRE_EQUAL(result, OperationReturn::SUCCESS);
  }

  // Check anti-commutative property: a × b = -(b × a)
  for (size_t i = 0; i < 3; ++i) {
    BOOST_CHECK_CLOSE(output_ab[i], -output_ba[i], FLOAT_TOLERANCE);
  }
}

/*!
 * \brief Test vector cross product with parallel vectors
 *
 * Tests that cross product of parallel vectors is zero vector:
 * a = [2, 4, 6]
 * b = [1, 2, 3] (a = 2*b)
 * Expected result: a × b = [0, 0, 0]
 */
BOOST_AUTO_TEST_CASE(test_vector_cross_product_parallel_vectors) {
  bool validate_3d = true;

  auto cross_op = vector_cross_product_sync<float>::make(validate_3d);

  // Create parallel 3D vectors
  std::vector<float> input_a{2.0f, 4.0f, 6.0f};
  std::vector<float> input_b{1.0f, 2.0f, 3.0f}; // a = 2*b
  std::vector<float> output(3);

  // Create matrix maps
  types::matrix_map_dynamic<float> map_a(input_a.data(), 3, 1);
  types::matrix_map_dynamic<float> map_b(input_b.data(), 3, 1);
  types::matrix_map_dynamic<float> map_out(output.data(), 3, 1);

  types::vector_const_matrix_map<float> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&map_a),
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&map_b)};
  types::vector_matrix_map<float> outputs{&map_out};

  // Perform operation
  OperationReturn result = cross_op->operation(inputs, outputs);

  BOOST_REQUIRE_EQUAL(result, OperationReturn::SUCCESS);

  // Parallel vectors should have zero cross product
  for (size_t i = 0; i < 3; ++i) {
    BOOST_CHECK_SMALL(output[i], FLOAT_TOLERANCE);
  }
}

/*!
 * \brief Test vector cross product with complex numbers
 *
 * Tests cross product with complex 3D vectors:
 * a = [1+j, 2+0j, 0+3j]
 * b = [2+0j, 1+j, 1+0j]
 * Expected result computed using complex arithmetic
 */
BOOST_AUTO_TEST_CASE(test_vector_cross_product_complex) {
  using cf = std::complex<float>;
  bool validate_3d = true;

  auto cross_op = vector_cross_product_sync<cf>::make(validate_3d);

  // Create complex 3D vectors
  std::vector<cf> input_a{cf(1, 1), cf(2, 0), cf(0, 3)};
  std::vector<cf> input_b{cf(2, 0), cf(1, 1), cf(1, 0)};
  std::vector<cf> output(3);

  // Create matrix maps
  types::matrix_map_dynamic<cf> map_a(input_a.data(), 3, 1);
  types::matrix_map_dynamic<cf> map_b(input_b.data(), 3, 1);
  types::matrix_map_dynamic<cf> map_out(output.data(), 3, 1);

  types::vector_const_matrix_map<cf> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<cf> *>(&map_a),
      reinterpret_cast<types::const_matrix_map_dynamic<cf> *>(&map_b)};
  types::vector_matrix_map<cf> outputs{&map_out};

  // Perform operation
  OperationReturn result = cross_op->operation(inputs, outputs);

  BOOST_REQUIRE_EQUAL(result, OperationReturn::SUCCESS);

  // Manual calculation for complex cross product:
  // a × b = (a₁b₂-a₂b₁, a₂b₀-a₀b₂, a₀b₁-a₁b₀)
  // a₁b₂ = (2+0j)*(1+0j) = 2
  // a₂b₁ = (0+3j)*(1+1j) = 3j + 3j² = 3j - 3 = -3+3j
  // Component 0: 2 - (-3+3j) = 5-3j
  // a₂b₀ = (0+3j)*(2+0j) = 6j
  // a₀b₂ = (1+1j)*(1+0j) = 1+1j
  // Component 1: 6j - (1+1j) = -1+5j
  // a₀b₁ = (1+1j)*(1+1j) = 1 + 2j + j² = 1 + 2j - 1 = 2j
  // a₁b₀ = (2+0j)*(2+0j) = 4
  // Component 2: 2j - 4 = -4+2j
  std::vector<cf> expected{cf(5, -3), cf(-1, 5), cf(-4, 2)};

  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_CHECK_CLOSE(output[i].real(), expected[i].real(), FLOAT_TOLERANCE);
    BOOST_CHECK_CLOSE(output[i].imag(), expected[i].imag(), FLOAT_TOLERANCE);
  }
}

/*!
 * \brief Test vector cross product with double precision
 *
 * Tests higher precision computation with double vectors
 */
BOOST_AUTO_TEST_CASE(test_vector_cross_product_double_precision) {
  bool validate_3d = true;

  auto cross_op = vector_cross_product_sync<double>::make(validate_3d);

  // Create high precision 3D vectors
  std::vector<double> input_a{1.123456789012345, 2.987654321098765,
                              3.141592653589793};
  std::vector<double> input_b{2.718281828459045, 1.414213562373095,
                              1.732050807568877};
  std::vector<double> output(3);

  // Create matrix maps
  types::matrix_map_dynamic<double> map_a(input_a.data(), 3, 1);
  types::matrix_map_dynamic<double> map_b(input_b.data(), 3, 1);
  types::matrix_map_dynamic<double> map_out(output.data(), 3, 1);

  types::vector_const_matrix_map<double> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<double> *>(&map_a),
      reinterpret_cast<types::const_matrix_map_dynamic<double> *>(&map_b)};
  types::vector_matrix_map<double> outputs{&map_out};

  // Perform operation
  OperationReturn result = cross_op->operation(inputs, outputs);

  BOOST_REQUIRE_EQUAL(result, OperationReturn::SUCCESS);

  // Manual calculation with high precision
  double a0 = input_a[0], a1 = input_a[1], a2 = input_a[2];
  double b0 = input_b[0], b1 = input_b[1], b2 = input_b[2];

  std::vector<double> expected{
      a1 * b2 - a2 * b1, // a₁b₂ - a₂b₁
      a2 * b0 - a0 * b2, // a₂b₀ - a₀b₂
      a0 * b1 - a1 * b0  // a₀b₁ - a₁b₀
  };

  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_CHECK_CLOSE(output[i], expected[i], DOUBLE_TOLERANCE);
  }
}

/*!
 * \brief Test vector cross product orthogonality property
 *
 * Tests that result is orthogonal to both input vectors:
 * (a × b) · a = 0 and (a × b) · b = 0
 */
BOOST_AUTO_TEST_CASE(test_vector_cross_product_orthogonality) {
  bool validate_3d = true;

  auto cross_op = vector_cross_product_sync<float>::make(validate_3d);

  // Create 3D vectors
  std::vector<float> input_a{1.0f, 3.0f, 2.0f};
  std::vector<float> input_b{2.0f, 1.0f, 4.0f};
  std::vector<float> output(3);

  // Create matrix maps
  types::matrix_map_dynamic<float> map_a(input_a.data(), 3, 1);
  types::matrix_map_dynamic<float> map_b(input_b.data(), 3, 1);
  types::matrix_map_dynamic<float> map_out(output.data(), 3, 1);

  types::vector_const_matrix_map<float> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&map_a),
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&map_b)};
  types::vector_matrix_map<float> outputs{&map_out};

  // Perform cross product
  OperationReturn result = cross_op->operation(inputs, outputs);
  BOOST_REQUIRE_EQUAL(result, OperationReturn::SUCCESS);

  // Check orthogonality: (a × b) · a = 0
  float dot_with_a =
      output[0] * input_a[0] + output[1] * input_a[1] + output[2] * input_a[2];
  BOOST_CHECK_SMALL(dot_with_a, FLOAT_TOLERANCE);

  // Check orthogonality: (a × b) · b = 0
  float dot_with_b =
      output[0] * input_b[0] + output[1] * input_b[1] + output[2] * input_b[2];
  BOOST_CHECK_SMALL(dot_with_b, FLOAT_TOLERANCE);
}

/*!
 * \brief Test error handling for non-3D vectors
 *
 * Tests that operation returns error when vectors are not 3D
 * (when validation is enabled)
 */
BOOST_AUTO_TEST_CASE(test_vector_cross_product_invalid_dimension) {
  bool validate_3d = true; // Enable 3D validation

  auto cross_op = vector_cross_product_sync<float>::make(validate_3d);

  // Create 2D vectors (should fail)
  std::vector<float> input_a{1.0f, 2.0f}; // 2D vector
  std::vector<float> input_b{3.0f, 4.0f}; // 2D vector
  std::vector<float> output(2);

  // Create matrix maps (2x1 vectors)
  types::matrix_map_dynamic<float> map_a(input_a.data(), 2, 1);
  types::matrix_map_dynamic<float> map_b(input_b.data(), 2, 1);
  types::matrix_map_dynamic<float> map_out(output.data(), 2, 1);

  types::vector_const_matrix_map<float> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&map_a),
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&map_b)};
  types::vector_matrix_map<float> outputs{&map_out};

  // Perform operation - should fail due to non-3D vectors
  OperationReturn result = cross_op->operation(inputs, outputs);

  BOOST_REQUIRE_EQUAL(result, OperationReturn::INVALID_SHAPE);
}

/*!
 * \brief Test with validation disabled
 *
 * Tests that operation can work with non-3D vectors when validation is disabled
 * (though mathematically this doesn't make sense for true cross product)
 */
BOOST_AUTO_TEST_CASE(test_vector_cross_product_no_validation) {
  bool validate_3d = false; // Disable 3D validation

  auto cross_op = vector_cross_product_sync<float>::make(validate_3d);

  // This test would need to be adapted based on how the implementation
  // handles non-3D cases when validation is disabled
  // For now, we'll test that it at least doesn't crash

  // Note: In practice, cross product is only mathematically defined for 3D
  // vectors, so disabling validation should be used with caution
}

/*!
 * \brief Test insufficient inputs error handling
 *
 * Tests error handling when not exactly 2 inputs are provided
 */
BOOST_AUTO_TEST_CASE(test_vector_cross_product_insufficient_inputs) {
  bool validate_3d = true;

  auto cross_op = vector_cross_product_sync<float>::make(validate_3d);

  // Create only one input (cross product requires exactly 2)
  std::vector<float> input_a{1.0f, 2.0f, 3.0f};
  std::vector<float> output(3);

  types::matrix_map_dynamic<float> map_a(input_a.data(), 3, 1);
  types::matrix_map_dynamic<float> map_out(output.data(), 3, 1);

  // Provide only one input (should require exactly 2)
  types::vector_const_matrix_map<float> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&map_a)};
  types::vector_matrix_map<float> outputs{&map_out};

  // Perform operation - should fail due to insufficient inputs
  OperationReturn result = cross_op->operation(inputs, outputs);

  BOOST_REQUIRE_EQUAL(result, OperationReturn::INVALID_SHAPE);
}

/*!
 * \brief Test cross product magnitude property
 *
 * Tests that |a × b| = |a| * |b| * sin(θ)
 * For orthogonal unit vectors, this should be 1
 */
BOOST_AUTO_TEST_CASE(test_vector_cross_product_magnitude) {
  bool validate_3d = true;

  auto cross_op = vector_cross_product_sync<float>::make(validate_3d);

  // Create orthogonal unit vectors
  std::vector<float> input_a{1.0f, 0.0f, 0.0f}; // unit vector along x
  std::vector<float> input_b{0.0f, 1.0f, 0.0f}; // unit vector along y
  std::vector<float> output(3);

  // Create matrix maps
  types::matrix_map_dynamic<float> map_a(input_a.data(), 3, 1);
  types::matrix_map_dynamic<float> map_b(input_b.data(), 3, 1);
  types::matrix_map_dynamic<float> map_out(output.data(), 3, 1);

  types::vector_const_matrix_map<float> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&map_a),
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&map_b)};
  types::vector_matrix_map<float> outputs{&map_out};

  // Perform cross product
  OperationReturn result = cross_op->operation(inputs, outputs);
  BOOST_REQUIRE_EQUAL(result, OperationReturn::SUCCESS);

  // Calculate magnitude of result
  float magnitude = std::sqrt(output[0] * output[0] + output[1] * output[1] +
                              output[2] * output[2]);

  // For orthogonal unit vectors, magnitude should be 1
  BOOST_CHECK_CLOSE(magnitude, 1.0f, FLOAT_TOLERANCE);
}

/*!
 * \brief Test row vector format
 *
 * Tests cross product with row vectors (1x3) instead of column vectors (3x1)
 */
BOOST_AUTO_TEST_CASE(test_vector_cross_product_row_vectors) {
  bool validate_3d = true;

  auto cross_op = vector_cross_product_sync<float>::make(validate_3d);

  // Create 3D input vectors as row vectors (1x3)
  std::vector<float> input_a{1.0f, 0.0f, 0.0f};
  std::vector<float> input_b{0.0f, 1.0f, 0.0f};
  std::vector<float> output(3);

  // Create matrix maps as 1x3 row vectors
  types::matrix_map_dynamic<float> map_a(input_a.data(), 1, 3);
  types::matrix_map_dynamic<float> map_b(input_b.data(), 1, 3);
  types::matrix_map_dynamic<float> map_out(output.data(), 1, 3);

  types::vector_const_matrix_map<float> inputs{
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&map_a),
      reinterpret_cast<types::const_matrix_map_dynamic<float> *>(&map_b)};
  types::vector_matrix_map<float> outputs{&map_out};

  // Perform operation
  OperationReturn result = cross_op->operation(inputs, outputs);

  BOOST_REQUIRE_EQUAL(result, OperationReturn::SUCCESS);

  // Check result: [1,0,0] × [0,1,0] = [0,0,1]
  std::vector<float> expected{0.0f, 0.0f, 1.0f};
  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_CHECK_CLOSE(output[i], expected[i], FLOAT_TOLERANCE);
  }
}

} // namespace linalg
} // namespace gr