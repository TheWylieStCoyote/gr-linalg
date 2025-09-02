/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <Eigen/Dense>
#include <boost/test/unit_test.hpp>
#include <complex>
#include <gnuradio/attributes.h>
#include <gnuradio/linalg/array_slice.h>
#include <gnuradio/linalg/types.h>
#include <vector>

namespace gr {
namespace linalg {

// Test utility to create a test matrix
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> create_test_matrix() {
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> matrix(4, 3);
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
    matrix << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
  } else {
    // Complex case
    matrix << T(1, 0), T(2, 0), T(3, 0), T(4, 0), T(5, 0), T(6, 0), T(7, 0),
        T(8, 0), T(9, 0), T(10, 0), T(11, 0), T(12, 0);
  }
  return matrix;
}

BOOST_AUTO_TEST_CASE(test_array_slice_1d_basic) {
  // Test basic 1D vector slicing
  types::shape input_shape{6}; // 1D vector of size 6

  array_slice<float> slicer(input_shape,
                            {array_slice<float>::SliceSpec(1, 4, 1)});

  // Create input vector [1, 2, 3, 4, 5, 6]
  Eigen::Matrix<float, Eigen::Dynamic, 1> input_vec(6, 1);
  input_vec << 1, 2, 3, 4, 5, 6;

  // Create output vector for slice [1:4] -> [2, 3, 4]
  Eigen::Matrix<float, Eigen::Dynamic, 1> output_vec(3, 1);

  // Create matrix maps with proper parameters
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      input_map(input_vec.data(), 6, 1);
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> output_map(
      output_vec.data(), 3, 1);

  types::vector_const_matrix_map<float> input_arrays{&input_map};
  types::vector_matrix_map<float> output_arrays{&output_map};

  OperationReturn result = slicer.operation(input_arrays, output_arrays);

  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);
  BOOST_CHECK_CLOSE(output_vec(0), 2.0f, 0.001);
  BOOST_CHECK_CLOSE(output_vec(1), 3.0f, 0.001);
  BOOST_CHECK_CLOSE(output_vec(2), 4.0f, 0.001);
}

BOOST_AUTO_TEST_CASE(test_array_slice_1d_strided) {
  // Test 1D vector slicing with stride
  types::shape input_shape{6}; // 1D vector of size 6

  array_slice<float> slicer(input_shape,
                            {array_slice<float>::SliceSpec(0, 6, 2)});

  // Create input vector [1, 2, 3, 4, 5, 6]
  Eigen::Matrix<float, Eigen::Dynamic, 1> input_vec(6, 1);
  input_vec << 1, 2, 3, 4, 5, 6;

  // Create output vector for slice [0:6:2] -> [1, 3, 5]
  Eigen::Matrix<float, Eigen::Dynamic, 1> output_vec(3, 1);

  // Create matrix maps with proper parameters
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      input_map(input_vec.data(), 6, 1);
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> output_map(
      output_vec.data(), 3, 1);

  types::vector_const_matrix_map<float> input_arrays{&input_map};
  types::vector_matrix_map<float> output_arrays{&output_map};

  OperationReturn result = slicer.operation(input_arrays, output_arrays);

  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);
  BOOST_CHECK_CLOSE(output_vec(0), 1.0f, 0.001);
  BOOST_CHECK_CLOSE(output_vec(1), 3.0f, 0.001);
  BOOST_CHECK_CLOSE(output_vec(2), 5.0f, 0.001);
}

BOOST_AUTO_TEST_CASE(test_array_slice_2d_basic) {
  // Test basic 2D matrix slicing
  types::shape input_shape{4, 3}; // 4x3 matrix

  // Slice rows 1-3, all columns
  array_slice<float> slicer(
      input_shape, {
                       array_slice<float>::SliceSpec(1, 3, 1), // rows 1-2
                       array_slice<float>::SliceSpec(0, 3, 1)  // all columns
                   });

  auto input_matrix = create_test_matrix<float>();

  // Expected output: 2x3 matrix with rows 1-2 from original
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> output_matrix(2, 3);

  // Create matrix maps with proper parameters
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      input_map(input_matrix.data(), 4, 3);
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> output_map(
      output_matrix.data(), 2, 3);

  types::vector_const_matrix_map<float> input_arrays{&input_map};
  types::vector_matrix_map<float> output_arrays{&output_map};

  OperationReturn result = slicer.operation(input_arrays, output_arrays);

  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Check that we got rows 1 and 2 from the original matrix
  BOOST_CHECK_CLOSE(output_matrix(0, 0), 4.0f, 0.001); // row 1, col 0
  BOOST_CHECK_CLOSE(output_matrix(0, 1), 5.0f, 0.001); // row 1, col 1
  BOOST_CHECK_CLOSE(output_matrix(0, 2), 6.0f, 0.001); // row 1, col 2
  BOOST_CHECK_CLOSE(output_matrix(1, 0), 7.0f, 0.001); // row 2, col 0
  BOOST_CHECK_CLOSE(output_matrix(1, 1), 8.0f, 0.001); // row 2, col 1
  BOOST_CHECK_CLOSE(output_matrix(1, 2), 9.0f, 0.001); // row 2, col 2
}

BOOST_AUTO_TEST_CASE(test_array_slice_2d_strided) {
  // Test 2D matrix slicing with stride
  types::shape input_shape{4, 3}; // 4x3 matrix

  // Slice every other row, every other column
  array_slice<float> slicer(
      input_shape, {
                       array_slice<float>::SliceSpec(0, 4, 2), // rows 0, 2
                       array_slice<float>::SliceSpec(0, 3, 2)  // cols 0, 2
                   });

  auto input_matrix = create_test_matrix<float>();

  // Expected output: 2x2 matrix
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> output_matrix(2, 2);

  // Create matrix maps with proper parameters
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      input_map(input_matrix.data(), 4, 3);
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> output_map(
      output_matrix.data(), 2, 2);

  types::vector_const_matrix_map<float> input_arrays{&input_map};
  types::vector_matrix_map<float> output_arrays{&output_map};

  OperationReturn result = slicer.operation(input_arrays, output_arrays);

  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Check the strided slice results
  BOOST_CHECK_CLOSE(output_matrix(0, 0), 1.0f, 0.001); // row 0, col 0
  BOOST_CHECK_CLOSE(output_matrix(0, 1), 3.0f, 0.001); // row 0, col 2
  BOOST_CHECK_CLOSE(output_matrix(1, 0), 7.0f, 0.001); // row 2, col 0
  BOOST_CHECK_CLOSE(output_matrix(1, 1), 9.0f, 0.001); // row 2, col 2
}

BOOST_AUTO_TEST_CASE(test_array_slice_negative_indices) {
  // Test negative indices (Python-like indexing from the end)
  types::shape input_shape{4}; // 1D vector of size 4

  // Slice from -3 to -1 (equivalent to 1 to 3 in 0-based indexing)
  array_slice<float> slicer(input_shape,
                            {array_slice<float>::SliceSpec(-3, -1, 1)});

  // Create input vector [1, 2, 3, 4]
  Eigen::Matrix<float, Eigen::Dynamic, 1> input_vec(4, 1);
  input_vec << 1, 2, 3, 4;

  // Create output vector for slice [-3:-1] -> [2, 3]
  Eigen::Matrix<float, Eigen::Dynamic, 1> output_vec(2, 1);

  // Create matrix maps with proper parameters
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      input_map(input_vec.data(), 4, 1);
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> output_map(
      output_vec.data(), 2, 1);

  types::vector_const_matrix_map<float> input_arrays{&input_map};
  types::vector_matrix_map<float> output_arrays{&output_map};

  OperationReturn result = slicer.operation(input_arrays, output_arrays);

  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);
  BOOST_CHECK_CLOSE(output_vec(0), 2.0f, 0.001);
  BOOST_CHECK_CLOSE(output_vec(1), 3.0f, 0.001);
}

BOOST_AUTO_TEST_CASE(test_array_slice_complex) {
  // Test with complex numbers
  types::shape input_shape{3}; // 1D complex vector

  array_slice<std::complex<float>> slicer(
      input_shape, {array_slice<std::complex<float>>::SliceSpec(0, 2, 1)});

  // Create input vector [(1+i), (2+2i), (3+3i)]
  Eigen::Matrix<std::complex<float>, Eigen::Dynamic, 1> input_vec(3, 1);
  input_vec << std::complex<float>(1, 1), std::complex<float>(2, 2),
      std::complex<float>(3, 3);

  // Create output vector for slice [0:2] -> [(1+i), (2+2i)]
  Eigen::Matrix<std::complex<float>, Eigen::Dynamic, 1> output_vec(2, 1);

  // Create matrix maps with proper parameters
  Eigen::Map<
      const Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic>>
      input_map(input_vec.data(), 3, 1);
  Eigen::Map<Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic>>
      output_map(output_vec.data(), 2, 1);

  types::vector_const_matrix_map<std::complex<float>> input_arrays{&input_map};
  types::vector_matrix_map<std::complex<float>> output_arrays{&output_map};

  OperationReturn result = slicer.operation(input_arrays, output_arrays);

  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);
  BOOST_CHECK_CLOSE(output_vec(0).real(), 1.0f, 0.001);
  BOOST_CHECK_CLOSE(output_vec(0).imag(), 1.0f, 0.001);
  BOOST_CHECK_CLOSE(output_vec(1).real(), 2.0f, 0.001);
  BOOST_CHECK_CLOSE(output_vec(1).imag(), 2.0f, 0.001);
}

BOOST_AUTO_TEST_CASE(test_array_slice_shape_computation) {
  // Test static shape computation methods
  types::vector_shapes input_shapes{{5}};
  std::vector<array_slice<float>::SliceSpec> specs{
      array_slice<float>::SliceSpec(1, 4, 1)};

  auto output_shapes =
      array_slice<float>::compute_output_shapes(input_shapes, specs);

  BOOST_CHECK_EQUAL(output_shapes.size(), 1);
  BOOST_CHECK_EQUAL(output_shapes[0].size(), 1);
  BOOST_CHECK_EQUAL(output_shapes[0][0], 3); // slice [1:4] has length 3

  int vlen = array_slice<float>::compute_output_vlen(input_shapes, specs);
  BOOST_CHECK_EQUAL(vlen, 3);
}

BOOST_AUTO_TEST_CASE(test_array_slice_integer_constructor) {
  // Test constructor with integer indices
  types::shape input_shape{3, 4}; // 3x4 matrix

  // Select element at [1, 2] - should create SliceSpecs (1,2,1) and (2,3,1)
  std::vector<int> indices = {1, 2};
  array_slice<float> slicer(input_shape, indices);

  auto input_matrix = create_test_matrix<float>();

  // Expected output: 1x1 matrix with element at [1, 2]
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> output_matrix(1, 1);

  // Create matrix maps
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      input_map(input_matrix.data(), 4,
                3); // Note: our test matrix is 4x3, not 3x4
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> output_map(
      output_matrix.data(), 1, 1);

  types::vector_const_matrix_map<float> input_arrays{&input_map};
  types::vector_matrix_map<float> output_arrays{&output_map};

  // This should select single element - adjust for 4x3 input matrix
  std::vector<int> indices_corrected = {1, 2};
  array_slice<float> slicer_corrected({4, 3}, indices_corrected);
  OperationReturn result =
      slicer_corrected.operation(input_arrays, output_arrays);

  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);
  BOOST_CHECK_CLOSE(output_matrix(0, 0), 6.0f,
                    0.001); // Element at row 1, col 2
}

BOOST_AUTO_TEST_CASE(test_array_slice_negative_integer_indices) {
  // Test constructor with negative integer indices
  types::shape input_shape{4}; // 1D vector of size 4

  // Select element at [-2] (equivalent to [2] in 0-based indexing)
  std::vector<int> negative_indices = {-2};
  array_slice<float> slicer(input_shape, negative_indices);

  // Create input vector [1, 2, 3, 4]
  Eigen::Matrix<float, Eigen::Dynamic, 1> input_vec(4, 1);
  input_vec << 1, 2, 3, 4;

  // Create output vector for single element
  Eigen::Matrix<float, Eigen::Dynamic, 1> output_vec(1, 1);

  // Create matrix maps
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      input_map(input_vec.data(), 4, 1);
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> output_map(
      output_vec.data(), 1, 1);

  types::vector_const_matrix_map<float> input_arrays{&input_map};
  types::vector_matrix_map<float> output_arrays{&output_map};

  OperationReturn result = slicer.operation(input_arrays, output_arrays);

  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);
  BOOST_CHECK_CLOSE(output_vec(0), 3.0f,
                    0.001); // Element at index 2 (3rd element)
}

BOOST_AUTO_TEST_CASE(test_array_slice_error_handling) {
  // Test error conditions
  types::shape input_shape{3};

  // Test with too many slice specs
  BOOST_CHECK_THROW(
      array_slice<float> slicer(input_shape,
                                {array_slice<float>::SliceSpec(0, 2, 1),
                                 array_slice<float>::SliceSpec(
                                     0, 1, 1), // Too many specs for 1D input
                                 array_slice<float>::SliceSpec(0, 1, 1)}),
      std::invalid_argument);

  // Test zero step size
  types::vector_shapes input_shapes{{3}};
  BOOST_CHECK_THROW(
      array_slice<float>::compute_output_shapes(
          input_shapes,
          {
              array_slice<float>::SliceSpec(0, 2, 0) // Invalid step = 0
          }),
      std::invalid_argument);

  // Test integer constructor with too many indices
  std::vector<int> too_many_indices = {0, 1, 2, 3};
  BOOST_CHECK_THROW(
      array_slice<float> slicer(
          input_shape, too_many_indices), // Too many indices for 1D input
      std::invalid_argument);
}

} /* namespace linalg */
} /* namespace gr */
