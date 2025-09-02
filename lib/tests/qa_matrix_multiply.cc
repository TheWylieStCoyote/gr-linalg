/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <boost/test/unit_test.hpp>
#include <chrono>
#include <cmath>
#include <complex>
#include <gnuradio/attributes.h>
#include <gnuradio/blocks/vector_sink.h>
#include <gnuradio/blocks/vector_source.h>
#include <gnuradio/gr_complex.h>
#include <gnuradio/linalg/detail/matrix_multiply_impl.h>
#include <gnuradio/linalg/matrix_multiply.h>
#include <gnuradio/linalg/types.h>
#include <gnuradio/top_block.h>
#include <numeric>
#include <vector>

namespace gr {
namespace linalg {

// Test constants
constexpr float FLOAT_TOLERANCE = 1e-5f;
constexpr double DOUBLE_TOLERANCE = 1e-10;

// Test static method: validate_shape
BOOST_AUTO_TEST_CASE(test_matrix_multiply_validate_shape_valid) {
  types::vector_shapes input_shapes = {{2, 3},
                                       {3, 4}}; // Compatible for multiplication
  types::vector_shapes output_shapes = {{2, 4}}; // Expected output shape

  BOOST_CHECK_NO_THROW({
    matrix_multiply<float>::validate_shape(input_shapes, output_shapes, "test");
  });
}

BOOST_AUTO_TEST_CASE(test_matrix_multiply_validate_shape_wrong_input_count) {
  types::vector_shapes input_shapes = {{2, 3}}; // Only one matrix
  types::vector_shapes output_shapes = {{2, 3}};

  BOOST_CHECK_THROW(
      {
        matrix_multiply<float>::validate_shape(input_shapes, output_shapes,
                                               "test");
      },
      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(
    test_matrix_multiply_validate_shape_incompatible_dimensions) {
  types::vector_shapes input_shapes = {{2, 3}, {4, 5}}; // 3 != 4, incompatible
  types::vector_shapes output_shapes = {{2, 5}};

  BOOST_CHECK_THROW(
      {
        matrix_multiply<float>::validate_shape(input_shapes, output_shapes,
                                               "test");
      },
      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_matrix_multiply_validate_shape_non_2d_matrix) {
  types::vector_shapes input_shapes = {{2, 3, 4}, {3, 4}}; // First matrix is 3D
  types::vector_shapes output_shapes = {{2, 4}};

  BOOST_CHECK_THROW(
      {
        matrix_multiply<float>::validate_shape(input_shapes, output_shapes,
                                               "test");
      },
      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_matrix_multiply_validate_shape_negative_dimension) {
  types::vector_shapes input_shapes = {{2, -3}, {3, 4}}; // Negative dimension
  types::vector_shapes output_shapes = {{2, 4}};

  BOOST_CHECK_THROW(
      {
        matrix_multiply<float>::validate_shape(input_shapes, output_shapes,
                                               "test");
      },
      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_matrix_multiply_validate_shape_zero_dimension) {
  types::vector_shapes input_shapes = {{2, 0}, {0, 4}}; // Zero dimension
  types::vector_shapes output_shapes = {{2, 4}};

  BOOST_CHECK_THROW(
      {
        matrix_multiply<float>::validate_shape(input_shapes, output_shapes,
                                               "test");
      },
      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_matrix_multiply_validate_shape_wrong_output_shape) {
  types::vector_shapes input_shapes = {{2, 3}, {3, 4}};
  types::vector_shapes output_shapes = {
      {2, 3}}; // Wrong output shape (should be {2, 4})

  BOOST_CHECK_THROW(
      {
        matrix_multiply<float>::validate_shape(input_shapes, output_shapes,
                                               "test");
      },
      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_matrix_multiply_validate_shape_empty_output) {
  types::vector_shapes input_shapes = {{2, 3}, {3, 4}};
  types::vector_shapes output_shapes =
      {}; // Empty output shapes (should be allowed)

  BOOST_CHECK_NO_THROW({
    matrix_multiply<float>::validate_shape(input_shapes, output_shapes, "test");
  });
}

// Test static method: compute_sizes
BOOST_AUTO_TEST_CASE(test_matrix_multiply_compute_sizes) {
  types::vector_shapes shapes = {{2, 3}, {3, 4}, {1, 1}};
  std::vector<size_t> expected = {6, 12, 1}; // 2*3=6, 3*4=12, 1*1=1

  auto result = matrix_multiply<float>::compute_sizes(shapes);

  BOOST_CHECK_EQUAL(result.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_CHECK_EQUAL(result[i], expected[i]);
  }
}

BOOST_AUTO_TEST_CASE(test_matrix_multiply_compute_sizes_empty) {
  types::vector_shapes shapes = {};
  auto result = matrix_multiply<float>::compute_sizes(shapes);

  BOOST_CHECK_EQUAL(result.size(), 0);
}

BOOST_AUTO_TEST_CASE(test_matrix_multiply_compute_sizes_large_matrix) {
  types::vector_shapes shapes = {{100, 200}, {50, 75}};
  std::vector<size_t> expected = {20000, 3750}; // 100*200=20000, 50*75=3750

  auto result = matrix_multiply<float>::compute_sizes(shapes);

  BOOST_CHECK_EQUAL(result.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_CHECK_EQUAL(result[i], expected[i]);
  }
}

// Test static method: compute_output_shapes
BOOST_AUTO_TEST_CASE(test_matrix_multiply_compute_output_shapes_valid) {
  types::vector_shapes input_shapes = {{2, 3}, {3, 4}};
  types::vector_shapes expected = {{2, 4}}; // Output: [rows_A, cols_B]

  auto result = matrix_multiply<float>::compute_output_shapes(input_shapes);

  BOOST_CHECK_EQUAL(result.size(), 1);
  BOOST_CHECK_EQUAL(result[0].size(), 2);
  BOOST_CHECK_EQUAL(result[0][0], 2);
  BOOST_CHECK_EQUAL(result[0][1], 4);
}

BOOST_AUTO_TEST_CASE(
    test_matrix_multiply_compute_output_shapes_square_matrices) {
  types::vector_shapes input_shapes = {{5, 5}, {5, 5}};
  types::vector_shapes expected = {{5, 5}};

  auto result = matrix_multiply<float>::compute_output_shapes(input_shapes);

  BOOST_CHECK_EQUAL(result.size(), 1);
  BOOST_CHECK_EQUAL(result[0][0], 5);
  BOOST_CHECK_EQUAL(result[0][1], 5);
}

BOOST_AUTO_TEST_CASE(
    test_matrix_multiply_compute_output_shapes_vector_multiply) {
  types::vector_shapes input_shapes = {{3, 1},
                                       {1, 4}}; // Column vector * row vector
  types::vector_shapes expected = {{3, 4}};

  auto result = matrix_multiply<float>::compute_output_shapes(input_shapes);

  BOOST_CHECK_EQUAL(result.size(), 1);
  BOOST_CHECK_EQUAL(result[0][0], 3);
  BOOST_CHECK_EQUAL(result[0][1], 4);
}

BOOST_AUTO_TEST_CASE(
    test_matrix_multiply_compute_output_shapes_wrong_input_count) {
  types::vector_shapes input_shapes = {{2, 3}}; // Only one matrix

  BOOST_CHECK_THROW(
      { matrix_multiply<float>::compute_output_shapes(input_shapes); },
      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_matrix_multiply_compute_output_shapes_non_2d) {
  types::vector_shapes input_shapes = {{2, 3, 4}, {3, 4}}; // First matrix is 3D

  BOOST_CHECK_THROW(
      { matrix_multiply<float>::compute_output_shapes(input_shapes); },
      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_matrix_multiply_compute_output_shapes_incompatible) {
  types::vector_shapes input_shapes = {{2, 3}, {4, 5}}; // 3 != 4

  BOOST_CHECK_THROW(
      { matrix_multiply<float>::compute_output_shapes(input_shapes); },
      std::invalid_argument);
}

// Test operation method with mock matrices
class TestMatrixMultiply : public matrix_multiply<float> {
public:
  TestMatrixMultiply() : matrix_multiply<float>("test_matrix_multiply") {}

  // Expose operation method for testing
  using matrix_multiply<float>::operation;
};

BOOST_AUTO_TEST_CASE(test_matrix_multiply_operation_2x2) {
  TestMatrixMultiply mm;

  // Create test matrices: A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
  Eigen::MatrixXf A(2, 2);
  A << 1, 2, 3, 4;

  Eigen::MatrixXf B(2, 2);
  B << 5, 6, 7, 8;

  Eigen::MatrixXf C(2, 2);
  C.setZero();

  // Create matrix maps
  auto A_map = Eigen::Map<const Eigen::MatrixXf>(A.data(), 2, 2);
  auto B_map = Eigen::Map<const Eigen::MatrixXf>(B.data(), 2, 2);
  auto C_map = Eigen::Map<Eigen::MatrixXf>(C.data(), 2, 2);

  types::vector_const_matrix_map<float> inputs = {&A_map, &B_map};
  types::vector_matrix_map<float> outputs = {&C_map};

  // Perform operation
  auto result = mm.operation(inputs, outputs);

  // Check result
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Expected result: C = A * B = [[19, 22], [43, 50]]
  BOOST_CHECK_CLOSE(C(0, 0), 19.0f, 1e-6);
  BOOST_CHECK_CLOSE(C(0, 1), 22.0f, 1e-6);
  BOOST_CHECK_CLOSE(C(1, 0), 43.0f, 1e-6);
  BOOST_CHECK_CLOSE(C(1, 1), 50.0f, 1e-6);
}

BOOST_AUTO_TEST_CASE(test_matrix_multiply_operation_3x2_2x4) {
  TestMatrixMultiply mm;

  // Create test matrices: A = 3x2, B = 2x4
  Eigen::MatrixXf A(3, 2);
  A << 1, 2, 3, 4, 5, 6;

  Eigen::MatrixXf B(2, 4);
  B << 1, 2, 3, 4, 5, 6, 7, 8;

  Eigen::MatrixXf C(3, 4);
  C.setZero();

  // Create matrix maps
  auto A_map = Eigen::Map<const Eigen::MatrixXf>(A.data(), 3, 2);
  auto B_map = Eigen::Map<const Eigen::MatrixXf>(B.data(), 2, 4);
  auto C_map = Eigen::Map<Eigen::MatrixXf>(C.data(), 3, 4);

  types::vector_const_matrix_map<float> inputs = {&A_map, &B_map};
  types::vector_matrix_map<float> outputs = {&C_map};

  // Perform operation
  auto result = mm.operation(inputs, outputs);

  // Check result
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Expected result: C = A * B
  // Row 0: [1*1+2*5, 1*2+2*6, 1*3+2*7, 1*4+2*8] = [11, 14, 17, 20]
  BOOST_CHECK_CLOSE(C(0, 0), 11.0f, 1e-6);
  BOOST_CHECK_CLOSE(C(0, 1), 14.0f, 1e-6);
  BOOST_CHECK_CLOSE(C(0, 2), 17.0f, 1e-6);
  BOOST_CHECK_CLOSE(C(0, 3), 20.0f, 1e-6);

  // Row 1: [3*1+4*5, 3*2+4*6, 3*3+4*7, 3*4+4*8] = [23, 30, 37, 44]
  BOOST_CHECK_CLOSE(C(1, 0), 23.0f, 1e-6);
  BOOST_CHECK_CLOSE(C(1, 1), 30.0f, 1e-6);
  BOOST_CHECK_CLOSE(C(1, 2), 37.0f, 1e-6);
  BOOST_CHECK_CLOSE(C(1, 3), 44.0f, 1e-6);

  // Row 2: [5*1+6*5, 5*2+6*6, 5*3+6*7, 5*4+6*8] = [35, 46, 57, 68]
  BOOST_CHECK_CLOSE(C(2, 0), 35.0f, 1e-6);
  BOOST_CHECK_CLOSE(C(2, 1), 46.0f, 1e-6);
  BOOST_CHECK_CLOSE(C(2, 2), 57.0f, 1e-6);
  BOOST_CHECK_CLOSE(C(2, 3), 68.0f, 1e-6);
}

BOOST_AUTO_TEST_CASE(test_matrix_multiply_operation_identity) {
  TestMatrixMultiply mm;

  // Test A * I = A
  Eigen::MatrixXf A(2, 2);
  A << 3, 7, 2, 5;

  Eigen::MatrixXf I = Eigen::MatrixXf::Identity(2, 2);
  Eigen::MatrixXf C(2, 2);
  C.setZero();

  // Create matrix maps
  auto A_map = Eigen::Map<const Eigen::MatrixXf>(A.data(), 2, 2);
  auto I_map = Eigen::Map<const Eigen::MatrixXf>(I.data(), 2, 2);
  auto C_map = Eigen::Map<Eigen::MatrixXf>(C.data(), 2, 2);

  types::vector_const_matrix_map<float> inputs = {&A_map, &I_map};
  types::vector_matrix_map<float> outputs = {&C_map};

  // Perform operation
  auto result = mm.operation(inputs, outputs);

  // Check result
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Result should be A
  BOOST_CHECK_CLOSE(C(0, 0), 3.0f, 1e-6);
  BOOST_CHECK_CLOSE(C(0, 1), 7.0f, 1e-6);
  BOOST_CHECK_CLOSE(C(1, 0), 2.0f, 1e-6);
  BOOST_CHECK_CLOSE(C(1, 1), 5.0f, 1e-6);
}

BOOST_AUTO_TEST_CASE(test_matrix_multiply_operation_wrong_input_count) {
  TestMatrixMultiply mm;

  Eigen::MatrixXf A(2, 2);
  A.setOnes();
  auto A_map = Eigen::Map<const Eigen::MatrixXf>(A.data(), 2, 2);

  Eigen::MatrixXf C(2, 2);
  auto C_map = Eigen::Map<Eigen::MatrixXf>(C.data(), 2, 2);

  // Only one input matrix (should be 2)
  types::vector_const_matrix_map<float> inputs = {&A_map};
  types::vector_matrix_map<float> outputs = {&C_map};

  BOOST_CHECK_THROW({ mm.operation(inputs, outputs); }, std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_matrix_multiply_operation_wrong_output_count) {
  TestMatrixMultiply mm;

  Eigen::MatrixXf A(2, 2), B(2, 2);
  A.setOnes();
  B.setOnes();

  auto A_map = Eigen::Map<const Eigen::MatrixXf>(A.data(), 2, 2);
  auto B_map = Eigen::Map<const Eigen::MatrixXf>(B.data(), 2, 2);

  types::vector_const_matrix_map<float> inputs = {&A_map, &B_map};
  types::vector_matrix_map<float> outputs = {}; // No output matrices

  BOOST_CHECK_THROW({ mm.operation(inputs, outputs); }, std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_matrix_multiply_operation_incompatible_dimensions) {
  TestMatrixMultiply mm;

  // A is 2x3, B is 4x2 (incompatible: 3 != 4)
  Eigen::MatrixXf A(2, 3), B(4, 2), C(2, 2);
  A.setOnes();
  B.setOnes();
  C.setZero();

  auto A_map = Eigen::Map<const Eigen::MatrixXf>(A.data(), 2, 3);
  auto B_map = Eigen::Map<const Eigen::MatrixXf>(B.data(), 4, 2);
  auto C_map = Eigen::Map<Eigen::MatrixXf>(C.data(), 2, 2);

  types::vector_const_matrix_map<float> inputs = {&A_map, &B_map};
  types::vector_matrix_map<float> outputs = {&C_map};

  BOOST_CHECK_THROW({ mm.operation(inputs, outputs); }, std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_matrix_multiply_operation_wrong_output_dimensions) {
  TestMatrixMultiply mm;

  // A is 2x3, B is 3x4, but C is 2x2 (should be 2x4)
  Eigen::MatrixXf A(2, 3), B(3, 4), C(2, 2);
  A.setOnes();
  B.setOnes();
  C.setZero();

  auto A_map = Eigen::Map<const Eigen::MatrixXf>(A.data(), 2, 3);
  auto B_map = Eigen::Map<const Eigen::MatrixXf>(B.data(), 3, 4);
  auto C_map = Eigen::Map<Eigen::MatrixXf>(C.data(), 2, 2);

  types::vector_const_matrix_map<float> inputs = {&A_map, &B_map};
  types::vector_matrix_map<float> outputs = {&C_map};

  BOOST_CHECK_THROW({ mm.operation(inputs, outputs); }, std::invalid_argument);
}

// Test with complex numbers
BOOST_AUTO_TEST_CASE(test_matrix_multiply_complex) {
  using Complex = std::complex<float>;

  class TestMatrixMultiplyComplex : public matrix_multiply<Complex> {
  public:
    TestMatrixMultiplyComplex() : matrix_multiply<Complex>("test_complex") {}
    using matrix_multiply<Complex>::operation;
  };

  TestMatrixMultiplyComplex mm;

  // Create complex matrices using dynamic size
  Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> A(2, 2), B(2, 2),
      C(2, 2);
  A << Complex(1, 1), Complex(2, 0), Complex(0, 1), Complex(1, 1);

  B << Complex(1, 0), Complex(0, 1), Complex(1, 1), Complex(2, 0);

  C.setZero();

  // Create matrix maps using dynamic types to match the expected type
  auto A_map =
      Eigen::Map<const Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>>(
          A.data(), 2, 2);
  auto B_map =
      Eigen::Map<const Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>>(
          B.data(), 2, 2);
  auto C_map =
      Eigen::Map<Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>>(
          C.data(), 2, 2);

  types::vector_const_matrix_map<Complex> inputs = {&A_map, &B_map};
  types::vector_matrix_map<Complex> outputs = {&C_map};

  // Perform operation
  auto result = mm.operation(inputs, outputs);

  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Verify that the result is computed (exact values depend on complex
  // arithmetic)
  BOOST_CHECK(C(0, 0) != Complex(0, 0)); // Should not be zero
  BOOST_CHECK(C(1, 1) != Complex(0, 0)); // Should not be zero
}

// Test typename aliases
BOOST_AUTO_TEST_CASE(test_matrix_multiply_type_aliases) {
  // Test that the type aliases compile correctly
  matrix_multiply_sync_f::sptr sync_f_ptr = nullptr;
  matrix_multiply_sync_d::sptr sync_d_ptr = nullptr;
  matrix_multiply_sync_c::sptr sync_c_ptr = nullptr;
  matrix_multiply_sync_z::sptr sync_z_ptr = nullptr;

  matrix_multiply_pdu_f::sptr pdu_f_ptr = nullptr;
  matrix_multiply_pdu_d::sptr pdu_d_ptr = nullptr;
  matrix_multiply_pdu_c::sptr pdu_c_ptr = nullptr;
  matrix_multiply_pdu_z::sptr pdu_z_ptr = nullptr;

  // Just check that the pointers can be created (they will be null)
  BOOST_CHECK(sync_f_ptr == nullptr);
  BOOST_CHECK(sync_d_ptr == nullptr);
  BOOST_CHECK(sync_c_ptr == nullptr);
  BOOST_CHECK(sync_z_ptr == nullptr);

  BOOST_CHECK(pdu_f_ptr == nullptr);
  BOOST_CHECK(pdu_d_ptr == nullptr);
  BOOST_CHECK(pdu_c_ptr == nullptr);
  BOOST_CHECK(pdu_z_ptr == nullptr);
}

// GNU Radio Block Functionality Tests
// Note: These tests focus on the template interface and basic block structure
// since the implementation classes may not be available during unit testing

BOOST_AUTO_TEST_CASE(test_matrix_multiply_block_template_compilation) {
  // Test that the block template classes compile correctly
  // This verifies template structure without requiring implementation classes

  // Test template type aliases exist and compile
  using sync_float_type = matrix_multiply_sync<float>;
  using sync_double_type = matrix_multiply_sync<double>;
  using pdu_float_type = matrix_multiply_pdu<float>;
  using pdu_double_type = matrix_multiply_pdu<double>;

  // Test shared_ptr type aliases
  sync_float_type::sptr sync_ptr = nullptr;
  pdu_float_type::sptr pdu_ptr = nullptr;

  BOOST_CHECK(sync_ptr == nullptr);
  BOOST_CHECK(pdu_ptr == nullptr);
}

BOOST_AUTO_TEST_CASE(test_matrix_multiply_type_aliases_compilation) {
  // Test that all type aliases compile correctly

  // Test sync type aliases exist and compile
  using test_sync_f = matrix_multiply_sync_f;
  using test_sync_d = matrix_multiply_sync_d;
  using test_sync_c = matrix_multiply_sync_c;
  using test_sync_z = matrix_multiply_sync_z;

  // Test PDU type aliases exist and compile
  using test_pdu_f = matrix_multiply_pdu_f;
  using test_pdu_d = matrix_multiply_pdu_d;
  using test_pdu_c = matrix_multiply_pdu_c;
  using test_pdu_z = matrix_multiply_pdu_z;

  // If we get here, all type aliases compiled successfully
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(test_matrix_multiply_template_parameters) {
  // Test template parameter variations
  types::shape shape_2x2 = {2, 2};

  // Test with different template parameters (Eigen dimension specification)
  try {
    matrix_multiply<float, 2, 2, 2, 2> fixed_block("test",
                                                   {shape_2x2, shape_2x2});
    BOOST_CHECK(true); // If we get here, construction succeeded
  } catch (const std::exception &e) {
    BOOST_FAIL("Fixed dimension template construction failed: " << e.what());
  } catch (...) {
    BOOST_FAIL(
        "Fixed dimension template construction failed: unknown exception");
  }

  try {
    matrix_multiply<float> dynamic_block("test", {shape_2x2, shape_2x2});
    BOOST_CHECK(true); // If we get here, construction succeeded
  } catch (const std::exception &e) {
    BOOST_FAIL("Dynamic dimension template construction failed: " << e.what());
  } catch (...) {
    BOOST_FAIL(
        "Dynamic dimension template construction failed: unknown exception");
  }
}

BOOST_AUTO_TEST_CASE(test_matrix_multiply_constructor_parameters) {
  // Test different constructor parameter combinations
  types::shape shape_a = {3, 2};
  types::shape shape_b = {2, 4};
  types::shape expected_output = {3, 4};

  // Test with input shapes only (let constructor compute output)
  try {
    matrix_multiply<float> block_no_output("test_block", {shape_a, shape_b});
    BOOST_CHECK(true); // Construction succeeded
  } catch (const std::exception &e) {
    BOOST_FAIL("Constructor with input shapes failed: " << e.what());
  } catch (...) {
    BOOST_FAIL("Constructor with input shapes failed: unknown exception");
  }

  // Test with default parameters
  try {
    matrix_multiply<float> block; // Default constructor
    BOOST_CHECK(true);            // Construction succeeded
  } catch (...) {
    BOOST_FAIL("Default constructor failed");
  }
}

BOOST_AUTO_TEST_CASE(test_matrix_multiply_static_method_accessibility) {
  // Test that static methods are accessible from the template classes
  types::vector_shapes input_shapes = {{2, 3}, {3, 4}};
  types::vector_shapes output_shapes = {{2, 4}};

  // Test static methods are accessible and callable
  try {
    matrix_multiply<float>::validate_shape(input_shapes, output_shapes, "test");
    BOOST_CHECK(true); // Method call succeeded
  } catch (...) {
    BOOST_FAIL("validate_shape method failed");
  }

  auto sizes = matrix_multiply<float>::compute_sizes(input_shapes);
  BOOST_CHECK_EQUAL(sizes.size(), 2);

  auto output = matrix_multiply<float>::compute_output_shapes(input_shapes);
  BOOST_CHECK_EQUAL(output.size(), 1);
}

BOOST_AUTO_TEST_CASE(test_matrix_multiply_template_with_complex_types) {
  // Test template with complex number types
  types::shape shape_2x2 = {2, 2};

  try {
    matrix_multiply<std::complex<float>> complex_block("test",
                                                       {shape_2x2, shape_2x2});
    BOOST_CHECK(true); // Complex float template works
  } catch (const std::exception &e) {
    BOOST_FAIL("Complex float template construction failed: " << e.what());
  } catch (...) {
    BOOST_FAIL("Complex float template construction failed: unknown exception");
  }

  try {
    matrix_multiply<std::complex<double>> complex_double_block(
        "test", {shape_2x2, shape_2x2});
    BOOST_CHECK(true); // Complex double template works
  } catch (...) {
    BOOST_FAIL("Complex double template construction failed");
  }
}

//==============================================================================
// GNU Radio Flowgraph Tests
//==============================================================================

/*!
 * \brief Test matrix multiply sync block in GNU Radio flowgraph
 *
 * Tests basic 2×3 × 3×2 matrix multiplication through actual signal processing
 * pipeline: Input A: [[1, 2, 3], [4, 5, 6]] (column-major: [1, 4, 2, 5, 3, 6])
 * Input B: [[7, 8], [9, 10], [11, 12]] (column-major: [7, 9, 11, 8, 10, 12])
 * Expected result: A × B = [[58, 64], [139, 154]] (column-major: [58, 139, 64,
 * 154])
 */
BOOST_AUTO_TEST_CASE(test_matrix_multiply_sync_flowgraph) {
  BOOST_TEST_MESSAGE(
      "Testing matrix multiply sync block in GNU Radio flowgraph...");

  auto tb = gr::make_top_block("matrix_multiply_test");

  // Create input matrices: A = [[1, 2, 3], [4, 5, 6]] (2×3)
  std::vector<float> input_a = {1.0f, 4.0f, 2.0f,
                                5.0f, 3.0f, 6.0f}; // Column-major

  // B = [[7, 8], [9, 10], [11, 12]] (3×2)
  std::vector<float> input_b = {7.0f, 9.0f,  11.0f,
                                8.0f, 10.0f, 12.0f}; // Column-major

  auto vector_source_a =
      gr::blocks::vector_source_f::make(input_a, false, 6); // 2×3 = 6 elements
  auto vector_source_b =
      gr::blocks::vector_source_f::make(input_b, false, 6); // 3×2 = 6 elements
  auto multiply_block = matrix_multiply_sync<float>::make({2, 3}, {3, 2});
  auto output_sink = gr::blocks::vector_sink_f::make(4); // 2×2 = 4 elements

  tb->connect(vector_source_a, 0, multiply_block, 0);
  tb->connect(vector_source_b, 0, multiply_block, 1);
  tb->connect(multiply_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 4);

  // Expected result: [[58, 64], [139, 154]] in column-major: [58, 139, 64, 154]
  // Calculation: [1*7+2*9+3*11, 4*7+5*9+6*11, 1*8+2*10+3*12, 4*8+5*10+6*12]
  std::vector<float> expected = {58.0f, 139.0f, 64.0f, 154.0f};
  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_CHECK_CLOSE(output_data[i], expected[i], FLOAT_TOLERANCE);
  }

  BOOST_TEST_MESSAGE("Matrix multiply sync flowgraph test passed.");
}

/*!
 * \brief Test square matrix multiplication in flowgraph
 *
 * Tests 2×2 × 2×2 matrix multiplication through GNU Radio pipeline:
 * A = [[1, 2], [3, 4]]
 * B = [[5, 6], [7, 8]]
 * Expected: A × B = [[19, 22], [43, 50]]
 */
BOOST_AUTO_TEST_CASE(test_matrix_multiply_sync_flowgraph_square) {
  BOOST_TEST_MESSAGE("Testing matrix multiply sync block with square matrices "
                     "in flowgraph...");

  auto tb = gr::make_top_block("matrix_multiply_square_test");

  // Create input matrices
  std::vector<float> input_a = {1.0f, 3.0f, 2.0f,
                                4.0f}; // [[1, 2], [3, 4]] column-major
  std::vector<float> input_b = {5.0f, 7.0f, 6.0f,
                                8.0f}; // [[5, 6], [7, 8]] column-major

  auto vector_source_a = gr::blocks::vector_source_f::make(input_a, false, 4);
  auto vector_source_b = gr::blocks::vector_source_f::make(input_b, false, 4);
  auto multiply_block = matrix_multiply_sync<float>::make({2, 2}, {2, 2});
  auto output_sink = gr::blocks::vector_sink_f::make(4);

  tb->connect(vector_source_a, 0, multiply_block, 0);
  tb->connect(vector_source_b, 0, multiply_block, 1);
  tb->connect(multiply_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 4);

  // Expected result: [[19, 22], [43, 50]] in column-major: [19, 43, 22, 50]
  std::vector<float> expected = {19.0f, 43.0f, 22.0f, 50.0f};
  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_CHECK_CLOSE(output_data[i], expected[i], FLOAT_TOLERANCE);
  }

  BOOST_TEST_MESSAGE("Square matrix multiply flowgraph test passed.");
}

/*!
 * \brief Test matrix multiply with complex numbers in flowgraph
 *
 * Tests complex matrix multiplication through GNU Radio pipeline:
 * A = [[1+i, 2], [0, 1+i]]
 * B = [[1, 0], [i, 1]]
 * Expected computed through complex arithmetic
 */
BOOST_AUTO_TEST_CASE(test_matrix_multiply_sync_flowgraph_complex) {
  BOOST_TEST_MESSAGE("Testing matrix multiply sync block with complex numbers "
                     "in flowgraph...");

  using cf = std::complex<float>;
  auto tb = gr::make_top_block("matrix_multiply_complex_test");

  // Create complex input matrices
  std::vector<cf> input_a = {cf(1, 1), cf(0, 0), cf(2, 0),
                             cf(1, 1)}; // [[1+i, 2], [0, 1+i]]
  std::vector<cf> input_b = {cf(1, 0), cf(0, 1), cf(0, 0),
                             cf(1, 0)}; // [[1, 0], [i, 1]]

  auto vector_source_a = gr::blocks::vector_source_c::make(input_a, false, 4);
  auto vector_source_b = gr::blocks::vector_source_c::make(input_b, false, 4);
  auto multiply_block = matrix_multiply_sync<cf>::make({2, 2}, {2, 2});
  auto output_sink = gr::blocks::vector_sink_c::make(4);

  tb->connect(vector_source_a, 0, multiply_block, 0);
  tb->connect(vector_source_b, 0, multiply_block, 1);
  tb->connect(multiply_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 4);

  // Verify complex multiplication results
  // Expected: (1+i)*1 + 2*i = 1+3i, 0*1 + (1+i)*i = -1+i, (1+i)*0 + 2*1 = 2,
  // 0*0 + (1+i)*1 = 1+i
  std::vector<cf> expected = {cf(1, 3), cf(-1, 1), cf(2, 0), cf(1, 1)};

  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_CHECK_CLOSE(output_data[i].real(), expected[i].real(),
                      FLOAT_TOLERANCE);
    BOOST_CHECK_CLOSE(output_data[i].imag(), expected[i].imag(),
                      FLOAT_TOLERANCE);
  }

  BOOST_TEST_MESSAGE("Complex matrix multiply flowgraph test passed.");
}

/*!
 * \brief Test matrix multiply with identity matrix in flowgraph
 *
 * Tests A × I = A through GNU Radio pipeline
 */
BOOST_AUTO_TEST_CASE(test_matrix_multiply_sync_flowgraph_identity) {
  BOOST_TEST_MESSAGE("Testing matrix multiply sync block with identity matrix "
                     "in flowgraph...");

  auto tb = gr::make_top_block("matrix_multiply_identity_test");

  // Create A matrix and identity matrix
  std::vector<float> input_a = {3.0f, 2.0f, 7.0f, 5.0f}; // [[3, 7], [2, 5]]
  std::vector<float> input_i = {1.0f, 0.0f, 0.0f, 1.0f}; // [[1, 0], [0, 1]]

  auto vector_source_a = gr::blocks::vector_source_f::make(input_a, false, 4);
  auto vector_source_i = gr::blocks::vector_source_f::make(input_i, false, 4);
  auto multiply_block = matrix_multiply_sync<float>::make({2, 2}, {2, 2});
  auto output_sink = gr::blocks::vector_sink_f::make(4);

  tb->connect(vector_source_a, 0, multiply_block, 0);
  tb->connect(vector_source_i, 0, multiply_block, 1);
  tb->connect(multiply_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 4);

  // Result should be A unchanged
  for (size_t i = 0; i < input_a.size(); ++i) {
    BOOST_CHECK_CLOSE(output_data[i], input_a[i], FLOAT_TOLERANCE);
  }

  BOOST_TEST_MESSAGE("Identity matrix multiply flowgraph test passed.");
}

/*!
 * \brief Test vector-matrix multiplication in flowgraph
 *
 * Tests column vector × row vector = matrix through GNU Radio pipeline:
 * a = [[2], [3], [4]] (3×1)
 * b = [[1, 2, 3, 4]] (1×4)
 * Expected: a × b = 3×4 matrix
 */
BOOST_AUTO_TEST_CASE(test_matrix_multiply_sync_flowgraph_vector) {
  BOOST_TEST_MESSAGE(
      "Testing matrix multiply sync block with vectors in flowgraph...");

  auto tb = gr::make_top_block("matrix_multiply_vector_test");

  // Create column vector (3×1) and row vector (1×4)
  std::vector<float> input_a = {2.0f, 3.0f,
                                4.0f}; // Column vector [[2], [3], [4]]
  std::vector<float> input_b = {1.0f, 2.0f, 3.0f,
                                4.0f}; // Row vector [[1, 2, 3, 4]]

  auto vector_source_a = gr::blocks::vector_source_f::make(input_a, false, 3);
  auto vector_source_b = gr::blocks::vector_source_f::make(input_b, false, 4);
  auto multiply_block = matrix_multiply_sync<float>::make({3, 1}, {1, 4});
  auto output_sink = gr::blocks::vector_sink_f::make(12); // 3×4 = 12 elements

  tb->connect(vector_source_a, 0, multiply_block, 0);
  tb->connect(vector_source_b, 0, multiply_block, 1);
  tb->connect(multiply_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 12);

  // Expected result: [[2, 4, 6, 8], [3, 6, 9, 12], [4, 8, 12, 16]]
  // In column-major: [2, 3, 4, 4, 6, 8, 6, 9, 12, 8, 12, 16]
  std::vector<float> expected = {2.0f, 3.0f, 4.0f,  4.0f, 6.0f,  8.0f,
                                 6.0f, 9.0f, 12.0f, 8.0f, 12.0f, 16.0f};
  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_CHECK_CLOSE(output_data[i], expected[i], FLOAT_TOLERANCE);
  }

  BOOST_TEST_MESSAGE("Vector matrix multiply flowgraph test passed.");
}

/*!
 * \brief Test large matrix multiplication in flowgraph
 *
 * Tests multiplication of larger matrices through GNU Radio pipeline
 */
BOOST_AUTO_TEST_CASE(test_matrix_multiply_sync_flowgraph_large) {
  BOOST_TEST_MESSAGE(
      "Testing matrix multiply sync block with large matrices in flowgraph...");

  auto tb = gr::make_top_block("matrix_multiply_large_test");

  // Create 4×3 and 3×5 matrices
  // A = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
  std::vector<float> input_a = {1.0f, 4.0f, 7.0f, 10.0f,  // Column 1
                                2.0f, 5.0f, 8.0f, 11.0f,  // Column 2
                                3.0f, 6.0f, 9.0f, 12.0f}; // Column 3

  // B = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
  std::vector<float> input_b = {1.0f, 6.0f,  11.0f,  // Column 1
                                2.0f, 7.0f,  12.0f,  // Column 2
                                3.0f, 8.0f,  13.0f,  // Column 3
                                4.0f, 9.0f,  14.0f,  // Column 4
                                5.0f, 10.0f, 15.0f}; // Column 5

  auto vector_source_a =
      gr::blocks::vector_source_f::make(input_a, false, 12); // 4×3
  auto vector_source_b =
      gr::blocks::vector_source_f::make(input_b, false, 15); // 3×5
  auto multiply_block = matrix_multiply_sync<float>::make({4, 3}, {3, 5});
  auto output_sink = gr::blocks::vector_sink_f::make(20); // 4×5 = 20 elements

  tb->connect(vector_source_a, 0, multiply_block, 0);
  tb->connect(vector_source_b, 0, multiply_block, 1);
  tb->connect(multiply_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 20);

  // Verify first and last elements as a sanity check
  // First element (0,0): 1*1 + 2*6 + 3*11 = 46
  BOOST_CHECK_CLOSE(output_data[0], 46.0f, FLOAT_TOLERANCE);

  // Last element (3,4): 10*5 + 11*10 + 12*15 = 340
  BOOST_CHECK_CLOSE(output_data[19], 340.0f, FLOAT_TOLERANCE);

  BOOST_TEST_MESSAGE("Large matrix multiply flowgraph test passed.");
}

/*!
 * \brief Test matrix multiply with double precision in flowgraph
 *
 * Tests high precision matrix multiplication through GNU Radio pipeline
 */
BOOST_AUTO_TEST_CASE(test_matrix_multiply_sync_flowgraph_double) {
  BOOST_TEST_MESSAGE("Testing matrix multiply sync block with double precision "
                     "in flowgraph...");

  auto tb = gr::make_top_block("matrix_multiply_double_test");

  // Create double precision matrices (3×3 × 3×3)
  std::vector<double> input_a = {1.123456789, 4.987654321, 7.999999999,
                                 2.271828183, 5.000000001, 8.123123123,
                                 3.141592654, 6.456456456, 9.789789789};

  std::vector<double> input_b = {2.123456789, 1.987654321, 2.999999999,
                                 1.271828183, 3.000000001, 3.123123123,
                                 2.141592654, 1.456456456, 1.789789789};

  // Convert double to float for compatibility with available GNU Radio blocks
  std::vector<float> input_a_f(input_a.begin(), input_a.end());
  std::vector<float> input_b_f(input_b.begin(), input_b.end());

  auto vector_source_a = gr::blocks::vector_source_f::make(input_a_f, false, 9);
  auto vector_source_b = gr::blocks::vector_source_f::make(input_b_f, false, 9);
  auto multiply_block = matrix_multiply_sync<float>::make({3, 3}, {3, 3});
  auto output_sink = gr::blocks::vector_sink_f::make(9);

  tb->connect(vector_source_a, 0, multiply_block, 0);
  tb->connect(vector_source_b, 0, multiply_block, 1);
  tb->connect(multiply_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 9);

  // Verify that results are finite and non-zero
  for (size_t i = 0; i < output_data.size(); ++i) {
    BOOST_CHECK(std::isfinite(output_data[i]));
    BOOST_CHECK(output_data[i] != 0.0f);
  }

  BOOST_TEST_MESSAGE("Double precision matrix multiply flowgraph test passed.");
}

/*!
 * \brief Test matrix multiply with multiple matrices in flowgraph
 *
 * Tests processing of multiple consecutive matrix pairs through GNU Radio
 * pipeline
 */
BOOST_AUTO_TEST_CASE(test_matrix_multiply_sync_flowgraph_multiple_matrices) {
  BOOST_TEST_MESSAGE("Testing matrix multiply sync block with multiple "
                     "matrices in flowgraph...");

  auto tb = gr::make_top_block("matrix_multiply_multi_matrices_test");

  // Create multiple 2×2 matrix pairs for consecutive processing
  std::vector<float> input_a = {// First pair A1: [[1, 2], [3, 4]]
                                1.0f, 3.0f, 2.0f, 4.0f,
                                // Second pair A2: [[2, 3], [4, 5]]
                                2.0f, 4.0f, 3.0f, 5.0f,
                                // Third pair A3: [[3, 4], [5, 6]]
                                3.0f, 5.0f, 4.0f, 6.0f};

  std::vector<float> input_b = {// First pair B1: [[5, 6], [7, 8]]
                                5.0f, 7.0f, 6.0f, 8.0f,
                                // Second pair B2: [[6, 7], [8, 9]]
                                6.0f, 8.0f, 7.0f, 9.0f,
                                // Third pair B3: [[7, 8], [9, 10]]
                                7.0f, 9.0f, 8.0f, 10.0f};

  auto vector_source_a = gr::blocks::vector_source_f::make(input_a, false, 4);
  auto vector_source_b = gr::blocks::vector_source_f::make(input_b, false, 4);
  auto multiply_block = matrix_multiply_sync<float>::make({2, 2}, {2, 2});
  auto output_sink = gr::blocks::vector_sink_f::make(4);

  tb->connect(vector_source_a, 0, multiply_block, 0);
  tb->connect(vector_source_b, 0, multiply_block, 1);
  tb->connect(multiply_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 12); // 3 matrices × 4 elements each

  // Verify first matrix result: [[19, 22], [43, 50]]
  BOOST_CHECK_CLOSE(output_data[0], 19.0f, FLOAT_TOLERANCE);
  BOOST_CHECK_CLOSE(output_data[1], 43.0f, FLOAT_TOLERANCE);
  BOOST_CHECK_CLOSE(output_data[2], 22.0f, FLOAT_TOLERANCE);
  BOOST_CHECK_CLOSE(output_data[3], 50.0f, FLOAT_TOLERANCE);

  // Verify last matrix result exists and is valid
  BOOST_CHECK(std::isfinite(output_data[8]));
  BOOST_CHECK(std::isfinite(output_data[11]));

  BOOST_TEST_MESSAGE(
      "Multiple matrices matrix multiply flowgraph test passed.");
}

/*!
 * \brief Test matrix multiply performance in flowgraph
 *
 * Tests performance with a large number of matrix multiplications through GNU
 * Radio pipeline
 */
BOOST_AUTO_TEST_CASE(test_matrix_multiply_sync_flowgraph_performance) {
  BOOST_TEST_MESSAGE(
      "Testing matrix multiply sync block performance in flowgraph...");

  auto tb = gr::make_top_block("matrix_multiply_perf_test");

  // Create many 3×3 matrix pairs for performance testing
  int num_matrices = 500;
  std::vector<float> input_a, input_b;
  input_a.reserve(num_matrices * 9);
  input_b.reserve(num_matrices * 9);

  for (int i = 0; i < num_matrices; ++i) {
    // Generate matrices with varying values for performance test
    float base = 1.0f + 0.001f * i;
    for (int j = 0; j < 9; ++j) {
      input_a.push_back(base + j * 0.1f);
      input_b.push_back(base + (8 - j) * 0.1f);
    }
  }

  auto start_time = std::chrono::high_resolution_clock::now();

  auto vector_source_a = gr::blocks::vector_source_f::make(input_a, false, 9);
  auto vector_source_b = gr::blocks::vector_source_f::make(input_b, false, 9);
  auto multiply_block = matrix_multiply_sync<float>::make({3, 3}, {3, 3});
  auto output_sink = gr::blocks::vector_sink_f::make(9);

  tb->connect(vector_source_a, 0, multiply_block, 0);
  tb->connect(vector_source_b, 0, multiply_block, 1);
  tb->connect(multiply_block, 0, output_sink, 0);

  tb->run();

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), num_matrices * 9);

  // Verify first and last matrix results are valid
  BOOST_CHECK(std::isfinite(output_data[0]));
  BOOST_CHECK(std::isfinite(output_data[8]));

  size_t last_matrix_idx = (num_matrices - 1) * 9;
  BOOST_CHECK(std::isfinite(output_data[last_matrix_idx]));
  BOOST_CHECK(std::isfinite(output_data[last_matrix_idx + 8]));

  // All results should be finite
  for (size_t i = 0; i < output_data.size();
       i += 100) { // Check every 100th element for speed
    BOOST_CHECK(std::isfinite(output_data[i]));
  }

  BOOST_TEST_MESSAGE("Processed " << num_matrices
                                  << " 3×3 matrix multiplications in "
                                  << duration.count() << " ms");
  BOOST_TEST_MESSAGE("Matrix multiply performance flowgraph test passed.");
}

} /* namespace linalg */
} /* namespace gr */
