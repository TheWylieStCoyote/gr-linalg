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
#include <gnuradio/linalg/matrix_covariance.h>
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
BOOST_AUTO_TEST_CASE(test_matrix_covariance_validate_shape_valid) {
  types::vector_shapes input_shapes = {{3, 2}};  // 3 samples, 2 features
  types::vector_shapes output_shapes = {{2, 2}}; // Expected output shape

  BOOST_CHECK_NO_THROW({
    matrix_covariance<float>::validate_shape(input_shapes, output_shapes,
                                             "test");
  });
}

BOOST_AUTO_TEST_CASE(test_matrix_covariance_validate_shape_wrong_input_count) {
  types::vector_shapes input_shapes = {{2, 3}, {3, 2}}; // Two matrices
  types::vector_shapes output_shapes = {{2, 2}};

  BOOST_CHECK_THROW(
      {
        matrix_covariance<float>::validate_shape(input_shapes, output_shapes,
                                                 "test");
      },
      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(
    test_matrix_covariance_validate_shape_insufficient_samples) {
  types::vector_shapes input_shapes = {{1, 3}}; // Only 1 sample
  types::vector_shapes output_shapes = {{3, 3}};

  BOOST_CHECK_THROW(
      {
        matrix_covariance<float>::validate_shape(input_shapes, output_shapes,
                                                 "test");
      },
      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_matrix_covariance_validate_shape_non_2d_matrix) {
  types::vector_shapes input_shapes = {{2, 3, 4}}; // 3D matrix
  types::vector_shapes output_shapes = {{3, 3}};

  BOOST_CHECK_THROW(
      {
        matrix_covariance<float>::validate_shape(input_shapes, output_shapes,
                                                 "test");
      },
      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_matrix_covariance_validate_shape_wrong_output_shape) {
  types::vector_shapes input_shapes = {{4, 3}}; // 4 samples, 3 features
  types::vector_shapes output_shapes = {
      {2, 2}}; // Wrong output shape (should be {3, 3})

  BOOST_CHECK_THROW(
      {
        matrix_covariance<float>::validate_shape(input_shapes, output_shapes,
                                                 "test");
      },
      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_matrix_covariance_validate_shape_empty_output) {
  types::vector_shapes input_shapes = {{4, 3}};
  types::vector_shapes output_shapes =
      {}; // Empty output shapes (should be allowed)

  BOOST_CHECK_NO_THROW({
    matrix_covariance<float>::validate_shape(input_shapes, output_shapes,
                                             "test");
  });
}

// Test static method: compute_sizes
BOOST_AUTO_TEST_CASE(test_matrix_covariance_compute_sizes) {
  types::vector_shapes shapes = {{3, 2}, {2, 2}};
  std::vector<size_t> expected = {6, 4}; // 3*2=6, 2*2=4

  auto result = matrix_covariance<float>::compute_sizes(shapes);

  BOOST_CHECK_EQUAL(result.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_CHECK_EQUAL(result[i], expected[i] * sizeof(float));
  }
}

// Test static method: compute_output_shapes
BOOST_AUTO_TEST_CASE(test_matrix_covariance_compute_output_shapes_valid) {
  types::vector_shapes input_shapes = {{4, 3}}; // 4 samples, 3 features
  types::vector_shapes expected = {{3, 3}};     // Output: [features, features]

  auto result = matrix_covariance<float>::compute_output_shapes(input_shapes);

  BOOST_CHECK_EQUAL(result.size(), 1);
  BOOST_CHECK_EQUAL(result[0].size(), 2);
  BOOST_CHECK_EQUAL(result[0][0], 3);
  BOOST_CHECK_EQUAL(result[0][1], 3);
}

BOOST_AUTO_TEST_CASE(
    test_matrix_covariance_compute_output_shapes_wrong_input_count) {
  types::vector_shapes input_shapes = {{2, 3}, {3, 2}}; // Two matrices

  BOOST_CHECK_THROW(
      { matrix_covariance<float>::compute_output_shapes(input_shapes); },
      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_matrix_covariance_compute_output_shapes_non_2d) {
  types::vector_shapes input_shapes = {{2, 3, 4}}; // 3D matrix

  BOOST_CHECK_THROW(
      { matrix_covariance<float>::compute_output_shapes(input_shapes); },
      std::invalid_argument);
}

// Test operation method with mock matrices
class TestMatrixCovariance : public matrix_covariance<float> {
public:
  TestMatrixCovariance(bool unbiased = true, bool center = true)
      : matrix_covariance<float>("test_matrix_covariance", {}, {}, unbiased,
                                 center) {}

  // Expose operation method for testing
  using matrix_covariance<float>::operation;
  using matrix_covariance<float>::set_unbiased;
  using matrix_covariance<float>::set_center;
  using matrix_covariance<float>::get_unbiased;
  using matrix_covariance<float>::get_center;
};

BOOST_AUTO_TEST_CASE(test_matrix_covariance_operation_2x2_centered_unbiased) {
  TestMatrixCovariance cov;

  // Create test data: 3 samples, 2 features
  // Data = [[1, 2], [3, 4], [5, 6]]
  Eigen::MatrixXf data(3, 2);
  data << 1, 2, 3, 4, 5, 6;

  Eigen::MatrixXf output(2, 2);
  output.setZero();

  // Create matrix maps
  auto data_map = Eigen::Map<const Eigen::MatrixXf>(data.data(), 3, 2);
  auto output_map = Eigen::Map<Eigen::MatrixXf>(output.data(), 2, 2);

  types::vector_const_matrix_map<float> inputs = {&data_map};
  types::vector_matrix_map<float> outputs = {&output_map};

  // Perform operation
  auto result = cov.operation(inputs, outputs);

  // Check result
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Expected unbiased covariance:
  // Mean = [3, 4], centered data = [[-2, -2], [0, 0], [2, 2]]
  // Cov = (centered^T * centered) / (N-1) = [[8, 8], [8, 8]] / 2 = [[4, 4], [4,
  // 4]]
  BOOST_CHECK_CLOSE(output(0, 0), 4.0f, 1e-6);
  BOOST_CHECK_CLOSE(output(0, 1), 4.0f, 1e-6);
  BOOST_CHECK_CLOSE(output(1, 0), 4.0f, 1e-6);
  BOOST_CHECK_CLOSE(output(1, 1), 4.0f, 1e-6);
}

BOOST_AUTO_TEST_CASE(test_matrix_covariance_operation_biased) {
  TestMatrixCovariance cov(false, true); // biased, centered

  // Create test data: 3 samples, 2 features
  Eigen::MatrixXf data(3, 2);
  data << 1, 2, 3, 4, 5, 6;

  Eigen::MatrixXf output(2, 2);
  output.setZero();

  auto data_map = Eigen::Map<const Eigen::MatrixXf>(data.data(), 3, 2);
  auto output_map = Eigen::Map<Eigen::MatrixXf>(output.data(), 2, 2);

  types::vector_const_matrix_map<float> inputs = {&data_map};
  types::vector_matrix_map<float> outputs = {&output_map};

  auto result = cov.operation(inputs, outputs);

  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Expected biased covariance: same as above but divided by N instead of N-1
  // Cov = [[8, 8], [8, 8]] / 3 = [[2.667, 2.667], [2.667, 2.667]]
  BOOST_CHECK_CLOSE(output(0, 0), 8.0f / 3.0f, 1e-5);
  BOOST_CHECK_CLOSE(output(0, 1), 8.0f / 3.0f, 1e-5);
  BOOST_CHECK_CLOSE(output(1, 0), 8.0f / 3.0f, 1e-5);
  BOOST_CHECK_CLOSE(output(1, 1), 8.0f / 3.0f, 1e-5);
}

BOOST_AUTO_TEST_CASE(test_matrix_covariance_operation_not_centered) {
  TestMatrixCovariance cov(true, false); // unbiased, not centered

  // Create test data: 3 samples, 2 features
  Eigen::MatrixXf data(3, 2);
  data << 1, 2, 3, 4, 5, 6;

  Eigen::MatrixXf output(2, 2);
  output.setZero();

  auto data_map = Eigen::Map<const Eigen::MatrixXf>(data.data(), 3, 2);
  auto output_map = Eigen::Map<Eigen::MatrixXf>(output.data(), 2, 2);

  types::vector_const_matrix_map<float> inputs = {&data_map};
  types::vector_matrix_map<float> outputs = {&output_map};

  auto result = cov.operation(inputs, outputs);

  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Without centering: Cov = (data^T * data) / (N-1)
  // data^T * data = [[35, 44], [44, 56]]
  // Cov = [[35, 44], [44, 56]] / 2 = [[17.5, 22], [22, 28]]
  BOOST_CHECK_CLOSE(output(0, 0), 17.5f, 1e-5);
  BOOST_CHECK_CLOSE(output(0, 1), 22.0f, 1e-5);
  BOOST_CHECK_CLOSE(output(1, 0), 22.0f, 1e-5);
  BOOST_CHECK_CLOSE(output(1, 1), 28.0f, 1e-5);
}

BOOST_AUTO_TEST_CASE(test_matrix_covariance_operation_wrong_input_count) {
  TestMatrixCovariance cov;

  Eigen::MatrixXf data1(3, 2), data2(3, 2);
  data1.setOnes();
  data2.setOnes();
  auto data1_map = Eigen::Map<const Eigen::MatrixXf>(data1.data(), 3, 2);
  auto data2_map = Eigen::Map<const Eigen::MatrixXf>(data2.data(), 3, 2);

  Eigen::MatrixXf output(2, 2);
  auto output_map = Eigen::Map<Eigen::MatrixXf>(output.data(), 2, 2);

  // Two input matrices (should be 1)
  types::vector_const_matrix_map<float> inputs = {&data1_map, &data2_map};
  types::vector_matrix_map<float> outputs = {&output_map};

  BOOST_CHECK_EQUAL(cov.operation(inputs, outputs),
                    OperationReturn::ERROR_INVALID_INPUT);
}

BOOST_AUTO_TEST_CASE(test_matrix_covariance_operation_wrong_output_count) {
  TestMatrixCovariance cov;

  Eigen::MatrixXf data(3, 2);
  data.setOnes();
  auto data_map = Eigen::Map<const Eigen::MatrixXf>(data.data(), 3, 2);

  types::vector_const_matrix_map<float> inputs = {&data_map};
  types::vector_matrix_map<float> outputs = {}; // No output matrices

  BOOST_CHECK_EQUAL(cov.operation(inputs, outputs),
                    OperationReturn::ERROR_INVALID_OUTPUT);
}

BOOST_AUTO_TEST_CASE(test_matrix_covariance_operation_insufficient_samples) {
  TestMatrixCovariance cov;

  // Only 1 sample
  Eigen::MatrixXf data(1, 2);
  data.setOnes();
  auto data_map = Eigen::Map<const Eigen::MatrixXf>(data.data(), 1, 2);

  Eigen::MatrixXf output(2, 2);
  auto output_map = Eigen::Map<Eigen::MatrixXf>(output.data(), 2, 2);

  types::vector_const_matrix_map<float> inputs = {&data_map};
  types::vector_matrix_map<float> outputs = {&output_map};

  BOOST_CHECK_EQUAL(cov.operation(inputs, outputs),
                    OperationReturn::ERROR_INVALID_INPUT);
}

BOOST_AUTO_TEST_CASE(test_matrix_covariance_operation_wrong_output_dimensions) {
  TestMatrixCovariance cov;

  // 3 samples, 2 features
  Eigen::MatrixXf data(3, 2);
  data.setOnes();
  auto data_map = Eigen::Map<const Eigen::MatrixXf>(data.data(), 3, 2);

  // Wrong output size (should be 2x2)
  Eigen::MatrixXf output(3, 3);
  auto output_map = Eigen::Map<Eigen::MatrixXf>(output.data(), 3, 3);

  types::vector_const_matrix_map<float> inputs = {&data_map};
  types::vector_matrix_map<float> outputs = {&output_map};

  BOOST_CHECK_EQUAL(cov.operation(inputs, outputs),
                    OperationReturn::ERROR_INVALID_OUTPUT);
}

// Test getter/setter methods
BOOST_AUTO_TEST_CASE(test_matrix_covariance_setters_getters) {
  TestMatrixCovariance cov;

  // Test unbiased flag
  BOOST_CHECK_EQUAL(cov.get_unbiased(), true); // Default
  cov.set_unbiased(false);
  BOOST_CHECK_EQUAL(cov.get_unbiased(), false);

  // Test center flag
  BOOST_CHECK_EQUAL(cov.get_center(), true); // Default
  cov.set_center(false);
  BOOST_CHECK_EQUAL(cov.get_center(), false);
}

// Test with complex numbers
BOOST_AUTO_TEST_CASE(test_matrix_covariance_complex) {
  using Complex = std::complex<float>;

  class TestMatrixCovarianceComplex : public matrix_covariance<Complex> {
  public:
    TestMatrixCovarianceComplex()
        : matrix_covariance<Complex>("test_complex") {}
    using matrix_covariance<Complex>::operation;
  };

  TestMatrixCovarianceComplex cov;

  // Create complex data matrix
  Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> data(3, 2);
  data << Complex(1, 1), Complex(2, 0), Complex(0, 1), Complex(1, 1),
      Complex(1, 0), Complex(0, 2);

  Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> output(2, 2);
  output.setZero();

  auto data_map =
      Eigen::Map<const Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>>(
          data.data(), 3, 2);
  auto output_map =
      Eigen::Map<Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>>(
          output.data(), 2, 2);

  types::vector_const_matrix_map<Complex> inputs = {&data_map};
  types::vector_matrix_map<Complex> outputs = {&output_map};

  auto result = cov.operation(inputs, outputs);

  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Verify that the result is computed (exact values depend on complex
  // arithmetic)
  BOOST_CHECK(output(0, 0) != Complex(0, 0)); // Should not be zero
  BOOST_CHECK(output(1, 1) != Complex(0, 0)); // Should not be zero
}

// Test typename aliases
BOOST_AUTO_TEST_CASE(test_matrix_covariance_type_aliases) {
  // Test that the type aliases compile correctly
  matrix_covariance_sync_f::sptr sync_f_ptr = nullptr;
  matrix_covariance_sync_d::sptr sync_d_ptr = nullptr;
  matrix_covariance_sync_cf::sptr sync_cf_ptr = nullptr;
  matrix_covariance_sync_cd::sptr sync_cd_ptr = nullptr;

  matrix_covariance_pdu_f::sptr pdu_f_ptr = nullptr;
  matrix_covariance_pdu_d::sptr pdu_d_ptr = nullptr;
  matrix_covariance_pdu_cf::sptr pdu_cf_ptr = nullptr;
  matrix_covariance_pdu_cd::sptr pdu_cd_ptr = nullptr;

  // Just check that the pointers can be created (they will be null)
  BOOST_CHECK(sync_f_ptr == nullptr);
  BOOST_CHECK(sync_d_ptr == nullptr);
  BOOST_CHECK(sync_cf_ptr == nullptr);
  BOOST_CHECK(sync_cd_ptr == nullptr);

  BOOST_CHECK(pdu_f_ptr == nullptr);
  BOOST_CHECK(pdu_d_ptr == nullptr);
  BOOST_CHECK(pdu_cf_ptr == nullptr);
  BOOST_CHECK(pdu_cd_ptr == nullptr);
}

//==============================================================================
// GNU Radio Flowgraph Tests
//==============================================================================

/*!
 * \brief Test matrix covariance sync block in GNU Radio flowgraph
 *
 * Tests basic covariance computation through signal processing pipeline
 * Input: 4 samples, 3 features
 * Expected: 3x3 covariance matrix
 */
BOOST_AUTO_TEST_CASE(test_matrix_covariance_sync_flowgraph) {
  BOOST_TEST_MESSAGE(
      "Testing matrix covariance sync block in GNU Radio flowgraph...");

  auto tb = gr::make_top_block("matrix_covariance_test");

  // Create input data: 4 samples, 3 features
  // Data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
  std::vector<float> input_data = {
      1.0f, 4.0f, 7.0f, 10.0f, // Column 1
      2.0f, 5.0f, 8.0f, 11.0f, // Column 2
      3.0f, 6.0f, 9.0f, 12.0f  // Column 3
  };

  auto vector_source = gr::blocks::vector_source_f::make(
      input_data, false, 12); // 4x3 = 12 elements
  auto covariance_block = matrix_covariance_sync<float>::make(
      {4, 3}, true, true);                               // unbiased, centered
  auto output_sink = gr::blocks::vector_sink_f::make(9); // 3x3 = 9 elements

  tb->connect(vector_source, 0, covariance_block, 0);
  tb->connect(covariance_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 9);

  // Expected covariance matrix for this simple arithmetic sequence
  // Mean = [5.5, 6.5, 7.5], perfect correlation between features
  // Each column has variance = 16.67 (unbiased), covariance between all = 16.67
  float expected_value = 50.0f / 3.0f; // ≈ 16.67

  for (size_t i = 0; i < 9; ++i) {
    BOOST_CHECK_CLOSE(output_data[i], expected_value, 1.0f); // 1% tolerance
  }

  BOOST_TEST_MESSAGE("Matrix covariance sync flowgraph test passed.");
}

/*!
 * \brief Test matrix covariance with biased estimation
 */
BOOST_AUTO_TEST_CASE(test_matrix_covariance_sync_flowgraph_biased) {
  BOOST_TEST_MESSAGE(
      "Testing matrix covariance sync block with biased estimation...");

  auto tb = gr::make_top_block("matrix_covariance_biased_test");

  // Simple 3x2 data for easy verification
  std::vector<float> input_data = {
      1.0f, 2.0f, 3.0f, // Column 1: [1, 2, 3]
      4.0f, 5.0f, 6.0f  // Column 2: [4, 5, 6]
  };

  auto vector_source = gr::blocks::vector_source_f::make(input_data, false, 6);
  auto covariance_block = matrix_covariance_sync<float>::make(
      {3, 2}, false, true);                              // biased, centered
  auto output_sink = gr::blocks::vector_sink_f::make(4); // 2x2 = 4 elements

  tb->connect(vector_source, 0, covariance_block, 0);
  tb->connect(covariance_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 4);

  // Expected biased covariance: [[2/3, 2/3], [2/3, 2/3]]
  float expected_value = 2.0f / 3.0f;

  for (size_t i = 0; i < 4; ++i) {
    BOOST_CHECK_CLOSE(output_data[i], expected_value, 1e-5);
  }

  BOOST_TEST_MESSAGE("Biased matrix covariance flowgraph test passed.");
}

} /* namespace linalg */
} /* namespace gr */
