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
#include <gnuradio/linalg/matrix_diag.h>
#include <gnuradio/linalg/types.h>
#include <gnuradio/top_block.h>
#include <vector>

namespace gr {
namespace linalg {

// Test constants
constexpr float FLOAT_TOLERANCE = 1e-5f;
constexpr double DOUBLE_TOLERANCE = 1e-10;

// Test static method: compute_output_shapes
BOOST_AUTO_TEST_CASE(test_matrix_diag_compute_output_shapes_square) {
  types::vector_shapes input_shapes = {{3, 3}}; // 3x3 matrix
  types::vector_shapes expected = {{3, 1}};     // Diagonal should be 3x1 vector

  auto result = matrix_diag<float>::compute_output_shapes(input_shapes);

  BOOST_CHECK_EQUAL(result.size(), 1);
  BOOST_CHECK_EQUAL(result[0].size(), 2);
  BOOST_CHECK_EQUAL(result[0][0], 3);
  BOOST_CHECK_EQUAL(result[0][1], 1);
}

BOOST_AUTO_TEST_CASE(test_matrix_diag_compute_output_shapes_rectangular) {
  types::vector_shapes input_shapes = {{4, 3}}; // 4x3 matrix
  types::vector_shapes expected = {
      {3, 1}}; // Diagonal should be min(4,3)=3 elements

  auto result = matrix_diag<float>::compute_output_shapes(input_shapes);

  BOOST_CHECK_EQUAL(result.size(), 1);
  BOOST_CHECK_EQUAL(result[0][0], 3);
  BOOST_CHECK_EQUAL(result[0][1], 1);
}

BOOST_AUTO_TEST_CASE(test_matrix_diag_compute_output_shapes_non_square) {
  types::vector_shapes input_shapes = {{2, 5}}; // 2x5 matrix
  types::vector_shapes expected = {
      {2, 1}}; // Diagonal should be min(2,5)=2 elements

  auto result = matrix_diag<float>::compute_output_shapes(input_shapes);

  BOOST_CHECK_EQUAL(result.size(), 1);
  BOOST_CHECK_EQUAL(result[0][0], 2);
  BOOST_CHECK_EQUAL(result[0][1], 1);
}

BOOST_AUTO_TEST_CASE(test_matrix_diag_compute_output_shapes_empty_input) {
  types::vector_shapes input_shapes = {};

  BOOST_CHECK_THROW(
      { matrix_diag<float>::compute_output_shapes(input_shapes); },
      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_matrix_diag_compute_output_shapes_non_2d) {
  types::vector_shapes input_shapes = {{2, 3, 4}}; // 3D tensor

  BOOST_CHECK_THROW(
      { matrix_diag<float>::compute_output_shapes(input_shapes); },
      std::invalid_argument);
}

// Test matrix_diag base class functionality
BOOST_AUTO_TEST_CASE(test_matrix_diag_k_parameter) {
  types::shape shape = {3, 3};
  matrix_diag<float> diag(shape, 1); // k=1 for super-diagonal

  BOOST_CHECK_EQUAL(diag.get_k(), 1);

  diag.set_k(-1); // k=-1 for sub-diagonal
  BOOST_CHECK_EQUAL(diag.get_k(), -1);
}

// Test factory methods
BOOST_AUTO_TEST_CASE(test_matrix_diag_sync_factory) {
  types::shape shape = {3, 3};

  try {
    auto block = matrix_diag_sync<float>::make(shape, 0);
    BOOST_CHECK(block != nullptr);
  } catch (const std::exception &e) {
    BOOST_FAIL("matrix_diag_sync factory failed: " << e.what());
  }
}

BOOST_AUTO_TEST_CASE(test_matrix_diag_pdu_factory) {
  types::shape shape = {3, 3};

  try {
    auto block = matrix_diag_pdu<float>::make(shape, 0);
    BOOST_CHECK(block != nullptr);
  } catch (const std::exception &e) {
    BOOST_FAIL("matrix_diag_pdu factory failed: " << e.what());
  }
}

//==============================================================================
// GNU Radio Flowgraph Tests
//==============================================================================

/*!
 * \brief Test matrix diag sync block in GNU Radio flowgraph
 *
 * Tests diagonal extraction from a 3×3 matrix through signal processing
 * pipeline: Input: [[1, 2, 3], [4, 5, 6], [7, 8, 9]] (column-major: [1, 4, 7,
 * 2, 5, 8, 3, 6, 9]) Expected diagonal: [1, 5, 9] (main diagonal elements)
 */
BOOST_AUTO_TEST_CASE(test_matrix_diag_sync_flowgraph) {
  BOOST_TEST_MESSAGE(
      "Testing matrix diag sync block in GNU Radio flowgraph...");

  auto tb = gr::make_top_block("matrix_diag_test");

  // Create input matrix: [[1, 2, 3], [4, 5, 6], [7, 8, 9]] in column-major
  // order
  std::vector<float> input_matrix = {1.0f, 4.0f, 7.0f,  // Column 1
                                     2.0f, 5.0f, 8.0f,  // Column 2
                                     3.0f, 6.0f, 9.0f}; // Column 3

  types::shape shape = {3, 3};
  auto vector_source = gr::blocks::vector_source_f::make(input_matrix, false,
                                                         9); // 3×3 = 9 elements
  auto diag_block =
      matrix_diag_sync<float>::make(shape, 0); // k=0 for main diagonal
  auto output_sink =
      gr::blocks::vector_sink_f::make(3); // Expect 3 diagonal elements

  tb->connect(vector_source, 0, diag_block, 0);
  tb->connect(diag_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 3);

  // Expected diagonal elements: [1, 5, 9]
  std::vector<float> expected = {1.0f, 5.0f, 9.0f};
  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_CHECK_CLOSE(output_data[i], expected[i], FLOAT_TOLERANCE);
  }

  BOOST_TEST_MESSAGE("Matrix diag sync flowgraph test passed.");
}

/*!
 * \brief Test matrix diag with rectangular matrix in flowgraph
 *
 * Tests diagonal extraction from 2×4 matrix:
 * Input: [[1, 2, 3, 4], [5, 6, 7, 8]]
 * Expected diagonal: [1, 6] (min(2,4)=2 elements)
 */
BOOST_AUTO_TEST_CASE(test_matrix_diag_sync_flowgraph_rectangular) {
  BOOST_TEST_MESSAGE(
      "Testing matrix diag sync block with rectangular matrix in flowgraph...");

  auto tb = gr::make_top_block("matrix_diag_rect_test");

  // Create 2×4 input matrix: [[1, 2, 3, 4], [5, 6, 7, 8]]
  std::vector<float> input_matrix = {1.0f, 5.0f,  // Column 1
                                     2.0f, 6.0f,  // Column 2
                                     3.0f, 7.0f,  // Column 3
                                     4.0f, 8.0f}; // Column 4

  types::shape shape = {2, 4};
  auto vector_source = gr::blocks::vector_source_f::make(input_matrix, false,
                                                         8); // 2×4 = 8 elements
  auto diag_block = matrix_diag_sync<float>::make(shape, 0);
  auto output_sink =
      gr::blocks::vector_sink_f::make(2); // Expect min(2,4)=2 diagonal elements

  tb->connect(vector_source, 0, diag_block, 0);
  tb->connect(diag_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 2);

  // Expected diagonal elements: [1, 6]
  std::vector<float> expected = {1.0f, 6.0f};
  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_CHECK_CLOSE(output_data[i], expected[i], FLOAT_TOLERANCE);
  }

  BOOST_TEST_MESSAGE("Rectangular matrix diag flowgraph test passed.");
}

/*!
 * \brief Test matrix diag with complex numbers in flowgraph
 *
 * Tests diagonal extraction from complex 2×2 matrix
 */
BOOST_AUTO_TEST_CASE(test_matrix_diag_sync_flowgraph_complex) {
  BOOST_TEST_MESSAGE(
      "Testing matrix diag sync block with complex numbers in flowgraph...");

  using cf = std::complex<float>;
  auto tb = gr::make_top_block("matrix_diag_complex_test");

  // Create complex 2×2 matrix: [[1+2j, 3+0j], [0+4j, 5+6j]]
  std::vector<cf> input_matrix = {cf(1, 2), cf(0, 4),  // Column 1
                                  cf(3, 0), cf(5, 6)}; // Column 2

  types::shape shape = {2, 2};
  auto vector_source =
      gr::blocks::vector_source_c::make(input_matrix, false, 4);
  auto diag_block = matrix_diag_sync<cf>::make(shape, 0);
  auto output_sink = gr::blocks::vector_sink_c::make(2);

  tb->connect(vector_source, 0, diag_block, 0);
  tb->connect(diag_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 2);

  // Expected diagonal elements: [1+2j, 5+6j]
  std::vector<cf> expected = {cf(1, 2), cf(5, 6)};
  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_CHECK_CLOSE(output_data[i].real(), expected[i].real(),
                      FLOAT_TOLERANCE);
    BOOST_CHECK_CLOSE(output_data[i].imag(), expected[i].imag(),
                      FLOAT_TOLERANCE);
  }

  BOOST_TEST_MESSAGE("Complex matrix diag flowgraph test passed.");
}

/*!
 * \brief Test matrix diag with double precision in flowgraph
 *
 * Tests high precision diagonal extraction
 */
BOOST_AUTO_TEST_CASE(test_matrix_diag_sync_flowgraph_double) {
  BOOST_TEST_MESSAGE(
      "Testing matrix diag sync block with double precision in flowgraph...");

  auto tb = gr::make_top_block("matrix_diag_double_test");

  // Create double precision 3×3 matrix with high precision values
  std::vector<double> input_matrix = {
      1.123456789, 4.987654321, 7.555555555, // Column 1
      2.271828183, 5.000000001, 8.777777777, // Column 2
      3.141592654, 6.456456456, 9.999999999  // Column 3
  };

  // Convert to float for GNU Radio compatibility
  std::vector<float> input_matrix_f(input_matrix.begin(), input_matrix.end());

  types::shape shape = {3, 3};
  auto vector_source =
      gr::blocks::vector_source_f::make(input_matrix_f, false, 9);
  auto diag_block = matrix_diag_sync<float>::make(shape, 0);
  auto output_sink = gr::blocks::vector_sink_f::make(3);

  tb->connect(vector_source, 0, diag_block, 0);
  tb->connect(diag_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 3);

  // Expected diagonal elements: [1.123456789, 5.000000001, 9.999999999]
  std::vector<float> expected = {
      static_cast<float>(input_matrix[0]), // 1.123456789
      static_cast<float>(input_matrix[4]), // 5.000000001
      static_cast<float>(input_matrix[8])  // 9.999999999
  };

  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_CHECK_CLOSE(output_data[i], expected[i],
                      1e-4f); // Relaxed tolerance for float precision
  }

  BOOST_TEST_MESSAGE("Double precision matrix diag flowgraph test passed.");
}

/*!
 * \brief Test matrix diag with identity matrix in flowgraph
 *
 * Tests diagonal extraction from identity matrix
 */
BOOST_AUTO_TEST_CASE(test_matrix_diag_sync_flowgraph_identity) {
  BOOST_TEST_MESSAGE(
      "Testing matrix diag sync block with identity matrix in flowgraph...");

  auto tb = gr::make_top_block("matrix_diag_identity_test");

  // Create 4×4 identity matrix
  std::vector<float> input_matrix = {
      1.0f, 0.0f, 0.0f, 0.0f, // Column 1
      0.0f, 1.0f, 0.0f, 0.0f, // Column 2
      0.0f, 0.0f, 1.0f, 0.0f, // Column 3
      0.0f, 0.0f, 0.0f, 1.0f  // Column 4
  };

  types::shape shape = {4, 4};
  auto vector_source =
      gr::blocks::vector_source_f::make(input_matrix, false, 16);
  auto diag_block = matrix_diag_sync<float>::make(shape, 0);
  auto output_sink = gr::blocks::vector_sink_f::make(4);

  tb->connect(vector_source, 0, diag_block, 0);
  tb->connect(diag_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 4);

  // Expected diagonal should be all ones: [1, 1, 1, 1]
  for (size_t i = 0; i < output_data.size(); ++i) {
    BOOST_CHECK_CLOSE(output_data[i], 1.0f, FLOAT_TOLERANCE);
  }

  BOOST_TEST_MESSAGE("Identity matrix diag flowgraph test passed.");
}

/*!
 * \brief Test matrix diag with zero matrix in flowgraph
 *
 * Tests diagonal extraction from all-zero matrix
 */
BOOST_AUTO_TEST_CASE(test_matrix_diag_sync_flowgraph_zeros) {
  BOOST_TEST_MESSAGE(
      "Testing matrix diag sync block with zero matrix in flowgraph...");

  auto tb = gr::make_top_block("matrix_diag_zeros_test");

  // Create 3×3 zero matrix
  std::vector<float> input_matrix(9, 0.0f);

  types::shape shape = {3, 3};
  auto vector_source =
      gr::blocks::vector_source_f::make(input_matrix, false, 9);
  auto diag_block = matrix_diag_sync<float>::make(shape, 0);
  auto output_sink = gr::blocks::vector_sink_f::make(3);

  tb->connect(vector_source, 0, diag_block, 0);
  tb->connect(diag_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), 3);

  // Expected diagonal should be all zeros: [0, 0, 0]
  for (size_t i = 0; i < output_data.size(); ++i) {
    BOOST_CHECK_SMALL(output_data[i], FLOAT_TOLERANCE);
  }

  BOOST_TEST_MESSAGE("Zero matrix diag flowgraph test passed.");
}

/*!
 * \brief Test matrix diag with multiple matrices in flowgraph
 *
 * Tests processing of multiple consecutive matrices
 */
BOOST_AUTO_TEST_CASE(test_matrix_diag_sync_flowgraph_multiple_matrices) {
  BOOST_TEST_MESSAGE(
      "Testing matrix diag sync block with multiple matrices in flowgraph...");

  auto tb = gr::make_top_block("matrix_diag_multi_test");

  // Create multiple 2×2 matrices
  std::vector<float> input_matrices = {// Matrix 1: [[1, 2], [3, 4]]
                                       1.0f, 3.0f, 2.0f, 4.0f,
                                       // Matrix 2: [[5, 6], [7, 8]]
                                       5.0f, 7.0f, 6.0f, 8.0f,
                                       // Matrix 3: [[9, 10], [11, 12]]
                                       9.0f, 11.0f, 10.0f, 12.0f};

  types::shape shape = {2, 2};
  auto vector_source = gr::blocks::vector_source_f::make(input_matrices, false,
                                                         4); // 2×2 per matrix
  auto diag_block = matrix_diag_sync<float>::make(shape, 0);
  auto output_sink =
      gr::blocks::vector_sink_f::make(2); // 2 diagonal elements per matrix

  tb->connect(vector_source, 0, diag_block, 0);
  tb->connect(diag_block, 0, output_sink, 0);

  tb->run();

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(),
                      6); // 3 matrices × 2 diagonal elements each

  // Expected diagonal elements: [1, 4] from matrix 1, [5, 8] from matrix 2, [9,
  // 12] from matrix 3
  std::vector<float> expected = {1.0f, 4.0f, 5.0f, 8.0f, 9.0f, 12.0f};
  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_CHECK_CLOSE(output_data[i], expected[i], FLOAT_TOLERANCE);
  }

  BOOST_TEST_MESSAGE("Multiple matrices diag flowgraph test passed.");
}

/*!
 * \brief Test matrix diag performance in flowgraph
 *
 * Tests performance with many matrix diagonal extractions
 */
BOOST_AUTO_TEST_CASE(test_matrix_diag_sync_flowgraph_performance) {
  BOOST_TEST_MESSAGE(
      "Testing matrix diag sync block performance in flowgraph...");

  auto tb = gr::make_top_block("matrix_diag_perf_test");

  // Create many 3×3 matrices for performance testing
  int num_matrices = 1000;
  std::vector<float> input_matrices;
  input_matrices.reserve(num_matrices * 9);

  for (int i = 0; i < num_matrices; ++i) {
    // Generate matrix with varying diagonal values
    float diag_val = 1.0f + 0.001f * i;
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        if (row == col) {
          input_matrices.push_back(diag_val + row); // Diagonal elements
        } else {
          input_matrices.push_back(0.5f *
                                   (row + col + 1)); // Off-diagonal elements
        }
      }
    }
  }

  auto start_time = std::chrono::high_resolution_clock::now();

  types::shape shape = {3, 3};
  auto vector_source =
      gr::blocks::vector_source_f::make(input_matrices, false, 9);
  auto diag_block = matrix_diag_sync<float>::make(shape, 0);
  auto output_sink = gr::blocks::vector_sink_f::make(3);

  tb->connect(vector_source, 0, diag_block, 0);
  tb->connect(diag_block, 0, output_sink, 0);

  tb->run();

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);

  auto output_data = output_sink->data();
  BOOST_REQUIRE_EQUAL(output_data.size(), num_matrices * 3);

  // Verify first and last matrix diagonal elements
  BOOST_CHECK_CLOSE(output_data[0], 1.0f,
                    FLOAT_TOLERANCE); // First matrix, first diagonal element
  BOOST_CHECK_CLOSE(output_data[1], 2.0f,
                    FLOAT_TOLERANCE); // First matrix, second diagonal element
  BOOST_CHECK_CLOSE(output_data[2], 3.0f,
                    FLOAT_TOLERANCE); // First matrix, third diagonal element

  size_t last_matrix_start = (num_matrices - 1) * 3;
  float last_base = 1.0f + 0.001f * (num_matrices - 1);
  BOOST_CHECK_CLOSE(output_data[last_matrix_start], last_base,
                    1e-2f); // Last matrix, first diagonal
  BOOST_CHECK_CLOSE(output_data[last_matrix_start + 1], last_base + 1,
                    1e-2f); // Last matrix, second diagonal
  BOOST_CHECK_CLOSE(output_data[last_matrix_start + 2], last_base + 2,
                    1e-2f); // Last matrix, third diagonal

  // All results should be finite
  for (size_t i = 0; i < output_data.size();
       i += 100) { // Check every 100th element for speed
    BOOST_CHECK(std::isfinite(output_data[i]));
  }

  BOOST_TEST_MESSAGE("Processed " << num_matrices
                                  << " 3×3 matrix diagonal extractions in "
                                  << duration.count() << " ms");
  BOOST_TEST_MESSAGE("Matrix diag performance flowgraph test passed.");
}

} /* namespace linalg */
} /* namespace gr */
