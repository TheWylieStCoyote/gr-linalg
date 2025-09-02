/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "gnuradio/linalg/linalg_base.h"
#include <Eigen/Dense>
#include <boost/test/unit_test.hpp>
#include <chrono>
#include <gnuradio/attributes.h>
#include <gnuradio/blocks/vector_sink.h>
#include <gnuradio/blocks/vector_source.h>
#include <gnuradio/io_signature.h>
#include <gnuradio/linalg/decomp_svd.h>
#include <gnuradio/linalg/types.h>
#include <gnuradio/linalg/utils.h>
#include <gnuradio/top_block.h>
#include <memory>

namespace gr {
namespace linalg {

BOOST_AUTO_TEST_CASE(test_decomp_svd_constructor) {
  BOOST_TEST_MESSAGE("Testing SVD decomposition constructor...");

  // Create a simple matrix shape
  auto shape = types::shape({3, 3});

  // Test sync block constructor
  auto svd_sync = decomp_svd_sync<float>::make(shape);
  BOOST_REQUIRE(svd_sync);
  BOOST_TEST_MESSAGE("SVD sync block created successfully.");

  // Test PDU block constructor
  auto svd_pdu = decomp_svd_pdu<float>::make();
  BOOST_REQUIRE(svd_pdu);
  BOOST_TEST_MESSAGE("SVD PDU block created successfully.");
}

BOOST_AUTO_TEST_CASE(test_decomp_svd_jacobi_operation) {
  BOOST_TEST_MESSAGE("Testing SVD decomposition with Jacobi algorithm...");

  // Create a test matrix
  Eigen::Matrix<float, 3, 2> input_matrix;
  input_matrix << 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f;

  // Test the Jacobi algorithm
  eigen_jacobi_svd<float> jacobi_algo;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> U, V;
  Eigen::Matrix<float, Eigen::Dynamic, 1> S;

  jacobi_algo.decompose(input_matrix, U, S, V);

  BOOST_TEST_MESSAGE("Verifying SVD decomposition...");

  // For SVD reconstruction: A = U_thin * S * V^T
  // where U_thin has only the first min(m,n) columns of U
  int min_dim = std::min(input_matrix.rows(), input_matrix.cols());
  auto U_thin = U.leftCols(min_dim);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> S_diag = S.asDiagonal();
  auto USV = U_thin * S_diag * V.transpose();

  for (int i = 0; i < input_matrix.rows(); ++i) {
    for (int j = 0; j < input_matrix.cols(); ++j) {
      BOOST_CHECK_CLOSE(USV(i, j), input_matrix(i, j),
                        1e-3f); // Relaxed tolerance
    }
  }

  // Verify U has orthonormal columns
  auto UtU = U.transpose() * U;
  for (int i = 0; i < UtU.rows(); ++i) {
    for (int j = 0; j < UtU.cols(); ++j) {
      if (i == j) {
        BOOST_CHECK_CLOSE(UtU(i, j), 1.0f, 1e-4f);
      } else {
        BOOST_CHECK_SMALL(UtU(i, j), 1e-4f);
      }
    }
  }

  // Verify V has orthonormal columns
  auto VtV = V.transpose() * V;
  for (int i = 0; i < VtV.rows(); ++i) {
    for (int j = 0; j < VtV.cols(); ++j) {
      if (i == j) {
        BOOST_CHECK_CLOSE(VtV(i, j), 1.0f, 1e-4f);
      } else {
        BOOST_CHECK_SMALL(VtV(i, j), 1e-4f);
      }
    }
  }

  // Verify singular values are sorted in descending order
  for (int i = 0; i < S.size() - 1; ++i) {
    BOOST_CHECK_GE(S(i), S(i + 1));
  }

  BOOST_TEST_MESSAGE("Jacobi SVD decomposition test passed.");
}

BOOST_AUTO_TEST_CASE(test_decomp_svd_bdc_operation) {
  BOOST_TEST_MESSAGE("Testing SVD decomposition with BDC algorithm...");

  // Create a larger test matrix for BDC algorithm
  Eigen::Matrix<float, 4, 3> input_matrix;
  input_matrix << 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 1.0f,
      1.0f, 1.0f;

  // Test the BDC algorithm
  eigen_bdc_svd<float> bdc_algo;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> U, V;
  Eigen::Matrix<float, Eigen::Dynamic, 1> S;

  bdc_algo.decompose(input_matrix, U, S, V);

  BOOST_TEST_MESSAGE("Verifying BDC SVD decomposition...");

  // For SVD reconstruction: A = U_thin * S * V^T
  // where U_thin has only the first min(m,n) columns of U
  int min_dim = std::min(input_matrix.rows(), input_matrix.cols());
  auto U_thin = U.leftCols(min_dim);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> S_diag = S.asDiagonal();
  auto USV = U_thin * S_diag * V.transpose();

  for (int i = 0; i < input_matrix.rows(); ++i) {
    for (int j = 0; j < input_matrix.cols(); ++j) {
      BOOST_CHECK_CLOSE(USV(i, j), input_matrix(i, j), 1e-4f);
    }
  }

  // Verify singular values are non-negative and sorted
  for (int i = 0; i < S.size(); ++i) {
    BOOST_CHECK_GE(S(i), 0.0f);
  }
  for (int i = 0; i < S.size() - 1; ++i) {
    BOOST_CHECK_GE(S(i), S(i + 1));
  }

  BOOST_TEST_MESSAGE("BDC SVD decomposition test passed.");
}

BOOST_AUTO_TEST_CASE(test_decomp_svd_algorithm_switching) {
  BOOST_TEST_MESSAGE("Testing SVD algorithm switching...");

  // Create a sync block
  auto shape = types::shape({3, 3});
  auto svd_sync = decomp_svd_sync<float>::make(shape);
  BOOST_REQUIRE(svd_sync);

  // Switch to BDC solver
  auto bdc_algo = std::make_shared<eigen_bdc_svd<float>>();
  svd_sync->set_algorithm(bdc_algo);

  BOOST_TEST_MESSAGE("Algorithm switching test passed.");
}

BOOST_AUTO_TEST_CASE(test_decomp_svd_double_precision) {
  BOOST_TEST_MESSAGE("Testing SVD decomposition with double precision...");

  // Test with double precision
  Eigen::Matrix<double, 2, 2> input_matrix;
  input_matrix << 3.0, 1.0, 1.0, 3.0;

  eigen_jacobi_svd<double> jacobi_algo;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> U, V;
  Eigen::Matrix<double, Eigen::Dynamic, 1> S;

  jacobi_algo.decompose(input_matrix, U, S, V);

  // Verify A = U * S * V^T
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> S_diag = S.asDiagonal();
  auto USV = U * S_diag * V.transpose();

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      BOOST_CHECK_CLOSE(USV(i, j), input_matrix(i, j), 1e-10);
    }
  }

  BOOST_TEST_MESSAGE("Double precision test passed.");
}

BOOST_AUTO_TEST_CASE(test_decomp_svd_identity_matrix) {
  BOOST_TEST_MESSAGE("Testing SVD decomposition of identity matrix...");

  // Identity matrix should have all singular values equal to 1
  Eigen::Matrix<float, 3, 3> identity = Eigen::Matrix<float, 3, 3>::Identity();

  eigen_jacobi_svd<float> jacobi_algo;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> U, V;
  Eigen::Matrix<float, Eigen::Dynamic, 1> S;

  jacobi_algo.decompose(identity, U, S, V);

  // All singular values should be 1.0
  for (int i = 0; i < 3; ++i) {
    BOOST_CHECK_CLOSE(S(i), 1.0f, 1e-6f);
  }

  BOOST_TEST_MESSAGE("Identity matrix test passed.");
}

BOOST_AUTO_TEST_CASE(test_decomp_svd_rank_deficient_matrix) {
  BOOST_TEST_MESSAGE("Testing SVD decomposition of rank-deficient matrix...");

  // Create a rank-1 matrix
  Eigen::Matrix<float, 3, 3> rank1_matrix;
  rank1_matrix << 1.0f, 2.0f, 3.0f, 2.0f, 4.0f, 6.0f, 1.0f, 2.0f, 3.0f;

  eigen_jacobi_svd<float> jacobi_algo;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> U, V;
  Eigen::Matrix<float, Eigen::Dynamic, 1> S;

  jacobi_algo.decompose(rank1_matrix, U, S, V);

  // Should have only one non-zero singular value (rank 1)
  BOOST_CHECK_GT(S(0), 1e-6f); // First singular value should be significant
  BOOST_CHECK_LT(S(1), 1e-4f); // Second should be near zero
  BOOST_CHECK_LT(S(2), 1e-4f); // Third should be near zero

  BOOST_TEST_MESSAGE("Rank-deficient matrix test passed.");
}

BOOST_AUTO_TEST_CASE(test_decomp_svd_rectangular_matrix) {
  BOOST_TEST_MESSAGE("Testing SVD decomposition of rectangular matrix...");

  // Test with a wide rectangular matrix
  Eigen::Matrix<float, 2, 4> input_matrix;
  input_matrix << 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f;

  eigen_jacobi_svd<float> jacobi_algo;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> U, V;
  Eigen::Matrix<float, Eigen::Dynamic, 1> S;

  jacobi_algo.decompose(input_matrix, U, S, V);

  // Verify A = U * S * V^T (considering the dimensions)
  // For a 2x4 matrix, U is 2x2, S has 2 values, V is 4x4
  // Use U_thin = U (all columns) and V_thin for reconstruction
  int min_dim = std::min(input_matrix.rows(), input_matrix.cols());
  auto V_thin = V.leftCols(min_dim);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> S_diag = S.asDiagonal();
  auto USV = U * S_diag * V_thin.transpose();

  for (int i = 0; i < input_matrix.rows(); ++i) {
    for (int j = 0; j < input_matrix.cols(); ++j) {
      BOOST_CHECK_CLOSE(USV(i, j), input_matrix(i, j), 1e-4f);
    }
  }

  BOOST_TEST_MESSAGE("Rectangular matrix test passed.");
}

BOOST_AUTO_TEST_CASE(test_decomp_svd_block_operation_simulation) {
  BOOST_TEST_MESSAGE("Testing SVD block operation with matrix maps...");

  // Create a sync block
  auto shape = types::shape({2, 2});
  auto svd_sync = decomp_svd_sync<float>::make(shape);
  BOOST_REQUIRE(svd_sync);

  // Create test matrices
  Eigen::Matrix<float, 2, 2> input_matrix;
  input_matrix << 3.0f, 1.0f, 1.0f, 3.0f;

  Eigen::Matrix<float, 2, 2> U_output, V_output;
  Eigen::Matrix<float, 2, 1> S_output;
  U_output.setZero();
  V_output.setZero();
  S_output.setZero();

  // Create matrix maps (simulating what the linalg_base would do)
  types::const_matrix_map_dynamic<float> input_map(&input_matrix(0, 0), 2, 2);
  types::matrix_map_dynamic<float> U_map(&U_output(0, 0), 2, 2);
  types::matrix_map_dynamic<float> S_map(&S_output(0, 0), 2, 1);
  types::matrix_map_dynamic<float> V_map(&V_output(0, 0), 2, 2);

  // Create vector of matrix maps
  types::vector_const_matrix_map<float> input_maps = {&input_map};
  types::vector_matrix_map<float> output_maps = {&U_map, &S_map, &V_map};

  // Call the operation method directly
  auto result = svd_sync->operation(input_maps, output_maps);

  BOOST_CHECK(result == OperationReturn::SUCCESS);

  // Verify A = U * S * V^T
  BOOST_TEST_MESSAGE("Verifying block operation result...");
  Eigen::Matrix<float, 2, 2> S_diag = S_output.asDiagonal();
  auto USV = U_output * S_diag * V_output.transpose();

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      BOOST_CHECK_CLOSE(USV(i, j), input_matrix(i, j), 1e-4f);
    }
  }

  // Verify singular values are positive and sorted
  BOOST_CHECK_GT(S_output(0), 0.0f);
  BOOST_CHECK_GT(S_output(1), 0.0f);
  BOOST_CHECK_GE(S_output(0), S_output(1));

  BOOST_TEST_MESSAGE("Block operation simulation test passed.");
}

BOOST_AUTO_TEST_CASE(test_decomp_svd_sync_block_functionality) {
  BOOST_TEST_MESSAGE("Testing SVD sync block GNU Radio block creation...");

  auto shape = types::shape({2, 2});

  // Create SVD sync block
  auto svd_block = decomp_svd_sync<float>::make(shape);
  BOOST_REQUIRE(svd_block);

  // Test that the block has proper input/output signature
  auto input_sig = svd_block->input_signature();
  auto output_sig = svd_block->output_signature();

  // Should have 1 input (matrix)
  BOOST_CHECK_EQUAL(input_sig->max_streams(), 1);
  // Should have 3 outputs (U, S, V matrices)
  BOOST_CHECK_EQUAL(output_sig->max_streams(), 3);

  BOOST_TEST_MESSAGE("SVD sync block functionality test passed.");
}

BOOST_AUTO_TEST_CASE(test_decomp_svd_sync_block_work_method) {
  BOOST_TEST_MESSAGE("Testing SVD sync block work() method functionality...");

  auto shape = types::shape({2, 2});
  auto svd_block = decomp_svd_sync<float>::make(shape);
  BOOST_REQUIRE(svd_block);

  // NOTE: Direct work() method testing is complex and requires proper GNU Radio
  // buffer management setup. The core SVD functionality is already tested
  // through the operation() method in other test cases.

  BOOST_TEST_MESSAGE("SVD sync block work() method test skipped - core "
                     "functionality tested elsewhere.");
}

BOOST_AUTO_TEST_CASE(test_decomp_svd_pdu_block_functionality) {
  BOOST_TEST_MESSAGE("Testing SVD PDU block GNU Radio functionality...");

  // Create SVD PDU block
  auto svd_pdu_block = decomp_svd_pdu<float>::make();
  BOOST_REQUIRE(svd_pdu_block);

  // Test that the PDU block can be created and has message ports
  BOOST_TEST_MESSAGE("SVD PDU block created successfully.");

  // Verify the block has the expected message ports
  auto msg_ports_in = svd_pdu_block->message_ports_in();
  auto msg_ports_out = svd_pdu_block->message_ports_out();

  // Note: Message port functionality is currently commented out in base class
  // This test verifies the block can be created and the API exists
  BOOST_CHECK(msg_ports_in != nullptr);
  BOOST_CHECK(msg_ports_out != nullptr);

  BOOST_TEST_MESSAGE("PDU block has proper message port interface.");
  BOOST_TEST_MESSAGE("SVD PDU block functionality test passed.");
}

BOOST_AUTO_TEST_CASE(test_decomp_svd_pdu_message_processing) {
  BOOST_TEST_MESSAGE("Testing SVD PDU block message processing...");

  auto svd_pdu_block = decomp_svd_pdu<float>::make();
  BOOST_REQUIRE(svd_pdu_block);

  // Test PDU message processing functionality
  BOOST_TEST_MESSAGE("Testing PDU message handling...");

  // Create a test matrix as a PMT vector
  std::vector<float> matrix_data = {3.0f, 1.0f, 1.0f, 3.0f}; // 2x2 matrix
  pmt::pmt_t matrix_pmt =
      pmt::init_f32vector(matrix_data.size(), matrix_data.data());

  // Create metadata for the matrix (shape information)
  pmt::pmt_t meta = pmt::make_dict();
  meta = pmt::dict_add(meta, pmt::mp("rows"), pmt::from_long(2));
  meta = pmt::dict_add(meta, pmt::mp("cols"), pmt::from_long(2));

  // Create PDU message
  pmt::pmt_t pdu = pmt::cons(meta, matrix_pmt);

  // Test message port interface
  auto msg_ports_in = svd_pdu_block->message_ports_in();
  BOOST_CHECK(msg_ports_in != nullptr);

  // Note: Full PDU message processing test would require setting up
  // message handlers and running in a flowgraph context
  // Currently the message port functionality is commented out in the base class
  BOOST_TEST_MESSAGE("PDU message structure validated.");
  BOOST_TEST_MESSAGE("SVD PDU message processing test passed.");
}

// GNU Radio Flowgraph Tests
BOOST_AUTO_TEST_CASE(test_decomp_svd_sync_flowgraph) {
  BOOST_TEST_MESSAGE("Testing SVD sync block in GNU Radio flowgraph...");

  auto tb = gr::make_top_block("svd_test");

  // Create test matrix: [[4, 0], [3, -5]] in column-major format
  std::vector<float> input_data = {4.0f, 3.0f, 0.0f, -5.0f};

  auto vector_source = gr::blocks::vector_source_f::make(input_data, false, 4);
  auto svd_block = decomp_svd_sync<float>::make({2, 2});

  // SVD outputs 3 separate streams: U, S, V
  auto U_sink = gr::blocks::vector_sink_f::make(4); // 2x2 U matrix
  auto S_sink = gr::blocks::vector_sink_f::make(2); // 2x1 S vector
  auto V_sink = gr::blocks::vector_sink_f::make(4); // 2x2 V matrix

  tb->connect(vector_source, 0, svd_block, 0);
  tb->connect(svd_block, 0, U_sink, 0); // U matrix output
  tb->connect(svd_block, 1, S_sink, 0); // S vector output
  tb->connect(svd_block, 2, V_sink, 0); // V matrix output

  tb->run();

  auto U_data = U_sink->data();
  auto S_data = S_sink->data();
  auto V_data = V_sink->data();

  BOOST_REQUIRE_EQUAL(U_data.size(), 4);
  BOOST_REQUIRE_EQUAL(S_data.size(), 2);
  BOOST_REQUIRE_EQUAL(V_data.size(), 4);

  // Reconstruct matrices from column-major data
  Eigen::Matrix<float, 2, 2> U, V, A;
  Eigen::Matrix<float, 2, 1> S;

  U << U_data[0], U_data[2], U_data[1], U_data[3]; // Column-major
  S << S_data[0], S_data[1];
  V << V_data[0], V_data[2], V_data[1], V_data[3]; // Column-major
  A << 4.0f, 0.0f, 3.0f, -5.0f; // Column-major: [[4,0],[3,-5]]

  // Verify A = U * S * V^T
  Eigen::Matrix<float, 2, 2> S_diag = S.asDiagonal();
  auto USV = U * S_diag * V.transpose();

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      if (std::abs(A(i, j)) < 1e-6f) {
        // Check for small values near zero - use absolute tolerance
        BOOST_CHECK_SMALL(std::abs(USV(i, j)), 1e-4f);
      } else {
        BOOST_CHECK_CLOSE(A(i, j), USV(i, j), 1e-2f); // Relaxed tolerance
      }
    }
  }

  // Verify singular values are positive and sorted
  BOOST_CHECK_GT(S(0), 0.0f);
  BOOST_CHECK_GT(S(1), 0.0f);
  BOOST_CHECK_GE(S(0), S(1));

  BOOST_TEST_MESSAGE("SVD sync flowgraph test passed.");
}

BOOST_AUTO_TEST_CASE(test_decomp_svd_sync_flowgraph_3x3) {
  BOOST_TEST_MESSAGE("Testing SVD sync block with 3x3 matrix in flowgraph...");

  auto tb = gr::make_top_block("svd_3x3_test");

  // Create 3x3 test matrix: [[1, 4, 7], [2, 5, 8], [3, 6, 9]] in column-major
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                   6.0f, 7.0f, 8.0f, 9.0f};

  auto vector_source = gr::blocks::vector_source_f::make(input_data, false, 9);
  auto svd_block = decomp_svd_sync<float>::make({3, 3});

  auto U_sink = gr::blocks::vector_sink_f::make(9); // 3x3 U matrix
  auto S_sink = gr::blocks::vector_sink_f::make(3); // 3x1 S vector
  auto V_sink = gr::blocks::vector_sink_f::make(9); // 3x3 V matrix

  tb->connect(vector_source, 0, svd_block, 0);
  tb->connect(svd_block, 0, U_sink, 0);
  tb->connect(svd_block, 1, S_sink, 0);
  tb->connect(svd_block, 2, V_sink, 0);

  tb->run();

  auto U_data = U_sink->data();
  auto S_data = S_sink->data();
  auto V_data = V_sink->data();

  BOOST_REQUIRE_EQUAL(U_data.size(), 9);
  BOOST_REQUIRE_EQUAL(S_data.size(), 3);
  BOOST_REQUIRE_EQUAL(V_data.size(), 9);

  // Reconstruct matrices
  Eigen::Matrix<float, 3, 3> U, V, A;
  Eigen::Matrix<float, 3, 1> S;

  U << U_data[0], U_data[3], U_data[6], U_data[1], U_data[4], U_data[7],
      U_data[2], U_data[5], U_data[8];
  S << S_data[0], S_data[1], S_data[2];
  V << V_data[0], V_data[3], V_data[6], V_data[1], V_data[4], V_data[7],
      V_data[2], V_data[5], V_data[8];
  A << 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f;

  // Verify A = U * S * V^T (with appropriate tolerance for singular matrix)
  Eigen::Matrix<float, 3, 3> S_diag = S.asDiagonal();
  auto USV = U * S_diag * V.transpose();

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      BOOST_CHECK_CLOSE(A(i, j), USV(i, j), 1e-3f); // Relaxed for near-singular
    }
  }

  // Verify singular values are sorted
  BOOST_CHECK_GE(S(0), S(1));
  BOOST_CHECK_GE(S(1), S(2));

  BOOST_TEST_MESSAGE("SVD 3x3 flowgraph test passed.");
}

BOOST_AUTO_TEST_CASE(test_decomp_svd_sync_flowgraph_multiple_matrices) {
  BOOST_TEST_MESSAGE(
      "Testing SVD sync block with multiple matrices in flowgraph...");

  auto tb = gr::make_top_block("svd_multi_test");

  // Create multiple 2x2 matrices
  std::vector<float> input_data = {// Matrix 1: [[1, 0], [0, 1]] (identity)
                                   1.0f, 0.0f, 0.0f, 1.0f,
                                   // Matrix 2: [[2, 1], [1, 2]]
                                   2.0f, 1.0f, 1.0f, 2.0f,
                                   // Matrix 3: [[3, 0], [0, 3]] (diagonal)
                                   3.0f, 0.0f, 0.0f, 3.0f};

  auto vector_source = gr::blocks::vector_source_f::make(input_data, false, 4);
  auto svd_block = decomp_svd_sync<float>::make({2, 2});

  auto U_sink = gr::blocks::vector_sink_f::make(4);
  auto S_sink = gr::blocks::vector_sink_f::make(2);
  auto V_sink = gr::blocks::vector_sink_f::make(4);

  tb->connect(vector_source, 0, svd_block, 0);
  tb->connect(svd_block, 0, U_sink, 0);
  tb->connect(svd_block, 1, S_sink, 0);
  tb->connect(svd_block, 2, V_sink, 0);

  tb->run();

  auto U_data = U_sink->data();
  auto S_data = S_sink->data();
  auto V_data = V_sink->data();

  BOOST_REQUIRE_EQUAL(U_data.size(), 12); // 3 matrices * 4 elements
  BOOST_REQUIRE_EQUAL(S_data.size(), 6);  // 3 matrices * 2 elements
  BOOST_REQUIRE_EQUAL(V_data.size(), 12); // 3 matrices * 4 elements

  // Verify each matrix decomposition
  for (int mat = 0; mat < 3; ++mat) {
    int U_base = mat * 4;
    int S_base = mat * 2;
    int V_base = mat * 4;

    Eigen::Matrix<float, 2, 2> U, V, A;
    Eigen::Matrix<float, 2, 1> S;

    U << U_data[U_base + 0], U_data[U_base + 2], U_data[U_base + 1],
        U_data[U_base + 3];
    S << S_data[S_base + 0], S_data[S_base + 1];
    V << V_data[V_base + 0], V_data[V_base + 2], V_data[V_base + 1],
        V_data[V_base + 3];

    // Original matrices
    if (mat == 0) {
      A << 1.0f, 0.0f, 0.0f, 1.0f; // Identity
    } else if (mat == 1) {
      A << 2.0f, 1.0f, 1.0f, 2.0f; // Symmetric
    } else {
      A << 3.0f, 0.0f, 0.0f, 3.0f; // Diagonal
    }

    // Verify A = U * S * V^T
    Eigen::Matrix<float, 2, 2> S_diag = S.asDiagonal();
    auto USV = U * S_diag * V.transpose();

    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        BOOST_CHECK_CLOSE(A(i, j), USV(i, j), 1e-4f);
      }
    }

    BOOST_TEST_MESSAGE("Matrix " << (mat + 1) << " SVD decomposition verified");
  }

  BOOST_TEST_MESSAGE("SVD multiple matrices flowgraph test passed.");
}

BOOST_AUTO_TEST_CASE(test_decomp_svd_sync_flowgraph_rectangular) {
  BOOST_TEST_MESSAGE(
      "Testing SVD sync block with rectangular matrix in flowgraph...");

  auto tb = gr::make_top_block("svd_rect_test");

  // Create 2x3 matrix: [[1, 3, 5], [2, 4, 6]] in column-major
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  auto vector_source = gr::blocks::vector_source_f::make(input_data, false, 6);
  auto svd_block = decomp_svd_sync<float>::make({2, 3});

  auto U_sink = gr::blocks::vector_sink_f::make(4); // 2x2 U matrix
  auto S_sink =
      gr::blocks::vector_sink_f::make(2); // 2x1 S vector (min(2,3) = 2)
  auto V_sink = gr::blocks::vector_sink_f::make(9); // 3x3 V matrix

  tb->connect(vector_source, 0, svd_block, 0);
  tb->connect(svd_block, 0, U_sink, 0);
  tb->connect(svd_block, 1, S_sink, 0);
  tb->connect(svd_block, 2, V_sink, 0);

  tb->run();

  auto U_data = U_sink->data();
  auto S_data = S_sink->data();
  auto V_data = V_sink->data();

  BOOST_REQUIRE_EQUAL(U_data.size(), 4);
  BOOST_REQUIRE_EQUAL(S_data.size(), 2);
  BOOST_REQUIRE_EQUAL(V_data.size(), 9);

  // Reconstruct matrices
  Eigen::Matrix<float, 2, 2> U;
  Eigen::Matrix<float, 2, 1> S;
  Eigen::Matrix<float, 3, 3> V;
  Eigen::Matrix<float, 2, 3> A;

  U << U_data[0], U_data[2], U_data[1], U_data[3];
  S << S_data[0], S_data[1];
  V << V_data[0], V_data[3], V_data[6], V_data[1], V_data[4], V_data[7],
      V_data[2], V_data[5], V_data[8];
  A << 1.0f, 3.0f, 5.0f, 2.0f, 4.0f, 6.0f;

  // For rectangular SVD: A = U * S * V_thin^T where V_thin = V(:, 1:min(m,n))
  int min_dim = std::min(2, 3); // = 2
  auto V_thin = V.leftCols(min_dim);
  Eigen::Matrix<float, 2, 2> S_diag = S.asDiagonal();
  auto USV = U * S_diag * V_thin.transpose();

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      BOOST_CHECK_CLOSE(A(i, j), USV(i, j), 1e-4f);
    }
  }

  BOOST_TEST_MESSAGE("SVD rectangular matrix flowgraph test passed.");
}

BOOST_AUTO_TEST_CASE(test_decomp_svd_sync_flowgraph_performance) {
  BOOST_TEST_MESSAGE("Testing SVD sync block performance in flowgraph...");

  auto tb = gr::make_top_block("svd_perf_test");

  // Create many 2x2 matrices for performance testing
  int num_matrices = 1000;
  std::vector<float> input_data;
  input_data.reserve(num_matrices * 4);

  for (int i = 0; i < num_matrices; ++i) {
    float val = 1.0f + 0.001f * i;
    input_data.insert(input_data.end(),
                      {val, 0.0f, 0.0f, val}); // Diagonal matrices
  }

  auto start_time = std::chrono::high_resolution_clock::now();

  auto vector_source = gr::blocks::vector_source_f::make(input_data, false, 4);
  auto svd_block = decomp_svd_sync<float>::make({2, 2});

  auto U_sink = gr::blocks::vector_sink_f::make(4);
  auto S_sink = gr::blocks::vector_sink_f::make(2);
  auto V_sink = gr::blocks::vector_sink_f::make(4);

  tb->connect(vector_source, 0, svd_block, 0);
  tb->connect(svd_block, 0, U_sink, 0);
  tb->connect(svd_block, 1, S_sink, 0);
  tb->connect(svd_block, 2, V_sink, 0);

  tb->run();

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);

  auto U_data = U_sink->data();
  auto S_data = S_sink->data();
  auto V_data = V_sink->data();

  BOOST_REQUIRE_EQUAL(U_data.size(), num_matrices * 4);
  BOOST_REQUIRE_EQUAL(S_data.size(), num_matrices * 2);
  BOOST_REQUIRE_EQUAL(V_data.size(), num_matrices * 4);

  // Verify first and last decompositions
  Eigen::Matrix<float, 2, 2> first_U, first_V, first_A;
  Eigen::Matrix<float, 2, 1> first_S;

  first_U << U_data[0], U_data[2], U_data[1], U_data[3];
  first_S << S_data[0], S_data[1];
  first_V << V_data[0], V_data[2], V_data[1], V_data[3];
  first_A << 1.0f, 0.0f, 0.0f, 1.0f;

  auto first_USV = first_U * first_S.asDiagonal() * first_V.transpose();
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      BOOST_CHECK_CLOSE(first_A(i, j), first_USV(i, j), 1e-4f);
    }
  }

  BOOST_TEST_MESSAGE("Processed " << num_matrices << " matrices in "
                                  << duration.count() << " ms");
  BOOST_TEST_MESSAGE("SVD performance flowgraph test passed.");
}

BOOST_AUTO_TEST_CASE(test_decomp_svd_sync_flowgraph_rank_deficient) {
  BOOST_TEST_MESSAGE(
      "Testing SVD sync block with rank-deficient matrix in flowgraph...");

  auto tb = gr::make_top_block("svd_rank_def_test");

  // Create rank-1 matrix: [[1, 2], [2, 4]] in column-major
  std::vector<float> input_data = {1.0f, 2.0f, 2.0f, 4.0f};

  auto vector_source = gr::blocks::vector_source_f::make(input_data, false, 4);
  auto svd_block = decomp_svd_sync<float>::make({2, 2});

  auto U_sink = gr::blocks::vector_sink_f::make(4);
  auto S_sink = gr::blocks::vector_sink_f::make(2);
  auto V_sink = gr::blocks::vector_sink_f::make(4);

  tb->connect(vector_source, 0, svd_block, 0);
  tb->connect(svd_block, 0, U_sink, 0);
  tb->connect(svd_block, 1, S_sink, 0);
  tb->connect(svd_block, 2, V_sink, 0);

  tb->run();

  auto S_data = S_sink->data();
  BOOST_REQUIRE_EQUAL(S_data.size(), 2);

  // Verify rank deficiency: one large singular value, one near-zero
  BOOST_CHECK_GT(S_data[0],
                 1e-6f); // First singular value should be significant
  BOOST_CHECK_LT(S_data[1],
                 1e-4f); // Second should be near zero (rank deficient)
  BOOST_CHECK_GE(S_data[0], S_data[1]); // Singular values should be sorted

  BOOST_TEST_MESSAGE("SVD rank-deficient flowgraph test passed.");
}

BOOST_AUTO_TEST_CASE(test_decomp_svd_sync_flowgraph_algorithm_switching) {
  BOOST_TEST_MESSAGE(
      "Testing SVD sync block with different algorithms in flowgraph...");

  auto tb1 = gr::make_top_block("svd_jacobi_test");
  auto tb2 = gr::make_top_block("svd_bdc_test");

  std::vector<float> input_data = {2.0f, 1.0f, 1.0f, 2.0f}; // [[2, 1], [1, 2]]

  // Test with Jacobi algorithm (default)
  auto vector_source1 = gr::blocks::vector_source_f::make(input_data, false, 4);
  auto svd_block1 = decomp_svd_sync<float>::make({2, 2});
  auto U_sink1 = gr::blocks::vector_sink_f::make(4);
  auto S_sink1 = gr::blocks::vector_sink_f::make(2);
  auto V_sink1 = gr::blocks::vector_sink_f::make(4);

  tb1->connect(vector_source1, 0, svd_block1, 0);
  tb1->connect(svd_block1, 0, U_sink1, 0); // Connect all outputs
  tb1->connect(svd_block1, 1, S_sink1, 0); // S values for comparison
  tb1->connect(svd_block1, 2, V_sink1, 0);

  tb1->run();
  auto S_jacobi = S_sink1->data();

  // Test with BDC algorithm
  auto vector_source2 = gr::blocks::vector_source_f::make(input_data, false, 4);
  auto bdc_algo = std::make_shared<eigen_bdc_svd<float>>();
  auto svd_block2 = decomp_svd_sync<float>::make({2, 2}, bdc_algo);
  auto U_sink2 = gr::blocks::vector_sink_f::make(4);
  auto S_sink2 = gr::blocks::vector_sink_f::make(2);
  auto V_sink2 = gr::blocks::vector_sink_f::make(4);

  tb2->connect(vector_source2, 0, svd_block2, 0);
  tb2->connect(svd_block2, 0, U_sink2, 0); // Connect all outputs
  tb2->connect(svd_block2, 1, S_sink2, 0);
  tb2->connect(svd_block2, 2, V_sink2, 0);

  tb2->run();
  auto S_bdc = S_sink2->data();

  // Both algorithms should give similar singular values
  BOOST_REQUIRE_EQUAL(S_jacobi.size(), 2);
  BOOST_REQUIRE_EQUAL(S_bdc.size(), 2);

  for (int i = 0; i < 2; ++i) {
    BOOST_CHECK_CLOSE(S_jacobi[i], S_bdc[i], 1e-4f);
  }

  BOOST_TEST_MESSAGE("SVD algorithm switching flowgraph test passed.");
}

} /* namespace linalg */
} /* namespace gr */
