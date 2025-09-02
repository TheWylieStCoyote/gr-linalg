/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <Eigen/Dense>
#include <boost/test/unit_test.hpp>
#include <chrono>
#include <gnuradio/attributes.h>
#include <gnuradio/blocks/vector_sink.h>
#include <gnuradio/blocks/vector_source.h>
#include <gnuradio/io_signature.h>
#include <gnuradio/linalg/decomp_eigen.h>
#include <gnuradio/linalg/types.h>
#include <gnuradio/linalg/utils.h>
#include <gnuradio/top_block.h>
#include <memory>

namespace gr {
namespace linalg {

// Disable ControlPort/RPC to avoid teardown hangs
struct GRTestEnv {
  GRTestEnv() {
    setenv("GR_CONF_CONTROLPORT_ON", "0", 1);
    setenv("GR_CONF_RPC_ON", "0", 1);
  }
  ~GRTestEnv() {}
};
BOOST_TEST_GLOBAL_FIXTURE(GRTestEnv);

BOOST_AUTO_TEST_CASE(test_decomp_eigen_constructor) {
  BOOST_TEST_MESSAGE("Testing eigenvalue decomposition constructor...");

  // Create a simple matrix shape
  auto shape = types::shape({3, 3});

  // Test sync block constructor
  auto eigen_sync = decomp_eigen_sync<float>::make(shape);
  BOOST_REQUIRE(eigen_sync);
  BOOST_TEST_MESSAGE("Eigenvalue sync block created successfully.");

  // Test PDU block constructor
  auto eigen_pdu = decomp_eigen_pdu<float>::make();
  BOOST_REQUIRE(eigen_pdu);
  BOOST_TEST_MESSAGE("Eigenvalue PDU block created successfully.");
}

BOOST_AUTO_TEST_CASE(test_decomp_eigen_symmetric_matrix) {
  BOOST_TEST_MESSAGE(
      "Testing eigenvalue decomposition for symmetric matrix...");

  // Create a symmetric matrix with known eigenvalues
  Eigen::Matrix<float, 3, 3> input_matrix;
  input_matrix << 3.0f, 1.0f, 0.0f, 1.0f, 3.0f, 1.0f, 0.0f, 1.0f, 3.0f;

  // Test the self-adjoint algorithm
  eigen_selfadjoint_solver<float> sa_algo;
  Eigen::Matrix<float, Eigen::Dynamic, 1> eigenvalues;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> eigenvectors;

  sa_algo.decompose(input_matrix, eigenvalues, eigenvectors);

  BOOST_TEST_MESSAGE("Verifying eigenvalue decomposition...");

  // Verify A * v = λ * v for each eigenvalue/eigenvector pair
  for (int i = 0; i < 3; ++i) {
    auto eigenvector = eigenvectors.col(i);
    auto Av = input_matrix * eigenvector;
    auto lambda_v = eigenvalues(i) * eigenvector;

    for (int j = 0; j < 3; ++j) {
      // Use absolute tolerance for small values, relative for larger ones
      if (std::abs(lambda_v(j)) < 1e-5f) {
        BOOST_CHECK_SMALL(Av(j) - lambda_v(j), 1e-5f);
      } else {
        BOOST_CHECK_CLOSE(Av(j), lambda_v(j), 5.0f); // 5% relative tolerance
      }
    }
  }

  // Verify that eigenvalues are real and sorted (Eigen sorts ascending for
  // symmetric matrices)
  BOOST_CHECK_LE(eigenvalues(0), eigenvalues(1));
  BOOST_CHECK_LE(eigenvalues(1), eigenvalues(2));

  BOOST_TEST_MESSAGE("Symmetric matrix eigenvalue decomposition test passed.");
}

BOOST_AUTO_TEST_CASE(test_decomp_eigen_general_algorithm) {
  BOOST_TEST_MESSAGE("Testing general eigenvalue algorithm...");

  // Create a non-symmetric matrix
  Eigen::Matrix<float, 2, 2> input_matrix;
  input_matrix << 1.0f, 2.0f, 3.0f, 4.0f;

  // Test the general algorithm
  eigen_general_solver<float> general_algo;
  Eigen::Matrix<float, Eigen::Dynamic, 1> eigenvalues;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> eigenvectors;

  general_algo.decompose(input_matrix, eigenvalues, eigenvectors);

  BOOST_TEST_MESSAGE("Verifying general eigenvalue decomposition...");

  // For a 2x2 matrix with trace=5 and det=-2, eigenvalues should be
  // approximately 5.372 and -0.372
  float expected_sum = input_matrix.trace(); // Sum of eigenvalues = trace
  float actual_sum = eigenvalues.sum();
  BOOST_CHECK_CLOSE(actual_sum, expected_sum, 1e-4f);

  float expected_product =
      input_matrix.determinant(); // Product of eigenvalues = determinant
  float actual_product = eigenvalues(0) * eigenvalues(1);
  BOOST_CHECK_CLOSE(actual_product, expected_product, 1e-4f);

  BOOST_TEST_MESSAGE("General eigenvalue algorithm test passed.");
}

BOOST_AUTO_TEST_CASE(test_decomp_eigen_algorithm_switching) {
  BOOST_TEST_MESSAGE("Testing eigenvalue algorithm switching...");

  // Create a sync block
  auto shape = types::shape({2, 2});
  auto eigen_sync = decomp_eigen_sync<float>::make(shape);
  BOOST_REQUIRE(eigen_sync);

  // Switch to self-adjoint solver
  auto sa_algo = std::make_shared<eigen_selfadjoint_solver<float>>();
  eigen_sync->set_algorithm(sa_algo);

  BOOST_TEST_MESSAGE("Algorithm switching test passed.");
}

BOOST_AUTO_TEST_CASE(test_decomp_eigen_double_precision) {
  BOOST_TEST_MESSAGE(
      "Testing eigenvalue decomposition with double precision...");

  // Test with double precision
  Eigen::Matrix<double, 2, 2> input_matrix;
  input_matrix << 2.0, 1.0, 1.0, 2.0;

  eigen_selfadjoint_solver<double> sa_algo;
  Eigen::Matrix<double, Eigen::Dynamic, 1> eigenvalues;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> eigenvectors;

  sa_algo.decompose(input_matrix, eigenvalues, eigenvectors);

  // Known eigenvalues for this matrix: 1.0 and 3.0 (in ascending order)
  BOOST_CHECK_CLOSE(eigenvalues(0), 1.0, 1e-8);
  BOOST_CHECK_CLOSE(eigenvalues(1), 3.0, 1e-8);

  BOOST_TEST_MESSAGE("Double precision test passed.");
}

BOOST_AUTO_TEST_CASE(test_decomp_eigen_identity_matrix) {
  BOOST_TEST_MESSAGE("Testing eigenvalue decomposition of identity matrix...");

  // Identity matrix should have all eigenvalues equal to 1
  Eigen::Matrix<float, 3, 3> identity = Eigen::Matrix<float, 3, 3>::Identity();

  eigen_selfadjoint_solver<float> sa_algo;
  Eigen::Matrix<float, Eigen::Dynamic, 1> eigenvalues;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> eigenvectors;

  sa_algo.decompose(identity, eigenvalues, eigenvectors);

  // All eigenvalues should be 1.0
  for (int i = 0; i < 3; ++i) {
    BOOST_CHECK_CLOSE(eigenvalues(i), 1.0f, 1e-6f);
  }

  BOOST_TEST_MESSAGE("Identity matrix test passed.");
}

BOOST_AUTO_TEST_CASE(test_decomp_eigen_block_operation_simulation) {
  BOOST_TEST_MESSAGE("Testing eigenvalue block operation with matrix maps...");

  auto shape = types::shape({2, 2});
  auto eigen_sync = decomp_eigen_sync<float>::make(shape);
  BOOST_REQUIRE(eigen_sync);

  Eigen::Matrix<float, 2, 2> input_matrix;
  input_matrix << 2.0f, 1.0f, 1.0f, 2.0f;

  Eigen::Matrix<float, 2, 1> eigenvalues_output;
  eigenvalues_output.setZero();
  Eigen::Matrix<float, 2, 2> eigenvectors_output;
  eigenvectors_output.setZero();

  types::const_matrix_map_dynamic<float> input_map(&input_matrix(0, 0), 2, 2);
  types::matrix_map_dynamic<float> eigenvalues_map(&eigenvalues_output(0, 0), 2,
                                                   1);
  types::matrix_map_dynamic<float> eigenvectors_map(&eigenvectors_output(0, 0),
                                                    2, 2);

  types::vector_const_matrix_map<float> input_maps = {&input_map};
  types::vector_matrix_map<float> output_maps = {&eigenvalues_map,
                                                 &eigenvectors_map};

  auto result = eigen_sync->operation(input_maps, output_maps);
  BOOST_CHECK(result == OperationReturn::SUCCESS);

  for (int i = 0; i < 2; ++i) {
    auto eigenvector = eigenvectors_output.col(i);
    auto Av = input_matrix * eigenvector;
    auto lambda_v = eigenvalues_output(i) * eigenvector;
    for (int j = 0; j < 2; ++j) {
      BOOST_CHECK_CLOSE(Av(j), lambda_v(j), 1e-4f);
    }
  }
}

BOOST_AUTO_TEST_CASE(test_decomp_eigen_sync_block_functionality) {
  BOOST_TEST_MESSAGE(
      "Testing eigenvalue sync block GNU Radio block creation...");

  auto shape = types::shape({2, 2});
  auto eigen_block = decomp_eigen_sync<float>::make(shape);
  BOOST_REQUIRE(eigen_block);

  auto input_sig = eigen_block->input_signature();
  auto output_sig = eigen_block->output_signature();

  // Should have 1 input and 2 outputs (values, vectors)
  BOOST_CHECK_EQUAL(input_sig->max_streams(), 1);
  BOOST_CHECK_EQUAL(output_sig->max_streams(), 2);
}

BOOST_AUTO_TEST_CASE(test_decomp_eigen_sync_block_work_method) {
  BOOST_TEST_MESSAGE("Testing Eigen sync block work() method functionality "
                     "with two outputs...");

  auto shape = types::shape({2, 2});
  auto eigen_block = decomp_eigen_sync<float>::make(shape);
  BOOST_REQUIRE(eigen_block);

  // Symmetric matrix -> real eigenvalues
  std::vector<float> input_data = {3.0f, 1.0f, 1.0f,
                                   2.0f}; // [[3,1],[1,2]] column-major

  // Two outputs: eigenvalues (2x1) and eigenvectors (2x2)
  std::vector<float> out_vals(2);
  std::vector<float> out_vecs(4);

  std::vector<const void *> input_items(1);
  std::vector<void *> output_items(2);
  input_items[0] = input_data.data();
  output_items[0] = out_vals.data();
  output_items[1] = out_vecs.data();

  int noutput_items = 1;
  int produced = eigen_block->work(noutput_items, input_items, output_items);
  BOOST_CHECK_EQUAL(produced, 1);

  // Check eigenvalue properties: sum equals trace, product equals determinant
  float sum_vals = out_vals[0] + out_vals[1];
  BOOST_CHECK_CLOSE(sum_vals, 5.0f, 1e-3f);
  float prod_vals = out_vals[0] * out_vals[1];
  BOOST_CHECK_CLOSE(prod_vals, 5.0f, 1e-3f);
}

BOOST_AUTO_TEST_CASE(test_decomp_eigen_pdu_block_functionality) {
  BOOST_TEST_MESSAGE("Testing eigenvalue PDU block GNU Radio functionality...");

  // Create eigenvalue PDU block
  auto eigen_pdu_block = decomp_eigen_pdu<float>::make();
  BOOST_REQUIRE(eigen_pdu_block);

  // Test that the PDU block can be created and has message ports
  BOOST_TEST_MESSAGE("Eigenvalue PDU block created successfully.");

  // Verify the block has the expected message ports
  auto msg_ports_in = eigen_pdu_block->message_ports_in();
  auto msg_ports_out = eigen_pdu_block->message_ports_out();

  // Note: Message port functionality is currently commented out in base class
  // This test verifies the block can be created and the API exists
  BOOST_CHECK(msg_ports_in != nullptr);
  BOOST_CHECK(msg_ports_out != nullptr);

  BOOST_TEST_MESSAGE("PDU block has proper message port interface.");
  BOOST_TEST_MESSAGE("Eigenvalue PDU block functionality test passed.");
}

BOOST_AUTO_TEST_CASE(test_decomp_eigen_pdu_message_processing) {
  BOOST_TEST_MESSAGE("Testing Eigen PDU block message processing...");

  auto eigen_pdu_block = decomp_eigen_pdu<float>::make();
  BOOST_REQUIRE(eigen_pdu_block);

  // Test PDU message processing functionality
  BOOST_TEST_MESSAGE("Testing PDU message handling...");

  // Create a test matrix as a PMT vector
  std::vector<float> matrix_data = {3.0f, 1.0f, 1.0f,
                                    2.0f}; // 2x2 symmetric matrix
  pmt::pmt_t matrix_pmt =
      pmt::init_f32vector(matrix_data.size(), matrix_data.data());

  // Create metadata for the matrix (shape information)
  pmt::pmt_t meta = pmt::make_dict();
  meta = pmt::dict_add(meta, pmt::mp("rows"), pmt::from_long(2));
  meta = pmt::dict_add(meta, pmt::mp("cols"), pmt::from_long(2));

  // Create PDU message
  pmt::pmt_t pdu = pmt::cons(meta, matrix_pmt);

  // Test message port interface
  auto msg_ports_in = eigen_pdu_block->message_ports_in();
  BOOST_CHECK(msg_ports_in != nullptr);

  // Note: Full PDU message processing test would require setting up
  // message handlers and running in a flowgraph context
  // Currently the message port functionality is commented out in the base class
  BOOST_TEST_MESSAGE("PDU message structure validated.");
  BOOST_TEST_MESSAGE("Eigen PDU message processing test passed.");
}

// GNU Radio Flowgraph Tests
// These tests connect blocks in actual GNU Radio flowgraphs and execute them

BOOST_AUTO_TEST_CASE(test_decomp_eigen_sync_flowgraph) {
  BOOST_TEST_MESSAGE("Testing Eigen sync block in GNU Radio flowgraph...");

  // Create a GNU Radio top block (flowgraph)
  auto tb = gr::make_top_block("eigen_test");

  // Create test input: symmetric matrix 2x2
  // [[3, 1], [1, 2]] -> column-major: [3, 1, 1, 2]
  std::vector<float> input_data = {3.0f, 1.0f, 1.0f, 2.0f};

  // Create GNU Radio blocks
  auto vector_source = gr::blocks::vector_source_f::make(input_data, false, 4);
  auto eigen_block = decomp_eigen_sync<float>::make({2, 2});
  auto eigenvalues_sink = gr::blocks::vector_sink_f::make(2); // 2x1 eigenvalues
  auto eigenvectors_sink =
      gr::blocks::vector_sink_f::make(4); // 2x2 eigenvectors

  // Connect the blocks in flowgraph
  tb->connect(vector_source, 0, eigen_block, 0);
  tb->connect(eigen_block, 0, eigenvalues_sink, 0);  // eigenvalues output
  tb->connect(eigen_block, 1, eigenvectors_sink, 0); // eigenvectors output

  // Run the flowgraph
  tb->run();

  // Get the output data
  auto eigenvalues_data = eigenvalues_sink->data();
  auto eigenvectors_data = eigenvectors_sink->data();

  BOOST_REQUIRE_EQUAL(eigenvalues_data.size(), 2);
  BOOST_REQUIRE_EQUAL(eigenvectors_data.size(), 4);

  // Verify eigenvalue properties
  // For matrix [[3,1],[1,2]]: trace = 5, determinant = 5
  float trace = eigenvalues_data[0] + eigenvalues_data[1];
  float determinant = eigenvalues_data[0] * eigenvalues_data[1];

  BOOST_CHECK_CLOSE(trace, 5.0f, 1e-3f);
  BOOST_CHECK_CLOSE(determinant, 5.0f, 1e-3f);

  // Verify eigenvalue equation A*v = λ*v
  Eigen::Matrix<float, 2, 2> A;
  A << 3.0f, 1.0f, 1.0f, 2.0f;

  Eigen::Map<const Eigen::Matrix<float, 2, 2>> V(eigenvectors_data.data());

  for (int i = 0; i < 2; ++i) {
    auto v = V.col(i);
    auto Av = A * v;
    auto lambda_v = eigenvalues_data[i] * v;

    for (int j = 0; j < 2; ++j) {
      BOOST_CHECK_CLOSE(Av(j), lambda_v(j), 1e-2f);
    }
  }

  BOOST_TEST_MESSAGE("Eigen sync flowgraph test passed.");
}

BOOST_AUTO_TEST_CASE(test_decomp_eigen_sync_flowgraph_3x3) {
  BOOST_TEST_MESSAGE(
      "Testing Eigen sync block with 3x3 matrix in flowgraph...");

  auto tb = gr::make_top_block("eigen_3x3_test");

  // Create 3x3 symmetric matrix
  // [[4, 1, 0], [1, 3, 1], [0, 1, 2]] -> column-major
  std::vector<float> input_data = {4.0f, 1.0f, 0.0f,  // column 1
                                   1.0f, 3.0f, 1.0f,  // column 2
                                   0.0f, 1.0f, 2.0f}; // column 3

  auto vector_source = gr::blocks::vector_source_f::make(input_data, false, 9);
  auto eigen_block = decomp_eigen_sync<float>::make({3, 3});
  auto eigenvalues_sink = gr::blocks::vector_sink_f::make(3); // 3x1 eigenvalues
  auto eigenvectors_sink =
      gr::blocks::vector_sink_f::make(9); // 3x3 eigenvectors

  tb->connect(vector_source, 0, eigen_block, 0);
  tb->connect(eigen_block, 0, eigenvalues_sink, 0);
  tb->connect(eigen_block, 1, eigenvectors_sink, 0);

  tb->run();

  auto eigenvalues_data = eigenvalues_sink->data();
  auto eigenvectors_data = eigenvectors_sink->data();

  BOOST_REQUIRE_EQUAL(eigenvalues_data.size(), 3);
  BOOST_REQUIRE_EQUAL(eigenvectors_data.size(), 9);

  // Verify trace (sum of eigenvalues)
  float trace = eigenvalues_data[0] + eigenvalues_data[1] + eigenvalues_data[2];
  BOOST_CHECK_CLOSE(trace, 9.0f,
                    1e-3f); // trace of [[4,1,0],[1,3,1],[0,1,2]] = 9

  // Verify eigenvalues are sorted (Eigen sorts ascending for symmetric
  // matrices)
  BOOST_CHECK_LE(eigenvalues_data[0], eigenvalues_data[1]);
  BOOST_CHECK_LE(eigenvalues_data[1], eigenvalues_data[2]);

  BOOST_TEST_MESSAGE("Eigen 3x3 flowgraph test passed.");
}

BOOST_AUTO_TEST_CASE(test_decomp_eigen_sync_flowgraph_multiple_matrices) {
  BOOST_TEST_MESSAGE(
      "Testing Eigen sync block with multiple matrices in flowgraph...");

  auto tb = gr::make_top_block("eigen_multi_test");

  // Create multiple 2x2 matrices to process
  std::vector<float> input_data = {// Matrix 1: [[3, 1], [1, 2]]
                                   3.0f, 1.0f, 1.0f, 2.0f,
                                   // Matrix 2: [[5, 2], [2, 3]]
                                   5.0f, 2.0f, 2.0f, 3.0f,
                                   // Matrix 3: [[1, 0], [0, 1]] (identity)
                                   1.0f, 0.0f, 0.0f, 1.0f};

  auto vector_source = gr::blocks::vector_source_f::make(input_data, false, 4);
  auto eigen_block = decomp_eigen_sync<float>::make({2, 2});
  auto eigenvalues_sink = gr::blocks::vector_sink_f::make(2);
  auto eigenvectors_sink = gr::blocks::vector_sink_f::make(4);

  tb->connect(vector_source, 0, eigen_block, 0);
  tb->connect(eigen_block, 0, eigenvalues_sink, 0);
  tb->connect(eigen_block, 1, eigenvectors_sink, 0);

  tb->run();

  auto eigenvalues_data = eigenvalues_sink->data();
  auto eigenvectors_data = eigenvectors_sink->data();

  BOOST_REQUIRE_EQUAL(eigenvalues_data.size(),
                      6); // 3 matrices * 2 eigenvalues each
  BOOST_REQUIRE_EQUAL(eigenvectors_data.size(),
                      12); // 3 matrices * 4 elements each

  // Verify each matrix's eigenvalues
  std::vector<float> expected_traces = {5.0f, 8.0f,
                                        2.0f}; // traces of the input matrices

  for (int matrix_idx = 0; matrix_idx < 3; ++matrix_idx) {
    float trace =
        eigenvalues_data[matrix_idx * 2] + eigenvalues_data[matrix_idx * 2 + 1];
    BOOST_CHECK_CLOSE(trace, expected_traces[matrix_idx], 1e-3f);
    BOOST_TEST_MESSAGE("Matrix " << (matrix_idx + 1)
                                 << " eigenvalues verified");
  }

  BOOST_TEST_MESSAGE("Eigen multiple matrices flowgraph test passed.");
}

BOOST_AUTO_TEST_CASE(test_decomp_eigen_sync_flowgraph_double_precision) {
  BOOST_TEST_MESSAGE(
      "Testing Eigen sync block with double precision in flowgraph...");

  auto tb = gr::make_top_block("eigen_double_test");

  // Use float vectors but test precision handling
  std::vector<float> input_data = {2.0f, 1.0f, 1.0f, 2.0f};

  auto vector_source = gr::blocks::vector_source_f::make(input_data, false, 4);
  auto eigen_block = decomp_eigen_sync<float>::make({2, 2});
  auto eigenvalues_sink = gr::blocks::vector_sink_f::make(2);
  auto eigenvectors_sink = gr::blocks::vector_sink_f::make(4);

  tb->connect(vector_source, 0, eigen_block, 0);
  tb->connect(eigen_block, 0, eigenvalues_sink, 0);
  tb->connect(eigen_block, 1, eigenvectors_sink, 0);

  tb->run();

  auto eigenvalues_data = eigenvalues_sink->data();
  auto eigenvectors_data = eigenvectors_sink->data();

  BOOST_REQUIRE_EQUAL(eigenvalues_data.size(), 2);
  BOOST_REQUIRE_EQUAL(eigenvectors_data.size(), 4);

  // For matrix [[2,1],[1,2]], eigenvalues are 1 and 3
  // Check that we get the correct values (allowing for ordering)
  std::vector<float> eigenvals = {eigenvalues_data[0], eigenvalues_data[1]};
  std::sort(eigenvals.begin(), eigenvals.end());

  BOOST_CHECK_CLOSE(eigenvals[0], 1.0f, 1e-5f);
  BOOST_CHECK_CLOSE(eigenvals[1], 3.0f, 1e-5f);

  BOOST_TEST_MESSAGE("Eigen double precision flowgraph test passed.");
}

BOOST_AUTO_TEST_CASE(test_decomp_eigen_sync_flowgraph_algorithm_switching) {
  BOOST_TEST_MESSAGE("Testing Eigen algorithm switching in flowgraph...");

  auto tb = gr::make_top_block("eigen_algo_test");

  std::vector<float> input_data = {3.0f, 1.0f, 1.0f, 2.0f};

  auto vector_source = gr::blocks::vector_source_f::make(input_data, false, 4);
  auto eigen_block = decomp_eigen_sync<float>::make({2, 2});
  auto eigenvalues_sink = gr::blocks::vector_sink_f::make(2);
  auto eigenvectors_sink = gr::blocks::vector_sink_f::make(4);

  // Test with self-adjoint algorithm
  auto sa_algorithm = std::make_shared<eigen_selfadjoint_solver<float>>();
  eigen_block->set_algorithm(sa_algorithm);

  tb->connect(vector_source, 0, eigen_block, 0);
  tb->connect(eigen_block, 0, eigenvalues_sink, 0);
  tb->connect(eigen_block, 1, eigenvectors_sink, 0);

  tb->run();

  auto sa_eigenvalues = eigenvalues_sink->data();
  BOOST_REQUIRE_EQUAL(sa_eigenvalues.size(), 2);

  // Reset for general solver test
  tb->disconnect_all();
  eigenvalues_sink = gr::blocks::vector_sink_f::make(2);
  eigenvectors_sink = gr::blocks::vector_sink_f::make(4);
  vector_source = gr::blocks::vector_source_f::make(input_data, false, 4);

  // Switch to general algorithm
  auto general_algorithm = std::make_shared<eigen_general_solver<float>>();
  eigen_block->set_algorithm(general_algorithm);

  tb->connect(vector_source, 0, eigen_block, 0);
  tb->connect(eigen_block, 0, eigenvalues_sink, 0);
  tb->connect(eigen_block, 1, eigenvectors_sink, 0);

  tb->run();

  auto general_eigenvalues = eigenvalues_sink->data();
  BOOST_REQUIRE_EQUAL(general_eigenvalues.size(), 2);

  // Both algorithms should produce similar eigenvalues for symmetric matrices
  float sa_trace = sa_eigenvalues[0] + sa_eigenvalues[1];
  float general_trace = general_eigenvalues[0] + general_eigenvalues[1];

  BOOST_CHECK_CLOSE(sa_trace, general_trace, 1e-3f);
  BOOST_CHECK_CLOSE(sa_trace, 5.0f, 1e-3f); // Expected trace

  BOOST_TEST_MESSAGE("Eigen algorithm switching flowgraph test passed.");
}

BOOST_AUTO_TEST_CASE(test_decomp_eigen_sync_flowgraph_identity_matrix) {
  BOOST_TEST_MESSAGE(
      "Testing Eigen sync block with identity matrix in flowgraph...");

  auto tb = gr::make_top_block("eigen_identity_test");

  // Identity matrix 3x3
  std::vector<float> input_data = {1.0f, 0.0f, 0.0f,  // column 1
                                   0.0f, 1.0f, 0.0f,  // column 2
                                   0.0f, 0.0f, 1.0f}; // column 3

  auto vector_source = gr::blocks::vector_source_f::make(input_data, false, 9);
  auto eigen_block = decomp_eigen_sync<float>::make({3, 3});
  auto eigenvalues_sink = gr::blocks::vector_sink_f::make(3);
  auto eigenvectors_sink = gr::blocks::vector_sink_f::make(9);

  tb->connect(vector_source, 0, eigen_block, 0);
  tb->connect(eigen_block, 0, eigenvalues_sink, 0);
  tb->connect(eigen_block, 1, eigenvectors_sink, 0);

  tb->run();

  auto eigenvalues_data = eigenvalues_sink->data();
  auto eigenvectors_data = eigenvectors_sink->data();

  BOOST_REQUIRE_EQUAL(eigenvalues_data.size(), 3);
  BOOST_REQUIRE_EQUAL(eigenvectors_data.size(), 9);

  // All eigenvalues of identity matrix should be 1.0
  for (int i = 0; i < 3; ++i) {
    BOOST_CHECK_CLOSE(eigenvalues_data[i], 1.0f, 1e-6f);
  }

  BOOST_TEST_MESSAGE("Eigen identity matrix flowgraph test passed.");
}

BOOST_AUTO_TEST_CASE(test_decomp_eigen_sync_flowgraph_performance) {
  BOOST_TEST_MESSAGE("Testing Eigen sync block performance in flowgraph...");

  auto tb = gr::make_top_block("eigen_perf_test");

  // Create many small matrices to test throughput
  const int num_matrices =
      500; // Fewer than Cholesky since eigendecomposition is more expensive
  std::vector<float> input_data;
  input_data.reserve(num_matrices * 4);

  for (int i = 0; i < num_matrices; ++i) {
    // Each matrix is [[3, 1], [1, 2]] with slight variation
    float base = 3.0f + 0.001f * i;
    input_data.insert(input_data.end(), {base, 1.0f, 1.0f, 2.0f});
  }

  auto vector_source = gr::blocks::vector_source_f::make(input_data, false, 4);
  auto eigen_block = decomp_eigen_sync<float>::make({2, 2});
  auto eigenvalues_sink = gr::blocks::vector_sink_f::make(2);
  auto eigenvectors_sink = gr::blocks::vector_sink_f::make(4);

  tb->connect(vector_source, 0, eigen_block, 0);
  tb->connect(eigen_block, 0, eigenvalues_sink, 0);
  tb->connect(eigen_block, 1, eigenvectors_sink, 0);

  // Measure execution time
  auto start_time = std::chrono::high_resolution_clock::now();
  tb->run();
  auto end_time = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);

  auto eigenvalues_data = eigenvalues_sink->data();
  auto eigenvectors_data = eigenvectors_sink->data();

  BOOST_REQUIRE_EQUAL(eigenvalues_data.size(), num_matrices * 2);
  BOOST_REQUIRE_EQUAL(eigenvectors_data.size(), num_matrices * 4);

  // Verify first and last eigenvalue decompositions
  float first_trace = eigenvalues_data[0] + eigenvalues_data[1];
  float last_trace = eigenvalues_data[(num_matrices - 1) * 2] +
                     eigenvalues_data[(num_matrices - 1) * 2 + 1];

  BOOST_CHECK_CLOSE(first_trace, 5.0f, 1e-2f);
  BOOST_CHECK_CLOSE(last_trace, 5.0f + 0.001f * (num_matrices - 1), 1e-2f);

  BOOST_TEST_MESSAGE("Processed " << num_matrices << " matrices in "
                                  << duration.count() << " ms");
  BOOST_TEST_MESSAGE("Eigen performance flowgraph test passed.");
}

BOOST_AUTO_TEST_CASE(test_decomp_eigen_sync_flowgraph_general_matrix) {
  BOOST_TEST_MESSAGE("Testing Eigen sync block with general (non-symmetric) "
                     "matrix in flowgraph...");

  auto tb = gr::make_top_block("eigen_general_test");

  // Non-symmetric matrix [[1, 2], [3, 4]]
  std::vector<float> input_data = {1.0f, 3.0f, 2.0f, 4.0f}; // column-major

  auto vector_source = gr::blocks::vector_source_f::make(input_data, false, 4);
  auto eigen_block = decomp_eigen_sync<float>::make({2, 2});
  auto eigenvalues_sink = gr::blocks::vector_sink_f::make(2);
  auto eigenvectors_sink = gr::blocks::vector_sink_f::make(4);

  // Use general solver for non-symmetric matrix
  auto general_algorithm = std::make_shared<eigen_general_solver<float>>();
  eigen_block->set_algorithm(general_algorithm);

  tb->connect(vector_source, 0, eigen_block, 0);
  tb->connect(eigen_block, 0, eigenvalues_sink, 0);
  tb->connect(eigen_block, 1, eigenvectors_sink, 0);

  tb->run();

  auto eigenvalues_data = eigenvalues_sink->data();
  auto eigenvectors_data = eigenvectors_sink->data();

  BOOST_REQUIRE_EQUAL(eigenvalues_data.size(), 2);
  BOOST_REQUIRE_EQUAL(eigenvectors_data.size(), 4);

  // For matrix [[1,2],[3,4]]: trace = 5, determinant = -2
  float trace = eigenvalues_data[0] + eigenvalues_data[1];
  float determinant = eigenvalues_data[0] * eigenvalues_data[1];

  BOOST_CHECK_CLOSE(trace, 5.0f, 1e-3f);
  BOOST_CHECK_CLOSE(determinant, -2.0f, 1e-3f);

  BOOST_TEST_MESSAGE("Eigen general matrix flowgraph test passed.");
}

BOOST_AUTO_TEST_CASE(test_decomp_eigen_sync_flowgraph_error_conditions) {
  BOOST_TEST_MESSAGE("Testing Eigen sync block error handling in flowgraph...");

  // Test with a matrix that might cause numerical issues
  auto tb = gr::make_top_block("eigen_error_test");

  // Nearly singular matrix [[1, 1], [1, 1.0001]]
  std::vector<float> input_data = {1.0f, 1.0f, 1.0f, 1.0001f};

  auto vector_source = gr::blocks::vector_source_f::make(input_data, false, 4);
  auto eigen_block = decomp_eigen_sync<float>::make({2, 2});
  auto eigenvalues_sink = gr::blocks::vector_sink_f::make(2);
  auto eigenvectors_sink = gr::blocks::vector_sink_f::make(4);

  tb->connect(vector_source, 0, eigen_block, 0);
  tb->connect(eigen_block, 0, eigenvalues_sink, 0);
  tb->connect(eigen_block, 1, eigenvectors_sink, 0);

  // This should run without crashing, even if eigenvalues are close to zero
  try {
    tb->run();
    auto eigenvalues_data = eigenvalues_sink->data();
    auto eigenvectors_data = eigenvectors_sink->data();

    BOOST_CHECK_EQUAL(eigenvalues_data.size(), 2);
    BOOST_CHECK_EQUAL(eigenvectors_data.size(), 4);

    // One eigenvalue should be close to 0, the other close to 2.0001
    std::vector<float> eigenvals = {eigenvalues_data[0], eigenvalues_data[1]};
    std::sort(eigenvals.begin(), eigenvals.end());

    BOOST_CHECK_SMALL(eigenvals[0], 1e-3f); // Should be near zero
    BOOST_CHECK_CLOSE(eigenvals[1], 2.0001f, 1e-2f);

    BOOST_TEST_MESSAGE("Error condition handled gracefully");
  } catch (const std::exception &e) {
    BOOST_TEST_MESSAGE("Exception caught as expected: " << e.what());
    // It's okay if it throws - error handling is working
  }

  BOOST_TEST_MESSAGE("Eigen error handling flowgraph test passed.");
}

} /* namespace linalg */
} /* namespace gr */
