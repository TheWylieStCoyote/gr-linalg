/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <Eigen/Dense>
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <complex>
#include <gnuradio/linalg/decomp_hessenberg.h>

namespace gr {
namespace linalg {

BOOST_AUTO_TEST_CASE(test_decomp_hessenberg_basic_creation) {
  // Test basic block creation with default options
  auto hess_block = decomp_hessenberg_sync_f::make({2, 2}, true);

  BOOST_REQUIRE(hess_block != nullptr);

  // Check basic parameters
  BOOST_CHECK(hess_block->get_compute_q() == true);

  // Check default options
  const auto &options = hess_block->get_options();
  BOOST_CHECK(options.balance_matrix == false);
  BOOST_CHECK(options.extract_reflectors == false);
  BOOST_CHECK(options.packed_storage == false);
  BOOST_CHECK(options.compute_condition == false);
  BOOST_CHECK_CLOSE(options.balance_tolerance, 1e-8, 1e-10);
}

BOOST_AUTO_TEST_CASE(test_decomp_hessenberg_advanced_options) {
  // Test block creation with advanced options
  hessenberg_options options;
  options.balance_matrix = true;
  options.extract_reflectors = true;
  options.packed_storage = true;
  options.compute_condition = true;
  options.balance_tolerance = 1e-12;

  auto hess_block =
      decomp_hessenberg_sync_d::make({3, 3}, true, nullptr, options);

  BOOST_REQUIRE(hess_block != nullptr);

  // Check advanced options were set
  const auto &block_options = hess_block->get_options();
  BOOST_CHECK(block_options.balance_matrix == true);
  BOOST_CHECK(block_options.extract_reflectors == true);
  BOOST_CHECK(block_options.packed_storage == true);
  BOOST_CHECK(block_options.compute_condition == true);
  BOOST_CHECK_CLOSE(block_options.balance_tolerance, 1e-12, 1e-14);
}

BOOST_AUTO_TEST_CASE(test_decomp_hessenberg_io_signatures) {
  // Test input/output signatures for different matrix sizes
  auto hess_2x2 = decomp_hessenberg_sync_f::make({2, 2});
  auto hess_3x3 = decomp_hessenberg_sync_d::make({3, 3});

  // Check 2x2 signatures
  auto input_sig_2x2 = hess_2x2->input_signature();
  auto output_sig_2x2 = hess_2x2->output_signature();

  BOOST_CHECK_EQUAL(input_sig_2x2->max_streams(), 1);
  BOOST_CHECK_EQUAL(output_sig_2x2->max_streams(), 2); // Q and H matrices
  BOOST_CHECK_EQUAL(input_sig_2x2->sizeof_stream_item(0), 4 * sizeof(float));
  BOOST_CHECK_EQUAL(output_sig_2x2->sizeof_stream_item(0),
                    4 * sizeof(float)); // Q matrix
  BOOST_CHECK_EQUAL(output_sig_2x2->sizeof_stream_item(1),
                    4 * sizeof(float)); // H matrix

  // Check 3x3 signatures
  auto input_sig_3x3 = hess_3x3->input_signature();
  auto output_sig_3x3 = hess_3x3->output_signature();

  BOOST_CHECK_EQUAL(input_sig_3x3->max_streams(), 1);
  BOOST_CHECK_EQUAL(output_sig_3x3->max_streams(), 2);
  BOOST_CHECK_EQUAL(input_sig_3x3->sizeof_stream_item(0), 9 * sizeof(double));
  BOOST_CHECK_EQUAL(output_sig_3x3->sizeof_stream_item(0), 9 * sizeof(double));
  BOOST_CHECK_EQUAL(output_sig_3x3->sizeof_stream_item(1), 9 * sizeof(double));
}

BOOST_AUTO_TEST_CASE(test_decomp_hessenberg_compute_q_option) {
  // Test with compute_q = false (don't compute Q matrix)
  auto hess_no_q = decomp_hessenberg_sync_f::make({2, 2}, false);

  BOOST_REQUIRE(hess_no_q != nullptr);
  BOOST_CHECK(hess_no_q->get_compute_q() == false);

  // Output should be 1 stream (only H matrix)
  auto output_sig = hess_no_q->output_signature();
  BOOST_CHECK_EQUAL(output_sig->max_streams(), 1);
}

BOOST_AUTO_TEST_CASE(test_decomp_hessenberg_complex_variants) {
  // Test complex float variant
  auto hess_cf = decomp_hessenberg_sync_cf::make({2, 2});
  BOOST_REQUIRE(hess_cf != nullptr);

  auto input_sig_cf = hess_cf->input_signature();
  auto output_sig_cf = hess_cf->output_signature();

  BOOST_CHECK_EQUAL(input_sig_cf->sizeof_stream_item(0),
                    4 * sizeof(std::complex<float>));
  BOOST_CHECK_EQUAL(output_sig_cf->sizeof_stream_item(0),
                    4 * sizeof(std::complex<float>));
  BOOST_CHECK_EQUAL(output_sig_cf->sizeof_stream_item(1),
                    4 * sizeof(std::complex<float>));

  // Test complex double variant
  auto hess_cd = decomp_hessenberg_sync_cd::make({3, 3});
  BOOST_REQUIRE(hess_cd != nullptr);

  auto input_sig_cd = hess_cd->input_signature();
  auto output_sig_cd = hess_cd->output_signature();

  BOOST_CHECK_EQUAL(input_sig_cd->sizeof_stream_item(0),
                    9 * sizeof(std::complex<double>));
  BOOST_CHECK_EQUAL(output_sig_cd->sizeof_stream_item(0),
                    9 * sizeof(std::complex<double>));
  BOOST_CHECK_EQUAL(output_sig_cd->sizeof_stream_item(1),
                    9 * sizeof(std::complex<double>));
}

BOOST_AUTO_TEST_CASE(test_decomp_hessenberg_pdu_variants) {
  // Test PDU variants
  auto pdu_f = decomp_hessenberg_pdu_f::make({2, 2}, true);
  auto pdu_d = decomp_hessenberg_pdu_d::make({3, 3}, false);
  auto pdu_cf = decomp_hessenberg_pdu_cf::make({2, 2}, true);
  auto pdu_cd = decomp_hessenberg_pdu_cd::make({4, 4}, false);

  BOOST_CHECK(pdu_f != nullptr);
  BOOST_CHECK(pdu_d != nullptr);
  BOOST_CHECK(pdu_cf != nullptr);
  BOOST_CHECK(pdu_cd != nullptr);

  // Test getter/setter methods
  BOOST_CHECK(pdu_f->get_compute_q() == true);
  BOOST_CHECK(pdu_d->get_compute_q() == false);

  pdu_f->set_compute_q(false);
  pdu_d->set_compute_q(true);

  BOOST_CHECK(pdu_f->get_compute_q() == false);
  BOOST_CHECK(pdu_d->get_compute_q() == true);
}

BOOST_AUTO_TEST_CASE(test_decomp_hessenberg_advanced_feature_setters) {
  auto hess_block = decomp_hessenberg_sync_f::make({2, 2});

  // Test advanced feature setters
  hess_block->set_balance_matrix(true);
  hess_block->set_extract_reflectors(true);
  hess_block->set_packed_storage(true);
  hess_block->set_compute_condition(true);
  hess_block->set_balance_tolerance(1e-14);

  const auto &options = hess_block->get_options();
  BOOST_CHECK(options.balance_matrix == true);
  BOOST_CHECK(options.extract_reflectors == true);
  BOOST_CHECK(options.packed_storage == true);
  BOOST_CHECK(options.compute_condition == true);
  BOOST_CHECK_CLOSE(options.balance_tolerance, 1e-14, 1e-16);
}

BOOST_AUTO_TEST_CASE(test_decomp_hessenberg_shape_validation) {
  // Test valid square matrix shapes
  BOOST_CHECK_NO_THROW(decomp_hessenberg_sync_f::make({2, 2}));
  BOOST_CHECK_NO_THROW(decomp_hessenberg_sync_d::make({3, 3}));
  BOOST_CHECK_NO_THROW(decomp_hessenberg_sync_cf::make({4, 4}));
  BOOST_CHECK_NO_THROW(decomp_hessenberg_sync_cd::make({5, 5}));

  // Non-square matrices should be rejected
  BOOST_CHECK_THROW(decomp_hessenberg_sync_f::make({2, 3}), std::exception);
  BOOST_CHECK_THROW(decomp_hessenberg_sync_d::make({3, 2}), std::exception);
  BOOST_CHECK_THROW(decomp_hessenberg_sync_cf::make({4, 3}), std::exception);
  BOOST_CHECK_THROW(decomp_hessenberg_sync_cd::make({2, 5}), std::exception);
}

BOOST_AUTO_TEST_CASE(test_decomp_hessenberg_compute_output_shapes) {
  // Test static shape computation
  types::vector_shapes input_shapes = {{3, 3}};
  auto output_shapes =
      decomp_hessenberg<float>::compute_output_shapes(input_shapes);

  BOOST_REQUIRE_EQUAL(output_shapes.size(), 2); // Q and H matrices
  BOOST_REQUIRE_EQUAL(output_shapes[0].size(), 2);
  BOOST_REQUIRE_EQUAL(output_shapes[1].size(), 2);

  // Both Q and H should have same shape as input
  BOOST_CHECK_EQUAL(output_shapes[0][0], 3);
  BOOST_CHECK_EQUAL(output_shapes[0][1], 3);
  BOOST_CHECK_EQUAL(output_shapes[1][0], 3);
  BOOST_CHECK_EQUAL(output_shapes[1][1], 3);

  // Test invalid input shapes
  BOOST_CHECK_THROW(decomp_hessenberg<double>::compute_output_shapes({{2, 3}}),
                    std::invalid_argument);
  BOOST_CHECK_THROW(
      decomp_hessenberg<float>::compute_output_shapes({{2, 2}, {3, 3}}),
      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_decomp_hessenberg_algorithm_interface) {
  // Test that we can create blocks with custom algorithms
  auto eigen_algorithm = std::make_shared<eigen_hessenberg<float>>();
  auto eigen_advanced_algorithm =
      std::make_shared<eigen_hessenberg_advanced<double>>();

  // Create blocks with custom algorithms
  auto hess_standard =
      decomp_hessenberg_sync_f::make({2, 2}, true, eigen_algorithm);
  auto hess_advanced =
      decomp_hessenberg_sync_d::make({3, 3}, true, eigen_advanced_algorithm);

  BOOST_CHECK(hess_standard != nullptr);
  BOOST_CHECK(hess_advanced != nullptr);
}

BOOST_AUTO_TEST_CASE(test_decomp_hessenberg_operation_interface) {
  // Test that the operation interface is working
  hessenberg_options options;
  options.extract_reflectors = true;
  options.compute_condition = true;

  auto hess_impl = decomp_hessenberg<float>("test_hessenberg", {2, 2}, true,
                                            nullptr, options);

  // Create test matrices
  std::vector<float> input_data = {2.0f, 1.0f, 1.0f, 2.0f}; // [[2, 1], [1, 2]]
  std::vector<float> output_q(4);
  std::vector<float> output_h(4);

  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      input_matrix(input_data.data(), 2, 2);
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> q_matrix(
      output_q.data(), 2, 2);
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> h_matrix(
      output_h.data(), 2, 2);

  types::vector_const_matrix_map<float> input_matrices = {&input_matrix};
  types::vector_matrix_map<float> output_matrices = {&q_matrix, &h_matrix};

  // Test operation call
  auto result = hess_impl.operation(input_matrices, output_matrices);

  BOOST_CHECK(result == OperationReturn::SUCCESS);

  // Check that we have advanced results available
  const auto &last_result = hess_impl.get_last_result();
  // For a 2x2 matrix, we expect some basic structure
  BOOST_CHECK(last_result.Q.rows() == 2);
  BOOST_CHECK(last_result.Q.cols() == 2);
  BOOST_CHECK(last_result.H.rows() == 2);
  BOOST_CHECK(last_result.H.cols() == 2);
}

BOOST_AUTO_TEST_CASE(test_decomp_hessenberg_result_accessors) {
  // Create a block with advanced options to test result accessors
  hessenberg_options options;
  options.balance_matrix = true;
  options.extract_reflectors = true;
  options.compute_condition = true;

  auto hess_block =
      decomp_hessenberg_sync_f::make({2, 2}, true, nullptr, options);
  BOOST_REQUIRE(hess_block != nullptr);

  // Test that accessors don't crash (results will be empty until operation is
  // called)
  BOOST_CHECK_NO_THROW(hess_block->get_condition_estimate());
  BOOST_CHECK_NO_THROW(hess_block->was_matrix_balanced());
  BOOST_CHECK_NO_THROW(hess_block->get_permutation());
  BOOST_CHECK_NO_THROW(hess_block->get_scale_factors());
  BOOST_CHECK_NO_THROW(hess_block->get_householder_reflectors());
  BOOST_CHECK_NO_THROW(hess_block->get_last_result());
}

BOOST_AUTO_TEST_CASE(test_decomp_hessenberg_mathematical_properties) {
  // Test basic mathematical properties of Hessenberg decomposition
  hessenberg_options options;
  options.compute_condition = true;

  auto hess_impl = decomp_hessenberg<double>("test_math_hessenberg", {3, 3},
                                             true, nullptr, options);

  // Create a symmetric test matrix (will have nice properties)
  std::vector<double> input_data = {
      2.0, 1.0, 0.0, // Column 1
      1.0, 2.0, 1.0, // Column 2
      0.0, 1.0, 2.0  // Column 3
  };
  std::vector<double> output_q(9);
  std::vector<double> output_h(9);

  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
      input_matrix(input_data.data(), 3, 3);
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> q_matrix(
      output_q.data(), 3, 3);
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> h_matrix(
      output_h.data(), 3, 3);

  types::vector_const_matrix_map<double> input_matrices = {&input_matrix};
  types::vector_matrix_map<double> output_matrices = {&q_matrix, &h_matrix};

  // Test operation call
  auto result = hess_impl.operation(input_matrices, output_matrices);
  BOOST_CHECK(result == OperationReturn::SUCCESS);

  // Verify H matrix is upper Hessenberg (zero below first subdiagonal)
  // H(i,j) should be zero for i > j+1
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (i > j + 1) {
        BOOST_CHECK_SMALL(std::abs(h_matrix(i, j)), 1e-10);
      }
    }
  }

  // Verify Q is orthogonal: Q^T * Q should be identity
  Eigen::Matrix<double, 3, 3> qtq = q_matrix.transpose() * q_matrix;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (i == j) {
        BOOST_CHECK_CLOSE(qtq(i, j), 1.0, 1e-10); // Diagonal should be 1
      } else {
        BOOST_CHECK_SMALL(std::abs(qtq(i, j)),
                          1e-10); // Off-diagonal should be 0
      }
    }
  }

  // Verify decomposition: A = Q * H * Q^T
  Eigen::Matrix<double, 3, 3> reconstructed =
      q_matrix * h_matrix * q_matrix.transpose();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      BOOST_CHECK_CLOSE(reconstructed(i, j), input_matrix(i, j), 1e-10);
    }
  }
}

} /* namespace linalg */
} /* namespace gr */