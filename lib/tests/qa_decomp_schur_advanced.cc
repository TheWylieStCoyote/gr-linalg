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
#include <gnuradio/linalg/decomp_schur.h>

namespace gr {
namespace linalg {

BOOST_AUTO_TEST_CASE(test_decomp_schur_basic_creation) {
  // Test basic block creation with default options
  auto schur_block = decomp_schur_sync_f::make({2, 2}, true);

  BOOST_REQUIRE(schur_block != nullptr);

  // Check basic parameters
  BOOST_CHECK(schur_block->get_compute_u() == true);

  // Check default options
  const auto &options = schur_block->get_options();
  BOOST_CHECK(options.balance_matrix == false);
  BOOST_CHECK(options.extract_eigenvalues == false);
  BOOST_CHECK(options.sort_eigenvalues == false);
  BOOST_CHECK(options.compute_condition == false);
  BOOST_CHECK(options.use_complex_schur == false);
}

BOOST_AUTO_TEST_CASE(test_decomp_schur_advanced_options) {
  // Test block creation with advanced options
  schur_options options;
  options.balance_matrix = true;
  options.extract_eigenvalues = true;
  options.sort_eigenvalues = true;
  options.compute_condition = true;
  options.balance_tolerance = 1e-10;
  options.sort_tolerance = 1e-14;

  auto schur_block = decomp_schur_sync_d::make({3, 3}, true, nullptr, options);

  BOOST_REQUIRE(schur_block != nullptr);

  // Check advanced options were set
  const auto &block_options = schur_block->get_options();
  BOOST_CHECK(block_options.balance_matrix == true);
  BOOST_CHECK(block_options.extract_eigenvalues == true);
  BOOST_CHECK(block_options.sort_eigenvalues == true);
  BOOST_CHECK(block_options.compute_condition == true);
  BOOST_CHECK_CLOSE(block_options.balance_tolerance, 1e-10, 1e-12);
  BOOST_CHECK_CLOSE(block_options.sort_tolerance, 1e-14, 1e-16);
}

BOOST_AUTO_TEST_CASE(test_decomp_schur_io_signatures) {
  // Test input/output signatures for different matrix sizes
  auto schur_2x2 = decomp_schur_sync_f::make({2, 2});
  auto schur_3x3 = decomp_schur_sync_d::make({3, 3});

  // Check 2x2 signatures
  auto input_sig_2x2 = schur_2x2->input_signature();
  auto output_sig_2x2 = schur_2x2->output_signature();

  BOOST_CHECK_EQUAL(input_sig_2x2->max_streams(), 1);
  BOOST_CHECK_EQUAL(output_sig_2x2->max_streams(), 2); // Q and T matrices
  BOOST_CHECK_EQUAL(input_sig_2x2->sizeof_stream_item(0), 4 * sizeof(float));
  BOOST_CHECK_EQUAL(output_sig_2x2->sizeof_stream_item(0),
                    4 * sizeof(float)); // Q matrix
  BOOST_CHECK_EQUAL(output_sig_2x2->sizeof_stream_item(1),
                    4 * sizeof(float)); // T matrix

  // Check 3x3 signatures
  auto input_sig_3x3 = schur_3x3->input_signature();
  auto output_sig_3x3 = schur_3x3->output_signature();

  BOOST_CHECK_EQUAL(input_sig_3x3->max_streams(), 1);
  BOOST_CHECK_EQUAL(output_sig_3x3->max_streams(), 2);
  BOOST_CHECK_EQUAL(input_sig_3x3->sizeof_stream_item(0), 9 * sizeof(double));
  BOOST_CHECK_EQUAL(output_sig_3x3->sizeof_stream_item(0), 9 * sizeof(double));
  BOOST_CHECK_EQUAL(output_sig_3x3->sizeof_stream_item(1), 9 * sizeof(double));
}

BOOST_AUTO_TEST_CASE(test_decomp_schur_compute_q_option) {
  // Test with compute_u = false (don't compute Q matrix)
  auto schur_no_q = decomp_schur_sync_f::make({2, 2}, false);

  BOOST_REQUIRE(schur_no_q != nullptr);
  BOOST_CHECK(schur_no_q->get_compute_u() == false);

  // Output should be 1 stream (only T matrix)
  auto output_sig = schur_no_q->output_signature();
  BOOST_CHECK_EQUAL(output_sig->max_streams(), 1);
}

BOOST_AUTO_TEST_CASE(test_decomp_schur_complex_variants) {
  // Test complex float variant
  auto schur_cf = decomp_schur_sync_cf::make({2, 2});
  BOOST_REQUIRE(schur_cf != nullptr);

  auto input_sig_cf = schur_cf->input_signature();
  auto output_sig_cf = schur_cf->output_signature();

  BOOST_CHECK_EQUAL(input_sig_cf->sizeof_stream_item(0),
                    4 * sizeof(std::complex<float>));
  BOOST_CHECK_EQUAL(output_sig_cf->sizeof_stream_item(0),
                    4 * sizeof(std::complex<float>));
  BOOST_CHECK_EQUAL(output_sig_cf->sizeof_stream_item(1),
                    4 * sizeof(std::complex<float>));

  // Test complex double variant
  auto schur_cd = decomp_schur_sync_cd::make({3, 3});
  BOOST_REQUIRE(schur_cd != nullptr);

  auto input_sig_cd = schur_cd->input_signature();
  auto output_sig_cd = schur_cd->output_signature();

  BOOST_CHECK_EQUAL(input_sig_cd->sizeof_stream_item(0),
                    9 * sizeof(std::complex<double>));
  BOOST_CHECK_EQUAL(output_sig_cd->sizeof_stream_item(0),
                    9 * sizeof(std::complex<double>));
  BOOST_CHECK_EQUAL(output_sig_cd->sizeof_stream_item(1),
                    9 * sizeof(std::complex<double>));
}

BOOST_AUTO_TEST_CASE(test_decomp_schur_pdu_variants) {
  // Test PDU variants
  auto pdu_f = decomp_schur_pdu_f::make({2, 2}, true);
  auto pdu_d = decomp_schur_pdu_d::make({3, 3}, false);
  auto pdu_cf = decomp_schur_pdu_cf::make({2, 2}, true);
  auto pdu_cd = decomp_schur_pdu_cd::make({4, 4}, false);

  BOOST_CHECK(pdu_f != nullptr);
  BOOST_CHECK(pdu_d != nullptr);
  BOOST_CHECK(pdu_cf != nullptr);
  BOOST_CHECK(pdu_cd != nullptr);

  // Test getter/setter methods
  BOOST_CHECK(pdu_f->get_compute_u() == true);
  BOOST_CHECK(pdu_d->get_compute_u() == false);

  pdu_f->set_compute_u(false);
  pdu_d->set_compute_u(true);

  BOOST_CHECK(pdu_f->get_compute_u() == false);
  BOOST_CHECK(pdu_d->get_compute_u() == true);
}

BOOST_AUTO_TEST_CASE(test_decomp_schur_advanced_feature_setters) {
  auto schur_block = decomp_schur_sync_f::make({2, 2});

  // Test advanced feature setters
  schur_block->set_balance_matrix(true);
  schur_block->set_extract_eigenvalues(true);
  schur_block->set_sort_eigenvalues(true);
  schur_block->set_compute_condition(true);
  schur_block->set_use_complex_schur(true);
  schur_block->set_balance_tolerance(1e-12);
  schur_block->set_sort_tolerance(1e-15);

  const auto &options = schur_block->get_options();
  BOOST_CHECK(options.balance_matrix == true);
  BOOST_CHECK(options.extract_eigenvalues == true);
  BOOST_CHECK(options.sort_eigenvalues == true);
  BOOST_CHECK(options.compute_condition == true);
  BOOST_CHECK(options.use_complex_schur == true);
  BOOST_CHECK_CLOSE(options.balance_tolerance, 1e-12, 1e-14);
  BOOST_CHECK_CLOSE(options.sort_tolerance, 1e-15, 1e-17);
}

BOOST_AUTO_TEST_CASE(test_decomp_schur_shape_validation) {
  // Test valid square matrix shapes
  BOOST_CHECK_NO_THROW(decomp_schur_sync_f::make({2, 2}));
  BOOST_CHECK_NO_THROW(decomp_schur_sync_d::make({3, 3}));
  BOOST_CHECK_NO_THROW(decomp_schur_sync_cf::make({4, 4}));
  BOOST_CHECK_NO_THROW(decomp_schur_sync_cd::make({5, 5}));

  // Non-square matrices should be rejected
  BOOST_CHECK_THROW(decomp_schur_sync_f::make({2, 3}), std::exception);
  BOOST_CHECK_THROW(decomp_schur_sync_d::make({3, 2}), std::exception);
  BOOST_CHECK_THROW(decomp_schur_sync_cf::make({4, 3}), std::exception);
  BOOST_CHECK_THROW(decomp_schur_sync_cd::make({2, 5}), std::exception);
}

BOOST_AUTO_TEST_CASE(test_decomp_schur_compute_output_shapes) {
  // Test static shape computation
  types::vector_shapes input_shapes = {{3, 3}};
  auto output_shapes = decomp_schur<float>::compute_output_shapes(input_shapes);

  BOOST_REQUIRE_EQUAL(output_shapes.size(), 2); // Q and T matrices
  BOOST_REQUIRE_EQUAL(output_shapes[0].size(), 2);
  BOOST_REQUIRE_EQUAL(output_shapes[1].size(), 2);

  // Both Q and T should have same shape as input
  BOOST_CHECK_EQUAL(output_shapes[0][0], 3);
  BOOST_CHECK_EQUAL(output_shapes[0][1], 3);
  BOOST_CHECK_EQUAL(output_shapes[1][0], 3);
  BOOST_CHECK_EQUAL(output_shapes[1][1], 3);

  // Test invalid input shapes
  BOOST_CHECK_THROW(decomp_schur<double>::compute_output_shapes({{2, 3}}),
                    std::invalid_argument);
  BOOST_CHECK_THROW(
      decomp_schur<float>::compute_output_shapes({{2, 2}, {3, 3}}),
      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_decomp_schur_algorithm_interface) {
  // Test that we can create blocks with custom algorithms
  auto eigen_real_algorithm = std::make_shared<eigen_real_schur<float>>();
  auto eigen_complex_algorithm =
      std::make_shared<eigen_complex_schur<std::complex<float>>>();
  auto eigen_advanced_algorithm =
      std::make_shared<eigen_schur_advanced<double>>();

  // Create blocks with custom algorithms
  auto schur_real =
      decomp_schur_sync_f::make({2, 2}, true, eigen_real_algorithm);
  auto schur_complex =
      decomp_schur_sync_cf::make({2, 2}, true, eigen_complex_algorithm);
  auto schur_advanced =
      decomp_schur_sync_d::make({3, 3}, true, eigen_advanced_algorithm);

  BOOST_CHECK(schur_real != nullptr);
  BOOST_CHECK(schur_complex != nullptr);
  BOOST_CHECK(schur_advanced != nullptr);
}

BOOST_AUTO_TEST_CASE(test_decomp_schur_operation_interface) {
  // Test that the operation interface is working
  schur_options options;
  options.extract_eigenvalues = true;

  auto schur_impl =
      decomp_schur<float>("test_schur", {2, 2}, true, nullptr, options);

  // Create test matrices
  std::vector<float> input_data = {2.0f, 1.0f, 1.0f, 2.0f}; // [[2, 1], [1, 2]]
  std::vector<float> output_q(4);
  std::vector<float> output_t(4);

  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      input_matrix(input_data.data(), 2, 2);
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> q_matrix(
      output_q.data(), 2, 2);
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> t_matrix(
      output_t.data(), 2, 2);

  types::vector_const_matrix_map<float> input_matrices = {&input_matrix};
  types::vector_matrix_map<float> output_matrices = {&q_matrix, &t_matrix};

  // Test operation call
  auto result = schur_impl.operation(input_matrices, output_matrices);

  BOOST_CHECK(result == OperationReturn::SUCCESS);

  // Check that we have advanced results available
  const auto &last_result = schur_impl.get_last_result();
  BOOST_CHECK(last_result.eigenvalues.size() ==
              2); // Should have extracted eigenvalues
}

BOOST_AUTO_TEST_CASE(test_decomp_schur_result_accessors) {
  // Create a block with advanced options to test result accessors
  schur_options options;
  options.balance_matrix = true;
  options.extract_eigenvalues = true;
  options.compute_condition = true;

  auto schur_block = decomp_schur_sync_f::make({2, 2}, true, nullptr, options);
  BOOST_REQUIRE(schur_block != nullptr);

  // Test that accessors don't crash (results will be empty until operation is
  // called)
  BOOST_CHECK_NO_THROW(schur_block->get_condition_estimate());
  BOOST_CHECK_NO_THROW(schur_block->was_matrix_balanced());
  BOOST_CHECK_NO_THROW(schur_block->is_complex_schur_form());
  BOOST_CHECK_NO_THROW(schur_block->get_permutation());
  BOOST_CHECK_NO_THROW(schur_block->get_scale_factors());
  BOOST_CHECK_NO_THROW(schur_block->get_eigenvalues());
  BOOST_CHECK_NO_THROW(schur_block->get_last_result());
}

} /* namespace linalg */
} /* namespace gr */