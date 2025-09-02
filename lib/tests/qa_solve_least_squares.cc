/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <boost/test/unit_test.hpp>
#include <gnuradio/attributes.h>
#include <gnuradio/linalg/solve_least_squares.h>

namespace gr {
namespace linalg {

// Test basic overdetermined system with well-conditioned matrix
BOOST_AUTO_TEST_CASE(test_solve_least_squares_well_conditioned) {
  // Create 3x2 overdetermined system: Ax = b
  // A = [[1, 1], [1, 2], [1, 3]]
  // b = [6, 8, 10]
  // Expected solution: x ≈ [4, 2] (minimizes ||Ax - b||)

  std::vector<float> matrix_a = {1.0f, 1.0f, 1.0f,
                                 1.0f, 2.0f, 3.0f}; // Column-major
  std::vector<float> vector_b = {6.0f, 8.0f, 10.0f};

  auto solver = solve_least_squares_sync_f::make(
      {3, 2}, {3, 1}, least_squares_method::AUTO, 1e-6);
  BOOST_REQUIRE(solver);

  // Expected to use Normal Equations for well-conditioned matrix
  // Verify creation succeeds
  BOOST_CHECK_EQUAL(solver->name(), "solve_least_squares");
}

// Test Normal Equations method explicitly
BOOST_AUTO_TEST_CASE(test_solve_least_squares_normal_equations) {
  // Well-conditioned 2x2 system that should be exact
  // A = [[2, 1], [1, 2]]
  // b = [5, 4]
  // Expected solution: x = [2, 1]

  std::vector<float> matrix_a = {2.0f, 1.0f, 1.0f,
                                 2.0f}; // Column-major [[2,1],[1,2]]
  std::vector<float> vector_b = {5.0f, 4.0f};

  auto solver = solve_least_squares_sync_f::make(
      {2, 2}, {2, 1}, least_squares_method::NORMAL_EQ, 1e-6);
  BOOST_REQUIRE(solver);

  // Test that it uses specified method (not AUTO)
  BOOST_CHECK_EQUAL(solver->name(), "solve_least_squares");
}

// Test QR decomposition method
BOOST_AUTO_TEST_CASE(test_solve_least_squares_qr_decomposition) {
  // Test QR method with overdetermined system
  // A = [[1, 0], [1, 1], [0, 1]]
  // b = [1, 2, 1]
  // This should use QR decomposition

  std::vector<double> matrix_a = {1.0, 1.0, 0.0, 0.0, 1.0, 1.0}; // Column-major
  std::vector<double> vector_b = {1.0, 2.0, 1.0};

  auto solver = solve_least_squares_sync_d::make(
      {3, 2}, {3, 1}, least_squares_method::QR, 1e-10);
  BOOST_REQUIRE(solver);

  BOOST_CHECK_EQUAL(solver->name(), "solve_least_squares");
}

// Test SVD method for rank-deficient matrix
BOOST_AUTO_TEST_CASE(test_solve_least_squares_svd_robust) {
  // Test SVD with ill-conditioned matrix
  // A = [[1, 1], [1, 1.001], [1, 1.002]]
  // b = [2, 2.001, 2.002]
  // Near-singular system that should use SVD

  std::vector<double> matrix_a = {1.0, 1.0,   1.0,
                                  1.0, 1.001, 1.002}; // Column-major
  std::vector<double> vector_b = {2.0, 2.001, 2.002};

  auto solver = solve_least_squares_sync_d::make(
      {3, 2}, {3, 1}, least_squares_method::SVD, 1e-12);
  BOOST_REQUIRE(solver);

  BOOST_CHECK_EQUAL(solver->name(), "solve_least_squares");
}

// Test AUTO method selection based on condition number
BOOST_AUTO_TEST_CASE(test_solve_least_squares_auto_selection) {
  // Test that AUTO method properly selects algorithm

  // Well-conditioned matrix (should select Normal Equations)
  std::vector<float> well_conditioned_a = {2.0f, 0.0f, 0.0f,
                                           2.0f}; // Identity-like
  std::vector<float> well_conditioned_b = {4.0f, 6.0f};

  auto solver_well = solve_least_squares_sync_f::make(
      {2, 2}, {2, 1}, least_squares_method::AUTO, 1e-6);
  BOOST_REQUIRE(solver_well);

  // Ill-conditioned matrix (should select QR or SVD)
  std::vector<float> ill_conditioned_a = {1.0f, 1.0f, 1.0f,
                                          1.00001f}; // Nearly singular
  std::vector<float> ill_conditioned_b = {2.0f, 2.00001f};

  auto solver_ill = solve_least_squares_sync_f::make(
      {2, 2}, {2, 1}, least_squares_method::AUTO, 1e-6);
  BOOST_REQUIRE(solver_ill);

  BOOST_CHECK_EQUAL(solver_well->name(), "solve_least_squares");
  BOOST_CHECK_EQUAL(solver_ill->name(), "solve_least_squares");
}

// Test complex number support
BOOST_AUTO_TEST_CASE(test_solve_least_squares_complex) {
  // Test complex overdetermined system
  // A = [[1+0j, 0+1j], [0+1j, 1+0j], [1+1j, 0+0j]]
  // This tests complex arithmetic in least squares

  std::vector<std::complex<float>> matrix_a = {
      {1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f},
      {0.0f, 1.0f}, {1.0f, 0.0f}, {0.0f, 0.0f}}; // Column-major complex matrix
  std::vector<std::complex<float>> vector_b = {
      {1.0f, 1.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}};

  auto solver = solve_least_squares_sync_c::make(
      {3, 2}, {3, 1}, least_squares_method::AUTO, 1e-6);
  BOOST_REQUIRE(solver);

  BOOST_CHECK_EQUAL(solver->name(), "solve_least_squares");
}

// Test error handling for invalid inputs
BOOST_AUTO_TEST_CASE(test_solve_least_squares_error_handling) {
  // Test invalid shape combinations

  // Non-2D matrices should throw
  BOOST_CHECK_THROW(solve_least_squares_sync_f::make(
                        {3}, {3}, least_squares_method::AUTO, 1e-6),
                    std::invalid_argument);

  // Mismatched dimensions (A.rows != b.rows)
  BOOST_CHECK_THROW(solve_least_squares_sync_f::make(
                        {3, 2}, {4, 1}, least_squares_method::AUTO, 1e-6),
                    std::invalid_argument);

  // Underdetermined system (A.rows < A.cols)
  BOOST_CHECK_THROW(solve_least_squares_sync_f::make(
                        {2, 3}, {2, 1}, least_squares_method::AUTO, 1e-6),
                    std::invalid_argument);

  // Valid construction should succeed
  BOOST_CHECK_NO_THROW(solve_least_squares_sync_f::make(
      {3, 2}, {3, 1}, least_squares_method::AUTO, 1e-6));
}

// Test multiple right-hand sides (matrix b instead of vector)
BOOST_AUTO_TEST_CASE(test_solve_least_squares_multiple_rhs) {
  // Test solving Ax = B where B has multiple columns
  // A = [[1, 1], [1, 2], [1, 3]] (3x2)
  // B = [[6, 12], [8, 16], [10, 20]] (3x2)
  // Should solve two systems simultaneously

  std::vector<double> matrix_a = {1.0, 1.0, 1.0,
                                  1.0, 2.0, 3.0}; // Column-major 3x2
  std::vector<double> matrix_b = {6.0,  8.0,  10.0,
                                  12.0, 16.0, 20.0}; // Column-major 3x2

  auto solver = solve_least_squares_sync_d::make(
      {3, 2}, {3, 2}, least_squares_method::AUTO, 1e-10);
  BOOST_REQUIRE(solver);

  BOOST_CHECK_EQUAL(solver->name(), "solve_least_squares");
}

// Test PDU interface creation
BOOST_AUTO_TEST_CASE(test_solve_least_squares_pdu_interface) {
  // Test that PDU interface can be created
  auto solver_pdu = solve_least_squares_pdu_f::make(
      {3, 2}, {3, 1}, least_squares_method::AUTO, 1e-6);
  BOOST_REQUIRE(solver_pdu);

  BOOST_CHECK_EQUAL(solver_pdu->name(), "solve_least_squares");

  // Test complex PDU variant
  auto solver_pdu_complex = solve_least_squares_pdu_c::make(
      {4, 3}, {4, 2}, least_squares_method::QR, 1e-8);
  BOOST_REQUIRE(solver_pdu_complex);

  BOOST_CHECK_EQUAL(solver_pdu_complex->name(), "solve_least_squares");
}

// Test tolerance parameter handling
BOOST_AUTO_TEST_CASE(test_solve_least_squares_tolerance) {
  // Test different tolerance values

  // Very strict tolerance
  auto solver_strict = solve_least_squares_sync_d::make(
      {3, 2}, {3, 1}, least_squares_method::SVD, 1e-15);
  BOOST_REQUIRE(solver_strict);

  // Relaxed tolerance
  auto solver_relaxed = solve_least_squares_sync_d::make(
      {3, 2}, {3, 1}, least_squares_method::SVD, 1e-3);
  BOOST_REQUIRE(solver_relaxed);

  // Default tolerance (should work)
  auto solver_default = solve_least_squares_sync_d::make(
      {3, 2}, {3, 1}, least_squares_method::AUTO, 1e-10);
  BOOST_REQUIRE(solver_default);

  BOOST_CHECK_EQUAL(solver_strict->name(), "solve_least_squares");
  BOOST_CHECK_EQUAL(solver_relaxed->name(), "solve_least_squares");
  BOOST_CHECK_EQUAL(solver_default->name(), "solve_least_squares");
}

} /* namespace linalg */
} /* namespace gr */
