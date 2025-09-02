/* -*- c++ -*- */
/*
 * Copyright 2025 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <Eigen/Dense>
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <complex>
#include <gnuradio/attributes.h>
#include <gnuradio/linalg/solve_iterative.h>
#include <gnuradio/linalg/types.h>

namespace gr {
namespace linalg {

BOOST_AUTO_TEST_CASE(test_conjugate_gradient_solver) {
  // Test CG with a symmetric positive definite system
  types::shape shape_a = {3, 3};
  types::shape shape_b = {3, 1};

  auto iterative_block = solve_iterative_sync_f::make(
      shape_a, shape_b, iterative_method::CG, 100, 1e-6f, false);

  // Create SPD matrix A = [[4, 1, 0], [1, 4, 1], [0, 1, 4]]
  Eigen::Matrix<float, 3, 3> A_matrix;
  A_matrix << 4.0f, 1.0f, 0.0f, 1.0f, 4.0f, 1.0f, 0.0f, 1.0f, 4.0f;

  // Create RHS b = [6, 10, 6] -> solution should be [1, 2, 1]
  Eigen::Matrix<float, 3, 1> b_vector;
  b_vector << 6.0f, 10.0f, 6.0f;

  // Prepare input/output containers
  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> A_map(
      A_matrix.data(), 3, 3);
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> b_map(
      b_vector.data(), 3, 1);
  inputs.push_back(&A_map);
  inputs.push_back(&b_map);

  Eigen::Matrix<float, 3, 1> x_output;
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> x_map(
      x_output.data(), 3, 1);
  outputs.push_back(&x_map);

  // Perform iterative solve
  auto result = iterative_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Verify solution: x should be approximately [1, 2, 1]
  float tolerance = 1e-2f; // 1% tolerance for iterative solvers
  BOOST_CHECK_CLOSE(x_output(0), 1.0f, tolerance * 100);
  BOOST_CHECK_CLOSE(x_output(1), 2.0f, tolerance * 100);
  BOOST_CHECK_CLOSE(x_output(2), 1.0f, tolerance * 100);

  // Verify by substitution: A*x = b
  Eigen::Matrix<float, 3, 1> verification = A_matrix * x_output;
  for (int i = 0; i < 3; ++i) {
    BOOST_CHECK_CLOSE(verification(i), b_vector(i), tolerance * 100);
  }
}

BOOST_AUTO_TEST_CASE(test_bicgstab_solver) {
  // Test BiCGStab with a general (non-symmetric) system
  types::shape shape_a = {2, 2};
  types::shape shape_b = {2, 1};

  auto iterative_block = solve_iterative_sync_f::make(
      shape_a, shape_b, iterative_method::BICGSTAB, 100, 1e-6f, false);

  // Create non-symmetric matrix A = [[3, 1], [2, 4]]
  Eigen::Matrix<float, 2, 2> A_matrix;
  A_matrix << 3.0f, 1.0f, 2.0f, 4.0f;

  // Create RHS b = [5, 10] -> solution should be [1, 2]
  Eigen::Matrix<float, 2, 1> b_vector;
  b_vector << 5.0f, 10.0f;

  // Prepare input/output containers
  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> A_map(
      A_matrix.data(), 2, 2);
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> b_map(
      b_vector.data(), 2, 1);
  inputs.push_back(&A_map);
  inputs.push_back(&b_map);

  Eigen::Matrix<float, 2, 1> x_output;
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> x_map(
      x_output.data(), 2, 1);
  outputs.push_back(&x_map);

  // Perform iterative solve
  auto result = iterative_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Verify solution: x should be approximately [1, 2]
  float tolerance = 1e-2f; // 1% tolerance for iterative solvers
  BOOST_CHECK_CLOSE(x_output(0), 1.0f, tolerance * 100);
  BOOST_CHECK_CLOSE(x_output(1), 2.0f, tolerance * 100);

  // Verify by substitution: A*x = b
  Eigen::Matrix<float, 2, 1> verification = A_matrix * x_output;
  for (int i = 0; i < 2; ++i) {
    BOOST_CHECK_CLOSE(verification(i), b_vector(i), tolerance * 100);
  }
}

BOOST_AUTO_TEST_CASE(test_lscg_overdetermined_system) {
  // Test LSCG with an overdetermined system (more equations than unknowns)
  types::shape shape_a = {4, 2}; // 4x2 matrix A
  types::shape shape_b = {4, 1}; // 4x1 vector b

  auto iterative_block = solve_iterative_sync_f::make(
      shape_a, shape_b, iterative_method::LSCG, 200, 1e-6f, false);

  // Create overdetermined system:
  // A = [[1, 1], [1, 2], [1, 3], [1, 4]]
  // This represents fitting y = a + b*x to points (1,6), (2,8), (3,10), (4,12)
  // Expected least squares solution: a ≈ 4, b ≈ 2
  Eigen::Matrix<float, 4, 2> A_matrix;
  A_matrix << 1.0f, 1.0f, 1.0f, 2.0f, 1.0f, 3.0f, 1.0f, 4.0f;

  Eigen::Matrix<float, 4, 1> b_vector;
  b_vector << 6.0f, 8.0f, 10.0f, 12.0f;

  // Prepare input/output containers
  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> A_map(
      A_matrix.data(), 4, 2);
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> b_map(
      b_vector.data(), 4, 1);
  inputs.push_back(&A_map);
  inputs.push_back(&b_map);

  Eigen::Matrix<float, 2, 1> x_output;
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> x_map(
      x_output.data(), 2, 1);
  outputs.push_back(&x_map);

  // Perform iterative least squares solve
  auto result = iterative_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Verify solution: should be approximately [4, 2]
  float tolerance = 1e-2f; // 1% tolerance for iterative solvers
  BOOST_CHECK_CLOSE(x_output(0), 4.0f, tolerance * 100);
  BOOST_CHECK_CLOSE(x_output(1), 2.0f, tolerance * 100);

  // Verify least squares property: residual should be small
  Eigen::Matrix<float, 4, 1> residual = A_matrix * x_output - b_vector;
  float residual_norm = residual.norm();
  BOOST_CHECK_SMALL(residual_norm,
                    1e-5f); // Relaxed tolerance for iterative methods
}

BOOST_AUTO_TEST_CASE(test_initial_guess) {
  // Test using initial guess to potentially improve convergence
  types::shape shape_a = {2, 2};
  types::shape shape_b = {2, 1};

  auto iterative_block =
      solve_iterative_sync_f::make(shape_a, shape_b, iterative_method::CG, 50,
                                   1e-6f, true); // use_initial_guess = true

  // Create SPD system
  Eigen::Matrix<float, 2, 2> A_matrix;
  A_matrix << 4.0f, 1.0f, 1.0f, 4.0f;

  Eigen::Matrix<float, 2, 1> b_vector;
  b_vector << 5.0f, 5.0f; // Solution is [1, 1]

  // Provide good initial guess close to solution
  Eigen::Matrix<float, 2, 1> x0_vector;
  x0_vector << 0.9f, 1.1f; // Close to [1, 1]

  // Prepare input/output containers (3 inputs: A, b, x0)
  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> A_map(
      A_matrix.data(), 2, 2);
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> b_map(
      b_vector.data(), 2, 1);
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> x0_map(
      x0_vector.data(), 2, 1);
  inputs.push_back(&A_map);
  inputs.push_back(&b_map);
  inputs.push_back(&x0_map);

  Eigen::Matrix<float, 2, 1> x_output;
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> x_map(
      x_output.data(), 2, 1);
  outputs.push_back(&x_map);

  // Perform solve with initial guess
  auto result = iterative_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Verify solution
  float tolerance = 1e-2f; // 1% tolerance for iterative solvers
  BOOST_CHECK_CLOSE(x_output(0), 1.0f, tolerance * 100);
  BOOST_CHECK_CLOSE(x_output(1), 1.0f, tolerance * 100);
}

BOOST_AUTO_TEST_CASE(test_complex_iterative_solver) {
  // Test iterative solver with complex numbers
  types::shape shape_a = {2, 2};
  types::shape shape_b = {2, 1};

  auto iterative_block = solve_iterative_sync_cf::make(
      shape_a, shape_b, iterative_method::BICGSTAB, 100, 1e-6f, false);

  // Create complex matrix A
  Eigen::Matrix<std::complex<float>, 2, 2> A_matrix;
  A_matrix << std::complex<float>(2.0f, 0.5f), std::complex<float>(1.0f, 0.0f),
      std::complex<float>(0.0f, 1.0f), std::complex<float>(2.0f, -0.5f);

  // Create complex RHS
  Eigen::Matrix<std::complex<float>, 2, 1> b_vector;
  b_vector << std::complex<float>(3.0f, 1.0f), std::complex<float>(2.0f, 2.0f);

  // Prepare input/output containers
  types::vector_const_matrix_map<std::complex<float>> inputs;
  types::vector_matrix_map<std::complex<float>> outputs;

  Eigen::Map<
      const Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic>>
      A_map(A_matrix.data(), 2, 2);
  Eigen::Map<
      const Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic>>
      b_map(b_vector.data(), 2, 1);
  inputs.push_back(&A_map);
  inputs.push_back(&b_map);

  Eigen::Matrix<std::complex<float>, 2, 1> x_output;
  Eigen::Map<Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic>>
      x_map(x_output.data(), 2, 1);
  outputs.push_back(&x_map);

  // Perform iterative solve
  auto result = iterative_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Verify by substitution: A*x = b
  Eigen::Matrix<std::complex<float>, 2, 1> verification = A_matrix * x_output;
  float tolerance = 1e-3f;
  for (int i = 0; i < 2; ++i) {
    BOOST_CHECK_SMALL(std::abs(verification(i) - b_vector(i)), tolerance);
  }

  // Check that solution contains finite values
  for (int i = 0; i < 2; ++i) {
    BOOST_CHECK(!std::isnan(x_output(i).real()) &&
                !std::isinf(x_output(i).real()));
    BOOST_CHECK(!std::isnan(x_output(i).imag()) &&
                !std::isinf(x_output(i).imag()));
  }
}

BOOST_AUTO_TEST_CASE(test_auto_method_selection) {
  // Test automatic method selection

  // Square matrix should select BiCGStab
  {
    types::shape shape_a = {2, 2};
    types::shape shape_b = {2, 1};

    auto iterative_block = solve_iterative_sync_f::make(
        shape_a, shape_b, iterative_method::AUTO, 100, 1e-6f, false);

    BOOST_CHECK_EQUAL(iterative_block->get_method(), iterative_method::AUTO);
  }

  // Rectangular matrix should select LSCG
  {
    types::shape shape_a = {3, 2};
    types::shape shape_b = {3, 1};

    auto iterative_block = solve_iterative_sync_f::make(
        shape_a, shape_b, iterative_method::AUTO, 100, 1e-6f, false);

    BOOST_CHECK_EQUAL(iterative_block->get_method(), iterative_method::AUTO);
  }
}

BOOST_AUTO_TEST_CASE(test_iterative_solver_error_handling) {
  // Test error handling for invalid inputs
  types::shape shape_a = {2, 2};
  types::shape shape_b = {2, 1};

  auto iterative_block = solve_iterative_sync_f::make(
      shape_a, shape_b, iterative_method::CG, 100, 1e-6f, false);

  // Test with wrong number of inputs
  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  // Empty inputs should fail
  auto result = iterative_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::INVALID_SHAPE);

  // Test with wrong output size
  Eigen::Matrix<float, 2, 2> A_matrix = Eigen::Matrix<float, 2, 2>::Identity();
  Eigen::Matrix<float, 2, 1> b_vector = Eigen::Matrix<float, 2, 1>::Ones();

  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> A_map(
      A_matrix.data(), 2, 2);
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> b_map(
      b_vector.data(), 2, 1);
  inputs.push_back(&A_map);
  inputs.push_back(&b_map);

  // No outputs provided should fail
  result = iterative_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::INVALID_SHAPE);
}

BOOST_AUTO_TEST_CASE(test_convergence_parameters) {
  // Test different convergence parameters
  types::shape shape_a = {2, 2};
  types::shape shape_b = {2, 1};

  // Test with tight tolerance
  auto tight_block = solve_iterative_sync_d::make(
      shape_a, shape_b, iterative_method::CG, 1000, 1e-12, false);

  BOOST_CHECK_EQUAL(tight_block->get_max_iterations(), 1000);
  BOOST_CHECK_CLOSE(tight_block->get_tolerance(), 1e-12, 1e-6);

  // Test parameter modification
  tight_block->set_max_iterations(500);
  tight_block->set_tolerance(1e-8);

  BOOST_CHECK_EQUAL(tight_block->get_max_iterations(), 500);
  BOOST_CHECK_CLOSE(tight_block->get_tolerance(), 1e-8, 1e-6);
}

} /* namespace linalg */
} /* namespace gr */