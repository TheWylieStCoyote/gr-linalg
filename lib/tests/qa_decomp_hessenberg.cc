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
#include <gnuradio/attributes.h>
#include <gnuradio/linalg/decomp_hessenberg.h>
#include <gnuradio/linalg/types.h>

namespace gr {
namespace linalg {

BOOST_AUTO_TEST_CASE(test_real_hessenberg_decomposition) {
  // Test real Hessenberg decomposition with a 3x3 matrix
  types::shape input_shape = {3, 3};
  auto hessenberg_block = decomp_hessenberg_sync_f::make(input_shape, true);

  // Create input matrix A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  Eigen::Matrix<float, 3, 3> input_matrix;
  input_matrix << 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f;

  // Prepare input/output containers
  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      input_map(input_matrix.data(), 3, 3);
  inputs.push_back(&input_map);

  Eigen::Matrix<float, 3, 3> Q_output, H_output;
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> Q_map(
      Q_output.data(), 3, 3);
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> H_map(
      H_output.data(), 3, 3);
  outputs.push_back(&Q_map);
  outputs.push_back(&H_map);

  // Perform decomposition
  auto result = hessenberg_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Verify A = Q * H * Q^T
  Eigen::Matrix<float, 3, 3> reconstructed =
      Q_output * H_output * Q_output.transpose();

  float tolerance = 1e-4f;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      BOOST_CHECK_CLOSE(reconstructed(i, j), input_matrix(i, j), tolerance);
    }
  }

  // Verify Q is orthogonal (Q^T * Q = I)
  Eigen::Matrix<float, 3, 3> identity_check = Q_output.transpose() * Q_output;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      float expected = (i == j) ? 1.0f : 0.0f;
      BOOST_CHECK_SMALL(identity_check(i, j) - expected, tolerance);
    }
  }

  // Verify H is upper Hessenberg (zero below first subdiagonal)
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (i > j + 1) { // Below first subdiagonal
        BOOST_CHECK_SMALL(std::abs(H_output(i, j)), tolerance);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(test_complex_hessenberg_decomposition) {
  // Test complex Hessenberg decomposition
  types::shape input_shape = {2, 2};
  auto hessenberg_block = decomp_hessenberg_sync_cf::make(input_shape, true);

  // Create complex input matrix
  Eigen::Matrix<std::complex<float>, 2, 2> input_matrix;
  input_matrix << std::complex<float>(1.0f, 0.5f),
      std::complex<float>(2.0f, 0.0f), std::complex<float>(0.0f, 1.0f),
      std::complex<float>(3.0f, -0.5f);

  // Prepare input/output containers
  types::vector_const_matrix_map<std::complex<float>> inputs;
  types::vector_matrix_map<std::complex<float>> outputs;

  Eigen::Map<
      const Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic>>
      input_map(input_matrix.data(), 2, 2);
  inputs.push_back(&input_map);

  Eigen::Matrix<std::complex<float>, 2, 2> Q_output, H_output;
  Eigen::Map<Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic>>
      Q_map(Q_output.data(), 2, 2);
  Eigen::Map<Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic>>
      H_map(H_output.data(), 2, 2);
  outputs.push_back(&Q_map);
  outputs.push_back(&H_map);

  // Perform decomposition
  auto result = hessenberg_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Verify A = Q * H * Q^H (conjugate transpose)
  Eigen::Matrix<std::complex<float>, 2, 2> reconstructed =
      Q_output * H_output * Q_output.adjoint();

  float tolerance = 1e-4f;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      BOOST_CHECK_SMALL(std::abs(reconstructed(i, j) - input_matrix(i, j)),
                        tolerance);
    }
  }

  // Verify Q is unitary (Q^H * Q = I)
  Eigen::Matrix<std::complex<float>, 2, 2> unitary_check =
      Q_output.adjoint() * Q_output;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      std::complex<float> expected = (i == j) ? std::complex<float>(1.0f, 0.0f)
                                              : std::complex<float>(0.0f, 0.0f);
      BOOST_CHECK_SMALL(std::abs(unitary_check(i, j) - expected), tolerance);
    }
  }

  // Verify H is upper Hessenberg
  // Note: For a 2x2 matrix, it's already in Hessenberg form, so H(1,0) doesn't
  // need to be zero We just verify the decomposition worked and matrices are
  // finite
  auto elem = H_output(1, 0);
  BOOST_CHECK(!std::isnan(elem.real()) && !std::isinf(elem.real()) &&
              !std::isnan(elem.imag()) && !std::isinf(elem.imag()));
}

BOOST_AUTO_TEST_CASE(test_hessenberg_without_q_matrix) {
  // Test Hessenberg decomposition returning only H matrix
  types::shape input_shape = {2, 2};
  auto hessenberg_block =
      decomp_hessenberg_sync_f::make(input_shape, false); // compute_q = false

  // Create input matrix
  Eigen::Matrix<float, 2, 2> input_matrix;
  input_matrix << 4.0f, 1.0f, 2.0f, 3.0f;

  // Prepare input/output containers
  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      input_map(input_matrix.data(), 2, 2);
  inputs.push_back(&input_map);

  Eigen::Matrix<float, 2, 2> H_output;
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> H_map(
      H_output.data(), 2, 2);
  outputs.push_back(&H_map); // Only H matrix output

  // Perform decomposition
  auto result = hessenberg_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Verify H is upper Hessenberg (for 2x2, already in Hessenberg form)
  // The result should be similar to input but may be transformed
  BOOST_CHECK(!std::isnan(H_output(0, 0)) && !std::isinf(H_output(0, 0)));
  BOOST_CHECK(!std::isnan(H_output(0, 1)) && !std::isinf(H_output(0, 1)));
  BOOST_CHECK(!std::isnan(H_output(1, 0)) && !std::isinf(H_output(1, 0)));
  BOOST_CHECK(!std::isnan(H_output(1, 1)) && !std::isinf(H_output(1, 1)));
}

BOOST_AUTO_TEST_CASE(test_hessenberg_error_handling) {
  // Test error handling for invalid input shapes
  types::shape input_shape = {2, 2};
  auto hessenberg_block = decomp_hessenberg_sync_f::make(input_shape, true);

  // Test with wrong input size
  types::vector_const_matrix_map<float> inputs;
  types::vector_matrix_map<float> outputs;

  // Empty inputs should fail
  auto result = hessenberg_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::INVALID_SHAPE);

  // Wrong number of outputs should fail
  Eigen::Matrix<float, 2, 2> input_matrix =
      Eigen::Matrix<float, 2, 2>::Identity();
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
      input_map(input_matrix.data(), 2, 2);
  inputs.push_back(&input_map);

  // Should expect 2 outputs for compute_q=true, but provide only 1
  Eigen::Matrix<float, 2, 2> output_matrix;
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> output_map(
      output_matrix.data(), 2, 2);
  outputs.push_back(&output_map);

  result = hessenberg_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::INVALID_SHAPE);
}

BOOST_AUTO_TEST_CASE(test_hessenberg_eigenvalue_prerequisite) {
  // Test that Hessenberg decomposition preserves eigenvalues
  // (This is important since it's used as a first step in eigenvalue
  // algorithms)
  types::shape input_shape = {3, 3};
  auto hessenberg_block = decomp_hessenberg_sync_d::make(input_shape, true);

  // Create a symmetric matrix with known eigenvalues
  Eigen::Matrix<double, 3, 3> input_matrix;
  input_matrix << 4.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0, 1.0, 4.0;

  // Compute eigenvalues of original matrix
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3>> original_eigen(
      input_matrix);
  auto original_eigenvalues = original_eigen.eigenvalues();

  // Prepare input/output containers
  types::vector_const_matrix_map<double> inputs;
  types::vector_matrix_map<double> outputs;

  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
      input_map(input_matrix.data(), 3, 3);
  inputs.push_back(&input_map);

  Eigen::Matrix<double, 3, 3> Q_output, H_output;
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> Q_map(
      Q_output.data(), 3, 3);
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> H_map(
      H_output.data(), 3, 3);
  outputs.push_back(&Q_map);
  outputs.push_back(&H_map);

  // Perform decomposition
  auto result = hessenberg_block->operation(inputs, outputs);
  BOOST_CHECK_EQUAL(result, OperationReturn::SUCCESS);

  // Compute eigenvalues of Hessenberg matrix
  Eigen::EigenSolver<Eigen::Matrix<double, 3, 3>> hess_eigen(H_output);
  auto hess_eigenvalues = hess_eigen.eigenvalues();

  // Sort both sets of eigenvalues for comparison
  std::vector<double> orig_vals(original_eigenvalues.data(),
                                original_eigenvalues.data() +
                                    original_eigenvalues.size());
  std::vector<std::complex<double>> hess_vals(hess_eigenvalues.data(),
                                              hess_eigenvalues.data() +
                                                  hess_eigenvalues.size());

  std::sort(orig_vals.begin(), orig_vals.end());
  std::sort(hess_vals.begin(), hess_vals.end(),
            [](const std::complex<double> &a, const std::complex<double> &b) {
              return a.real() < b.real();
            });

  // Verify eigenvalues are preserved (should be real for symmetric matrix)
  double tolerance = 1e-10;
  for (size_t i = 0; i < orig_vals.size(); ++i) {
    BOOST_CHECK_SMALL(std::abs(hess_vals[i].imag()),
                      tolerance); // Should be real
    BOOST_CHECK_CLOSE(orig_vals[i], hess_vals[i].real(), tolerance * 100);
  }
}

} /* namespace linalg */
} /* namespace gr */