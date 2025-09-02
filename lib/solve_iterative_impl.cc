/* -*- c++ -*- */
/*
 * Copyright 2025 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <complex>
#include <gnuradio/io_signature.h>
#include <gnuradio/linalg/performance_profiler.h>
#include <gnuradio/linalg/solve_iterative.h>
#include <sstream>
#include <stdexcept>

namespace gr {
namespace linalg {

// Conjugate Gradient algorithm implementation
template <typename Scalar>
int cg_algorithm<Scalar>::solve(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &A,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &b,
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &x, int max_iterations,
    typename Eigen::NumTraits<Scalar>::Real tolerance) {
  Eigen::ConjugateGradient<
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>,
      Eigen::Lower | Eigen::Upper>
      solver;
  solver.compute(A);
  solver.setMaxIterations(max_iterations);
  solver.setTolerance(tolerance);

  x = solver.solveWithGuess(b, x);

  return (solver.info() == Eigen::Success) ? solver.iterations() : -1;
}

// BiCGStab algorithm implementation
template <typename Scalar>
int bicgstab_algorithm<Scalar>::solve(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &A,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &b,
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &x, int max_iterations,
    typename Eigen::NumTraits<Scalar>::Real tolerance) {
  Eigen::BiCGSTAB<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> solver;
  solver.compute(A);
  solver.setMaxIterations(max_iterations);
  solver.setTolerance(tolerance);

  x = solver.solveWithGuess(b, x);

  return (solver.info() == Eigen::Success) ? solver.iterations() : -1;
}

// GMRES algorithm implementation
template <typename Scalar>
int gmres_algorithm<Scalar>::solve(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &A,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &b,
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &x, int max_iterations,
    typename Eigen::NumTraits<Scalar>::Real tolerance) {
  // Note: Eigen doesn't have built-in GMRES, so we use BiCGSTAB as fallback
  // In a full implementation, we would implement GMRES from scratch
  Eigen::BiCGSTAB<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> solver;
  solver.compute(A);
  solver.setMaxIterations(max_iterations);
  solver.setTolerance(tolerance);

  x = solver.solveWithGuess(b, x);

  return (solver.info() == Eigen::Success) ? solver.iterations() : -1;
}

// LSCG algorithm implementation
template <typename Scalar>
int lscg_algorithm<Scalar>::solve(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &A,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &b,
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &x, int max_iterations,
    typename Eigen::NumTraits<Scalar>::Real tolerance) {
  Eigen::LeastSquaresConjugateGradient<
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
      solver;
  solver.compute(A);
  solver.setMaxIterations(max_iterations);
  solver.setTolerance(tolerance);

  x = solver.solveWithGuess(b, x);

  return (solver.info() == Eigen::Success) ? solver.iterations() : -1;
}

// Base class implementation
template <typename Scalar>
solve_iterative<Scalar>::solve_iterative(
    const types::shape &shape_a, const types::shape &shape_b,
    iterative_method method, int max_iterations,
    typename Eigen::NumTraits<Scalar>::Real tolerance, bool use_initial_guess,
    std::shared_ptr<iterative_algorithm<Scalar>> algorithm)
    : linalg_base<Scalar>(), shape_a_(shape_a), shape_b_(shape_b),
      method_(method), max_iterations_(max_iterations), tolerance_(tolerance),
      use_initial_guess_(use_initial_guess), algorithm_(algorithm) {
  if (!algorithm_) {
    set_method(method_);
  }
}

template <typename Scalar>
void solve_iterative<Scalar>::set_method(iterative_method method) {
  method_ = method;

  // Create appropriate algorithm based on method
  switch (method_) {
  case iterative_method::CG:
    algorithm_ = std::make_shared<cg_algorithm<Scalar>>();
    break;
  case iterative_method::BICGSTAB:
    algorithm_ = std::make_shared<bicgstab_algorithm<Scalar>>();
    break;
  case iterative_method::GMRES:
    algorithm_ = std::make_shared<gmres_algorithm<Scalar>>();
    break;
  case iterative_method::LSCG:
    algorithm_ = std::make_shared<lscg_algorithm<Scalar>>();
    break;
  case iterative_method::AUTO:
  default:
    // Auto-select based on matrix properties
    if (shape_a_[0] == shape_a_[1]) {
      // Square matrix - use BiCGStab as general purpose
      algorithm_ = std::make_shared<bicgstab_algorithm<Scalar>>();
    } else {
      // Rectangular matrix - use LSCG
      algorithm_ = std::make_shared<lscg_algorithm<Scalar>>();
    }
    break;
  }
}

template <typename Scalar>
void solve_iterative<Scalar>::set_algorithm(
    std::shared_ptr<iterative_algorithm<Scalar>> algorithm) {
  if (!algorithm) {
    throw std::invalid_argument("solve_iterative: algorithm cannot be null");
  }
  algorithm_ = algorithm;
}

template <typename Scalar>
OperationReturn solve_iterative<Scalar>::operation(
    types::vector_const_matrix_map<Scalar> &input_matrices,
    types::vector_matrix_map<Scalar> &output_matrices) {
  // Validate inputs
  size_t expected_inputs = use_initial_guess_ ? 3 : 2; // A, b, [x0]
  if (input_matrices.size() != expected_inputs) {
    return OperationReturn::INVALID_SHAPE;
  }

  if (output_matrices.size() != 1) {
    return OperationReturn::INVALID_SHAPE;
  }

  const auto &A = *input_matrices[0];
  PROFILE_LINALG_OPERATION("solve_iterative", A.rows() * A.cols());
  const auto &b = *input_matrices[1];
  auto &x = *output_matrices[0];

  // Validate matrix dimensions
  if (A.rows() != static_cast<Eigen::Index>(shape_a_[0]) ||
      A.cols() != static_cast<Eigen::Index>(shape_a_[1])) {
    return OperationReturn::INVALID_SHAPE;
  }

  if (b.rows() != static_cast<Eigen::Index>(shape_b_[0]) ||
      b.cols() != static_cast<Eigen::Index>(shape_b_[1])) {
    return OperationReturn::INVALID_SHAPE;
  }

  if (A.rows() != b.rows()) {
    return OperationReturn::INVALID_SHAPE;
  }

  try {
    // Convert to column vector for solver
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> b_vec = b.col(0);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> x_vec(A.cols());

    // Set initial guess
    if (use_initial_guess_ && input_matrices.size() == 3) {
      const auto &x0 = *input_matrices[2];
      x_vec = x0.col(0);
    } else {
      x_vec.setZero();
    }

    // Solve the system
    int iterations =
        algorithm_->solve(A, b_vec, x_vec, max_iterations_, tolerance_);

    if (iterations < 0) {
      return OperationReturn::FAILURE;
    }

    // Copy result to output
    x = x_vec;

    return OperationReturn::SUCCESS;

  } catch (const std::exception &e) {
    return OperationReturn::FAILURE;
  }
}

template <typename Scalar>
void solve_iterative<Scalar>::validate_shape(
    const types::vector_shapes &input_shapes,
    const types::vector_shapes &output_shapes, const std::string &name) {
  if (input_shapes.size() < 2 || input_shapes.size() > 3) {
    std::ostringstream oss;
    oss << name << ": Expected 2-3 input matrices, got " << input_shapes.size();
    throw std::invalid_argument(oss.str());
  }

  if (output_shapes.size() != 1) {
    std::ostringstream oss;
    oss << name << ": Expected 1 output matrix, got " << output_shapes.size();
    throw std::invalid_argument(oss.str());
  }

  const auto &shape_a = input_shapes[0];
  const auto &shape_b = input_shapes[1];
  const auto &shape_x = output_shapes[0];

  if (shape_a.size() != 2) {
    std::ostringstream oss;
    oss << name << ": Matrix A must be 2D, got " << shape_a.size() << "D";
    throw std::invalid_argument(oss.str());
  }

  if (shape_b.size() != 2 || shape_b[1] != 1) {
    std::ostringstream oss;
    oss << name << ": Vector b must be [m, 1], got [" << shape_b[0] << ", "
        << shape_b[1] << "]";
    throw std::invalid_argument(oss.str());
  }

  if (shape_a[0] != shape_b[0]) {
    std::ostringstream oss;
    oss << name << ": Matrix A rows (" << shape_a[0]
        << ") must match vector b rows (" << shape_b[0] << ")";
    throw std::invalid_argument(oss.str());
  }

  if (shape_x[0] != shape_a[1] || shape_x[1] != 1) {
    std::ostringstream oss;
    oss << name << ": Output x must be [" << shape_a[1] << ", 1], got ["
        << shape_x[0] << ", " << shape_x[1] << "]";
    throw std::invalid_argument(oss.str());
  }

  // Validate initial guess if provided
  if (input_shapes.size() == 3) {
    const auto &shape_x0 = input_shapes[2];
    if (shape_x0 != shape_x) {
      std::ostringstream oss;
      oss << name << ": Initial guess x0 shape [" << shape_x0[0] << ", "
          << shape_x0[1] << "] must match output x shape [" << shape_x[0]
          << ", " << shape_x[1] << "]";
      throw std::invalid_argument(oss.str());
    }
  }
}

template <typename Scalar>
types::vector_shapes solve_iterative<Scalar>::compute_output_shapes(
    const types::vector_shapes &input_shapes) {
  // Don't validate output shapes when computing them - just validate inputs
  if (input_shapes.size() < 2 || input_shapes.size() > 3) {
    std::ostringstream oss;
    oss << "solve_iterative::compute_output_shapes: Expected 2-3 input "
           "matrices, got "
        << input_shapes.size();
    throw std::invalid_argument(oss.str());
  }

  const auto &shape_a = input_shapes[0];

  if (shape_a.size() != 2) {
    std::ostringstream oss;
    oss << "solve_iterative::compute_output_shapes: Matrix A must be 2D, got "
        << shape_a.size() << "D";
    throw std::invalid_argument(oss.str());
  }

  // Output x has shape [n, 1] where A is [m, n]
  types::shape output_shape = {shape_a[1], 1};

  return {output_shape};
}

// Sync block implementation
template <typename Scalar>
typename solve_iterative_sync<Scalar>::sptr solve_iterative_sync<Scalar>::make(
    const types::shape &shape_a, const types::shape &shape_b,
    iterative_method method, int max_iterations,
    typename Eigen::NumTraits<Scalar>::Real tolerance, bool use_initial_guess,
    std::shared_ptr<iterative_algorithm<Scalar>> algorithm) {
  return std::make_shared<solve_iterative_sync<Scalar>>(
      shape_a, shape_b, method, max_iterations, tolerance, use_initial_guess,
      algorithm);
}

template <typename Scalar>
solve_iterative_sync<Scalar>::solve_iterative_sync(
    const types::shape &shape_a, const types::shape &shape_b,
    iterative_method method, int max_iterations,
    typename Eigen::NumTraits<Scalar>::Real tolerance, bool use_initial_guess,
    std::shared_ptr<iterative_algorithm<Scalar>> algorithm)
    : linalg_base<Scalar>(
          "solve_iterative_sync",
          use_initial_guess
              ? types::vector_shapes{shape_a, shape_b, {shape_a[1], 1}}
              : types::vector_shapes{shape_a, shape_b},
          std::vector<std::string>{},
          solve_iterative<Scalar>::compute_output_shapes({shape_a, shape_b}),
          std::vector<std::string>{}, array_broadcast_type::CUSTOM),
      solve_iterative<Scalar>(shape_a, shape_b, method, max_iterations,
                              tolerance, use_initial_guess, algorithm),
      linalg_base_sync<Scalar>(
          "solve_iterative_sync",
          use_initial_guess
              ? types::vector_shapes{shape_a, shape_b, {shape_a[1], 1}}
              : types::vector_shapes{shape_a, shape_b},
          solve_iterative<Scalar>::compute_output_shapes({shape_a, shape_b})) {}

// PDU block implementation
template <typename Scalar>
typename solve_iterative_pdu<Scalar>::sptr solve_iterative_pdu<Scalar>::make(
    const types::shape &shape_a, const types::shape &shape_b,
    iterative_method method, int max_iterations,
    typename Eigen::NumTraits<Scalar>::Real tolerance, bool use_initial_guess,
    std::shared_ptr<iterative_algorithm<Scalar>> algorithm) {
  return std::make_shared<solve_iterative_pdu<Scalar>>(
      shape_a, shape_b, method, max_iterations, tolerance, use_initial_guess,
      algorithm);
}

template <typename Scalar>
solve_iterative_pdu<Scalar>::solve_iterative_pdu(
    const types::shape &shape_a, const types::shape &shape_b,
    iterative_method method, int max_iterations,
    typename Eigen::NumTraits<Scalar>::Real tolerance, bool use_initial_guess,
    std::shared_ptr<iterative_algorithm<Scalar>> algorithm)
    : linalg_base<Scalar>(
          "solve_iterative_pdu",
          use_initial_guess
              ? types::vector_shapes{shape_a, shape_b, {shape_a[1], 1}}
              : types::vector_shapes{shape_a, shape_b},
          use_initial_guess ? std::vector<std::string>{"A", "b", "x0"}
                            : std::vector<std::string>{"A", "b"},
          solve_iterative<Scalar>::compute_output_shapes({shape_a, shape_b}),
          std::vector<std::string>{"x"}, array_broadcast_type::CUSTOM),
      solve_iterative<Scalar>(shape_a, shape_b, method, max_iterations,
                              tolerance, use_initial_guess, algorithm),
      linalg_base_pdu<Scalar>(
          "solve_iterative_pdu",
          use_initial_guess
              ? types::vector_shapes{shape_a, shape_b, {shape_a[1], 1}}
              : types::vector_shapes{shape_a, shape_b},
          use_initial_guess ? std::vector<std::string>{"A", "b", "x0"}
                            : std::vector<std::string>{"A", "b"},
          solve_iterative<Scalar>::compute_output_shapes({shape_a, shape_b}),
          std::vector<std::string>{"x"}, array_broadcast_type::NONE,
          error_tag_t::NONE, error_pdu_p::NONE, PDU_UPDATE::DEFAULT,
          MESSAGE_HANDLER_MODE::DEFAULT) {}

// Explicit template instantiations
template class cg_algorithm<float>;
template class cg_algorithm<double>;
template class cg_algorithm<std::complex<float>>;
template class cg_algorithm<std::complex<double>>;

template class bicgstab_algorithm<float>;
template class bicgstab_algorithm<double>;
template class bicgstab_algorithm<std::complex<float>>;
template class bicgstab_algorithm<std::complex<double>>;

template class gmres_algorithm<float>;
template class gmres_algorithm<double>;
template class gmres_algorithm<std::complex<float>>;
template class gmres_algorithm<std::complex<double>>;

template class lscg_algorithm<float>;
template class lscg_algorithm<double>;
template class lscg_algorithm<std::complex<float>>;
template class lscg_algorithm<std::complex<double>>;

template class solve_iterative<float>;
template class solve_iterative<double>;
template class solve_iterative<std::complex<float>>;
template class solve_iterative<std::complex<double>>;

template class solve_iterative_sync<float>;
template class solve_iterative_sync<double>;
template class solve_iterative_sync<std::complex<float>>;
template class solve_iterative_sync<std::complex<double>>;

template class solve_iterative_pdu<float>;
template class solve_iterative_pdu<double>;
template class solve_iterative_pdu<std::complex<float>>;
template class solve_iterative_pdu<std::complex<double>>;

} /* namespace linalg */
} /* namespace gr */