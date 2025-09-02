/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "decomp_schur_impl.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <gnuradio/io_signature.h>
#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/types.h>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace gr {
namespace linalg {

namespace {
static inline types::vector_shapes _single_shape(const types::shape &s) {
  types::vector_shapes v;
  v.push_back(s);
  return v;
}

static inline types::vector_shapes _schur_out(const types::shape &shape,
                                              bool compute_u) {
  if (shape.size() != 2)
    throw std::invalid_argument("decomp_schur requires 2D shape");
  const int n = shape[0];
  if (shape[1] != n)
    throw std::invalid_argument("decomp_schur requires square matrix");

  types::vector_shapes out;
  if (compute_u) {
    out.push_back({n, n}); // Q matrix
  }
  out.push_back({n, n}); // T matrix
  return out;
}
} // namespace

// Advanced Schur algorithm implementation

template <typename Scalar>
void eigen_schur_advanced<Scalar>::decompose(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &Q,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &T) {
  // Fallback to basic decomposition for legacy interface
  schur_result<Scalar> result = decompose_advanced(input, schur_options{});
  Q = result.Q;
  T = result.T;
}

template <typename Scalar>
schur_result<Scalar> eigen_schur_advanced<Scalar>::decompose_advanced(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input,
    const schur_options &options) {

  schur_result<Scalar> result;
  auto working_matrix = input;

  // Balance matrix if requested
  if (options.balance_matrix) {
    result.balanced = balance_matrix(working_matrix, result.permutation,
                                     result.scale, options.balance_tolerance);
  }

  // Perform Schur decomposition based on scalar type and options
  if constexpr (std::is_same_v<Scalar, std::complex<float>> ||
                std::is_same_v<Scalar, std::complex<double>>) {
    // Complex matrices always get complex Schur form
    Eigen::ComplexSchur<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
        solver;
    solver.compute(working_matrix);

    if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Complex Schur decomposition failed");
    }

    result.Q = solver.matrixU();
    result.T = solver.matrixT();
    result.is_complex_schur = true;
  } else {
    // Real matrices: use complex Schur if requested, otherwise real Schur
    if (options.use_complex_schur) {
      // Force complex Schur for real matrices
      using ComplexScalar = std::complex<Scalar>;
      Eigen::Matrix<ComplexScalar, Eigen::Dynamic, Eigen::Dynamic>
          complex_input = working_matrix.template cast<ComplexScalar>();

      Eigen::ComplexSchur<
          Eigen::Matrix<ComplexScalar, Eigen::Dynamic, Eigen::Dynamic>>
          solver;
      solver.compute(complex_input);

      if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Forced complex Schur decomposition failed");
      }

      // Cast back to real (this may lose information if there are complex
      // components)
      result.Q = solver.matrixU().real().template cast<Scalar>();
      result.T = solver.matrixT().real().template cast<Scalar>();
      result.is_complex_schur = true;
    } else {
      // Standard real Schur
      Eigen::RealSchur<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
          solver;
      solver.compute(working_matrix);

      if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Real Schur decomposition failed");
      }

      result.Q = solver.matrixU();
      result.T = solver.matrixT();
      result.is_complex_schur = false;
    }
  }

  // Extract eigenvalues if requested
  if (options.extract_eigenvalues) {
    result.eigenvalues = extract_eigenvalues_from_schur(result.T);

    // Sort eigenvalues if requested
    if (options.sort_eigenvalues) {
      sort_schur_decomposition(result.Q, result.T, result.eigenvalues,
                               options.sort_tolerance);
    }
  }

  // Estimate condition number if requested
  if (options.compute_condition) {
    result.condition_estimate = estimate_condition_number(result.T);
  }

  return result;
}

template <typename Scalar>
bool eigen_schur_advanced<Scalar>::balance_matrix(
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &matrix,
    Eigen::VectorXd &permutation, Eigen::VectorXd &scale,
    double tolerance) const {

  const int n = matrix.rows();
  permutation.resize(n);
  scale.resize(n);

  // Initialize permutation and scale
  for (int i = 0; i < n; ++i) {
    permutation[i] = i;
    scale[i] = 1.0;
  }

  // Simple balancing algorithm (similar to LAPACK's DGEBAL)
  bool balanced = false;
  const int max_iterations = 10;

  for (int iter = 0; iter < max_iterations; ++iter) {
    bool converged = true;

    for (int i = 0; i < n; ++i) {
      typename Eigen::NumTraits<Scalar>::Real row_norm = 0;
      typename Eigen::NumTraits<Scalar>::Real col_norm = 0;

      // Calculate row and column norms
      for (int j = 0; j < n; ++j) {
        if (i != j) {
          row_norm += std::abs(matrix(i, j));
          col_norm += std::abs(matrix(j, i));
        }
      }

      if (row_norm > 0 && col_norm > 0) {
        typename Eigen::NumTraits<Scalar>::Real scale_factor =
            std::sqrt(col_norm / row_norm);

        if (std::abs(scale_factor - 1.0) > tolerance) {
          // Apply scaling
          matrix.row(i) *= scale_factor;
          matrix.col(i) /= scale_factor;
          scale[i] *= scale_factor;
          converged = false;
          balanced = true;
        }
      }
    }

    if (converged)
      break;
  }

  return balanced;
}

template <typename Scalar>
Eigen::Vector<std::complex<typename Eigen::NumTraits<Scalar>::Real>,
              Eigen::Dynamic>
eigen_schur_advanced<Scalar>::extract_eigenvalues_from_schur(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &T) const {

  using RealType = typename Eigen::NumTraits<Scalar>::Real;
  using ComplexType = std::complex<RealType>;

  const int n = T.rows();
  Eigen::Vector<ComplexType, Eigen::Dynamic> eigenvalues(n);

  if constexpr (std::is_same_v<Scalar, std::complex<float>> ||
                std::is_same_v<Scalar, std::complex<double>>) {
    // For complex Schur form, eigenvalues are simply the diagonal elements
    for (int i = 0; i < n; ++i) {
      eigenvalues[i] = static_cast<ComplexType>(T(i, i));
    }
  } else {
    // For real Schur form, extract eigenvalues from quasi-triangular matrix
    int i = 0;
    while (i < n) {
      if (i == n - 1 || std::abs(T(i + 1, i)) < 1e-12) {
        // Real eigenvalue
        eigenvalues[i] = ComplexType(T(i, i), 0);
        ++i;
      } else {
        // Complex conjugate pair
        RealType a = T(i, i);
        RealType b = T(i, i + 1);
        RealType c = T(i + 1, i);
        RealType d = T(i + 1, i + 1);

        // Solve the 2x2 eigenvalue problem
        RealType trace = a + d;
        RealType det = a * d - b * c;
        RealType discriminant = trace * trace - 4 * det;

        if (discriminant >= 0) {
          // Real eigenvalues (shouldn't happen in quasi-triangular form)
          RealType sqrt_disc = std::sqrt(discriminant);
          eigenvalues[i] = ComplexType((trace + sqrt_disc) / 2, 0);
          eigenvalues[i + 1] = ComplexType((trace - sqrt_disc) / 2, 0);
        } else {
          // Complex conjugate pair
          RealType real_part = trace / 2;
          RealType imag_part = std::sqrt(-discriminant) / 2;
          eigenvalues[i] = ComplexType(real_part, imag_part);
          eigenvalues[i + 1] = ComplexType(real_part, -imag_part);
        }
        i += 2;
      }
    }
  }

  return eigenvalues;
}

template <typename Scalar>
bool eigen_schur_advanced<Scalar>::sort_schur_decomposition(
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &Q,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &T,
    Eigen::Vector<std::complex<typename Eigen::NumTraits<Scalar>::Real>,
                  Eigen::Dynamic> &eigenvalues,
    double tolerance) const {

  // Simple sorting by magnitude (descending order)
  const int n = eigenvalues.size();
  std::vector<int> indices(n);
  std::iota(indices.begin(), indices.end(), 0);

  // Sort indices by eigenvalue magnitude
  std::sort(indices.begin(), indices.end(), [&eigenvalues](int i, int j) {
    return std::abs(eigenvalues[i]) > std::abs(eigenvalues[j]);
  });

  // Check if already sorted
  bool already_sorted = true;
  for (int i = 0; i < n; ++i) {
    if (indices[i] != i) {
      already_sorted = false;
      break;
    }
  }

  if (already_sorted) {
    return false;
  }

  // Apply permutation to eigenvalues
  auto sorted_eigenvalues = eigenvalues;
  for (int i = 0; i < n; ++i) {
    eigenvalues[i] = sorted_eigenvalues[indices[i]];
  }

  // Note: Full Schur vector sorting would require more complex operations
  // This is a simplified version that only sorts the eigenvalues

  return true;
}

template <typename Scalar>
typename Eigen::NumTraits<Scalar>::Real
eigen_schur_advanced<Scalar>::estimate_condition_number(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &T) const {

  // Simple condition number estimation based on diagonal elements of Schur form
  using RealType = typename Eigen::NumTraits<Scalar>::Real;

  const int n = T.rows();
  RealType min_abs = std::numeric_limits<RealType>::max();
  RealType max_abs = 0;

  for (int i = 0; i < n; ++i) {
    RealType abs_val = std::abs(T(i, i));
    if (abs_val > 0) {
      min_abs = std::min(min_abs, abs_val);
      max_abs = std::max(max_abs, abs_val);
    }
  }

  if (min_abs == std::numeric_limits<RealType>::max() || min_abs == 0) {
    return std::numeric_limits<RealType>::infinity();
  }

  return max_abs / min_abs;
}

// Basic algorithms (legacy)

template <typename Scalar>
void eigen_real_schur<Scalar>::decompose(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &Q,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &T) {
  static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>,
                "Real Schur only supports real types");

  using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  Eigen::RealSchur<Mat> solver(input);

  if (solver.info() != Eigen::Success)
    throw std::runtime_error("Real Schur decomposition failed");

  Q = solver.matrixU();
  T = solver.matrixT();
}

template <typename Scalar>
void eigen_complex_schur<Scalar>::decompose(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &Q,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &T) {
  using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  Eigen::ComplexSchur<Mat> solver(input);

  if (solver.info() != Eigen::Success)
    throw std::runtime_error("Complex Schur decomposition failed");

  Q = solver.matrixU();
  T = solver.matrixT();
}

// Base

template <typename Scalar>
decomp_schur<Scalar>::decomp_schur(
    const types::shape &shape, bool compute_u,
    std::shared_ptr<schur_algorithm<Scalar>> algorithm,
    const schur_options &options)
    : linalg_base<Scalar>(
          "decomp_schur", _single_shape(shape), std::vector<std::string>{},
          _schur_out(shape, compute_u), std::vector<std::string>{},
          array_broadcast_type::CUSTOM, error_tag_t::NONE, error_pdu_p::NONE),
      compute_u_(compute_u), options_(options) {
  if (shape.size() != 2 || shape[0] != shape[1])
    throw std::invalid_argument("decomp_schur requires square 2D shape");

  // Auto-select algorithm based on scalar type and options
  if (!algorithm) {
    // Check if advanced features are requested
    bool needs_advanced =
        options_.balance_matrix || options_.extract_eigenvalues ||
        options_.sort_eigenvalues || options_.compute_condition ||
        options_.use_complex_schur;

    if (needs_advanced) {
      // Use advanced algorithm for enhanced features
      algorithm_ = std::make_shared<eigen_schur_advanced<Scalar>>();
    } else {
      // Use basic algorithm for legacy compatibility
      if constexpr (std::is_same_v<Scalar, float> ||
                    std::is_same_v<Scalar, double>) {
        algorithm_ = std::make_shared<eigen_real_schur<Scalar>>();
      } else {
        algorithm_ = std::make_shared<eigen_complex_schur<Scalar>>();
      }
    }
  } else {
    algorithm_ = algorithm;
  }
}

template <typename Scalar>
OperationReturn
decomp_schur<Scalar>::operation(types::vector_const_matrix_map<Scalar> &ins,
                                types::vector_matrix_map<Scalar> &outs) {
  if (ins.size() != 1)
    return OperationReturn::INVALID_SHAPE;

  const int expected_outputs = compute_u_ ? 2 : 1;
  if (static_cast<int>(outs.size()) != expected_outputs)
    return OperationReturn::INVALID_SHAPE;

  try {
    const auto &A = *ins[0];
    const int n = A.rows();

    if (A.cols() != n)
      return OperationReturn::INVALID_SHAPE;

    // Map input matrix for Eigen processing
    Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
        input_matrix(A.data(), A.rows(), A.cols());

    // Check if advanced features are requested
    bool needs_advanced =
        options_.balance_matrix || options_.extract_eigenvalues ||
        options_.sort_eigenvalues || options_.compute_condition ||
        options_.use_complex_schur;

    if (needs_advanced) {
      // Use advanced decomposition with full results
      last_result_ =
          std::dynamic_pointer_cast<eigen_schur_advanced<Scalar>>(algorithm_)
              ->decompose_advanced(input_matrix, options_);

      // Copy results to outputs
      if (compute_u_) {
        auto &Q_out = *outs[0];
        auto &T_out = *outs[1];

        if (Q_out.rows() != n || Q_out.cols() != n || T_out.rows() != n ||
            T_out.cols() != n)
          return OperationReturn::INVALID_SHAPE;

        Q_out = last_result_.Q;
        T_out = last_result_.T;
      } else {
        auto &T_out = *outs[0];

        if (T_out.rows() != n || T_out.cols() != n)
          return OperationReturn::INVALID_SHAPE;

        T_out = last_result_.T;
      }
    } else {
      // Use basic decomposition for backward compatibility
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Q, T;
      algorithm_->decompose(input_matrix, Q, T);

      // Store basic results in last_result_ for consistency
      last_result_.Q = Q;
      last_result_.T = T;
      last_result_.balanced = false;
      last_result_.is_complex_schur = false;
      last_result_.condition_estimate = 0.0;

      // Copy results to outputs
      if (compute_u_) {
        auto &Q_out = *outs[0];
        auto &T_out = *outs[1];

        if (Q_out.rows() != n || Q_out.cols() != n || T_out.rows() != n ||
            T_out.cols() != n)
          return OperationReturn::INVALID_SHAPE;

        Q_out = Q;
        T_out = T;
      } else {
        auto &T_out = *outs[0];

        if (T_out.rows() != n || T_out.cols() != n)
          return OperationReturn::INVALID_SHAPE;

        T_out = T;
      }
    }

    return OperationReturn::SUCCESS;
  } catch (const std::exception &) {
    return OperationReturn::FAILURE;
  }
}

template <typename Scalar>
void decomp_schur<Scalar>::set_algorithm(
    std::shared_ptr<schur_algorithm<Scalar>> algo) {
  if (algo)
    algorithm_ = algo;
}

template <typename Scalar>
void decomp_schur<Scalar>::validate_shape(const types::vector_shapes &in,
                                          const types::vector_shapes &out,
                                          const std::string &) {
  linalg_base<Scalar>::validate_shapes(in, out, "decomp_schur",
                                       array_broadcast_type::CUSTOM);
  if (in.size() != 1 || in[0].size() != 2)
    throw std::invalid_argument("decomp_schur expects one 2D input shape");
  if (in[0][0] != in[0][1])
    throw std::invalid_argument("decomp_schur requires square matrix");
}

template <typename Scalar>
types::vector_shapes
decomp_schur<Scalar>::compute_output_shapes(const types::vector_shapes &in) {
  if (in.size() != 1)
    return {};
  // Note: We need to determine compute_u from context - for static method,
  // assume true by default
  return _schur_out(in[0], true);
}

// Sync impl

template <typename Scalar>
decomp_schur_sync_impl<Scalar>::decomp_schur_sync_impl(
    const types::shape &shape, bool compute_u,
    std::shared_ptr<schur_algorithm<Scalar>> algorithm,
    const schur_options &options)
    : linalg_base<Scalar>( // MUST explicitly initialize virtual base!
          "decomp_schur_sync", _single_shape(shape), std::vector<std::string>{},
          _schur_out(shape, compute_u), std::vector<std::string>{},
          array_broadcast_type::CUSTOM, error_tag_t::NONE, error_pdu_p::NONE),
      decomp_schur<Scalar>(shape, compute_u, algorithm, options),
      linalg_base_sync<Scalar>(
          "decomp_schur_sync", _single_shape(shape), std::vector<std::string>{},
          _schur_out(shape, compute_u), std::vector<std::string>{},
          array_broadcast_type::CUSTOM, error_tag_t::NONE, error_pdu_p::NONE,
          gr::block::TPP_ALL_TO_ALL),
      decomp_schur_sync<Scalar>(shape, compute_u, algorithm, options) {
  // Algorithm initialization is now handled in decomp_schur base constructor
}

// Sync class constructor implementations

template <typename Scalar>
decomp_schur_sync<Scalar>::decomp_schur_sync(
    const types::shape &shape, bool compute_u,
    std::shared_ptr<schur_algorithm<Scalar>> algorithm,
    const schur_options &options)
    : decomp_schur<Scalar>(shape, compute_u, algorithm, options) {}

// PDU class constructor implementations

template <typename Scalar>
decomp_schur_pdu<Scalar>::decomp_schur_pdu(
    const types::shape &shape, bool compute_u,
    std::shared_ptr<schur_algorithm<Scalar>> algorithm,
    const schur_options &options)
    : linalg_base_pdu<Scalar>("decomp_schur_pdu", {}, {}, {}, {},
                              array_broadcast_type::NONE, error_tag_t::NONE,
                              error_pdu_p::NONE, PDU_UPDATE::DEFAULT,
                              MESSAGE_HANDLER_MODE::DEFAULT),
      decomp_schur<Scalar>(shape, compute_u, algorithm, options) {}

// Factory

template <typename Scalar>
typename decomp_schur_sync<Scalar>::sptr decomp_schur_sync<Scalar>::make(
    const types::shape &shape, bool compute_u,
    std::shared_ptr<schur_algorithm<Scalar>> algorithm,
    const schur_options &options) {
  return gnuradio::make_block_sptr<decomp_schur_sync_impl<Scalar>>(
      shape, compute_u, algorithm, options);
}

template <typename Scalar>
typename decomp_schur_pdu<Scalar>::sptr decomp_schur_pdu<Scalar>::make(
    const types::shape &shape, bool compute_u,
    std::shared_ptr<schur_algorithm<Scalar>> algorithm,
    const schur_options &options) {
  return gnuradio::make_block_sptr<decomp_schur_pdu<Scalar>>(
      shape, compute_u, algorithm, options);
}

// Explicit instantiations

template class eigen_real_schur<float>;
template class eigen_real_schur<double>;

template class eigen_complex_schur<std::complex<float>>;
template class eigen_complex_schur<std::complex<double>>;

template class eigen_schur_advanced<float>;
template class eigen_schur_advanced<double>;
template class eigen_schur_advanced<std::complex<float>>;
template class eigen_schur_advanced<std::complex<double>>;

template class decomp_schur<float>;
template class decomp_schur<double>;
template class decomp_schur<std::complex<float>>;
template class decomp_schur<std::complex<double>>;

template class decomp_schur_sync<float>;
template class decomp_schur_sync<double>;
template class decomp_schur_sync<std::complex<float>>;
template class decomp_schur_sync<std::complex<double>>;

template class decomp_schur_pdu<float>;
template class decomp_schur_pdu<double>;
template class decomp_schur_pdu<std::complex<float>>;
template class decomp_schur_pdu<std::complex<double>>;

template class decomp_schur_sync_impl<float>;
template class decomp_schur_sync_impl<double>;
template class decomp_schur_sync_impl<std::complex<float>>;
template class decomp_schur_sync_impl<std::complex<double>>;

} // namespace linalg
} // namespace gr
