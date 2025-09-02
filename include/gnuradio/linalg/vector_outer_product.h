/* -*- c++ -*- */
/*
 * Copyright 2024 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_LINALG_VECTOR_OUTER_PRODUCT_H
#define INCLUDED_LINALG_VECTOR_OUTER_PRODUCT_H

#include <gnuradio/linalg/api.h>
#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <memory>

namespace gr {
namespace linalg {

/**
 * @brief Vector outer product operation (tensor product)
 * @ingroup linalg
 *
 * Computes the outer product of two vectors: result(i,j) = a(i) * b(j)
 *
 * The outer product of vectors a ∈ ℝᵐ and b ∈ ℝⁿ produces a matrix C ∈ ℝᵐˣⁿ
 * where C(i,j) = a(i) × b(j). This is also known as the tensor product of
 * vectors.
 *
 * **Mathematical Operation:**
 * Given vectors a = [a₁, a₂, ..., aₘ]ᵀ and b = [b₁, b₂, ..., bₙ]ᵀ
 * Result matrix C = a ⊗ b where:
 * C = [a₁b₁  a₁b₂  ...  a₁bₙ]
 *     [a₂b₁  a₂b₂  ...  a₂bₙ]
 *     [  ⋮     ⋮    ⋱    ⋮ ]
 *     [aₘb₁  aₘb₂  ...  aₘbₙ]
 *
 * **Key Properties:**
 * - Input vectors can be of any length (including different lengths)
 * - Output is always a matrix of size m×n where m=len(a), n=len(b)
 * - Rank of output matrix is 1 (unless either vector is zero)
 * - Preserves complex arithmetic: (a+ib) ⊗ (c+id) handled correctly
 * - Commutative up to transpose: a ⊗ b = (b ⊗ a)ᵀ
 *
 * **Supported Types:** float, double, complex<float>, complex<double>
 *
 * **Input Requirements:**
 * - Two vectors (can be different lengths)
 * - Vectors are treated as column vectors regardless of input shape
 * - 1×N matrices are treated as N-element vectors
 * - N×1 matrices are treated as N-element vectors
 *
 * **Output Format:**
 * - Matrix of size m×n where m and n are the input vector lengths
 * - Element (i,j) = vector_a[i] × vector_b[j]
 *
 * **Applications:**
 * - Rank-1 matrix construction for matrix approximations
 * - Projection matrix construction: P = v⊗v / (v·v)
 * - Covariance matrix estimation: Σ ≈ (x-μ)⊗(x-μ)ᵀ
 * - Beamforming weight matrices
 * - Tensor product spaces in quantum computing
 * - Statistical analysis and pattern recognition
 *
 * **Performance Notes:**
 * - Time complexity: O(mn) where m, n are vector lengths
 * - Space complexity: O(mn) for output matrix
 * - No temporary matrices needed - direct computation
 * - Memory-efficient for large vectors due to streaming computation
 *
 * **Examples:**
 * ```
 * a = [1, 2, 3]ᵀ, b = [4, 5]ᵀ
 * Result = [1×4  1×5]   = [4   5 ]
 *          [2×4  2×5]     [8   10]
 *          [3×4  3×5]     [12  15]
 * ```
 *
 * ```cpp
 * // Usage example
 * auto block = vector_outer_product_sync<float>::make();
 * // Connect two vector inputs, get matrix output
 * ```
 */
template <typename Scalar>
class LINALG_API vector_outer_product : virtual public linalg_base<Scalar> {
public:
  typedef std::shared_ptr<vector_outer_product<Scalar>> sptr;

  /**
   * @brief Default constructor
   */
  vector_outer_product();

  /**
   * @brief Perform vector outer product operation
   * @param input_matrices Input vectors [vector_a, vector_b]
   * @param output_matrices Output matrix [result]
   * @return OperationReturn::SUCCESS on success, error code on failure
   */
  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;

  // Static validation and computation helpers
  static void validate_shapes(const types::vector_shapes &input_shapes,
                              const types::vector_shapes &output_shapes,
                              const std::string &name);

  static std::vector<size_t> compute_sizes(const types::vector_shapes &shapes);
  static types::vector_shapes
  compute_output_shapes(const types::vector_shapes &input_shapes);

  /**
   * @brief Check if shape represents a vector (1D array)
   * @param shape Shape to check
   * @return true if shape is 1×n, n×1, or conceptually 1D
   */
  static bool is_vector_shape(const types::shape &shape);

  /**
   * @brief Get vector length from shape, treating matrices as vectors
   * appropriately
   * @param shape Shape of input
   * @return Vector length (total elements for valid vector shapes)
   */
  static size_t get_vector_length(const types::shape &shape);

private:
  /**
   * @brief Core outer product computation
   * @param vector_a First vector (m elements)
   * @param vector_b Second vector (n elements)
   * @param result Output matrix (m×n)
   * @return OperationReturn status
   */
  OperationReturn compute_outer_product(
      const Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic,
                                           Eigen::Dynamic>> &vector_a,
      const Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic,
                                           Eigen::Dynamic>> &vector_b,
      Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
          &result);
};

/**
 * @brief Sync implementation of vector outer product
 */
template <typename Scalar>
class LINALG_API vector_outer_product_sync
    : virtual public vector_outer_product<Scalar>,
      virtual public linalg_base_sync<Scalar> {
public:
  typedef std::shared_ptr<vector_outer_product_sync<Scalar>> sptr;

  static sptr make();
  vector_outer_product_sync();
};

/**
 * @brief PDU implementation of vector outer product
 */
template <typename Scalar>
class LINALG_API vector_outer_product_pdu
    : virtual public vector_outer_product<Scalar>,
      virtual public linalg_base_pdu<Scalar> {
public:
  typedef std::shared_ptr<vector_outer_product_pdu<Scalar>> sptr;

  static sptr make();

protected:
  vector_outer_product_pdu();
};

// Type aliases for convenience
typedef vector_outer_product<float> vector_outer_product_f;
typedef vector_outer_product<double> vector_outer_product_d;
typedef vector_outer_product<std::complex<float>> vector_outer_product_cf;
typedef vector_outer_product<std::complex<double>> vector_outer_product_cd;

typedef vector_outer_product_sync<float> vector_outer_product_sync_f;
typedef vector_outer_product_sync<double> vector_outer_product_sync_d;
typedef vector_outer_product_sync<std::complex<float>>
    vector_outer_product_sync_cf;
typedef vector_outer_product_sync<std::complex<double>>
    vector_outer_product_sync_cd;

typedef vector_outer_product_pdu<float> vector_outer_product_pdu_f;
typedef vector_outer_product_pdu<double> vector_outer_product_pdu_d;
typedef vector_outer_product_pdu<std::complex<float>>
    vector_outer_product_pdu_cf;
typedef vector_outer_product_pdu<std::complex<double>>
    vector_outer_product_pdu_cd;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_VECTOR_OUTER_PRODUCT_H */