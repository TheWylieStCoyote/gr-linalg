/* -*- c++ -*- */
/*
 * Copyright 2025 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_LINALG_MATRIX_KRONECKER_PRODUCT_H
#define INCLUDED_LINALG_MATRIX_KRONECKER_PRODUCT_H

#include <gnuradio/linalg/api.h>
#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/types.h>
#include <memory>

namespace gr {
namespace linalg {

/*!
 * \brief Matrix Kronecker Product computation A ⊗ B
 * \ingroup linalg
 *
 * This block computes the Kronecker product (tensor product) of two matrices:
 *
 * If A is m×n and B is p×q, then A ⊗ B is (mp)×(nq):
 *
 *     A ⊗ B = [a₁₁B  a₁₂B  ...  a₁ₙB ]
 *             [a₂₁B  a₂₂B  ...  a₂ₙB ]
 *             [ ⋮     ⋮    ⋱    ⋮   ]
 *             [aₘ₁B  aₘ₂B  ...  aₘₙB ]
 *
 * Mathematical Properties:
 * - (A ⊗ B)ᵀ = Aᵀ ⊗ Bᵀ  (transpose property)
 * - (A ⊗ B)(C ⊗ D) = (AC) ⊗ (BD)  (product property, when dimensions match)
 * - det(A ⊗ B) = det(A)ᵖ det(B)ᵐ  (determinant property)
 * - rank(A ⊗ B) = rank(A) × rank(B)  (rank property)
 *
 * Applications:
 * - Tensor algebra and multilinear operations
 * - Signal processing (2D filtering, image processing)
 * - Quantum mechanics (tensor product states)
 * - Graph theory (Cartesian product of graphs)
 * - Numerical methods (solving systems with structured matrices)
 */
template <typename Scalar>
class LINALG_API matrix_kronecker_product : virtual public linalg_base<Scalar> {
public:
  using sptr = std::shared_ptr<matrix_kronecker_product<Scalar>>;
};

// Forward declarations for implementation classes
template <typename Scalar>
class matrix_kronecker_product_sync_impl;
template <typename Scalar>
class matrix_kronecker_product_pdu_impl;

// Sync block type aliases with make functions
template <typename Scalar>
class LINALG_API matrix_kronecker_product_sync
    : public matrix_kronecker_product<Scalar>,
      public linalg_base_sync<Scalar> {
public:
  using sptr = std::shared_ptr<matrix_kronecker_product_sync<Scalar>>;
  matrix_kronecker_product_sync(const types::shape &shape_a,
                                const types::shape &shape_b);
  static sptr make(const types::shape &shape_a, const types::shape &shape_b);
};

using matrix_kronecker_product_sync_f = matrix_kronecker_product_sync<float>;
using matrix_kronecker_product_sync_d = matrix_kronecker_product_sync<double>;
using matrix_kronecker_product_sync_c =
    matrix_kronecker_product_sync<std::complex<float>>;
using matrix_kronecker_product_sync_cd =
    matrix_kronecker_product_sync<std::complex<double>>;

// PDU block type aliases with make functions
template <typename Scalar>
class LINALG_API matrix_kronecker_product_pdu
    : public matrix_kronecker_product<Scalar>,
      public linalg_base_pdu<Scalar> {
public:
  using sptr = std::shared_ptr<matrix_kronecker_product_pdu<Scalar>>;
  matrix_kronecker_product_pdu(const types::shape &shape_a,
                               const types::shape &shape_b);
  static sptr make(const types::shape &shape_a, const types::shape &shape_b);
};

using matrix_kronecker_product_pdu_f = matrix_kronecker_product_pdu<float>;
using matrix_kronecker_product_pdu_d = matrix_kronecker_product_pdu<double>;
using matrix_kronecker_product_pdu_c =
    matrix_kronecker_product_pdu<std::complex<float>>;
using matrix_kronecker_product_pdu_cd =
    matrix_kronecker_product_pdu<std::complex<double>>;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_KRONECKER_PRODUCT_H */