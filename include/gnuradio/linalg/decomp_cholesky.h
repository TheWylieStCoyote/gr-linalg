/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_DECOMP_CHOLESKY_H
#define INCLUDED_LINALG_DECOMP_CHOLESKY_H

#include <Eigen/Dense>
#include <complex>
#include <gnuradio/linalg/api.h>
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/types.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace gr {
namespace linalg {

/*!
 * Cholesky decomposition interfaces and blocks
 *
 * Design:
 * - Strategy interface cholesky_algorithm<Scalar>
 * - Concrete algorithms: eigen_llt_cholesky (LLT) and eigen_ldlt_cholesky
 * (LDLT)
 * - Base block decomp_cholesky<Scalar> implements operation() and validation
 * - Sync block decomp_cholesky_sync<Scalar> constructs io_signatures and calls
 * base
 *
 * Outputs: single lower-triangular factor L with the same shape as input (NxN)
 */

template <typename Scalar>
class LINALG_API cholesky_algorithm {
public:
  virtual ~cholesky_algorithm() = default;
  virtual void
  decompose(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input,
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &L) = 0;
};

// Standard Eigen LLT (requires SPD/Hermitian PD)
template <typename Scalar>
class LINALG_API eigen_llt_cholesky : public cholesky_algorithm<Scalar> {
public:
  void
  decompose(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input,
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &L) override;
};

// Eigen LDLT (more robust close to singular)
template <typename Scalar>
class LINALG_API eigen_ldlt_cholesky : public cholesky_algorithm<Scalar> {
public:
  void
  decompose(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input,
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &L) override;
};

// Base class matching LU/QR patterns
// Single input (NxN), single output (NxN)
template <typename Scalar>
class LINALG_API decomp_cholesky : virtual public linalg_base<Scalar> {
protected:
  std::shared_ptr<cholesky_algorithm<Scalar>> algorithm_;

public:
  typedef std::shared_ptr<decomp_cholesky<Scalar>> sptr;

  decomp_cholesky(const types::shape &shape,
                  const std::string &name = "decomp_cholesky");

  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;

  void set_algorithm(std::shared_ptr<cholesky_algorithm<Scalar>> algo);

  static types::vector_shapes
  compute_output_shapes(const types::vector_shapes &input_shapes);

  static void validate_shape(const types::vector_shapes &input_shapes = {},
                             const types::vector_shapes &output_shapes = {},
                             const std::string &name = "decomp_cholesky");

  static std::vector<size_t> compute_sizes(const types::vector_shapes &shapes);
};

// Sync block
// Most-derived impl initializes linalg_base and linalg_base_sync
// Factory-only construction

template <typename Scalar>
class LINALG_API decomp_cholesky_sync
    : virtual public decomp_cholesky<Scalar>,
      virtual public linalg_base_sync<Scalar> {
public:
  typedef std::shared_ptr<decomp_cholesky_sync<Scalar>> sptr;

  static sptr make(const types::shape &shape = {});

protected:
  decomp_cholesky_sync(const types::shape &shape)
      : decomp_cholesky<Scalar>(shape) {}
};

// Minimal PDU block to satisfy constructor/message-port tests
// Does not perform signal processing yet.

template <typename Scalar>
class LINALG_API decomp_cholesky_pdu : virtual public linalg_base_pdu<Scalar> {
public:
  typedef std::shared_ptr<decomp_cholesky_pdu<Scalar>> sptr;

  static sptr make() {
    return gnuradio::make_block_sptr<decomp_cholesky_pdu<Scalar>>();
  }

public:
  decomp_cholesky_pdu()
      : linalg_base_pdu<Scalar>(
            "decomp_cholesky_pdu", /*shape_inputs*/ {}, /*input_names*/ {},
            /*shape_outputs*/ {}, /*output_names*/ {},
            array_broadcast_type::NONE, error_tag_t::NONE, error_pdu_p::NONE,
            PDU_UPDATE::DEFAULT, MESSAGE_HANDLER_MODE::DEFAULT) {}

public:
  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> & /*ins*/,
            types::vector_matrix_map<Scalar> & /*outs*/) override {
    return OperationReturn::SUCCESS;
  }
};

// Convenient aliases
using decomp_cholesky_sync_f = decomp_cholesky_sync<float>;
using decomp_cholesky_sync_d = decomp_cholesky_sync<double>;
using decomp_cholesky_sync_c = decomp_cholesky_sync<std::complex<float>>;
using decomp_cholesky_sync_z = decomp_cholesky_sync<std::complex<double>>;
using decomp_cholesky_pdu_f = decomp_cholesky_pdu<float>;
using decomp_cholesky_pdu_d = decomp_cholesky_pdu<double>;
using decomp_cholesky_pdu_c = decomp_cholesky_pdu<std::complex<float>>;
using decomp_cholesky_pdu_z = decomp_cholesky_pdu<std::complex<double>>;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_DECOMP_CHOLESKY_H */