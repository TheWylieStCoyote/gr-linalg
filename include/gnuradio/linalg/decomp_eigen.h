/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_DECOMP_EIGEN_H
#define INCLUDED_LINALG_DECOMP_EIGEN_H

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
 * \brief Eigenvalue decomposition algorithm interface
 * \ingroup linalg
 */
template <typename Scalar>
class LINALG_API eigen_algorithm {
public:
  virtual ~eigen_algorithm() = default;

  virtual std::string name() const = 0;

  /*!
   * \brief Perform the eigenvalue decomposition
   * \param input Input matrix
   * \param eigenvalues Output eigenvalues
   * \param eigenvectors Output eigenvectors
   */
  virtual void decompose(
      const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input,
      Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &eigenvalues,
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &eigenvectors) = 0;
};

/*!
 * \brief General eigenvalue solver using ComplexEigenSolver
 */
template <typename Scalar>
class LINALG_API eigen_general_solver : public eigen_algorithm<Scalar> {
public:
  std::string name() const override { return "general"; }

  void
  decompose(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input,
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &eigenvalues,
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &eigenvectors)
      override;
};

/*!
 * \brief Self-adjoint eigenvalue solver (for symmetric/Hermitian matrices)
 */
template <typename Scalar>
class LINALG_API eigen_selfadjoint_solver : public eigen_algorithm<Scalar> {
public:
  std::string name() const override { return "selfadjoint"; }

  void
  decompose(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input,
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &eigenvalues,
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &eigenvectors)
      override;
};

/*!
 * \brief Base block: single input (NxN), two outputs: (Nx1) eigenvalues and
 * (NxN) eigenvectors
 */
template <typename Scalar>
class LINALG_API decomp_eigen : virtual public linalg_base<Scalar> {
protected:
  std::shared_ptr<eigen_algorithm<Scalar>> algorithm_;

public:
  typedef std::shared_ptr<decomp_eigen<Scalar>> sptr;

  decomp_eigen(const types::shape &shape,
               const std::string &name = "decomp_eigen");

  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;

  void set_algorithm(std::shared_ptr<eigen_algorithm<Scalar>> algo) {
    algorithm_ = std::move(algo);
  }

  static types::vector_shapes
  compute_output_shapes(const types::vector_shapes &input_shapes);
  static void validate_shape(const types::vector_shapes &input_shapes = {},
                             const types::vector_shapes &output_shapes = {},
                             const std::string &name = "decomp_eigen");
  static std::vector<size_t> compute_sizes(const types::vector_shapes &shapes);
};

/*!
 * \brief Synchronous eigenvalue decomposition block (factory-only)
 */
template <typename Scalar>
class LINALG_API decomp_eigen_sync : virtual public decomp_eigen<Scalar>,
                                     virtual public linalg_base_sync<Scalar> {
public:
  typedef std::shared_ptr<decomp_eigen_sync<Scalar>> sptr;
  static sptr make(const types::shape &shape = {});

protected:
  decomp_eigen_sync(const types::shape &shape) : decomp_eigen<Scalar>(shape) {}
};

/*!
 * \brief Minimal PDU eigenvalue decomposition block (for tests)
 */
template <typename Scalar>
class LINALG_API decomp_eigen_pdu : virtual public linalg_base_pdu<Scalar> {
public:
  typedef std::shared_ptr<decomp_eigen_pdu<Scalar>> sptr;
  static sptr make() {
    return gnuradio::make_block_sptr<decomp_eigen_pdu<Scalar>>();
  }

  decomp_eigen_pdu()
      : linalg_base_pdu<Scalar>("decomp_eigen_pdu", {}, {}, {}, {},
                                array_broadcast_type::NONE, error_tag_t::NONE,
                                error_pdu_p::NONE, PDU_UPDATE::DEFAULT,
                                MESSAGE_HANDLER_MODE::DEFAULT) {}

  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> & /*ins*/,
            types::vector_matrix_map<Scalar> & /*outs*/) override {
    return OperationReturn::SUCCESS;
  }
};

// Convenient type aliases
using decomp_eigen_sync_f = decomp_eigen_sync<float>;
using decomp_eigen_sync_d = decomp_eigen_sync<double>;
// Complex types are not instantiated yet for multi-output API
// using decomp_eigen_sync_c = decomp_eigen_sync<std::complex<float>>;
// using decomp_eigen_sync_z = decomp_eigen_sync<std::complex<double>>;
using decomp_eigen_pdu_f = decomp_eigen_pdu<float>;
using decomp_eigen_pdu_d = decomp_eigen_pdu<double>;
// using decomp_eigen_pdu_c = decomp_eigen_pdu<std::complex<float>>;
// using decomp_eigen_pdu_z = decomp_eigen_pdu<std::complex<double>>;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_DECOMP_EIGEN_H */
