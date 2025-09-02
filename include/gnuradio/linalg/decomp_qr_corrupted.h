/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_DECOMP_QR_H
#define INCLUDED_LINALG_DECOMP_QR_H

#include <Eigen/Dense>
#include <boost/shared_ptr.hpp>
#include <complex>
#include <gnuradio/block.h>
#include <gnuradio/linalg/api.h>
#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/types.h>
#include <gnuradio/sync_block.h>
#include <memory>

namespace gr {
namespace linalg {

/*!
 * \brief QR decomposition algorithm interface
 * \ingroup linalg
 */
template <typename Scalar, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic>
class LINALG_API qr_algorithm {
public:
  virtual ~qr_algorithm() = default;

  /*!
   * \brief Perform the QR decomposition
   * \param input Input matrix
   * \param Q Output orthogonal matrix
   * \param R Output upper triangular matrix
   */
  virtual void decompose(const Eigen::Matrix<Scalar, Rows, Cols> &input,
                         Eigen::Matrix<Scalar, Rows, Cols> &Q,
                         Eigen::Matrix<Scalar, Rows, Cols> &R) = 0;
};

/*!
 * \brief Householder QR decomposition (fast, stable)
 */
template <typename Scalar, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic>
class LINALG_API eigen_householder_qr
    : public qr_algorithm<Scalar, Rows, Cols> {
public:
  void decompose(const Eigen::Matrix<Scalar, Rows, Cols> &input,
                 Eigen::Matrix<Scalar, Rows, Cols> &Q,
                 Eigen::Matrix<Scalar, Rows, Cols> &R) override;
};

/*!
 * \brief Column pivoting QR decomposition (more robust for rank-deficient
 * matrices)
 */
template <typename Scalar, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic>
class LINALG_API eigen_colpivhouseholder_qr
    : public qr_algorithm<Scalar, Rows, Cols> {
public:
  void decompose(const Eigen::Matrix<Scalar, Rows, Cols> &input,
                 Eigen::Matrix<Scalar, Rows, Cols> &Q,
                 Eigen::Matrix<Scalar, Rows, Cols> &R) override;
};

template <typename Scalar>
class LINALG_API decomp_qr : virtual public linalg_base<Scalar> {
public:
  typedef boost::shared_ptr<decomp_qr<Scalar>> sptr;

  // Trivial inline ctor to allow most-derived impl to construct the virtual
  // base
  decomp_qr() {}
  decomp_qr(const types::shape & /*shape*/) {}

  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices);

  // Static helpers delegate to linalg_base/types
  static void validate_shape(const types::vector_shapes &input_shapes = {},
                             const types::vector_shapes &output_shapes = {},
                             const std::string &name = "decomp_qr") {
    linalg_base<Scalar>::validate_shapes(input_shapes, output_shapes, name);
  }

  static std::vector<size_t> compute_sizes(const types::vector_shapes &shapes) {
    return types::compute_sizes<Scalar>(shapes);
  }

  static types::vector_shapes
  compute_output_shapes(const types::vector_shapes &input_shapes) {
    return linalg_base<Scalar>::compute_output_shapes(input_shapes);
  }
};

/*!
 * \brief QR decomposition sync block
 * \ingroup linalg
 */
template <typename Scalar>
class LINALG_API decomp_qr_sync : virtual public decomp_qr<Scalar>,
                                  virtual public linalg_base_sync<Scalar> {
public:
  typedef boost::shared_ptr<decomp_qr_sync<Scalar>> sptr;

  // Constructor used by factory
  decomp_qr_sync(const types::shape &shape);

  // Factory
  static sptr make(const types::shape &shape);
};

template <typename Scalar>
class LINALG_API decomp_qr_pdu : virtual public decomp_qr<Scalar>,
                                 virtual public linalg_base_pdu<Scalar> {
public:
  // Convenience typedefs
  typedef boost::shared_ptr<decomp_qr_pdu<Scalar>> sptr;
  // Factory
  static sptr make();
};

// Sync blocks
using decomp_qr_sync_i = decomp_qr_sync<int>;
using decomp_qr_sync_f = decomp_qr_sync<float>;
using decomp_qr_sync_d = decomp_qr_sync<double>;
using decomp_qr_sync_c = decomp_qr_sync<std::complex<float>>;
using decomp_qr_sync_cd = decomp_qr_sync<std::complex<double>>;
// PDU blocks
using decomp_qr_pdu_i = decomp_qr_pdu<int>;
using decomp_qr_pdu_f = decomp_qr_pdu<float>;
using decomp_qr_pdu_d = decomp_qr_pdu<double>;
using decomp_qr_pdu_c = decomp_qr_pdu<std::complex<float>>;
using decomp_qr_pdu_cd = decomp_qr_pdu<std::complex<double>>;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_DECOMP_QR_H */
* \brief QR decomposition algorithm interface * \ingroup linalg * /
    template <typename Scalar, int Rows = Eigen::Dynamic,
              int Cols = Eigen::Dynamic>
    class LINALG_API qr_algorithm {
public:
  virtual ~qr_algorithm() = default;

  /*!
   * \brief Perform the QR decomposition
   * \param input Input matrix
   * \param Q Output orthogonal matrix
   * \param R Output upper triangular matrix
   */
  virtual void decompose(const Eigen::Matrix<Scalar, Rows, Cols> &input,
                         Eigen::Matrix<Scalar, Rows, Cols> &Q,
                         Eigen::Matrix<Scalar, Rows, Cols> &R) = 0;
};

/*!
 * \brief Householder QR decomposition (fast, stable)
 */
template <typename Scalar, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic>
class LINALG_API eigen_householder_qr
    : public qr_algorithm<Scalar, Rows, Cols> {
public:
  void decompose(const Eigen::Matrix<Scalar, Rows, Cols> &input,
                 Eigen::Matrix<Scalar, Rows, Cols> &Q,
                 Eigen::Matrix<Scalar, Rows, Cols> &R) override;
};

/*!
 * \brief Column pivoting QR decomposition (more robust for rank-deficient
 * matrices)
 */
template <typename Scalar, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic>
class LINALG_API eigen_colpivhouseholder_qr
    : public qr_algorithm<Scalar, Rows, Cols> {
public:
  void decompose(const Eigen::Matrix<Scalar, Rows, Cols> &input,
                 Eigen::Matrix<Scalar, Rows, Cols> &Q,
                 Eigen::Matrix<Scalar, Rows, Cols> &R) override;
};

/*!
 * \brief Base class for QR decomposition blocks
 */
template <typename Scalar>
class LINALG_API decomp_qr : virtual public linalg_base<Scalar> {
protected:
  std::shared_ptr<qr_algorithm<Scalar>> algorithm_;

public:
  typedef std::shared_ptr<decomp_qr<Scalar>> sptr;

  // Trivial inline ctor to allow most-derived impl to construct the virtual
  // base
  decomp_qr() {}
  decomp_qr(const types::shape &shape) {}

  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices);

  /*!
   * \brief Set a custom algorithm (for advanced users)
   */
  void set_algorithm(std::shared_ptr<qr_algorithm<Scalar>> algo);

  static types::vector_shapes
  compute_output_shapes(const types::vector_shapes &input_shapes);

  static void validate_shape(const types::vector_shapes &input_shapes = {},
                             const types::vector_shapes &output_shapes = {},
                             const std::string &name = "decomp_qr");

  static std::vector<size_t> compute_sizes(const types::vector_shapes &shapes);

  /*!
   * \brief Synchronous QR decomposition block
   */
  template <typename Scalar, int Rows = Eigen::Dynamic,
            int Cols = Eigen::Dynamic,
            class QRAlgorithm = eigen_householder_qr<Scalar>,
            class QRBackend = void, class QRBackendOptions = void>
  class LINALG_API decomp_qr_sync : public linalg_base_sync<Scalar> {
  private:
    std::shared_ptr<qr_algorithm<Scalar>> algorithm_;

  public:
    typedef std::shared_ptr<decomp_qr_sync<Scalar>> sptr;

    /*!
     * \brief Create a new QR decomposition sync block
     * \param shape Shape of the input matrix
     */
    static sptr make(const types::shape &shape = {}) {
      return gnuradio::make_block_sptr<decomp_qr_sync<Scalar>>(shape);
    }

    decomp_qr_sync(const types::shape &shape)
        : linalg_base_sync<Scalar>("decomp_qr_sync", {shape}, {shape}),
          algorithm_(std::make_shared<eigen_householder_qr<Scalar>>()) {

      // Validate matrix dimensions
      if (shape.size() != 2) {
        throw std::invalid_argument("QR decomposition requires matrix input");
      }
    }

    /*!
     * \brief Perform the QR decomposition operation
     */
    int operation(types::vector_const_matrix_map<Scalar> &input_matrices,
                  types::vector_matrix_map<Scalar> &output_matrices) override {

      if (input_matrices.empty() || output_matrices.size() < 2) {
        return 0; // Need at least 2 outputs: Q, R
      }

      try {
        // Work directly with mapped matrices to avoid unnecessary copies
        const auto &input_matrix = *input_matrices[0];
        auto &Q_output = *output_matrices[0]; // Q matrix
        auto &R_output = *output_matrices[1]; // R matrix

        // Create temporary matrices for algorithm interface
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> temp_input =
            input_matrix;
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> temp_Q, temp_R;

        algorithm_->decompose(temp_input, temp_Q, temp_R);

        Q_output = temp_Q;
        R_output = temp_R;

        return 1; // Successfully processed one matrix
      } catch (const std::exception &e) {
        // Log error and return 0
        return 0;
      }
    }

    /*!
     * \brief Set a custom algorithm (for advanced users)
     */
    void set_algorithm(std::shared_ptr<qr_algorithm<Scalar>> algo) {
      if (!algo) {
        throw std::invalid_argument("Algorithm cannot be null");
      }
      algorithm_ = algo;
    }
  };

  /*!
   * \brief PDU QR decomposition block
   */
  template <typename Scalar, int Rows = Eigen::Dynamic,
            int Cols = Eigen::Dynamic,
            class QRAlgorithm = eigen_householder_qr<Scalar>,
            class QRBackend = void, class QRBackendOptions = void>
  class LINALG_API decomp_qr_pdu : public linalg_base_pdu<Scalar> {
  private:
    std::shared_ptr<qr_algorithm<Scalar>> algorithm_;

  public:
    typedef std::shared_ptr<decomp_qr_pdu<Scalar>> sptr;

    static sptr make() {
      return gnuradio::make_block_sptr<decomp_qr_pdu<Scalar>>();
    }

    decomp_qr_pdu()
        : linalg_base_pdu<Scalar>("decomp_qr_pdu", {}, {}),
          algorithm_(std::make_shared<eigen_householder_qr<Scalar>>()) {}

    /*!
     * \brief Perform the QR decomposition operation for PDU
     */
    int operation(types::vector_const_matrix_map<Scalar> &input_matrices,
                  types::vector_matrix_map<Scalar> &output_matrices) override {

      if (input_matrices.empty() || output_matrices.size() < 2) {
        return 0; // Need at least 2 outputs: Q, R
      }

      try {
        // Work directly with mapped matrices to avoid unnecessary copies
        const auto &input_matrix = *input_matrices[0];
        auto &Q_output = *output_matrices[0]; // Q matrix
        auto &R_output = *output_matrices[1]; // R matrix

        // Create temporary matrices for algorithm interface
        Eigen::Matrix<Scalar, Rows, Cols> temp_input = input_matrix;
        Eigen::Matrix<Scalar, Rows, Cols> temp_Q, temp_R;

        algorithm_->decompose(temp_input, temp_Q, temp_R);

        Q_output = temp_Q;
        R_output = temp_R;

        return 1;
      } catch (const std::exception &e) {
        // Log error and return 0
        return 0;
      }
    }

    void set_algorithm(std::shared_ptr<qr_algorithm<Scalar>> algo) {
      if (!algo) {
        throw std::invalid_argument("Algorithm cannot be null");
      }
      algorithm_ = algo;
    }
  };

  // Convenient type aliases
  using decomp_qr_sync_i = decomp_qr_sync<int>;
  using decomp_qr_sync_f = decomp_qr_sync<float>;
  using decomp_qr_sync_d = decomp_qr_sync<double>;
  using decomp_qr_sync_c = decomp_qr_sync<std::complex<float>>;
  using decomp_qr_sync_z = decomp_qr_sync<std::complex<double>>;
  using decomp_qr_pdu_i = decomp_qr_pdu<int>;
  using decomp_qr_pdu_f = decomp_qr_pdu<float>;
  using decomp_qr_pdu_d = decomp_qr_pdu<double>;
  using decomp_qr_pdu_c = decomp_qr_pdu<std::complex<float>>;
  using decomp_qr_pdu_z = decomp_qr_pdu<std::complex<double>>;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_DECOMP_QR_H */
