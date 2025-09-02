/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_DECOMP_SVD_H
#define INCLUDED_LINALG_DECOMP_SVD_H

#include "gnuradio/linalg/linalg_base.h"
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
 * \brief SVD algorithm interface
 * \ingroup linalg
 */
template <typename Scalar>
class LINALG_API svd_algorithm {
public:
  typedef std::shared_ptr<svd_algorithm<Scalar>> sptr;
  virtual ~svd_algorithm() = default;
  virtual void
  decompose(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input,
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &U,
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &S,
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &V) = 0;
};

/*!
 * \brief JacobiSVD decomposition (accurate for small matrices)
 */
template <typename Scalar>
class LINALG_API eigen_jacobi_svd : public svd_algorithm<Scalar> {
public:
  void
  decompose(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input,
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &U,
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &S,
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &V) override;
};

/*!
 * \brief BDCSVD decomposition (divide-and-conquer, fast for large matrices)
 */
template <typename Scalar>
class LINALG_API eigen_bdc_svd : public svd_algorithm<Scalar> {
public:
  void
  decompose(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &input,
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &U,
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &S,
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &V) override;
};

template <typename Scalar>
class LINALG_API decomp_svd : virtual public linalg_base<Scalar> {
protected:
  std::shared_ptr<svd_algorithm<Scalar>> algorithm_;

public:
  decomp_svd(const types::shape &shape,
             std::shared_ptr<svd_algorithm<Scalar>> algorithm = nullptr);
  ~decomp_svd() override = default;

  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;

  void set_algorithm(std::shared_ptr<svd_algorithm<Scalar>> algo);

  static void validate_shape(const types::vector_shapes &input_shapes = {},
                             const types::vector_shapes &output_shapes = {},
                             const std::string &name = "decomp_svd");
  static types::vector_shapes
  compute_output_shapes(const types::vector_shapes &input_shapes);
};

/*!
 * \brief Synchronous SVD decomposition block
 */
template <typename Scalar>
class LINALG_API decomp_svd_sync : virtual public decomp_svd<Scalar>,
                                   virtual public linalg_base_sync<Scalar> {
public:
  typedef std::shared_ptr<decomp_svd_sync<Scalar>> sptr;

  /*!
   * \brief Create a new SVD decomposition sync block
   * \param shape Shape of the input matrix
   */
  static sptr make(const types::shape &shape,
                   std::shared_ptr<svd_algorithm<Scalar>> algorithm = nullptr);

  decomp_svd_sync(const types::shape &shape,
                  std::shared_ptr<svd_algorithm<Scalar>> algorithm = nullptr)
      : decomp_svd<Scalar>(shape, algorithm) {}
};

/*!
 * \brief PDU SVD decomposition block
 */
template <typename Scalar>
class LINALG_API decomp_svd_pdu : virtual public linalg_base_pdu<Scalar> {
public:
  typedef std::shared_ptr<decomp_svd_pdu<Scalar>> sptr;

  static sptr make() {
    return gnuradio::make_block_sptr<decomp_svd_pdu<Scalar>>();
  }

  decomp_svd_pdu()
      : linalg_base_pdu<Scalar>("decomp_svd_pdu", {}, {}, {}, {},
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
using decomp_svd_sync_f = decomp_svd_sync<float>;
using decomp_svd_sync_d = decomp_svd_sync<double>;
using decomp_svd_pdu_f = decomp_svd_pdu<float>;
using decomp_svd_pdu_d = decomp_svd_pdu<double>;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_DECOMP_SVD_H */