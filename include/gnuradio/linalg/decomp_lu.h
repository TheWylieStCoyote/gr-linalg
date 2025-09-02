/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_DECOMP_LU_H
#define INCLUDED_LINALG_DECOMP_LU_H

#include <Eigen/Dense>
#include <complex>
#include <gnuradio/linalg/api.h>
#include <gnuradio/linalg/linalg_base.h>
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
 * \brief LU decomposition algorithm interface
 * \ingroup linalg
 */
template <typename Scalar, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic>
class LINALG_API lu_algorithm {
public:
  virtual ~lu_algorithm() = default;

  /*!
   * \brief Perform the LU decomposition
   * \param input Input matrix
   * \param L Output lower triangular matrix
   * \param U Output upper triangular matrix
   * \param P Output permutation matrix
   */
  virtual void decompose(const Eigen::Matrix<Scalar, Rows, Cols> &input,
                         Eigen::Matrix<Scalar, Rows, Cols> &L,
                         Eigen::Matrix<Scalar, Rows, Cols> &U,
                         Eigen::Matrix<Scalar, Rows, Cols> &P) = 0;
};

/*!
 * \brief Standard Eigen LU decomposition with partial pivoting
 */
template <typename Scalar, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic>
class LINALG_API eigen_partial_pivlu : public lu_algorithm<Scalar, Rows, Cols> {
public:
  void decompose(const Eigen::Matrix<Scalar, Rows, Cols> &input,
                 Eigen::Matrix<Scalar, Rows, Cols> &L,
                 Eigen::Matrix<Scalar, Rows, Cols> &U,
                 Eigen::Matrix<Scalar, Rows, Cols> &P) override;
};

/*!
 * \brief Full pivoting LU decomposition (more stable but slower)
 */
template <typename Scalar, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic>
class LINALG_API eigen_full_pivlu : public lu_algorithm<Scalar, Rows, Cols> {
public:
  void decompose(const Eigen::Matrix<Scalar, Rows, Cols> &input,
                 Eigen::Matrix<Scalar, Rows, Cols> &L,
                 Eigen::Matrix<Scalar, Rows, Cols> &U,
                 Eigen::Matrix<Scalar, Rows, Cols> &P) override;
};

/*!
 * \brief Base class for LU decomposition blocks
 */
template <typename Scalar>
class LINALG_API decomp_lu : virtual public linalg_base<Scalar> {
protected:
  std::shared_ptr<lu_algorithm<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
      algorithm_;

public:
  typedef std::shared_ptr<decomp_lu<Scalar>> sptr;

  // Construct base with shapes so the most-derived impl doesn't need to touch
  // linalg_base directly
  decomp_lu(const types::shape &shape, const std::string &name = "decomp_lu");

  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices);

  /*!
   * \brief Set a custom algorithm (for advanced users)
   */
  void set_algorithm(
      std::shared_ptr<lu_algorithm<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
          algo);

  static types::vector_shapes
  compute_output_shapes(const types::vector_shapes &input_shapes);

  static void validate_shape(const types::vector_shapes &input_shapes = {},
                             const types::vector_shapes &output_shapes = {},
                             const std::string &name = "decomp_lu");

  static std::vector<size_t> compute_sizes(const types::vector_shapes &shapes);
};

/*!
 * \brief Synchronous LU decomposition block
 */
template <typename Scalar>
class LINALG_API decomp_lu_sync : virtual public decomp_lu<Scalar>,
                                  virtual public linalg_base_sync<Scalar> {

public:
  typedef std::shared_ptr<decomp_lu_sync<Scalar>> sptr;

  // Factory only
  static sptr make(const types::shape &shape = {});

protected:
  // Only implementations construct this virtual base
  decomp_lu_sync(const types::shape &shape) : decomp_lu<Scalar>(shape) {}
};

/*!
 * \brief PDU LU decomposition block - NOT YET IMPLEMENTED
 */
template <typename Scalar>
class LINALG_API decomp_lu_pdu : virtual public decomp_lu<Scalar>,
                                 public linalg_base_pdu<Scalar> {

public:
  typedef std::shared_ptr<decomp_lu_pdu<Scalar>> sptr;

  static sptr make(const types::shape &shape = {});
};

// Convenient type aliases
using decomp_lu_sync_i = decomp_lu_sync<int>;
using decomp_lu_sync_f = decomp_lu_sync<float>;
using decomp_lu_sync_c = decomp_lu_sync<std::complex<float>>;
using decomp_lu_sync_d = decomp_lu_sync<double>;
using decomp_lu_sync_cd = decomp_lu_sync<std::complex<double>>;
// PDU blocks - NOT YET IMPLEMENTED
/*
using decomp_lu_pdu_i = decomp_lu_pdu<int>;
using decomp_lu_pdu_f = decomp_lu_pdu<float>;
using decomp_lu_pdu_c = decomp_lu_pdu<std::complex<float>>;
using decomp_lu_pdu_d = decomp_lu_pdu<double>;
using decomp_lu_pdu_cd = decomp_lu_pdu<std::complex<double>>;
*/

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_DECOMP_LU_H */
