/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_VECTOR_TO_MATRIX_H
#define INCLUDED_LINALG_VECTOR_TO_MATRIX_H

#include <complex>
#include <gnuradio/linalg/api.h>
#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/types.h>
#include <memory>

namespace gr {
namespace linalg {

/*!\
 * \brief Convert vector to matrix format through reshaping
 * \ingroup linalg
 *
 * This block converts input vectors into matrices by arranging the vector
 * elements into the specified matrix shape. The total number of elements
 * must be preserved (input vector length = output matrix rows × cols).
 *
 * **Mathematical Operation:**
 * reshape(vector[n]) → matrix[rows, cols] where n = rows * cols
 *
 * **Input/Output:**
 * - Input: Vector [n] where n is the total number of elements
 * - Output: Matrix [rows, cols] where rows * cols = n
 *
 * **Performance:**
 * - Time complexity: O(1) - simple data reshaping
 * - Memory efficient: No additional memory allocation
 * - No mathematical computation involved, just data reorganization
 *
 * **Applications:**
 * - Converting 1D signal data to 2D matrix format
 * - Preparing data for matrix operations
 * - Signal processing applications requiring matrix structure
 * - Converting between vector and matrix representations
 */
template <typename Scalar>
class LINALG_API vector_to_matrix : virtual public linalg_base<Scalar> {
public:
  typedef std::shared_ptr<vector_to_matrix<Scalar>> sptr;

  /*!
   * \brief Constructor
   * \param input_shape Input vector shape [n]
   * \param output_shape Output matrix shape [rows, cols]
   */
  vector_to_matrix(const types::shape &input_shape,
                   const types::shape &output_shape);

  virtual ~vector_to_matrix() = default;

protected:
  /*!
   * \brief Perform the vector to matrix reshape operation
   */
  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;

  /*!
   * \brief Compute output shapes based on input shapes
   */
  static types::vector_shapes
  compute_output_shapes(const types::shape &input_shape,
                        const types::shape &output_shape);

private:
  types::shape d_input_shape;
  types::shape d_output_shape;
};

/*!
 * \brief Sync block wrapper for vector_to_matrix
 */
template <typename Scalar>
class LINALG_API vector_to_matrix_sync : public linalg_base_sync<Scalar>,
                                         public vector_to_matrix<Scalar> {
public:
  typedef std::shared_ptr<vector_to_matrix_sync<Scalar>> sptr;

  static sptr make(const types::shape &input_shape,
                   const types::shape &output_shape);

  vector_to_matrix_sync(const types::shape &input_shape,
                        const types::shape &output_shape);
};

/*!
 * \brief PDU block wrapper for vector_to_matrix
 */
template <typename Scalar>
class LINALG_API vector_to_matrix_pdu : public linalg_base_pdu<Scalar>,
                                        public vector_to_matrix<Scalar> {
public:
  typedef std::shared_ptr<vector_to_matrix_pdu<Scalar>> sptr;

  static sptr make(const types::shape &input_shape,
                   const types::shape &output_shape);

  vector_to_matrix_pdu(const types::shape &input_shape,
                       const types::shape &output_shape);
};

// Type aliases for common use cases
typedef vector_to_matrix_sync<float> vector_to_matrix_sync_f;
typedef vector_to_matrix_sync<double> vector_to_matrix_sync_d;
typedef vector_to_matrix_sync<std::complex<float>> vector_to_matrix_sync_cf;
typedef vector_to_matrix_sync<std::complex<double>> vector_to_matrix_sync_cd;

typedef vector_to_matrix_pdu<float> vector_to_matrix_pdu_f;
typedef vector_to_matrix_pdu<double> vector_to_matrix_pdu_d;
typedef vector_to_matrix_pdu<std::complex<float>> vector_to_matrix_pdu_cf;
typedef vector_to_matrix_pdu<std::complex<double>> vector_to_matrix_pdu_cd;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_VECTOR_TO_MATRIX_H */
