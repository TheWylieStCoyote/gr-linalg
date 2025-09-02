/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_MATRIX_MULTIPLY_H
#define INCLUDED_LINALG_MATRIX_MULTIPLY_H

#include <Eigen/Dense>
#include <gnuradio/linalg/api.h>
#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/types.h>
#include <gnuradio/sync_block.h>

namespace gr {
namespace linalg {

/*!
 * \brief Matrix multiplication block base class
 * \ingroup linalg
 * This class provides a base implementation for matrix multiplication blocks.
 * It inherits from linalg_base and provides the operation method for matrix
 * multiplication.
 * \tparam Scalar The scalar type of the matrices (e.g., float, double)
 * \tparam A_Rows Number of rows in the first matrix (default: Eigen::Dynamic)
 * \tparam A_Cols Number of columns in the first matrix (default:
 * Eigen::Dynamic) \tparam B_Rows Number of rows in the second matrix (default:
 * Eigen::Dynamic) \tparam B_Cols Number of columns in the second matrix
 * (default: Eigen::Dynamic)
 *
 */
template <class Scalar, int A_Rows = Eigen::Dynamic,
          int A_Cols = Eigen::Dynamic, int B_Rows = Eigen::Dynamic,
          int B_Cols = Eigen::Dynamic>
class LINALG_API matrix_multiply : virtual public linalg_base<Scalar> {
public:
  typedef std::shared_ptr<matrix_multiply<Scalar>> sptr;

  /*!
   * \brief Constructor for matrix_multiply
   * \param name Name of the block
   * \param shape_inputs Vector of input shapes of the input matrices
   * \param shape_outputs Vector of output shapes of the output matrices
   *
   * This constructor initializes the base class with the given name and shapes.
   * It is expected to be called by derived classes to set up the block's
   * properties.
   * \note The input and output shapes are expected to be in the format
   * expected by the block, and the block should handle any necessary type
   * conversions internally.
   * \throws std::invalid_argument if the shapes are invalid
   * \throws std::out_of_range if the output items are not large enough
   * \throws std::runtime_error if the operation cannot be performed
   * \throws std::logic_error if the block is not properly initialized
   * \throws std::bad_alloc if memory allocation fails
   * \throws std::exception for any other errors
   */
  matrix_multiply(const std::string &name = "matrix_multiply",
                  const types::vector_shapes &shape_inputs = {},
                  const types::vector_shapes &shape_outputs = {})
      : linalg_base<Scalar>(name, shape_inputs, shape_outputs){};

  /*! \brief Perform the matrix multiplication operation
   * \param input_matrices Vector of input matrices
   * \param output_matrices Vector of output matrices
   * \return OperationReturn status code
   * This method computes the product of the first input matrix with each
   * subsequent input matrix and stores the result in the corresponding
   * output matrix.
   * \throws std::invalid_argument if the input matrices are not compatible for
   * multiplication (i.e., the number of columns in the first matrix does not
   * match the number of rows in the second matrix)
   * \throws std::out_of_range if the output matrices are not large enough to
   * hold the result of the multiplication
   * \throws std::runtime_error if the operation cannot be performed
   */
  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override;

  /*!
   * \brief Validate the shapes of input and output matrices
   * \param input_shapes Vector of input shapes
   * \param output_shapes Vector of output shapes (optional)
   * \param name Name for error messages
   * This method checks that all dimensions in the input and output shapes are
   * positive and throws an exception if any dimension is non-positive.
   * It also checks that the output shapes match the expected output shapes
   * computed from the input shapes.
   * \throws std::invalid_argument if the shapes are invalid
   */
  static void validate_shape(const types::vector_shapes &input_shapes,
                             const types::vector_shapes &output_shapes = {},
                             const std::string &name = "linalg_base");

  /*!
   * \brief Calculate total number of elements in multiple shapes
   * \param shapes Vector of shape vectors
   * \return Vector of total sizes for each shape
   * \required implimented by Superclasses
   * This method computes the product of dimensions for each shape in the
   * vector and returns a vector of sizes.
   */
  static std::vector<size_t> compute_sizes(const types::vector_shapes &shapes);

  /*!
   * \brief Compute output shapes based on input shapes
   * \param input_shapes Vector of input shapes
   * \return Vector of output shapes
   * This method computes the expected output shapes based on the input shapes
   * and the specific operation being performed.
   */
  static types::vector_shapes
  compute_output_shapes(const types::vector_shapes &input_shapes);
};

/*!
 * \brief Matrix multiplication sync block
 * \ingroup linalg
 * This class provides a synchronous block for matrix multiplication.
 * It inherits from matrix_multiply and implements the linalg_base_sync
 * interface for synchronous operation.
 * \tparam Scalar The scalar type of the matrices (e.g., float, double)
 * \tparam A_Rows Number of rows in the first matrix (default: Eigen::Dynamic)
 * \tparam A_Cols Number of columns in the first matrix (default:
 * Eigen::Dynamic) \tparam B_Rows Number of rows in the second matrix (default:
 * Eigen::Dynamic) \tparam B_Cols Number of columns in the second matrix
 * (default: Eigen::Dynamic) This class provides a synchronous interface for
 * matrix multiplication, allowing it to be used in a synchronous processing
 * context. It is designed to be used with the GNU Radio framework and provides
 * the necessary methods for processing input and output matrices in a
 * synchronous manner.
 * \note The input and output shapes are expected to be in the format
 * expected by the block, and the block should handle any necessary type
 * conversions internally.
 * \throws std::runtime_error if the operation cannot be performed
 * \throws std::invalid_argument if the input shapes are invalid
 * \throws std::out_of_range if the output items are not large enough
 * \throws std::logic_error if the block is not properly initialized
 * \throws std::bad_alloc if memory allocation fails
 * \throws std::exception for any other errors
 */
template <class Scalar, int A_Rows = Eigen::Dynamic,
          int A_Cols = Eigen::Dynamic, int B_Rows = Eigen::Dynamic,
          int B_Cols = Eigen::Dynamic>
class LINALG_API matrix_multiply_sync
    : virtual public matrix_multiply<Scalar, A_Rows, A_Cols, B_Rows, B_Cols>,
      virtual public linalg_base_sync<Scalar> {
public:
  typedef std::shared_ptr<
      matrix_multiply_sync<Scalar, A_Rows, A_Cols, B_Rows, B_Cols>>
      sptr;

  /*! \brief Return a shared_ptr to a new instance of
   * linalg::matrix_multiply_sync. \param shape_0 Shape of the first input
   * matrix \param shape_1 Shape of the second input matrix This method creates
   * a new instance of the matrix_multiply_sync block with the specified shapes
   * for the input matrices. It is the public interface for creating new
   * instances of the block. \note The input shapes are expected to be in the
   * format expected by the block, and the block should handle any necessary
   * type conversions internally. \throws std::invalid_argument if the shapes
   * are invalid \throws std::out_of_range if the output items are not large
   * enough \throws std::runtime_error if the operation cannot be performed
   * \throws std::logic_error if the block is not properly initialized
   * \throws std::bad_alloc if memory allocation fails
   * \throws std::exception for any other errors
   */
  static sptr make(const types::shape &shape_0, const types::shape &shape_1);

protected:
  matrix_multiply_sync(const types::shape &shape_0, const types::shape &shape_1)
      : linalg_base<Scalar>("matrix_multiply_sync", {shape_0, shape_1},
                            {{shape_0[0], shape_1[1]}}),
        matrix_multiply<Scalar>("matrix_multiply_sync", {shape_0, shape_1},
                                {{shape_0[0], shape_1[1]}}),
        linalg_base_sync<Scalar>("matrix_multiply_sync", {shape_0, shape_1},
                                 {{shape_0[0], shape_1[1]}}) {}
};

/*!
 * \brief Matrix multiplication PDU block
 * \ingroup linalg
 * This class provides a PDU (Protocol Data Unit) block for matrix
 * multiplication. It inherits from matrix_multiply and implements the
 * linalg_base_pdu interface for PDU operation. \tparam Scalar The scalar type
 * of the matrices (e.g., float, double) \tparam A_Rows Number of rows in the
 * first matrix (default: Eigen::Dynamic) \tparam A_Cols Number of columns in
 * the first matrix (default: Eigen::Dynamic) \tparam B_Rows Number of rows in
 * the second matrix (default: Eigen::Dynamic) \tparam B_Cols Number of columns
 * in the second matrix (default: Eigen::Dynamic) This class provides a PDU
 * interface for matrix multiplication, allowing it to be used in a PDU
 * processing context. It is designed to be used with the GNU Radio framework
 * and provides the necessary methods for processing input and output matrices
 * in a PDU manner. \note The input and output shapes are expected to be in the
 * format expected by the block, and the block should handle any necessary type
 * conversions internally.
 * \throws std::invalid_argument if the shapes are invalid
 * \throws std::out_of_range if the output items are not large enough
 * \throws std::runtime_error if the operation cannot be performed
 * \throws std::logic_error if the block is not properly initialized
 * \throws std::bad_alloc if memory allocation fails
 * \throws std::exception for any other errors
 */
template <class Scalar, int A_Rows = Eigen::Dynamic,
          int A_Cols = Eigen::Dynamic, int B_Rows = Eigen::Dynamic,
          int B_Cols = Eigen::Dynamic>
class LINALG_API matrix_multiply_pdu
    : virtual public matrix_multiply<Scalar, A_Rows, A_Cols, B_Rows, B_Cols>,
      virtual public linalg_base_pdu<Scalar> {
public:
  typedef std::shared_ptr<
      matrix_multiply_pdu<Scalar, A_Rows, A_Cols, B_Rows, B_Cols>>
      sptr;

  /*!
   * \brief Return a shared_ptr to a new instance of
   * linalg::matrix_multiply_pdu. \param shape_0 Shape of the first input matrix
   * \param shape_1 Shape of the second input matrix
   * This method creates a new instance of the matrix_multiply_pdu block with
   * the specified shapes for the input matrices. It is the public interface for
   * creating new instances of the block.
   * \note The input shapes are expected to be in the format expected by the
   * block, and the block should handle any necessary type conversions
   * internally.
   * \throws std::invalid_argument if the shapes are invalid
   * \throws std::out_of_range if the output items are not large enough
   * \throws std::runtime_error if the operation cannot be performed
   * \throws std::logic_error if the block is not properly initialized
   * \throws std::bad_alloc if memory allocation fails
   * \throws std::exception for any other errors
   */
  static sptr make(const types::shape &shape_0, const types::shape &shape_1);

protected:
  matrix_multiply_pdu(const types::shape &shape_0, const types::shape &shape_1)
      : linalg_base<Scalar>("matrix_multiply_pdu", {shape_0, shape_1},
                            {{shape_0[0], shape_1[1]}}),
        matrix_multiply<Scalar>("matrix_multiply_pdu", {shape_0, shape_1},
                                {{shape_0[0], shape_1[1]}}),
        linalg_base_pdu<Scalar>(
            "matrix_multiply_pdu", {shape_0, shape_1}, // Input shapes
            {"in0", "in1"},                            // Input names
            {{shape_0[0], shape_1[1]}},                // Output shapes
            {"out"},                                   // Output names
            array_broadcast_type::CUSTOM,              // Skip strict validation
            error_tag_t::NONE, error_pdu_p::NONE) {}
};

// Sync blocks
using matrix_multiply_sync_i = matrix_multiply_sync<int>;
using matrix_multiply_sync_f = matrix_multiply_sync<float>;
using matrix_multiply_sync_d = matrix_multiply_sync<double>;
using matrix_multiply_sync_c = matrix_multiply_sync<std::complex<float>>;
using matrix_multiply_sync_z = matrix_multiply_sync<std::complex<double>>;

// PDU blocks
using matrix_multiply_pdu_i = matrix_multiply_pdu<int>;
using matrix_multiply_pdu_f = matrix_multiply_pdu<float>;
using matrix_multiply_pdu_d = matrix_multiply_pdu<double>;
using matrix_multiply_pdu_c = matrix_multiply_pdu<std::complex<float>>;
using matrix_multiply_pdu_z = matrix_multiply_pdu<std::complex<double>>;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_MULTIPLY_H */
