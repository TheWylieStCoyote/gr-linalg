/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_BASE_H
#define INCLUDED_LINALG_BASE_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cstddef>
#include <gnuradio/block.h>
#include <gnuradio/linalg/api.h>
#include <gnuradio/linalg/types.h>
#include <iostream>
#include <string>
#include <vector>

namespace gr {
namespace linalg {

// Broadcasting policy for shapes
enum class array_broadcast_type {
  NONE = 0, // No broadcasting, input shapes must be compatible with each other
  VECTOR =
      1, // Broadcast output shapes to each input, preserving vector structure
  MATRIX = 2, // Broadcast output shapes to input, preserving matrix structure
  CUSTOM = -1 // Custom broadcasting logic defined by derived class
};

// Whether a derived operation handles broadcasting internally
enum class broadcast_in_operation {
  IMPLEMENTS_BROADCAST = 0,
  DOES_NOT_IMPLEMENT_BROADCAST = 1
};

// Basic return codes for operations
enum class OperationReturn {
  SUCCESS = 0,
  FAILURE = -1,
  NOT_IMPLEMENTED = -2,
  INVALID_SHAPE = -3,
  OUT_OF_RANGE = -4
};

// Stream operator for OperationReturn (needed for Boost Test)
inline std::ostream &operator<<(std::ostream &os, const OperationReturn &op) {
  switch (op) {
  case OperationReturn::SUCCESS:
    return os << "SUCCESS";
  case OperationReturn::FAILURE:
    return os << "FAILURE";
  case OperationReturn::NOT_IMPLEMENTED:
    return os << "NOT_IMPLEMENTED";
  case OperationReturn::INVALID_SHAPE:
    return os << "INVALID_SHAPE";
  case OperationReturn::OUT_OF_RANGE:
    return os << "OUT_OF_RANGE";
  default:
    return os << "UNKNOWN(" << static_cast<int>(op) << ")";
  }
}

// Error handling policies (defined here to avoid cross-header coupling)
enum class error_tag_t {
  NONE = 0,   // No error tags
  TAGS = 1,   // Properly tagged update constant
  CUSTOM = -1 // Custom error tag logic defined by derived class
};

enum class error_pdu_p {
  NONE = 0,   // No PDUs on errors
  PDU = 1,    // Emit PDUs on errors for operation calls
  CUSTOM = -1 // Custom PDU error logic defined by derived class
};

/*!
 * \brief Utility base class for linear algebra blocks
 *
 * This class provides common utility functions for all linear algebra blocks.
 * It is designed to be inherited alongside gr::sync_block using multiple
 * inheritance. It doesn't change the work() method behavior - just provides
 * utilities.
 */
template <class Scalar>
class LINALG_API linalg_base {

protected:
  std::string d_name; // Name of the block
  types::vector_shapes shape_inputs;
  std::vector<std::string> input_names;
  std::vector<size_t> input_sizes; // Sizes (bytes) of input matrices
  types::vector_shapes shape_outputs;
  std::vector<std::string> output_names;
  std::vector<size_t> output_sizes;    // Sizes (bytes) of output matrices
  array_broadcast_type broadcast_type; // Broadcasting to apply to output shapes
  error_tag_t tag_errors;              // Tag errors for operation calls
  error_pdu_p pdu_errors; // Emit PDUs on errors for operation calls

  // Protected default constructor for virtual inheritance scenarios
  // This is used by intermediate classes that need to delegate virtual base
  // initialization to the most-derived class
  linalg_base();

public:
  // Convenience constructor (names defaulted to in_X/out_X)
  linalg_base(
      const std::string &name, const types::vector_shapes &shape_inputs,
      const types::vector_shapes &shape_outputs,
      const array_broadcast_type &broadcast_type = array_broadcast_type::NONE);

  // Full constructor (with explicit port names and error policies)
  linalg_base(
      const std::string &name,                      // Name of the block
      const types::vector_shapes &shape_inputs,     // Input shapes
      const std::vector<std::string> &input_names,  // Input port names
      const types::vector_shapes &shape_outputs,    // Output shapes
      const std::vector<std::string> &output_names, // Output port names
      const array_broadcast_type &broadcast_type = array_broadcast_type::NONE,
      const error_tag_t &tag_errors = error_tag_t::NONE,
      const error_pdu_p &pdu_errors = error_pdu_p::NONE);

  virtual ~linalg_base() = default;

  // Main operation to be implemented by derived classes
  virtual OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) = 0;

  // Compute output shapes from input shapes (default: identity)
  static types::vector_shapes compute_output_shapes(
      const types::vector_shapes &input_shapes,
      const array_broadcast_type &broadcast_type = array_broadcast_type::NONE);

  // Validate shape vectors and broadcast expectations
  static void validate_shapes(
      const types::vector_shapes &input_shapes,
      const types::vector_shapes &output_shapes = {},
      const std::string &name = "linalg_base",
      const array_broadcast_type &broadcast_type = array_broadcast_type::NONE);
};

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_BASE_H */
