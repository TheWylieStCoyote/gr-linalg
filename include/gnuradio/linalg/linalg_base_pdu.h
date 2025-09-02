/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_BLOCK_SYNC_BASE_PDU_H
#define INCLUDED_BLOCK_SYNC_BASE_PDU_H

#include <functional>
#include <gnuradio/block.h>
#include <gnuradio/linalg/api.h>
#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/types.h>
#include <memory>
#include <pmt/pmt.h>
#include <string>
#include <vector>

namespace gr {
namespace linalg {

enum class PDU_UPDATE {
  DEFAULT,    // Default behavior: update output if any input is updated
  ANY_INPUT,  // Update output if any input is updated
  ALL_INPUTS, // Update output only if all inputs are updated
  CUSTOM      // Custom handling
};

enum class MESSAGE_HANDLER_MODE {
  NONE,    // No message handlers
  DEFAULT, // Use default message handlers
  CUSTOM   // Custom message handlers
};

enum class PDU_MATRIX_SIZE {
  FIXED,                   // Fixed size
  DYNAMIC = Eigen::Dynamic // Dynamic size
};

/*!
 * \brief Base class for synchronous linear algebra blocks
 *
 * This class provides the basic interface and functionality for synchronous
 * linear algebra blocks.
 */
template <class Scalar>
class LINALG_API linalg_base_pdu : virtual public linalg_base<Scalar>,
                                   public gr::block {

public:
  typedef std::shared_ptr<linalg_base_pdu<Scalar>> sptr;
  // typedef std::shared_ptr<std::function<void(pmt::pmt_t)>>
  // message_handler_sptr;

protected:
  PDU_UPDATE d_pdu_update; // Update strategy for PDUs
  MESSAGE_HANDLER_MODE
  d_generate_message_handlers;            // Message handler generation strategy
  std::vector<std::string> d_input_names; // Names of input ports
  std::vector<std::string> d_output_names; // Names of output ports

  // Vector of function pointers for handling messages
  std::vector<std::shared_ptr<std::function<void(pmt::pmt_t)>>>
      d_message_handlers;

  // Vector to store last seen values for each input port
  std::vector<std::vector<Scalar>> d_last_values; // Last seen values
  std::vector<bool> d_seen; // Track if inputs have been seen

public:
  /*!
   * \brief Constructor for linalg_base_pdu
   * \param name Name of the block
   * \param shape_inputs Vector of input shapes
   * \param shape_outputs Vector of output shapes
   * \param input_names Vector of input port names
   * \param output_names Vector of output port names
   * \param broadcast_type Type of broadcasting to apply to output shapes
   * \param tag_errors Tag errors for operation calls
   * \param pdu_errors Emit PDUs on errors for operation calls
   * \param pdu_update Update strategy for PDUs
   * \param generate_message_handlers Message handler generation strategy
   */
  linalg_base_pdu(
      const std::string &name,                      // Name of the block
      const types::vector_shapes &shape_inputs,     // Input shapes
      const std::vector<std::string> &input_names,  // Input port names
      const types::vector_shapes &shape_outputs,    // Output shapes
      const std::vector<std::string> &output_names, // Output port names
      array_broadcast_type
          broadcast_type,     // Type of broadcasting to apply to output shapes
      error_tag_t tag_errors, // Tag errors for operation calls
      error_pdu_p pdu_errors, // Emit PDUs on errors for operation calls
      PDU_UPDATE pdu_update = PDU_UPDATE::DEFAULT, // Update strategy for PDUs
      MESSAGE_HANDLER_MODE
          generate_message_handlers = // Message handler generation strategy
      MESSAGE_HANDLER_MODE::DEFAULT);

  virtual ~linalg_base_pdu() = default;

protected:
  /*!
   * \brief Handle incoming PDU messages if using default message handlers
   * \param msg The incoming PDU message
   * \param port_idx The index of the input port number corresponding to the
   * message port
   */
  virtual void _handle_pdu(const pmt::pmt_t &msg, const int port_idx);

  /*!
   * \brief Handle incoming PDU messages if using custom message handlers
   * \param msg The incoming PDU message
   */
  virtual int _work();
};

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_BLOCK_SYNC_BASE_PDU_H */
