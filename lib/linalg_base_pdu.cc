/*
 * Copyright 2025 Wylie Standage-Beier.
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * linalg_base_pdu.cc - Stub implementation for PDU base template instantiations
 */

#include <complex>
#include <gnuradio/io_signature.h> // added
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <pmt/pmt.h> // ensure pmt::mp is available

namespace gr {
namespace linalg {

template <typename Scalar>
linalg_base_pdu<Scalar>::linalg_base_pdu(
    const std::string &name,                      // Name of the block
    const types::vector_shapes &shape_inputs,     // Input shapes
    const std::vector<std::string> &input_names,  // Input port names
    const types::vector_shapes &shape_outputs,    // Output shapes
    const std::vector<std::string> &output_names, // Output port names
    array_broadcast_type
        broadcast_type,     // Type of broadcasting to apply to output shapes
    error_tag_t tag_errors, // Tag errors for operation calls
    error_pdu_p pdu_errors, // Emit PDUs on errors for operation calls
    PDU_UPDATE pdu_update,  // Update strategy for PDUs
    MESSAGE_HANDLER_MODE
        generate_message_handlers // Message handler generation strategy
    )
    : linalg_base<Scalar>(name, shape_inputs, input_names, shape_outputs,
                          output_names, broadcast_type, tag_errors, pdu_errors),
      gr::block(name, gr::io_signature::make(0, 0, 0),
                gr::io_signature::make(0, 0, 0)),
      d_pdu_update(pdu_update),
      d_generate_message_handlers(generate_message_handlers),
      d_input_names(input_names), d_output_names(output_names) {
  // Register input and output message ports
  for (size_t i = 0; i < d_input_names.size(); ++i) {
    this->message_port_register_in(pmt::mp(d_input_names[i]));
  }
  for (size_t i = 0; i < d_output_names.size(); ++i) {
    this->message_port_register_out(pmt::mp(d_output_names[i]));
  }

  // Initialize last seen and seen vectors
  d_seen.resize(shape_inputs.size());
  d_last_values.resize(shape_inputs.size());
  std::fill(d_seen.begin(), d_seen.end(), false);
  // Allocate last seen vectors
  for (size_t i = 0; i < shape_inputs.size(); ++i) {
    const auto shape = shape_inputs[i];
    const auto size = types::compute_size(shape); // Total number of elements
    // FIXME: Unused variable warning - 'name' is declared but never used
    // Problem: Variable 'name' is captured but not used in this scope
    // Solution: Either remove the variable or use it, or add [[maybe_unused]]
    // attribute const auto &name = d_input_names[i];
    d_last_values[i].resize(size);
    // Use a unique negative value per input index as a sentinel for
    // uninitialized last values
    auto uninitialized_last_value = [](size_t idx) {
      return -static_cast<int>(idx) - 1;
    };
    for (size_t i = 0; i < shape_inputs.size(); ++i) {
      const auto shape = shape_inputs[i];
      const auto size = types::compute_size(shape); // Total number of elements
      // FIXME: Unused variable warning - 'name' is declared but never used
      // Problem: Variable 'name' is captured but not used in this scope
      // Solution: Either remove the variable or use it, or add [[maybe_unused]]
      // attribute
      const auto &name = d_input_names[i];
      d_last_values[i].resize(size);
      std::fill(d_last_values[i].begin(), d_last_values[i].end(),
                uninitialized_last_value(i));
    }

    if (d_generate_message_handlers == MESSAGE_HANDLER_MODE::DEFAULT) {
      // Install message handlers for each input port
      for (size_t i = 0; i < d_input_names.size(); ++i) {
        const auto &name = d_input_names[i];
        auto port = pmt::mp(name);
        this->set_msg_handler(
            port, [this, i](pmt::pmt_t msg) { this->_handle_pdu(msg, i); });
      }
    }
  }
}

template <typename Scalar>
void linalg_base_pdu<Scalar>::_handle_pdu(const pmt::pmt_t &msg,
                                          const int port_idx) {
  // Process the PDU data and metadata
  // ...
}

template <typename Scalar>
int linalg_base_pdu<Scalar>::_work() {
  // This is a stub implementation, actual work will depend on derived classes
  // and their specific PDU handling logic.
  // For now, just return 0 to indicate no output produced.

  // TODO: Implement the work method
  throw std::runtime_error(
      "linalg_base_pdu::_work() is not implemented. This is a stub class.");
  return 0;
}

// Explicit instantiation for common scalar types
template class linalg_base_pdu<float>;
template class linalg_base_pdu<double>;
template class linalg_base_pdu<std::complex<float>>;
template class linalg_base_pdu<std::complex<double>>;

} // namespace linalg
} // namespace gr