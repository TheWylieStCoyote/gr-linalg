#include <algorithm>
#include <complex>
#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/types.h>
#include <gnuradio/linalg/utils.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace gr {
namespace linalg {

// Note: For class templates, definitions are provided here and explicitly
// instantiated for the supported Scalar types below.

// Protected default constructor
template <class Scalar>
linalg_base<Scalar>::linalg_base()
    : d_name(""), shape_inputs{}, input_names{}, input_sizes{}, shape_outputs{},
      output_names{}, output_sizes{},
      broadcast_type(array_broadcast_type::NONE), tag_errors(error_tag_t::NONE),
      pdu_errors(error_pdu_p::NONE) {}

// Convenience constructor delegating to full constructor
template <class Scalar>
linalg_base<Scalar>::linalg_base(const std::string &name,
                                 const types::vector_shapes &shape_inputs_,
                                 const types::vector_shapes &shape_outputs_,
                                 const array_broadcast_type &broadcast_type_)
    : linalg_base(name, shape_inputs_, std::vector<std::string>{},
                  shape_outputs_, std::vector<std::string>{}, broadcast_type_,
                  error_tag_t::NONE, error_pdu_p::NONE) {}

// Full constructor with validation
template <class Scalar>
linalg_base<Scalar>::linalg_base(
    const std::string &name,                       // NOLINT
    const types::vector_shapes &shape_inputs_,     // NOLINT
    const std::vector<std::string> &input_names_,  // NOLINT
    const types::vector_shapes &shape_outputs_,    // NOLINT
    const std::vector<std::string> &output_names_, // NOLINT
    const array_broadcast_type &broadcast_type_, const error_tag_t &tag_errors_,
    const error_pdu_p &pdu_errors_)
    : d_name(name), shape_inputs(shape_inputs_), input_names(input_names_),
      input_sizes(types::compute_sizes<Scalar>(shape_inputs_)),
      shape_outputs(shape_outputs_), output_names(output_names_),
      output_sizes(types::compute_sizes<Scalar>(shape_outputs_)),
      broadcast_type(broadcast_type_), tag_errors(tag_errors_),
      pdu_errors(pdu_errors_) {
  // Supply default port names if not provided
  if (this->input_names.empty()) {
    this->input_names.reserve(this->shape_inputs.size());
    for (size_t i = 0; i < this->shape_inputs.size(); ++i) {
      this->input_names.push_back("in_" + std::to_string(i));
    }
  }
  if (this->output_names.empty()) {
    this->output_names.reserve(this->shape_outputs.size());
    for (size_t i = 0; i < this->shape_outputs.size(); ++i) {
      this->output_names.push_back("out_" + std::to_string(i));
    }
  }

  // Basic shape/name consistency checks
  if (this->input_names.size() != this->shape_inputs.size()) {
    throw std::invalid_argument(
        "Input names size does not match input shapes size");
  }
  if (this->output_names.size() != this->shape_outputs.size()) {
    throw std::invalid_argument(
        "Output names size does not match output shapes size");
  }

  // Validate shapes vs expected
  validate_shapes(this->shape_inputs, this->shape_outputs, this->d_name,
                  this->broadcast_type);
}

// Static: compute_output_shapes (with basic broadcast policies)
template <class Scalar>
types::vector_shapes linalg_base<Scalar>::compute_output_shapes(
    const types::vector_shapes &input_shapes,
    const array_broadcast_type &broadcast_type) {
  const auto n_inputs = input_shapes.size();
  for (const auto &shape : input_shapes) {
    if (shape.empty()) {
      throw std::invalid_argument("Input shape cannot be empty");
    }
    for (auto dim : shape) {
      if (dim == 0) {
        // This still supports Eigen::Dynamic shapes (typically -1)
        throw std::invalid_argument("Input shape dimensions must be positive");
      }
    }
  }
  if (broadcast_type == array_broadcast_type::VECTOR) {
    if (n_inputs < 2) {
      throw std::invalid_argument(
          "At least 2 input shapes required for VECTOR broadcast");
    }
    // Compute a single broadcasted vector shape and wrap it into vector_shapes
    const auto out_shape = utils::compute_broadcast(input_shapes, 1);
    return utils::shapes_from_shapes(out_shape);
  }
  if (broadcast_type == array_broadcast_type::MATRIX) {
    if (n_inputs < 2) {
      throw std::invalid_argument(
          "At least 2 input shapes required for MATRIX broadcast");
    }
    // Compute a single broadcasted matrix shape and wrap it into vector_shapes
    const auto out_shape = utils::compute_broadcast(input_shapes, 2);
    return utils::shapes_from_shapes(out_shape);
  } else if (broadcast_type == array_broadcast_type::NONE) {
    // For NONE, return the input shapes as output
    return input_shapes;
  } else if (broadcast_type == array_broadcast_type::CUSTOM) {
    // CUSTOM broadcast requires specific handling in derived classes
    return {};
  } else {
    return {};
  }
}

// Static: validate_shapes with optional expected outputs and broadcast policy
template <class Scalar>
void linalg_base<Scalar>::validate_shapes(
    const types::vector_shapes &input_shapes,  // vector of input shapes
    const types::vector_shapes &output_shapes, // vector of output shapes
    const std::string &name,                   // name of the block for errors
    const array_broadcast_type &broadcast_type) {
  // Check non-negative dims
  auto check_dims = [&](const types::vector_shapes &shapes, const char *label) {
    for (size_t i = 0; i < shapes.size(); ++i) {
      for (auto d : shapes[i]) {
        if (d < 0) {
          throw std::invalid_argument(
              std::string(label) + "[" + std::to_string(i) +
              "] has negative dim (" + std::to_string(d) + ") in " + name);
        }
      }
    }
  };
  check_dims(input_shapes, "input");
  check_dims(output_shapes, "output");

  // Compare against expected broadcasted outputs
  const auto expected = compute_output_shapes(input_shapes, broadcast_type);
  if (!output_shapes.empty()) {
    if (broadcast_type == array_broadcast_type::CUSTOM) {
      return; // Skip strict validation for custom operations
    }
    if (expected.size() != output_shapes.size()) {
      throw std::invalid_argument(
          "Output shapes count does not match expected count for " + name);
    }
    for (size_t i = 0; i < expected.size(); ++i) {
      if (expected[i] != output_shapes[i]) {
        throw std::invalid_argument("Output shape[" + std::to_string(i) +
                                    "] does not match expected shape for " +
                                    name);
      }
    }
  }
}

// Explicit template instantiations for common types
template class linalg_base<float>;
template class linalg_base<double>;
template class linalg_base<std::complex<float>>;
template class linalg_base<std::complex<double>>;

} // namespace linalg
} // namespace gr
