/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <boost/test/unit_test.hpp>
#include <complex>
#include <gnuradio/attributes.h>
#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/types.h>
#include <stdexcept>

namespace gr {
namespace linalg {

// Test fixture class that inherits from linalg_base to test the abstract base
// class
template <class Scalar>
class test_linalg_base : public linalg_base<Scalar> {
public:
  test_linalg_base() : linalg_base<Scalar>() {}

  test_linalg_base(
      const std::string &name, const types::vector_shapes &shape_inputs,
      const types::vector_shapes &shape_outputs,
      const array_broadcast_type &broadcast_type = array_broadcast_type::NONE)
      : linalg_base<Scalar>(name, shape_inputs, shape_outputs, broadcast_type) {
  }

  test_linalg_base(
      const std::string &name, const types::vector_shapes &shape_inputs,
      const std::vector<std::string> &input_names,
      const types::vector_shapes &shape_outputs,
      const std::vector<std::string> &output_names,
      const array_broadcast_type &broadcast_type = array_broadcast_type::NONE,
      const error_tag_t &tag_errors = error_tag_t::NONE,
      const error_pdu_p &pdu_errors = error_pdu_p::NONE)
      : linalg_base<Scalar>(name, shape_inputs, input_names, shape_outputs,
                            output_names, broadcast_type, tag_errors,
                            pdu_errors) {}

  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_matrices,
            types::vector_matrix_map<Scalar> &output_matrices) override {
    return OperationReturn::SUCCESS;
  }
};

BOOST_AUTO_TEST_CASE(test_linalg_base_default_constructor) {
  test_linalg_base<float> base;
  BOOST_CHECK_NO_THROW(base);
}

BOOST_AUTO_TEST_CASE(test_linalg_base_convenience_constructor) {
  types::vector_shapes inputs = {{2, 2}, {2, 2}};
  types::vector_shapes outputs = {
      {2, 2}, {2, 2}}; // NONE broadcast returns input shapes

  test_linalg_base<float> base("test_block", inputs, outputs,
                               array_broadcast_type::NONE);
  BOOST_CHECK_NO_THROW(base);
}

BOOST_AUTO_TEST_CASE(test_linalg_base_full_constructor_valid) {
  types::vector_shapes inputs = {{3, 3}, {3, 3}};
  types::vector_shapes outputs = {
      {3, 3}, {3, 3}}; // NONE broadcast returns input shapes
  std::vector<std::string> input_names = {"matrix_a", "matrix_b"};
  std::vector<std::string> output_names = {"result_a", "result_b"};

  BOOST_CHECK_NO_THROW({
    test_linalg_base<double> base("matrix_op", inputs, input_names, outputs,
                                  output_names, array_broadcast_type::NONE,
                                  error_tag_t::NONE, error_pdu_p::NONE);
  });
}

BOOST_AUTO_TEST_CASE(test_linalg_base_default_port_names) {
  types::vector_shapes inputs = {{2, 2}, {2, 2}};
  types::vector_shapes outputs = {{2, 2}, {2, 2}};

  test_linalg_base<float> base("test_block", inputs, {}, outputs, {},
                               array_broadcast_type::NONE);
  BOOST_CHECK_NO_THROW(base);
}

BOOST_AUTO_TEST_CASE(test_linalg_base_input_names_size_mismatch) {
  types::vector_shapes inputs = {{2, 2}, {2, 2}};
  types::vector_shapes outputs = {{2, 2}};
  std::vector<std::string> input_names = {"only_one_name"};
  std::vector<std::string> output_names = {"result"};

  BOOST_CHECK_THROW(
      {
        test_linalg_base<float> base("test_block", inputs, input_names, outputs,
                                     output_names);
      },
      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_linalg_base_output_names_size_mismatch) {
  types::vector_shapes inputs = {{2, 2}};
  types::vector_shapes outputs = {{2, 2}, {2, 2}};
  std::vector<std::string> input_names = {"input"};
  std::vector<std::string> output_names = {"only_one_name"};

  BOOST_CHECK_THROW(
      {
        test_linalg_base<float> base("test_block", inputs, input_names, outputs,
                                     output_names);
      },
      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_compute_output_shapes_none_broadcast) {
  types::vector_shapes inputs = {{2, 2}, {3, 3}};

  auto outputs = test_linalg_base<float>::compute_output_shapes(
      inputs, array_broadcast_type::NONE);
  BOOST_CHECK_EQUAL(outputs.size(), 2);
  BOOST_CHECK(outputs[0] == types::shape({2, 2}));
  BOOST_CHECK(outputs[1] == types::shape({3, 3}));
}

BOOST_AUTO_TEST_CASE(test_compute_output_shapes_vector_broadcast) {
  types::vector_shapes inputs = {{3}, {3}}; // Compatible shapes for broadcast

  auto outputs = test_linalg_base<float>::compute_output_shapes(
      inputs, array_broadcast_type::VECTOR);
  BOOST_CHECK_EQUAL(outputs.size(), 1);
}

BOOST_AUTO_TEST_CASE(test_compute_output_shapes_matrix_broadcast) {
  types::vector_shapes inputs = {{2, 3},
                                 {2, 3}}; // Compatible shapes for broadcast

  auto outputs = test_linalg_base<float>::compute_output_shapes(
      inputs, array_broadcast_type::MATRIX);
  BOOST_CHECK_EQUAL(outputs.size(), 1);
}

BOOST_AUTO_TEST_CASE(test_compute_output_shapes_custom_broadcast) {
  types::vector_shapes inputs = {{2, 2}};

  auto outputs = test_linalg_base<float>::compute_output_shapes(
      inputs, array_broadcast_type::CUSTOM);
  BOOST_CHECK(outputs.empty());
}

BOOST_AUTO_TEST_CASE(test_compute_output_shapes_empty_input) {
  types::vector_shapes inputs = {{}};

  BOOST_CHECK_THROW(
      {
        test_linalg_base<float>::compute_output_shapes(
            inputs, array_broadcast_type::NONE);
      },
      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_compute_output_shapes_zero_dimension) {
  types::vector_shapes inputs = {{2, 0, 3}};

  BOOST_CHECK_THROW(
      {
        test_linalg_base<float>::compute_output_shapes(
            inputs, array_broadcast_type::NONE);
      },
      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_compute_output_shapes_insufficient_inputs_vector) {
  types::vector_shapes inputs = {{2}};

  BOOST_CHECK_THROW(
      {
        test_linalg_base<float>::compute_output_shapes(
            inputs, array_broadcast_type::VECTOR);
      },
      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_compute_output_shapes_insufficient_inputs_matrix) {
  types::vector_shapes inputs = {{2, 2}};

  BOOST_CHECK_THROW(
      {
        test_linalg_base<float>::compute_output_shapes(
            inputs, array_broadcast_type::MATRIX);
      },
      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_validate_shapes_valid) {
  types::vector_shapes inputs = {{2, 2}, {2, 2}};
  types::vector_shapes outputs = {{2, 2}, {2, 2}};

  BOOST_CHECK_NO_THROW({
    test_linalg_base<float>::validate_shapes(inputs, outputs, "test_block",
                                             array_broadcast_type::NONE);
  });
}

BOOST_AUTO_TEST_CASE(test_validate_shapes_negative_input_dimension) {
  types::vector_shapes inputs = {{2, -1}};
  types::vector_shapes outputs = {{2, 2}};

  BOOST_CHECK_THROW(
      {
        test_linalg_base<float>::validate_shapes(inputs, outputs, "test_block");
      },
      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_validate_shapes_negative_output_dimension) {
  types::vector_shapes inputs = {{2, 2}};
  types::vector_shapes outputs = {{2, -1}};

  BOOST_CHECK_THROW(
      {
        test_linalg_base<float>::validate_shapes(inputs, outputs, "test_block");
      },
      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(test_validate_shapes_custom_broadcast_skip) {
  types::vector_shapes inputs = {{2, 2}};
  types::vector_shapes outputs = {{3, 3}};

  BOOST_CHECK_NO_THROW({
    test_linalg_base<float>::validate_shapes(inputs, outputs, "test_block",
                                             array_broadcast_type::CUSTOM);
  });
}

BOOST_AUTO_TEST_CASE(test_template_instantiation_float) {
  BOOST_CHECK_NO_THROW({ test_linalg_base<float> base; });
}

BOOST_AUTO_TEST_CASE(test_template_instantiation_double) {
  BOOST_CHECK_NO_THROW({ test_linalg_base<double> base; });
}

BOOST_AUTO_TEST_CASE(test_template_instantiation_complex_float) {
  BOOST_CHECK_NO_THROW({ test_linalg_base<std::complex<float>> base; });
}

BOOST_AUTO_TEST_CASE(test_template_instantiation_complex_double) {
  BOOST_CHECK_NO_THROW({ test_linalg_base<std::complex<double>> base; });
}

BOOST_AUTO_TEST_CASE(test_operation_return_enum_output) {
  std::ostringstream oss;
  oss << OperationReturn::SUCCESS;
  BOOST_CHECK_EQUAL(oss.str(), "SUCCESS");

  oss.str("");
  oss << OperationReturn::FAILURE;
  BOOST_CHECK_EQUAL(oss.str(), "FAILURE");

  oss.str("");
  oss << OperationReturn::NOT_IMPLEMENTED;
  BOOST_CHECK_EQUAL(oss.str(), "NOT_IMPLEMENTED");

  oss.str("");
  oss << OperationReturn::INVALID_SHAPE;
  BOOST_CHECK_EQUAL(oss.str(), "INVALID_SHAPE");

  oss.str("");
  oss << OperationReturn::OUT_OF_RANGE;
  BOOST_CHECK_EQUAL(oss.str(), "OUT_OF_RANGE");
}

} /* namespace linalg */
} /* namespace gr */
