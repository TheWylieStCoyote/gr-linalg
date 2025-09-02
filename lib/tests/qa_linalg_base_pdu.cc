/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <Eigen/Dense>
#include <boost/test/unit_test.hpp>
#include <complex>
#include <cstdlib>
#include <gnuradio/attributes.h>
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/types.h>
#include <pmt/pmt.h>
#include <string>
#include <vector>

namespace {
struct GRTestEnv {
  GRTestEnv() {
    setenv("GR_DONT_LOAD_PREFS", "0", 1);
    setenv("GR_CONF_CONTROLPORT_ON", "0", 1);
    setenv("GR_RPCMANAGER_ENABLED", "0", 1);
  }
  ~GRTestEnv() {}
};
} // namespace

BOOST_TEST_GLOBAL_FIXTURE(GRTestEnv);

namespace gr {
namespace linalg {

class PduTestBase : public linalg_base_pdu<float> {
public:
  PduTestBase(const std::string &name = "linalg_base_pdu_test")
      : linalg_base_pdu<float>(
            name,
            /*shape_inputs*/ {{2, 2}, {2, 2}},
            /*input_names*/ std::vector<std::string>{},
            /*shape_outputs*/ {{2, 2}},
            /*output_names*/ std::vector<std::string>{},
            /*broadcast_type*/ array_broadcast_type::CUSTOM,
            /*tag_errors*/ error_tag_t::NONE,
            /*pdu_errors*/ error_pdu_p::NONE,
            /*pdu_update*/ PDU_UPDATE::DEFAULT,
            /*message handlers*/ MESSAGE_HANDLER_MODE::DEFAULT) {}

  OperationReturn
  operation(types::vector_const_matrix_map<float> & /*ins*/,
            types::vector_matrix_map<float> & /*outs*/) override {
    return OperationReturn::SUCCESS;
  }

  static types::vector_shapes
  compute_output_shapes_helper(const types::vector_shapes &input_shapes) {
    if (input_shapes.empty()) {
      return {{2, 2}}; // Default output shape
    }
    auto rows = input_shapes[0].size() > 0 ? input_shapes[0][0] : 2;
    auto cols = input_shapes[0].size() > 1 ? input_shapes[0][1] : 2;
    return types::vector_shapes{{rows, cols}};
  }

  void test_handle_input_0(const pmt::pmt_t &pdu) {
    if (!pmt::is_pair(pdu))
      return;
    auto data = pmt::cdr(pdu);
    if (pmt::is_f32vector(data)) {
      size_t len;
      (void)len;
      const float *vec_data = pmt::f32vector_elements(data, len);
      (void)vec_data;
    }
  }

  void test_handle_input_1(const pmt::pmt_t &pdu) {
    if (!pmt::is_pair(pdu))
      return;
    auto data = pmt::cdr(pdu);
    if (pmt::is_f32vector(data)) {
      size_t len;
      (void)len;
      const float *vec_data = pmt::f32vector_elements(data, len);
      (void)vec_data;
    }
  }
};

} /* namespace linalg */
} /* namespace gr */

BOOST_AUTO_TEST_CASE(test_linalg_base_pdu_constructor) {
  gr::linalg::PduTestBase base;
  BOOST_CHECK_NO_THROW(gr::linalg::PduTestBase("test_name"));
}

BOOST_AUTO_TEST_CASE(test_linalg_base_pdu_operation) {
  gr::linalg::PduTestBase base("test_pdu");
  gr::linalg::types::vector_const_matrix_map<float> input_matrices;
  gr::linalg::types::vector_matrix_map<float> output_matrices;
  auto result = base.operation(input_matrices, output_matrices);
  BOOST_CHECK(result == gr::linalg::OperationReturn::SUCCESS);
}

BOOST_AUTO_TEST_CASE(test_linalg_base_pdu_shapes) {
  gr::linalg::types::vector_shapes input_shapes = {{3, 4}};
  auto output_shapes =
      gr::linalg::PduTestBase::compute_output_shapes_helper(input_shapes);
  BOOST_CHECK_EQUAL(output_shapes.size(), 1);
  BOOST_CHECK_EQUAL(output_shapes[0][0], 3);
  BOOST_CHECK_EQUAL(output_shapes[0][1], 4);

  gr::linalg::types::vector_shapes empty_shapes;
  auto default_shapes =
      gr::linalg::PduTestBase::compute_output_shapes_helper(empty_shapes);
  BOOST_CHECK_EQUAL(default_shapes[0][0], 2);
  BOOST_CHECK_EQUAL(default_shapes[0][1], 2);
}

BOOST_AUTO_TEST_CASE(test_linalg_base_pdu_message_handling) {
  gr::linalg::PduTestBase base("test_pdu");
  pmt::pmt_t metadata = pmt::make_dict();
  metadata =
      pmt::dict_add(metadata, pmt::mp("test_key"), pmt::mp("test_value"));
  std::vector<float> test_data = {1.0f, 2.0f, 3.0f, 4.0f};
  pmt::pmt_t data_vector =
      pmt::init_f32vector(test_data.size(), test_data.data());
  pmt::pmt_t pdu = pmt::cons(metadata, data_vector);
  BOOST_CHECK_NO_THROW(base.test_handle_input_0(pdu));
  BOOST_CHECK_NO_THROW(base.test_handle_input_1(pdu));
  pmt::pmt_t invalid_pdu = pmt::mp("not_a_pair");
  BOOST_CHECK_NO_THROW(base.test_handle_input_0(invalid_pdu));
  BOOST_CHECK_NO_THROW(base.test_handle_input_1(invalid_pdu));
}