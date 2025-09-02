/* -*- c++ -*- */
/*
 * Copyright 2025
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <boost/test/unit_test.hpp>
#include <cstdlib>
#include <gnuradio/attributes.h>
#include <gnuradio/linalg/matrix_determinant.h>
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

BOOST_AUTO_TEST_CASE(test_matrix_determinant_pdu_block_functionality) {
  auto det_pdu = matrix_determinant_pdu<float>::make({2, 2});
  BOOST_REQUIRE(det_pdu);

  // Basic sanity check that the API exists and is callable
  (void)det_pdu->message_ports_in();
  (void)det_pdu->message_ports_out();
}

BOOST_AUTO_TEST_CASE(test_matrix_determinant_pdu_invalid_shape_throws) {
  BOOST_CHECK_THROW((void)matrix_determinant_pdu<float>::make({2, 3}),
                    std::exception);
}

} // namespace linalg
} // namespace gr
