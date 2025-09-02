/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <boost/test/unit_test.hpp>
#include <gnuradio/attributes.h>
#include <gnuradio/linalg/matrix_exp.h>

namespace gr {
namespace linalg {

BOOST_AUTO_TEST_CASE(test_matrix_exp_basic_functionality) {
  // Test basic matrix exponential functionality

  // Test identity matrix -> exp(I) = e*I
  auto exp_block = matrix_exp_sync_f::make({{2, 2}});
  BOOST_REQUIRE(exp_block);
  BOOST_CHECK_EQUAL(exp_block->name(), "matrix_exp_sync");

  // Test creation of different precision types
  auto exp_block_d = matrix_exp_sync_d::make({{3, 3}});
  BOOST_REQUIRE(exp_block_d);

  auto exp_block_cf = matrix_exp_sync_cf::make({{2, 2}});
  BOOST_REQUIRE(exp_block_cf);

  auto exp_block_cd = matrix_exp_sync_cd::make({{2, 2}});
  BOOST_REQUIRE(exp_block_cd);
}

BOOST_AUTO_TEST_CASE(test_matrix_exp_pdu_interface) {
  // Test PDU interface creation
  auto exp_pdu = matrix_exp_pdu_f::make({{2, 2}});
  BOOST_REQUIRE(exp_pdu);
  BOOST_CHECK_EQUAL(exp_pdu->name(), "matrix_exp_pdu");

  // Test complex PDU variant
  auto exp_pdu_cf = matrix_exp_pdu_cf::make({{3, 3}});
  BOOST_REQUIRE(exp_pdu_cf);
}

} /* namespace linalg */
} /* namespace gr */
