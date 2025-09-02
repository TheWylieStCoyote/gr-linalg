/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <boost/test/tools/old/interface.hpp>
#include <boost/test/unit_test.hpp>
#include <gnuradio/linalg/utils.h>
#include <limits>

namespace gr {
namespace linalg {

BOOST_AUTO_TEST_CASE(test_utils_pmt_to_vector) {
  // Test PMT to vector_shapes conversion
  // pmt_to_std_vector
  std::vector<float> float_limits = {
      0.0f,                                   // Zero
      -1.0f,                                  // Negative value
      1.0f,                                   // Positive value
      std::numeric_limits<float>::min(),      // Minimum positive value
      std::numeric_limits<float>::max(),      // Maximum value
      std::numeric_limits<float>::epsilon(),  // Epsilon
      std::numeric_limits<float>::infinity(), // Infinity
      std::numeric_limits<float>::quiet_NaN() // NaN
  };
  // pmt::pmt_t pmt_vector = pmt::make_vector(float_limits.size(), 0.0f);
  // auto result_vector = utils::pmt_to_vector(pmt_vector);
  // BOOST_CHECK_EQUAL(result_vector.size(), 5);
  // BOOST_CHECK_EQUAL(result_vector[0], float_limits.min());
  // BOOST_CHECK_EQUAL(result_vector[1], float_limits.max());
  // BOOST_CHECK_EQUAL(result_vector[2], float_limits.epsilon());
  // BOOST_CHECK_EQUAL(result_vector[3], float_limits.infinity());
  // BOOST_CHECK(std::isnan(result_vector[4])); // Check for NaN
}

BOOST_AUTO_TEST_CASE(test_utils_pmt_vector_to_pmt) {
  // Test vector_shapes to PMT conversion
  // std_vector_to_pmt
}

} // namespace linalg
} // namespace gr
