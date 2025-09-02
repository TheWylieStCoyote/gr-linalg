/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <boost/test/unit_test.hpp>
#include <gnuradio/attributes.h>
#include <gnuradio/linalg/matrix_subtract.h>
#include <gnuradio/linalg/types.h>

namespace gr {
namespace linalg {

using matrix_subtract_f = gr::linalg::matrix_subtract_sync<float>;

BOOST_AUTO_TEST_CASE(test_matrix_subtract_sync_block) {
  auto shape = types::shape({2, 2});
  auto block = matrix_subtract_f::make(shape, 2);
  BOOST_REQUIRE(block);
  auto in_sig = block->input_signature();
  auto out_sig = block->output_signature();
  BOOST_CHECK_EQUAL(in_sig->max_streams(), 2);
  BOOST_CHECK_EQUAL(out_sig->max_streams(), 1);
}

BOOST_AUTO_TEST_CASE(test_matrix_subtract_work) {
  auto shape = types::shape({2, 2});
  auto block = matrix_subtract_f::make(shape, 2);
  BOOST_REQUIRE(block);

  std::vector<float> A = {1, 2, 3, 4};
  std::vector<float> B = {5, 6, 7, 8};
  std::vector<float> C(4, 0.0f);

  std::vector<const void *> in(2);
  std::vector<void *> out(1);
  in[0] = A.data();
  in[1] = B.data();
  out[0] = C.data();

  int r = block->work(1, in, out);
  BOOST_CHECK_EQUAL(r, 1);

  std::vector<float> expected = {1 - 5, 2 - 6, 3 - 7, 4 - 8};
  for (size_t i = 0; i < expected.size(); ++i) {
    BOOST_CHECK_CLOSE(C[i], expected[i], 1e-5f);
  }
}

} /* namespace linalg */
} /* namespace gr */
