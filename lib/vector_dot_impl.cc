/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "vector_dot_impl.h"
#include <gnuradio/io_signature.h>
#include <gnuradio/linalg/types.h>

namespace gr {
namespace linalg {

#pragma message("set the following appropriately and remove this warning")
using input_type = float;
#pragma message("set the following appropriately and remove this warning")
using output_type = float;
vector_dot::sptr vector_dot::make(const types::shape &shape_a,
                                  const types::shape &shape_b) {
  return gnuradio::make_block_sptr<vector_dot_impl>(shape_a, shape_b);
}

/*
 * The private constructor
 */
vector_dot_impl::vector_dot_impl(const types::shape &shape_a,
                                 const types::shape &shape_b)
    : gr::sync_block(
          "vector_dot",
          gr::io_signature::make(2 /* min inputs */, 2 /* max inputs */,
                                 sizeof(input_type)),
          gr::io_signature::make(1 /* min outputs */, 1 /*max outputs */,
                                 sizeof(output_type))) {
  (void)shape_a;
  (void)shape_b;
}

/*
 * Our virtual destructor.
 */
vector_dot_impl::~vector_dot_impl() {}

int vector_dot_impl::work(int noutput_items,
                          gr_vector_const_void_star &input_items,
                          gr_vector_void_star &output_items) {
  auto in = static_cast<const input_type *>(input_items[0]);
  auto out = static_cast<output_type *>(output_items[0]);

#pragma message(                                                               \
    "Implement the signal processing in your block and remove this warning")
  // Do <+signal processing+>

  // Tell runtime system how many output items we produced.
  return noutput_items;
}

} /* namespace linalg */
} /* namespace gr */
