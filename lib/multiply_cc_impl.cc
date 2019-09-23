/* -*- c++ -*- */
/*
 * Copyright 2019 Wylie Standage-Beier.
 *
 * This is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this software; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gnuradio/io_signature.h>
#include "multiply_cc_impl.h"

namespace gr {
  namespace linalg {

    multiply_cc::sptr
    multiply_cc::make(std::vector<int> shape_A, std::vector<int> shape_B)
    {
      return gnuradio::get_initial_sptr
        (new multiply_cc_impl(shape_A, shape_B));
    }


    /*
     * The private constructor
     */
    multiply_cc_impl::multiply_cc_impl(std::vector<int> shape_A, std::vector<int> shape_B)
      : gr::sync_block("multiply_cc",
              gr::io_signature::make(1, 2, sizeof(gr_complex)),
              gr::io_signature::make(1, 2, sizeof(gr_complex)))
    {}

    /*
     * Our virtual destructor.
     */
    multiply_cc_impl::~multiply_cc_impl()
    {
    }

    int
    multiply_cc_impl::work(int noutput_items,
        gr_vector_const_void_star &input_items,
        gr_vector_void_star &output_items)
    {
      const gr_complex *in = (const gr_complex *) input_items[0];
      gr_complex *out = (gr_complex *) output_items[0];

      // Do <+signal processing+>

      // Tell runtime system how many output items we produced.
      return noutput_items;
    }

  } /* namespace linalg */
} /* namespace gr */

