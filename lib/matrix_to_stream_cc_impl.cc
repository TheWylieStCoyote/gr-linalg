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
#include "matrix_to_stream_cc_impl.h"

namespace gr {
  namespace linalg {

    matrix_to_stream_cc::sptr
    matrix_to_stream_cc::make(shape)
    {
      return gnuradio::get_initial_sptr
        (new matrix_to_stream_cc_impl(shape));
    }


    /*
     * The private constructor
     */
    matrix_to_stream_cc_impl::matrix_to_stream_cc_impl(shape)
      : gr::sync_interpolator("matrix_to_stream_cc",
              gr::io_signature::make(<+MIN_IN+>, <+MAX_IN+>, sizeof(<+ITYPE+>)),
              gr::io_signature::make(<+MIN_OUT+>, <+MAX_OUT+>, sizeof(<+OTYPE+>)), <+interpolation+>)
    {}

    /*
     * Our virtual destructor.
     */
    matrix_to_stream_cc_impl::~matrix_to_stream_cc_impl()
    {
    }

    int
    matrix_to_stream_cc_impl::work(int noutput_items,
        gr_vector_const_void_star &input_items,
        gr_vector_void_star &output_items)
    {
      const <+ITYPE+> *in = (const <+ITYPE+> *) input_items[0];
      <+OTYPE+> *out = (<+OTYPE+> *) output_items[0];

      // Do <+signal processing+>

      // Tell runtime system how many output items we produced.
      return noutput_items;
    }

  } /* namespace linalg */
} /* namespace gr */

