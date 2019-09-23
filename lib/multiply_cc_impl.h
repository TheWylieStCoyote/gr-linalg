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

#ifndef INCLUDED_LINALG_MULTIPLY_CC_IMPL_H
#define INCLUDED_LINALG_MULTIPLY_CC_IMPL_H

#include <linalg/multiply_cc.h>

namespace gr {
  namespace linalg {

    class multiply_cc_impl : public multiply_cc
    {
     private:
      // Nothing to declare in this block.

     public:
      multiply_cc_impl(std::vector<int> shape_A, std::vector<int> shape_B);
      ~multiply_cc_impl();

      // Where all the action really happens
      int work(
              int noutput_items,
              gr_vector_const_void_star &input_items,
              gr_vector_void_star &output_items
      );
    };

  } // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MULTIPLY_CC_IMPL_H */

