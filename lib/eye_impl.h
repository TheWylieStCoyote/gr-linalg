/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_EYE_IMPL_H
#define INCLUDED_LINALG_EYE_IMPL_H

#include <gnuradio/linalg/eye.h>
#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/types.h>
#include <gnuradio/linalg/utils.h>

namespace gr {
namespace linalg {

template <class T>
class eye_impl : public eye<T> {
private:
  types::shape d_shape;
  types::shape d_input_shape;
  types::shape d_output_shape;
  size_t d_input_size;
  size_t d_output_size;

public:
  eye_impl(const types::shape &shape);
  ~eye_impl();

  // Where all the action really happens
  int work(int noutput_items, gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);
};

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_EYE_IMPL_H */
