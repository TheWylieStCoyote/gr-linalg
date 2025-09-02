/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_ONES_IMPL_H
#define INCLUDED_LINALG_ONES_IMPL_H

#include <gnuradio/linalg/ones.h>

namespace gr {
namespace linalg {

template <class T>
class ones_impl : public ones<T> {
private:
  types::shape d_shape;
  types::shape d_input_shape;
  types::shape d_output_shape;
  size_t d_input_size;
  size_t d_output_size;

public:
  ones_impl(const types::shape &shape);
  ~ones_impl();

  // Where all the action really happens
  int work(int noutput_items, gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);
};

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_ONES_IMPL_H */
