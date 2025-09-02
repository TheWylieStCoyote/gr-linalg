/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_VECTOR_DOT_IMPL_H
#define INCLUDED_LINALG_VECTOR_DOT_IMPL_H

#include <gnuradio/linalg/types.h>
#include <gnuradio/linalg/vector_dot.h>
#include <memory>

namespace gr {
namespace linalg {

class vector_dot_impl : public vector_dot {
private:
  // Nothing to declare in this block.

public:
  vector_dot_impl(const types::shape &shape_a, const types::shape &shape_b);
  ~vector_dot_impl();

  // Where all the action really happens
  int work(int noutput_items, gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);
};

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_VECTOR_DOT_IMPL_H */
