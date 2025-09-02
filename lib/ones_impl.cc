/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "ones_impl.h"
#include <algorithm>
#include <complex>
#include <gnuradio/buffer_type.h>
#include <gnuradio/io_signature.h>
#include <gnuradio/linalg/utils.h>
#include <stdexcept>

namespace gr {
namespace linalg {

// #pragma message("set the following appropriately and remove this warning")
// using output_type = float;
template <class T>
typename ones<T>::sptr ones<T>::make(const types::shape &shape) {
  return typename ones<T>::sptr(new ones_impl<T>(shape));
}

/*
 * The private constructor
 */
template <class T>
ones_impl<T>::ones_impl(const types::shape &shape)
    : gr::sync_block("ones", gr::io_signature::make(0, 0, 0),
                     gr::io_signature::make(
                         1 /* min outputs */, 1 /*max outputs */,
                         sizeof(T) * gr::linalg::utils::product(shape))) {
  // Initialize the output shape to be the same as the input shape
  d_shape = shape;

  // Validate the shape
  if (shape.empty()) {
    throw std::invalid_argument("Ones block: shape cannot be empty");
  }

  d_output_size = gr::linalg::utils::product(shape);
  d_input_size = 0;       // No input required for ones block
  d_input_shape = {};     // Empty input shape (source block)
  d_output_shape = shape; // Output shape is the same as input shape
}

/*
 * Our virtual destructor.
 */
template <class T>
ones_impl<T>::~ones_impl() {}

template <class T>
int ones_impl<T>::work(int noutput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items) {
  auto out = static_cast<T *>(output_items[0]);

  // Generate matrices filled with ones
  for (int item = 0; item < noutput_items; item++) {
    T *matrix = &out[item * d_output_size];

    // Fill all elements with one
    std::fill(matrix, matrix + d_output_size, static_cast<T>(1));
  }

  // Tell runtime system how many output items we produced.
  return noutput_items;
}

// Explicit template instantiations
template class ones_impl<float>;
template class ones_impl<double>;
template class ones_impl<std::complex<float>>;
template class ones_impl<std::complex<double>>;

// Explicitly instantiate the make functions
template typename ones<float>::sptr ones<float>::make(const types::shape &);
template typename ones<double>::sptr ones<double>::make(const types::shape &);
template typename ones<std::complex<float>>::sptr
ones<std::complex<float>>::make(const types::shape &);
template typename ones<std::complex<double>>::sptr
ones<std::complex<double>>::make(const types::shape &);

} /* namespace linalg */
} /* namespace gr */
