/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "zeros_impl.h"
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
typename zeros<T>::sptr zeros<T>::make(const types::shape &shape) {
  return typename zeros<T>::sptr(new zeros_impl<T>(shape));
}

/*
 * The private constructor
 */
template <class T>
zeros_impl<T>::zeros_impl(const types::shape &shape)
    : gr::sync_block("zeros", gr::io_signature::make(0, 0, 0),
                     gr::io_signature::make(
                         1 /* min outputs */, 1 /*max outputs */,
                         sizeof(T) * gr::linalg::utils::product(shape))) {
  // Initialize the output shape to be the same as the input shape
  d_shape = shape;

  // Validate the shape
  if (shape.empty()) {
    throw std::invalid_argument("Zeros block: shape cannot be empty");
  }

  d_output_size = gr::linalg::utils::product(shape);
  d_input_size = 0;       // No input required for zeros block
  d_input_shape = {};     // Empty input shape (source block)
  d_output_shape = shape; // Output shape is the same as input shape
}

/*
 * Our virtual destructor.
 */
template <class T>
zeros_impl<T>::~zeros_impl() {}

template <class T>
int zeros_impl<T>::work(int noutput_items,
                        gr_vector_const_void_star &input_items,
                        gr_vector_void_star &output_items) {
  auto out = static_cast<T *>(output_items[0]);

  // Generate matrices filled with zeros
  for (int item = 0; item < noutput_items; item++) {
    T *matrix = &out[item * d_output_size];

    // Fill all elements with zero
    std::fill(matrix, matrix + d_output_size, static_cast<T>(0));
  }

  // Tell runtime system how many output items we produced.
  return noutput_items;
}

// Explicit template instantiations
template class zeros_impl<float>;
template class zeros_impl<double>;
template class zeros_impl<std::complex<float>>;
template class zeros_impl<std::complex<double>>;

// Explicitly instantiate the make functions
template typename zeros<float>::sptr zeros<float>::make(const types::shape &);
template typename zeros<double>::sptr zeros<double>::make(const types::shape &);
template typename zeros<std::complex<float>>::sptr
zeros<std::complex<float>>::make(const types::shape &);
template typename zeros<std::complex<double>>::sptr
zeros<std::complex<double>>::make(const types::shape &);

} /* namespace linalg */
} /* namespace gr */
