/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "eye_impl.h"
#include <algorithm>
#include <complex>
#include <gnuradio/buffer_type.h>
#include <gnuradio/io_signature.h>
#include <gnuradio/linalg/utils.h>
#include <stdexcept>

namespace gr {
namespace linalg {

template <class T>
typename eye<T>::sptr eye<T>::make(const types::shape &shape) {
  return typename eye<T>::sptr(new eye_impl<T>(shape));
}

/*
 * The private constructor
 */
template <class T>
eye_impl<T>::eye_impl(const types::shape &shape)
    : gr::sync_block("eye", gr::io_signature::make(0, 0, 0),
                     gr::io_signature::make(
                         1 /* min outputs */, 1 /*max outputs */,
                         sizeof(T) * gr::linalg::utils::product(shape))) {
  // Initialize the output shape to be the same as the input shape
  d_shape = shape;

  // Validate the shape
  if (shape.empty()) {
    throw std::invalid_argument("Eye block: shape cannot be empty");
  }

  // Validate that it's a 2D square matrix
  if (shape.size() != 2 || shape[0] != shape[1]) {
    throw std::invalid_argument(
        "Eye block requires square 2D matrix shape [n, n]");
  }

  d_output_size = gr::linalg::utils::product(shape);
  d_input_size = 0;       // No input required for eye block
  d_input_shape = {};     // Empty input shape (source block)
  d_output_shape = shape; // Output shape is the same as input shape
} /*
   * Our virtual destructor.
   */
template <class T>
eye_impl<T>::~eye_impl() {}

template <class T>
int eye_impl<T>::work(int noutput_items, gr_vector_const_void_star &input_items,
                      gr_vector_void_star &output_items) {
  auto out = static_cast<T *>(output_items[0]);

  // Generate identity matrices
  for (int item = 0; item < noutput_items; item++) {
    T *matrix = &out[item * d_output_size];

    // Initialize all elements to zero
    std::fill(matrix, matrix + d_output_size, static_cast<T>(0));

    // Set diagonal elements to one (assuming square matrix)
    int matrix_size = d_shape[0]; // Assume square matrix
    for (int i = 0; i < matrix_size; i++) {
      matrix[i * matrix_size + i] = static_cast<T>(1);
    }
  }

  // Tell runtime system how many output items we produced.
  return noutput_items;
}

// Explicit template instantiations
template class eye_impl<float>;
template class eye_impl<double>;
template class eye_impl<std::complex<float>>;
template class eye_impl<std::complex<double>>;

// Explicitly instantiate the make functions
template typename eye<float>::sptr eye<float>::make(const types::shape &);
template typename eye<double>::sptr eye<double>::make(const types::shape &);
template typename eye<std::complex<float>>::sptr
eye<std::complex<float>>::make(const types::shape &);
template typename eye<std::complex<double>>::sptr
eye<std::complex<double>>::make(const types::shape &);

} /* namespace linalg */
} /* namespace gr */
