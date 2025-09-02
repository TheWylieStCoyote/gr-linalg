/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_ZEROS_H
#define INCLUDED_LINALG_ZEROS_H

#include <gnuradio/linalg/api.h>
#include <gnuradio/linalg/types.h>
#include <gnuradio/sync_block.h>

namespace gr {
namespace linalg {

/*!
 * \brief <+description of block+>
 * \ingroup linalg
 *
 */
template <class T>
class LINALG_API zeros : virtual public gr::sync_block {
public:
  typedef std::shared_ptr<zeros<T>> sptr;

  /*!
   * \brief Return a shared_ptr to a new instance of linalg::zeros.
   *
   * To avoid accidental use of raw pointers, linalg::zeros's
   * constructor is in a private implementation
   * class. linalg::zeros::make is the public interface for
   * creating new instances.
   */
  static sptr make(const types::shape &shape);
};

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_ZEROS_H */
