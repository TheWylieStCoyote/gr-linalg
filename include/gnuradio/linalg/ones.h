/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_ONES_H
#define INCLUDED_LINALG_ONES_H

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
class LINALG_API ones : virtual public gr::sync_block {
public:
  typedef std::shared_ptr<ones<T>> sptr;

  /*!
   * \brief Return a shared_ptr to a new instance of linalg::ones.
   *
   * To avoid accidental use of raw pointers, linalg::ones's
   * constructor is in a private implementation
   * class. linalg::ones::make is the public interface for
   * creating new instances.
   */
  static sptr make(const types::shape &shape);
};

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_ONES_H */
