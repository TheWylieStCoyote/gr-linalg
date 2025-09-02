/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_EYE_H
#define INCLUDED_LINALG_EYE_H

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
class LINALG_API eye : virtual public gr::sync_block {
public:
  typedef std::shared_ptr<eye<T>> sptr;

  /*!
   * \brief Return a shared_ptr to a new instance of linalg::eye.
   *
   * To avoid accidental use of raw pointers, linalg::eye's
   * constructor is in a private implementation
   * class. linalg::eye::make is the public interface for
   * creating new instances.
   */
  static sptr make(const types::shape &shape);
};

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_EYE_H */
