/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_VECTOR_DOT_H
#define INCLUDED_LINALG_VECTOR_DOT_H

#include <gnuradio/linalg/api.h>
#include <gnuradio/linalg/types.h>
#include <gnuradio/sync_block.h>
#include <memory>

namespace gr {
namespace linalg {

/*!
 * \brief <+description of block+>
 * \ingroup linalg
 *
 */
class LINALG_API vector_dot : virtual public gr::sync_block {
public:
  typedef std::shared_ptr<vector_dot> sptr;

  /*!
   * \brief Return a shared_ptr to a new instance of linalg::vector_dot.
   *
   * To avoid accidental use of raw pointers, linalg::vector_dot's
   * constructor is in a private implementation
   * class. linalg::vector_dot::make is the public interface for
   * creating new instances.
   */
  static sptr make(const types::shape &shape_a, const types::shape &shape_b);
};

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_VECTOR_DOT_H */
