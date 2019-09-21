/* -*- c++ -*- */
/*
 * Copyright 2019 Wylie Standage-Beier.
 *
 * This is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this software; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#ifndef INCLUDED_LINALG_STREAM_TO_MATRIX_CC_H
#define INCLUDED_LINALG_STREAM_TO_MATRIX_CC_H

#include <linalg/api.h>
#include <gnuradio/sync_decimator.h>

namespace gr {
  namespace linalg {

    /*!
     * \brief <+description of block+>
     * \ingroup linalg
     *
     */
    class LINALG_API stream_to_matrix_cc : virtual public gr::sync_decimator
    {
     public:
      typedef boost::shared_ptr<stream_to_matrix_cc> sptr;

      /*!
       * \brief Return a shared_ptr to a new instance of linalg::stream_to_matrix_cc.
       *
       * To avoid accidental use of raw pointers, linalg::stream_to_matrix_cc's
       * constructor is in a private implementation
       * class. linalg::stream_to_matrix_cc::make is the public interface for
       * creating new instances.
       */
      static sptr make(shape);
    };

  } // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_STREAM_TO_MATRIX_CC_H */

