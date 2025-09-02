/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_BLOCK_SYNC_BASE_SYNC_H
#define INCLUDED_BLOCK_SYNC_BASE_SYNC_H

#include <gnuradio/linalg/api.h>
#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/types.h>
#include <gnuradio/sync_decimator.h>

namespace gr {
namespace linalg {

// Use the same enum types as the base class for consistency
using ERROR_TAG = error_tag_t;
using ERROR_PDU = error_pdu_p;

/*!
 * \brief Base class for synchronous linear algebra blocks
 *
 * This class provides the basic interface and functionality for synchronous
 * linear algebra blocks.
 */
template <class Scalar>
class LINALG_API linalg_base_decim : virtual public linalg_base<Scalar>,
                                     public gr::sync_decimator {
public:
  /*!
   * \brief Constructor for linalg_base_decim
   * \param name Name of the block
   * \param shape_inputs Vector of input shapes
   * \param shape_outputs Vector of output shapes
   * \param broadcast_type Type of broadcasting to apply to output shapes
   * \param tag_errors Tag errors for operation calls
   * \param pdu_errors Emit PDUs on errors for operation calls
   */
  linalg_base_decim(const std::string &name,
                    const types::vector_shapes &shape_inputs,
                    const types::vector_shapes &shape_outputs, int decimation);

  /*!
   * \brief Constructor for linalg_base_decim with input/output names
   * \param name Name of the block
   * \param shape_inputs Vector of input shapes
   * \param input_names Vector of input port names
   * \param shape_outputs Vector of output shapes
   * \param output_names Vector of output port names
   * \param broadcast_type Type of broadcasting to apply to output shapes
   * \param tag_errors Tag errors for operation calls
   * \param pdu_errors Emit PDUs on errors for operation calls
   */
  linalg_base_decim(
      const std::string &name, const types::vector_shapes &shape_inputs,
      const std::vector<std::string> &input_names,
      const types::vector_shapes &shape_outputs,
      const std::vector<std::string> &output_names, int decimation,
      const array_broadcast_type &broadcast_type = array_broadcast_type::NONE,
      const ERROR_TAG &tag_errors = error_tag_t::NONE,
      const ERROR_PDU &pdu_errors = error_pdu_p::NONE,
      const gr::block::tag_propagation_policy_t tag_propagation_policy =
          gr::block::TPP_ALL_TO_ALL);

protected:
  /*!
   * \brief Protected default constructor for virtual inheritance scenarios
   *
   * This constructor is used when virtual inheritance requires intermediate
   * classes to not initialize the virtual base, leaving that responsibility
   * to the most-derived class.
   */
  // linalg_base_decim()
  //     : gr::sync_block("", gr::io_signature::make(0, 0, 0),
  //                      gr::io_signature::make(0, 0, 0)) {}

public:
  /*!
   * \brief Destructor for linalg_base_decim
   */
  virtual ~linalg_base_decim() = default;

  /*!
   * \brief Work function for linalg_base_decim
   * \param noutput_items Number of output items to produce
   * \param input_items Vector of input item pointers
   * \param output_items Vector of output item pointers
   * \return Number of output items produced
   */
  int work(int noutput_items, gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items) override;

protected:
};

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_BLOCK_SYNC_BASE_SYNC_H */
