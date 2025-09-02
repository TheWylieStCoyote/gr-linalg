/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_LINALG_BASE_CONST_SYNC_H
#define INCLUDED_LINALG_LINALG_BASE_CONST_SYNC_H

#include <gnuradio/linalg/api.h>
#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/types.h>
#include <gnuradio/linalg/utils.h>
#include <gnuradio/sync_block.h>
#include <string>
#include <vector>

namespace gr {
namespace linalg {

// Use the same enum types as the base class for consistency
enum class const_tag_t {
  NONE = 0,   // No constant tags
  TAGS = 1,   // Properly tagged update constant
  CUSTOM = -1 // Custom constant tag logic defined by derived class
};

enum class const_pdu_t {
  NONE = 0,   // No constant PDUs
  TAGS = 1,   // Properly tagged update constant
  CUSTOM = -1 // Custom constant PDU logic defined by derived class
};

/*!
 * \class linalg_base_const_sync
 * \brief Synchronous linear–algebra processing base class with one or more
 *        compile-/runtime configurable constant (non–stream) data inputs.
 *
 * This helper/base block mixes ordinary streaming input ports with a fixed
 * collection of constant data arrays (e.g., coefficient vectors, matrices,
 * lookup tables) that participate in the work() computation each call. The
 * constant data can be supplied at construction and updated later (per array
 * or wholesale) without needing to reinstantiate the block. Tag and PDU
 * emission behaviors on both normal operation and error conditions are
 * controlled through policy enums provided at construction.
 *
 * Design notes:
 * - Constant data are stored internally as a std::vector<std::vector<Scalar>>
 *   where the outer index selects one "const input" (0 .. num_const_data-1)
 *   and the inner vector holds that input's elements.
 * - Broadcasting of constant data to output shapes (or between differing
 *   dimensionalities) is governed by array_broadcast_type.
 * - All constant data supplied (initially or via setters) are validated for
 *   size / shape consistency by validate_const_data(); invalid inputs raise
 *   (or log) an error according to the configured policies.
 *
 * Thread-safety:
 * - set_const_data() is not inherently thread-safe relative to work(). If
 *   runtime updates can overlap with processing, the caller must externally
 *   synchronize or ensure no concurrent access.
 */
template <class Scalar>
class LINALG_API linalg_base_const_sync
    : virtual public linalg_base_sync<Scalar> {
protected:
  const_tag_t d_tag_const; // Constant tag policy for operation calls
  const_pdu_t d_pdu_const; // Constant PDU policy for operation calls

  size_t d_num_const_data; // Number of constant data elements
  std::vector<std::vector<Scalar>> d_const_data; // Constant data for operations
  types::vector_shapes
      d_const_data_shapes; // Shapes for constant data validation

  void validate_const_data(const std::vector<std::vector<Scalar>> &data) const;

public:
  /*!
   * \brief Full constructor providing maximal control over names, shapes, tag /
   *        PDU policies, broadcasting, and error handling.
   *
   * Use this overload when:
   * - You need custom (human-readable) port names.
   * - You want to specify both input and output shapes explicitly.
   * - You must control tag propagation, tag constancy, and PDU emission
   * policies.
   *
   * Parameter semantics:
   * \param name Human-readable block name (also used for logging / tags).
   * \param num_const_data Number of constant data "ports" (logical constant
   * inputs). Must match the size of the provided constant-data container.
   * \param const_data (overload 1 - outer container) Collection of
   * constant data arrays: const_data.size() == num_const_data. Each inner
   * vector contains the elements for one constant data input. \param const_data
   * (overload 2 - flattened form) Optional single constant data array when only
   * one constant dataset is supplied (present due to API design; if both forms
   * are provided, the vector-of-vectors takes precedence). NOTE: The duplicated
   * name in the signature may reflect an overload or a pending refactor; ensure
   * only one form is used coherently. \param shape_inputs Vector of expected
   * input stream shapes (one per streaming input port). \param input_names
   * Optional vector of user-defined input port names; size must equal
   * shape_inputs.size() if non-empty. \param shape_outputs Vector of output
   * shapes (one per output port). \param output_names Optional vector of
   * user-defined output port names; size must equal shape_outputs.size() if
   * non-empty. \param tag_const Policy controlling how (or if) existing tags on
   * input streams are reused / fixed across operations. \param pdu_const Policy
   * controlling interpretation / reuse of PDUs as constant data inputs. \param
   * broadcast_type Broadcasting rule applied when constant data shapes differ
   * from required output shapes (e.g., NONE, ROW, COLUMN, AUTO). \param
   * tag_errors Policy describing whether and how error conditions generate
   * diagnostic tags. \param pdu_errors Policy describing whether error
   * conditions emit PDUs. \param tag_propagation_policy GNU Radio core tag
   * propagation behavior (e.g., TPP_ALL_TO_ALL).
   *
   * Validation & errors:
   * - Throws / signals (according to policies) if num_const_data mismatches the
   *   supplied constant-data container size or shape inconsistencies arise.
   * - Broadcasting incompatibilities result in error handling per tag_errors /
   *   pdu_errors.
   */
  linalg_base_const_sync(
      const std::string &name, // Name of the block
      size_t num_const_data,   // Number of constant data elements streams
      const std::vector<std::vector<Scalar>>
          &const_data,                          // Constant data for operations
      const types::vector_shapes &shape_inputs, // Input shapes
      const std::vector<std::string> &input_names,  // Input port names
      const types::vector_shapes &shape_outputs,    // Output shapes
      const std::vector<std::string> &output_names, // Output port names
      const_tag_t tag_const = const_tag_t::TAGS,    // Constant tag policy
      const_pdu_t pdu_const = const_pdu_t::NONE,    // Constant PDU policy
      array_broadcast_type broadcast_type =
          array_broadcast_type::NONE, // Broadcasting to apply to output shapes
      error_tag_t tag_errors = error_tag_t::NONE, // Error tag policy
      error_pdu_p pdu_errors = error_pdu_p::NONE, // Error PDU policy
      gr::block::tag_propagation_policy_t tag_propagation_policy =
          gr::block::TPP_ALL_TO_ALL // Tag propagation policy
  );

  /*!
   * \brief Convenience constructor for minimal setup (no custom port names,
   *        default policies, single constant-data container).
   *
   * Use when default tag/PDU policies and automatic tag propagation are
   * acceptable and you only need to specify shapes and initial constants.
   *
   * \param name Block name.
   * \param num_const_data Number of constant data arrays.
   * \param const_data Single flattened constant dataset (use when
   *        num_const_data == 1) OR the first dataset if subsequent ones will
   *        be assigned later via set_const_data().
   * \param shape_inputs Input stream shapes.
   * \param shape_outputs Output stream shapes.
   */
  linalg_base_const_sync(
      const std::string &name, // Name of the block
      size_t num_const_data,   // Number of constant data elements streams
      const std::vector<std::vector<Scalar>>
          &const_data,                          // Constant data for operations
      const types::vector_shapes &shape_inputs, // Input shapes
      const types::vector_shapes &shape_outputs // Output shapes
  );

  /*!
   * \brief Virtual destructor to allow proper cleanup in derived classes.
   *
   * Ensures any resources or dynamic allocations introduced by subclasses are
   * released correctly.
   */
  virtual ~linalg_base_const_sync();

  /*!
   * \brief Replace the entire collection of constant data arrays.
   *
   * \param new_const_data New container of constant datasets. Must have
   *        new_const_data.size() == num_const_data. Each inner vector must
   *        conform to dimensionality / broadcasting requirements established
   *        at construction.
   *
   * Effects:
   * - Validates sizes and shapes before committing changes.
   * - On validation failure, leaves prior data intact and emits errors / tags
   * / PDUs per configured policies.
   *
   * Performance:
   * - Copies all provided arrays; prefer per-index updates for large datasets
   *   when only a subset changes.
   */

  /* End of public interface documentation */
  void set_const_data(const std::vector<std::vector<Scalar>>
                          &new_const_data); // Update constant data

  /*!
   * \brief Update a single constant data array in-place.
   *
   * \param index Zero-based constant data index to replace.
   * \param data New data vector for that constant input.
   *
   * Requirements:
   * - index < num_const_data.
   * - data length must be compatible with existing shapes (or broadcast rules).
   *
   * Failure handling mirrors set_const_data(new_const_data).
   */
  void
  set_const_data(size_t index,
                 const std::vector<Scalar> &data); // Set constant data at index

  /*!
   * \brief Core synchronous processing function integrating stream inputs with
   *        the current constant data arrays.
   *
   * \param noutput_items Specified number of output items to produce per output
   *        port (GNU Radio scheduler contract).
   * \param input_items Array of raw pointers (one per streaming input port).
   * \param output_items Array of raw pointers (one per output port) to be
   *        populated.
   *
   * Responsibilities (typical expected behavior for derived implementations):
   * - Read up to noutput_items elements from each input port.
   * - Apply linear-algebra operations that combine streaming inputs with each
   *   constant dataset (e.g., scaling, matrix multiply, vector add).
   * - Respect tag_const and tag_propagation_policy for tag forwarding.
   * - Emit or suppress error tags / PDUs per policy settings.
   *
   * Return value:
   * - Number of output items actually produced (<= noutput_items).
   * - A negative return may signal a processing error (handled by runtime).
   *
   * Assumptions:
   * - Constant data remain immutable during a single work() call; external
   *   synchronization is required if updates can occur concurrently.
   */
  int work(int noutput_items, gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items) override;

private:
};

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_LINALG_BASE_CONST_SYNC_H */
