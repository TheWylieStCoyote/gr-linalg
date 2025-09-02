/* -*- c++ -*- */
/*
 * Copyright 2025 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_LINALG_ARRAY_SLICE_H
#define INCLUDED_LINALG_ARRAY_SLICE_H

#include <Eigen/Dense>
#include <gnuradio/linalg/api.h>
#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/types.h>
#include <memory>
#include <vector>

namespace gr {
namespace linalg {

/**
 * @brief Array slicing operations for matrices and vectors
 *
 * Extracts subarrays from input matrices or vectors using Python-like slicing
 * syntax. Supports advanced indexing including start:stop:step notation for
 * each dimension.
 *
 * **Mathematical Background:**
 * For an input array A with shape [M, N], array slicing extracts a subarray B
 * using index ranges: B = A[row_start:row_stop:row_step,
 * col_start:col_stop:col_step]
 *
 * **Slice Parameters:**
 * Each slice specification consists of (start, stop, step) tuples:
 * - start: First index to include (0-based, negative values count from end)
 * - stop: First index to exclude (exclusive upper bound)
 * - step: Stride between indices (default 1, can be negative for reverse)
 *
 * **Applications:**
 * - Extract regions of interest from images/matrices
 * - Subsample signals by taking every nth element
 * - Reverse arrays using negative step values
 * - Extract rows/columns from matrices
 * - Windowing operations for signal processing
 * - Data preprocessing and feature extraction
 *
 * @tparam Scalar The input/output scalar type (float, double, complex<float>,
 * complex<double>)
 */
template <typename Scalar>
class LINALG_API array_slice : virtual public linalg_base<Scalar> {
public:
  typedef std::shared_ptr<array_slice<Scalar>> sptr;

  /**
   * @brief Slice specification for one dimension
   */
  struct SliceSpec {
    int start; ///< Starting index (inclusive)
    int stop;  ///< Stopping index (exclusive)
    int step;  ///< Step size (stride)

    SliceSpec(int start_ = 0, int stop_ = -1, int step_ = 1)
        : start(start_), stop(stop_), step(step_) {}
  };

  // Trivial inline ctor to allow most-derived impl to construct the virtual
  // base
  array_slice();
  array_slice(const types::shape &input_shape,
              const std::vector<SliceSpec> &slice_specs);
  array_slice(const types::shape &input_shape, const std::vector<int> &indices);

  /**
   * @brief Perform array slicing operation
   *
   * @param input_arrays Input arrays to slice
   * @param output_arrays Output sliced arrays
   * @return Operation status
   */
  OperationReturn
  operation(types::vector_const_matrix_map<Scalar> &input_arrays,
            types::vector_matrix_map<Scalar> &output_arrays);

  // Static helpers delegate to linalg_base/types
  static const types::vector_shapes
  compute_output_shapes(const types::vector_shapes &input_shapes,
                        const std::vector<SliceSpec> &slice_specs);

  static const int
  compute_output_vlen(const types::vector_shapes &input_shapes,
                      const std::vector<SliceSpec> &slice_specs);

private:
  types::shape d_input_shape;
  std::vector<SliceSpec> d_slice_specs;

  // Helper methods
  int normalize_index(int index, int dimension_size) const;
  SliceSpec normalize_slice(const SliceSpec &spec, int dimension_size) const;

  void perform_2d_slice(
      const Eigen::Map<
          const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &input,
      Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
          &output);
};

// Sync block implementation
template <typename Scalar>
class LINALG_API array_slice_sync : public linalg_base_sync<Scalar>,
                                    public array_slice<Scalar> {
public:
  typedef std::shared_ptr<array_slice_sync<Scalar>> sptr;

  static sptr
  make(const types::shape &input_shape,
       const std::vector<typename array_slice<Scalar>::SliceSpec> &slice_specs);

  array_slice_sync(
      const types::shape &input_shape,
      const std::vector<typename array_slice<Scalar>::SliceSpec> &slice_specs);
};

// PDU block implementation
template <typename Scalar>
class LINALG_API array_slice_pdu : public linalg_base_pdu<Scalar>,
                                   public array_slice<Scalar> {
public:
  typedef std::shared_ptr<array_slice_pdu<Scalar>> sptr;

  static sptr
  make(const types::vector_shapes &input_shapes,
       const std::vector<typename array_slice<Scalar>::SliceSpec> &slice_specs);

  array_slice_pdu(
      const types::vector_shapes &input_shapes,
      const std::vector<typename array_slice<Scalar>::SliceSpec> &slice_specs);
};

// Type aliases for common use cases
typedef array_slice_sync<float> array_slice_sync_f;
typedef array_slice_sync<double> array_slice_sync_d;
typedef array_slice_sync<std::complex<float>> array_slice_sync_c;
typedef array_slice_sync<std::complex<double>> array_slice_sync_cd;

typedef array_slice_pdu<float> array_slice_pdu_f;
typedef array_slice_pdu<double> array_slice_pdu_d;
typedef array_slice_pdu<std::complex<float>> array_slice_pdu_c;
typedef array_slice_pdu<std::complex<double>> array_slice_pdu_cd;

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_ARRAY_SLICE_H */
