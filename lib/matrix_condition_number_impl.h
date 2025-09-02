/* -*- c++ -*- */
/*
 * Copyright 2024 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_LINALG_MATRIX_CONDITION_NUMBER_IMPL_H
#define INCLUDED_LINALG_MATRIX_CONDITION_NUMBER_IMPL_H

#include <gnuradio/linalg/matrix_condition_number.h>

namespace gr {
namespace linalg {

template <typename Scalar>
class matrix_condition_number_sync_impl
    : public matrix_condition_number_sync<Scalar> {
public:
  matrix_condition_number_sync_impl();
  matrix_condition_number_sync_impl(const types::shape &shape);
  ~matrix_condition_number_sync_impl() = default;
};

template <typename Scalar>
class matrix_condition_number_pdu_impl
    : public matrix_condition_number_pdu<Scalar> {
public:
  matrix_condition_number_pdu_impl();
  ~matrix_condition_number_pdu_impl() = default;
};

} // namespace linalg
} // namespace gr

#endif /* INCLUDED_LINALG_MATRIX_CONDITION_NUMBER_IMPL_H */