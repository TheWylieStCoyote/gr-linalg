/* -*- c++ -*- */
/*
 * Copyright 2025 Wylie Standage-Beier.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LINALG_H
#define INCLUDED_LINALG_H

#include <gnuradio/linalg/linalg_base.h>
#include <gnuradio/linalg/linalg_base_pdu.h>
#include <gnuradio/linalg/linalg_base_sync.h>
#include <gnuradio/linalg/types.h>
#include <gnuradio/linalg/utils.h>

// Existing blocks
#include <gnuradio/linalg/decomp_cholesky.h>
#include <gnuradio/linalg/decomp_eigen.h>
#include <gnuradio/linalg/decomp_lu.h>
#include <gnuradio/linalg/decomp_qr.h>
#include <gnuradio/linalg/decomp_svd.h>
#include <gnuradio/linalg/dot_product.h>
#include <gnuradio/linalg/eye.h>
#include <gnuradio/linalg/matrix_add.h>
#include <gnuradio/linalg/matrix_determinant.h>
#include <gnuradio/linalg/matrix_inverse.h>
#include <gnuradio/linalg/matrix_multiply.h>
#include <gnuradio/linalg/matrix_norm.h>
#include <gnuradio/linalg/matrix_pseudo_inverse.h>
#include <gnuradio/linalg/matrix_rank.h>
#include <gnuradio/linalg/matrix_source_const.h>
#include <gnuradio/linalg/matrix_subtract.h>
#include <gnuradio/linalg/matrix_trace.h>
#include <gnuradio/linalg/matrix_transpose.h>
#include <gnuradio/linalg/ones.h>
#include <gnuradio/linalg/vector_dot.h>
#include <gnuradio/linalg/vector_norm.h>
#include <gnuradio/linalg/vector_normalize.h>
#include <gnuradio/linalg/zeros.h>

// New blocks (Advanced Matrix Operations)
#include <gnuradio/linalg/matrix_elementwise_divide.h>
#include <gnuradio/linalg/matrix_elementwise_multiply.h>

// New blocks (Advanced Vector Operations)
#include <gnuradio/linalg/vector_cross_product.h>

#endif /* INCLUDED_LINALG_H */
