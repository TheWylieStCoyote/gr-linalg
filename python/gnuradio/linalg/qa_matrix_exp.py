#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2025 Wylie Standage-Beier.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

from gnuradio import gr, gr_unittest
import numpy as np
import math

# from gnuradio import blocks
try:
    # Try to import matrix_exp classes
    from gnuradio.linalg import matrix_exp_sync_f

    HAVE_MATRIX_EXP = True
    # If basic import works, try the others
    try:
        from gnuradio.linalg import matrix_exp_sync_d

        HAVE_ALL_TYPES = True
    except ImportError:
        HAVE_ALL_TYPES = False
except ImportError:
    HAVE_MATRIX_EXP = False
    HAVE_ALL_TYPES = False


class qa_matrix_exp(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance(self):
        # Test matrix exponential block creation with valid shapes
        if not HAVE_MATRIX_EXP:
            self.skipTest("matrix_exp_sync_f not available")

        # Create matrix exponential: 2x2 matrix
        shape = [2, 2]
        instance = matrix_exp_sync_f(shape)
        self.assertIsNotNone(instance)

    def test_zero_matrix_exponential(self):
        # Test matrix exponential of zero matrix (should be identity)
        if not HAVE_MATRIX_EXP:
            self.skipTest("matrix_exp_sync_f not available")

        from gnuradio import blocks

        # Zero matrix: [[0,0],[0,0]]
        shape = [2, 2]
        zero_matrix = [0.0, 0.0, 0.0, 0.0]  # Column-major
        expected_identity = [1.0, 0.0, 0.0, 1.0]  # exp(0) = I

        src = blocks.vector_source_f(zero_matrix, False, 4)
        exp_block = matrix_exp_sync_f(shape)
        sink = blocks.vector_sink_f(4)

        self.tb.connect(src, exp_block)
        self.tb.connect(exp_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), len(expected_identity))
        for i, expected in enumerate(expected_identity):
            self.assertAlmostEqual(result[i], expected, places=5)

    def test_diagonal_matrix_exponential(self):
        # Test matrix exponential of diagonal matrix
        if not HAVE_MATRIX_EXP:
            self.skipTest("matrix_exp_sync_f not available")

        from gnuradio import blocks

        # Diagonal matrix: [[1,0],[0,2]]
        shape = [2, 2]
        diag_matrix = [1.0, 0.0, 0.0, 2.0]  # Column-major
        expected_exp = [
            math.e,
            0.0,
            0.0,
            math.e**2,
        ]  # exp(diag([1,2])) = diag([e, e^2])

        src = blocks.vector_source_f(diag_matrix, False, 4)
        exp_block = matrix_exp_sync_f(shape)
        sink = blocks.vector_sink_f(4)

        self.tb.connect(src, exp_block)
        self.tb.connect(exp_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), len(expected_exp))
        for i, expected in enumerate(expected_exp):
            self.assertAlmostEqual(result[i], expected, places=4)

    def test_small_matrix_exponential(self):
        # Test matrix exponential with small values
        if not HAVE_MATRIX_EXP:
            self.skipTest("matrix_exp_sync_f not available")

        from gnuradio import blocks

        # Small scalar matrix: [[0.1,0],[0,0.1]] = 0.1 * I
        shape = [2, 2]
        small_matrix = [0.1, 0.0, 0.0, 0.1]
        expected_exp = [
            math.exp(0.1),
            0.0,
            0.0,
            math.exp(0.1),
        ]  # exp(0.1*I) = exp(0.1)*I

        src = blocks.vector_source_f(small_matrix, False, 4)
        exp_block = matrix_exp_sync_f(shape)
        sink = blocks.vector_sink_f(4)

        self.tb.connect(src, exp_block)
        self.tb.connect(exp_block, sink)
        self.tb.run()

        result = sink.data()
        for i, expected in enumerate(expected_exp):
            self.assertAlmostEqual(result[i], expected, places=5)

    def test_identity_matrix_exponential(self):
        # Test matrix exponential of identity matrix
        if not HAVE_MATRIX_EXP:
            self.skipTest("matrix_exp_sync_f not available")

        from gnuradio import blocks

        # Identity matrix: [[1,0],[0,1]]
        shape = [2, 2]
        identity_matrix = [1.0, 0.0, 0.0, 1.0]
        expected_exp = [math.e, 0.0, 0.0, math.e]  # exp(I) = e*I

        src = blocks.vector_source_f(identity_matrix, False, 4)
        exp_block = matrix_exp_sync_f(shape)
        sink = blocks.vector_sink_f(4)

        self.tb.connect(src, exp_block)
        self.tb.connect(exp_block, sink)
        self.tb.run()

        result = sink.data()
        for i, expected in enumerate(expected_exp):
            self.assertAlmostEqual(result[i], expected, places=4)

    def test_matrix_exponential_properties(self):
        # Test mathematical properties: exp(0) = I
        if not HAVE_MATRIX_EXP:
            self.skipTest("matrix_exp_sync_f not available")

        from gnuradio import blocks

        # 3x3 zero matrix
        shape = [3, 3]
        zero_matrix = [0.0] * 9

        src = blocks.vector_source_f(zero_matrix, False, 9)
        exp_block = matrix_exp_sync_f(shape)
        sink = blocks.vector_sink_f(9)

        self.tb.connect(src, exp_block)
        self.tb.connect(exp_block, sink)
        self.tb.run()

        result = sink.data()

        # Result should be 3x3 identity matrix
        expected_identity_3x3 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        for i, expected in enumerate(expected_identity_3x3):
            self.assertAlmostEqual(result[i], expected, places=5)


if __name__ == "__main__":
    gr_unittest.run(qa_matrix_exp)
