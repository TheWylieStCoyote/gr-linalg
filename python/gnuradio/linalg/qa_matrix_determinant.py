#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2025 Wylie Standage-Beier.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

from gnuradio import gr, gr_unittest, blocks

try:
    from gnuradio.linalg import matrix_determinant
except ImportError:
    import os
    import sys

    dirname, filename = os.path.split(os.path.abspath(__file__))
    sys.path.append(os.path.join(dirname, "bindings"))
    from gnuradio.linalg import matrix_determinant


class qa_matrix_determinant(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance(self):
        """Instantiate determinant block or skip if unavailable."""
        blk = matrix_determinant([2, 2])
        if blk is None:
            self.skipTest("matrix_determinant factory returned None")
        self.assertIsNotNone(blk)

    def test_multiple_matrix_determinants(self):
        """Feed a stream of 2x2 matrices and verify determinants.

        Input layout: matrices flattened row-major with vlen = rows*cols.
        Matrices:
          A = [[1, 2], [3, 4]] det = -2
          B = [[2, 0], [0, 2]] det = 4
          C = [[0, 1], [1, 0]] det = -1
        """
        blk = matrix_determinant([2, 2])
        if blk is None:
            self.skipTest("matrix_determinant factory returned None")

        # Row-major flattened data for 3 matrices
        data = [1.0, 2.0, 3.0, 4.0, 2.0, 0.0, 0.0, 2.0, 0.0, 1.0, 1.0, 0.0]
        src = blocks.vector_source_f(data, False, 4)  # vlen=4 for 2x2
        sink = blocks.vector_sink_f()

        self.tb.connect(src, blk)
        self.tb.connect(blk, sink)
        self.tb.run()

        out = list(sink.data())
        # Expect one determinant per input matrix
        self.assertEqual(len(out), 3)
        expected = [-2.0, 4.0, -1.0]
        for got, exp in zip(out, expected):
            self.assertAlmostEqual(got, exp, places=6)


if __name__ == "__main__":
    gr_unittest.run(qa_matrix_determinant)
