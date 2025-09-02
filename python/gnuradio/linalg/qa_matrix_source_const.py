#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2025 Wylie Standage-Beier.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

from gnuradio import gr, gr_unittest, blocks  # type: ignore

try:
    from gnuradio.linalg import matrix_source_const
except ImportError:
    import os
    import sys

    dirname, filename = os.path.split(os.path.abspath(__file__))
    sys.path.append(os.path.join(dirname, "bindings"))
    from gnuradio.linalg import matrix_source_const


class qa_matrix_source_const(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance(self):
        """Instantiate with simple 2x2 matrix or skip if unavailable."""
        matrix = [[1.0, 2.0], [3.0, 4.0]]
        shape = [2, 2]
        blk = matrix_source_const(matrix, shape)
        if blk is None:
            self.skipTest("matrix_source_const factory returned None (bindings absent)")
        self.assertIsNotNone(blk)

    def test_constant_matrix_output(self):
        """Verify output delivers constant matrix data.

        Assumptions:
        - One output port exists (index 0)
        - Output emits matrices flattened column-major per item

        If the block is unbound, the test skips.
        """
        matrix = [[5.0, -1.0, 0.5], [2.25, 3.5, 7.0]]  # 2x3
        shape = [2, 3]
        # Column-major order: col0=[5.0, 2.25], col1=[-1.0, 3.5], col2=[0.5, 7.0]
        expected_flat_col_major = [5.0, 2.25, -1.0, 3.5, 0.5, 7.0]

        src = matrix_source_const(matrix, shape)
        if src is None:
            self.skipTest("matrix_source_const factory returned None (bindings absent)")

        sink = blocks.vector_sink_f(vlen=6)  # 2x3 = 6 elements
        head = blocks.head(gr.sizeof_float * 6, 1)  # Limit to 1 matrix

        self.tb.connect(src, head)
        self.tb.connect(head, sink)
        self.tb.run()

        output = list(sink.data())

        self.assertGreaterEqual(len(output), len(expected_flat_col_major))

        slice_output = output[: len(expected_flat_col_major)]

        # Test that the output matches expected column-major order
        self.assertEqual(slice_output, expected_flat_col_major)


if __name__ == "__main__":
    gr_unittest.run(qa_matrix_source_const)
