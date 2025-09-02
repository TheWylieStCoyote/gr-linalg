#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2025 Wylie Standage-Beier.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

from gnuradio import gr, gr_unittest, blocks
import numpy as np

try:
    from gnuradio.linalg import matrix_trace
except ImportError:
    import os
    import sys

    dirname, filename = os.path.split(os.path.abspath(__file__))
    sys.path.append(os.path.join(dirname, "bindings"))
    from gnuradio.linalg import matrix_trace


class qa_matrix_trace(gr_unittest.TestCase):
    """Flowgraph tests for matrix_trace sync block.

    ASSUMPTION (updated): Block exposes TWO outputs:
      output 0: trace scalar (vlen = 1)
      output 1: diagonal elements vector (vlen = N) OR unused placeholder.
    Current C++ impl provides only ONE output (trace scalar). Test will fail
    until block is refactored to produce the second stream.
    Refactor summary:
      * Provide second output shape {N,1}
      * Fill with diagonal in operation().
    """

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    @staticmethod
    def _flatten_col_major(M):
        return M.T.flatten()

    def _run_single_matrix(self, A):
        n = A.shape[0]
        assert A.shape[1] == n, "Square matrix required"
        flat = self._flatten_col_major(A)
        vlen_in = n * n
        src = blocks.vector_source_f(
            flat.astype(np.float32).tolist(), repeat=False, vlen=vlen_in
        )
        blk = matrix_trace([n, n])
        sink_trace = blocks.vector_sink_f(vlen=1)
        self.tb.connect(src, blk)
        self.tb.connect(blk, sink_trace)
        self.tb.run()
        trace_vals = sink_trace.data()
        self.assertEqual(len(trace_vals), 1, "Trace output size")
        # Expected trace
        expected_trace = np.trace(A)
        self.assertTrue(
            np.allclose(
                trace_vals[0], expected_trace, atol=1e-6
            ),  # Reduced precision for float32
            f"Trace mismatch: got {trace_vals[0]}, expected {expected_trace}",
        )

    def test_trace_2x2(self):
        A = np.array([[4.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        self._run_single_matrix(A)

    def test_trace_3x3(self):
        A = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            dtype=np.float32,
        )
        self._run_single_matrix(A)


if __name__ == "__main__":
    gr_unittest.run(qa_matrix_trace)
