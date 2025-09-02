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
    from gnuradio.linalg import matrix_inverse
except ImportError:
    import os
    import sys

    dirname, filename = os.path.split(os.path.abspath(__file__))
    sys.path.append(os.path.join(dirname, "bindings"))
    try:
        from gnuradio.linalg import matrix_inverse
    except ImportError:
        matrix_inverse = None


class qa_matrix_inverse(gr_unittest.TestCase):
    """Flowgraph tests for matrix_inverse sync block.

    ASSUMPTION: Block provides ONE output:
      output 0: inverse matrix vlen = N*N
    Input: one N x N matrix per stream item (vlen = N*N) column-major.
    """

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    @staticmethod
    def _col_major_to_matrix(flat, n):
        M = np.zeros((n, n), dtype=np.float64)
        for j in range(n):
            M[:, j] = flat[j * n : (j + 1) * n]
        return M

    def _run_single_matrix(self, A):
        n = A.shape[0]
        assert A.shape[1] == n, "Square matrix required"
        flat_A = A.T.flatten().astype(
            np.float32
        )  # column-major serialization, convert to float32
        vlen = n * n
        src = blocks.vector_source_f(flat_A.tolist(), repeat=False, vlen=vlen)
        blk = matrix_inverse([n, n])
        if blk is None:
            self.skipTest("matrix_inverse not available in Python bindings")
        sink_inv = blocks.vector_sink_f(vlen=vlen)
        self.tb.connect(src, blk)
        self.tb.connect(blk, sink_inv)
        self.tb.run()
        data_inv = sink_inv.data()
        self.assertEqual(len(data_inv), vlen, "Inverse matrix size mismatch")
        A_inv = self._col_major_to_matrix(data_inv, n)
        I_recon = A @ A_inv
        self.assertTrue(
            np.allclose(I_recon, np.eye(n), atol=1e-6),
            "Inverse check failed",
        )

    def test_inverse_2x2(self):
        if matrix_inverse is None:
            self.skipTest("matrix_inverse not available in Python bindings")
        A = np.array([[4.0, 7.0], [2.0, 6.0]], dtype=np.float32)
        self._run_single_matrix(A)

    def test_inverse_3x3(self):
        if matrix_inverse is None:
            self.skipTest("matrix_inverse not available in Python bindings")
        A = np.array(
            [
                [3.0, 0.0, 2.0],
                [2.0, 0.0, -2.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        self._run_single_matrix(A)


if __name__ == "__main__":
    gr_unittest.run(qa_matrix_inverse)
