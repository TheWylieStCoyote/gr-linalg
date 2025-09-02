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
    from gnuradio.linalg import decomp_qr
except ImportError:
    import os
    import sys

    dirname, filename = os.path.split(os.path.abspath(__file__))
    sys.path.append(os.path.join(dirname, "bindings"))
    from gnuradio.linalg import decomp_qr


class qa_decomp_qr(gr_unittest.TestCase):
    """Flowgraph tests for QR decomposition sync block.

    ASSUMPTION (updated): Block now has TWO output streams:
      output 0: Q matrix (vlen = N*N, column-major)
      output 1: R matrix (vlen = N*N, column-major; upper part used)
    Input: one N x N matrix per stream item (vlen = N*N, column-major).

    Validation per matrix: A ≈ Q @ R, Q^T Q ≈ I, R lower part ≈ 0.
    NOTE: Current C++ impl concatenates Q|R in one output. This test will
    fail unless refactored accordingly.
    """

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    @staticmethod
    def _col_major_to_matrix(flat, n):
        M = np.zeros((n, n), dtype=np.float32)
        for j in range(n):
            M[:, j] = flat[j * n : (j + 1) * n]
        return M

    def _run_single_matrix(self, A):
        n, m = A.shape
        assert n == m, "Square matrix expected"
        flat_A = A.T.flatten()
        src = blocks.vector_source_f(
            flat_A.astype(np.float32).tolist(), repeat=False, vlen=n * n
        )
        blk = decomp_qr([n, n])
        # QR block outputs concatenated Q|R in single output stream
        sink_QR = blocks.vector_sink_f(vlen=2 * n * n)  # Q + R matrices concatenated
        self.tb.connect(src, blk)
        self.tb.connect(blk, sink_QR)
        self.tb.run()
        qr_flat = sink_QR.data()
        self.assertEqual(len(qr_flat), 2 * n * n, "QR size mismatch")
        # Split concatenated Q|R output
        q_flat = qr_flat[: n * n]
        r_flat = qr_flat[n * n :]
        Q = self._col_major_to_matrix(q_flat, n)
        R = self._col_major_to_matrix(r_flat, n)
        A_recon = Q @ R
        self.assertTrue(np.allclose(A, A_recon, atol=1e-6), "A != Q R")
        QtQ = Q.T @ Q
        self.assertTrue(
            np.allclose(QtQ, np.eye(n), atol=1e-6),
            "Q not orthogonal",
        )
        self.assertTrue(
            np.allclose(np.tril(R, -1), 0.0, atol=1e-9),
            "R not upper",
        )

    def test_qr_2x2(self):
        A = np.array([[1.0, 2.0], [4.0, 5.0]], dtype=np.float32)
        self._run_single_matrix(A)

    def test_qr_3x3(self):
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
    gr_unittest.run(qa_decomp_qr)
