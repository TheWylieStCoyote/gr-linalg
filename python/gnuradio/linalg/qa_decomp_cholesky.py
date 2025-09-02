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
    from gnuradio.linalg import decomp_cholesky
except ImportError:  # Fallback for local run without install
    import os
    import sys

    dirname, filename = os.path.split(os.path.abspath(__file__))
    sys.path.append(os.path.join(dirname, "bindings"))
    from gnuradio.linalg import decomp_cholesky


class qa_decomp_cholesky(gr_unittest.TestCase):
    """Flowgraph tests for Cholesky decomposition sync block.

    The factory decomp_cholesky() returns the double precision
    sync block (decomp_cholesky_sync_d) by default. We feed full
    matrices as single stream items using vector_source_d with
    vlen = N*N (column-major ordering) and expect an output item
    of the same length containing lower-triangular factor L.
    Validation: reconstruct A' = L * L^T and compare to A.
    """

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    @staticmethod
    def _col_major_to_matrix(data, n):
        """Convert flat column-major list/tuple to square matrix."""
        M = np.zeros((n, n), dtype=np.float32)
        for j in range(n):
            col = data[j * n : (j + 1) * n]
            M[:, j] = col
        return M

    def _run_single_matrix(self, A):
        n = A.shape[0]
        # A.T.flatten(): numpy row-major -> column-major serialization
        flat_col_major = A.T.flatten()
        vlen = n * n

        src = blocks.vector_source_f(
            flat_col_major.astype(np.float32).tolist(), repeat=False, vlen=vlen
        )
        blk = decomp_cholesky([n, n])
        snk = blocks.vector_sink_f(vlen=vlen)

        self.tb.connect(src, blk)
        self.tb.connect(blk, snk)
        self.tb.run()

        out_data = snk.data()
        self.assertEqual(
            len(out_data), vlen, "Expected single output item of size vlen"
        )

        L = self._col_major_to_matrix(out_data, n)
        A_recon = L @ L.T
        self.assertTrue(
            np.allclose(A, A_recon, atol=1e-8),
            f"Cholesky mismatch\nA=\n{A}\nL=\n{L}\nL L^T=\n{A_recon}",
        )
        self.assertTrue(np.allclose(L, np.tril(L)), "L is not lower-triangular")
        return L

    def test_cholesky_2x2(self):
        A = np.array([[4.0, 2.0], [2.0, 3.0]], dtype=np.float32)
        self._run_single_matrix(A)

    def test_cholesky_3x3(self):
        A = np.array(
            [
                [6.0, 3.0, 4.0],
                [3.0, 6.0, 5.0],
                [4.0, 5.0, 10.0],
            ],
            dtype=np.float32,
        )
        self._run_single_matrix(A)

    def test_multiple_matrices(self):
        matrices = [
            np.array([[4.0, 2.0], [2.0, 3.0]], dtype=np.float32),
            np.array([[9.0, 3.0], [3.0, 5.0]], dtype=np.float32),
            np.array([[25.0, 5.0], [5.0, 26.0]], dtype=np.float32),
        ]
        n = 2
        vlen = n * n
        concatenated = []
        for A in matrices:
            concatenated.extend(A.T.flatten())

        src = blocks.vector_source_f(concatenated, repeat=False, vlen=vlen)
        blk = decomp_cholesky([n, n])
        snk = blocks.vector_sink_f(vlen=vlen)

        self.tb.connect(src, blk)
        self.tb.connect(blk, snk)
        self.tb.run()

        out_data = list(snk.data())
        self.assertEqual(len(out_data), len(matrices) * vlen)

        for idx, A in enumerate(matrices):
            item = out_data[idx * vlen : (idx + 1) * vlen]
            L = self._col_major_to_matrix(item, n)
            A_recon = L @ L.T
            self.assertTrue(
                np.allclose(A, A_recon, atol=1e-8),
                f"Mismatch matrix {idx}\nA=\n{A}\nL=\n{L}\n",
            )


if __name__ == "__main__":
    gr_unittest.run(qa_decomp_cholesky)
