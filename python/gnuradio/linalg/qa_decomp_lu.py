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
    from gnuradio.linalg import decomp_lu
except ImportError:
    import os
    import sys

    dirname, filename = os.path.split(os.path.abspath(__file__))
    sys.path.append(os.path.join(dirname, "bindings"))
    from gnuradio.linalg import decomp_lu


class qa_decomp_lu(gr_unittest.TestCase):
    """Flowgraph tests for LU decomposition sync block.

    Output layout: concatenated N x (3N) matrix [L|U|P] column-major.
    Length per item = 3 * N^2. Validate P*A == L*U and structure.
    """

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    @staticmethod
    def _col_major_to_matrix(flat, rows, cols):
        M = np.zeros((rows, cols), dtype=np.float32)
        for j in range(cols):
            M[:, j] = flat[j * rows : (j + 1) * rows]
        return M

    def _run_single_matrix(self, A):
        n = A.shape[0]
        assert A.shape[1] == n
        flat_A = A.T.flatten()
        vlen_in = n * n
        src = blocks.vector_source_f(
            flat_A.astype(np.float32).tolist(), repeat=False, vlen=vlen_in
        )
        blk = decomp_lu([n, n])
        vlen_out = 3 * n * n
        snk = blocks.vector_sink_f(vlen=vlen_out)
        self.tb.connect(src, blk)
        self.tb.connect(blk, snk)
        self.tb.run()
        out = snk.data()
        self.assertEqual(
            len(out), vlen_out, f"Unexpected output size {len(out)} vs {vlen_out}"
        )
        # Segments: first n^2 -> L, next -> U, last -> P
        L_flat = out[0 : n * n]
        U_flat = out[n * n : 2 * n * n]
        P_flat = out[2 * n * n : 3 * n * n]
        L = self._col_major_to_matrix(L_flat, n, n)
        U = self._col_major_to_matrix(U_flat, n, n)
        P = self._col_major_to_matrix(P_flat, n, n)
        PA = P @ A
        LU = L @ U
        self.assertTrue(np.allclose(PA, LU, atol=1e-6), "P*A != L*U")
        self.assertTrue(np.allclose(np.diag(L), 1.0, atol=1e-9))
        self.assertTrue(np.allclose(np.triu(L, 1), 0.0, atol=1e-9))
        self.assertTrue(np.allclose(np.tril(U, -1), 0.0, atol=1e-9))
        self.assertTrue(np.allclose(P @ P.T, np.eye(n)), "P not permutation")

    def test_lu_2x2(self):
        A = np.array([[4.0, 3.0], [2.0, 1.0]], dtype=np.float32)
        self._run_single_matrix(A)

    def test_lu_3x3(self):
        A = np.array(
            [
                [2.0, 1.0, 1.0],
                [4.0, 3.0, 3.0],
                [8.0, 7.0, 9.0],
            ],
            dtype=np.float32,
        )
        self._run_single_matrix(A)

    def test_multiple_matrices(self):
        matrices = [
            np.array([[4.0, 3.0], [2.0, 1.0]], dtype=np.float32),
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([[5.0, 1.0], [1.0, 5.0]], dtype=np.float32),
        ]
        n = 2
        vlen_in = n * n
        vlen_out = 3 * n * n
        flat_inputs = []
        for A in matrices:
            flat_inputs.extend(A.T.flatten())
        src = blocks.vector_source_f(flat_inputs, repeat=False, vlen=vlen_in)
        blk = decomp_lu([n, n])
        snk = blocks.vector_sink_f(vlen=vlen_out)
        self.tb.connect(src, blk)
        self.tb.connect(blk, snk)
        self.tb.run()
        out = list(snk.data())
        self.assertEqual(len(out), len(matrices) * vlen_out)
        for idx, A in enumerate(matrices):
            seg = out[idx * vlen_out : (idx + 1) * vlen_out]
            L_flat = seg[0 : n * n]
            U_flat = seg[n * n : 2 * n * n]
            P_flat = seg[2 * n * n : 3 * n * n]
            L = self._col_major_to_matrix(L_flat, n, n)
            U = self._col_major_to_matrix(U_flat, n, n)
            P = self._col_major_to_matrix(P_flat, n, n)
            PA = P @ A
            LU = L @ U
            self.assertTrue(np.allclose(PA, LU, atol=1e-6), "Mismatch")


if __name__ == "__main__":
    gr_unittest.run(qa_decomp_lu)
