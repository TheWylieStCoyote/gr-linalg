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
    from gnuradio.linalg import matrix_rank
except ImportError:
    import os
    import sys

    dirname, filename = os.path.split(os.path.abspath(__file__))
    sys.path.append(os.path.join(dirname, "bindings"))
    from gnuradio.linalg import matrix_rank


class qa_matrix_rank(gr_unittest.TestCase):
    """Flowgraph tests for matrix_rank sync block.

    ASSUMPTION (updated): Block exposes TWO output streams:
      output 0: rank scalar (vlen = 1)
      output 1: singular values (vlen = min(m,n)) or diagnostics vector.
    Current C++ impl only provides ONE output (rank scalar {1,1}). Test will
    fail until block is refactored accordingly. Refactor summary:
      * Adjust linalg_base_sync ctor to pass two output shapes
      * Modify operation() to emit singular values to second output.
    """

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    @staticmethod
    def _flatten_col_major(M):
        return M.T.flatten()  # column-major serialization

    def _run_single_matrix(self, A, expected_rank):
        m, n = A.shape
        flat = self._flatten_col_major(A)
        vlen_in = m * n
        src = blocks.vector_source_f(
            flat.astype(np.float32).tolist(), repeat=False, vlen=vlen_in
        )
        blk = matrix_rank([m, n])
        sink_rank = blocks.vector_sink_f(vlen=1)
        self.tb.connect(src, blk)
        self.tb.connect(blk, sink_rank)
        self.tb.run()
        rank_vals = sink_rank.data()
        self.assertEqual(len(rank_vals), 1, "Rank output size")
        self.assertAlmostEqual(
            rank_vals[0],
            expected_rank,
            places=5,
            msg=f"Rank mismatch: got {rank_vals[0]}, expected {expected_rank}",
        )

    def test_full_rank_3x3(self):
        A = np.array(
            [
                [1.0, 2.0, 3.0],
                [0.0, 1.0, 4.0],
                [5.0, 6.0, 0.0],
            ],
            dtype=np.float32,
        )
        self._run_single_matrix(A, expected_rank=3)

    def test_rank_deficient_3x3(self):
        # Third row = sum of first two -> rank 2
        A = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [5.0, 7.0, 9.0],
            ],
            dtype=np.float32,
        )
        self._run_single_matrix(A, expected_rank=2)

    def test_multiple_matrices(self):
        mats = [
            (np.eye(2, dtype=np.float32), 2),
            (np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float32), 1),
            (np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32), 1),
        ]
        vlen_in = 4  # 2x2
        concatenated = []
        for M, _r in mats:
            concatenated.extend(self._flatten_col_major(M).tolist())
        src = blocks.vector_source_f(concatenated, repeat=False, vlen=vlen_in)
        blk = matrix_rank([2, 2])
        sink_rank = blocks.vector_sink_f(vlen=1)
        self.tb.connect(src, blk)
        self.tb.connect(blk, sink_rank)
        self.tb.run()
        rank_data = list(sink_rank.data())
        self.assertEqual(len(rank_data), len(mats))
        for i, (_M, r) in enumerate(mats):
            self.assertAlmostEqual(
                rank_data[i],
                r,
                places=5,
                msg=f"Rank mismatch matrix {i}: got {rank_data[i]}, expected {r}",
            )


if __name__ == "__main__":
    gr_unittest.run(qa_matrix_rank)
