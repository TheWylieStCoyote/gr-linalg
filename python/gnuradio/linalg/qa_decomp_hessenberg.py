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
    # Try to import Hessenberg decomposition classes
    from gnuradio.linalg import decomp_hessenberg_sync_f

    HAVE_DECOMP_HESSENBERG = True
    # If basic import works, try the others
    try:
        from gnuradio.linalg import decomp_hessenberg_sync_d

        HAVE_ALL_TYPES = True
    except ImportError:
        HAVE_ALL_TYPES = False
except ImportError:
    HAVE_DECOMP_HESSENBERG = False
    HAVE_ALL_TYPES = False


class qa_decomp_hessenberg(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance(self):
        # Test Hessenberg decomposition block creation with valid shapes
        if not HAVE_DECOMP_HESSENBERG:
            self.skipTest("decomp_hessenberg_sync_f not available")

        # Create Hessenberg decomposition: 3x3 matrix
        shape = [3, 3]
        upper_hessenberg = True  # Upper Hessenberg form
        instance = decomp_hessenberg_sync_f(shape, upper_hessenberg)
        self.assertIsNotNone(instance)

    def test_hessenberg_decomposition_validation(self):
        # Test Hessenberg decomposition with known matrix
        if not HAVE_DECOMP_HESSENBERG:
            self.skipTest("decomp_hessenberg_sync_f not available")

        from gnuradio import blocks

        # Test matrix: [[2,1,0],[1,2,1],[0,1,2]] (tridiagonal, should be close to Hessenberg)
        shape = [3, 3]
        input_matrix = [2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0]  # Column-major

        # Create blocks - Hessenberg outputs Q (orthogonal) and H (Hessenberg)
        src = blocks.vector_source_f(input_matrix, False, 9)
        hess_block = decomp_hessenberg_sync_f(shape, True)  # Upper Hessenberg

        # Hessenberg produces two outputs: Q (3x3), H (3x3)
        sink_q = blocks.vector_sink_f(9)  # Q matrix (3x3)
        sink_h = blocks.vector_sink_f(9)  # H matrix (3x3)

        # Connect flowgraph
        self.tb.connect(src, hess_block)
        self.tb.connect((hess_block, 0), sink_q)  # Q output
        self.tb.connect((hess_block, 1), sink_h)  # H output

        # Run flowgraph
        self.tb.run()

        # Verify results
        q_result = list(sink_q.data())
        h_result = list(sink_h.data())

        # Check that we got the right number of outputs
        self.assertEqual(len(q_result), 9)  # 3x3 Q matrix
        self.assertEqual(len(h_result), 9)  # 3x3 H matrix

    def test_orthogonality_property(self):
        # Test that Q matrix is orthogonal (Q^T * Q = I)
        if not HAVE_DECOMP_HESSENBERG:
            self.skipTest("decomp_hessenberg_sync_f not available")

        from gnuradio import blocks

        # Test matrix: [[1,2],[2,1]]
        shape = [2, 2]
        input_matrix = [1.0, 2.0, 2.0, 1.0]  # Column-major

        src = blocks.vector_source_f(input_matrix, False, 4)
        hess_block = decomp_hessenberg_sync_f(shape, True)
        sink_q = blocks.vector_sink_f(4)  # Q matrix
        sink_h = blocks.vector_sink_f(4)  # H matrix

        self.tb.connect(src, hess_block)
        self.tb.connect((hess_block, 0), sink_q)
        self.tb.connect((hess_block, 1), sink_h)
        self.tb.run()

        q_result = list(sink_q.data())

        # Convert to NumPy matrix for orthogonality check
        Q = np.array(q_result).reshape(2, 2, order="F")  # Column-major

        # Test orthogonality: Q^T * Q should be approximately identity
        QTQ = Q.T @ Q
        identity = np.eye(2)

        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(QTQ[i, j], identity[i, j], places=4)

    def test_hessenberg_reconstruction(self):
        # Test that A = Q * H * Q^T reconstruction works
        if not HAVE_DECOMP_HESSENBERG:
            self.skipTest("decomp_hessenberg_sync_f not available")

        from gnuradio import blocks

        # Test matrix: [[3,1],[1,3]]
        shape = [2, 2]
        input_matrix = [3.0, 1.0, 1.0, 3.0]  # Column-major

        src = blocks.vector_source_f(input_matrix, False, 4)
        hess_block = decomp_hessenberg_sync_f(shape, True)
        sink_q = blocks.vector_sink_f(4)  # Q matrix
        sink_h = blocks.vector_sink_f(4)  # H matrix

        self.tb.connect(src, hess_block)
        self.tb.connect((hess_block, 0), sink_q)
        self.tb.connect((hess_block, 1), sink_h)
        self.tb.run()

        q_result = list(sink_q.data())
        h_result = list(sink_h.data())

        # Reconstruct original matrix: A = Q * H * Q^T
        Q = np.array(q_result).reshape(2, 2, order="F")
        H = np.array(h_result).reshape(2, 2, order="F")

        reconstructed = Q @ H @ Q.T
        original = np.array(input_matrix).reshape(2, 2, order="F")

        # Check reconstruction accuracy
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(reconstructed[i, j], original[i, j], places=4)

    def test_hessenberg_structure(self):
        # Test that H matrix has proper Hessenberg structure (zeros below subdiagonal)
        if not HAVE_DECOMP_HESSENBERG:
            self.skipTest("decomp_hessenberg_sync_f not available")

        from gnuradio import blocks

        # Test 3x3 matrix
        shape = [3, 3]
        input_matrix = [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]  # Column-major

        src = blocks.vector_source_f(input_matrix, False, 9)
        hess_block = decomp_hessenberg_sync_f(shape, True)
        sink_q = blocks.vector_sink_f(9)  # Q matrix
        sink_h = blocks.vector_sink_f(9)  # H matrix

        self.tb.connect(src, hess_block)
        self.tb.connect((hess_block, 0), sink_q)
        self.tb.connect((hess_block, 1), sink_h)
        self.tb.run()

        h_result = list(sink_h.data())

        # Convert to NumPy matrix
        H = np.array(h_result).reshape(3, 3, order="F")

        # For 3x3 upper Hessenberg, H[2,0] should be approximately zero
        # (element below the first subdiagonal)
        self.assertAlmostEqual(H[2, 0], 0.0, places=4)

    def test_identity_matrix_hessenberg(self):
        # Test Hessenberg decomposition of identity matrix
        if not HAVE_DECOMP_HESSENBERG:
            self.skipTest("decomp_hessenberg_sync_f not available")

        from gnuradio import blocks

        # 2x2 identity matrix
        shape = [2, 2]
        identity_matrix = [1.0, 0.0, 0.0, 1.0]  # Column-major

        src = blocks.vector_source_f(identity_matrix, False, 4)
        hess_block = decomp_hessenberg_sync_f(shape, True)
        sink_q = blocks.vector_sink_f(4)  # Q matrix
        sink_h = blocks.vector_sink_f(4)  # H matrix

        self.tb.connect(src, hess_block)
        self.tb.connect((hess_block, 0), sink_q)
        self.tb.connect((hess_block, 1), sink_h)
        self.tb.run()

        h_result = list(sink_h.data())

        # For identity matrix, H should also be identity (already in Hessenberg form)
        H = np.array(h_result).reshape(2, 2, order="F")
        expected_identity = np.eye(2)

        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(H[i, j], expected_identity[i, j], places=4)

    def test_different_sizes(self):
        # Test Hessenberg decomposition with different matrix sizes
        if not HAVE_DECOMP_HESSENBERG:
            self.skipTest("decomp_hessenberg_sync_f not available")

        from gnuradio import blocks

        # Test with 4x4 matrix
        shape = [4, 4]
        input_matrix = [
            1.0,
            2.0,
            3.0,
            4.0,
            2.0,
            3.0,
            4.0,
            1.0,
            3.0,
            4.0,
            1.0,
            2.0,
            4.0,
            1.0,
            2.0,
            3.0,
        ]  # Column-major

        src = blocks.vector_source_f(input_matrix, False, 16)
        hess_block = decomp_hessenberg_sync_f(shape, True)
        sink_q = blocks.vector_sink_f(16)  # Q matrix (4x4)
        sink_h = blocks.vector_sink_f(16)  # H matrix (4x4)

        self.tb.connect(src, hess_block)
        self.tb.connect((hess_block, 0), sink_q)
        self.tb.connect((hess_block, 1), sink_h)
        self.tb.run()

        # Verify we got the right dimensions
        q_result = list(sink_q.data())
        h_result = list(sink_h.data())

        self.assertEqual(len(q_result), 16)  # 4x4 Q matrix
        self.assertEqual(len(h_result), 16)  # 4x4 H matrix


if __name__ == "__main__":
    gr_unittest.run(qa_decomp_hessenberg)
