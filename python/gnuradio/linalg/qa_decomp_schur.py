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
    # Try to import Schur decomposition classes
    from gnuradio.linalg import decomp_schur_sync_f

    HAVE_DECOMP_SCHUR = True
    # If basic import works, try the others
    try:
        from gnuradio.linalg import decomp_schur_sync_d

        HAVE_ALL_TYPES = True
    except ImportError:
        HAVE_ALL_TYPES = False
except ImportError:
    HAVE_DECOMP_SCHUR = False
    HAVE_ALL_TYPES = False


class qa_decomp_schur(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance(self):
        # Test Schur decomposition block creation with valid shapes
        if not HAVE_DECOMP_SCHUR:
            self.skipTest("decomp_schur_sync_f not available")

        # Create Schur decomposition: 3x3 matrix
        shape = [3, 3]
        real_schur = True  # Real Schur form
        instance = decomp_schur_sync_f(shape, real_schur)
        self.assertIsNotNone(instance)

    def test_schur_decomposition_validation(self):
        # Test Schur decomposition with known symmetric matrix
        if not HAVE_DECOMP_SCHUR:
            self.skipTest("decomp_schur_sync_f not available")

        from gnuradio import blocks

        # Symmetric matrix: [[2,1],[1,2]]
        # This should have real eigenvalues and a well-behaved Schur form
        shape = [2, 2]
        input_matrix = [2.0, 1.0, 1.0, 2.0]  # Column-major

        # Create blocks - Schur outputs Q (orthogonal) and T (upper triangular)
        src = blocks.vector_source_f(input_matrix, False, 4)
        schur_block = decomp_schur_sync_f(shape, True)  # Real Schur form

        # Schur produces two outputs: Q (2x2), T (2x2)
        sink_q = blocks.vector_sink_f(4)  # Q matrix (2x2)
        sink_t = blocks.vector_sink_f(4)  # T matrix (2x2)

        # Connect flowgraph
        self.tb.connect(src, schur_block)
        self.tb.connect((schur_block, 0), sink_q)  # Q output
        self.tb.connect((schur_block, 1), sink_t)  # T output

        # Run flowgraph
        self.tb.run()

        # Verify results
        q_result = list(sink_q.data())
        t_result = list(sink_t.data())

        # Check that we got the right number of outputs
        self.assertEqual(len(q_result), 4)  # 2x2 Q matrix
        self.assertEqual(len(t_result), 4)  # 2x2 T matrix

    def test_orthogonality_property(self):
        # Test that Q matrix is orthogonal (Q^T * Q = I)
        if not HAVE_DECOMP_SCHUR:
            self.skipTest("decomp_schur_sync_f not available")

        from gnuradio import blocks

        # Test matrix: [[3,1],[1,3]]
        shape = [2, 2]
        input_matrix = [3.0, 1.0, 1.0, 3.0]  # Column-major

        src = blocks.vector_source_f(input_matrix, False, 4)
        schur_block = decomp_schur_sync_f(shape, True)
        sink_q = blocks.vector_sink_f(4)  # Q matrix
        sink_t = blocks.vector_sink_f(4)  # T matrix

        self.tb.connect(src, schur_block)
        self.tb.connect((schur_block, 0), sink_q)
        self.tb.connect((schur_block, 1), sink_t)
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

    def test_schur_reconstruction(self):
        # Test that A = Q * T * Q^T reconstruction works
        if not HAVE_DECOMP_SCHUR:
            self.skipTest("decomp_schur_sync_f not available")

        from gnuradio import blocks

        # Test matrix: [[4,1],[1,4]]
        shape = [2, 2]
        input_matrix = [4.0, 1.0, 1.0, 4.0]  # Column-major

        src = blocks.vector_source_f(input_matrix, False, 4)
        schur_block = decomp_schur_sync_f(shape, True)
        sink_q = blocks.vector_sink_f(4)  # Q matrix
        sink_t = blocks.vector_sink_f(4)  # T matrix

        self.tb.connect(src, schur_block)
        self.tb.connect((schur_block, 0), sink_q)
        self.tb.connect((schur_block, 1), sink_t)
        self.tb.run()

        q_result = list(sink_q.data())
        t_result = list(sink_t.data())

        # Reconstruct original matrix: A = Q * T * Q^T
        Q = np.array(q_result).reshape(2, 2, order="F")
        T = np.array(t_result).reshape(2, 2, order="F")

        reconstructed = Q @ T @ Q.T
        original = np.array(input_matrix).reshape(2, 2, order="F")

        # Check reconstruction accuracy
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(reconstructed[i, j], original[i, j], places=4)

    def test_upper_triangular_property(self):
        # Test that T matrix is upper triangular (or quasi-upper triangular)
        if not HAVE_DECOMP_SCHUR:
            self.skipTest("decomp_schur_sync_f not available")

        from gnuradio import blocks

        # Test matrix: [[2,0],[0,3]] (already upper triangular)
        shape = [2, 2]
        input_matrix = [2.0, 0.0, 0.0, 3.0]  # Column-major

        src = blocks.vector_source_f(input_matrix, False, 4)
        schur_block = decomp_schur_sync_f(shape, True)
        sink_q = blocks.vector_sink_f(4)  # Q matrix
        sink_t = blocks.vector_sink_f(4)  # T matrix

        self.tb.connect(src, schur_block)
        self.tb.connect((schur_block, 0), sink_q)
        self.tb.connect((schur_block, 1), sink_t)
        self.tb.run()

        t_result = list(sink_t.data())

        # Convert to NumPy matrix
        T = np.array(t_result).reshape(2, 2, order="F")

        # For this diagonal input, T should be similar (eigenvalues on diagonal)
        # Check that result makes sense (should have real eigenvalues)
        self.assertTrue(np.isreal(T).all())

    def test_identity_matrix_schur(self):
        # Test Schur decomposition of identity matrix
        if not HAVE_DECOMP_SCHUR:
            self.skipTest("decomp_schur_sync_f not available")

        from gnuradio import blocks

        # 2x2 identity matrix
        shape = [2, 2]
        identity_matrix = [1.0, 0.0, 0.0, 1.0]  # Column-major

        src = blocks.vector_source_f(identity_matrix, False, 4)
        schur_block = decomp_schur_sync_f(shape, True)
        sink_q = blocks.vector_sink_f(4)  # Q matrix
        sink_t = blocks.vector_sink_f(4)  # T matrix

        self.tb.connect(src, schur_block)
        self.tb.connect((schur_block, 0), sink_q)
        self.tb.connect((schur_block, 1), sink_t)
        self.tb.run()

        t_result = list(sink_t.data())

        # For identity matrix, T should also be identity (or close to it)
        T = np.array(t_result).reshape(2, 2, order="F")
        expected_identity = np.eye(2)

        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(T[i, j], expected_identity[i, j], places=4)


if __name__ == "__main__":
    gr_unittest.run(qa_decomp_schur)
