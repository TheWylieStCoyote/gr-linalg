#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2025 Wylie Standage-Beier.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

from gnuradio import gr, gr_unittest
import numpy as np

# from gnuradio import blocks
try:
    # Try to import SVD decomposition classes
    from gnuradio.linalg import decomp_svd_sync_f

    HAVE_DECOMP_SVD = True
    # If basic import works, try the others
    try:
        from gnuradio.linalg import decomp_svd_sync_d

        HAVE_ALL_TYPES = True
    except ImportError:
        HAVE_ALL_TYPES = False
except ImportError:
    HAVE_DECOMP_SVD = False
    HAVE_ALL_TYPES = False


class qa_decomp_svd(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance(self):
        # Test SVD decomposition block creation with valid shapes
        if not HAVE_DECOMP_SVD:
            self.skipTest("decomp_svd_sync_f not available")

        # Create SVD decomposition: 3x2 matrix
        shape = [3, 2]
        instance = decomp_svd_sync_f(shape)
        self.assertIsNotNone(instance)

    def test_svd_decomposition_validation(self):
        # Test SVD decomposition with known input
        if not HAVE_DECOMP_SVD:
            self.skipTest("decomp_svd_sync_f not available")

        from gnuradio import blocks

        # Create test matrix: 2x2 matrix [[3,1],[1,3]]
        # This has known SVD: singular values should be 4 and 2
        shape = [2, 2]

        # Input matrix: [[3,1],[1,3]]
        input_matrix = [3.0, 1.0, 1.0, 3.0]

        # Create blocks - SVD outputs U, S, V matrices
        src = blocks.vector_source_f(input_matrix, False, 4)  # 2x2 = 4 elements
        svd_block = decomp_svd_sync_f(shape)

        # SVD produces multiple outputs: U (2x2), S (min(m,n)), V^T (2x2)
        sink_u = blocks.vector_sink_f(4)  # U matrix (2x2)
        sink_s = blocks.vector_sink_f(2)  # Singular values (min(2,2) = 2)
        sink_vt = blocks.vector_sink_f(4)  # V^T matrix (2x2)

        # Connect flowgraph
        self.tb.connect(src, svd_block)
        self.tb.connect((svd_block, 0), sink_u)  # U output
        self.tb.connect((svd_block, 1), sink_s)  # S output
        self.tb.connect((svd_block, 2), sink_vt)  # V^T output

        # Run flowgraph
        self.tb.run()

        # Verify results - singular values should be ordered (largest first)
        u_result = list(sink_u.data())
        s_result = list(sink_s.data())
        vt_result = list(sink_vt.data())

        # Check that we got the right number of outputs
        self.assertEqual(len(u_result), 4)  # 2x2 U matrix
        self.assertEqual(len(s_result), 2)  # 2 singular values
        self.assertEqual(len(vt_result), 4)  # 2x2 V^T matrix

        # Singular values should be positive and in descending order
        self.assertGreater(s_result[0], 0)
        self.assertGreater(s_result[1], 0)
        self.assertGreaterEqual(s_result[0], s_result[1])

        # For the symmetric matrix [[3,1],[1,3]], singular values should be 4 and 2
        expected_singular_values = [4.0, 2.0]
        for i, expected in enumerate(expected_singular_values):
            self.assertAlmostEqual(s_result[i], expected, places=3)

    def test_rank_deficient_matrix(self):
        # Test SVD of rank-deficient matrix
        if not HAVE_DECOMP_SVD:
            self.skipTest("decomp_svd_sync_f not available")

        from gnuradio import blocks

        # Rank-1 matrix: [[2,4],[1,2]] (second row is half of first)
        shape = [2, 2]

        input_matrix = [2.0, 4.0, 1.0, 2.0]

        src = blocks.vector_source_f(input_matrix, False, 4)
        svd_block = decomp_svd_sync_f(shape)
        sink_u = blocks.vector_sink_f(4)  # U matrix (2x2)
        sink_s = blocks.vector_sink_f(2)  # Singular values
        sink_vt = blocks.vector_sink_f(4)  # V^T matrix (2x2)

        self.tb.connect(src, svd_block)
        self.tb.connect((svd_block, 0), sink_u)
        self.tb.connect((svd_block, 1), sink_s)  # S output
        self.tb.connect((svd_block, 2), sink_vt)

        self.tb.run()

        s_result = list(sink_s.data())

        # For rank-1 matrix, one singular value should be much larger than the other
        # The smaller one should be close to zero
        self.assertGreater(s_result[0], s_result[1])
        self.assertLess(s_result[1], 0.1)  # Should be close to zero

    def test_identity_matrix_svd(self):
        # Test SVD of identity matrix
        if not HAVE_DECOMP_SVD:
            self.skipTest("decomp_svd_sync_f not available")

        from gnuradio import blocks

        # 2x2 identity matrix
        shape = [2, 2]

        # Identity matrix: [[1,0],[0,1]]
        identity_matrix = [1.0, 0.0, 0.0, 1.0]

        src = blocks.vector_source_f(identity_matrix, False, 4)
        svd_block = decomp_svd_sync_f(shape)
        sink_u = blocks.vector_sink_f(4)  # U matrix (2x2)
        sink_s = blocks.vector_sink_f(2)  # Singular values
        sink_vt = blocks.vector_sink_f(4)  # V^T matrix (2x2)

        self.tb.connect(src, svd_block)
        self.tb.connect((svd_block, 0), sink_u)
        self.tb.connect((svd_block, 1), sink_s)  # S output
        self.tb.connect((svd_block, 2), sink_vt)

        self.tb.run()

        s_result = list(sink_s.data())

        # For identity matrix, all singular values should be 1
        for s_val in s_result:
            self.assertAlmostEqual(s_val, 1.0, places=5)

    def test_rectangular_matrix_svd(self):
        # Test SVD of rectangular matrix
        if not HAVE_DECOMP_SVD:
            self.skipTest("decomp_svd_sync_f not available")

        from gnuradio import blocks

        # 3x2 matrix
        shape = [3, 2]

        # Matrix: [[1,2],[3,4],[5,6]]
        input_matrix = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        src = blocks.vector_source_f(input_matrix, False, 6)
        svd_block = decomp_svd_sync_f(shape)

        # For 3x2 matrix: U is 3x3, S has 2 values, V^T is 2x2
        sink_u = blocks.vector_sink_f(9)  # U matrix (3x3)
        sink_s = blocks.vector_sink_f(2)  # Singular values (min(3,2) = 2)
        sink_vt = blocks.vector_sink_f(4)  # V^T matrix (2x2)

        self.tb.connect(src, svd_block)
        self.tb.connect((svd_block, 0), sink_u)
        self.tb.connect((svd_block, 1), sink_s)
        self.tb.connect((svd_block, 2), sink_vt)

        self.tb.run()

        u_result = list(sink_u.data())
        s_result = list(sink_s.data())
        vt_result = list(sink_vt.data())

        # Check dimensions
        self.assertEqual(len(u_result), 9)  # 3x3 U matrix
        self.assertEqual(len(s_result), 2)  # 2 singular values
        self.assertEqual(len(vt_result), 4)  # 2x2 V^T matrix

        # Singular values should be positive and ordered
        self.assertGreater(s_result[0], 0)
        self.assertGreater(s_result[1], 0)
        self.assertGreaterEqual(s_result[0], s_result[1])

    def test_orthogonality_properties(self):
        # Test that SVD produces orthogonal U and V matrices
        if not HAVE_DECOMP_SVD:
            self.skipTest("decomp_svd_sync_f not available")

        from gnuradio import blocks

        # Test with 2x2 matrix
        shape = [2, 2]
        input_matrix = [2.0, 1.0, 1.0, 2.0]  # [[2,1],[1,2]]

        src = blocks.vector_source_f(input_matrix, False, 4)
        svd_block = decomp_svd_sync_f(shape)
        sink_u = blocks.vector_sink_f(4)  # U matrix (2x2)
        sink_s = blocks.vector_sink_f(2)  # Singular values
        sink_vt = blocks.vector_sink_f(4)  # V^T matrix (2x2)

        self.tb.connect(src, svd_block)
        self.tb.connect((svd_block, 0), sink_u)
        self.tb.connect((svd_block, 1), sink_s)
        self.tb.connect((svd_block, 2), sink_vt)

        self.tb.run()

        u_result = list(sink_u.data())
        vt_result = list(sink_vt.data())

        # Convert to NumPy matrices for orthogonality check
        U = np.array(u_result).reshape(2, 2, order="F")  # Column-major
        VT = np.array(vt_result).reshape(2, 2, order="F")  # Column-major

        # Test orthogonality: U^T * U should be approximately identity
        UTU = U.T @ U
        identity = np.eye(2)

        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(UTU[i, j], identity[i, j], places=4)

    def test_reconstruction_property(self):
        # Test that A = U * S * V^T reconstruction works
        if not HAVE_DECOMP_SVD:
            self.skipTest("decomp_svd_sync_f not available")

        from gnuradio import blocks

        # Test with 2x2 matrix
        shape = [2, 2]
        input_matrix = [3.0, 1.0, 1.0, 3.0]  # [[3,1],[1,3]]

        src = blocks.vector_source_f(input_matrix, False, 4)
        svd_block = decomp_svd_sync_f(shape)
        sink_u = blocks.vector_sink_f(4)  # U matrix (2x2)
        sink_s = blocks.vector_sink_f(2)  # Singular values
        sink_vt = blocks.vector_sink_f(4)  # V^T matrix (2x2)

        self.tb.connect(src, svd_block)
        self.tb.connect((svd_block, 0), sink_u)
        self.tb.connect((svd_block, 1), sink_s)
        self.tb.connect((svd_block, 2), sink_vt)

        self.tb.run()

        u_result = list(sink_u.data())
        s_result = list(sink_s.data())
        vt_result = list(sink_vt.data())

        # Reconstruct original matrix: A = U * S * V^T
        U = np.array(u_result).reshape(2, 2, order="F")
        S = np.diag(s_result)
        VT = np.array(vt_result).reshape(2, 2, order="F")

        reconstructed = U @ S @ VT
        original = np.array(input_matrix).reshape(2, 2, order="F")

        # Check reconstruction accuracy
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(reconstructed[i, j], original[i, j], places=4)


if __name__ == "__main__":
    gr_unittest.run(qa_decomp_svd)
