#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2025 Wylie Standage-Beier.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import unittest
from gnuradio import gr, gr_unittest
import numpy as np

# from gnuradio import blocks
try:
    # Try to import eye (identity matrix) generator classes
    from gnuradio.linalg import eye_sync_f

    HAVE_EYE = True
    # If basic import works, try the others
    try:
        from gnuradio.linalg import eye_sync_d

        HAVE_ALL_TYPES = True
    except ImportError:
        HAVE_ALL_TYPES = False
except ImportError:
    HAVE_EYE = False
    HAVE_ALL_TYPES = False


class qa_eye(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance(self):
        # Test eye (identity) matrix generator block creation with valid shapes
        if not HAVE_EYE:
            self.skipTest("eye_sync_f not available")

        # Create eye generator: 2x2 identity matrix
        shape = [2, 2]
        instance = eye_sync_f(shape)
        self.assertIsNotNone(instance)

    def test_eye_matrix_2x2(self):
        # Test 2x2 identity matrix generation
        if not HAVE_EYE:
            self.skipTest("eye_sync_f not available")

        from gnuradio import blocks

        # Expected 2x2 identity matrix in column-major: [[1,0],[0,1]] = [1,0,0,1]
        shape = [2, 2]
        expected_identity = [1.0, 0.0, 0.0, 1.0]

        eye_block = eye_sync_f(shape)
        head = blocks.head(gr.sizeof_float * 4, 1)  # Limit to 1 matrix
        sink = blocks.vector_sink_f(4)  # 2x2 = 4 elements

        # Connect flowgraph (eye block is a source - no input)
        self.tb.connect(eye_block, head)
        self.tb.connect(head, sink)

        # Run flowgraph
        self.tb.run()

        # Verify results
        result = list(sink.data())
        self.assertEqual(len(result), 4)  # 2x2 = 4 elements
        for i, expected in enumerate(expected_identity):
            self.assertAlmostEqual(result[i], expected, places=6)

    def test_eye_matrix_3x3(self):
        # Test 3x3 identity matrix generation
        if not HAVE_EYE:
            self.skipTest("eye_sync_f not available")

        from gnuradio import blocks

        # Expected 3x3 identity matrix in column-major
        # [[1,0,0],[0,1,0],[0,0,1]] = [1,0,0,0,1,0,0,0,1]
        shape = [3, 3]
        expected_identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

        eye_block = eye_sync_f(shape)
        head = blocks.head(gr.sizeof_float * 9, 1)  # Limit to 1 matrix
        sink = blocks.vector_sink_f(9)  # 3x3 = 9 elements

        self.tb.connect(eye_block, head)
        self.tb.connect(head, sink)
        self.tb.run()

        result = list(sink.data())
        self.assertEqual(len(result), 9)  # 3x3 = 9 elements
        for i, expected in enumerate(expected_identity):
            self.assertAlmostEqual(result[i], expected, places=6)

    def test_eye_matrix_4x4(self):
        # Test 4x4 identity matrix generation
        if not HAVE_EYE:
            self.skipTest("eye_sync_f not available")

        from gnuradio import blocks

        # Generate 4x4 identity matrix
        shape = [4, 4]
        eye_block = eye_sync_f(shape)
        head = blocks.head(gr.sizeof_float * 16, 1)  # Limit to 1 matrix
        sink = blocks.vector_sink_f(16)  # 4x4 = 16 elements

        self.tb.connect(eye_block, head)
        self.tb.connect(head, sink)
        self.tb.run()

        result = list(sink.data())
        self.assertEqual(len(result), 16)

        # Convert to NumPy and verify it's an identity matrix
        identity_matrix = np.array(result).reshape(4, 4, order="F")  # Column-major
        expected_identity = np.eye(4)

        self.assertTrue(np.allclose(identity_matrix, expected_identity))

    def test_eye_matrix_properties(self):
        # Test mathematical properties of identity matrix
        if not HAVE_EYE:
            self.skipTest("eye_sync_f not available")

        from gnuradio import blocks

        # Generate 3x3 identity matrix and verify properties
        shape = [3, 3]
        eye_block = eye_sync_f(shape)
        head = blocks.head(gr.sizeof_float * 9, 1)  # Limit to 1 matrix
        sink = blocks.vector_sink_f(9)

        self.tb.connect(eye_block, head)
        self.tb.connect(head, sink)
        self.tb.run()

        result = list(sink.data())

        # Convert to NumPy matrix for analysis
        identity_matrix = np.array(result).reshape(3, 3, order="F")  # Column-major

        # Property 1: Diagonal elements should be 1
        diagonal_elements = np.diag(identity_matrix)
        for diag_element in diagonal_elements:
            self.assertAlmostEqual(diag_element, 1.0, places=6)

        # Property 2: Off-diagonal elements should be 0
        n = identity_matrix.shape[0]
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.assertAlmostEqual(identity_matrix[i, j], 0.0, places=6)

        # Property 3: Trace (sum of diagonal) should equal matrix dimension
        trace = np.trace(identity_matrix)
        self.assertAlmostEqual(trace, 3.0, places=6)

        # Property 4: Determinant should be 1
        determinant = np.linalg.det(identity_matrix)
        self.assertAlmostEqual(determinant, 1.0, places=6)

        # Property 5: Matrix should be orthogonal (I^T * I = I)
        transpose_product = identity_matrix.T @ identity_matrix
        self.assertTrue(np.allclose(transpose_product, identity_matrix))

    def test_eye_multiplicative_identity(self):
        # Test identity matrix as multiplicative identity (A * I = I * A = A)
        if not HAVE_EYE:
            self.skipTest("eye_sync_f not available")

        from gnuradio import blocks

        # Generate 2x2 identity matrix
        shape = [2, 2]
        eye_block = eye_sync_f(shape)
        head = blocks.head(gr.sizeof_float * 4, 1)  # Limit to 1 matrix
        sink = blocks.vector_sink_f(4)

        self.tb.connect(eye_block, head)
        self.tb.connect(head, sink)
        self.tb.run()

        identity_result = list(sink.data())
        identity_matrix = np.array(identity_result).reshape(2, 2, order="F")

        # Create arbitrary test matrix
        test_matrix = np.array([[2, 3], [4, 5]], dtype=float)

        # Test multiplicative identity property: A * I = A
        result_matrix_right = test_matrix @ identity_matrix
        self.assertTrue(np.allclose(result_matrix_right, test_matrix))

        # Test: I * A = A
        result_matrix_left = identity_matrix @ test_matrix
        self.assertTrue(np.allclose(result_matrix_left, test_matrix))

        # Test: I * I = I
        identity_squared = identity_matrix @ identity_matrix
        self.assertTrue(np.allclose(identity_squared, identity_matrix))

    def test_eye_inverse_property(self):
        # Test that identity matrix is its own inverse (I^(-1) = I)
        if not HAVE_EYE:
            self.skipTest("eye_sync_f not available")

        from gnuradio import blocks

        # Generate 3x3 identity matrix
        shape = [3, 3]
        eye_block = eye_sync_f(shape)
        head = blocks.head(gr.sizeof_float * 9, 1)  # Limit to 1 matrix
        sink = blocks.vector_sink_f(9)

        self.tb.connect(eye_block, head)
        self.tb.connect(head, sink)
        self.tb.run()

        result = list(sink.data())
        identity_matrix = np.array(result).reshape(3, 3, order="F")

        # Calculate inverse
        inverse_matrix = np.linalg.inv(identity_matrix)

        # Identity matrix should be its own inverse
        self.assertTrue(np.allclose(inverse_matrix, identity_matrix))

        # Verify I * I^(-1) = I
        product = identity_matrix @ inverse_matrix
        self.assertTrue(np.allclose(product, identity_matrix))

    def test_eye_eigenvalue_properties(self):
        # Test eigenvalue properties of identity matrix (all eigenvalues should be 1)
        if not HAVE_EYE:
            self.skipTest("eye_sync_f not available")

        from gnuradio import blocks

        # Generate 3x3 identity matrix
        shape = [3, 3]
        eye_block = eye_sync_f(shape)
        head = blocks.head(gr.sizeof_float * 9, 1)  # Limit to 1 matrix
        sink = blocks.vector_sink_f(9)

        self.tb.connect(eye_block, head)
        self.tb.connect(head, sink)
        self.tb.run()

        result = list(sink.data())
        identity_matrix = np.array(result).reshape(3, 3, order="F")

        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvals(identity_matrix)

        # All eigenvalues should be 1
        for eigenvalue in eigenvalues:
            self.assertAlmostEqual(eigenvalue.real, 1.0, places=6)
            self.assertAlmostEqual(eigenvalue.imag, 0.0, places=6)

    @unittest.expectedFailure
    def test_eye_double_precision(self):
        # Test double precision identity matrix generation
        if not HAVE_ALL_TYPES:
            self.skipTest("eye_sync_d not available")

        from gnuradio import blocks

        # Generate 2x2 identity matrix with double precision
        shape = [2, 2]
        expected_identity = [1.0, 0.0, 0.0, 1.0]

        eye_block = eye_sync_d(shape)
        head = blocks.head(gr.sizeof_double * 4, 1)  # Limit to 1 matrix
        sink = blocks.vector_sink_d(4)  # Use double sink

        self.tb.connect(eye_block, head)
        self.tb.connect(head, sink)
        self.tb.run()

        result = list(sink.data())
        self.assertEqual(len(result), 4)
        for i, expected in enumerate(expected_identity):
            self.assertAlmostEqual(result[i], expected, places=12)  # Higher precision

    def test_eye_large_matrix(self):
        # Test larger identity matrix generation (within reasonable limits)
        if not HAVE_EYE:
            self.skipTest("eye_sync_f not available")

        from gnuradio import blocks

        # Generate 5x5 identity matrix
        shape = [5, 5]
        matrix_size = 5 * 5  # 25 elements

        eye_block = eye_sync_f(shape)
        head = blocks.head(gr.sizeof_float * matrix_size, 1)  # Limit to 1 matrix
        sink = blocks.vector_sink_f(matrix_size)

        self.tb.connect(eye_block, head)
        self.tb.connect(head, sink)
        self.tb.run()

        result = list(sink.data())
        self.assertEqual(len(result), matrix_size)

        # Verify matrix structure using NumPy
        identity_matrix = np.array(result).reshape(5, 5, order="F")
        expected_identity = np.eye(5)
        self.assertTrue(np.allclose(identity_matrix, expected_identity))

        # Additional verification: check trace
        trace = np.trace(identity_matrix)
        self.assertAlmostEqual(trace, 5.0, places=6)

    def test_eye_single_element(self):
        # Test 1x1 identity matrix (single element = 1)
        if not HAVE_EYE:
            self.skipTest("eye_sync_f not available")

        from gnuradio import blocks

        shape = [1, 1]
        eye_block = eye_sync_f(shape)
        head = blocks.head(gr.sizeof_float * 1, 1)  # Limit to 1 matrix
        sink = blocks.vector_sink_f(1)

        self.tb.connect(eye_block, head)
        self.tb.connect(head, sink)
        self.tb.run()

        result = list(sink.data())
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], 1.0, places=6)

    def test_eye_norm_properties(self):
        # Test norm properties of identity matrix
        if not HAVE_EYE:
            self.skipTest("eye_sync_f not available")

        from gnuradio import blocks

        # Generate 3x3 identity matrix
        shape = [3, 3]
        eye_block = eye_sync_f(shape)
        head = blocks.head(gr.sizeof_float * 9, 1)  # Limit to 1 matrix
        sink = blocks.vector_sink_f(9)

        self.tb.connect(eye_block, head)
        self.tb.connect(head, sink)
        self.tb.run()

        result = list(sink.data())
        identity_matrix = np.array(result).reshape(3, 3, order="F")

        # Frobenius norm should be sqrt(n) where n is dimension
        frobenius_norm = np.linalg.norm(identity_matrix, "fro")
        expected_frobenius = np.sqrt(3.0)  # sqrt(number of 1's on diagonal)
        self.assertAlmostEqual(frobenius_norm, expected_frobenius, places=6)

        # Spectral norm (largest singular value) should be 1
        spectral_norm = np.linalg.norm(identity_matrix, 2)
        self.assertAlmostEqual(spectral_norm, 1.0, places=6)

        # Nuclear norm (sum of singular values) should be n
        nuclear_norm = np.linalg.norm(identity_matrix, "nuc")
        self.assertAlmostEqual(nuclear_norm, 3.0, places=6)


if __name__ == "__main__":
    gr_unittest.run(qa_eye)
