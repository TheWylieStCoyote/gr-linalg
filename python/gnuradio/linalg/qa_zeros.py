#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2025 Wylie Standage-Beier.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

from gnuradio import gr, gr_unittest
import numpy as np
import unittest

# from gnuradio import blocks
try:
    # Try to import zeros generator classes
    from gnuradio.linalg import zeros_sync_f

    HAVE_ZEROS = True
    # If basic import works, try the others
    try:
        from gnuradio.linalg import zeros_sync_d

        HAVE_ALL_TYPES = True
    except ImportError:
        HAVE_ALL_TYPES = False
except ImportError:
    HAVE_ZEROS = False
    HAVE_ALL_TYPES = False


class qa_zeros(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance(self):
        # Test zeros matrix generator block creation with valid shapes
        if not HAVE_ZEROS:
            self.skipTest("zeros_sync_f not available")

        # Create zeros generator: 2x2 matrix
        shape = [2, 2]
        instance = zeros_sync_f(shape)
        self.assertIsNotNone(instance)

    def test_zeros_matrix_2x2(self):
        # Test 2x2 zeros matrix generation
        if not HAVE_ZEROS:
            self.skipTest("zeros_sync_f not available")

        from gnuradio import blocks

        # Expected 2x2 zeros matrix in column-major: [[0,0],[0,0]] = [0,0,0,0]
        shape = [2, 2]
        expected_zeros = [0.0, 0.0, 0.0, 0.0]

        zeros_block = zeros_sync_f(shape)
        head = blocks.head(gr.sizeof_float * 4, 1)  # Limit to 1 matrix
        sink = blocks.vector_sink_f(4)  # 2x2 = 4 elements

        # Connect flowgraph (zeros block is a source - no input)
        self.tb.connect(zeros_block, head)
        self.tb.connect(head, sink)

        # Run flowgraph
        self.tb.run()

        # Verify results
        result = list(sink.data())
        self.assertEqual(len(result), 4)  # 2x2 = 4 elements
        for i, expected in enumerate(expected_zeros):
            self.assertAlmostEqual(result[i], expected, places=6)

    def test_zeros_matrix_3x3(self):
        # Test 3x3 zeros matrix generation
        if not HAVE_ZEROS:
            self.skipTest("zeros_sync_f not available")

        from gnuradio import blocks

        # Expected 3x3 zeros matrix
        shape = [3, 3]
        expected_zeros = [0.0] * 9  # All zeros

        zeros_block = zeros_sync_f(shape)
        head = blocks.head(gr.sizeof_float * 9, 1)  # Limit to 1 matrix
        sink = blocks.vector_sink_f(9)  # 3x3 = 9 elements

        self.tb.connect(zeros_block, head)
        self.tb.connect(head, sink)
        self.tb.run()

        result = list(sink.data())
        self.assertEqual(len(result), 9)  # 3x3 = 9 elements
        for i, expected in enumerate(expected_zeros):
            self.assertAlmostEqual(result[i], expected, places=6)

    def test_zeros_vector_1d(self):
        # Test 1D zeros vector generation (column vector)
        if not HAVE_ZEROS:
            self.skipTest("zeros_sync_f not available")

        from gnuradio import blocks

        # Expected 5x1 zeros vector
        shape = [5, 1]
        expected_zeros = [0.0, 0.0, 0.0, 0.0, 0.0]

        zeros_block = zeros_sync_f(shape)
        head = blocks.head(gr.sizeof_float * 5, 1)  # Limit to 1 matrix
        sink = blocks.vector_sink_f(5)

        self.tb.connect(zeros_block, head)
        self.tb.connect(head, sink)
        self.tb.run()

        result = list(sink.data())
        self.assertEqual(len(result), 5)
        for i, expected in enumerate(expected_zeros):
            self.assertAlmostEqual(result[i], expected, places=6)

    def test_zeros_row_vector(self):
        # Test row vector generation
        if not HAVE_ZEROS:
            self.skipTest("zeros_sync_f not available")

        from gnuradio import blocks

        # Expected 1x4 row vector
        shape = [1, 4]
        expected_zeros = [0.0, 0.0, 0.0, 0.0]

        zeros_block = zeros_sync_f(shape)
        head = blocks.head(gr.sizeof_float * 4, 1)
        sink = blocks.vector_sink_f(4)

        self.tb.connect(zeros_block, head)
        self.tb.connect(head, sink)
        self.tb.run()

        result = list(sink.data())
        self.assertEqual(len(result), 4)
        for i, expected in enumerate(expected_zeros):
            self.assertAlmostEqual(result[i], expected, places=6)

    def test_zeros_rectangular_matrix(self):
        # Test non-square matrix generation
        if not HAVE_ZEROS:
            self.skipTest("zeros_sync_f not available")

        from gnuradio import blocks

        # Expected 3x2 zeros matrix
        shape = [3, 2]
        expected_zeros = [0.0] * 6  # All zeros

        zeros_block = zeros_sync_f(shape)
        head = blocks.head(gr.sizeof_float * 6, 1)  # Limit to 1 matrix
        sink = blocks.vector_sink_f(6)  # 3x2 = 6 elements

        self.tb.connect(zeros_block, head)
        self.tb.connect(head, sink)
        self.tb.run()

        result = list(sink.data())
        self.assertEqual(len(result), 6)
        for i, expected in enumerate(expected_zeros):
            self.assertAlmostEqual(result[i], expected, places=6)

    def test_zeros_matrix_properties(self):
        # Test mathematical properties of zeros matrix
        if not HAVE_ZEROS:
            self.skipTest("zeros_sync_f not available")

        from gnuradio import blocks

        # Generate 4x3 zeros matrix and verify properties
        shape = [4, 3]
        zeros_block = zeros_sync_f(shape)
        head = blocks.head(gr.sizeof_float * 12, 1)  # Limit to 1 matrix
        sink = blocks.vector_sink_f(12)  # 4x3 = 12 elements

        self.tb.connect(zeros_block, head)
        self.tb.connect(head, sink)
        self.tb.run()

        result = list(sink.data())

        # Convert to NumPy matrix for analysis
        zeros_matrix = np.array(result).reshape(4, 3, order="F")  # Column-major

        # Property 1: All elements should be 0
        self.assertTrue(np.allclose(zeros_matrix, 0.0))

        # Property 2: Sum of all elements should be 0
        total_sum = np.sum(zeros_matrix)
        self.assertAlmostEqual(total_sum, 0.0, places=6)

        # Property 3: All row sums should be 0
        row_sums = np.sum(zeros_matrix, axis=1)
        for row_sum in row_sums:
            self.assertAlmostEqual(row_sum, 0.0, places=6)

        # Property 4: All column sums should be 0
        col_sums = np.sum(zeros_matrix, axis=0)
        for col_sum in col_sums:
            self.assertAlmostEqual(col_sum, 0.0, places=6)

        # Property 5: Matrix norm should be 0
        matrix_norm = np.linalg.norm(zeros_matrix)
        self.assertAlmostEqual(matrix_norm, 0.0, places=6)

    def test_zeros_additive_identity(self):
        # Test zeros matrix as additive identity (A + 0 = A)
        if not HAVE_ZEROS:
            self.skipTest("zeros_sync_f not available")

        from gnuradio import blocks

        # Generate 2x2 zeros matrix
        shape = [2, 2]
        zeros_block = zeros_sync_f(shape)
        head = blocks.head(gr.sizeof_float * 4, 1)  # Limit to 1 matrix
        sink = blocks.vector_sink_f(4)

        self.tb.connect(zeros_block, head)
        self.tb.connect(head, sink)
        self.tb.run()

        zeros_result = list(sink.data())
        zeros_matrix = np.array(zeros_result).reshape(2, 2, order="F")

        # Create arbitrary test matrix
        test_matrix = np.array([[1, 2], [3, 4]], dtype=float)

        # Test additive identity property: A + 0 = A
        result_matrix = test_matrix + zeros_matrix
        self.assertTrue(np.allclose(result_matrix, test_matrix))

        # Test: 0 + A = A
        result_matrix2 = zeros_matrix + test_matrix
        self.assertTrue(np.allclose(result_matrix2, test_matrix))

    @unittest.expectedFailure
    def test_zeros_double_precision(self):
        # Test double precision zeros generation
        if not HAVE_ALL_TYPES:
            self.skipTest("zeros_sync_d not available")

        from gnuradio import blocks

        # Generate 2x2 zeros matrix with double precision
        shape = [2, 2]
        expected_zeros = [0.0, 0.0, 0.0, 0.0]

        zeros_block = zeros_sync_d(shape)
        head = blocks.head(gr.sizeof_double * 4, 1)  # Limit to 1 matrix
        sink = blocks.vector_sink_d(4)  # Use double sink

        self.tb.connect(zeros_block, head)
        self.tb.connect(head, sink)
        self.tb.run()

        result = list(sink.data())
        self.assertEqual(len(result), 4)
        for i, expected in enumerate(expected_zeros):
            self.assertAlmostEqual(result[i], expected, places=12)  # Higher precision

    def test_zeros_large_matrix(self):
        # Test larger matrix generation (within reasonable limits)
        if not HAVE_ZEROS:
            self.skipTest("zeros_sync_f not available")

        from gnuradio import blocks

        # Generate 6x5 zeros matrix
        shape = [6, 5]
        matrix_size = 6 * 5  # 30 elements
        expected_zeros = [0.0] * matrix_size

        zeros_block = zeros_sync_f(shape)
        head = blocks.head(gr.sizeof_float * matrix_size, 1)  # Limit to 1 matrix
        sink = blocks.vector_sink_f(matrix_size)

        self.tb.connect(zeros_block, head)
        self.tb.connect(head, sink)
        self.tb.run()

        result = list(sink.data())
        self.assertEqual(len(result), matrix_size)

        # Verify all elements are zeros
        for i, expected in enumerate(expected_zeros):
            self.assertAlmostEqual(result[i], expected, places=6)

        # Verify matrix structure using NumPy
        zeros_matrix = np.array(result).reshape(6, 5, order="F")
        self.assertTrue(np.allclose(zeros_matrix, 0.0))

        # Additional verification: check that no elements are accidentally non-zero
        max_abs_value = np.max(np.abs(zeros_matrix))
        self.assertAlmostEqual(max_abs_value, 0.0, places=6)

    def test_zeros_single_element(self):
        # Test 1x1 matrix (single element)
        if not HAVE_ZEROS:
            self.skipTest("zeros_sync_f not available")

        from gnuradio import blocks

        shape = [1, 1]
        zeros_block = zeros_sync_f(shape)
        head = blocks.head(gr.sizeof_float * 1, 1)  # Limit to 1 matrix
        sink = blocks.vector_sink_f(1)

        self.tb.connect(zeros_block, head)
        self.tb.connect(head, sink)
        self.tb.run()

        result = list(sink.data())
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], 0.0, places=6)

    def test_zeros_multiplication_property(self):
        # Test zeros matrix multiplication property (A * 0 = 0, 0 * A = 0)
        if not HAVE_ZEROS:
            self.skipTest("zeros_sync_f not available")

        from gnuradio import blocks

        # Generate 2x2 zeros matrix
        shape = [2, 2]
        zeros_block = zeros_sync_f(shape)
        head = blocks.head(gr.sizeof_float * 4, 1)  # Limit to 1 matrix
        sink = blocks.vector_sink_f(4)

        self.tb.connect(zeros_block, head)
        self.tb.connect(head, sink)
        self.tb.run()

        zeros_result = list(sink.data())
        zeros_matrix = np.array(zeros_result).reshape(2, 2, order="F")

        # Create arbitrary test matrices
        test_matrix_A = np.array([[1, 2], [3, 4]], dtype=float)
        test_matrix_B = np.array([[5, 6], [7, 8]], dtype=float)

        # Test multiplication properties with zeros
        # Note: These are conceptual tests since we can't directly multiply
        # with our GNU Radio blocks, but we verify the zeros matrix structure

        # Property: Element-wise multiplication with zeros gives zeros
        elementwise_result = test_matrix_A * zeros_matrix
        self.assertTrue(np.allclose(elementwise_result, zeros_matrix))

        # Verify zeros matrix maintains its zero property
        self.assertTrue(np.allclose(zeros_matrix, 0.0))


if __name__ == "__main__":
    gr_unittest.run(qa_zeros)
