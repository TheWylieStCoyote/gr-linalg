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
    # Try to import ones generator classes
    from gnuradio.linalg import ones_sync_f

    HAVE_ONES = True
    # If basic import works, try the others
    try:
        from gnuradio.linalg import ones_sync_d

        HAVE_ALL_TYPES = True
    except ImportError:
        HAVE_ALL_TYPES = False
except ImportError:
    HAVE_ONES = False
    HAVE_ALL_TYPES = False


class qa_ones(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance(self):
        # Test ones matrix generator block creation with valid shapes
        if not HAVE_ONES:
            self.skipTest("ones_sync_f not available")

        # Create ones generator: 2x2 matrix
        shape = [2, 2]
        instance = ones_sync_f(shape)
        self.assertIsNotNone(instance)

    def test_ones_matrix_2x2(self):
        # Test 2x2 ones matrix generation
        if not HAVE_ONES:
            self.skipTest("ones_sync_f not available")

        from gnuradio import blocks

        # Expected 2x2 ones matrix in column-major: [[1,1],[1,1]] = [1,1,1,1]
        shape = [2, 2]
        expected_ones = [1.0, 1.0, 1.0, 1.0]

        ones_block = ones_sync_f(shape)
        head = blocks.head(gr.sizeof_float * 4, 1)  # Limit to 4 outputs
        sink = blocks.vector_sink_f(4)  # 2x2 = 4 elements

        # Connect flowgraph (ones block is a source - no input)
        self.tb.connect(ones_block, head)
        self.tb.connect(head, sink)

        # Run flowgraph
        self.tb.run()

        # Verify results
        result = list(sink.data())
        self.assertEqual(len(result), 4)  # 2x2 = 4 elements
        for i, expected in enumerate(expected_ones):
            self.assertAlmostEqual(result[i], expected, places=6)

    def test_ones_matrix_3x3(self):
        # Test 3x3 ones matrix generation
        if not HAVE_ONES:
            self.skipTest("ones_sync_f not available")

        from gnuradio import blocks

        # Expected 3x3 ones matrix
        shape = [3, 3]
        expected_ones = [1.0] * 9  # All ones

        ones_block = ones_sync_f(shape)
        head = blocks.head(gr.sizeof_float * 9, 1)  # Limit to 9 outputs
        sink = blocks.vector_sink_f(9)  # 3x3 = 9 elements

        self.tb.connect(ones_block, head)
        self.tb.connect(head, sink)
        self.tb.run()

        result = list(sink.data())
        self.assertEqual(len(result), 9)  # 3x3 = 9 elements
        for i, expected in enumerate(expected_ones):
            self.assertAlmostEqual(result[i], expected, places=6)

    def test_ones_vector_1d(self):
        # Test 1D ones vector generation (column vector)
        if not HAVE_ONES:
            self.skipTest("ones_sync_f not available")

        from gnuradio import blocks

        # Expected 4x1 ones vector
        shape = [4, 1]
        expected_ones = [1.0, 1.0, 1.0, 1.0]

        ones_block = ones_sync_f(shape)
        head = blocks.head(gr.sizeof_float * 4, 1)  # Limit to 4 outputs
        sink = blocks.vector_sink_f(4)

        self.tb.connect(ones_block, head)
        self.tb.connect(head, sink)
        self.tb.run()

        result = list(sink.data())
        self.assertEqual(len(result), 4)
        for i, expected in enumerate(expected_ones):
            self.assertAlmostEqual(result[i], expected, places=6)

    def test_ones_row_vector(self):
        # Test row vector generation
        if not HAVE_ONES:
            self.skipTest("ones_sync_f not available")

        from gnuradio import blocks

        # Expected 1x3 row vector
        shape = [1, 3]
        expected_ones = [1.0, 1.0, 1.0]

        ones_block = ones_sync_f(shape)
        head = blocks.head(gr.sizeof_float * 3, 1)  # Limit to 3 outputs
        sink = blocks.vector_sink_f(3)

        self.tb.connect(ones_block, head)
        self.tb.connect(head, sink)
        self.tb.run()

        result = list(sink.data())
        self.assertEqual(len(result), 3)
        for i, expected in enumerate(expected_ones):
            self.assertAlmostEqual(result[i], expected, places=6)

    def test_ones_rectangular_matrix(self):
        # Test non-square matrix generation
        if not HAVE_ONES:
            self.skipTest("ones_sync_f not available")

        from gnuradio import blocks

        # Expected 2x3 ones matrix
        shape = [2, 3]
        expected_ones = [1.0] * 6  # All ones

        ones_block = ones_sync_f(shape)
        head = blocks.head(gr.sizeof_float * 6, 1)  # Limit to 6 outputs
        sink = blocks.vector_sink_f(6)  # 2x3 = 6 elements

        self.tb.connect(ones_block, head)
        self.tb.connect(head, sink)
        self.tb.run()

        result = list(sink.data())
        self.assertEqual(len(result), 6)
        for i, expected in enumerate(expected_ones):
            self.assertAlmostEqual(result[i], expected, places=6)

    def test_ones_matrix_properties(self):
        # Test mathematical properties of ones matrix
        if not HAVE_ONES:
            self.skipTest("ones_sync_f not available")

        from gnuradio import blocks

        # Generate 3x3 ones matrix and verify properties
        shape = [3, 3]
        ones_block = ones_sync_f(shape)
        head = blocks.head(gr.sizeof_float * 9, 1)  # Limit to 9 outputs
        sink = blocks.vector_sink_f(9)

        self.tb.connect(ones_block, head)
        self.tb.connect(head, sink)
        self.tb.run()

        result = list(sink.data())

        # Convert to NumPy matrix for analysis
        ones_matrix = np.array(result).reshape(3, 3, order="F")  # Column-major

        # Property 1: All elements should be 1
        self.assertTrue(np.allclose(ones_matrix, 1.0))

        # Property 2: Sum of all elements should be rows * cols
        total_sum = np.sum(ones_matrix)
        expected_sum = 3 * 3  # 9
        self.assertAlmostEqual(total_sum, expected_sum, places=6)

        # Property 3: Row sums should all be equal to number of columns
        row_sums = np.sum(ones_matrix, axis=1)
        expected_row_sum = 3  # Number of columns
        for row_sum in row_sums:
            self.assertAlmostEqual(row_sum, expected_row_sum, places=6)

        # Property 4: Column sums should all be equal to number of rows
        col_sums = np.sum(ones_matrix, axis=0)
        expected_col_sum = 3  # Number of rows
        for col_sum in col_sums:
            self.assertAlmostEqual(col_sum, expected_col_sum, places=6)

    @unittest.expectedFailure
    def test_ones_double_precision(self):
        # Test double precision ones generation
        if not HAVE_ALL_TYPES:
            self.skipTest("ones_sync_d not available")

        from gnuradio import blocks

        # Generate 2x2 ones matrix with double precision
        shape = [2, 2]
        expected_ones = [1.0, 1.0, 1.0, 1.0]

        ones_block = ones_sync_d(shape)
        head = blocks.head(gr.sizeof_double * 4, 1)  # Limit to 4 outputs
        sink = blocks.vector_sink_d(4)  # Use double sink

        self.tb.connect(ones_block, head)
        self.tb.connect(head, sink)
        self.tb.run()

        result = list(sink.data())
        self.assertEqual(len(result), 4)
        for i, expected in enumerate(expected_ones):
            self.assertAlmostEqual(result[i], expected, places=12)  # Higher precision

    def test_ones_large_matrix(self):
        # Test larger matrix generation (within reasonable limits)
        if not HAVE_ONES:
            self.skipTest("ones_sync_f not available")

        from gnuradio import blocks

        # Generate 5x4 ones matrix
        shape = [5, 4]
        matrix_size = 5 * 4  # 20 elements
        expected_ones = [1.0] * matrix_size

        ones_block = ones_sync_f(shape)
        head = blocks.head(gr.sizeof_float * matrix_size, 1)  # Limit to 20 outputs
        sink = blocks.vector_sink_f(matrix_size)

        self.tb.connect(ones_block, head)
        self.tb.connect(head, sink)
        self.tb.run()

        result = list(sink.data())
        self.assertEqual(len(result), matrix_size)

        # Verify all elements are ones
        for i, expected in enumerate(expected_ones):
            self.assertAlmostEqual(result[i], expected, places=6)

        # Verify matrix structure using NumPy
        ones_matrix = np.array(result).reshape(5, 4, order="F")
        self.assertTrue(np.allclose(ones_matrix, 1.0))

    def test_ones_single_element(self):
        # Test 1x1 matrix (single element)
        if not HAVE_ONES:
            self.skipTest("ones_sync_f not available")

        from gnuradio import blocks

        shape = [1, 1]
        ones_block = ones_sync_f(shape)
        head = blocks.head(gr.sizeof_float, 1)  # Limit to 1 output
        sink = blocks.vector_sink_f(1)

        self.tb.connect(ones_block, head)
        self.tb.connect(head, sink)
        self.tb.run()

        result = list(sink.data())
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], 1.0, places=6)


if __name__ == "__main__":
    gr_unittest.run(qa_ones)
