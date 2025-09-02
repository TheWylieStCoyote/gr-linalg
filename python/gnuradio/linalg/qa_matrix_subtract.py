#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2025 Wylie Standage-Beier.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

from gnuradio import gr, gr_unittest

# from gnuradio import blocks
try:
    # Try to import all matrix_subtract classes
    from gnuradio.linalg import matrix_subtract_sync_f

    HAVE_MATRIX_SUBTRACT = True
    # If basic import works, try the others
    try:
        from gnuradio.linalg import matrix_subtract_sync_d

        HAVE_ALL_TYPES = True
    except ImportError:
        HAVE_ALL_TYPES = False
except ImportError:
    HAVE_MATRIX_SUBTRACT = False
    HAVE_ALL_TYPES = False


class qa_matrix_subtract(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance(self):
        # Test matrix subtract block creation with valid shapes
        if not HAVE_MATRIX_SUBTRACT:
            self.skipTest("matrix_subtract_sync_f not available")

        # Create matrix subtraction: 2x2 matrices
        shape = [2, 2]
        instance = matrix_subtract_sync_f(
            shape, 2
        )  # Default to 2 inputs for subtraction
        self.assertIsNotNone(instance)

    def test_matrix_subtraction_validation(self):
        # Test matrix subtraction with known input/output
        if not HAVE_MATRIX_SUBTRACT:
            self.skipTest("matrix_subtract_sync_f not available")

        from gnuradio import blocks

        # Create test matrices: A - B where A = [[5,6],[7,8]], B = [[1,2],[3,4]]
        # Expected: A - B = [[4,4],[4,4]]
        shape = [2, 2]

        matrix_a = [5.0, 6.0, 7.0, 8.0]
        matrix_b = [1.0, 2.0, 3.0, 4.0]
        expected_output = [4.0, 4.0, 4.0, 4.0]

        # Create blocks
        src_a = blocks.vector_source_f(matrix_a, False, 4)  # 2x2 = 4 elements
        src_b = blocks.vector_source_f(matrix_b, False, 4)
        matrix_subtract = matrix_subtract_sync_f(shape, 2)
        sink = blocks.vector_sink_f(4)  # 2x2 = 4 elements output

        # Connect flowgraph
        self.tb.connect(src_a, (matrix_subtract, 0))
        self.tb.connect(src_b, (matrix_subtract, 1))
        self.tb.connect(matrix_subtract, sink)

        # Run flowgraph
        self.tb.run()

        # Verify results
        result = sink.data()
        self.assertEqual(len(result), len(expected_output))
        for i, expected in enumerate(expected_output):
            self.assertAlmostEqual(result[i], expected, places=5)

    def test_zero_matrix_subtraction(self):
        # Test subtracting zero matrix (should leave matrix unchanged)
        if not HAVE_MATRIX_SUBTRACT:
            self.skipTest("matrix_subtract_sync_f not available")

        from gnuradio import blocks

        shape = [2, 2]

        # A - 0 = A
        matrix_a = [1.0, 2.0, 3.0, 4.0]
        matrix_zero = [0.0, 0.0, 0.0, 0.0]
        expected_output = matrix_a  # Should be unchanged

        src_a = blocks.vector_source_f(matrix_a, False, 4)
        src_zero = blocks.vector_source_f(matrix_zero, False, 4)
        matrix_subtract = matrix_subtract_sync_f(shape, 2)
        sink = blocks.vector_sink_f(4)

        self.tb.connect(src_a, (matrix_subtract, 0))
        self.tb.connect(src_zero, (matrix_subtract, 1))
        self.tb.connect(matrix_subtract, sink)

        self.tb.run()

        result = sink.data()
        for i, expected in enumerate(expected_output):
            self.assertAlmostEqual(result[i], expected, places=5)

    def test_self_subtraction(self):
        # Test subtracting matrix from itself (should be zero)
        if not HAVE_MATRIX_SUBTRACT:
            self.skipTest("matrix_subtract_sync_f not available")

        from gnuradio import blocks

        shape = [2, 2]

        # A - A = 0
        matrix_a = [5.0, 3.0, 7.0, 2.0]
        expected_output = [0.0, 0.0, 0.0, 0.0]

        src_a1 = blocks.vector_source_f(matrix_a, False, 4)
        src_a2 = blocks.vector_source_f(matrix_a, False, 4)
        matrix_subtract = matrix_subtract_sync_f(shape, 2)
        sink = blocks.vector_sink_f(4)

        self.tb.connect(src_a1, (matrix_subtract, 0))
        self.tb.connect(src_a2, (matrix_subtract, 1))
        self.tb.connect(matrix_subtract, sink)

        self.tb.run()

        result = sink.data()
        for i, expected in enumerate(expected_output):
            self.assertAlmostEqual(result[i], expected, places=5)

    def test_different_matrix_sizes(self):
        # Test different matrix sizes
        if not HAVE_MATRIX_SUBTRACT:
            self.skipTest("matrix_subtract_sync_f not available")

        from gnuradio import blocks

        # Test 3x2 matrices
        shape = [3, 2]

        # A = [[6,8],[4,2],[9,1]], B = [[1,3],[2,1],[4,0]]
        # A - B = [[5,5],[2,1],[5,1]]
        matrix_a = [6.0, 8.0, 4.0, 2.0, 9.0, 1.0]  # 6 elements
        matrix_b = [1.0, 3.0, 2.0, 1.0, 4.0, 0.0]
        expected_output = [5.0, 5.0, 2.0, 1.0, 5.0, 1.0]

        src_a = blocks.vector_source_f(matrix_a, False, 6)
        src_b = blocks.vector_source_f(matrix_b, False, 6)
        matrix_subtract = matrix_subtract_sync_f(shape, 2)
        sink = blocks.vector_sink_f(6)  # 3x2 = 6 elements

        self.tb.connect(src_a, (matrix_subtract, 0))
        self.tb.connect(src_b, (matrix_subtract, 1))
        self.tb.connect(matrix_subtract, sink)

        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), len(expected_output))
        for i, expected in enumerate(expected_output):
            self.assertAlmostEqual(result[i], expected, places=5)


if __name__ == "__main__":
    gr_unittest.run(qa_matrix_subtract)
