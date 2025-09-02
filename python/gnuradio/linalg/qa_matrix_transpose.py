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
    # Try to import all matrix_transpose classes
    from gnuradio.linalg import matrix_transpose_sync_f

    HAVE_MATRIX_TRANSPOSE = True
    # If basic import works, try the others
    try:
        from gnuradio.linalg import matrix_transpose_sync_d

        HAVE_ALL_TYPES = True
    except ImportError:
        HAVE_ALL_TYPES = False
except ImportError:
    HAVE_MATRIX_TRANSPOSE = False
    HAVE_ALL_TYPES = False


class qa_matrix_transpose(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance(self):
        # Test matrix transpose block creation with valid shapes
        if not HAVE_MATRIX_TRANSPOSE:
            self.skipTest("matrix_transpose_sync_f not available")

        # Create matrix transpose: 3x2 matrix -> 2x3 matrix
        shape = [3, 2]
        instance = matrix_transpose_sync_f(shape)
        self.assertIsNotNone(instance)

    def test_matrix_transpose_validation(self):
        # Test matrix transpose with known input/output
        if not HAVE_MATRIX_TRANSPOSE:
            self.skipTest("matrix_transpose_sync_f not available")

        from gnuradio import blocks

        # Create test matrix: 2x3 matrix (Eigen column-major storage)
        shape = [2, 3]

        # Input matrix: [1,2,3,4,5,6] represents 2x3 matrix [[1,3,5],[2,4,6]] in column-major
        input_matrix = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        # Expected output: transpose gives 3x2 matrix [[1,2],[3,4],[5,6]] = [1,3,5,2,4,6] in column-major
        expected_output = [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]

        # Create blocks
        src = blocks.vector_source_f(input_matrix, False, 6)  # 2x3 = 6 elements
        transpose_block = matrix_transpose_sync_f(shape)
        sink = blocks.vector_sink_f(6)  # 3x2 = 6 elements output

        # Connect flowgraph
        self.tb.connect(src, transpose_block)
        self.tb.connect(transpose_block, sink)

        # Run flowgraph
        self.tb.run()

        # Verify results
        result = sink.data()
        self.assertEqual(len(result), len(expected_output))
        for i, expected in enumerate(expected_output):
            self.assertAlmostEqual(result[i], expected, places=5)

    def test_square_matrix_transpose(self):
        # Test transpose of square matrix
        if not HAVE_MATRIX_TRANSPOSE:
            self.skipTest("matrix_transpose_sync_f not available")

        from gnuradio import blocks

        # Square matrix: 2x2 [[1,2],[3,4]] -> [[1,3],[2,4]]
        shape = [2, 2]

        input_matrix = [1.0, 2.0, 3.0, 4.0]
        expected_output = [1.0, 3.0, 2.0, 4.0]

        src = blocks.vector_source_f(input_matrix, False, 4)
        transpose_block = matrix_transpose_sync_f(shape)
        sink = blocks.vector_sink_f(4)

        self.tb.connect(src, transpose_block)
        self.tb.connect(transpose_block, sink)

        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), len(expected_output))
        for i, expected in enumerate(expected_output):
            self.assertAlmostEqual(result[i], expected, places=5)

    def test_double_type(self):
        # Test matrix_transpose_sync_d (double precision)
        if not HAVE_ALL_TYPES:
            self.skipTest("matrix_transpose_sync_d not available")

        from gnuradio import blocks

        shape = [2, 3]

        input_matrix = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        expected_output = [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]

        # Note: GNU Radio only has vector_source_f, so we use float precision throughout
        src = blocks.vector_source_f([float(x) for x in input_matrix], False, 6)
        transpose_block = matrix_transpose_sync_f(shape)  # Use float precision
        sink = blocks.vector_sink_f(6)

        self.tb.connect(src, transpose_block)
        self.tb.connect(transpose_block, sink)

        self.tb.run()

        result = sink.data()
        for i, expected in enumerate(expected_output):
            self.assertAlmostEqual(result[i], expected, places=10)

    def test_identity_matrix_transpose(self):
        # Test transpose of identity matrix (should be unchanged)
        if not HAVE_MATRIX_TRANSPOSE:
            self.skipTest("matrix_transpose_sync_f not available")

        from gnuradio import blocks

        # 3x3 identity matrix
        shape = [3, 3]

        # Identity matrix: [[1,0,0],[0,1,0],[0,0,1]]
        identity_matrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        expected_output = identity_matrix  # Transpose of identity is itself

        src = blocks.vector_source_f(identity_matrix, False, 9)
        transpose_block = matrix_transpose_sync_f(shape)
        sink = blocks.vector_sink_f(9)

        self.tb.connect(src, transpose_block)
        self.tb.connect(transpose_block, sink)

        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), len(expected_output))
        for i, expected in enumerate(expected_output):
            self.assertAlmostEqual(result[i], expected, places=5)

    def test_rectangular_matrices(self):
        # Test transpose with different rectangular matrices
        if not HAVE_MATRIX_TRANSPOSE:
            self.skipTest("matrix_transpose_sync_f not available")

        from gnuradio import blocks

        # Test 1x4 matrix -> 4x1 matrix
        shape = [1, 4]

        # Row vector: [1, 2, 3, 4] -> Column vector
        input_matrix = [1.0, 2.0, 3.0, 4.0]
        expected_output = [1.0, 2.0, 3.0, 4.0]  # Same data, different shape

        src = blocks.vector_source_f(input_matrix, False, 4)
        transpose_block = matrix_transpose_sync_f(shape)
        sink = blocks.vector_sink_f(4)

        self.tb.connect(src, transpose_block)
        self.tb.connect(transpose_block, sink)

        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), len(expected_output))
        for i, expected in enumerate(expected_output):
            self.assertAlmostEqual(result[i], expected, places=5)


if __name__ == "__main__":
    gr_unittest.run(qa_matrix_transpose)
