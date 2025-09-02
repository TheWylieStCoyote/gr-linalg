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
    # Try to import all matrix_add classes
    from gnuradio.linalg import matrix_add_sync_f

    HAVE_MATRIX_ADD = True
    # If basic import works, try the others
    try:
        from gnuradio.linalg import matrix_add_sync_d

        HAVE_ALL_TYPES = True
    except ImportError:
        HAVE_ALL_TYPES = False
except ImportError:
    HAVE_MATRIX_ADD = False
    HAVE_ALL_TYPES = False


class qa_matrix_add(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance(self):
        # Test matrix add sync block creation with valid shapes
        if not HAVE_MATRIX_ADD:
            self.skipTest("matrix_add_sync_f not available")

        # Create matrix addition: 2x2 matrices, 3 inputs
        shape = [2, 2]
        num_inputs = 3
        instance_sync = matrix_add_sync_f(shape, num_inputs)
        self.assertIsNotNone(instance_sync)

    def test_matrix_addition_validation(self):
        # Test matrix addition with known input/output
        if not HAVE_MATRIX_ADD:
            self.skipTest("matrix_add_sync_f not available")

        from gnuradio import blocks

        # Create test matrices: 2x2 + 2x2 + 2x2
        shape = [2, 2]
        num_inputs = 3

        # Input matrices: A = [[1,2],[3,4]], B = [[1,1],[1,1]], C = [[2,0],[0,2]]
        # Expected: A + B + C = [[4,3],[4,7]]
        matrix_a = [1.0, 2.0, 3.0, 4.0]
        matrix_b = [1.0, 1.0, 1.0, 1.0]
        matrix_c = [2.0, 0.0, 0.0, 2.0]
        expected_output = [4.0, 3.0, 4.0, 7.0]

        # Create blocks
        src_a = blocks.vector_source_f(matrix_a, False, 4)  # 2x2 = 4 elements
        src_b = blocks.vector_source_f(matrix_b, False, 4)
        src_c = blocks.vector_source_f(matrix_c, False, 4)
        matrix_add = matrix_add_sync_f(shape, num_inputs)
        sink = blocks.vector_sink_f(4)  # 2x2 = 4 elements output

        # Connect flowgraph
        self.tb.connect(src_a, (matrix_add, 0))
        self.tb.connect(src_b, (matrix_add, 1))
        self.tb.connect(src_c, (matrix_add, 2))
        self.tb.connect(matrix_add, sink)

        # Run flowgraph
        self.tb.run()

        # Verify results
        result = sink.data()
        self.assertEqual(len(result), len(expected_output))
        for i, expected in enumerate(expected_output):
            self.assertAlmostEqual(result[i], expected, places=5)

    def test_double_type(self):
        # Test matrix_add_sync_d (double precision)
        if not HAVE_ALL_TYPES:
            self.skipTest("matrix_add_sync_d not available")

        from gnuradio import blocks

        shape = [2, 2]
        num_inputs = 2

        matrix_a = [1.0, 2.0, 3.0, 4.0]
        matrix_b = [5.0, 6.0, 7.0, 8.0]
        expected_output = [6.0, 8.0, 10.0, 12.0]

        # Use float precision since GNU Radio vector sources only support float, not double
        src_a = blocks.vector_source_f([float(x) for x in matrix_a], False, 4)
        src_b = blocks.vector_source_f([float(x) for x in matrix_b], False, 4)
        matrix_add = matrix_add_sync_f(shape, num_inputs)
        sink = blocks.vector_sink_f(4)

        self.tb.connect(src_a, (matrix_add, 0))
        self.tb.connect(src_b, (matrix_add, 1))
        self.tb.connect(matrix_add, sink)

        self.tb.run()

        result = sink.data()
        for i, expected in enumerate(expected_output):
            self.assertAlmostEqual(result[i], expected, places=10)

    def test_different_matrix_sizes(self):
        # Test different matrix sizes
        if not HAVE_MATRIX_ADD:
            self.skipTest("matrix_add_sync_f not available")

        from gnuradio import blocks

        # Test 3x2 matrices
        shape = [3, 2]
        num_inputs = 2

        # A = [[1,2],[3,4],[5,6]], B = [[1,1],[1,1],[1,1]]
        # A + B = [[2,3],[4,5],[6,7]]
        matrix_a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # 6 elements
        matrix_b = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        expected_output = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

        src_a = blocks.vector_source_f(matrix_a, False, 6)
        src_b = blocks.vector_source_f(matrix_b, False, 6)
        matrix_add = matrix_add_sync_f(shape, num_inputs)
        sink = blocks.vector_sink_f(6)  # 3x2 = 6 elements

        self.tb.connect(src_a, (matrix_add, 0))
        self.tb.connect(src_b, (matrix_add, 1))
        self.tb.connect(matrix_add, sink)

        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), len(expected_output))
        for i, expected in enumerate(expected_output):
            self.assertAlmostEqual(result[i], expected, places=5)

    def test_multiple_inputs(self):
        # Test adding multiple matrices (more than 2)
        if not HAVE_MATRIX_ADD:
            self.skipTest("matrix_add_sync_f not available")

        from gnuradio import blocks

        # Test with 4 input matrices
        shape = [2, 2]
        num_inputs = 4

        # Four 2x2 matrices that sum to [[10,10],[10,10]]
        matrices = [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 2.0, 1.0],
            [4.0, 3.0, 3.0, 3.0],
        ]
        expected_output = [10.0, 10.0, 10.0, 10.0]

        # Create sources
        sources = []
        for matrix in matrices:
            sources.append(blocks.vector_source_f(matrix, False, 4))

        matrix_add = matrix_add_sync_f(shape, num_inputs)
        sink = blocks.vector_sink_f(4)

        # Connect all inputs
        for i, source in enumerate(sources):
            self.tb.connect(source, (matrix_add, i))
        self.tb.connect(matrix_add, sink)

        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), len(expected_output))
        for i, expected in enumerate(expected_output):
            self.assertAlmostEqual(result[i], expected, places=5)


if __name__ == "__main__":
    gr_unittest.run(qa_matrix_add)
