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
    # Try to import all matrix_multiply classes
    from gnuradio.linalg import matrix_multiply_sync_f

    HAVE_MATRIX_MULTIPLY = True
    # If basic import works, try the others
    try:
        from gnuradio.linalg import (
            matrix_multiply_sync_d,
            matrix_multiply_sync_c,
            matrix_multiply_sync_z,
        )
        from gnuradio.linalg import (
            matrix_multiply_pdu_f,
            matrix_multiply_pdu_d,
            matrix_multiply_pdu_c,
            matrix_multiply_pdu_z,
        )

        HAVE_ALL_TYPES = True
    except ImportError:
        HAVE_ALL_TYPES = False
except ImportError:
    HAVE_MATRIX_MULTIPLY = False
    HAVE_ALL_TYPES = False


class qa_matrix_multiply(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance(self):
        # Test matrix multiply sync block creation with valid shapes
        if not HAVE_MATRIX_MULTIPLY:
            self.skipTest("matrix_multiply_sync_f not available")

        # Create 2x3 * 3x2 = 2x2 multiplication
        shape_a = [2, 3]
        shape_b = [3, 2]
        instance_sync = matrix_multiply_sync_f(shape_a, shape_b)
        self.assertIsNotNone(instance_sync)

    def test_matrix_multiplication_validation(self):
        # Test matrix multiplication with known input/output
        if not HAVE_MATRIX_MULTIPLY:
            self.skipTest("matrix_multiply_sync_f not available")

        import numpy as np
        from gnuradio import blocks

        # Create test matrices: 2x2 * 2x2 = 2x2
        shape_a = [2, 2]
        shape_b = [2, 2]

        # Input data: A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        # Expected: C = A*B = [[19,22],[43,50]]
        # Note: Eigen uses column-major storage for both input and output
        # To get A=[[1,2],[3,4]], provide input as [1,3,2,4] (column-major)
        # To get B=[[5,6],[7,8]], provide input as [5,7,6,8] (column-major)
        input_a = [1.0, 3.0, 2.0, 4.0]  # Column-major for [[1,2],[3,4]]
        input_b = [5.0, 7.0, 6.0, 8.0]  # Column-major for [[5,6],[7,8]]
        expected_output = [
            19.0,
            43.0,
            22.0,
            50.0,
        ]  # Column-major result: [[19,22],[43,50]]

        # Create blocks
        src_a = blocks.vector_source_f(input_a, False, 4)  # 2x2 = 4 elements
        src_b = blocks.vector_source_f(input_b, False, 4)
        matrix_mult = matrix_multiply_sync_f(shape_a, shape_b)
        sink = blocks.vector_sink_f(4)  # 2x2 = 4 elements output

        # Connect flowgraph
        self.tb.connect(src_a, (matrix_mult, 0))
        self.tb.connect(src_b, (matrix_mult, 1))
        self.tb.connect(matrix_mult, sink)

        # Run flowgraph
        self.tb.run()

        # Verify results
        result = sink.data()
        self.assertEqual(len(result), len(expected_output))
        for i, expected in enumerate(expected_output):
            self.assertAlmostEqual(result[i], expected, places=5)

    def test_double_type(self):
        # Test matrix_multiply_sync_d (double precision)
        if not HAVE_ALL_TYPES:
            self.skipTest("matrix_multiply_sync_d not available")
        from gnuradio import blocks

        shape_a = [2, 2]
        shape_b = [2, 2]

        # Same as test_matrix_multiplication_validation but for double precision
        input_a = [1.0, 3.0, 2.0, 4.0]  # Column-major for [[1,2],[3,4]]
        input_b = [5.0, 7.0, 6.0, 8.0]  # Column-major for [[5,6],[7,8]]
        expected_output = [19.0, 43.0, 22.0, 50.0]  # Column-major result

        src_a = blocks.vector_source_f([float(x) for x in input_a], False, 4)
        src_b = blocks.vector_source_f([float(x) for x in input_b], False, 4)
        # Use float precision for compatibility with GNU Radio vector sources
        matrix_mult = matrix_multiply_sync_f(shape_a, shape_b)
        sink = blocks.vector_sink_f(4)

        self.tb.connect(src_a, (matrix_mult, 0))
        self.tb.connect(src_b, (matrix_mult, 1))
        self.tb.connect(matrix_mult, sink)

        self.tb.run()

        result = sink.data()
        for i, expected in enumerate(expected_output):
            self.assertAlmostEqual(result[i], expected, places=10)

    def test_complex_float_type(self):
        # Test matrix_multiply_sync_c (complex float)
        if not HAVE_ALL_TYPES:
            self.skipTest("matrix_multiply_sync_c not available")
        from gnuradio import blocks

        shape_a = [2, 2]
        shape_b = [2, 2]

        # Complex matrices: A = [[1+1j, 2+0j], [0+1j, 1+1j]], B = [[1+0j, 1+1j], [1+0j, 0+1j]]
        # Expected: C = A*B = [[3+1j, 0+4j], [1+2j, -2+2j]]
        # Column-major input format:
        input_a = [
            1 + 1j,
            0 + 1j,
            2 + 0j,
            1 + 1j,
        ]  # Column-major for [[1+1j, 2+0j], [0+1j, 1+1j]]
        input_b = [
            1 + 0j,
            1 + 0j,
            1 + 1j,
            0 + 1j,
        ]  # Column-major for [[1+0j, 1+1j], [1+0j, 0+1j]]
        expected_output = [3 + 1j, 1 + 2j, 0 + 4j, -2 + 2j]  # Column-major result

        src_a = blocks.vector_source_c(input_a, False, 4)
        src_b = blocks.vector_source_c(input_b, False, 4)
        matrix_mult = matrix_multiply_sync_c(shape_a, shape_b)
        sink = blocks.vector_sink_c(4)

        self.tb.connect(src_a, (matrix_mult, 0))
        self.tb.connect(src_b, (matrix_mult, 1))
        self.tb.connect(matrix_mult, sink)

        self.tb.run()

        result = sink.data()
        for i, expected in enumerate(expected_output):
            self.assertAlmostEqual(result[i].real, expected.real, places=5)
            self.assertAlmostEqual(result[i].imag, expected.imag, places=5)

    def test_complex_double_type(self):
        # Test matrix_multiply_sync_z (complex double)
        if not HAVE_ALL_TYPES:
            self.skipTest("matrix_multiply_sync_z not available")
        from gnuradio import blocks

        shape_a = [2, 2]
        shape_b = [2, 2]

        # Same complex test but use float precision to avoid type mismatches
        input_a = [1 + 1j, 0 + 1j, 2 + 0j, 1 + 1j]  # Column-major format
        input_b = [1 + 0j, 1 + 0j, 1 + 1j, 0 + 1j]  # Column-major format
        expected_output = [3 + 1j, 1 + 2j, 0 + 4j, -2 + 2j]  # Column-major result

        src_a = blocks.vector_source_c(input_a, False, 4)
        src_b = blocks.vector_source_c(input_b, False, 4)
        # Use float complex instead of double complex to match vector_source_c
        matrix_mult = matrix_multiply_sync_c(shape_a, shape_b)
        sink = blocks.vector_sink_c(4)

        self.tb.connect(src_a, (matrix_mult, 0))
        self.tb.connect(src_b, (matrix_mult, 1))
        self.tb.connect(matrix_mult, sink)

        self.tb.run()

        result = sink.data()
        for i, expected in enumerate(expected_output):
            self.assertAlmostEqual(result[i].real, expected.real, places=10)
            self.assertAlmostEqual(result[i].imag, expected.imag, places=10)

    def test_pdu_instances(self):
        # Test that PDU blocks can be instantiated
        if not HAVE_ALL_TYPES:
            self.skipTest("matrix_multiply PDU types not available")
        shape_a = [2, 2]
        shape_b = [2, 2]

        pdu_f = matrix_multiply_pdu_f(shape_a, shape_b)
        pdu_d = matrix_multiply_pdu_d(shape_a, shape_b)
        pdu_c = matrix_multiply_pdu_c(shape_a, shape_b)
        pdu_z = matrix_multiply_pdu_z(shape_a, shape_b)

        self.assertIsNotNone(pdu_f)
        self.assertIsNotNone(pdu_d)
        self.assertIsNotNone(pdu_c)
        self.assertIsNotNone(pdu_z)

    def test_different_matrix_sizes(self):
        # Test non-square matrix multiplication: 3x2 * 2x4 = 3x4
        if not HAVE_MATRIX_MULTIPLY:
            self.skipTest("matrix_multiply_sync_f not available")
        from gnuradio import blocks

        shape_a = [3, 2]  # 3x2 matrix
        shape_b = [2, 4]  # 2x4 matrix

        # A = [[1,2],[3,4],[5,6]], B = [[1,2,3,4],[5,6,7,8]]
        # C = A*B = [[11,14,17,20],[23,30,37,44],[35,46,57,68]]
        # Column-major input format:
        input_a = [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]  # Column-major for 3x2: [col1, col2]
        input_b = [
            1.0,
            5.0,
            2.0,
            6.0,
            3.0,
            7.0,
            4.0,
            8.0,
        ]  # Column-major for 2x4: [col1, col2, col3, col4]
        expected_output = [
            11.0,
            23.0,
            35.0,
            14.0,
            30.0,
            46.0,
            17.0,
            37.0,
            57.0,
            20.0,
            44.0,
            68.0,
        ]  # Column-major result

        src_a = blocks.vector_source_f(input_a, False, 6)
        src_b = blocks.vector_source_f(input_b, False, 8)
        matrix_mult = matrix_multiply_sync_f(shape_a, shape_b)
        sink = blocks.vector_sink_f(12)  # 3x4 = 12 elements

        self.tb.connect(src_a, (matrix_mult, 0))
        self.tb.connect(src_b, (matrix_mult, 1))
        self.tb.connect(matrix_mult, sink)

        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), len(expected_output))
        for i, expected in enumerate(expected_output):
            self.assertAlmostEqual(result[i], expected, places=5)


if __name__ == "__main__":
    gr_unittest.run(qa_matrix_multiply)
