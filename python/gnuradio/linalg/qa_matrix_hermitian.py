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
    # Try to import all matrix_hermitian classes from direct C++ bindings
    import sys
    import os

    dirname = os.path.dirname(os.path.abspath(__file__))
    test_modules_path = os.path.join(
        dirname, "../../build/test_modules/gnuradio/linalg"
    )
    sys.path.insert(0, test_modules_path)

    import linalg_python

    matrix_hermitian_sync_f = linalg_python.matrix_hermitian_sync_f
    matrix_hermitian_sync_d = linalg_python.matrix_hermitian_sync_d
    matrix_hermitian_sync_cf = linalg_python.matrix_hermitian_sync_cf
    matrix_hermitian_sync_cd = linalg_python.matrix_hermitian_sync_cd

    HAVE_MATRIX_HERMITIAN = True
    HAVE_ALL_TYPES = True
except ImportError as e:
    print(f"Import error: {e}")
    try:
        # Fallback to try gnuradio.linalg module
        from gnuradio.linalg import matrix_hermitian_sync_f

        HAVE_MATRIX_HERMITIAN = True
        HAVE_ALL_TYPES = False
    except ImportError:
        HAVE_MATRIX_HERMITIAN = False
        HAVE_ALL_TYPES = False


class qa_matrix_hermitian(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance(self):
        # Test matrix hermitian block creation with valid shapes
        if not HAVE_MATRIX_HERMITIAN:
            self.skipTest("matrix_hermitian_sync_f not available")

        # Create matrix hermitian: 3x2 matrix -> 2x3 matrix
        shape = [3, 2]
        instance = matrix_hermitian_sync_f(shape)
        self.assertIsNotNone(instance)

    def test_real_matrix_hermitian(self):
        # Test hermitian (conjugate transpose) on real matrix - should be same as transpose
        if not HAVE_MATRIX_HERMITIAN:
            self.skipTest("matrix_hermitian_sync_f not available")

        from gnuradio import blocks

        # Create test matrix: 2x3 matrix (Eigen column-major storage)
        shape = [2, 3]

        # Input matrix: [1,2,3,4,5,6] represents 2x3 matrix [[1,3,5],[2,4,6]] in column-major
        input_matrix = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        # For real matrices, hermitian = transpose
        # Expected output: transpose gives 3x2 matrix [[1,2],[3,4],[5,6]] = [1,3,5,2,4,6] in column-major
        expected_output = [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]

        # Create blocks
        src = blocks.vector_source_f(input_matrix, False, 6)  # 2x3 = 6 elements
        hermitian_block = matrix_hermitian_sync_f(shape)
        sink = blocks.vector_sink_f(6)  # 3x2 = 6 elements output

        # Connect flowgraph
        self.tb.connect(src, hermitian_block)
        self.tb.connect(hermitian_block, sink)

        # Run flowgraph
        self.tb.run()

        # Verify results
        result = sink.data()
        self.assertEqual(len(result), len(expected_output))
        for i, expected in enumerate(expected_output):
            self.assertAlmostEqual(result[i], expected, places=5)

    def test_square_real_matrix_hermitian(self):
        # Test hermitian of square real matrix
        if not HAVE_MATRIX_HERMITIAN:
            self.skipTest("matrix_hermitian_sync_f not available")

        from gnuradio import blocks

        # Square matrix: 2x2 [[1,2],[3,4]] -> [[1,3],[2,4]]
        shape = [2, 2]

        input_matrix = [1.0, 2.0, 3.0, 4.0]
        expected_output = [1.0, 3.0, 2.0, 4.0]

        src = blocks.vector_source_f(input_matrix, False, 4)
        hermitian_block = matrix_hermitian_sync_f(shape)
        sink = blocks.vector_sink_f(4)

        self.tb.connect(src, hermitian_block)
        self.tb.connect(hermitian_block, sink)

        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), len(expected_output))
        for i, expected in enumerate(expected_output):
            self.assertAlmostEqual(result[i], expected, places=5)

    def test_complex_matrix_hermitian(self):
        # Test hermitian (conjugate transpose) on complex matrix
        if not HAVE_ALL_TYPES:
            self.skipTest("matrix_hermitian_sync_cf not available")

        from gnuradio import blocks

        # Create test matrix: 2x2 complex matrix
        shape = [2, 2]

        # Input complex matrix: [[1+2j, 3+4j], [5+6j, 7+8j]]
        # In column-major: [1+2j, 5+6j, 3+4j, 7+8j]
        # Real and imag parts interleaved: [1,2,5,6,3,4,7,8]
        input_matrix = [complex(1, 2), complex(5, 6), complex(3, 4), complex(7, 8)]

        # Hermitian (conjugate transpose): [[1-2j, 5-6j], [3-4j, 7-8j]]
        # In column-major: [1-2j, 3-4j, 5-6j, 7-8j]
        expected_output = [
            complex(1, -2),
            complex(3, -4),
            complex(5, -6),
            complex(7, -8),
        ]

        # Create blocks (using complex floats)
        src = blocks.vector_source_c(input_matrix, False, 4)
        hermitian_block = matrix_hermitian_sync_cf(shape)
        sink = blocks.vector_sink_c(4)

        # Connect flowgraph
        self.tb.connect(src, hermitian_block)
        self.tb.connect(hermitian_block, sink)

        # Run flowgraph
        self.tb.run()

        # Verify results
        result = sink.data()
        self.assertEqual(len(result), 4)

        for i, expected in enumerate(expected_output):
            self.assertAlmostEqual(result[i].real, expected.real, places=5)
            self.assertAlmostEqual(result[i].imag, expected.imag, places=5)

    def test_identity_matrix_hermitian(self):
        # Test hermitian of identity matrix (should be unchanged)
        if not HAVE_MATRIX_HERMITIAN:
            self.skipTest("matrix_hermitian_sync_f not available")

        from gnuradio import blocks

        # 3x3 identity matrix
        shape = [3, 3]

        # Identity matrix: [[1,0,0],[0,1,0],[0,0,1]]
        identity_matrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        expected_output = identity_matrix  # Hermitian of identity is itself

        src = blocks.vector_source_f(identity_matrix, False, 9)
        hermitian_block = matrix_hermitian_sync_f(shape)
        sink = blocks.vector_sink_f(9)

        self.tb.connect(src, hermitian_block)
        self.tb.connect(hermitian_block, sink)

        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), len(expected_output))
        for i, expected in enumerate(expected_output):
            self.assertAlmostEqual(result[i], expected, places=5)

    def test_hermitian_symmetric_real_matrix(self):
        # Test hermitian of symmetric real matrix (should be unchanged)
        if not HAVE_MATRIX_HERMITIAN:
            self.skipTest("matrix_hermitian_sync_f not available")

        from gnuradio import blocks

        # 3x3 symmetric matrix
        shape = [3, 3]

        # Symmetric matrix: [[1,2,3],[2,4,5],[3,5,6]]
        # In column-major: [1,2,3,2,4,5,3,5,6]
        symmetric_matrix = [1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0]
        expected_output = (
            symmetric_matrix  # Hermitian of symmetric real matrix is itself
        )

        src = blocks.vector_source_f(symmetric_matrix, False, 9)
        hermitian_block = matrix_hermitian_sync_f(shape)
        sink = blocks.vector_sink_f(9)

        self.tb.connect(src, hermitian_block)
        self.tb.connect(hermitian_block, sink)

        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), len(expected_output))
        for i, expected in enumerate(expected_output):
            self.assertAlmostEqual(result[i], expected, places=5)

    def test_rectangular_matrices(self):
        # Test hermitian with different rectangular matrices
        if not HAVE_MATRIX_HERMITIAN:
            self.skipTest("matrix_hermitian_sync_f not available")

        from gnuradio import blocks

        # Test 1x4 matrix -> 4x1 matrix
        shape = [1, 4]

        # Row vector: [1, 2, 3, 4] -> Column vector
        input_matrix = [1.0, 2.0, 3.0, 4.0]
        expected_output = [1.0, 2.0, 3.0, 4.0]  # Same data, different shape

        src = blocks.vector_source_f(input_matrix, False, 4)
        hermitian_block = matrix_hermitian_sync_f(shape)
        sink = blocks.vector_sink_f(4)

        self.tb.connect(src, hermitian_block)
        self.tb.connect(hermitian_block, sink)

        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), len(expected_output))
        for i, expected in enumerate(expected_output):
            self.assertAlmostEqual(result[i], expected, places=5)

    def test_complex_hermitian_property(self):
        # Test that (A^H)^H = A for complex matrices
        if not HAVE_ALL_TYPES:
            self.skipTest("matrix_hermitian_sync_cf not available")

        from gnuradio import blocks

        shape = [2, 2]

        # Input complex matrix
        input_complex = [complex(1, 2), complex(5, 6), complex(3, 4), complex(7, 8)]

        # Create blocks for double hermitian
        src = blocks.vector_source_c(input_complex, False, 4)
        hermitian_block1 = matrix_hermitian_sync_cf(shape)
        hermitian_block2 = matrix_hermitian_sync_cf([2, 2])  # Transposed shape
        sink = blocks.vector_sink_c(4)

        # Connect flowgraph: A -> A^H -> (A^H)^H = A
        self.tb.connect(src, hermitian_block1)
        self.tb.connect(hermitian_block1, hermitian_block2)
        self.tb.connect(hermitian_block2, sink)

        self.tb.run()

        # Result should be the original matrix
        result = sink.data()
        self.assertEqual(len(result), len(input_complex))

        for i, expected in enumerate(input_complex):
            self.assertAlmostEqual(result[i].real, expected.real, places=5)
            self.assertAlmostEqual(result[i].imag, expected.imag, places=5)


if __name__ == "__main__":
    gr_unittest.run(qa_matrix_hermitian)
