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
    # Try to import vector_Kronecker classes (matrix_kronecker_product alias)
    from gnuradio.linalg import matrix_kronecker_product_sync_f

    HAVE_VECTOR_KRONECKER = True
    # If basic import works, try the others
    try:
        from gnuradio.linalg import matrix_kronecker_product_sync_d

        HAVE_ALL_TYPES = True
    except ImportError:
        HAVE_ALL_TYPES = False
except ImportError:
    HAVE_VECTOR_KRONECKER = False
    HAVE_ALL_TYPES = False


class qa_vector_Kronecker(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance(self):
        # Test vector_Kronecker (matrix_kronecker_product) block creation with valid parameters
        if not HAVE_VECTOR_KRONECKER:
            self.skipTest("matrix_kronecker_product_sync_f not available")

        # Create Kronecker product: 2x2 ⊗ 2x2 -> 4x4
        shape_a = [2, 2]
        shape_b = [2, 2]
        instance = matrix_kronecker_product_sync_f.make(shape_a, shape_b)
        self.assertIsNotNone(instance)

    def test_kronecker_product_2x2_identity(self):
        # Test Kronecker product with identity matrices
        if not HAVE_VECTOR_KRONECKER:
            self.skipTest("matrix_kronecker_product_sync_f not available")

        from gnuradio import blocks

        # Identity ⊗ Identity = Identity (scaled)
        # I₂ = [[1,0],[0,1]], I₂ ⊗ I₂ = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        identity_2x2 = [1.0, 0.0, 0.0, 1.0]  # Column-major
        expected_result = [
            1.0,
            0.0,
            0.0,
            0.0,  # Column 1
            0.0,
            1.0,
            0.0,
            0.0,  # Column 2
            0.0,
            0.0,
            1.0,
            0.0,  # Column 3
            0.0,
            0.0,
            0.0,
            1.0,
        ]  # Column 4

        src_a = blocks.vector_source_f(identity_2x2, False, 4)
        src_b = blocks.vector_source_f(identity_2x2, False, 4)
        kronecker_block = matrix_kronecker_product_sync_f.make([2, 2], [2, 2])
        sink = blocks.vector_sink_f(16)  # 4x4 = 16 elements

        self.tb.connect(src_a, (kronecker_block, 0))
        self.tb.connect(src_b, (kronecker_block, 1))
        self.tb.connect(kronecker_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 16)
        for i, expected in enumerate(expected_result):
            self.assertAlmostEqual(result[i], expected, places=6)

    def test_kronecker_product_basic_2x2(self):
        # Test basic 2x2 Kronecker product with known result
        if not HAVE_VECTOR_KRONECKER:
            self.skipTest("matrix_kronecker_product_sync_f not available")

        from gnuradio import blocks

        # A = [[1,2],[3,4]], B = [[0,5],[6,7]]
        # A ⊗ B = [[A₁₁*B, A₁₂*B], [A₂₁*B, A₂₂*B]]
        matrix_a = [1.0, 3.0, 2.0, 4.0]  # Column-major: [[1,2],[3,4]]
        matrix_b = [0.0, 6.0, 5.0, 7.0]  # Column-major: [[0,5],[6,7]]

        # Expected result: 4x4 matrix
        # [[0,5,0,10], [6,7,12,14], [0,15,0,20], [18,21,24,28]]
        expected_result = [
            0.0,
            6.0,
            0.0,
            18.0,  # Column 1: [1*0, 1*6, 3*0, 3*6]
            5.0,
            7.0,
            15.0,
            21.0,  # Column 2: [1*5, 1*7, 3*5, 3*7]
            0.0,
            12.0,
            0.0,
            24.0,  # Column 3: [2*0, 2*6, 4*0, 4*6]
            10.0,
            14.0,
            20.0,
            28.0,
        ]  # Column 4: [2*5, 2*7, 4*5, 4*7]

        src_a = blocks.vector_source_f(matrix_a, False, 4)
        src_b = blocks.vector_source_f(matrix_b, False, 4)
        kronecker_block = matrix_kronecker_product_sync_f.make([2, 2], [2, 2])
        sink = blocks.vector_sink_f(16)

        self.tb.connect(src_a, (kronecker_block, 0))
        self.tb.connect(src_b, (kronecker_block, 1))
        self.tb.connect(kronecker_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 16)
        for i, expected in enumerate(expected_result):
            self.assertAlmostEqual(result[i], expected, places=6)

    def test_kronecker_product_rectangular_matrices(self):
        # Test Kronecker product with rectangular matrices
        if not HAVE_VECTOR_KRONECKER:
            self.skipTest("matrix_kronecker_product_sync_f not available")

        from gnuradio import blocks

        # A = [[1,2]] (1x2), B = [[3],[4]] (2x1)
        # A ⊗ B should be 2x2: [[3,6],[4,8]]
        matrix_a = [1.0, 2.0]  # 1x2 matrix in column-major
        matrix_b = [3.0, 4.0]  # 2x1 matrix in column-major
        expected_result = [3.0, 4.0, 6.0, 8.0]  # [[3,6],[4,8]] in column-major

        src_a = blocks.vector_source_f(matrix_a, False, 2)
        src_b = blocks.vector_source_f(matrix_b, False, 2)
        kronecker_block = matrix_kronecker_product_sync_f.make([1, 2], [2, 1])
        sink = blocks.vector_sink_f(4)  # 2x2 = 4 elements

        self.tb.connect(src_a, (kronecker_block, 0))
        self.tb.connect(src_b, (kronecker_block, 1))
        self.tb.connect(kronecker_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 4)
        for i, expected in enumerate(expected_result):
            self.assertAlmostEqual(result[i], expected, places=6)

    def test_kronecker_product_dimension_property(self):
        # Test that dimensions are correct: (m×n) ⊗ (p×q) = (mp×nq)
        if not HAVE_VECTOR_KRONECKER:
            self.skipTest("matrix_kronecker_product_sync_f not available")

        from gnuradio import blocks

        # A = 3x2 matrix, B = 2x3 matrix -> result should be 6x6
        matrix_a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # 3x2 in column-major
        matrix_b = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # 2x3 ones matrix

        src_a = blocks.vector_source_f(matrix_a, False, 6)
        src_b = blocks.vector_source_f(matrix_b, False, 6)
        kronecker_block = matrix_kronecker_product_sync_f.make([3, 2], [2, 3])
        sink = blocks.vector_sink_f(36)  # 6x6 = 36 elements

        self.tb.connect(src_a, (kronecker_block, 0))
        self.tb.connect(src_b, (kronecker_block, 1))
        self.tb.connect(kronecker_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 36)  # Verify correct dimensions

        # Convert result to matrix for structure verification
        result_matrix = np.array(result).reshape(6, 6, order="F")

        # Verify the block structure of Kronecker product
        # Each element A[i,j] should multiply the entire matrix B
        # Top-left 2x3 block should be A[0,0] * B = 1 * B = B
        top_left_block = result_matrix[0:2, 0:3]
        expected_block = np.ones((2, 3))
        self.assertTrue(np.allclose(top_left_block, expected_block, atol=1e-6))

    def test_kronecker_product_zero_matrix(self):
        # Test Kronecker product with zero matrix: A ⊗ 0 = 0
        if not HAVE_VECTOR_KRONECKER:
            self.skipTest("matrix_kronecker_product_sync_f not available")

        from gnuradio import blocks

        # A = [[1,2],[3,4]], 0 = [[0,0],[0,0]]
        matrix_a = [1.0, 3.0, 2.0, 4.0]
        zero_matrix = [0.0, 0.0, 0.0, 0.0]
        expected_result = [0.0] * 16  # All zeros

        src_a = blocks.vector_source_f(matrix_a, False, 4)
        src_b = blocks.vector_source_f(zero_matrix, False, 4)
        kronecker_block = matrix_kronecker_product_sync_f.make([2, 2], [2, 2])
        sink = blocks.vector_sink_f(16)

        self.tb.connect(src_a, (kronecker_block, 0))
        self.tb.connect(src_b, (kronecker_block, 1))
        self.tb.connect(kronecker_block, sink)
        self.tb.run()

        result = sink.data()
        for i, expected in enumerate(expected_result):
            self.assertAlmostEqual(result[i], expected, places=6)

    def test_kronecker_product_distributivity(self):
        # Test distributive property: A ⊗ (B + C) = (A ⊗ B) + (A ⊗ C)
        # We can't directly test this in flowgraph, but verify structure
        if not HAVE_VECTOR_KRONECKER:
            self.skipTest("matrix_kronecker_product_sync_f not available")

        from gnuradio import blocks

        # Test with simple matrices to verify consistency
        matrix_a = [2.0, 0.0, 0.0, 2.0]  # 2*I₂
        matrix_b = [1.0, 1.0, 1.0, 1.0]  # All ones matrix

        # 2*I₂ ⊗ ones₂ should give predictable pattern
        expected_result = [
            2.0,
            2.0,
            0.0,
            0.0,  # Column 1
            2.0,
            2.0,
            0.0,
            0.0,  # Column 2
            0.0,
            0.0,
            2.0,
            2.0,  # Column 3
            0.0,
            0.0,
            2.0,
            2.0,
        ]  # Column 4

        src_a = blocks.vector_source_f(matrix_a, False, 4)
        src_b = blocks.vector_source_f(matrix_b, False, 4)
        kronecker_block = matrix_kronecker_product_sync_f.make([2, 2], [2, 2])
        sink = blocks.vector_sink_f(16)

        self.tb.connect(src_a, (kronecker_block, 0))
        self.tb.connect(src_b, (kronecker_block, 1))
        self.tb.connect(kronecker_block, sink)
        self.tb.run()

        result = sink.data()
        for i, expected in enumerate(expected_result):
            self.assertAlmostEqual(result[i], expected, places=6)

    def test_kronecker_product_scalar_multiplication(self):
        # Test scalar multiplication property: (cA) ⊗ B = c(A ⊗ B) = A ⊗ (cB)
        if not HAVE_VECTOR_KRONECKER:
            self.skipTest("matrix_kronecker_product_sync_f not available")

        from gnuradio import blocks

        # A = [[1,0],[0,1]], B = [[2,0],[0,2]] = 2*I₂
        matrix_a = [1.0, 0.0, 0.0, 1.0]  # Identity
        matrix_b = [2.0, 0.0, 0.0, 2.0]  # 2*Identity

        # I₂ ⊗ (2*I₂) should give 2*I₄
        expected_result = [
            2.0,
            0.0,
            0.0,
            0.0,  # 2*I₄
            0.0,
            2.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0,
        ]

        src_a = blocks.vector_source_f(matrix_a, False, 4)
        src_b = blocks.vector_source_f(matrix_b, False, 4)
        kronecker_block = matrix_kronecker_product_sync_f.make([2, 2], [2, 2])
        sink = blocks.vector_sink_f(16)

        self.tb.connect(src_a, (kronecker_block, 0))
        self.tb.connect(src_b, (kronecker_block, 1))
        self.tb.connect(kronecker_block, sink)
        self.tb.run()

        result = sink.data()
        for i, expected in enumerate(expected_result):
            self.assertAlmostEqual(result[i], expected, places=6)

    def test_kronecker_product_asymmetry(self):
        # Test that A ⊗ B ≠ B ⊗ A in general (non-commutative)
        if not HAVE_VECTOR_KRONECKER:
            self.skipTest("matrix_kronecker_product_sync_f not available")

        from gnuradio import blocks

        matrix_a = [1.0, 0.0, 2.0, 0.0]  # [[1,2],[0,0]]
        matrix_b = [0.0, 1.0, 0.0, 3.0]  # [[0,0],[1,3]]

        # Test A ⊗ B
        src_a1 = blocks.vector_source_f(matrix_a, False, 4)
        src_b1 = blocks.vector_source_f(matrix_b, False, 4)
        kronecker_block1 = matrix_kronecker_product_sync_f.make([2, 2], [2, 2])
        sink1 = blocks.vector_sink_f(16)

        self.tb.connect(src_a1, (kronecker_block1, 0))
        self.tb.connect(src_b1, (kronecker_block1, 1))
        self.tb.connect(kronecker_block1, sink1)
        self.tb.run()

        result_ab = list(sink1.data())

        # Reset flowgraph for B ⊗ A
        self.tb = gr.top_block()

        src_a2 = blocks.vector_source_f(matrix_b, False, 4)  # Swap A and B
        src_b2 = blocks.vector_source_f(matrix_a, False, 4)
        kronecker_block2 = matrix_kronecker_product_sync_f.make([2, 2], [2, 2])
        sink2 = blocks.vector_sink_f(16)

        self.tb.connect(src_a2, (kronecker_block2, 0))
        self.tb.connect(src_b2, (kronecker_block2, 1))
        self.tb.connect(kronecker_block2, sink2)
        self.tb.run()

        result_ba = list(sink2.data())

        # A ⊗ B should not equal B ⊗ A for these matrices
        self.assertFalse(np.allclose(result_ab, result_ba, atol=1e-6))

    def test_kronecker_product_vector_case(self):
        # Test Kronecker product with vectors (1D case)
        if not HAVE_VECTOR_KRONECKER:
            self.skipTest("matrix_kronecker_product_sync_f not available")

        from gnuradio import blocks

        # a = [1, 2] (2x1), b = [3, 4, 5] (3x1)
        # a ⊗ b = [3, 4, 5, 6, 8, 10] (6x1)
        vector_a = [1.0, 2.0]  # 2x1
        vector_b = [3.0, 4.0, 5.0]  # 3x1
        expected_result = [3.0, 4.0, 5.0, 6.0, 8.0, 10.0]  # 6x1

        src_a = blocks.vector_source_f(vector_a, False, 2)
        src_b = blocks.vector_source_f(vector_b, False, 3)
        kronecker_block = matrix_kronecker_product_sync_f.make([2, 1], [3, 1])
        sink = blocks.vector_sink_f(6)

        self.tb.connect(src_a, (kronecker_block, 0))
        self.tb.connect(src_b, (kronecker_block, 1))
        self.tb.connect(kronecker_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 6)
        for i, expected in enumerate(expected_result):
            self.assertAlmostEqual(result[i], expected, places=6)

    def test_kronecker_product_numerical_accuracy(self):
        # Test numerical accuracy with precise values
        if not HAVE_VECTOR_KRONECKER:
            self.skipTest("matrix_kronecker_product_sync_f not available")

        from gnuradio import blocks

        # Use rational values that have exact floating point representation
        matrix_a = [0.5, 0.0, 0.0, 0.25]  # [[0.5, 0], [0, 0.25]]
        matrix_b = [2.0, 4.0, 8.0, 16.0]  # [[2, 8], [4, 16]]

        # Expected exact result
        expected_result = [
            1.0,
            2.0,
            0.0,
            0.0,  # 0.5*[2,4] and 0*[2,4]
            4.0,
            8.0,
            0.0,
            0.0,  # 0.5*[8,16] and 0*[8,16]
            0.0,
            0.0,
            0.5,
            1.0,  # 0*[2,4] and 0.25*[2,4]
            0.0,
            0.0,
            2.0,
            4.0,
        ]  # 0*[8,16] and 0.25*[8,16]

        src_a = blocks.vector_source_f(matrix_a, False, 4)
        src_b = blocks.vector_source_f(matrix_b, False, 4)
        kronecker_block = matrix_kronecker_product_sync_f.make([2, 2], [2, 2])
        sink = blocks.vector_sink_f(16)

        self.tb.connect(src_a, (kronecker_block, 0))
        self.tb.connect(src_b, (kronecker_block, 1))
        self.tb.connect(kronecker_block, sink)
        self.tb.run()

        result = sink.data()
        for i, expected in enumerate(expected_result):
            self.assertAlmostEqual(result[i], expected, places=6)

    def test_kronecker_product_double_precision(self):
        # Test double precision Kronecker product
        if not HAVE_ALL_TYPES:
            self.skipTest("matrix_kronecker_product_sync_d not available")

        # GNU Radio doesn't provide double precision vector sources and sinks
        self.skipTest(
            "GNU Radio doesn't provide vector_source_d/vector_sink_d for double precision testing"
        )

        from gnuradio import blocks

        # Simple test with double precision
        matrix_a = [1.0, 0.0, 0.0, 2.0]  # [[1,0],[0,2]]
        matrix_b = [3.0, 0.0, 0.0, 4.0]  # [[3,0],[0,4]]
        expected_result = [
            3.0,
            0.0,
            0.0,
            0.0,  # [[3,0,0,0],
            0.0,
            4.0,
            0.0,
            0.0,  #  [0,4,0,0],
            0.0,
            0.0,
            6.0,
            0.0,  #  [0,0,6,0],
            0.0,
            0.0,
            0.0,
            8.0,
        ]  #  [0,0,0,8]]

        src_a = blocks.vector_source_d(matrix_a, False, 4)
        src_b = blocks.vector_source_d(matrix_b, False, 4)
        kronecker_block = matrix_kronecker_product_sync_d.make([2, 2], [2, 2])
        sink = blocks.vector_sink_d(16)

        self.tb.connect(src_a, (kronecker_block, 0))
        self.tb.connect(src_b, (kronecker_block, 1))
        self.tb.connect(kronecker_block, sink)
        self.tb.run()

        result = sink.data()
        for i, expected in enumerate(expected_result):
            self.assertAlmostEqual(result[i], expected, places=12)  # Higher precision

    def test_kronecker_product_structure_verification(self):
        # Test structural properties of Kronecker product result
        if not HAVE_VECTOR_KRONECKER:
            self.skipTest("matrix_kronecker_product_sync_f not available")

        from gnuradio import blocks

        # A = [[1,2,3],[4,5,6]] (2x3), B = [[7,8],[9,10]] (2x2)
        # Result should be 4x6
        matrix_a = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]  # 2x3 in column-major
        matrix_b = [7.0, 9.0, 8.0, 10.0]  # 2x2 in column-major

        src_a = blocks.vector_source_f(matrix_a, False, 6)
        src_b = blocks.vector_source_f(matrix_b, False, 4)
        kronecker_block = matrix_kronecker_product_sync_f.make([2, 3], [2, 2])
        sink = blocks.vector_sink_f(24)  # 4x6 = 24 elements

        self.tb.connect(src_a, (kronecker_block, 0))
        self.tb.connect(src_b, (kronecker_block, 1))
        self.tb.connect(kronecker_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 24)

        # Convert to matrix and verify block structure
        result_matrix = np.array(result).reshape(4, 6, order="F")

        # The top-left 2x2 block should be A[0,0] * B = 1 * B
        top_left = result_matrix[0:2, 0:2]
        expected_top_left = np.array([[7.0, 8.0], [9.0, 10.0]])
        self.assertTrue(np.allclose(top_left, expected_top_left, atol=1e-6))

        # The bottom-right 2x2 block should be A[1,2] * B = 6 * B
        bottom_right = result_matrix[2:4, 4:6]
        expected_bottom_right = 6.0 * np.array([[7.0, 8.0], [9.0, 10.0]])
        self.assertTrue(np.allclose(bottom_right, expected_bottom_right, atol=1e-6))


if __name__ == "__main__":
    gr_unittest.run(qa_vector_Kronecker)
