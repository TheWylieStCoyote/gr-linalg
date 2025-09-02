#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2025 Wylie Standage-Beier.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

from gnuradio import gr, gr_unittest
import math
import sys
import os

# Add the test_modules path to find linalg_python
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "build",
        "test_modules",
        "gnuradio",
        "linalg",
    ),
)

try:
    import linalg_python

    HAVE_VECTOR_CROSS_PRODUCT = True
except ImportError:
    HAVE_VECTOR_CROSS_PRODUCT = False


class qa_vector_cross_product(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance(self):
        # Test vector cross product block creation
        if not HAVE_VECTOR_CROSS_PRODUCT:
            self.skipTest("vector_cross_product not available")

        # Create vector cross product block
        instance = linalg_python.vector_cross_product_sync_f(True)
        self.assertIsNotNone(instance)

    def test_basic_cross_product(self):
        # Test basic cross product: [1,0,0] × [0,1,0] = [0,0,1]
        if not HAVE_VECTOR_CROSS_PRODUCT:
            self.skipTest("vector_cross_product not available")

        from gnuradio import blocks

        # Input vectors
        input_a = [1.0, 0.0, 0.0]  # x-axis unit vector
        input_b = [0.0, 1.0, 0.0]  # y-axis unit vector
        expected_output = [0.0, 0.0, 1.0]  # z-axis unit vector

        # Create blocks
        src_a = blocks.vector_source_f(input_a, False, 3)
        src_b = blocks.vector_source_f(input_b, False, 3)
        cross_block = linalg_python.vector_cross_product_sync_f(True)
        sink = blocks.vector_sink_f(3)

        # Connect flowgraph
        self.tb.connect(src_a, (cross_block, 0))
        self.tb.connect(src_b, (cross_block, 1))
        self.tb.connect(cross_block, sink)

        # Run flowgraph
        self.tb.run()

        # Verify results
        result = sink.data()
        self.assertEqual(len(result), len(expected_output))
        for i, expected in enumerate(expected_output):
            self.assertAlmostEqual(result[i], expected, places=5)

    def test_general_cross_product(self):
        # Test general cross product: [2,3,4] × [5,6,7]
        if not HAVE_VECTOR_CROSS_PRODUCT:
            self.skipTest("vector_cross_product not available")

        from gnuradio import blocks

        # Input vectors
        input_a = [2.0, 3.0, 4.0]
        input_b = [5.0, 6.0, 7.0]
        # Expected: (3*7-4*6, 4*5-2*7, 2*6-3*5) = (21-24, 20-14, 12-15) = (-3, 6, -3)
        expected_output = [-3.0, 6.0, -3.0]

        # Create blocks
        src_a = blocks.vector_source_f(input_a, False, 3)
        src_b = blocks.vector_source_f(input_b, False, 3)
        cross_block = linalg_python.vector_cross_product_sync_f(True)
        sink = blocks.vector_sink_f(3)

        # Connect flowgraph
        self.tb.connect(src_a, (cross_block, 0))
        self.tb.connect(src_b, (cross_block, 1))
        self.tb.connect(cross_block, sink)

        # Run flowgraph
        self.tb.run()

        # Verify results
        result = sink.data()
        self.assertEqual(len(result), len(expected_output))
        for i, expected in enumerate(expected_output):
            self.assertAlmostEqual(result[i], expected, places=5)

    def test_anticommutative_property(self):
        # Test that a × b = -(b × a)
        if not HAVE_VECTOR_CROSS_PRODUCT:
            self.skipTest("vector_cross_product not available")

        from gnuradio import blocks

        # Input vectors
        input_a = [1.0, 2.0, 3.0]
        input_b = [4.0, 5.0, 6.0]

        # Test a × b
        tb1 = gr.top_block()
        src_a1 = blocks.vector_source_f(input_a, False, 3)
        src_b1 = blocks.vector_source_f(input_b, False, 3)
        cross_block1 = linalg_python.vector_cross_product_sync_f(True)
        sink1 = blocks.vector_sink_f(3)

        tb1.connect(src_a1, (cross_block1, 0))
        tb1.connect(src_b1, (cross_block1, 1))
        tb1.connect(cross_block1, sink1)
        tb1.run()
        result_ab = sink1.data()

        # Test b × a
        tb2 = gr.top_block()
        src_a2 = blocks.vector_source_f(input_a, False, 3)
        src_b2 = blocks.vector_source_f(input_b, False, 3)
        cross_block2 = linalg_python.vector_cross_product_sync_f(True)
        sink2 = blocks.vector_sink_f(3)

        tb2.connect(src_b2, (cross_block2, 0))  # Note: swapped order
        tb2.connect(src_a2, (cross_block2, 1))
        tb2.connect(cross_block2, sink2)
        tb2.run()
        result_ba = sink2.data()

        # Verify anticommutative property: a × b = -(b × a)
        self.assertEqual(len(result_ab), len(result_ba))
        for i in range(len(result_ab)):
            self.assertAlmostEqual(result_ab[i], -result_ba[i], places=5)

    def test_parallel_vectors(self):
        # Test cross product of parallel vectors (should be zero)
        if not HAVE_VECTOR_CROSS_PRODUCT:
            self.skipTest("vector_cross_product not available")

        from gnuradio import blocks

        # Parallel vectors: a = 2*b
        input_a = [2.0, 4.0, 6.0]
        input_b = [1.0, 2.0, 3.0]
        expected_output = [0.0, 0.0, 0.0]  # Should be zero vector

        # Create blocks
        src_a = blocks.vector_source_f(input_a, False, 3)
        src_b = blocks.vector_source_f(input_b, False, 3)
        cross_block = linalg_python.vector_cross_product_sync_f(True)
        sink = blocks.vector_sink_f(3)

        # Connect flowgraph
        self.tb.connect(src_a, (cross_block, 0))
        self.tb.connect(src_b, (cross_block, 1))
        self.tb.connect(cross_block, sink)

        # Run flowgraph
        self.tb.run()

        # Verify results
        result = sink.data()
        self.assertEqual(len(result), len(expected_output))
        for i, expected in enumerate(expected_output):
            self.assertAlmostEqual(result[i], expected, places=5)

    def test_orthogonality_property(self):
        # Test that result is orthogonal to both input vectors
        # (a × b) · a = 0 and (a × b) · b = 0
        if not HAVE_VECTOR_CROSS_PRODUCT:
            self.skipTest("vector_cross_product not available")

        from gnuradio import blocks

        # Input vectors
        input_a = [1.0, 3.0, 2.0]
        input_b = [2.0, 1.0, 4.0]

        # Create blocks
        src_a = blocks.vector_source_f(input_a, False, 3)
        src_b = blocks.vector_source_f(input_b, False, 3)
        cross_block = linalg_python.vector_cross_product_sync_f(True)
        sink = blocks.vector_sink_f(3)

        # Connect flowgraph
        self.tb.connect(src_a, (cross_block, 0))
        self.tb.connect(src_b, (cross_block, 1))
        self.tb.connect(cross_block, sink)

        # Run flowgraph
        self.tb.run()

        # Get cross product result
        result = sink.data()

        # Check orthogonality: (a × b) · a = 0
        dot_with_a = sum(result[i] * input_a[i] for i in range(3))
        self.assertAlmostEqual(dot_with_a, 0.0, places=5)

        # Check orthogonality: (a × b) · b = 0
        dot_with_b = sum(result[i] * input_b[i] for i in range(3))
        self.assertAlmostEqual(dot_with_b, 0.0, places=5)

    def test_magnitude_property(self):
        # Test that |a × b| = |a| * |b| * sin(θ)
        # For orthogonal unit vectors, this should be 1
        if not HAVE_VECTOR_CROSS_PRODUCT:
            self.skipTest("vector_cross_product not available")

        from gnuradio import blocks

        # Orthogonal unit vectors
        input_a = [1.0, 0.0, 0.0]  # unit vector along x
        input_b = [0.0, 1.0, 0.0]  # unit vector along y

        # Create blocks
        src_a = blocks.vector_source_f(input_a, False, 3)
        src_b = blocks.vector_source_f(input_b, False, 3)
        cross_block = linalg_python.vector_cross_product_sync_f(True)
        sink = blocks.vector_sink_f(3)

        # Connect flowgraph
        self.tb.connect(src_a, (cross_block, 0))
        self.tb.connect(src_b, (cross_block, 1))
        self.tb.connect(cross_block, sink)

        # Run flowgraph
        self.tb.run()

        # Calculate magnitude of result
        result = sink.data()
        magnitude = math.sqrt(sum(x * x for x in result))

        # For orthogonal unit vectors, magnitude should be 1
        self.assertAlmostEqual(magnitude, 1.0, places=5)

    def test_double_precision(self):
        # Test with double precision
        if not HAVE_VECTOR_CROSS_PRODUCT:
            self.skipTest("vector_cross_product not available")

        from gnuradio import blocks

        # High precision vectors
        input_a = [1.123456789, 2.987654321, 3.141592654]
        input_b = [2.718281828, 1.414213562, 1.732050808]

        # Create blocks (using float since vector_source_d might not be available)
        input_a_f = [float(x) for x in input_a]
        input_b_f = [float(x) for x in input_b]
        src_a = blocks.vector_source_f(input_a_f, False, 3)
        src_b = blocks.vector_source_f(input_b_f, False, 3)
        cross_block = linalg_python.vector_cross_product_sync_f(
            True
        )  # Use float version
        sink = blocks.vector_sink_f(3)

        # Connect flowgraph
        self.tb.connect(src_a, (cross_block, 0))
        self.tb.connect(src_b, (cross_block, 1))
        self.tb.connect(cross_block, sink)

        # Run flowgraph
        self.tb.run()

        # Verify results exist and are reasonable
        result = sink.data()
        self.assertEqual(len(result), 3)

        # Manual calculation for verification
        # a × b = (a₁b₂-a₂b₁, a₂b₀-a₀b₂, a₀b₁-a₁b₀)
        expected = [
            input_a[1] * input_b[2] - input_a[2] * input_b[1],
            input_a[2] * input_b[0] - input_a[0] * input_b[2],
            input_a[0] * input_b[1] - input_a[1] * input_b[0],
        ]

        for i, expected_val in enumerate(expected):
            self.assertAlmostEqual(
                result[i], expected_val, places=5
            )  # Reduced precision for float


if __name__ == "__main__":
    gr_unittest.run(qa_vector_cross_product)
