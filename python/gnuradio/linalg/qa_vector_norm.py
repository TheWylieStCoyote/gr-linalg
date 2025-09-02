#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2025 Wylie Standage-Beier.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

from gnuradio import gr, gr_unittest
import math

# from gnuradio import blocks
try:
    # Try to import vector_norm class
    from gnuradio.linalg import vector_norm

    HAVE_VECTOR_NORM = True
except ImportError:
    HAVE_VECTOR_NORM = False


class qa_vector_norm(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance(self):
        # Test vector norm block creation with valid shapes
        if not HAVE_VECTOR_NORM:
            self.skipTest("vector_norm not available")

        # Create vector norm: 3-element vector
        shape = [3]
        instance = vector_norm(shape)
        if instance is None:
            self.skipTest("vector_norm not available in Python bindings")
        if instance is None:
            self.skipTest("vector_norm not available in Python bindings")
        self.assertIsNotNone(instance)

    def test_euclidean_norm_validation(self):
        # Test vector norm (Euclidean/L2 norm) with known input/output
        if not HAVE_VECTOR_NORM:
            self.skipTest("vector_norm not available")

        from gnuradio import blocks

        # Create test vector: [3, 4] -> norm = sqrt(3^2 + 4^2) = sqrt(25) = 5
        shape = [2]

        input_vector = [3.0, 4.0]
        expected_output = [5.0]  # Euclidean norm

        # Create blocks
        src = blocks.vector_source_f(input_vector, False, 2)  # 2 elements
        norm_block = vector_norm(shape)
        if norm_block is None:
            self.skipTest("vector_norm not available in Python bindings")
        sink = blocks.vector_sink_f(1)  # Scalar output

        # Connect flowgraph
        self.tb.connect(src, norm_block)
        self.tb.connect(norm_block, sink)

        # Run flowgraph
        self.tb.run()

        # Verify results
        result = sink.data()
        self.assertEqual(len(result), len(expected_output))
        for i, expected in enumerate(expected_output):
            self.assertAlmostEqual(result[i], expected, places=5)

    def test_unit_vector_norm(self):
        # Test norm of unit vector (should be 1)
        if not HAVE_VECTOR_NORM:
            self.skipTest("vector_norm not available")

        from gnuradio import blocks

        # Unit vector: [1, 0] -> norm = 1
        shape = [2]

        input_vector = [1.0, 0.0]
        expected_output = [1.0]

        src = blocks.vector_source_f(input_vector, False, 2)
        norm_block = vector_norm(shape)
        if norm_block is None:
            self.skipTest("vector_norm not available in Python bindings")
        sink = blocks.vector_sink_f(1)

        self.tb.connect(src, norm_block)
        self.tb.connect(norm_block, sink)

        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), len(expected_output))
        for i, expected in enumerate(expected_output):
            self.assertAlmostEqual(result[i], expected, places=5)

    def test_zero_vector_norm(self):
        # Test norm of zero vector (should be 0)
        if not HAVE_VECTOR_NORM:
            self.skipTest("vector_norm not available")

        from gnuradio import blocks

        # Zero vector: [0, 0, 0] -> norm = 0
        shape = [3]

        input_vector = [0.0, 0.0, 0.0]
        expected_output = [0.0]

        src = blocks.vector_source_f(input_vector, False, 3)
        norm_block = vector_norm(shape)
        if norm_block is None:
            self.skipTest("vector_norm not available in Python bindings")
        sink = blocks.vector_sink_f(1)

        self.tb.connect(src, norm_block)
        self.tb.connect(norm_block, sink)

        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), len(expected_output))
        for i, expected in enumerate(expected_output):
            self.assertAlmostEqual(result[i], expected, places=5)

    def test_high_dimensional_norm(self):
        # Test norm of higher dimensional vector
        if not HAVE_VECTOR_NORM:
            self.skipTest("vector_norm not available")

        from gnuradio import blocks

        # 5D vector: [1, 1, 1, 1, 1] -> norm = sqrt(5) ≈ 2.236
        shape = [5]

        input_vector = [1.0, 1.0, 1.0, 1.0, 1.0]
        expected_output = [math.sqrt(5.0)]  # ≈ 2.236

        src = blocks.vector_source_f(input_vector, False, 5)
        norm_block = vector_norm(shape)
        if norm_block is None:
            self.skipTest("vector_norm not available in Python bindings")
        sink = blocks.vector_sink_f(1)

        self.tb.connect(src, norm_block)
        self.tb.connect(norm_block, sink)

        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), len(expected_output))
        for i, expected in enumerate(expected_output):
            self.assertAlmostEqual(result[i], expected, places=5)

    def test_known_pythagorean_triple(self):
        # Test with Pythagorean triple [5, 12, 13] -> norm = sqrt(5^2 + 12^2 + 13^2) = sqrt(338) ≈ 18.385
        if not HAVE_VECTOR_NORM:
            self.skipTest("vector_norm not available")

        from gnuradio import blocks

        shape = [3]

        input_vector = [5.0, 12.0, 13.0]
        expected_output = [math.sqrt(5 * 5 + 12 * 12 + 13 * 13)]  # sqrt(338) ≈ 18.385

        src = blocks.vector_source_f(input_vector, False, 3)
        norm_block = vector_norm(shape)
        if norm_block is None:
            self.skipTest("vector_norm not available in Python bindings")
        sink = blocks.vector_sink_f(1)

        self.tb.connect(src, norm_block)
        self.tb.connect(norm_block, sink)

        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), len(expected_output))
        for i, expected in enumerate(expected_output):
            self.assertAlmostEqual(result[i], expected, places=4)


if __name__ == "__main__":
    gr_unittest.run(qa_vector_norm)
