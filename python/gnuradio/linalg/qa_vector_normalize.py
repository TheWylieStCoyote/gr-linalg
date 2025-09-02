#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2025 Wylie Standage-Beier.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

from gnuradio import gr, gr_unittest
import numpy as np
import math

# from gnuradio import blocks
try:
    # Try to import vector_normalize classes
    from gnuradio.linalg import vector_normalize_sync_f

    HAVE_VECTOR_NORMALIZE = True
    # If basic import works, try the others
    try:
        from gnuradio.linalg import vector_normalize_sync_d

        HAVE_ALL_TYPES = True
    except ImportError:
        HAVE_ALL_TYPES = False
except ImportError:
    HAVE_VECTOR_NORMALIZE = False
    HAVE_ALL_TYPES = False


class qa_vector_normalize(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance(self):
        # Test vector_normalize block creation with valid parameters
        if not HAVE_VECTOR_NORMALIZE:
            self.skipTest("vector_normalize_sync_f not available")

        # Create vector_normalize: 3D vector with L2 norm
        shape = [3]
        norm_type = 2  # L2 norm
        instance = vector_normalize_sync_f(shape, norm_type)
        self.assertIsNotNone(instance)

    def test_l2_normalize_3d_vector(self):
        # Test L2 normalization of 3D vector
        if not HAVE_VECTOR_NORMALIZE:
            self.skipTest("vector_normalize_sync_f not available")

        from gnuradio import blocks

        # Input vector [3, 4, 0] -> L2 norm = sqrt(9+16+0) = 5
        # Normalized: [3/5, 4/5, 0/5] = [0.6, 0.8, 0.0]
        input_vector = [3.0, 4.0, 0.0]
        expected_normalized = [0.6, 0.8, 0.0]

        src = blocks.vector_source_f(input_vector, False, 3)
        normalize_block = vector_normalize_sync_f([3], 2)  # L2 norm
        sink = blocks.vector_sink_f(3)

        self.tb.connect(src, normalize_block)
        self.tb.connect(normalize_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 3)
        for i, expected in enumerate(expected_normalized):
            self.assertAlmostEqual(result[i], expected, places=6)

    def test_l2_normalize_2d_vector(self):
        # Test L2 normalization of 2D vector
        if not HAVE_VECTOR_NORMALIZE:
            self.skipTest("vector_normalize_sync_f not available")

        from gnuradio import blocks

        # Input vector [1, 1] -> L2 norm = sqrt(1+1) = sqrt(2)
        # Normalized: [1/sqrt(2), 1/sqrt(2)] = [0.7071..., 0.7071...]
        input_vector = [1.0, 1.0]
        expected_normalized = [1.0 / math.sqrt(2), 1.0 / math.sqrt(2)]

        src = blocks.vector_source_f(input_vector, False, 2)
        normalize_block = vector_normalize_sync_f([2], 2)  # L2 norm
        sink = blocks.vector_sink_f(2)

        self.tb.connect(src, normalize_block)
        self.tb.connect(normalize_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 2)
        for i, expected in enumerate(expected_normalized):
            self.assertAlmostEqual(result[i], expected, places=6)

    def test_l1_normalize_vector(self):
        # Test L1 (Manhattan) normalization
        if not HAVE_VECTOR_NORMALIZE:
            self.skipTest("vector_normalize_sync_f not available")

        from gnuradio import blocks

        # Input vector [2, 3, 1] -> L1 norm = 2+3+1 = 6
        # Normalized: [2/6, 3/6, 1/6] = [0.333..., 0.5, 0.166...]
        input_vector = [2.0, 3.0, 1.0]
        expected_normalized = [2.0 / 6.0, 3.0 / 6.0, 1.0 / 6.0]

        src = blocks.vector_source_f(input_vector, False, 3)
        normalize_block = vector_normalize_sync_f([3], 1)  # L1 norm
        sink = blocks.vector_sink_f(3)

        self.tb.connect(src, normalize_block)
        self.tb.connect(normalize_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 3)
        for i, expected in enumerate(expected_normalized):
            self.assertAlmostEqual(result[i], expected, places=6)

    def test_linf_normalize_vector(self):
        # Test L∞ (max) normalization
        if not HAVE_VECTOR_NORMALIZE:
            self.skipTest("vector_normalize_sync_f not available")

        from gnuradio import blocks

        # Input vector [1, 5, 2] -> L∞ norm = max(1,5,2) = 5
        # Normalized: [1/5, 5/5, 2/5] = [0.2, 1.0, 0.4]
        input_vector = [1.0, 5.0, 2.0]
        expected_normalized = [0.2, 1.0, 0.4]

        src = blocks.vector_source_f(input_vector, False, 3)
        normalize_block = vector_normalize_sync_f(
            [3], 0
        )  # L∞ norm (usually 0 or special value)
        sink = blocks.vector_sink_f(3)

        self.tb.connect(src, normalize_block)
        self.tb.connect(normalize_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 3)
        for i, expected in enumerate(expected_normalized):
            self.assertAlmostEqual(result[i], expected, places=6)

    def test_unit_vector_unchanged(self):
        # Test that unit vector remains unchanged
        if not HAVE_VECTOR_NORMALIZE:
            self.skipTest("vector_normalize_sync_f not available")

        from gnuradio import blocks

        # Input vector [1, 0, 0] already has L2 norm = 1
        # Should remain [1, 0, 0] after normalization
        input_vector = [1.0, 0.0, 0.0]
        expected_normalized = [1.0, 0.0, 0.0]

        src = blocks.vector_source_f(input_vector, False, 3)
        normalize_block = vector_normalize_sync_f([3], 2)  # L2 norm
        sink = blocks.vector_sink_f(3)

        self.tb.connect(src, normalize_block)
        self.tb.connect(normalize_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 3)
        for i, expected in enumerate(expected_normalized):
            self.assertAlmostEqual(result[i], expected, places=6)

    def test_normalized_vector_properties(self):
        # Test mathematical properties of normalized vector
        if not HAVE_VECTOR_NORMALIZE:
            self.skipTest("vector_normalize_sync_f not available")

        from gnuradio import blocks

        # Input vector [6, 8] -> L2 norm = 10
        # Normalized: [0.6, 0.8] -> should have L2 norm = 1
        input_vector = [6.0, 8.0]

        src = blocks.vector_source_f(input_vector, False, 2)
        normalize_block = vector_normalize_sync_f([2], 2)  # L2 norm
        sink = blocks.vector_sink_f(2)

        self.tb.connect(src, normalize_block)
        self.tb.connect(normalize_block, sink)
        self.tb.run()

        result = sink.data()

        # Verify the normalized vector has unit norm
        normalized_norm = math.sqrt(result[0] ** 2 + result[1] ** 2)
        self.assertAlmostEqual(normalized_norm, 1.0, places=6)

        # Verify direction is preserved (angle unchanged)
        original_angle = math.atan2(input_vector[1], input_vector[0])
        normalized_angle = math.atan2(result[1], result[0])
        self.assertAlmostEqual(original_angle, normalized_angle, places=6)

    def test_zero_vector_handling(self):
        # Test handling of zero vector (edge case)
        if not HAVE_VECTOR_NORMALIZE:
            self.skipTest("vector_normalize_sync_f not available")

        from gnuradio import blocks

        # Zero vector [0, 0, 0] -> undefined normalization
        # Implementation should handle gracefully (often returns zero or NaN)
        input_vector = [0.0, 0.0, 0.0]

        src = blocks.vector_source_f(input_vector, False, 3)
        normalize_block = vector_normalize_sync_f([3], 2)  # L2 norm
        sink = blocks.vector_sink_f(3)

        self.tb.connect(src, normalize_block)
        self.tb.connect(normalize_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 3)

        # Result should be either all zeros or all NaN
        # Check if all elements are the same (consistent behavior)
        self.assertTrue(all(abs(result[i] - result[0]) < 1e-10 for i in range(3)))

    def test_negative_components_preserved(self):
        # Test that negative components are preserved in normalization
        if not HAVE_VECTOR_NORMALIZE:
            self.skipTest("vector_normalize_sync_f not available")

        from gnuradio import blocks

        # Input vector [-3, 4] -> L2 norm = 5
        # Normalized: [-3/5, 4/5] = [-0.6, 0.8]
        input_vector = [-3.0, 4.0]
        expected_normalized = [-0.6, 0.8]

        src = blocks.vector_source_f(input_vector, False, 2)
        normalize_block = vector_normalize_sync_f([2], 2)  # L2 norm
        sink = blocks.vector_sink_f(2)

        self.tb.connect(src, normalize_block)
        self.tb.connect(normalize_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 2)
        for i, expected in enumerate(expected_normalized):
            self.assertAlmostEqual(result[i], expected, places=6)

    def test_double_precision_normalize(self):
        # Test double precision normalization
        # Note: GNU Radio doesn't provide vector_source_d/vector_sink_d, so we skip this test
        if not HAVE_ALL_TYPES:
            self.skipTest("vector_normalize_sync_d not available")

        self.skipTest(
            "GNU Radio doesn't provide vector_source_d/vector_sink_d for double precision testing"
        )

    def test_large_vector_normalization(self):
        # Test normalization of larger vectors
        if not HAVE_VECTOR_NORMALIZE:
            self.skipTest("vector_normalize_sync_f not available")

        from gnuradio import blocks

        # 5D vector [1, 2, 3, 4, 5] -> L2 norm = sqrt(55) ≈ 7.416
        input_vector = [1.0, 2.0, 3.0, 4.0, 5.0]
        norm_value = math.sqrt(sum(x * x for x in input_vector))
        expected_normalized = [x / norm_value for x in input_vector]

        src = blocks.vector_source_f(input_vector, False, 5)
        normalize_block = vector_normalize_sync_f([5], 2)  # L2 norm
        sink = blocks.vector_sink_f(5)

        self.tb.connect(src, normalize_block)
        self.tb.connect(normalize_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 5)
        for i, expected in enumerate(expected_normalized):
            self.assertAlmostEqual(result[i], expected, places=6)

        # Verify unit norm
        computed_norm = math.sqrt(sum(x * x for x in result))
        self.assertAlmostEqual(computed_norm, 1.0, places=6)

    def test_single_element_vector(self):
        # Test normalization of 1D vector (scalar)
        if not HAVE_VECTOR_NORMALIZE:
            self.skipTest("vector_normalize_sync_f not available")

        from gnuradio import blocks

        # Single element vector [7.0] -> normalized: [1.0] (sign preserved)
        input_vector = [7.0]
        expected_normalized = [1.0]

        src = blocks.vector_source_f(input_vector, False, 1)
        normalize_block = vector_normalize_sync_f([1], 2)  # L2 norm
        sink = blocks.vector_sink_f(1)

        self.tb.connect(src, normalize_block)
        self.tb.connect(normalize_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], expected_normalized[0], places=6)

        # Test negative single element
        self.tb = gr.top_block()  # Reset flowgraph
        input_vector_neg = [-5.0]
        expected_normalized_neg = [-1.0]

        src_neg = blocks.vector_source_f(input_vector_neg, False, 1)
        normalize_block_neg = vector_normalize_sync_f([1], 2)
        sink_neg = blocks.vector_sink_f(1)

        self.tb.connect(src_neg, normalize_block_neg)
        self.tb.connect(normalize_block_neg, sink_neg)
        self.tb.run()

        result_neg = sink_neg.data()
        self.assertAlmostEqual(result_neg[0], expected_normalized_neg[0], places=6)

    def test_normalization_idempotent(self):
        # Test that normalizing a normalized vector doesn't change it
        if not HAVE_VECTOR_NORMALIZE:
            self.skipTest("vector_normalize_sync_f not available")

        from gnuradio import blocks

        # Start with arbitrary vector, normalize it twice
        input_vector = [2.0, 3.0, 6.0]

        # First normalization
        src1 = blocks.vector_source_f(input_vector, False, 3)
        normalize_block1 = vector_normalize_sync_f([3], 2)
        sink1 = blocks.vector_sink_f(3)

        self.tb.connect(src1, normalize_block1)
        self.tb.connect(normalize_block1, sink1)
        self.tb.run()

        first_result = list(sink1.data())

        # Second normalization
        self.tb = gr.top_block()  # Reset flowgraph
        src2 = blocks.vector_source_f(first_result, False, 3)
        normalize_block2 = vector_normalize_sync_f([3], 2)
        sink2 = blocks.vector_sink_f(3)

        self.tb.connect(src2, normalize_block2)
        self.tb.connect(normalize_block2, sink2)
        self.tb.run()

        second_result = list(sink2.data())

        # Results should be identical (idempotent property)
        self.assertEqual(len(first_result), len(second_result))
        for i in range(len(first_result)):
            self.assertAlmostEqual(first_result[i], second_result[i], places=6)


if __name__ == "__main__":
    gr_unittest.run(qa_vector_normalize)
