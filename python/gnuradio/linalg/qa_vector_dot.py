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
    # Try to import vector_dot class
    from gnuradio.linalg import vector_dot

    HAVE_VECTOR_DOT = True
except ImportError:
    HAVE_VECTOR_DOT = False


class qa_vector_dot(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance(self):
        # Test vector dot product block creation with valid shapes
        if not HAVE_VECTOR_DOT:
            self.skipTest("vector_dot not available")

        # Create vector dot product: 3-element vectors
        shape_a = [3]
        shape_b = [3]
        instance = vector_dot(shape_a, shape_b)
        if instance is None:
            self.skipTest("vector_dot not available in Python bindings")
        self.assertIsNotNone(instance)

    def test_vector_dot_product_validation(self):
        # Test vector dot product with known input/output
        if not HAVE_VECTOR_DOT:
            self.skipTest("vector_dot not available")

        from gnuradio import blocks

        # Create test vectors: [1, 2, 3] · [4, 5, 6] = 1*4 + 2*5 + 3*6 = 32
        shape_a = [3]
        shape_b = [3]

        vector_a = [1.0, 2.0, 3.0]
        vector_b = [4.0, 5.0, 6.0]
        expected_output = [32.0]  # Scalar result

        # Create blocks
        src_a = blocks.vector_source_f(vector_a, False, 3)  # 3 elements
        src_b = blocks.vector_source_f(vector_b, False, 3)
        dot_product = vector_dot(shape_a, shape_b)
        if dot_product is None:
            self.skipTest("vector_dot not available in Python bindings")
        sink = blocks.vector_sink_f(1)  # Scalar output

        # Connect flowgraph
        self.tb.connect(src_a, (dot_product, 0))
        self.tb.connect(src_b, (dot_product, 1))
        self.tb.connect(dot_product, sink)

        # Run flowgraph
        self.tb.run()

        # Verify results
        result = sink.data()
        self.assertEqual(len(result), len(expected_output))
        for i, expected in enumerate(expected_output):
            self.assertAlmostEqual(result[i], expected, places=5)

    def test_orthogonal_vectors(self):
        # Test dot product of orthogonal vectors (should be 0)
        if not HAVE_VECTOR_DOT:
            self.skipTest("vector_dot not available")

        from gnuradio import blocks

        # Orthogonal vectors: [1, 0] · [0, 1] = 0
        shape_a = [2]
        shape_b = [2]

        vector_a = [1.0, 0.0]
        vector_b = [0.0, 1.0]
        expected_output = [0.0]

        src_a = blocks.vector_source_f(vector_a, False, 2)
        src_b = blocks.vector_source_f(vector_b, False, 2)
        dot_product = vector_dot(shape_a, shape_b)
        if dot_product is None:
            self.skipTest("vector_dot not available in Python bindings")
        sink = blocks.vector_sink_f(1)

        self.tb.connect(src_a, (dot_product, 0))
        self.tb.connect(src_b, (dot_product, 1))
        self.tb.connect(dot_product, sink)

        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), len(expected_output))
        for i, expected in enumerate(expected_output):
            self.assertAlmostEqual(result[i], expected, places=5)

    def test_parallel_vectors(self):
        # Test dot product of parallel vectors
        if not HAVE_VECTOR_DOT:
            self.skipTest("vector_dot not available")

        from gnuradio import blocks

        # Parallel vectors: [3, 4] · [3, 4] = 9 + 16 = 25 (magnitude squared)
        shape_a = [2]
        shape_b = [2]

        vector_a = [3.0, 4.0]
        vector_b = [3.0, 4.0]
        expected_output = [25.0]

        src_a = blocks.vector_source_f(vector_a, False, 2)
        src_b = blocks.vector_source_f(vector_b, False, 2)
        dot_product = vector_dot(shape_a, shape_b)
        if dot_product is None:
            self.skipTest("vector_dot not available in Python bindings")
        sink = blocks.vector_sink_f(1)

        self.tb.connect(src_a, (dot_product, 0))
        self.tb.connect(src_b, (dot_product, 1))
        self.tb.connect(dot_product, sink)

        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), len(expected_output))
        for i, expected in enumerate(expected_output):
            self.assertAlmostEqual(result[i], expected, places=5)

    def test_different_vector_sizes(self):
        # Test different vector sizes
        if not HAVE_VECTOR_DOT:
            self.skipTest("vector_dot not available")

        from gnuradio import blocks

        # Test 5-element vectors: [1, 1, 1, 1, 1] · [2, 2, 2, 2, 2] = 10
        shape_a = [5]
        shape_b = [5]

        vector_a = [1.0, 1.0, 1.0, 1.0, 1.0]
        vector_b = [2.0, 2.0, 2.0, 2.0, 2.0]
        expected_output = [10.0]

        src_a = blocks.vector_source_f(vector_a, False, 5)
        src_b = blocks.vector_source_f(vector_b, False, 5)
        dot_product = vector_dot(shape_a, shape_b)
        if dot_product is None:
            self.skipTest("vector_dot not available in Python bindings")
        sink = blocks.vector_sink_f(1)

        self.tb.connect(src_a, (dot_product, 0))
        self.tb.connect(src_b, (dot_product, 1))
        self.tb.connect(dot_product, sink)

        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), len(expected_output))
        for i, expected in enumerate(expected_output):
            self.assertAlmostEqual(result[i], expected, places=5)


if __name__ == "__main__":
    gr_unittest.run(qa_vector_dot)
