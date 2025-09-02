#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2025 Wylie Standage-Beier.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

from gnuradio import gr, gr_unittest, blocks

# from gnuradio import blocks
try:
    from gnuradio.linalg import dot_product
except ImportError:
    try:
        # Fallback to direct import from linalg module
        import sys

        sys.path.insert(0, "python")
        from linalg import dot_product
    except ImportError:
        import os
        import sys

        dirname, filename = os.path.split(os.path.abspath(__file__))
        sys.path.append(os.path.join(dirname, "bindings"))
        from gnuradio.linalg import dot_product


class qa_dot_product(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance(self):
        # Test creating an instance with proper arguments
        shape_0 = [3]  # Vector of length 3
        shape_1 = [3]  # Vector of length 3
        instance = dot_product(shape_0, shape_1)
        if instance is None:
            self.skipTest("dot_product not available in Python bindings")
        self.assertIsNotNone(instance)

    def test_001_descriptive_test_name(self):
        # set up fg
        self.tb.run()
        # check data

    def test_dot_product_simple(self):
        # Test the dot product of two simple vectors
        shape_0 = [3]
        shape_1 = [3]
        source_vector = [1, 2, 3]
        instance = dot_product(shape_0, shape_1)
        if instance is None:
            self.skipTest("dot_product not available in Python bindings")
        source_block_0 = blocks.vector_source_f(source_vector, False, 3)
        source_block_1 = blocks.vector_source_f(source_vector, False, 3)
        sink_block = blocks.vector_sink_f(1)

        self.tb.connect(source_block_0, (instance, 0))
        self.tb.connect(source_block_1, (instance, 1))
        self.tb.connect(instance, sink_block)
        self.tb.run()

        result = sink_block.data()
        expected_result = [14.0]  # 1*1 + 2*2 + 3*3
        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    gr_unittest.run(qa_dot_product)
