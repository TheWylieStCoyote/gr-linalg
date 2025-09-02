#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2025 Wylie Standage-Beier.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import numpy as np
from gnuradio import gr, gr_unittest, blocks

try:
    from gnuradio.linalg import (
        matrix_norm_sync_f,
        matrix_norm_sync_d,
        matrix_norm_sync_c,
        matrix_norm_sync_cd,
        norm_type,
    )
except ImportError:
    import os
    import sys

    # Try importing directly from the C++ bindings as fallback
    try:
        dirname, filename = os.path.split(os.path.abspath(__file__))
        build_path = os.path.join(
            dirname, "..", "..", "build", "test_modules", "gnuradio", "linalg"
        )
        if os.path.exists(build_path):
            sys.path.insert(0, build_path)
        import linalg_python

        # Create aliases for easier testing
        matrix_norm_sync_f = linalg_python.matrix_norm_sync_f
        matrix_norm_sync_d = linalg_python.matrix_norm_sync_d
        matrix_norm_sync_c = linalg_python.matrix_norm_sync_c
        matrix_norm_sync_cd = linalg_python.matrix_norm_sync_cd
        norm_type = linalg_python.norm_type
    except ImportError as e:
        print(f"Could not import matrix_norm classes: {e}")
        raise


class qa_matrix_norm(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance_float(self):
        """Test that we can create matrix_norm instances with proper constructors"""
        # Test float version
        instance_f = matrix_norm_sync_f([2, 2], norm_type.Frobenius)
        self.assertIsNotNone(instance_f)

        # Test double version
        instance_d = matrix_norm_sync_d([3, 3], norm_type.L1)
        self.assertIsNotNone(instance_d)

        # Test complex versions
        instance_c = matrix_norm_sync_c([2, 2], norm_type.Max)
        self.assertIsNotNone(instance_c)

        instance_cd = matrix_norm_sync_cd([2, 2], norm_type.L2)
        self.assertIsNotNone(instance_cd)

    def test_frobenius_norm_2x2_float(self):
        """Test Frobenius norm computation for 2x2 float matrix"""
        # Create a 2x2 identity matrix
        # Input: [[1, 0], [0, 1]] -> Frobenius norm = sqrt(1^2 + 0^2 + 0^2 + 1^2) = sqrt(2)
        input_data = [1.0, 0.0, 0.0, 1.0]  # Row-major order
        expected_norm = np.sqrt(2.0)

        # Create flowgraph
        src = blocks.vector_source_f(input_data, False, 4)  # 4 elements for 2x2 matrix
        norm_block = matrix_norm_sync_f([2, 2], norm_type.Frobenius)
        sink = blocks.vector_sink_f(1)  # 1 element output (scalar)

        self.tb.connect(src, norm_block)
        self.tb.connect(norm_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 1, "Should output exactly one norm value")
        self.assertAlmostEqual(
            result[0],
            expected_norm,
            places=5,
            msg=f"Frobenius norm should be {expected_norm}, got {result[0]}",
        )

    def test_l1_norm_2x2_float(self):
        """Test L1 norm computation for 2x2 float matrix"""
        # Input data [1, 2, 3, 4] is interpreted as column-major: [[1, 3], [2, 4]]
        # L1 norm = max(|1|+|2|, |3|+|4|) = max(3, 7) = 7
        input_data = [1.0, 2.0, 3.0, 4.0]
        expected_norm = 7.0

        # Create flowgraph
        src = blocks.vector_source_f(input_data, False, 4)
        norm_block = matrix_norm_sync_f([2, 2], norm_type.L1)
        sink = blocks.vector_sink_f(1)

        self.tb.connect(src, norm_block)
        self.tb.connect(norm_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(
            result[0],
            expected_norm,
            places=5,
            msg=f"L1 norm should be {expected_norm}, got {result[0]}",
        )

    def test_max_norm_3x3_float(self):
        """Test Max norm computation for 3x3 float matrix"""
        # Input: [[1, -5, 2], [0, 3, -1], [4, 0, 2]]
        # Max norm = max(|1|, |-5|, |2|, |0|, |3|, |-1|, |4|, |0|, |2|) = 5
        input_data = [1.0, -5.0, 2.0, 0.0, 3.0, -1.0, 4.0, 0.0, 2.0]
        expected_norm = 5.0

        src = blocks.vector_source_f(input_data, False, 9)
        norm_block = matrix_norm_sync_f([3, 3], norm_type.Max)
        sink = blocks.vector_sink_f(1)

        self.tb.connect(src, norm_block)
        self.tb.connect(norm_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(
            result[0],
            expected_norm,
            places=5,
            msg=f"Max norm should be {expected_norm}, got {result[0]}",
        )

    def test_complex_frobenius_norm(self):
        """Test Frobenius norm for complex matrix"""
        # Input: [[1+2j, 0], [0, 3+4j]]
        # Frobenius norm = sqrt(|1+2j|^2 + |0|^2 + |0|^2 + |3+4j|^2)
        #                = sqrt(5 + 25) = sqrt(30)
        input_data = [1 + 2j, 0 + 0j, 0 + 0j, 3 + 4j]
        expected_norm = np.sqrt(30.0)

        src = blocks.vector_source_c(input_data, False, 4)
        norm_block = matrix_norm_sync_c([2, 2], norm_type.Frobenius)
        sink = blocks.vector_sink_c(1)

        self.tb.connect(src, norm_block)
        self.tb.connect(norm_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 1)
        # For complex output, norm should be real part only
        self.assertAlmostEqual(
            result[0].real,
            expected_norm,
            places=5,
            msg=f"Complex Frobenius norm should be {expected_norm}, got {result[0].real}",
        )
        self.assertAlmostEqual(
            result[0].imag, 0.0, places=5, msg="Imaginary part should be zero"
        )

    def test_multiple_matrices_stream(self):
        """Test processing multiple matrices in a stream"""
        # Two 2x2 matrices: [[1, 0], [0, 1]] and [[2, 0], [0, 2]]
        # Frobenius norms: sqrt(2) and sqrt(8) = 2*sqrt(2)
        input_data = [
            1.0,
            0.0,
            0.0,
            1.0,  # First matrix
            2.0,
            0.0,
            0.0,
            2.0,
        ]  # Second matrix
        expected_norms = [np.sqrt(2.0), 2.0 * np.sqrt(2.0)]

        src = blocks.vector_source_f(input_data, False, 4)
        norm_block = matrix_norm_sync_f([2, 2], norm_type.Frobenius)
        sink = blocks.vector_sink_f(1)

        self.tb.connect(src, norm_block)
        self.tb.connect(norm_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 2, "Should output two norm values")
        for i, expected in enumerate(expected_norms):
            self.assertAlmostEqual(
                result[i],
                expected,
                places=5,
                msg=f"Matrix {i}: norm should be {expected}, got {result[i]}",
            )

    def test_default_norm_type(self):
        """Test that default norm type is Frobenius"""
        # Create block without specifying norm type
        norm_block = matrix_norm_sync_f([2, 2])  # Should default to Frobenius
        self.assertIsNotNone(norm_block)

        # Test with a simple matrix
        input_data = [3.0, 4.0, 0.0, 0.0]  # [[3, 4], [0, 0]]
        expected_norm = 5.0  # sqrt(3^2 + 4^2) = 5

        src = blocks.vector_source_f(input_data, False, 4)
        sink = blocks.vector_sink_f(1)

        self.tb.connect(src, norm_block)
        self.tb.connect(norm_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(
            result[0],
            expected_norm,
            places=5,
            msg=f"Default (Frobenius) norm should be {expected_norm}, got {result[0]}",
        )

    def test_norm_types_available(self):
        """Test that all norm types are available and can be used"""
        # Test that all norm types exist and can be used
        norm_types = [norm_type.L1, norm_type.L2, norm_type.Frobenius, norm_type.Max]

        for nt in norm_types:
            with self.subTest(norm_type=nt):
                # Should be able to create block with each norm type
                norm_block = matrix_norm_sync_f([2, 2], nt)
                self.assertIsNotNone(
                    norm_block, f"Failed to create block with norm type {nt}"
                )


if __name__ == "__main__":
    gr_unittest.run(qa_matrix_norm)
