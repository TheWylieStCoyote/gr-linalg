#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2025 Wylie Standage-Beier.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

from gnuradio import gr, gr_unittest
from gnuradio import blocks

try:
    from gnuradio import linalg
except ImportError:
    import os
    import sys

    dirname, filename = os.path.split(os.path.abspath(__file__))
    sys.path.append(os.path.join(dirname, "bindings"))
    from linalg import *

import numpy as np


class qa_matrix_eigenvalues(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_matrix_eigenvalues_identity_float(self):
        """Test eigenvalues of identity matrix (should all be 1.0)"""
        # Identity matrix: [[1, 0], [0, 1]] in column-major format
        input_data = [1.0, 0.0, 0.0, 1.0]

        src = blocks.vector_source_f(input_data, False, 4)
        eigenvalues_block = matrix_eigenvalues_ff([2, 2])
        sink = blocks.vector_sink_c(2)  # Eigenvalues are complex

        self.tb.connect(src, eigenvalues_block)
        self.tb.connect(eigenvalues_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 2, "Should have 2 eigenvalues")

        # All eigenvalues should be 1.0 (within tolerance)
        for eigenvalue in result:
            self.assertAlmostEqual(
                abs(eigenvalue),
                1.0,
                places=5,
                msg=f"Identity matrix eigenvalue {eigenvalue} should have magnitude 1.0",
            )
            self.assertAlmostEqual(
                eigenvalue.imag,
                0.0,
                places=5,
                msg="Identity matrix eigenvalues should be real",
            )

    def test_matrix_eigenvalues_diagonal_double(self):
        """Test eigenvalues of diagonal matrix"""
        # Diagonal matrix: [[2, 0], [0, 3]] in column-major format
        input_data = [2.0, 0.0, 0.0, 3.0]

        src = blocks.vector_source_f(input_data, False, 4)
        eigenvalues_block = matrix_eigenvalues_dd([2, 2])
        sink = blocks.vector_sink_c(2)  # Eigenvalues are complex

        self.tb.connect(src, eigenvalues_block)
        self.tb.connect(eigenvalues_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 2, "Should have 2 eigenvalues")

        # Eigenvalues should be 2.0 and 3.0 (in some order)
        real_eigenvalues = [eigenvalue.real for eigenvalue in result]
        real_eigenvalues.sort()

        self.assertAlmostEqual(real_eigenvalues[0], 2.0, places=5)
        self.assertAlmostEqual(real_eigenvalues[1], 3.0, places=5)

        # All should be real
        for eigenvalue in result:
            self.assertAlmostEqual(
                eigenvalue.imag,
                0.0,
                places=5,
                msg="Diagonal matrix eigenvalues should be real",
            )

    def test_matrix_eigenvalues_complex_float(self):
        """Test eigenvalues of complex matrix"""
        # Complex matrix: [[1+i, 0], [0, 1-i]] in column-major format
        input_data = [complex(1, 1), complex(0, 0), complex(0, 0), complex(1, -1)]

        src = blocks.vector_source_c(input_data, False, 4)
        eigenvalues_block = matrix_eigenvalues_cc([2, 2])
        sink = blocks.vector_sink_c(2)  # Eigenvalues are complex

        self.tb.connect(src, eigenvalues_block)
        self.tb.connect(eigenvalues_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 2, "Should have 2 eigenvalues")

        # Eigenvalues should be 1+i and 1-i (in some order)
        expected_eigenvalues = [complex(1, 1), complex(1, -1)]
        result_list = list(result)

        # Check that we have the expected eigenvalues (order may vary)
        for expected in expected_eigenvalues:
            found = False
            for actual in result_list:
                if abs(actual - expected) < 1e-5:
                    found = True
                    break
            self.assertTrue(
                found,
                f"Expected eigenvalue {expected} not found in result {result_list}",
            )

    def test_matrix_eigenvalues_complex_double(self):
        """Test eigenvalues of complex double matrix"""
        # Complex matrix: [[2+0i, 1+0i], [0+0i, 2+0i]] (upper triangular)
        input_data = [complex(2, 0), complex(0, 0), complex(1, 0), complex(2, 0)]

        src = blocks.vector_source_c(input_data, False, 4)
        eigenvalues_block = matrix_eigenvalues_zz([2, 2])
        sink = blocks.vector_sink_c(2)

        self.tb.connect(src, eigenvalues_block)
        self.tb.connect(eigenvalues_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 2, "Should have 2 eigenvalues")

        # For upper triangular matrix, eigenvalues are the diagonal elements (2, 2)
        for eigenvalue in result:
            self.assertAlmostEqual(
                eigenvalue.real,
                2.0,
                places=5,
                msg=f"Eigenvalue {eigenvalue} should have real part 2.0",
            )
            self.assertAlmostEqual(
                eigenvalue.imag,
                0.0,
                places=5,
                msg=f"Eigenvalue {eigenvalue} should have zero imaginary part",
            )

    def test_matrix_eigenvalues_sorting_magnitude(self):
        """Test eigenvalue sorting by magnitude"""
        # Matrix with eigenvalues of different magnitudes
        # Use a matrix that we can verify: [[3, 1], [0, 1]]
        # Eigenvalues are 3 and 1 (magnitudes 3 and 1)
        input_data = [3.0, 0.0, 1.0, 1.0]  # Column-major format

        # Test with sorting enabled (default)
        src = blocks.vector_source_f(input_data, False, 4)
        eigenvalues_block = matrix_eigenvalues_ff(
            [2, 2], True
        )  # sort_by_magnitude=True
        sink = blocks.vector_sink_c(2)

        self.tb.connect(src, eigenvalues_block)
        self.tb.connect(eigenvalues_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 2, "Should have 2 eigenvalues")

        # With sorting, larger magnitude should come first
        magnitudes = [abs(eigenvalue) for eigenvalue in result]
        self.assertGreaterEqual(
            magnitudes[0],
            magnitudes[1],
            "Eigenvalues should be sorted by magnitude (descending)",
        )

    def test_matrix_eigenvalues_3x3_numpy_comparison(self):
        """Test 3x3 matrix eigenvalues against NumPy"""
        # Test matrix: [[1, 2, 0], [0, 1, 3], [0, 0, 1]] (upper triangular)
        # Column-major format: [1, 0, 0, 2, 1, 0, 0, 3, 1]
        input_data = [1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 3.0, 1.0]

        src = blocks.vector_source_f(input_data, False, 9)
        eigenvalues_block = matrix_eigenvalues_ff([3, 3], False)  # Don't sort
        sink = blocks.vector_sink_c(3)

        self.tb.connect(src, eigenvalues_block)
        self.tb.connect(eigenvalues_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 3, "Should have 3 eigenvalues")

        # Compare with NumPy
        numpy_matrix = np.array([[1, 2, 0], [0, 1, 3], [0, 0, 1]], dtype=np.float32)
        numpy_eigenvalues = np.linalg.eigvals(numpy_matrix)

        # Sort both results for comparison (since order may vary)
        result_real = sorted([eigenvalue.real for eigenvalue in result])
        numpy_real = sorted(numpy_eigenvalues.real)

        for i in range(3):
            self.assertAlmostEqual(
                result_real[i],
                numpy_real[i],
                places=4,
                msg=f"Eigenvalue {i}: {result_real[i]} vs NumPy {numpy_real[i]}",
            )

    def test_matrix_eigenvalues_symmetric_matrix(self):
        """Test eigenvalues of symmetric matrix (should be real)"""
        # Symmetric matrix: [[4, 2], [2, 3]] in column-major format
        input_data = [4.0, 2.0, 2.0, 3.0]

        src = blocks.vector_source_f(input_data, False, 4)
        eigenvalues_block = matrix_eigenvalues_ff([2, 2])
        sink = blocks.vector_sink_c(2)

        self.tb.connect(src, eigenvalues_block)
        self.tb.connect(eigenvalues_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 2, "Should have 2 eigenvalues")

        # Symmetric matrix should have real eigenvalues
        for eigenvalue in result:
            self.assertAlmostEqual(
                eigenvalue.imag,
                0.0,
                places=5,
                msg=f"Symmetric matrix eigenvalue {eigenvalue} should be real",
            )

        # Verify eigenvalues are reasonable
        real_eigenvalues = [eigenvalue.real for eigenvalue in result]
        real_eigenvalues.sort()

        # For this symmetric matrix, eigenvalues should be positive
        for eigenval in real_eigenvalues:
            self.assertGreater(
                eigenval, 0, "Symmetric matrix eigenvalues should be positive"
            )

    def test_matrix_eigenvalues_error_handling(self):
        """Test error handling for invalid inputs"""
        # This test verifies the block handles edge cases gracefully

        # Test with zero matrix (should not crash)
        zero_input = [0.0, 0.0, 0.0, 0.0]

        src = blocks.vector_source_f(zero_input, False, 4)
        eigenvalues_block = matrix_eigenvalues_ff([2, 2])
        sink = blocks.vector_sink_c(2)

        self.tb.connect(src, eigenvalues_block)
        self.tb.connect(eigenvalues_block, sink)

        try:
            self.tb.run()
            result = sink.data()
            self.assertEqual(len(result), 2, "Should handle zero matrix")

            # All eigenvalues should be zero
            for eigenvalue in result:
                self.assertAlmostEqual(
                    abs(eigenvalue),
                    0.0,
                    places=5,
                    msg="Zero matrix should have zero eigenvalues",
                )
        except Exception as e:
            self.fail(f"Zero matrix test should not raise exception: {e}")

    def test_matrix_eigenvalues_factory_functions(self):
        """Test factory functions for different precisions"""
        # Test that all factory functions work
        try:
            block_f = matrix_eigenvalues_ff([2, 2])
            block_d = matrix_eigenvalues_dd([2, 2])
            block_c = matrix_eigenvalues_cc([2, 2])
            block_z = matrix_eigenvalues_zz([2, 2])

            self.assertIsNotNone(block_f, "Float factory function should work")
            self.assertIsNotNone(block_d, "Double factory function should work")
            self.assertIsNotNone(block_c, "Complex float factory function should work")
            self.assertIsNotNone(block_z, "Complex double factory function should work")

            # Test generic function
            generic_block = matrix_eigenvalues([2, 2])
            self.assertIsNotNone(generic_block, "Generic factory function should work")

        except Exception as e:
            self.fail(f"Factory function test failed: {e}")


if __name__ == "__main__":
    gr_unittest.run(qa_matrix_eigenvalues)
