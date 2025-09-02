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


class qa_decomp_schur_advanced(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_decomp_schur_basic_creation(self):
        """Test basic Schur decomposition block creation"""
        try:
            schur_block = decomp_schur_ff([2, 2])
            self.assertIsNotNone(schur_block, "Basic Schur block creation should work")

            # Test different precisions
            schur_d = decomp_schur_dd([3, 3])
            schur_c = decomp_schur_cc([2, 2])
            schur_z = decomp_schur_zz([4, 4])

            self.assertIsNotNone(schur_d, "Double precision should work")
            self.assertIsNotNone(schur_c, "Complex float should work")
            self.assertIsNotNone(schur_z, "Complex double should work")

        except Exception as e:
            self.fail(f"Basic Schur block creation failed: {e}")

    def test_decomp_schur_identity_matrix(self):
        """Test Schur decomposition of identity matrix"""
        # Identity matrix: [[1, 0], [0, 1]] in column-major format
        input_data = [1.0, 0.0, 0.0, 1.0]

        src = blocks.vector_source_f(input_data, False, 4)
        schur_block = decomp_schur_ff([2, 2])
        sink_q = blocks.vector_sink_f(4)  # Q matrix (2x2)
        sink_t = blocks.vector_sink_f(4)  # T matrix (2x2)

        self.tb.connect(src, schur_block)
        self.tb.connect((schur_block, 0), sink_q)  # Q matrix
        self.tb.connect((schur_block, 1), sink_t)  # T matrix
        self.tb.run()

        q_result = sink_q.data()
        t_result = sink_t.data()

        self.assertEqual(len(q_result), 4, "Q matrix should have 4 elements")
        self.assertEqual(len(t_result), 4, "T matrix should have 4 elements")

        # For identity matrix, T should be identity (or close to it)
        # and Q should be orthogonal
        q_matrix = np.array(q_result).reshape(2, 2, order="F")  # Column-major
        t_matrix = np.array(t_result).reshape(2, 2, order="F")

        # Check Q is orthogonal: Q^T * Q should be identity
        qtq = np.dot(q_matrix.T, q_matrix)
        np.testing.assert_allclose(
            qtq, np.eye(2), rtol=1e-5, atol=1e-5, err_msg="Q should be orthogonal"
        )

    def test_decomp_schur_diagonal_matrix(self):
        """Test Schur decomposition of diagonal matrix"""
        # Diagonal matrix: [[2, 0], [0, 3]] in column-major format
        input_data = [2.0, 0.0, 0.0, 3.0]

        src = blocks.vector_source_f(input_data, False, 4)
        schur_block = decomp_schur_ff([2, 2])
        sink_q = blocks.vector_sink_f(4)
        sink_t = blocks.vector_sink_f(4)

        self.tb.connect(src, schur_block)
        self.tb.connect((schur_block, 0), sink_q)
        self.tb.connect((schur_block, 1), sink_t)
        self.tb.run()

        q_result = sink_q.data()
        t_result = sink_t.data()

        q_matrix = np.array(q_result).reshape(2, 2, order="F")
        t_matrix = np.array(t_result).reshape(2, 2, order="F")
        input_matrix = np.array([[2, 0], [0, 3]], dtype=np.float32)

        # Verify decomposition: A = Q * T * Q^T
        reconstructed = np.dot(q_matrix, np.dot(t_matrix, q_matrix.T))
        np.testing.assert_allclose(
            reconstructed,
            input_matrix,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Decomposition should reconstruct original matrix",
        )

    def test_decomp_schur_complex_matrix(self):
        """Test Schur decomposition of complex matrix"""
        # Complex matrix: [[1+i, 0], [0, 1-i]] in column-major format
        input_data = [complex(1, 1), complex(0, 0), complex(0, 0), complex(1, -1)]

        src = blocks.vector_source_c(input_data, False, 4)
        schur_block = decomp_schur_cc([2, 2])
        sink_q = blocks.vector_sink_c(4)
        sink_t = blocks.vector_sink_c(4)

        self.tb.connect(src, schur_block)
        self.tb.connect((schur_block, 0), sink_q)
        self.tb.connect((schur_block, 1), sink_t)
        self.tb.run()

        q_result = sink_q.data()
        t_result = sink_t.data()

        self.assertEqual(len(q_result), 4, "Complex Q matrix should have 4 elements")
        self.assertEqual(len(t_result), 4, "Complex T matrix should have 4 elements")

        q_matrix = np.array(q_result).reshape(2, 2, order="F")
        t_matrix = np.array(t_result).reshape(2, 2, order="F")

        # Check Q is unitary: Q^H * Q should be identity
        qhq = np.dot(q_matrix.conj().T, q_matrix)
        np.testing.assert_allclose(
            qhq,
            np.eye(2),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Q should be unitary for complex matrices",
        )

    def test_decomp_schur_double_precision(self):
        """Test Schur decomposition with double precision"""
        # Test matrix: [[4, 1], [0, 3]] (upper triangular)
        input_data = [4.0, 0.0, 1.0, 3.0]  # Column-major format

        src = blocks.vector_source_d(input_data, False, 4)
        schur_block = decomp_schur_dd([2, 2])
        sink_q = blocks.vector_sink_d(4)
        sink_t = blocks.vector_sink_d(4)

        self.tb.connect(src, schur_block)
        self.tb.connect((schur_block, 0), sink_q)
        self.tb.connect((schur_block, 1), sink_t)
        self.tb.run()

        q_result = sink_q.data()
        t_result = sink_t.data()

        q_matrix = np.array(q_result, dtype=np.float64).reshape(2, 2, order="F")
        t_matrix = np.array(t_result, dtype=np.float64).reshape(2, 2, order="F")
        input_matrix = np.array([[4, 1], [0, 3]], dtype=np.float64)

        # Verify decomposition
        reconstructed = np.dot(q_matrix, np.dot(t_matrix, q_matrix.T))
        np.testing.assert_allclose(
            reconstructed,
            input_matrix,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Double precision decomposition should be accurate",
        )

    def test_decomp_schur_3x3_matrix(self):
        """Test Schur decomposition of 3x3 matrix"""
        # Test with a 3x3 upper triangular matrix
        # [[2, 1, 0], [0, 2, 1], [0, 0, 2]] in column-major format
        input_data = [2.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 1.0, 2.0]

        src = blocks.vector_source_f(input_data, False, 9)
        schur_block = decomp_schur_ff([3, 3])
        sink_q = blocks.vector_sink_f(9)
        sink_t = blocks.vector_sink_f(9)

        self.tb.connect(src, schur_block)
        self.tb.connect((schur_block, 0), sink_q)
        self.tb.connect((schur_block, 1), sink_t)
        self.tb.run()

        q_result = sink_q.data()
        t_result = sink_t.data()

        self.assertEqual(len(q_result), 9, "3x3 Q matrix should have 9 elements")
        self.assertEqual(len(t_result), 9, "3x3 T matrix should have 9 elements")

        q_matrix = np.array(q_result).reshape(3, 3, order="F")
        t_matrix = np.array(t_result).reshape(3, 3, order="F")

        # Check Q is orthogonal
        qtq = np.dot(q_matrix.T, q_matrix)
        np.testing.assert_allclose(
            qtq,
            np.eye(3),
            rtol=1e-4,
            atol=1e-4,
            err_msg="3x3 Q matrix should be orthogonal",
        )

    def test_decomp_schur_symmetric_matrix(self):
        """Test Schur decomposition of symmetric matrix"""
        # Symmetric matrix: [[3, 1], [1, 3]] in column-major format
        input_data = [3.0, 1.0, 1.0, 3.0]

        src = blocks.vector_source_f(input_data, False, 4)
        schur_block = decomp_schur_ff([2, 2])
        sink_q = blocks.vector_sink_f(4)
        sink_t = blocks.vector_sink_f(4)

        self.tb.connect(src, schur_block)
        self.tb.connect((schur_block, 0), sink_q)
        self.tb.connect((schur_block, 1), sink_t)
        self.tb.run()

        q_result = sink_q.data()
        t_result = sink_t.data()

        q_matrix = np.array(q_result).reshape(2, 2, order="F")
        t_matrix = np.array(t_result).reshape(2, 2, order="F")
        input_matrix = np.array([[3, 1], [1, 3]], dtype=np.float32)

        # Verify decomposition
        reconstructed = np.dot(q_matrix, np.dot(t_matrix, q_matrix.T))
        np.testing.assert_allclose(
            reconstructed,
            input_matrix,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Symmetric matrix decomposition should be accurate",
        )

        # For real symmetric matrices, T should be diagonal (or block diagonal)
        # and contain the eigenvalues
        eigenvalues_t = np.diag(t_matrix)
        eigenvalues_numpy = np.linalg.eigvals(input_matrix)
        eigenvalues_t_sorted = np.sort(eigenvalues_t)
        eigenvalues_numpy_sorted = np.sort(eigenvalues_numpy)

        np.testing.assert_allclose(
            eigenvalues_t_sorted,
            eigenvalues_numpy_sorted,
            rtol=1e-4,
            atol=1e-4,
            err_msg="T matrix diagonal should contain eigenvalues",
        )

    def test_decomp_schur_numpy_comparison(self):
        """Test Schur decomposition against NumPy/SciPy"""
        # Test matrix
        input_matrix = np.array([[2, 1, 0], [0, 2, 1], [1, 0, 2]], dtype=np.float32)
        input_data = input_matrix.flatten(order="F").tolist()  # Column-major

        src = blocks.vector_source_f(input_data, False, 9)
        schur_block = decomp_schur_ff([3, 3])
        sink_q = blocks.vector_sink_f(9)
        sink_t = blocks.vector_sink_f(9)

        self.tb.connect(src, schur_block)
        self.tb.connect((schur_block, 0), sink_q)
        self.tb.connect((schur_block, 1), sink_t)
        self.tb.run()

        q_result = sink_q.data()
        t_result = sink_t.data()

        q_matrix = np.array(q_result).reshape(3, 3, order="F")
        t_matrix = np.array(t_result).reshape(3, 3, order="F")

        # Verify basic properties
        # 1. Q is orthogonal
        qtq = np.dot(q_matrix.T, q_matrix)
        np.testing.assert_allclose(
            qtq, np.eye(3), rtol=1e-4, atol=1e-4, err_msg="Q should be orthogonal"
        )

        # 2. A = Q * T * Q^T
        reconstructed = np.dot(q_matrix, np.dot(t_matrix, q_matrix.T))
        np.testing.assert_allclose(
            reconstructed,
            input_matrix,
            rtol=1e-4,
            atol=1e-4,
            err_msg="Decomposition should reconstruct original",
        )

        # 3. T should be quasi-upper triangular (for real Schur form)
        # Check that T has zeros below the first subdiagonal
        for i in range(3):
            for j in range(3):
                if i > j + 1:  # Below first subdiagonal
                    self.assertAlmostEqual(
                        t_matrix[i, j], 0.0, places=4, msg=f"T[{i},{j}] should be zero"
                    )

    def test_decomp_schur_error_handling(self):
        """Test error handling for edge cases"""
        # Test with near-singular matrix (should not crash)
        near_singular = [1.0, 1.0, 1.0, 1.001]  # Almost singular

        src = blocks.vector_source_f(near_singular, False, 4)
        schur_block = decomp_schur_ff([2, 2])
        sink_q = blocks.vector_sink_f(4)
        sink_t = blocks.vector_sink_f(4)

        self.tb.connect(src, schur_block)
        self.tb.connect((schur_block, 0), sink_q)
        self.tb.connect((schur_block, 1), sink_t)

        try:
            self.tb.run()
            q_result = sink_q.data()
            t_result = sink_t.data()

            self.assertEqual(len(q_result), 4, "Should handle near-singular matrix")
            self.assertEqual(len(t_result), 4, "Should handle near-singular matrix")
        except Exception as e:
            self.fail(f"Near-singular matrix test should not crash: {e}")

    def test_decomp_schur_factory_functions(self):
        """Test factory functions for different precisions"""
        try:
            # Test precision variants
            block_f = decomp_schur_ff([2, 2])
            block_d = decomp_schur_dd([2, 2])
            block_c = decomp_schur_cc([2, 2])
            block_z = decomp_schur_zz([2, 2])

            self.assertIsNotNone(block_f, "Float factory should work")
            self.assertIsNotNone(block_d, "Double factory should work")
            self.assertIsNotNone(block_c, "Complex float factory should work")
            self.assertIsNotNone(block_z, "Complex double factory should work")

            # Test generic function
            generic_block = decomp_schur([3, 3])
            self.assertIsNotNone(generic_block, "Generic factory should work")

        except Exception as e:
            self.fail(f"Factory function test failed: {e}")

    def test_decomp_schur_compute_q_disabled(self):
        """Test Schur decomposition with Q computation disabled"""
        # This test would require access to the compute_u parameter
        # For now, just test that the basic functionality works
        input_data = [2.0, 1.0, 1.0, 2.0]

        src = blocks.vector_source_f(input_data, False, 4)
        schur_block = decomp_schur_ff([2, 2])
        sink_q = blocks.vector_sink_f(4)
        sink_t = blocks.vector_sink_f(4)

        self.tb.connect(src, schur_block)
        self.tb.connect((schur_block, 0), sink_q)
        self.tb.connect((schur_block, 1), sink_t)
        self.tb.run()

        q_result = sink_q.data()
        t_result = sink_t.data()

        self.assertEqual(len(q_result), 4, "Q output should be present")
        self.assertEqual(len(t_result), 4, "T output should be present")

    def test_decomp_schur_mathematical_properties(self):
        """Test mathematical properties of Schur decomposition"""
        # Use a well-conditioned test matrix
        test_matrix = np.array([[4, 1, 2], [0, 3, 1], [0, 0, 2]], dtype=np.float32)
        input_data = test_matrix.flatten(order="F").tolist()

        src = blocks.vector_source_f(input_data, False, 9)
        schur_block = decomp_schur_ff([3, 3])
        sink_q = blocks.vector_sink_f(9)
        sink_t = blocks.vector_sink_f(9)

        self.tb.connect(src, schur_block)
        self.tb.connect((schur_block, 0), sink_q)
        self.tb.connect((schur_block, 1), sink_t)
        self.tb.run()

        q_result = sink_q.data()
        t_result = sink_t.data()

        q_matrix = np.array(q_result).reshape(3, 3, order="F")
        t_matrix = np.array(t_result).reshape(3, 3, order="F")

        # Property 1: Q is orthogonal (Q^T * Q = I)
        qtq = np.dot(q_matrix.T, q_matrix)
        np.testing.assert_allclose(qtq, np.eye(3), rtol=1e-5, atol=1e-5)

        # Property 2: det(Q) = ±1 (orthogonal matrices have unit determinant)
        det_q = np.linalg.det(q_matrix)
        self.assertAlmostEqual(
            abs(det_q),
            1.0,
            places=4,
            msg="Orthogonal matrix should have unit determinant",
        )

        # Property 3: A = Q * T * Q^T
        reconstructed = np.dot(q_matrix, np.dot(t_matrix, q_matrix.T))
        np.testing.assert_allclose(reconstructed, test_matrix, rtol=1e-5, atol=1e-5)

        # Property 4: T is quasi-upper triangular
        for i in range(3):
            for j in range(3):
                if i > j + 1:  # Below first subdiagonal
                    self.assertAlmostEqual(
                        t_matrix[i, j], 0.0, places=4, msg=f"T[{i},{j}] should be zero"
                    )

        # Property 5: Trace(A) = Trace(T) (similarity transformation preserves trace)
        trace_a = np.trace(test_matrix)
        trace_t = np.trace(t_matrix)
        self.assertAlmostEqual(
            trace_a, trace_t, places=4, msg="Trace should be preserved in Schur form"
        )


if __name__ == "__main__":
    gr_unittest.run(qa_decomp_schur_advanced)
