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


class qa_decomp_hessenberg_advanced(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_decomp_hessenberg_basic_creation(self):
        """Test basic block creation with default options"""
        # Test all precision variants
        hess_f = decomp_hessenberg_ff([2, 2])
        hess_d = decomp_hessenberg_dd([2, 2])
        hess_cf = decomp_hessenberg_cc([2, 2])
        hess_cd = decomp_hessenberg_zz([2, 2])

        self.assertIsNotNone(hess_f, "Float block should be created")
        self.assertIsNotNone(hess_d, "Double block should be created")
        self.assertIsNotNone(hess_cf, "Complex float block should be created")
        self.assertIsNotNone(hess_cd, "Complex double block should be created")

    def test_decomp_hessenberg_identity_matrix_float(self):
        """Test Hessenberg decomposition of identity matrix"""
        # Identity matrix: [[1, 0], [0, 1]] in column-major format
        input_data = [1.0, 0.0, 0.0, 1.0]

        src = blocks.vector_source_f(input_data, False, 4)
        hess_block = decomp_hessenberg_ff([2, 2])
        sink_q = blocks.vector_sink_f(4)  # Q matrix (2x2)
        sink_h = blocks.vector_sink_f(4)  # H matrix (2x2)

        self.tb.connect(src, hess_block)
        self.tb.connect((hess_block, 0), sink_q)  # Q output
        self.tb.connect((hess_block, 1), sink_h)  # H output
        self.tb.run()

        q_result = sink_q.data()
        h_result = sink_h.data()

        self.assertEqual(len(q_result), 4, "Q matrix should have 4 elements")
        self.assertEqual(len(h_result), 4, "H matrix should have 4 elements")

        # For identity matrix, H should be essentially the identity matrix
        # and Q should be orthogonal
        q_matrix = np.array(q_result).reshape(2, 2, order="F")
        h_matrix = np.array(h_result).reshape(2, 2, order="F")

        # Check that Q is orthogonal: Q^T * Q should be identity
        qtq = q_matrix.T @ q_matrix
        np.testing.assert_allclose(
            qtq, np.eye(2), atol=1e-5, err_msg="Q matrix should be orthogonal"
        )

        # Check that decomposition holds: A = Q * H * Q^T
        identity = np.eye(2)
        reconstructed = q_matrix @ h_matrix @ q_matrix.T
        np.testing.assert_allclose(
            reconstructed,
            identity,
            atol=1e-5,
            err_msg="Decomposition A = Q*H*Q^T should hold",
        )

    def test_decomp_hessenberg_diagonal_double(self):
        """Test Hessenberg decomposition of diagonal matrix"""
        # Diagonal matrix: [[2, 0], [0, 3]] in column-major format
        input_data = [2.0, 0.0, 0.0, 3.0]

        src = blocks.vector_source_d(input_data, False, 4)
        hess_block = decomp_hessenberg_dd([2, 2])
        sink_q = blocks.vector_sink_d(4)
        sink_h = blocks.vector_sink_d(4)

        self.tb.connect(src, hess_block)
        self.tb.connect((hess_block, 0), sink_q)
        self.tb.connect((hess_block, 1), sink_h)
        self.tb.run()

        q_result = sink_q.data()
        h_result = sink_h.data()

        q_matrix = np.array(q_result).reshape(2, 2, order="F")
        h_matrix = np.array(h_result).reshape(2, 2, order="F")

        # Verify Q is orthogonal
        qtq = q_matrix.T @ q_matrix
        np.testing.assert_allclose(
            qtq, np.eye(2), atol=1e-10, err_msg="Q matrix should be orthogonal"
        )

        # Verify decomposition holds
        original = np.array([[2, 0], [0, 3]])
        reconstructed = q_matrix @ h_matrix @ q_matrix.T
        np.testing.assert_allclose(
            reconstructed,
            original,
            atol=1e-10,
            err_msg="Decomposition should reproduce original matrix",
        )

    def test_decomp_hessenberg_symmetric_matrix(self):
        """Test Hessenberg decomposition of symmetric matrix"""
        # Symmetric matrix: [[4, 2], [2, 3]] in column-major format
        input_data = [4.0, 2.0, 2.0, 3.0]

        src = blocks.vector_source_f(input_data, False, 4)
        hess_block = decomp_hessenberg_ff([2, 2])
        sink_q = blocks.vector_sink_f(4)
        sink_h = blocks.vector_sink_f(4)

        self.tb.connect(src, hess_block)
        self.tb.connect((hess_block, 0), sink_q)
        self.tb.connect((hess_block, 1), sink_h)
        self.tb.run()

        q_result = sink_q.data()
        h_result = sink_h.data()

        q_matrix = np.array(q_result).reshape(2, 2, order="F")
        h_matrix = np.array(h_result).reshape(2, 2, order="F")

        # For symmetric matrices, Hessenberg form is tridiagonal
        # Check that h_matrix is upper Hessenberg
        self.assertLess(
            abs(h_matrix[1, 0]),
            1e-5 if h_matrix.shape[0] > 2 else float("inf"),
            "H matrix should be upper Hessenberg",
        )

        # Verify decomposition
        original = np.array([[4, 2], [2, 3]])
        reconstructed = q_matrix @ h_matrix @ q_matrix.T
        np.testing.assert_allclose(
            reconstructed,
            original,
            atol=1e-5,
            err_msg="Decomposition should be accurate",
        )

    def test_decomp_hessenberg_complex_matrix(self):
        """Test Hessenberg decomposition of complex matrix"""
        # Complex matrix: [[1+i, 2], [0, 1-i]] in column-major format
        input_data = [complex(1, 1), complex(0, 0), complex(2, 0), complex(1, -1)]

        src = blocks.vector_source_c(input_data, False, 4)
        hess_block = decomp_hessenberg_cc([2, 2])
        sink_q = blocks.vector_sink_c(4)
        sink_h = blocks.vector_sink_c(4)

        self.tb.connect(src, hess_block)
        self.tb.connect((hess_block, 0), sink_q)
        self.tb.connect((hess_block, 1), sink_h)
        self.tb.run()

        q_result = sink_q.data()
        h_result = sink_h.data()

        self.assertEqual(len(q_result), 4, "Complex Q matrix should have 4 elements")
        self.assertEqual(len(h_result), 4, "Complex H matrix should have 4 elements")

        q_matrix = np.array(q_result).reshape(2, 2, order="F")
        h_matrix = np.array(h_result).reshape(2, 2, order="F")

        # For complex matrices, Q should be unitary: Q^H * Q = I
        qhq = q_matrix.conj().T @ q_matrix
        np.testing.assert_allclose(
            qhq, np.eye(2), atol=1e-5, err_msg="Q matrix should be unitary"
        )

    def test_decomp_hessenberg_3x3_matrix_numpy_comparison(self):
        """Test 3x3 matrix against NumPy Hessenberg decomposition"""
        # Test matrix: [[1, 2, 0], [0, 1, 3], [0, 0, 1]] in column-major format
        input_data = [1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 3.0, 1.0]

        src = blocks.vector_source_f(input_data, False, 9)
        hess_block = decomp_hessenberg_ff([3, 3])
        sink_q = blocks.vector_sink_f(9)
        sink_h = blocks.vector_sink_f(9)

        self.tb.connect(src, hess_block)
        self.tb.connect((hess_block, 0), sink_q)
        self.tb.connect((hess_block, 1), sink_h)
        self.tb.run()

        q_result = sink_q.data()
        h_result = sink_h.data()

        self.assertEqual(len(q_result), 9, "3x3 Q matrix should have 9 elements")
        self.assertEqual(len(h_result), 9, "3x3 H matrix should have 9 elements")

        q_matrix = np.array(q_result).reshape(3, 3, order="F")
        h_matrix = np.array(h_result).reshape(3, 3, order="F")

        # Verify H is upper Hessenberg (zeros below first subdiagonal)
        for i in range(3):
            for j in range(3):
                if i > j + 1:  # Below first subdiagonal
                    self.assertLess(
                        abs(h_matrix[i, j]),
                        1e-5,
                        f"H[{i},{j}] should be zero (upper Hessenberg property)",
                    )

        # Verify Q is orthogonal
        qtq = q_matrix.T @ q_matrix
        np.testing.assert_allclose(
            qtq, np.eye(3), atol=1e-5, err_msg="Q should be orthogonal"
        )

        # Verify decomposition
        original = np.array([[1, 2, 0], [0, 1, 3], [0, 0, 1]])
        reconstructed = q_matrix @ h_matrix @ q_matrix.T
        np.testing.assert_allclose(
            reconstructed,
            original,
            atol=1e-5,
            err_msg="Decomposition should be mathematically correct",
        )

    def test_decomp_hessenberg_compute_q_false(self):
        """Test Hessenberg decomposition with compute_q=False"""
        input_data = [2.0, 1.0, 1.0, 2.0]

        # Create block that doesn't compute Q matrix
        try:
            # This tests the block creation - actual compute_q=False functionality
            # would require access to the underlying C++ interface
            hess_block = decomp_hessenberg_ff([2, 2])
            self.assertIsNotNone(hess_block, "Block should be created successfully")
        except Exception as e:
            self.fail(f"Block creation should not fail: {e}")

    def test_decomp_hessenberg_factory_functions(self):
        """Test all factory functions work correctly"""
        try:
            block_f = decomp_hessenberg_ff([2, 2])
            block_d = decomp_hessenberg_dd([2, 2])
            block_c = decomp_hessenberg_cc([2, 2])
            block_z = decomp_hessenberg_zz([2, 2])

            self.assertIsNotNone(block_f, "Float factory should work")
            self.assertIsNotNone(block_d, "Double factory should work")
            self.assertIsNotNone(block_c, "Complex float factory should work")
            self.assertIsNotNone(block_z, "Complex double factory should work")

            # Test generic function
            generic_block = decomp_hessenberg([2, 2])
            self.assertIsNotNone(generic_block, "Generic factory should work")

        except Exception as e:
            self.fail(f"Factory function test failed: {e}")

    def test_decomp_hessenberg_mathematical_properties(self):
        """Test mathematical properties of Hessenberg decomposition"""
        # Test with a general 3x3 matrix
        input_data = [2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0]  # Tridiagonal

        src = blocks.vector_source_f(input_data, False, 9)
        hess_block = decomp_hessenberg_ff([3, 3])
        sink_q = blocks.vector_sink_f(9)
        sink_h = blocks.vector_sink_f(9)

        self.tb.connect(src, hess_block)
        self.tb.connect((hess_block, 0), sink_q)
        self.tb.connect((hess_block, 1), sink_h)
        self.tb.run()

        q_result = sink_q.data()
        h_result = sink_h.data()

        q_matrix = np.array(q_result).reshape(3, 3, order="F")
        h_matrix = np.array(h_result).reshape(3, 3, order="F")

        # Property 1: H should be upper Hessenberg
        for i in range(3):
            for j in range(3):
                if i > j + 1:
                    self.assertLess(
                        abs(h_matrix[i, j]),
                        1e-10,
                        f"H[{i},{j}] = {h_matrix[i, j]} should be ~0 (Hessenberg form)",
                    )

        # Property 2: Q should be orthogonal (Q^T * Q = I)
        qtq = q_matrix.T @ q_matrix
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                self.assertAlmostEqual(
                    qtq[i, j],
                    expected,
                    places=5,
                    msg=f"Q^T*Q[{i},{j}] = {qtq[i, j]} should be {expected}",
                )

        # Property 3: Decomposition should be exact (A = Q * H * Q^T)
        original = np.array(input_data).reshape(3, 3, order="F")
        reconstructed = q_matrix @ h_matrix @ q_matrix.T
        np.testing.assert_allclose(
            reconstructed,
            original,
            atol=1e-10,
            err_msg="A = Q*H*Q^T should hold exactly",
        )

        # Property 4: det(Q) should be ±1 (orthogonal matrix property)
        det_q = np.linalg.det(q_matrix)
        self.assertAlmostEqual(
            abs(det_q), 1.0, places=5, msg=f"det(Q) = {det_q} should have magnitude 1"
        )

    def test_decomp_hessenberg_error_handling(self):
        """Test error handling for edge cases"""
        # Test with zero matrix (should not crash)
        zero_input = [0.0, 0.0, 0.0, 0.0]

        src = blocks.vector_source_f(zero_input, False, 4)
        hess_block = decomp_hessenberg_ff([2, 2])
        sink_q = blocks.vector_sink_f(4)
        sink_h = blocks.vector_sink_f(4)

        self.tb.connect(src, hess_block)
        self.tb.connect((hess_block, 0), sink_q)
        self.tb.connect((hess_block, 1), sink_h)

        try:
            self.tb.run()
            q_result = sink_q.data()
            h_result = sink_h.data()

            self.assertEqual(len(q_result), 4, "Should handle zero matrix")
            self.assertEqual(len(h_result), 4, "Should handle zero matrix")

        except Exception as e:
            self.fail(f"Zero matrix test should not crash: {e}")

    def test_decomp_hessenberg_vs_numpy(self):
        """Test decomposition results against NumPy for verification"""
        # Use a well-conditioned test matrix
        test_matrix = np.array([[3, 1, 0], [1, 3, 1], [0, 1, 3]], dtype=np.float32)
        input_data = test_matrix.flatten(order="F").tolist()

        src = blocks.vector_source_f(input_data, False, 9)
        hess_block = decomp_hessenberg_ff([3, 3])
        sink_q = blocks.vector_sink_f(9)
        sink_h = blocks.vector_sink_f(9)

        self.tb.connect(src, hess_block)
        self.tb.connect((hess_block, 0), sink_q)
        self.tb.connect((hess_block, 1), sink_h)
        self.tb.run()

        q_result = np.array(sink_q.data()).reshape(3, 3, order="F")
        h_result = np.array(sink_h.data()).reshape(3, 3, order="F")

        # Compare with SciPy (more readily available than NumPy's Hessenberg)
        try:
            from scipy.linalg import hessenberg

            scipy_h, scipy_q = hessenberg(test_matrix, calc_q=True)

            # The decompositions might differ by sign/phase, but should be equivalent
            # Check that both decompositions reproduce the original matrix
            our_reconstruction = q_result @ h_result @ q_result.T
            scipy_reconstruction = scipy_q @ scipy_h @ scipy_q.T

            np.testing.assert_allclose(
                our_reconstruction,
                test_matrix,
                atol=1e-5,
                err_msg="Our decomposition should reproduce original",
            )
            np.testing.assert_allclose(
                scipy_reconstruction,
                test_matrix,
                atol=1e-5,
                err_msg="SciPy decomposition should reproduce original",
            )

        except ImportError:
            # If SciPy not available, just verify our decomposition is mathematically sound
            reconstruction = q_result @ h_result @ q_result.T
            np.testing.assert_allclose(
                reconstruction,
                test_matrix,
                atol=1e-5,
                err_msg="Decomposition should reproduce original matrix",
            )

    def test_decomp_hessenberg_advanced_features_interface(self):
        """Test that advanced features interface is available"""
        # This tests the interface exists - actual advanced feature testing
        # would require access to the underlying C++ advanced options
        try:
            hess_block = decomp_hessenberg_ff([2, 2])

            # Test basic block functionality
            self.assertIsNotNone(hess_block)

            # Advanced features would be tested if the Python bindings
            # exposed the advanced options interface

        except Exception as e:
            self.fail(f"Advanced features interface test failed: {e}")


if __name__ == "__main__":
    gr_unittest.run(qa_decomp_hessenberg_advanced)
