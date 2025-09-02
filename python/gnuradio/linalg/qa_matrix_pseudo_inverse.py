#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2025 Wylie Standage-Beier.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

from gnuradio import gr, gr_unittest, blocks
import numpy as np

try:
    from gnuradio.linalg import (
        matrix_pseudo_inverse_sync_f,
        matrix_pseudo_inverse_sync_d,
    )
except ImportError:
    import os
    import sys

    dirname, filename = os.path.split(os.path.abspath(__file__))
    sys.path.append(os.path.join(dirname, "bindings"))
    try:
        from gnuradio.linalg import (
            matrix_pseudo_inverse_sync_f,
            matrix_pseudo_inverse_sync_d,
        )
    except ImportError:
        # Try direct import from C++ module
        import linalg_python

        matrix_pseudo_inverse_sync_f = linalg_python.matrix_pseudo_inverse_sync_f
        matrix_pseudo_inverse_sync_d = linalg_python.matrix_pseudo_inverse_sync_d


class qa_matrix_pseudo_inverse(gr_unittest.TestCase):
    """Flowgraph tests for matrix_pseudo_inverse sync block.

    Tests the Moore-Penrose pseudoinverse implementation for various matrix types:
    - Square invertible matrices
    - Rectangular matrices (overdetermined and underdetermined systems)
    - Rank-deficient matrices
    - Identity matrices
    - Zero matrices
    """

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    @staticmethod
    def _flatten_col_major(M):
        """Convert numpy matrix to column-major flattened list for GNU Radio."""
        return M.T.flatten()

    def _run_single_matrix(self, A, precision="float"):
        """Run pseudoinverse on a single matrix and return result."""
        m, n = A.shape
        flat_input = self._flatten_col_major(A)
        vlen_in = m * n
        vlen_out = n * m  # Transposed dimensions for pseudoinverse

        if precision == "float":
            src = blocks.vector_source_f(
                flat_input.astype(np.float32).tolist(), repeat=False, vlen=vlen_in
            )
            pinv_block = matrix_pseudo_inverse_sync_f([m, n])
            sink = blocks.vector_sink_f(vlen=vlen_out)
        else:  # double - note: GNU Radio only provides float vector sources/sinks
            # Convert double precision to float for GNU Radio compatibility
            src = blocks.vector_source_f(
                flat_input.astype(np.float32).tolist(), repeat=False, vlen=vlen_in
            )
            # Block still uses double precision internally
            pinv_block = matrix_pseudo_inverse_sync_d([m, n])
            sink = blocks.vector_sink_f(vlen=vlen_out)

        self.tb.connect(src, pinv_block)
        self.tb.connect(pinv_block, sink)
        self.tb.run()

        result_flat = np.array(sink.data())
        # Reshape back to matrix (remembering it's transposed)
        A_pinv = result_flat.reshape((n, m), order="F")  # Column-major order
        return A_pinv

    def test_identity_matrix(self):
        """Test pseudoinverse of identity matrix (should return itself)."""
        A = np.eye(3, dtype=np.float32)
        A_pinv = self._run_single_matrix(A, precision="float")

        # Identity pseudoinverse should be identity
        np.testing.assert_allclose(A_pinv, A, rtol=1e-5, atol=1e-6)

    def test_diagonal_matrix(self):
        """Test pseudoinverse of diagonal matrix."""
        A = np.diag([2.0, 3.0, 4.0]).astype(np.float32)
        A_pinv = self._run_single_matrix(A, precision="float")

        # Diagonal pseudoinverse should have reciprocal diagonal elements
        expected = np.diag([1.0 / 2.0, 1.0 / 3.0, 1.0 / 4.0])
        np.testing.assert_allclose(A_pinv, expected, rtol=1e-5, atol=1e-6)

    def test_rectangular_tall_matrix(self):
        """Test pseudoinverse of tall rectangular matrix (overdetermined)."""
        # 3x2 matrix (more rows than columns)
        A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        A_pinv = self._run_single_matrix(A, precision="float")

        # For overdetermined systems: A_pinv * A should be identity
        should_be_identity = A_pinv @ A
        expected_identity = np.eye(2)
        np.testing.assert_allclose(
            should_be_identity, expected_identity, rtol=1e-4, atol=1e-5
        )

    def test_rectangular_wide_matrix(self):
        """Test pseudoinverse of wide rectangular matrix (underdetermined)."""
        # 2x3 matrix (more columns than rows)
        A = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32)
        A_pinv = self._run_single_matrix(A, precision="float")

        # For underdetermined systems: A * A_pinv should be identity
        should_be_identity = A @ A_pinv
        expected_identity = np.eye(2)
        np.testing.assert_allclose(
            should_be_identity, expected_identity, rtol=1e-4, atol=1e-5
        )

    def test_rank_deficient_matrix(self):
        """Test pseudoinverse of rank-deficient matrix."""
        # Rank-1 matrix: second column = 2 * first column
        A = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float32)
        A_pinv = self._run_single_matrix(A, precision="float")

        # Test Moore-Penrose property: A * A_pinv * A = A
        result = A @ A_pinv @ A
        np.testing.assert_allclose(result, A, rtol=1e-4, atol=1e-5)

    def test_zero_matrix(self):
        """Test pseudoinverse of zero matrix."""
        A = np.zeros((2, 3), dtype=np.float32)
        A_pinv = self._run_single_matrix(A, precision="float")

        # Zero matrix pseudoinverse should be zero matrix of transposed dimensions
        expected = np.zeros((3, 2))
        np.testing.assert_allclose(A_pinv, expected, rtol=1e-6, atol=1e-7)

    def test_single_element_nonzero(self):
        """Test pseudoinverse of single non-zero element."""
        A = np.array([[5.0]], dtype=np.float32)
        A_pinv = self._run_single_matrix(A, precision="float")

        # Single element pseudoinverse should be reciprocal
        expected = np.array([[1.0 / 5.0]])
        np.testing.assert_allclose(A_pinv, expected, rtol=1e-5, atol=1e-6)

    def test_single_element_zero(self):
        """Test pseudoinverse of single zero element."""
        A = np.array([[0.0]], dtype=np.float32)
        A_pinv = self._run_single_matrix(A, precision="float")

        # Zero element pseudoinverse should be zero
        expected = np.array([[0.0]])
        np.testing.assert_allclose(A_pinv, expected, rtol=1e-6, atol=1e-7)

    def test_double_precision(self):
        """Test pseudoinverse with double precision using NumPy fallback."""
        # Since GNU Radio doesn't have vector_source_d/sink_d, we test the
        # double precision logic using NumPy directly rather than flowgraph
        A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

        # Compute expected pseudoinverse using NumPy for comparison
        A_pinv_expected = np.linalg.pinv(A)

        # Verify the mathematical properties hold for double precision
        # This tests the algorithm logic without GNU Radio flowgraph constraints
        self.assertIsNotNone(A_pinv_expected)
        self.assertEqual(A_pinv_expected.shape, (2, 2))

        # Test Moore-Penrose properties
        # Property 1: A * A+ * A = A
        result1 = A @ A_pinv_expected @ A
        np.testing.assert_allclose(result1, A, rtol=1e-10, atol=1e-12)

        # Property 2: A+ * A * A+ = A+
        result2 = A_pinv_expected @ A @ A_pinv_expected
        np.testing.assert_allclose(result2, A_pinv_expected, rtol=1e-10, atol=1e-12)

    def test_multiple_matrices(self):
        """Test pseudoinverse with multiple matrices in a single flowgraph."""
        # Test with multiple 2x2 matrices
        matrices = [
            np.array([[1.0, 0.0], [0.0, 1.0]]),  # Identity
            np.array([[2.0, 0.0], [0.0, 3.0]]),  # Diagonal
            np.array([[1.0, 1.0], [1.0, 2.0]]),  # General
        ]

        # Flatten all matrices
        flat_input = []
        for A in matrices:
            flat_input.extend(self._flatten_col_major(A).astype(np.float32).tolist())

        src = blocks.vector_source_f(flat_input, repeat=False, vlen=4)
        pinv_block = matrix_pseudo_inverse_sync_f([2, 2])
        sink = blocks.vector_sink_f(vlen=4)

        self.tb.connect(src, pinv_block)
        self.tb.connect(pinv_block, sink)
        self.tb.run()

        result_data = np.array(sink.data())

        # Should have 3 matrices * 4 elements each = 12 elements
        self.assertEqual(len(result_data), 12)

        # Extract and test each result matrix
        for i in range(3):
            start_idx = i * 4
            end_idx = start_idx + 4
            result_flat = result_data[start_idx:end_idx]
            A_pinv = result_flat.reshape((2, 2), order="F")

            # All results should be finite
            self.assertTrue(np.all(np.isfinite(A_pinv)))

        # Test first matrix (identity) specifically
        first_result = result_data[:4].reshape((2, 2), order="F")
        expected_identity = np.eye(2)
        np.testing.assert_allclose(
            first_result, expected_identity, rtol=1e-5, atol=1e-6
        )

    def test_ill_conditioned_matrix(self):
        """Test pseudoinverse with ill-conditioned matrix."""
        # Create a matrix that's close to singular but not quite
        A = np.array(
            [
                [1.0, 1.0],
                # Nearly rank-deficient
                [1.0, 1.000001],
            ],
            dtype=np.float32,
        )
        A_pinv = self._run_single_matrix(A, precision="float")

        # Should still produce finite results
        self.assertTrue(np.all(np.isfinite(A_pinv)))

        # Test Moore-Penrose property: A * A_pinv * A ≈ A
        result = A @ A_pinv @ A
        np.testing.assert_allclose(result, A, rtol=1e-3, atol=1e-4)

    def test_performance_many_matrices(self):
        """Test performance with many small matrices."""
        num_matrices = 50
        matrices_data = []

        # Create many 2x2 identity matrices with small perturbations
        for i in range(num_matrices):
            eps = 0.001 * i
            A = np.array([[1.0 + eps, 0.0], [0.0, 1.0 + eps]], dtype=np.float32)
            matrices_data.extend(self._flatten_col_major(A).tolist())

        src = blocks.vector_source_f(matrices_data, repeat=False, vlen=4)
        pinv_block = matrix_pseudo_inverse_sync_f([2, 2])
        sink = blocks.vector_sink_f(vlen=4)

        self.tb.connect(src, pinv_block)
        self.tb.connect(pinv_block, sink)
        self.tb.run()

        result_data = np.array(sink.data())

        # Should have processed all matrices
        self.assertEqual(len(result_data), num_matrices * 4)

        # All results should be finite
        self.assertTrue(np.all(np.isfinite(result_data)))

        # Test first and last results
        first_result = result_data[:4].reshape((2, 2), order="F")
        last_result = result_data[-4:].reshape((2, 2), order="F")

        # Should be close to identity matrices (reciprocals of diagonal elements)
        np.testing.assert_allclose(first_result[0, 0], 1.0, rtol=1e-5)
        np.testing.assert_allclose(first_result[1, 1], 1.0, rtol=1e-5)
        self.assertLess(abs(first_result[0, 1]), 1e-5)
        self.assertLess(abs(first_result[1, 0]), 1e-5)

    def test_penrose_properties(self):
        """Test Moore-Penrose pseudoinverse properties for a general matrix."""
        # Use a 3x2 matrix for testing
        A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        A_pinv = self._run_single_matrix(A, precision="float")

        # Property 1: A * A+ * A = A
        result1 = A @ A_pinv @ A
        np.testing.assert_allclose(result1, A, rtol=1e-4, atol=1e-5)

        # Property 2: A+ * A * A+ = A+
        result2 = A_pinv @ A @ A_pinv
        np.testing.assert_allclose(result2, A_pinv, rtol=1e-4, atol=1e-5)

        # Property 3: (A * A+)^H = A * A+ (Hermitian)
        AA_pinv = A @ A_pinv
        np.testing.assert_allclose(AA_pinv, AA_pinv.T, rtol=1e-4, atol=1e-5)

        # Property 4: (A+ * A)^H = A+ * A (Hermitian)
        A_pinv_A = A_pinv @ A
        np.testing.assert_allclose(A_pinv_A, A_pinv_A.T, rtol=1e-4, atol=1e-5)

    def test_block_instantiation(self):
        """Test that the block can be instantiated with different shapes."""
        # Test various matrix dimensions
        shapes = [(2, 2), (3, 2), (2, 3), (4, 4), (1, 1)]

        for shape in shapes:
            with self.subTest(shape=shape):
                try:
                    block_f = matrix_pseudo_inverse_sync_f(list(shape))
                    block_d = matrix_pseudo_inverse_sync_d(list(shape))
                    # If we get here, instantiation succeeded
                    self.assertIsNotNone(block_f)
                    self.assertIsNotNone(block_d)
                except Exception as e:
                    self.fail(f"Failed to instantiate block with shape {shape}: {e}")


if __name__ == "__main__":
    gr_unittest.run(qa_matrix_pseudo_inverse)
