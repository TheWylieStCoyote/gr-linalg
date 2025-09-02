#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNU Radio Integration Test: Matrix Operations Flowgraph

This test demonstrates a complete GNU Radio flowgraph using multiple linear algebra blocks
to perform common matrix operations in a signal processing context.

Scenario: Matrix-based signal processing chain
- Generate test matrices
- Perform addition, multiplication, and transpose operations
- Verify results using our linear algebra blocks
"""

import numpy as np
from gnuradio import gr, blocks
from gnuradio.linalg import (
    matrix_add_sync_d,
    matrix_multiply_sync_d,
    matrix_transpose_sync_d,
    matrix_determinant_sync_d,
    matrix_condition_number_sync_d,
)
import unittest
import time
import sys
import os


class MatrixOperationsFlowgraph(gr.top_block):
    """
    Complete flowgraph demonstrating matrix operations in GNU Radio
    """

    def __init__(self):
        gr.top_block.__init__(self, "Matrix Operations Integration Test")

        # Matrix dimensions for testing
        self.matrix_rows = 3
        self.matrix_cols = 3
        self.matrix_size = self.matrix_rows * self.matrix_cols

        # Test data: create well-conditioned test matrices
        self.test_matrix_a = np.array(
            [[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]], dtype=np.float64
        ).flatten()

        self.test_matrix_b = np.array(
            [[1.0, 0.0, 1.0], [0.0, 2.0, 0.0], [1.0, 0.0, 1.0]], dtype=np.float64
        ).flatten()

        # Expected results for verification
        self.expected_add = self.test_matrix_a.reshape(
            3, 3
        ) + self.test_matrix_b.reshape(3, 3)
        self.expected_multiply = np.dot(
            self.test_matrix_a.reshape(3, 3), self.test_matrix_b.reshape(3, 3)
        )
        self.expected_transpose = self.test_matrix_a.reshape(3, 3).T

        # Create GNU Radio blocks
        self._create_sources()
        self._create_linalg_blocks()
        self._create_sinks()
        self._connect_flowgraph()

    def _create_sources(self):
        """Create vector sources for test matrices"""
        # Source for matrix A
        self.source_a = blocks.vector_source_f(
            self.test_matrix_a.astype(np.float32).tolist(),
            repeat=False,
            vlen=self.matrix_size,
        )

        # Source for matrix B
        self.source_b = blocks.vector_source_f(
            self.test_matrix_b.astype(np.float32).tolist(),
            repeat=False,
            vlen=self.matrix_size,
        )

        # Convert to double precision for linalg blocks
        self.float_to_double_a = blocks.float_to_double(self.matrix_size)
        self.float_to_double_b = blocks.float_to_double(self.matrix_size)

    def _create_linalg_blocks(self):
        """Create linear algebra processing blocks"""
        matrix_shape = [self.matrix_rows, self.matrix_cols]

        # Matrix addition: C = A + B
        self.matrix_add = matrix_add_sync_d(matrix_shape, 2)

        # Matrix multiplication: D = A * B
        self.matrix_multiply = matrix_multiply_sync_d(matrix_shape, 2)

        # Matrix transpose: E = A^T
        self.matrix_transpose = matrix_transpose_sync_d(matrix_shape)

        # Matrix determinant: det_A = det(A)
        self.matrix_determinant = matrix_determinant_sync_d(matrix_shape)

        # Matrix condition number: cond_A = cond(A)
        self.matrix_condition_number = matrix_condition_number_sync_d()

    def _create_sinks(self):
        """Create sinks to capture results"""
        # Sinks for matrix results
        self.sink_add = blocks.vector_sink_d(self.matrix_size)
        self.sink_multiply = blocks.vector_sink_d(self.matrix_size)
        self.sink_transpose = blocks.vector_sink_d(self.matrix_size)

        # Sinks for scalar results
        self.sink_determinant = blocks.vector_sink_d(1)
        self.sink_condition_number = blocks.vector_sink_d(1)

    def _connect_flowgraph(self):
        """Connect all blocks in the flowgraph"""
        # Convert sources to double precision
        self.connect((self.source_a, 0), (self.float_to_double_a, 0))
        self.connect((self.source_b, 0), (self.float_to_double_b, 0))

        # Matrix addition: A + B
        self.connect((self.float_to_double_a, 0), (self.matrix_add, 0))
        self.connect((self.float_to_double_b, 0), (self.matrix_add, 1))
        self.connect((self.matrix_add, 0), (self.sink_add, 0))

        # Matrix multiplication: A * B
        self.connect((self.float_to_double_a, 0), (self.matrix_multiply, 0))
        self.connect((self.float_to_double_b, 0), (self.matrix_multiply, 1))
        self.connect((self.matrix_multiply, 0), (self.sink_multiply, 0))

        # Matrix transpose: A^T
        self.connect((self.float_to_double_a, 0), (self.matrix_transpose, 0))
        self.connect((self.matrix_transpose, 0), (self.sink_transpose, 0))

        # Matrix determinant: det(A)
        self.connect((self.float_to_double_a, 0), (self.matrix_determinant, 0))
        self.connect((self.matrix_determinant, 0), (self.sink_determinant, 0))

        # Matrix condition number: cond(A)
        self.connect((self.float_to_double_a, 0), (self.matrix_condition_number, 0))
        self.connect((self.matrix_condition_number, 0), (self.sink_condition_number, 0))

    def get_results(self):
        """Retrieve results from sinks"""
        return {
            "add": np.array(self.sink_add.data()).reshape(
                self.matrix_rows, self.matrix_cols
            ),
            "multiply": np.array(self.sink_multiply.data()).reshape(
                self.matrix_rows, self.matrix_cols
            ),
            "transpose": np.array(self.sink_transpose.data()).reshape(
                self.matrix_rows, self.matrix_cols
            ),
            "determinant": (
                self.sink_determinant.data()[0]
                if self.sink_determinant.data()
                else None
            ),
            "condition_number": (
                self.sink_condition_number.data()[0]
                if self.sink_condition_number.data()
                else None
            ),
        }


class TestMatrixOperationsIntegration(unittest.TestCase):
    """Integration test for matrix operations in GNU Radio flowgraph"""

    def setUp(self):
        """Set up test flowgraph"""
        self.tb = MatrixOperationsFlowgraph()

    def tearDown(self):
        """Clean up test flowgraph"""
        self.tb.stop()
        self.tb.wait()
        self.tb = None

    def test_matrix_operations_flowgraph(self):
        """Test complete matrix operations flowgraph"""
        # Run the flowgraph
        self.tb.start()

        # Wait for processing to complete
        time.sleep(1.0)

        self.tb.stop()
        self.tb.wait()

        # Get results
        results = self.tb.get_results()

        # Verify matrix addition
        np.testing.assert_array_almost_equal(
            results["add"],
            self.tb.expected_add,
            decimal=6,
            err_msg="Matrix addition failed in flowgraph",
        )

        # Verify matrix multiplication
        np.testing.assert_array_almost_equal(
            results["multiply"],
            self.tb.expected_multiply,
            decimal=6,
            err_msg="Matrix multiplication failed in flowgraph",
        )

        # Verify matrix transpose
        np.testing.assert_array_almost_equal(
            results["transpose"],
            self.tb.expected_transpose,
            decimal=6,
            err_msg="Matrix transpose failed in flowgraph",
        )

        # Verify determinant is reasonable
        self.assertIsNotNone(results["determinant"], "Determinant calculation failed")
        self.assertGreater(
            abs(results["determinant"]), 0.1, "Determinant seems too small"
        )

        # Verify condition number is reasonable
        self.assertIsNotNone(
            results["condition_number"], "Condition number calculation failed"
        )
        self.assertGreater(
            results["condition_number"], 1.0, "Condition number should be >= 1"
        )
        self.assertLess(
            results["condition_number"],
            100.0,
            "Condition number suggests ill-conditioning",
        )

        print("✓ All matrix operations completed successfully in GNU Radio flowgraph")
        print(f"  Matrix addition: {results['add'].shape} matrix")
        print(f"  Matrix multiplication: {results['multiply'].shape} matrix")
        print(f"  Matrix transpose: {results['transpose'].shape} matrix")
        print(f"  Determinant: {results['determinant']:.6f}")
        print(f"  Condition number: {results['condition_number']:.6f}")


def run_integration_test():
    """Run the integration test standalone"""
    print("GNU Radio Linear Algebra Integration Test")
    print("=========================================")
    print()

    try:
        # Import test to verify all dependencies
        from gnuradio.linalg import matrix_add_sync_d

        print("✓ Successfully imported GNU Radio linalg module")

        # Run the test
        unittest.main(verbosity=2, exit=False)

    except ImportError as e:
        print(f"✗ Failed to import required modules: {e}")
        print("  Make sure GNU Radio and gr-linalg are properly installed")
        return False
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

    return True


if __name__ == "__main__":
    success = run_integration_test()
    sys.exit(0 if success else 1)
