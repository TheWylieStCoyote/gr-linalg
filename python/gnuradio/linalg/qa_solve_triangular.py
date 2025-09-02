#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2025 Wylie Standage-Beier.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

from gnuradio import gr, gr_unittest
import numpy as np

# from gnuradio import blocks
try:
    # Try to import solve_triangular classes
    from gnuradio.linalg import solve_triangular_sync_f

    HAVE_SOLVE_TRIANGULAR = True
    # If basic import works, try the others
    try:
        from gnuradio.linalg import solve_triangular_sync_d

        HAVE_ALL_TYPES = True
    except ImportError:
        HAVE_ALL_TYPES = False
except ImportError:
    HAVE_SOLVE_TRIANGULAR = False
    HAVE_ALL_TYPES = False


class qa_solve_triangular(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance(self):
        # Test solve_triangular block creation with valid parameters
        if not HAVE_SOLVE_TRIANGULAR:
            self.skipTest("solve_triangular_sync_f not available")

        # Create solve_triangular: 3x3 upper triangular system
        shape_a = [3, 3]
        shape_b = [3, 1]
        instance = solve_triangular_sync_f(shape_a, shape_b, True)  # upper=True
        self.assertIsNotNone(instance)

    def test_upper_triangular_2x2(self):
        # Test solving upper triangular 2x2 system
        if not HAVE_SOLVE_TRIANGULAR:
            self.skipTest("solve_triangular_sync_f not available")

        from gnuradio import blocks

        # Upper triangular system: [[2,3],[0,4]] * x = [8,4]
        # Solution: x = [1, 1] (back substitution: 4*x2=4 -> x2=1; 2*x1+3*1=8 -> x1=2.5)
        # Wait, let me recalculate: 2*x1 + 3*x2 = 8, 0*x1 + 4*x2 = 4
        # From second: x2 = 1, from first: 2*x1 + 3*1 = 8 -> x1 = 2.5
        shape_a = [2, 2]
        shape_b = [2, 1]
        upper_matrix = [2.0, 0.0, 3.0, 4.0]  # Column-major: [[2,3],[0,4]]
        rhs_vector = [8.0, 4.0]
        expected_solution = [2.5, 1.0]

        src_a = blocks.vector_source_f(upper_matrix, False, 4)
        src_b = blocks.vector_source_f(rhs_vector, False, 2)
        solve_block = solve_triangular_sync_f(shape_a, shape_b, True)  # upper=True
        sink = blocks.vector_sink_f(2)

        self.tb.connect(src_a, (solve_block, 0))
        self.tb.connect(src_b, (solve_block, 1))
        self.tb.connect(solve_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 2)
        for i, expected in enumerate(expected_solution):
            self.assertAlmostEqual(result[i], expected, places=5)

    def test_lower_triangular_2x2(self):
        # Test solving lower triangular 2x2 system
        if not HAVE_SOLVE_TRIANGULAR:
            self.skipTest("solve_triangular_sync_f not available")

        from gnuradio import blocks

        # Lower triangular system: [[3,0],[2,4]] * x = [6,10]
        # Solution: 3*x1 = 6 -> x1 = 2; 2*x1 + 4*x2 = 10 -> 2*2 + 4*x2 = 10 -> x2 = 1.5
        shape_a = [2, 2]
        shape_b = [2, 1]
        lower_matrix = [3.0, 2.0, 0.0, 4.0]  # Column-major: [[3,0],[2,4]]
        rhs_vector = [6.0, 10.0]
        expected_solution = [2.0, 1.5]

        src_a = blocks.vector_source_f(lower_matrix, False, 4)
        src_b = blocks.vector_source_f(rhs_vector, False, 2)
        solve_block = solve_triangular_sync_f(
            shape_a, shape_b, False
        )  # upper=False (lower)
        sink = blocks.vector_sink_f(2)

        self.tb.connect(src_a, (solve_block, 0))
        self.tb.connect(src_b, (solve_block, 1))
        self.tb.connect(solve_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 2)
        for i, expected in enumerate(expected_solution):
            self.assertAlmostEqual(result[i], expected, places=5)

    def test_upper_triangular_3x3(self):
        # Test solving larger upper triangular system
        if not HAVE_SOLVE_TRIANGULAR:
            self.skipTest("solve_triangular_sync_f not available")

        from gnuradio import blocks

        # Upper triangular 3x3: [[2,1,3],[0,4,2],[0,0,1]] * x = [14,10,3]
        # Back substitution: x3=3, 4*x2+2*3=10 -> x2=1, 2*x1+1*1+3*3=14 -> x1=2
        shape_a = [3, 3]
        shape_b = [3, 1]
        upper_matrix = [2.0, 0.0, 0.0, 1.0, 4.0, 0.0, 3.0, 2.0, 1.0]  # Column-major
        rhs_vector = [14.0, 10.0, 3.0]
        expected_solution = [2.0, 1.0, 3.0]

        src_a = blocks.vector_source_f(upper_matrix, False, 9)
        src_b = blocks.vector_source_f(rhs_vector, False, 3)
        solve_block = solve_triangular_sync_f(shape_a, shape_b, True)  # upper=True
        sink = blocks.vector_sink_f(3)

        self.tb.connect(src_a, (solve_block, 0))
        self.tb.connect(src_b, (solve_block, 1))
        self.tb.connect(solve_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 3)
        for i, expected in enumerate(expected_solution):
            self.assertAlmostEqual(result[i], expected, places=5)

    def test_lower_triangular_3x3(self):
        # Test solving larger lower triangular system
        if not HAVE_SOLVE_TRIANGULAR:
            self.skipTest("solve_triangular_sync_f not available")

        from gnuradio import blocks

        # Lower triangular 3x3: [[1,0,0],[2,3,0],[1,2,4]] * x = [1,8,17]
        # Forward substitution: x1=1, 2*1+3*x2=8 -> x2=2, 1*1+2*2+4*x3=17 -> x3=3
        shape_a = [3, 3]
        shape_b = [3, 1]
        lower_matrix = [1.0, 2.0, 1.0, 0.0, 3.0, 2.0, 0.0, 0.0, 4.0]  # Column-major
        rhs_vector = [1.0, 8.0, 17.0]
        expected_solution = [1.0, 2.0, 3.0]

        src_a = blocks.vector_source_f(lower_matrix, False, 9)
        src_b = blocks.vector_source_f(rhs_vector, False, 3)
        solve_block = solve_triangular_sync_f(shape_a, shape_b, False)  # upper=False
        sink = blocks.vector_sink_f(3)

        self.tb.connect(src_a, (solve_block, 0))
        self.tb.connect(src_b, (solve_block, 1))
        self.tb.connect(solve_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 3)
        for i, expected in enumerate(expected_solution):
            self.assertAlmostEqual(result[i], expected, places=5)

    def test_identity_matrix_solve(self):
        # Test solving with identity matrix (trivial case)
        if not HAVE_SOLVE_TRIANGULAR:
            self.skipTest("solve_triangular_sync_f not available")

        from gnuradio import blocks

        # Identity matrix [[1,0],[0,1]] * x = [5,3] -> solution: x = [5,3]
        shape_a = [2, 2]
        shape_b = [2, 1]
        identity_matrix = [1.0, 0.0, 0.0, 1.0]  # Column-major identity
        rhs_vector = [5.0, 3.0]
        expected_solution = [5.0, 3.0]

        src_a = blocks.vector_source_f(identity_matrix, False, 4)
        src_b = blocks.vector_source_f(rhs_vector, False, 2)
        solve_block = solve_triangular_sync_f(
            shape_a, shape_b, True
        )  # Can be upper or lower
        sink = blocks.vector_sink_f(2)

        self.tb.connect(src_a, (solve_block, 0))
        self.tb.connect(src_b, (solve_block, 1))
        self.tb.connect(solve_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 2)
        for i, expected in enumerate(expected_solution):
            self.assertAlmostEqual(result[i], expected, places=6)

    def test_solution_verification(self):
        # Test that solution actually satisfies the triangular equation
        if not HAVE_SOLVE_TRIANGULAR:
            self.skipTest("solve_triangular_sync_f not available")

        from gnuradio import blocks

        # Upper triangular system for verification
        shape_a = [2, 2]
        shape_b = [2, 1]
        upper_matrix = [3.0, 0.0, 2.0, 5.0]  # [[3,2],[0,5]]
        rhs_vector = [
            11.0,
            10.0,
        ]  # Should give x = [1,2]: 3*1+2*2=7+4=11 ✗ Let me fix this
        # Actually: 3*x1 + 2*x2 = 11, 0*x1 + 5*x2 = 10
        # From second: x2 = 2, from first: 3*x1 + 2*2 = 11 -> x1 = 7/3 ≈ 2.33
        expected_solution = [7.0 / 3.0, 2.0]

        src_a = blocks.vector_source_f(upper_matrix, False, 4)
        src_b = blocks.vector_source_f(rhs_vector, False, 2)
        solve_block = solve_triangular_sync_f(shape_a, shape_b, True)
        sink = blocks.vector_sink_f(2)

        self.tb.connect(src_a, (solve_block, 0))
        self.tb.connect(src_b, (solve_block, 1))
        self.tb.connect(solve_block, sink)
        self.tb.run()

        result = sink.data()

        # Verify the solution by substitution
        A = np.array([[3.0, 2.0], [0.0, 5.0]])
        x_computed = np.array(result)
        b_original = np.array(rhs_vector)
        b_computed = A @ x_computed

        # Check that A*x ≈ b
        self.assertTrue(np.allclose(b_computed, b_original, atol=1e-5))

    def test_double_precision(self):
        # Test double precision triangular solve
        if not HAVE_ALL_TYPES:
            self.skipTest("solve_triangular_sync_d not available")

        from gnuradio import blocks

        # Simple upper triangular system with double precision
        shape_a = [2, 2]
        shape_b = [2, 1]
        upper_matrix = [2.0, 0.0, 1.0, 3.0]  # [[2,1],[0,3]]
        rhs_vector = [5.0, 9.0]  # 2*x1+1*x2=5, 0*x1+3*x2=9 -> x2=3, x1=1
        expected_solution = [1.0, 3.0]

        src_a = blocks.vector_source_d(upper_matrix, False, 4)
        src_b = blocks.vector_source_d(rhs_vector, False, 2)
        solve_block = solve_triangular_sync_d(shape_a, shape_b, True)
        sink = blocks.vector_sink_d(2)

        self.tb.connect(src_a, (solve_block, 0))
        self.tb.connect(src_b, (solve_block, 1))
        self.tb.connect(solve_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 2)
        for i, expected in enumerate(expected_solution):
            self.assertAlmostEqual(result[i], expected, places=12)  # Higher precision

    def test_multiple_rhs_vectors(self):
        # Test solving system with multiple right-hand side vectors
        if not HAVE_SOLVE_TRIANGULAR:
            self.skipTest("solve_triangular_sync_f not available")

        from gnuradio import blocks

        # Upper triangular 2x2 with 2x2 RHS matrix (2 RHS vectors)
        shape_a = [2, 2]
        shape_b = [2, 2]  # 2 RHS vectors
        upper_matrix = [4.0, 0.0, 2.0, 3.0]  # [[4,2],[0,3]]
        rhs_matrix = [8.0, 6.0, 10.0, 9.0]  # [[8,10],[6,9]] - two RHS vectors
        # For RHS1 [8,6]: 4*x1+2*x2=8, 3*x2=6 -> x2=2, x1=1
        # For RHS2 [10,9]: 4*x1+2*x2=10, 3*x2=9 -> x2=3, x1=1
        expected_solution = [1.0, 1.0, 2.0, 3.0]  # Two solution vectors concatenated

        src_a = blocks.vector_source_f(upper_matrix, False, 4)
        src_b = blocks.vector_source_f(rhs_matrix, False, 4)
        solve_block = solve_triangular_sync_f(shape_a, shape_b, True)
        sink = blocks.vector_sink_f(4)

        self.tb.connect(src_a, (solve_block, 0))
        self.tb.connect(src_b, (solve_block, 1))
        self.tb.connect(solve_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 4)
        for i, expected in enumerate(expected_solution):
            self.assertAlmostEqual(result[i], expected, places=5)

    def test_numerical_accuracy(self):
        # Test numerical accuracy with well-conditioned triangular matrix
        if not HAVE_SOLVE_TRIANGULAR:
            self.skipTest("solve_triangular_sync_f not available")

        from gnuradio import blocks

        # Well-conditioned upper triangular matrix
        shape_a = [3, 3]
        shape_b = [3, 1]
        # Diagonal-dominant upper triangular matrix
        upper_matrix = [5.0, 0.0, 0.0, 0.5, 4.0, 0.0, 0.2, 0.3, 6.0]
        rhs_vector = [5.7, 4.3, 6.0]  # Known solution: approximately [1, 1, 1]

        src_a = blocks.vector_source_f(upper_matrix, False, 9)
        src_b = blocks.vector_source_f(rhs_vector, False, 3)
        solve_block = solve_triangular_sync_f(shape_a, shape_b, True)
        sink = blocks.vector_sink_f(3)

        self.tb.connect(src_a, (solve_block, 0))
        self.tb.connect(src_b, (solve_block, 1))
        self.tb.connect(solve_block, sink)
        self.tb.run()

        result = sink.data()

        # Verify solution by back-substitution check
        A = np.array(upper_matrix).reshape(3, 3, order="F")  # Column-major to row-major
        x_computed = np.array(result)
        b_original = np.array(rhs_vector)
        b_computed = A @ x_computed

        # Should satisfy Ax = b within numerical precision
        self.assertTrue(np.allclose(b_computed, b_original, rtol=1e-5))


if __name__ == "__main__":
    gr_unittest.run(qa_solve_triangular)
