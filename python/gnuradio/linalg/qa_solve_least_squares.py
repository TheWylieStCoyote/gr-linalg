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
    # Try to import solve_least_squares classes
    from gnuradio.linalg import solve_least_squares_sync_f

    HAVE_SOLVE_LEAST_SQUARES = True
    # If basic import works, try the others
    try:
        from gnuradio.linalg import solve_least_squares_sync_d

        HAVE_ALL_TYPES = True
    except ImportError:
        HAVE_ALL_TYPES = False
except ImportError:
    HAVE_SOLVE_LEAST_SQUARES = False
    HAVE_ALL_TYPES = False


class qa_solve_least_squares(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_instance(self):
        # Test solve_least_squares block creation with valid parameters
        if not HAVE_SOLVE_LEAST_SQUARES:
            self.skipTest("solve_least_squares_sync_f not available")

        # Create solve_least_squares: 3x2 overdetermined system
        shape_a = [3, 2]
        shape_b = [3, 1]
        instance = solve_least_squares_sync_f(shape_a, shape_b)
        self.assertIsNotNone(instance)

    def test_overdetermined_system_2x2(self):
        # Test basic overdetermined system solving
        if not HAVE_SOLVE_LEAST_SQUARES:
            self.skipTest("solve_least_squares_sync_f not available")

        from gnuradio import blocks

        # Overdetermined system: A = [[1,1],[1,2],[1,3]], b = [6,8,10]
        # Expected least squares solution: x ≈ [4, 2] (line fitting)
        shape_a = [3, 2]
        shape_b = [3, 1]
        matrix_a = [1.0, 1.0, 1.0, 1.0, 2.0, 3.0]  # Column-major: [[1,1],[1,2],[1,3]]
        vector_b = [6.0, 8.0, 10.0]
        expected_solution = [4.0, 2.0]  # Least squares solution

        src_a = blocks.vector_source_f(matrix_a, False, 6)
        src_b = blocks.vector_source_f(vector_b, False, 3)
        solve_block = solve_least_squares_sync_f(shape_a, shape_b)
        sink = blocks.vector_sink_f(2)

        self.tb.connect(src_a, (solve_block, 0))
        self.tb.connect(src_b, (solve_block, 1))
        self.tb.connect(solve_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 2)
        for i, expected in enumerate(expected_solution):
            self.assertAlmostEqual(result[i], expected, places=4)

    def test_line_fitting_problem(self):
        # Test least squares line fitting y = mx + c
        if not HAVE_SOLVE_LEAST_SQUARES:
            self.skipTest("solve_least_squares_sync_f not available")

        from gnuradio import blocks

        # Data points: (0,1), (1,3), (2,5) -> should fit line y = 2x + 1
        # A = [[1,0],[1,1],[1,2]] (augmented matrix for y = c + m*x)
        # b = [1,3,5]
        # Expected solution: [c,m] = [1,2]
        shape_a = [3, 2]
        shape_b = [3, 1]
        matrix_a = [1.0, 1.0, 1.0, 0.0, 1.0, 2.0]  # Column-major
        vector_b = [1.0, 3.0, 5.0]
        expected_solution = [1.0, 2.0]  # [intercept, slope]

        src_a = blocks.vector_source_f(matrix_a, False, 6)
        src_b = blocks.vector_source_f(vector_b, False, 3)
        solve_block = solve_least_squares_sync_f(shape_a, shape_b)
        sink = blocks.vector_sink_f(2)

        self.tb.connect(src_a, (solve_block, 0))
        self.tb.connect(src_b, (solve_block, 1))
        self.tb.connect(solve_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 2)
        for i, expected in enumerate(expected_solution):
            self.assertAlmostEqual(result[i], expected, places=5)

    def test_exact_solution_case(self):
        # Test when overdetermined system has exact solution (consistent)
        if not HAVE_SOLVE_LEAST_SQUARES:
            self.skipTest("solve_least_squares_sync_f not available")

        from gnuradio import blocks

        # System where all points lie exactly on line y = 3x + 2
        # Points: (0,2), (1,5), (2,8) -> A*x = b has exact solution x = [2,3]
        shape_a = [3, 2]
        shape_b = [3, 1]
        matrix_a = [1.0, 1.0, 1.0, 0.0, 1.0, 2.0]  # Column-major: [[1,0],[1,1],[1,2]]
        vector_b = [2.0, 5.0, 8.0]  # Perfect line data
        expected_solution = [2.0, 3.0]  # [intercept, slope]

        src_a = blocks.vector_source_f(matrix_a, False, 6)
        src_b = blocks.vector_source_f(vector_b, False, 3)
        solve_block = solve_least_squares_sync_f(shape_a, shape_b)
        sink = blocks.vector_sink_f(2)

        self.tb.connect(src_a, (solve_block, 0))
        self.tb.connect(src_b, (solve_block, 1))
        self.tb.connect(solve_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 2)
        for i, expected in enumerate(expected_solution):
            self.assertAlmostEqual(result[i], expected, places=6)

    def test_underdetermined_system(self):
        # Test underdetermined system (fewer equations than unknowns)
        if not HAVE_SOLVE_LEAST_SQUARES:
            self.skipTest("solve_least_squares_sync_f not available")

        from gnuradio import blocks

        # System: 2 equations, 3 unknowns - least norm solution
        # A = [[1,2,1],[2,1,3]], b = [4,7]
        # Should find minimum norm solution
        shape_a = [2, 3]
        shape_b = [2, 1]
        matrix_a = [1.0, 2.0, 2.0, 1.0, 1.0, 3.0]  # Column-major
        vector_b = [4.0, 7.0]

        src_a = blocks.vector_source_f(matrix_a, False, 6)
        src_b = blocks.vector_source_f(vector_b, False, 2)
        solve_block = solve_least_squares_sync_f(shape_a, shape_b)
        sink = blocks.vector_sink_f(3)

        self.tb.connect(src_a, (solve_block, 0))
        self.tb.connect(src_b, (solve_block, 1))
        self.tb.connect(solve_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 3)

        # Verify solution satisfies the equations (within tolerance)
        A = np.array(matrix_a).reshape(2, 3, order="F")
        x = np.array(result)
        b_computed = A @ x
        b_original = np.array(vector_b)

        self.assertTrue(np.allclose(b_computed, b_original, rtol=1e-4))

    def test_residual_computation(self):
        # Test that computed solution minimizes residual ||Ax - b||
        if not HAVE_SOLVE_LEAST_SQUARES:
            self.skipTest("solve_least_squares_sync_f not available")

        from gnuradio import blocks

        # Inconsistent system with known residual
        shape_a = [4, 2]
        shape_b = [4, 1]
        matrix_a = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0]  # Column-major
        vector_b = [1.0, 2.0, 4.0, 8.0]  # Data with some noise

        src_a = blocks.vector_source_f(matrix_a, False, 8)
        src_b = blocks.vector_source_f(vector_b, False, 4)
        solve_block = solve_least_squares_sync_f(shape_a, shape_b)
        sink = blocks.vector_sink_f(2)

        self.tb.connect(src_a, (solve_block, 0))
        self.tb.connect(src_b, (solve_block, 1))
        self.tb.connect(solve_block, sink)
        self.tb.run()

        result = sink.data()

        # Verify solution quality by comparing with NumPy
        A = np.array(matrix_a).reshape(4, 2, order="F")
        b = np.array(vector_b)
        x_numpy, residuals_numpy, rank, s = np.linalg.lstsq(A, b, rcond=None)

        # Our solution should be close to NumPy's solution
        for i in range(2):
            self.assertAlmostEqual(result[i], x_numpy[i], places=4)

    def test_rank_deficient_matrix(self):
        # Test handling of rank-deficient matrices
        if not HAVE_SOLVE_LEAST_SQUARES:
            self.skipTest("solve_least_squares_sync_f not available")

        from gnuradio import blocks

        # Rank-deficient system (columns are linearly dependent)
        # A = [[1,2],[2,4],[3,6]], b = [1,2,3] - second column is 2*first
        shape_a = [3, 2]
        shape_b = [3, 1]
        matrix_a = [1.0, 2.0, 3.0, 2.0, 4.0, 6.0]  # Column-major, rank 1
        vector_b = [1.0, 2.0, 3.0]

        src_a = blocks.vector_source_f(matrix_a, False, 6)
        src_b = blocks.vector_source_f(vector_b, False, 3)
        solve_block = solve_least_squares_sync_f(shape_a, shape_b)
        sink = blocks.vector_sink_f(2)

        self.tb.connect(src_a, (solve_block, 0))
        self.tb.connect(src_b, (solve_block, 1))
        self.tb.connect(solve_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 2)

        # Solution should still minimize residual (though not unique)
        A = np.array(matrix_a).reshape(3, 2, order="F")
        x = np.array(result)
        b_original = np.array(vector_b)
        residual = np.linalg.norm(A @ x - b_original)

        # Residual should be small for this consistent (though degenerate) system
        self.assertLess(residual, 1e-6)

    def test_multiple_rhs_vectors(self):
        # Test solving system with multiple right-hand side vectors
        if not HAVE_SOLVE_LEAST_SQUARES:
            self.skipTest("solve_least_squares_sync_f not available")

        from gnuradio import blocks

        # System with 2 RHS vectors
        shape_a = [3, 2]
        shape_b = [3, 2]  # 2 RHS vectors
        matrix_a = [1.0, 1.0, 1.0, 1.0, 2.0, 3.0]  # Same A matrix
        rhs_matrix = [6.0, 8.0, 10.0, 3.0, 5.0, 7.0]  # Two RHS vectors

        src_a = blocks.vector_source_f(matrix_a, False, 6)
        src_b = blocks.vector_source_f(rhs_matrix, False, 6)
        solve_block = solve_least_squares_sync_f(shape_a, shape_b)
        sink = blocks.vector_sink_f(4)  # 2 solution vectors

        self.tb.connect(src_a, (solve_block, 0))
        self.tb.connect(src_b, (solve_block, 1))
        self.tb.connect(solve_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 4)

        # Verify both solutions independently
        A = np.array(matrix_a).reshape(3, 2, order="F")
        B = np.array(rhs_matrix).reshape(3, 2, order="F")
        X = np.array(result).reshape(2, 2, order="F")

        # Check both solutions
        for j in range(2):
            residual = np.linalg.norm(A @ X[:, j] - B[:, j])
            self.assertLess(residual, 1e-4)

    def test_double_precision_solve(self):
        # Test double precision least squares solving
        if not HAVE_ALL_TYPES:
            self.skipTest("solve_least_squares_sync_d not available")

        from gnuradio import blocks

        # High precision test case
        shape_a = [3, 2]
        shape_b = [3, 1]
        matrix_a = [1.0, 1.0, 1.0, 0.5, 1.0, 1.5]  # Column-major
        vector_b = [2.5, 3.0, 3.5]

        src_a = blocks.vector_source_d(matrix_a, False, 6)
        src_b = blocks.vector_source_d(vector_b, False, 3)
        solve_block = solve_least_squares_sync_d(shape_a, shape_b)
        sink = blocks.vector_sink_d(2)

        self.tb.connect(src_a, (solve_block, 0))
        self.tb.connect(src_b, (solve_block, 1))
        self.tb.connect(solve_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 2)

        # Verify with NumPy double precision
        A = np.array(matrix_a, dtype=np.float64).reshape(3, 2, order="F")
        b = np.array(vector_b, dtype=np.float64)
        x_numpy = np.linalg.lstsq(A, b, rcond=None)[0]

        for i in range(2):
            self.assertAlmostEqual(result[i], x_numpy[i], places=10)

    def test_numerical_stability(self):
        # Test numerical stability with well-conditioned vs ill-conditioned matrices
        if not HAVE_SOLVE_LEAST_SQUARES:
            self.skipTest("solve_least_squares_sync_f not available")

        from gnuradio import blocks

        # Well-conditioned matrix
        shape_a = [4, 2]
        shape_b = [4, 1]
        matrix_a = [1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0]  # Well-conditioned
        vector_b = [10.0, 20.0, 30.0, 40.0]

        src_a = blocks.vector_source_f(matrix_a, False, 8)
        src_b = blocks.vector_source_f(vector_b, False, 4)
        solve_block = solve_least_squares_sync_f(shape_a, shape_b)
        sink = blocks.vector_sink_f(2)

        self.tb.connect(src_a, (solve_block, 0))
        self.tb.connect(src_b, (solve_block, 1))
        self.tb.connect(solve_block, sink)
        self.tb.run()

        result = sink.data()

        # Solution should be numerically stable
        A = np.array(matrix_a).reshape(4, 2, order="F")
        b = np.array(vector_b)
        x = np.array(result)
        residual = np.linalg.norm(A @ x - b)

        # Residual should be very small for well-conditioned system
        self.assertLess(residual, 1e-6)

    def test_zero_solution_case(self):
        # Test case where least squares solution is zero vector
        if not HAVE_SOLVE_LEAST_SQUARES:
            self.skipTest("solve_least_squares_sync_f not available")

        from gnuradio import blocks

        # System Ax = 0 with overdetermined A (homogeneous system)
        shape_a = [3, 2]
        shape_b = [3, 1]
        matrix_a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # Any matrix
        vector_b = [0.0, 0.0, 0.0]  # Zero RHS
        expected_solution = [0.0, 0.0]  # Should be zero

        src_a = blocks.vector_source_f(matrix_a, False, 6)
        src_b = blocks.vector_source_f(vector_b, False, 3)
        solve_block = solve_least_squares_sync_f(shape_a, shape_b)
        sink = blocks.vector_sink_f(2)

        self.tb.connect(src_a, (solve_block, 0))
        self.tb.connect(src_b, (solve_block, 1))
        self.tb.connect(solve_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 2)
        for i, expected in enumerate(expected_solution):
            self.assertAlmostEqual(result[i], expected, places=6)

    def test_polynomial_fitting(self):
        # Test polynomial fitting using least squares
        if not HAVE_SOLVE_LEAST_SQUARES:
            self.skipTest("solve_least_squares_sync_f not available")

        from gnuradio import blocks

        # Fit quadratic y = ax² + bx + c to points (0,1), (1,3), (2,9), (3,19)
        # Should fit y = 2x² + x + 1 exactly
        shape_a = [4, 3]  # Vandermonde matrix
        shape_b = [4, 1]
        # A = [[1,0,0],[1,1,1],[1,2,4],[1,3,9]] for [c,b,a] coefficients
        matrix_a = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 9.0]
        vector_b = [1.0, 3.0, 9.0, 19.0]
        expected_solution = [1.0, 1.0, 2.0]  # [c, b, a] for y = 2x² + x + 1

        src_a = blocks.vector_source_f(matrix_a, False, 12)
        src_b = blocks.vector_source_f(vector_b, False, 4)
        solve_block = solve_least_squares_sync_f(shape_a, shape_b)
        sink = blocks.vector_sink_f(3)

        self.tb.connect(src_a, (solve_block, 0))
        self.tb.connect(src_b, (solve_block, 1))
        self.tb.connect(solve_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertEqual(len(result), 3)
        for i, expected in enumerate(expected_solution):
            self.assertAlmostEqual(result[i], expected, places=5)


if __name__ == "__main__":
    gr_unittest.run(qa_solve_least_squares)
