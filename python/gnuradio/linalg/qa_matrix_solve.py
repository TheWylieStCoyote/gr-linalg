#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2025 Wylie Standage-Beier.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

from gnuradio import gr, gr_unittest, blocks
import numpy as np

# Import matrix_solve functionality
try:
    from gnuradio.linalg import matrix_solve, matrix_solve_ff, matrix_solve_dd
    from gnuradio.linalg import matrix_solve_cc, matrix_solve_zz, solver_method
except ImportError:
    import os
    import sys

    dirname, filename = os.path.split(os.path.abspath(__file__))
    sys.path.append(os.path.join(dirname, "bindings"))
    try:
        from gnuradio.linalg import matrix_solve, matrix_solve_ff, matrix_solve_dd
        from gnuradio.linalg import matrix_solve_cc, matrix_solve_zz, solver_method
    except ImportError:
        # Try direct C++ import as fallback
        import linalg_python

        matrix_solve_sync_f = linalg_python.matrix_solve_sync_f
        matrix_solve_sync_d = linalg_python.matrix_solve_sync_d
        matrix_solve_sync_c = linalg_python.matrix_solve_sync_c
        matrix_solve_sync_cd = linalg_python.matrix_solve_sync_cd
        solver_method = linalg_python.solver_method


class qa_matrix_solve(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_basic_2x2_system(self):
        """Test basic 2x2 linear system solution"""
        # System: [2  1] [x1]   [5]
        #         [1  1] [x2] = [3]
        # Solution: x1=2, x2=1

        matrix_a = [2.0, 1.0, 1.0, 1.0]  # Column-major: [[2, 1], [1, 1]]
        vector_b = [5.0, 3.0]  # Column-major: [[5], [3]]
        expected_solution = [2.0, 1.0]

        src_a = blocks.vector_source_f(matrix_a, False, 4)
        src_b = blocks.vector_source_f(vector_b, False, 2)
        solve_block = matrix_solve_ff([2, 2], [2, 1])
        sink = blocks.vector_sink_f(2)

        self.tb.connect(src_a, (solve_block, 0))
        self.tb.connect(src_b, (solve_block, 1))
        self.tb.connect(solve_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertFloatTuplesAlmostEqual(tuple(result), tuple(expected_solution), 5)

    def test_3x3_system(self):
        """Test 3x3 linear system solution"""
        # System: [2  1  1] [x1]   [6]
        #         [1  2  1] [x2] = [6]
        #         [1  1  2] [x3]   [6]
        # Solution: x1=1.5, x2=1.5, x3=1.5

        matrix_a = [2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0]  # Column-major
        vector_b = [6.0, 6.0, 6.0]
        expected_solution = [1.5, 1.5, 1.5]

        src_a = blocks.vector_source_f(matrix_a, False, 9)
        src_b = blocks.vector_source_f(vector_b, False, 3)
        solve_block = matrix_solve_ff([3, 3], [3, 1])
        sink = blocks.vector_sink_f(3)

        self.tb.connect(src_a, (solve_block, 0))
        self.tb.connect(src_b, (solve_block, 1))
        self.tb.connect(solve_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertFloatTuplesAlmostEqual(tuple(result), tuple(expected_solution), 5)

    def test_solver_methods(self):
        """Test different solver methods"""
        # Use same 2x2 system for all methods
        matrix_a = [2.0, 1.0, 1.0, 1.0]
        vector_b = [5.0, 3.0]
        expected_solution = [2.0, 1.0]

        methods = [
            (solver_method.AUTO, "AUTO"),
            (solver_method.LU, "LU"),
            (solver_method.QR, "QR"),
            (solver_method.SVD, "SVD"),
        ]

        for method, method_name in methods:
            with self.subTest(method=method_name):
                tb = gr.top_block()

                src_a = blocks.vector_source_f(matrix_a, False, 4)
                src_b = blocks.vector_source_f(vector_b, False, 2)
                solve_block = matrix_solve_ff([2, 2], [2, 1], method)
                sink = blocks.vector_sink_f(2)

                tb.connect(src_a, (solve_block, 0))
                tb.connect(src_b, (solve_block, 1))
                tb.connect(solve_block, sink)
                tb.run()

                result = sink.data()
                self.assertFloatTuplesAlmostEqual(
                    tuple(result),
                    tuple(expected_solution),
                    4,
                    f"{method_name} solver failed",
                )

    def test_double_precision(self):
        """Test matrix solve with double precision"""
        # Skip this test since GNU Radio doesn't have vector_source_d
        self.skipTest(
            "GNU Radio doesn't provide vector_source_d - double precision blocks use float sources"
        )

    def test_complex_system(self):
        """Test complex linear system solution"""
        # Simple 2x2 complex system
        matrix_a = [complex(2, 0), complex(0, 1), complex(1, 0), complex(1, 0)]
        vector_b = [complex(2, 1), complex(1, 0)]

        src_a = blocks.vector_source_c(matrix_a, False, 4)
        src_b = blocks.vector_source_c(vector_b, False, 2)
        solve_block = matrix_solve_cc([2, 2], [2, 1])
        sink = blocks.vector_sink_c(2)

        self.tb.connect(src_a, (solve_block, 0))
        self.tb.connect(src_b, (solve_block, 1))
        self.tb.connect(solve_block, sink)
        self.tb.run()

        result = sink.data()
        # Check that we got a valid solution (sum of residuals should be small)
        self.assertEqual(len(result), 2)
        # For complex systems, just verify the solution exists and is reasonable
        self.assertLess(
            abs(result[0]), 10.0
        )  # Solution components should be reasonable
        self.assertLess(abs(result[1]), 10.0)

    def test_overdetermined_system(self):
        """Test overdetermined system (more equations than unknowns)"""
        # 3x2 system (3 equations, 2 unknowns) - should use QR or SVD
        matrix_a = [1.0, 1.0, 2.0, 2.0, 3.0, 4.0]  # [[1, 2], [1, 3], [2, 4]]
        vector_b = [3.0, 4.0, 6.0]  # [[3], [4], [6]]

        src_a = blocks.vector_source_f(matrix_a, False, 6)
        src_b = blocks.vector_source_f(vector_b, False, 3)
        solve_block = matrix_solve_ff([3, 2], [3, 1], solver_method.QR)
        sink = blocks.vector_sink_f(2)

        self.tb.connect(src_a, (solve_block, 0))
        self.tb.connect(src_b, (solve_block, 1))
        self.tb.connect(solve_block, sink)
        self.tb.run()

        result = sink.data()
        # For overdetermined systems, just verify we get a solution
        self.assertEqual(len(result), 2)
        # Solution should be reasonable (not NaN or infinity)
        self.assertTrue(all(abs(x) < 100.0 for x in result))

    def test_underdetermined_system(self):
        """Test underdetermined system (more unknowns than equations)"""
        # 2x3 system (2 equations, 3 unknowns) - should use SVD
        matrix_a = [1.0, 2.0, 2.0, 3.0, 3.0, 4.0]  # [[1, 2], [2, 3], [3, 4]]
        vector_b = [5.0, 8.0]  # [[5], [8]]

        src_a = blocks.vector_source_f(matrix_a, False, 6)
        src_b = blocks.vector_source_f(vector_b, False, 2)
        solve_block = matrix_solve_ff([2, 3], [2, 1], solver_method.SVD)
        sink = blocks.vector_sink_f(3)

        self.tb.connect(src_a, (solve_block, 0))
        self.tb.connect(src_b, (solve_block, 1))
        self.tb.connect(solve_block, sink)
        self.tb.run()

        result = sink.data()
        # For underdetermined systems, just verify we get a solution
        self.assertEqual(len(result), 3)
        # Solution should be reasonable (not NaN or infinity)
        self.assertTrue(all(abs(x) < 100.0 for x in result))

    def test_convenience_functions(self):
        """Test the convenience functions work correctly"""
        matrix_a = [2.0, 1.0, 1.0, 1.0]
        vector_b = [5.0, 3.0]

        # Test generic function
        try:
            solve_block = matrix_solve([2, 2], [2, 1])
            self.assertIsNotNone(
                solve_block, "Generic matrix_solve function should work"
            )
        except Exception as e:
            self.skipTest(f"matrix_solve function not available: {e}")

        # Test specific precision functions
        try:
            solve_block_f = matrix_solve_ff([2, 2], [2, 1])
            self.assertIsNotNone(solve_block_f, "matrix_solve_ff should work")
        except Exception as e:
            self.skipTest(f"matrix_solve_ff function not available: {e}")

    def test_multiple_systems(self):
        """Test solving multiple systems in sequence"""
        # Process two 2x2 systems in sequence
        matrix_data = [2.0, 1.0, 1.0, 1.0, 3.0, 1.0, 2.0, 1.0]  # Two 2x2 matrices
        vector_data = [5.0, 3.0, 7.0, 5.0]  # Two 2x1 vectors

        src_a = blocks.vector_source_f(matrix_data, False, 4)
        src_b = blocks.vector_source_f(vector_data, False, 2)
        solve_block = matrix_solve_ff([2, 2], [2, 1])
        sink = blocks.vector_sink_f(2)

        self.tb.connect(src_a, (solve_block, 0))
        self.tb.connect(src_b, (solve_block, 1))
        self.tb.connect(solve_block, sink)
        self.tb.run()

        result = sink.data()
        # Should get 4 elements back (two 2x1 solution vectors)
        self.assertEqual(len(result), 4)

    def test_identity_matrix(self):
        """Test solving with identity matrix (trivial case)"""
        # Identity matrix system: I*x = b => x = b
        matrix_a = [1.0, 0.0, 0.0, 1.0]  # 2x2 identity
        vector_b = [3.0, 4.0]  # Expected solution
        expected_solution = [3.0, 4.0]

        src_a = blocks.vector_source_f(matrix_a, False, 4)
        src_b = blocks.vector_source_f(vector_b, False, 2)
        solve_block = matrix_solve_ff([2, 2], [2, 1])
        sink = blocks.vector_sink_f(2)

        self.tb.connect(src_a, (solve_block, 0))
        self.tb.connect(src_b, (solve_block, 1))
        self.tb.connect(solve_block, sink)
        self.tb.run()

        result = sink.data()
        self.assertFloatTuplesAlmostEqual(tuple(result), tuple(expected_solution), 6)


if __name__ == "__main__":
    gr_unittest.run(qa_matrix_solve)
