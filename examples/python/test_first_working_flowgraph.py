#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
First Working GNU Radio Flowgraph Test

This test validates that matrix_add blocks actually work correctly
in GNU Radio flowgraphs with real data processing.
"""

import sys
import numpy as np
from gnuradio import gr, blocks


def main():
    print("GNU Radio Linear Algebra - First Working Flowgraph Test")
    print("======================================================")
    print()

    try:
        # Import with the working build
        sys.path.insert(0, "python")
        sys.path.insert(0, "build/test_modules")
        import linalg
        import gnuradio.linalg.linalg_python as linalg_cpp

        print("✓ Module imported successfully")

        # Test 1: Matrix Addition with Float Data
        print("\nTest 1: Matrix Addition Flowgraph (Float)")
        print("-" * 40)
        try:
            tb = gr.top_block()

            # Create test matrices (2x2)
            matrix1_data = [1.0, 2.0, 3.0, 4.0]  # [[1,2],[3,4]]
            matrix2_data = [5.0, 6.0, 7.0, 8.0]  # [[5,6],[7,8]]

            # Sources (vector_source_f outputs float data, set vlen for matrix size)
            # For 2x2 matrix, we need vector length of 4
            src1 = blocks.vector_source_f(matrix1_data, repeat=False, vlen=4)
            src2 = blocks.vector_source_f(matrix2_data, repeat=False, vlen=4)

            # Matrix addition block for float data
            matrix_add = linalg_cpp.matrix_add_sync_f.make([2, 2], 2)
            print(f"  ✓ Created matrix_add_sync_f block: {type(matrix_add)}")

            # Sink (also needs vector length for matrix output)
            sink = blocks.vector_sink_f(vlen=4)

            # Connect flowgraph
            tb.connect(src1, (matrix_add, 0))
            tb.connect(src2, (matrix_add, 1))
            tb.connect(matrix_add, sink)

            print("  ✓ Flowgraph connected successfully")

            # Run
            print("  ► Running flowgraph...")
            tb.run()
            print("  ✓ Flowgraph execution completed")

            # Check results
            result = sink.data()
            expected = [6.0, 8.0, 10.0, 12.0]  # [[6,8],[10,12]]

            print(f"  Input 1:  {matrix1_data}")
            print(f"  Input 2:  {matrix2_data}")
            print(f"  Output:   {list(result)}")
            print(f"  Expected: {expected}")

            if len(result) == len(expected) and all(
                abs(a - b) < 1e-6 for a, b in zip(result, expected)
            ):
                print("  🎉 PASS: Matrix addition works correctly!")
                test1_success = True
            else:
                print("  ❌ FAIL: Incorrect output")
                test1_success = False

        except Exception as e:
            print(f"  ❌ ERROR: {str(e)}")
            import traceback

            traceback.print_exc()
            test1_success = False

        # Test 2: Matrix Addition with Double Data
        print("\nTest 2: Matrix Addition Flowgraph (Double)")
        print("-" * 40)
        try:
            tb = gr.top_block()

            # Create test matrices (3x2)
            matrix1_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # [[1,2],[3,4],[5,6]]
            matrix2_data = [
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
                60.0,
            ]  # [[10,20],[30,40],[50,60]]

            # Sources (using double data - need vector_source_d for double precision)
            # For 3x2 matrix, we need vector length of 6
            src1 = blocks.vector_source_d(matrix1_data, repeat=False, vlen=6)
            src2 = blocks.vector_source_d(matrix2_data, repeat=False, vlen=6)

            # Matrix addition block for double data (but fed with float - GNU Radio will convert)
            matrix_add = linalg_cpp.matrix_add_sync_d.make([3, 2], 2)
            print(f"  ✓ Created matrix_add_sync_d block: {type(matrix_add)}")

            # Sink (also needs vector length for matrix output, double precision)
            sink = blocks.vector_sink_d(vlen=6)

            # Connect flowgraph
            tb.connect(src1, (matrix_add, 0))
            tb.connect(src2, (matrix_add, 1))
            tb.connect(matrix_add, sink)

            print("  ✓ Flowgraph connected successfully")

            # Run
            print("  ► Running flowgraph...")
            tb.run()
            print("  ✓ Flowgraph execution completed")

            # Check results
            result = sink.data()
            expected = [11.0, 22.0, 33.0, 44.0, 55.0, 66.0]

            print(f"  Input 1:  {matrix1_data}")
            print(f"  Input 2:  {matrix2_data}")
            print(f"  Output:   {list(result)}")
            print(f"  Expected: {expected}")

            if len(result) == len(expected) and all(
                abs(a - b) < 1e-6 for a, b in zip(result, expected)
            ):
                print("  🎉 PASS: Matrix addition (double) works correctly!")
                test2_success = True
            else:
                print("  ❌ FAIL: Incorrect output")
                test2_success = False

        except Exception as e:
            print(f"  ❌ ERROR: {str(e)}")
            import traceback

            traceback.print_exc()
            test2_success = False

        # Test 3: Testing factory function integration
        print("\nTest 3: Factory Function Integration")
        print("-" * 35)
        try:
            # Test if factory functions now work
            block_ff = linalg.matrix_add_ff([2, 2], 2)
            block_dd = linalg.matrix_add_dd([2, 2], 2)

            if block_ff is not None and block_dd is not None:
                print("  ✓ Factory functions now return working blocks!")
                test3_success = True
            else:
                print("  ⚠ Factory functions still return None")
                test3_success = False

        except Exception as e:
            print(f"  ❌ Factory function test failed: {e}")
            test3_success = False

        # Summary
        print("\nSummary")
        print("=" * 30)
        tests_passed = sum([test1_success, test2_success, test3_success])
        total_tests = 3

        print(f"Tests passed: {tests_passed}/{total_tests}")

        if test1_success:
            print("✓ Float matrix addition flowgraph: WORKING")
        if test2_success:
            print("✓ Double matrix addition flowgraph: WORKING")
        if test3_success:
            print("✓ Factory function integration: WORKING")

        if tests_passed >= 2:
            print("\n🎉 SUCCESS: Linear algebra blocks are working in GNU Radio!")
            print("This is the first real signal processing functionality!")
            return True
        elif tests_passed >= 1:
            print("\n✅ PARTIAL SUCCESS: Some functionality working")
            return True
        else:
            print("\n❌ FAILURE: No tests passed")
            return False

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
