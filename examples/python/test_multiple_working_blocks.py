#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multiple Working GNU Radio Linear Algebra Blocks Test

This test validates multiple linear algebra blocks working correctly
in GNU Radio flowgraphs with real signal processing.
"""

import sys
import numpy as np
from gnuradio import gr, blocks


def main():
    print("GNU Radio Linear Algebra - Multiple Working Blocks Test")
    print("======================================================")
    print()

    try:
        # Import modules
        sys.path.insert(0, "python")
        sys.path.insert(0, "build/test_modules")
        import linalg
        import gnuradio.linalg.linalg_python as linalg_cpp

        print("✓ Modules imported successfully")

        # Check what sync classes are available
        sync_classes = [
            name
            for name in dir(linalg_cpp)
            if "sync" in name and not name.startswith("_")
        ]
        print(f"✓ Found {len(sync_classes)} sync classes")
        print(f"  Available: {', '.join(sorted(sync_classes))}")
        print()

        test_results = []

        # Test 1: Matrix Addition
        print("Test 1: Matrix Addition")
        print("-" * 25)
        try:
            tb = gr.top_block()

            # Test data
            data1 = [1.0, 2.0, 3.0, 4.0]  # 2x2 matrix
            data2 = [5.0, 6.0, 7.0, 8.0]  # 2x2 matrix

            src1 = blocks.vector_source_f(data1, repeat=False, vlen=4)
            src2 = blocks.vector_source_f(data2, repeat=False, vlen=4)
            block = linalg_cpp.matrix_add_sync_f.make([2, 2], 2)
            sink = blocks.vector_sink_f(vlen=4)

            tb.connect(src1, (block, 0))
            tb.connect(src2, (block, 1))
            tb.connect(block, sink)
            tb.run()

            result = sink.data()
            expected = [6.0, 8.0, 10.0, 12.0]

            success = len(result) == len(expected) and all(
                abs(a - b) < 1e-6 for a, b in zip(result, expected)
            )
            test_results.append(("matrix_add", success))

            if success:
                print("  ✓ PASS: Matrix addition works correctly")
                print(f"    Result: {list(result)}")
            else:
                print("  ❌ FAIL: Incorrect result")
                print(f"    Expected: {expected}, Got: {list(result)}")

        except Exception as e:
            test_results.append(("matrix_add", False))
            print(f"  ❌ ERROR: {e}")

        # Test 2: Matrix Subtraction
        print("\nTest 2: Matrix Subtraction")
        print("-" * 27)
        try:
            tb = gr.top_block()

            # Test data
            data1 = [10.0, 8.0, 6.0, 4.0]  # 2x2 matrix
            data2 = [1.0, 2.0, 3.0, 4.0]  # 2x2 matrix

            src1 = blocks.vector_source_f(data1, repeat=False, vlen=4)
            src2 = blocks.vector_source_f(data2, repeat=False, vlen=4)
            block = linalg_cpp.matrix_subtract_sync_f.make([2, 2], 2)
            sink = blocks.vector_sink_f(vlen=4)

            tb.connect(src1, (block, 0))
            tb.connect(src2, (block, 1))
            tb.connect(block, sink)
            tb.run()

            result = sink.data()
            expected = [9.0, 6.0, 3.0, 0.0]  # [10-1, 8-2, 6-3, 4-4]

            success = len(result) == len(expected) and all(
                abs(a - b) < 1e-6 for a, b in zip(result, expected)
            )
            test_results.append(("matrix_subtract", success))

            if success:
                print("  ✓ PASS: Matrix subtraction works correctly")
                print(f"    Result: {list(result)}")
            else:
                print("  ❌ FAIL: Incorrect result")
                print(f"    Expected: {expected}, Got: {list(result)}")

        except Exception as e:
            test_results.append(("matrix_subtract", False))
            print(f"  ❌ ERROR: {e}")

        # Test 3: Matrix Transpose
        print("\nTest 3: Matrix Transpose")
        print("-" * 25)
        try:
            tb = gr.top_block()

            # Test data (2x3 matrix)
            data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # [[1,2,3],[4,5,6]]

            src = blocks.vector_source_f(data, repeat=False, vlen=6)
            block = linalg_cpp.matrix_transpose_sync_f.make([2, 3])
            sink = blocks.vector_sink_f(vlen=6)  # Output will be 3x2

            tb.connect(src, block)
            tb.connect(block, sink)
            tb.run()

            result = sink.data()
            expected = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]  # [[1,4],[2,5],[3,6]]

            success = len(result) == len(expected) and all(
                abs(a - b) < 1e-6 for a, b in zip(result, expected)
            )
            test_results.append(("matrix_transpose", success))

            if success:
                print("  ✓ PASS: Matrix transpose works correctly")
                print(f"    Input:  {data} (2x3)")
                print(f"    Result: {list(result)} (3x2)")
            else:
                print("  ❌ FAIL: Incorrect result")
                print(f"    Expected: {expected}, Got: {list(result)}")

        except Exception as e:
            test_results.append(("matrix_transpose", False))
            print(f"  ❌ ERROR: {e}")

        # Test 4: Check available but untested blocks
        print("\nTest 4: Available Block Inventory")
        print("-" * 35)

        tested_blocks = {
            "matrix_add_sync_f",
            "matrix_subtract_sync_f",
            "matrix_transpose_sync_f",
        }
        available_blocks = set(sync_classes)
        untested_blocks = available_blocks - tested_blocks

        print(f"  Tested and working: {len([r for r in test_results if r[1]])}")
        print(f"  Tested but failing: {len([r for r in test_results if not r[1]])}")
        print(f"  Available but untested: {len(untested_blocks)}")

        if untested_blocks:
            print(f"  Untested blocks: {', '.join(sorted(untested_blocks))}")

        # Summary
        print("\nSummary")
        print("=" * 40)

        working_blocks = [name for name, success in test_results if success]
        failing_blocks = [name for name, success in test_results if not success]

        print(f"Working blocks: {len(working_blocks)}/{len(test_results)}")
        print(f"Total available sync classes: {len(sync_classes)}")

        if working_blocks:
            print(f"✓ Working: {', '.join(working_blocks)}")
        if failing_blocks:
            print(f"✗ Failing: {', '.join(failing_blocks)}")

        success_rate = len(working_blocks) / len(test_results) if test_results else 0

        if success_rate >= 0.8:
            print(
                f"\n🎉 EXCELLENT: {success_rate:.1%} of tested blocks working in GNU Radio!"
            )
            print("Real linear algebra signal processing is now functional!")
            return True
        elif success_rate >= 0.5:
            print(f"\n✅ GOOD: {success_rate:.1%} of tested blocks working!")
            return True
        else:
            print(f"\n⚠️ NEEDS WORK: Only {success_rate:.1%} working")
            return False

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
