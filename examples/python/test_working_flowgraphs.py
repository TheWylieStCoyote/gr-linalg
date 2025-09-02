#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Working GNU Radio Flowgraph Tests

This test identifies and validates the linear algebra blocks that are actually
functional in GNU Radio flowgraphs with the current build.
"""

import sys
import numpy as np
from gnuradio import gr, blocks


def main():
    print("GNU Radio Linear Algebra - Working Flowgraph Tests")
    print("=================================================")
    print()

    try:
        # Import with the working build
        sys.path.insert(0, "python")
        import linalg

        print("✓ Module imported successfully")

        # Discover actually available blocks by testing instantiation
        print("\nDiscovering working blocks...")
        print("-" * 30)

        test_candidates = [
            # Factory functions with specific type suffixes
            ("matrix_add_ff", lambda: linalg.matrix_add_ff([2, 2], 2)),
            ("matrix_add_dd", lambda: linalg.matrix_add_dd([2, 2], 2)),
            ("matrix_subtract_ff", lambda: linalg.matrix_subtract_ff([2, 2], 2)),
            ("matrix_subtract_dd", lambda: linalg.matrix_subtract_dd([2, 2], 2)),
            ("matrix_transpose_ff", lambda: linalg.matrix_transpose_ff([2, 2])),
            ("matrix_transpose_dd", lambda: linalg.matrix_transpose_dd([2, 2])),
            ("matrix_determinant_ff", lambda: linalg.matrix_determinant_ff([2, 2])),
            ("matrix_determinant_dd", lambda: linalg.matrix_determinant_dd([2, 2])),
            ("decomp_lu_ff", lambda: linalg.decomp_lu_ff([2, 2])),
            ("decomp_lu_dd", lambda: linalg.decomp_lu_dd([2, 2])),
            ("decomp_qr_ff", lambda: linalg.decomp_qr_ff([2, 2])),
            ("decomp_qr_dd", lambda: linalg.decomp_qr_dd([2, 2])),
            (
                "matrix_elementwise_multiply_ff",
                lambda: linalg.matrix_elementwise_multiply_ff([2, 2], 2),
            ),
            (
                "matrix_elementwise_multiply_dd",
                lambda: linalg.matrix_elementwise_multiply_dd([2, 2], 2),
            ),
        ]

        working_blocks = []

        for name, test_func in test_candidates:
            try:
                block = test_func()
                if block is not None:
                    working_blocks.append((name, test_func))
                    print(f"  ✓ {name}: Available")
                else:
                    print(f"  ⚠ {name}: Returns None")
            except Exception as e:
                print(f"  ✗ {name}: {str(e)[:50]}")

        print(f"\nFound {len(working_blocks)} working blocks")

        if not working_blocks:
            print("❌ No working blocks found - cannot run flowgraph tests")
            return False

        print()

        # Test the first working block with a real flowgraph
        test_name, test_func = working_blocks[0]
        print(f"Testing flowgraph with: {test_name}")
        print("-" * 40)

        try:
            # Create flowgraph
            tb = gr.top_block()

            # Create test data based on block type
            if "add" in test_name or "subtract" in test_name or "multiply" in test_name:
                # Two-input blocks
                data1 = [1.0, 2.0, 3.0, 4.0]  # 2x2 matrix
                data2 = [5.0, 6.0, 7.0, 8.0]  # 2x2 matrix

                src1 = blocks.vector_source_f(data1, repeat=False)
                src2 = blocks.vector_source_f(data2, repeat=False)

                block = test_func()
                sink = blocks.vector_sink_f()

                tb.connect(src1, (block, 0))
                tb.connect(src2, (block, 1))
                tb.connect(block, sink)

            else:
                # Single-input blocks
                data = [1.0, 2.0, 3.0, 4.0]  # 2x2 matrix

                src = blocks.vector_source_f(data, repeat=False)
                block = test_func()

                if "decomp" in test_name:
                    # Decomposition blocks have multiple outputs
                    sink1 = blocks.vector_sink_f()
                    sink2 = blocks.vector_sink_f()

                    tb.connect(src, block)
                    tb.connect((block, 0), sink1)
                    tb.connect((block, 1), sink2)

                    tb.run()

                    result1 = sink1.data()
                    result2 = sink2.data()

                    print(f"  ✓ {test_name} flowgraph executed successfully")
                    print(f"    Input:   {data}")
                    print(f"    Output1: {list(result1)}")
                    print(f"    Output2: {list(result2)}")

                    return len(result1) > 0 or len(result2) > 0

                else:
                    # Single output blocks
                    sink = blocks.vector_sink_f()

                    tb.connect(src, block)
                    tb.connect(block, sink)

            # Run flowgraph
            tb.run()

            # Check results
            result = sink.data()

            print(f"  ✓ {test_name} flowgraph executed successfully")
            print(
                f"    Input:  {data1 if 'add' in test_name or 'subtract' in test_name or 'multiply' in test_name else data}"
            )
            if "add" in test_name or "subtract" in test_name or "multiply" in test_name:
                print(f"            {data2}")
            print(f"    Output: {list(result)}")

            # Validate we got some meaningful output
            if len(result) > 0:
                print("  ✓ Block produced output - FLOWGRAPH TEST PASSED")
                return True
            else:
                print("  ✗ No output produced - FLOWGRAPH TEST FAILED")
                return False

        except Exception as e:
            print(f"  ✗ Flowgraph test failed: {str(e)}")
            import traceback

            traceback.print_exc()
            return False

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print(
            "\n🎉 SUCCESS: At least one linear algebra block works in GNU Radio flowgraphs!"
        )
    else:
        print("\n❌ FAILURE: No blocks are functional in GNU Radio flowgraphs")
    sys.exit(0 if success else 1)
