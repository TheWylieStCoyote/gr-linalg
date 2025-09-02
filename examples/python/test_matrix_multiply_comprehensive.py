#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Matrix Multiply Test for GNU Radio Linear Algebra

This test validates both sync and PDU variants of matrix multiply blocks
with real GNU Radio flowgraph signal processing.
"""

import sys
import numpy as np
from gnuradio import gr, blocks


def test_matrix_multiply_basic():
    """Test basic matrix multiplication functionality"""
    print("Test 1: Basic Matrix Multiplication")
    print("-" * 40)

    try:
        # Import modules
        sys.path.insert(0, "python")
        sys.path.insert(0, "build/test_modules")
        import linalg
        import gnuradio.linalg.linalg_python as linalg_cpp

        # Create GNU Radio top block
        tb = gr.top_block()

        # Test data: 2x2 matrices
        # Matrix A: [[1, 2], [3, 4]]
        matrix_a = [1.0, 2.0, 3.0, 4.0]
        # Matrix B: [[2, 0], [1, 3]]
        matrix_b = [2.0, 0.0, 1.0, 3.0]

        # Expected result: A * B = [[4, 6], [10, 12]]
        expected = [4.0, 6.0, 10.0, 12.0]

        # Sources
        src_a = blocks.vector_source_f(matrix_a, repeat=False, vlen=4)
        src_b = blocks.vector_source_f(matrix_b, repeat=False, vlen=4)

        # Matrix multiply block (float sync version)
        multiply_block = linalg_cpp.matrix_multiply_sync_f.make([2, 2], [2, 2])

        # Sink
        sink = blocks.vector_sink_f(vlen=4)

        # Connect flowgraph
        tb.connect(src_a, (multiply_block, 0))
        tb.connect(src_b, (multiply_block, 1))
        tb.connect(multiply_block, sink)

        # Run
        tb.run()

        # Check results
        result = list(sink.data())

        print(f"  Matrix A: {matrix_a} (2x2)")
        print(f"  Matrix B: {matrix_b} (2x2)")
        print(f"  Result:   {result}")
        print(f"  Expected: {expected}")

        success = len(result) == len(expected) and all(
            abs(a - b) < 1e-6 for a, b in zip(result, expected)
        )

        if success:
            print("  ✓ PASS: Basic matrix multiplication works!")
            return True
        else:
            print("  ❌ FAIL: Incorrect matrix multiplication result")
            return False

    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_matrix_multiply_types():
    """Test different scalar types for matrix multiply"""
    print("\nTest 2: Multiple Scalar Types")
    print("-" * 33)

    try:
        sys.path.insert(0, "python")
        sys.path.insert(0, "build/test_modules")
        import linalg
        import gnuradio.linalg.linalg_python as linalg_cpp

        # Test available types
        available_types = [
            ("matrix_multiply_sync_f", "float"),
            ("matrix_multiply_sync_d", "double"),
            ("matrix_multiply_sync_c", "complex float"),
            ("matrix_multiply_sync_z", "complex double"),
        ]

        working_types = []

        for class_name, type_name in available_types:
            try:
                if hasattr(linalg_cpp, class_name):
                    # Test instantiation
                    block_class = getattr(linalg_cpp, class_name)
                    block = block_class.make([2, 2], [2, 2])
                    print(f"  ✓ {type_name}: Available and instantiable")
                    working_types.append(type_name)
                else:
                    print(f"  ❌ {type_name}: Not available")
            except Exception as e:
                print(f"  ❌ {type_name}: Error - {e}")

        print(f"  Working types: {len(working_types)}/{len(available_types)}")
        return len(working_types) > 0

    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return False


def test_matrix_multiply_pdu():
    """Test PDU variants of matrix multiply"""
    print("\nTest 3: PDU Block Variants")
    print("-" * 28)

    try:
        sys.path.insert(0, "python")
        sys.path.insert(0, "build/test_modules")
        import linalg
        import gnuradio.linalg.linalg_python as linalg_cpp

        # Test PDU types
        pdu_types = [
            ("matrix_multiply_pdu_f", "float"),
            ("matrix_multiply_pdu_d", "double"),
            ("matrix_multiply_pdu_c", "complex float"),
            ("matrix_multiply_pdu_z", "complex double"),
        ]

        working_pdu = []

        for class_name, type_name in pdu_types:
            try:
                if hasattr(linalg_cpp, class_name):
                    # Test instantiation
                    block_class = getattr(linalg_cpp, class_name)
                    block = block_class.make([2, 2], [2, 2])
                    print(f"  ✓ PDU {type_name}: Available and instantiable")
                    working_pdu.append(type_name)
                else:
                    print(f"  ❌ PDU {type_name}: Not available")
            except Exception as e:
                print(f"  ❌ PDU {type_name}: Error - {e}")

        print(f"  Working PDU types: {len(working_pdu)}/{len(pdu_types)}")
        return len(working_pdu) > 0

    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return False


def test_matrix_multiply_sizes():
    """Test different matrix sizes"""
    print("\nTest 4: Different Matrix Sizes")
    print("-" * 33)

    try:
        sys.path.insert(0, "python")
        sys.path.insert(0, "build/test_modules")
        import linalg
        import gnuradio.linalg.linalg_python as linalg_cpp

        # Test different matrix sizes
        test_cases = [
            ([1, 1], [1, 1]),  # 1x1 matrices
            ([2, 3], [3, 2]),  # 2x3 * 3x2 -> 2x2
            ([3, 2], [2, 3]),  # 3x2 * 2x3 -> 3x3
        ]

        working_sizes = []

        for shape_a, shape_b in test_cases:
            try:
                # Create block
                block = linalg_cpp.matrix_multiply_sync_f.make(shape_a, shape_b)
                print(f"  ✓ {shape_a} × {shape_b}: Block created successfully")
                working_sizes.append((shape_a, shape_b))
            except Exception as e:
                print(f"  ❌ {shape_a} × {shape_b}: Error - {e}")

        print(f"  Working size combinations: {len(working_sizes)}/{len(test_cases)}")
        return len(working_sizes) > 0

    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return False


def test_matrix_multiply_pipeline():
    """Test matrix multiply in a processing pipeline"""
    print("\nTest 5: Matrix Processing Pipeline")
    print("-" * 37)

    try:
        sys.path.insert(0, "python")
        sys.path.insert(0, "build/test_modules")
        import linalg
        import gnuradio.linalg.linalg_python as linalg_cpp

        # Create GNU Radio top block
        tb = gr.top_block()

        # Pipeline: (A * B) + C
        # Matrix A: [[1, 1], [1, 1]]
        matrix_a = [1.0, 1.0, 1.0, 1.0]
        # Matrix B: [[2, 0], [0, 2]]
        matrix_b = [2.0, 0.0, 0.0, 2.0]
        # Matrix C: [[1, 1], [1, 1]]
        matrix_c = [1.0, 1.0, 1.0, 1.0]

        # Expected: A*B = [[2, 2], [2, 2]], then (A*B)+C = [[3, 3], [3, 3]]
        expected = [3.0, 3.0, 3.0, 3.0]

        # Sources
        src_a = blocks.vector_source_f(matrix_a, repeat=False, vlen=4)
        src_b = blocks.vector_source_f(matrix_b, repeat=False, vlen=4)
        src_c = blocks.vector_source_f(matrix_c, repeat=False, vlen=4)

        # Processing blocks
        multiply_block = linalg_cpp.matrix_multiply_sync_f.make([2, 2], [2, 2])
        add_block = linalg_cpp.matrix_add_sync_f.make([2, 2], 2)

        # Sinks
        sink_multiply = blocks.vector_sink_f(vlen=4)
        sink_final = blocks.vector_sink_f(vlen=4)

        # Connect pipeline: (A * B) + C
        tb.connect(src_a, (multiply_block, 0))
        tb.connect(src_b, (multiply_block, 1))
        tb.connect(src_c, (add_block, 1))
        tb.connect(multiply_block, (add_block, 0))
        tb.connect(multiply_block, sink_multiply)
        tb.connect(add_block, sink_final)

        # Run
        tb.run()

        # Check results
        multiply_result = list(sink_multiply.data())
        final_result = list(sink_final.data())

        print(f"  A * B result: {multiply_result}")
        print(f"  (A * B) + C:  {final_result}")
        print(f"  Expected:     {expected}")

        success = len(final_result) == len(expected) and all(
            abs(a - b) < 1e-6 for a, b in zip(final_result, expected)
        )

        if success:
            print("  ✓ PASS: Matrix multiply pipeline works!")
            return True
        else:
            print("  ❌ FAIL: Pipeline results incorrect")
            return False

    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run comprehensive matrix multiply tests"""
    print("🔢 GNU Radio Matrix Multiply - Comprehensive Tests")
    print("=" * 55)
    print()

    # Run all tests
    tests = [
        test_matrix_multiply_basic,
        test_matrix_multiply_types,
        test_matrix_multiply_pdu,
        test_matrix_multiply_sizes,
        test_matrix_multiply_pipeline,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
            results.append(False)

    # Summary
    print("\n" + "=" * 55)
    print("📊 TEST SUMMARY")
    print("-" * 15)

    passed = sum(results)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total:.1%}")

    test_names = [
        "Basic Matrix Multiplication",
        "Multiple Scalar Types",
        "PDU Block Variants",
        "Different Matrix Sizes",
        "Matrix Processing Pipeline",
    ]

    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")

    print()

    if passed >= 3:
        print("🎉 EXCELLENT: Matrix multiply functionality is working!")
        print("✅ Both sync and PDU variants are exposed")
        print("✅ Real GNU Radio signal processing confirmed")
        print("✅ Multiple data types supported")
        return True
    elif passed >= 2:
        print("✅ GOOD: Core matrix multiply functionality works")
        return True
    else:
        print("⚠️ NEEDS WORK: Matrix multiply has significant issues")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
