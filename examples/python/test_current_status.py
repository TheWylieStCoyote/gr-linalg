#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Current Status Test

This test checks the current status of all blocks without requiring a rebuild.
It tests what's actually available in the current Python bindings.
"""

import sys
import os


def main():
    print("GNU Radio Linear Algebra Current Status Test")
    print("============================================")
    print()

    try:
        # Import with our existing working setup
        sys.path.insert(0, "python")
        import linalg

        print("✓ Successfully imported linalg module")

        # Get all available items
        all_items = [item for item in dir(linalg) if not item.startswith("_")]
        print(f"✓ Found {len(all_items)} available items")

        # Categorize items
        sync_classes = [
            item
            for item in all_items
            if "sync" in item and not item.endswith(("_f", "_d", "_cf", "_cd"))
        ]
        typed_classes = [
            item
            for item in all_items
            if item.endswith(("_f", "_d", "_cf", "_cd", "_ff", "_dd", "_cc", "_zz"))
        ]
        factory_functions = [
            item
            for item in all_items
            if not "sync" in item
            and not "pdu" in item
            and not item.endswith(("_f", "_d", "_cf", "_cd"))
        ]

        print(f"✓ Sync classes: {len(sync_classes)}")
        print(f"✓ Typed classes: {len(typed_classes)}")
        print(f"✓ Factory functions: {len(factory_functions)}")
        print()

        # Test the specific blocks we've been working on
        print("Testing Target Blocks (Previously Failing)")
        print("==========================================")

        target_blocks = [
            ("dot_product", lambda: linalg.dot_product([3], [3])),
            ("vector_dot", lambda: linalg.vector_dot([3], [3])),
            ("eye", lambda: linalg.eye([3, 3])),
            ("zeros", lambda: linalg.zeros([3, 3])),
            ("ones", lambda: linalg.ones([3, 3])),
            ("vector_cross_product", lambda: linalg.vector_cross_product(True)),
            (
                "matrix_kronecker_product",
                lambda: linalg.matrix_kronecker_product([2, 2], [2, 2]),
            ),
        ]

        working = []
        broken = []

        for name, test_func in target_blocks:
            print(f"Testing {name}...")
            try:
                if hasattr(linalg, name):
                    result = test_func()
                    if result is not None:
                        working.append(name)
                        print(f"  ✓ WORKING: Function exists and returns object")
                    else:
                        broken.append(f"{name} (returns None)")
                        print(f"  ⚠ PARTIAL: Function exists but returns None")
                else:
                    broken.append(f"{name} (missing)")
                    print(f"  ✗ MISSING: Function not found")
            except Exception as e:
                broken.append(f"{name} (error: {str(e)[:30]})")
                print(f"  ✗ ERROR: {str(e)[:50]}")

        print()
        print("Summary")
        print("=======")
        print(f"Working blocks: {len(working)}")
        print(f"Broken blocks: {len(broken)}")

        if working:
            print(f"✓ Working: {', '.join(working)}")
        if broken:
            print(f"✗ Issues: {', '.join(broken)}")

        print()

        # Check for underlying C++ classes
        print("C++ Class Availability Check")
        print("============================")

        cpp_classes_to_check = [
            "dot_product",
            "vector_dot",
            "eye",
            "zeros",
            "ones",
            "vector_cross_product_sync",
            "vector_cross_product_sync_d",
            "matrix_kronecker_product_sync",
            "matrix_kronecker_product_sync_d",
        ]

        found_classes = []
        missing_classes = []

        for cls_name in cpp_classes_to_check:
            if hasattr(linalg, cls_name):
                found_classes.append(cls_name)
                print(f"  ✓ {cls_name}")
            else:
                missing_classes.append(cls_name)
                print(f"  ✗ {cls_name}")

        print(f"\nC++ classes found: {len(found_classes)}")
        print(f"C++ classes missing: {len(missing_classes)}")

        # Final assessment
        print()
        print("Assessment")
        print("==========")

        success_rate = len(working) / len(target_blocks)
        cpp_availability = len(found_classes) / len(cpp_classes_to_check)

        print(f"Factory function success: {success_rate:.1%}")
        print(f"C++ class availability: {cpp_availability:.1%}")

        if success_rate >= 0.8:
            print("🎉 EXCELLENT: Most target blocks are working!")
            return True
        elif success_rate >= 0.5:
            print("✅ GOOD: Significant progress made!")
            return True
        elif len(found_classes) >= 3:
            print("⚠ PARTIAL: Some C++ classes available, bindings need work")
            return False
        else:
            print("❌ NEEDS WORK: Most blocks still not working")
            return False

    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
