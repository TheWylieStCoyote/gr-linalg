#!/usr/bin/env python3

"""
Test script to analyze GNU Radio Linear Algebra module implementation status
Tests which blocks can be imported and instantiated vs which fail
"""

import sys
import traceback
from collections import defaultdict


def test_block_import(block_name):
    """Test if a block can be imported from linalg_python C++ module"""
    try:
        # Try to import the C++ linalg_python module directly
        import linalg_python

        # Try to get the block class
        if hasattr(linalg_python, block_name):
            block_class = getattr(linalg_python, block_name)
            result = {
                "import_success": True,
                "class_exists": True,
                "block_class": block_class,
                "implementation": "c++",
                "error": None,
            }
        else:
            # Also check the linalg module for factory functions
            try:
                import linalg

                if hasattr(linalg, block_name):
                    block_class = getattr(linalg, block_name)
                    result = {
                        "import_success": True,
                        "class_exists": True,
                        "block_class": block_class,
                        "implementation": "factory",
                        "error": None,
                    }
                else:
                    result = {
                        "import_success": True,
                        "class_exists": False,
                        "block_class": None,
                        "implementation": "none",
                        "error": f"Class {block_name} not found in linalg_python or linalg modules",
                    }
            except:
                result = {
                    "import_success": True,
                    "class_exists": False,
                    "block_class": None,
                    "implementation": "none",
                    "error": f"Class {block_name} not found in linalg_python module",
                }

    except Exception as e:
        result = {
            "import_success": False,
            "class_exists": False,
            "block_class": None,
            "implementation": "none",
            "error": str(e),
        }

    return result


def test_block_instantiation(block_class, test_args):
    """Test if a block can be instantiated with given arguments"""
    if block_class is None:
        return {"instantiation_success": False, "error": "No block class"}

    try:
        # Try to instantiate with test arguments
        instance = block_class(*test_args)
        return {"instantiation_success": True, "instance": instance, "error": None}
    except Exception as e:
        return {"instantiation_success": False, "instance": None, "error": str(e)}


def analyze_linalg_blocks():
    """Analyze all major linalg blocks"""

    print("=" * 80)
    print("GNU Radio Linear Algebra Module - Block Functionality Analysis")
    print("=" * 80)

    # List of blocks to test with their expected class names and test parameters
    # Using actual C++ class names from linalg_python module
    blocks_to_test = [
        # Core Matrix Operations (float versions available)
        ("matrix_add_sync_f", [(2, 2), (2, 2)]),
        ("matrix_subtract_sync_f", [(2, 2), (2, 2)]),
        ("matrix_multiply_sync_f", [(2, 3), (3, 2)]),
        ("matrix_inverse_sync_f", [(2, 2)]),
        ("matrix_transpose_sync_f", [(2, 3)]),
        ("matrix_determinant_sync_f", [(2, 2)]),
        ("matrix_trace_sync_f", [(2, 2)]),
        ("matrix_norm_sync_f", [(2, 2)]),
        ("matrix_rank_sync_f", [(2, 2)]),
        ("matrix_condition_number_sync_f", [(2, 2)]),
        # Element-wise Operations
        ("matrix_elementwise_multiply_sync_f", [(2, 2), (2, 2)]),
        ("matrix_elementwise_divide_sync_f", [(2, 2), (2, 2)]),
        # Advanced Matrix Operations
        ("matrix_power_sync_f", [(2, 2)]),
        ("matrix_solve_sync_f", [(2, 2), (2, 1)]),
        ("matrix_hermitian_sync_f", [(2, 2)]),
        ("matrix_pseudo_inverse_sync_f", [(2, 2)]),
        ("matrix_reshape_sync_f", [(2, 3)]),
        # Vector Operations
        ("dot_product_sync_f", [(3,), (3,)]),
        ("vector_norm_sync_f", [(3,)]),
        ("vector_normalize_sync_f", [(3,)]),
        ("vector_cross_product_sync_f", [(3,), (3,)]),
        ("vector_outer_product_sync_f", [(3,), (4,)]),
        ("vector_angle_sync_f", [(3,), (3,)]),
        # Decompositions
        ("decomp_svd_sync_f", [(2, 2)]),
        ("decomp_qr_sync_f", [(2, 2)]),
        ("decomp_lu_sync_f", [(2, 2)]),
        ("decomp_cholesky_sync_f", [(2, 2)]),
        ("decomp_eigen_sync_f", [(2, 2)]),
        # Utility Blocks
        ("matrix_source_const_sync_f", [(2, 2)]),
        ("eye_sync_f", [(2,)]),
        ("zeros_sync_f", [(2, 2)]),
        ("ones_sync_f", [(2, 2)]),
        # Complex versions (some key ones)
        ("matrix_multiply_sync_c", [(2, 3), (3, 2)]),
        ("matrix_inverse_sync_c", [(2, 2)]),
        ("decomp_lu_sync_c", [(2, 2)]),
        ("dot_product_sync_c", [(3,), (3,)]),
    ]

    # Test results categorization
    results = {
        "fully_working": [],
        "import_ok_instantiate_fail": [],
        "import_fail": [],
        "class_missing": [],
    }

    for block_name, test_args in blocks_to_test:
        print(f"\nTesting: {block_name}")
        print("-" * 50)

        # Test import and class existence
        import_result = test_block_import(block_name)

        if not import_result["import_success"]:
            print(f"❌ IMPORT FAILED: {import_result['error']}")
            results["import_fail"].append(
                ("linalg", block_name, import_result["error"])
            )
            continue

        if not import_result["class_exists"]:
            print(f"❌ CLASS MISSING: {import_result['error']}")
            results["class_missing"].append(
                ("linalg", block_name, import_result["error"])
            )
            continue

        impl_type = import_result.get("implementation", "unknown")
        print(f"✅ Import successful: {import_result['block_class']} ({impl_type})")

        # Test instantiation
        instantiation_result = test_block_instantiation(
            import_result["block_class"], test_args
        )

        if instantiation_result["instantiation_success"]:
            print(f"✅ Instantiation successful")
            results["fully_working"].append(("linalg", block_name))
        else:
            print(f"❌ INSTANTIATION FAILED: {instantiation_result['error']}")
            results["import_ok_instantiate_fail"].append(
                ("linalg", block_name, instantiation_result["error"])
            )

    return results


def print_summary(results):
    """Print a summary of test results"""

    print("\n" + "=" * 80)
    print("SUMMARY - GNU Radio Linear Algebra Module Status")
    print("=" * 80)

    print(f"\n🟢 FULLY WORKING BLOCKS ({len(results['fully_working'])})")
    print("-" * 50)
    for module_path, block_name in results["fully_working"]:
        print(f"  ✅ {block_name}")

    print(
        f"\n🟡 IMPORT OK, INSTANTIATION FAILS ({len(results['import_ok_instantiate_fail'])})"
    )
    print("-" * 50)
    for module_path, block_name, error in results["import_ok_instantiate_fail"]:
        print(f"  ⚠️  {block_name}: {error[:60]}...")

    print(f"\n🔴 CLASS MISSING IN MODULE ({len(results['class_missing'])})")
    print("-" * 50)
    for module_path, block_name, error in results["class_missing"]:
        print(f"  ❌ {block_name}: {error}")

    print(f"\n🔴 MODULE IMPORT FAILS ({len(results['import_fail'])})")
    print("-" * 50)
    for module_path, block_name, error in results["import_fail"]:
        print(f"  ❌ {module_path}: {error[:60]}...")

    # Calculate percentages
    total_blocks = sum(len(results[key]) for key in results.keys())
    if total_blocks > 0:
        working_pct = (len(results["fully_working"]) / total_blocks) * 100
        print(f"\n📊 OVERALL STATUS")
        print(
            f"   Working: {len(results['fully_working'])}/{total_blocks} ({working_pct:.1f}%)"
        )


if __name__ == "__main__":
    try:
        # Add the Python module paths
        import os

        os.environ["PYTHONPATH"] = (
            "/home/wylie/Documents/work/gnuradio/_gr-linalg/build/python/linalg/bindings:"
            + os.environ.get("PYTHONPATH", "")
        )
        sys.path.insert(0, "/home/wylie/Documents/work/gnuradio/_gr-linalg/python")
        sys.path.insert(
            0,
            "/home/wylie/Documents/work/gnuradio/_gr-linalg/build/python/linalg/bindings",
        )

        results = analyze_linalg_blocks()
        print_summary(results)

    except Exception as e:
        print(f"Critical error during testing: {e}")
        traceback.print_exc()
