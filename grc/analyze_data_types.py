#!/usr/bin/env python3
"""
Analyze GRC blocks for incorrect input/output data types.
"""

import yaml
import os
import re
from pathlib import Path


def analyze_data_types(filepath):
    """Analyze a GRC block for data type issues."""
    issues = []

    try:
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        return [f"YAML error: {e}"]

    if not data:
        return ["Empty YAML file"]

    # Skip files with placeholders
    if "parametername_replace_me" in str(data) or "FIX ME" in str(data):
        return ["Contains placeholders - skipping"]

    # Check inputs
    if "inputs" in data:
        for i, inp in enumerate(data["inputs"]):
            if isinstance(inp, dict):
                if "dtype" in inp:
                    dtype = inp["dtype"]
                    # Common issues:
                    # 1. Using ${type} instead of ${type.t}
                    if dtype == "${type}":
                        issues.append(
                            f"Input {i}: dtype should be '${{type.t}}' not '${{type}}'"
                        )
                    # 2. Hardcoded types when should be variable
                    elif dtype in ["float", "double", "complex"]:
                        issues.append(
                            f"Input {i}: hardcoded dtype '{dtype}' should be '${{type.t}}'"
                        )
                    # 3. Missing dtype
                else:
                    issues.append(f"Input {i}: missing dtype field")

    # Check outputs
    if "outputs" in data:
        for i, out in enumerate(data["outputs"]):
            if isinstance(out, dict):
                if "dtype" in out:
                    dtype = out["dtype"]
                    # Common issues:
                    # 1. Using ${type} instead of ${type.t}
                    if dtype == "${type}":
                        issues.append(
                            f"Output {i}: dtype should be '${{type.t}}' not '${{type}}'"
                        )
                    # 2. Hardcoded types when should be variable
                    elif dtype in ["float", "double", "complex"]:
                        issues.append(
                            f"Output {i}: hardcoded dtype '{dtype}' should be '${{type.t}}'"
                        )
                    # 3. Missing dtype
                else:
                    issues.append(f"Output {i}: missing dtype field")

    # Check for type parameter consistency
    if "parameters" in data:
        has_type_param = False
        for param in data["parameters"]:
            if isinstance(param, dict) and param.get("id") == "type":
                has_type_param = True
                # Check if type parameter has correct attributes
                if "option_attributes" in param:
                    attrs = param["option_attributes"]
                    if "t" not in attrs:
                        issues.append(
                            "Type parameter missing 't' attribute for data types"
                        )
                break

        # If using ${type.t} but no type parameter
        if not has_type_param and ("${type.t}" in str(data) or "${type}" in str(data)):
            issues.append("Uses type variables but has no 'type' parameter")

    return issues


def main():
    """Main analysis function."""
    grc_dir = Path("grc")
    if not grc_dir.exists():
        print("❌ GRC directory not found!")
        return

    yaml_files = list(grc_dir.glob("*.block.yml"))
    if not yaml_files:
        print("❌ No .block.yml files found!")
        return

    print(f"🔍 Analyzing {len(yaml_files)} GRC blocks for data type issues...")
    print("=" * 70)

    blocks_with_issues = []
    blocks_with_placeholders = []
    clean_blocks = []

    for filepath in sorted(yaml_files):
        print(f"📄 {filepath.name}")
        issues = analyze_data_types(filepath)

        if not issues:
            clean_blocks.append(filepath.name)
            print("   ✅ Data types correct")
        elif len(issues) == 1 and "placeholders" in issues[0]:
            blocks_with_placeholders.append(filepath.name)
            print(f"   ⚠️  {issues[0]}")
        else:
            blocks_with_issues.append((filepath.name, issues))
            print(f"   ❌ {len(issues)} issues:")
            for issue in issues:
                print(f"      • {issue}")
        print()

    print("=" * 70)
    print(f"📊 ANALYSIS SUMMARY:")
    print(f"   📁 Files analyzed: {len(yaml_files)}")
    print(f"   ✅ Clean blocks: {len(clean_blocks)}")
    print(f"   ❌ Blocks with data type issues: {len(blocks_with_issues)}")
    print(f"   ⚠️  Blocks with placeholders: {len(blocks_with_placeholders)}")

    if blocks_with_issues:
        print(f"\n🔧 BLOCKS NEEDING FIXES:")
        for block_name, issues in blocks_with_issues:
            print(f"   • {block_name}: {len(issues)} issues")

    return blocks_with_issues, clean_blocks, blocks_with_placeholders


if __name__ == "__main__":
    main()
