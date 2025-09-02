#!/usr/bin/env python3
"""
Comprehensive validation script for all GRC block YAML files.
Checks syntax, completeness, and proper structure.
"""

import yaml
import os
import sys
from pathlib import Path


def validate_grc_file(filepath):
    """Validate a single GRC YAML file."""
    errors = []
    warnings = []

    try:
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return [f"YAML syntax error: {e}"], []
    except FileNotFoundError:
        return [f"File not found: {filepath}"], []

    # Check required top-level fields
    required_fields = [
        "id",
        "label",
        "category",
        "templates",
        "parameters",
        "inputs",
        "outputs",
        "documentation",
    ]
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    # Check for placeholder content
    if "FIX ME" in str(data) or "parametername_replace_me" in str(data):
        errors.append(
            "Contains placeholder content (FIX ME or parametername_replace_me)"
        )

    # Validate templates section
    if "templates" in data:
        templates = data["templates"]
        if "imports" not in templates:
            errors.append("Missing templates.imports")
        elif "from gnuradio import linalg" not in templates["imports"]:
            warnings.append(
                "Templates.imports doesn't include 'from gnuradio import linalg'"
            )

        if "make" not in templates:
            errors.append("Missing templates.make")
        elif not templates["make"].startswith("linalg."):
            warnings.append("Templates.make doesn't start with 'linalg.'")

    # Validate parameters section
    if "parameters" in data:
        params = data["parameters"]
        if not isinstance(params, list):
            errors.append("Parameters should be a list")
        else:
            # Check for type parameter
            has_type_param = any(
                p.get("id") == "type" for p in params if isinstance(p, dict)
            )
            if not has_type_param:
                warnings.append(
                    "No 'type' parameter found (may be intentional for some blocks)"
                )

    # Validate inputs/outputs
    for section in ["inputs", "outputs"]:
        if section in data:
            ios = data[section]
            if not isinstance(ios, list):
                errors.append(f"{section} should be a list")
            else:
                for i, io in enumerate(ios):
                    if not isinstance(io, dict):
                        errors.append(f"{section}[{i}] should be a dictionary")
                    elif section == "inputs" and "domain" not in io:
                        errors.append(f"{section}[{i}] missing 'domain' field")
                    elif section == "outputs" and "domain" not in io:
                        errors.append(f"{section}[{i}] missing 'domain' field")

    # Check documentation
    if "documentation" in data:
        doc = data["documentation"]
        if not isinstance(doc, str) or len(doc.strip()) < 50:
            warnings.append("Documentation appears to be too short or not a string")

    return errors, warnings


def main():
    """Main validation function."""
    grc_dir = Path("grc")
    if not grc_dir.exists():
        print("❌ GRC directory not found!")
        return 1

    yaml_files = list(grc_dir.glob("*.block.yml"))
    if not yaml_files:
        print("❌ No .block.yml files found in grc directory!")
        return 1

    print(f"🔍 Validating {len(yaml_files)} GRC block files...")
    print("=" * 60)

    total_errors = 0
    total_warnings = 0

    for filepath in sorted(yaml_files):
        print(f"📄 {filepath.name}")
        errors, warnings = validate_grc_file(filepath)

        if errors:
            total_errors += len(errors)
            print(f"   ❌ {len(errors)} errors:")
            for error in errors:
                print(f"      • {error}")

        if warnings:
            total_warnings += len(warnings)
            print(f"   ⚠️  {len(warnings)} warnings:")
            for warning in warnings:
                print(f"      • {warning}")

        if not errors and not warnings:
            print("   ✅ Valid")

        print()

    print("=" * 60)
    print(f"📊 VALIDATION SUMMARY:")
    print(f"   📁 Files checked: {len(yaml_files)}")
    print(f"   ❌ Total errors: {total_errors}")
    print(f"   ⚠️  Total warnings: {total_warnings}")

    if total_errors == 0:
        print("🎉 ALL GRC FILES ARE VALID!")
        return 0
    else:
        print("❌ Some files have errors that need to be fixed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
