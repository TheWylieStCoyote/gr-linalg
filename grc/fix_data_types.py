#!/usr/bin/env python3
"""
Fix incorrect data types in GRC blocks.
"""

import yaml
import os
import re
from pathlib import Path


def fix_data_types(filepath):
    """Fix data type issues in a GRC block file."""

    try:
        with open(filepath, "r") as f:
            content = f.read()
    except Exception as e:
        return f"Error reading {filepath.name}: {e}"

    # Skip files with YAML errors (placeholder files)
    if "parametername_replace_me" in content or "FIX ME:" in content:
        return f"Skipped {filepath.name} - contains placeholders"

    original_content = content
    changes_made = []

    # Fix 1: Change ${type} to ${type.t} in dtype fields
    type_pattern = r"(dtype: )\$\{type\}(?!\.|_)"
    if re.search(type_pattern, content):
        content = re.sub(type_pattern, r"\1${type.t}", content)
        changes_made.append("Changed dtype: ${type} to dtype: ${type.t}")

    # Fix 2: Update import statements to use proper format
    import_pattern = r"imports: import linalg"
    if re.search(import_pattern, content):
        content = re.sub(
            import_pattern, "imports: from gnuradio import linalg", content
        )
        changes_made.append("Fixed import statement")

    # Fix 3: Update make templates that are malformed
    # Look for make statements that don't follow the pattern
    make_pattern = r"(make: linalg\.)(\w+)_sync_(\$\{type\.t\})"
    if re.search(make_pattern, content):
        content = re.sub(make_pattern, r"\1\2_${type.fcn}", content)
        changes_made.append("Fixed make template pattern")

    # Fix 4: Add missing dtype fields to inputs/outputs that have domain but no dtype
    # This is more complex and requires YAML parsing for precision
    try:
        data = yaml.safe_load(content)
        if data and "inputs" in data:
            for i, inp in enumerate(data["inputs"]):
                if isinstance(inp, dict) and "domain" in inp and "dtype" not in inp:
                    # Add default dtype for missing fields
                    if "inputs:" in content:
                        # Find the input section and add dtype
                        input_section_pattern = r"(inputs:\n(?:.*?\n)*?- domain: \w+\n(?:.*?\n)*?)(  vlen:|outputs:|documentation:|\n\n|\nfile_format:)"

                        def add_dtype(match):
                            input_part = match.group(1)
                            rest = match.group(2) if match.group(2) else ""
                            if "dtype:" not in input_part:
                                input_part += "  dtype: ${type.t}\n"
                                changes_made.append(f"Added missing dtype to input {i}")
                            return input_part + rest

                        content = re.sub(
                            input_section_pattern,
                            add_dtype,
                            content,
                            flags=re.MULTILINE | re.DOTALL,
                        )

        if data and "outputs" in data:
            for i, out in enumerate(data["outputs"]):
                if isinstance(out, dict) and "domain" in out and "dtype" not in out:
                    # Add default dtype for missing fields
                    if "outputs:" in content:
                        output_section_pattern = r"(outputs:\n(?:.*?\n)*?- domain: \w+\n(?:.*?\n)*?)(  vlen:|documentation:|\nasserts:|\n\n|\nfile_format:)"

                        def add_output_dtype(match):
                            output_part = match.group(1)
                            rest = match.group(2) if match.group(2) else ""
                            if "dtype:" not in output_part:
                                output_part += "  dtype: ${type.t}\n"
                                changes_made.append(
                                    f"Added missing dtype to output {i}"
                                )
                            return output_part + rest

                        content = re.sub(
                            output_section_pattern,
                            add_output_dtype,
                            content,
                            flags=re.MULTILINE | re.DOTALL,
                        )

    except yaml.YAMLError:
        # Skip YAML processing if file has syntax errors
        pass

    # Write back if changes were made
    if content != original_content:
        with open(filepath, "w") as f:
            f.write(content)
        return f"Fixed {filepath.name}: {', '.join(changes_made)}"
    else:
        return f"No changes needed for {filepath.name}"


def main():
    """Main fix function."""
    grc_dir = Path("grc")
    if not grc_dir.exists():
        print("❌ GRC directory not found!")
        return

    yaml_files = list(grc_dir.glob("*.block.yml"))
    if not yaml_files:
        print("❌ No .block.yml files found!")
        return

    print(f"🔧 Fixing data type issues in {len(yaml_files)} GRC blocks...")
    print("=" * 70)

    fixed_blocks = []
    skipped_blocks = []
    no_change_blocks = []

    for filepath in sorted(yaml_files):
        result = fix_data_types(filepath)

        if "Fixed" in result:
            fixed_blocks.append(result)
            print(f"✅ {result}")
        elif "Skipped" in result:
            skipped_blocks.append(result)
            print(f"⚠️  {result}")
        else:
            no_change_blocks.append(result)
            print(f"ℹ️  {result}")

    print("=" * 70)
    print(f"📊 FIX SUMMARY:")
    print(f"   🔧 Blocks fixed: {len(fixed_blocks)}")
    print(f"   ⚠️  Blocks skipped: {len(skipped_blocks)}")
    print(f"   ✅ Blocks already correct: {len(no_change_blocks)}")

    if fixed_blocks:
        print(f"\n🎉 SUCCESSFULLY FIXED:")
        for result in fixed_blocks:
            print(f"   • {result.split(': ', 1)[0]}")


if __name__ == "__main__":
    main()
