#!/usr/bin/env python3
"""
Fix output domains to be conditional on interface type.
"""

import os
import re
from pathlib import Path


def fix_output_domains(filepath):
    """Fix output domains to be conditional."""

    with open(filepath, "r") as f:
        content = f.read()

    # Skip if no interface selector
    if "interface" not in content or "sync, pdu" not in content:
        return f"Skipped {filepath.name} - no interface selector"

    # Fix output domains that are still hardcoded to 'stream'
    # This handles both single and multiple outputs
    content = re.sub(
        r"(\n  domain: )stream",
        r"\1${'stream' if interface == 'sync' else 'message'}",
        content,
    )

    # Also fix the initial output domain line
    content = re.sub(
        r"(\noutputs:\n- .*?\n  domain: )stream",
        r"\1${'stream' if interface == 'sync' else 'message'}",
        content,
        flags=re.MULTILINE | re.DOTALL,
    )

    with open(filepath, "w") as f:
        f.write(content)

    return f"Fixed output domains in {filepath.name}"


def main():
    grc_dir = Path("grc")
    fixed_count = 0

    # Process all .block.yml files
    for filepath in grc_dir.glob("*.block.yml"):
        result = fix_output_domains(filepath)
        if "Fixed" in result:
            print(result)
            fixed_count += 1
        elif "Skipped" not in result:
            print(result)

    print(f"\nFixed output domains in {fixed_count} files")


if __name__ == "__main__":
    main()
