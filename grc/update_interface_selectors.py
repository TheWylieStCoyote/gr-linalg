#!/usr/bin/env python3
"""
Script to add interface selectors (sync/pdu) to GRC blocks that support both interfaces.
"""

import os
import re
from pathlib import Path

# Blocks that support both sync and PDU interfaces based on header analysis
DUAL_INTERFACE_BLOCKS = [
    "matrix_add",
    "matrix_subtract",
    "matrix_multiply",
    "matrix_transpose",
    "matrix_determinant",  # Already done
    "matrix_diag",
    "matrix_elementwise_multiply",
    "matrix_elementwise_divide",
    "matrix_inverse",
    "matrix_norm",
    "matrix_trace",
    "matrix_rank",
    "vector_norm",
    "vector_normalize",
    "vector_cross_product",
    "vector_outer_product",
    "dot_product",
    "decomp_svd",
    "decomp_eigen",
    "decomp_lu",
    "decomp_qr",
    "decomp_cholesky",
]


def update_block_with_interface_selector(filepath):
    """Add interface selector to a GRC block file."""

    with open(filepath, "r") as f:
        content = f.read()

    # Skip if already has interface selector
    if "interface" in content and "sync, pdu" in content:
        return f"Skipped {filepath.name} - already has interface selector"

    # Skip if contains placeholders
    if "parametername_replace_me" in content or "FIX ME" in content:
        return f"Skipped {filepath.name} - contains placeholders"

    # Pattern 1: Update make template to include interface
    make_pattern = r"(make: linalg\.[\w_]+)_(\$\{type\.fcn\})"
    make_replacement = r"\1_${interface}_\2"
    content = re.sub(make_pattern, make_replacement, content)

    # Pattern 2: Add interface parameter after any existing parameters but before type
    type_param_pattern = r"(parameters:\n)(- id: type)"
    interface_param = """- id: interface
  label: Interface
  dtype: enum
  default: sync
  options: [sync, pdu]
  option_labels: [Synchronous, PDU Messages]

"""
    content = re.sub(type_param_pattern, r"\1" + interface_param + r"\2", content)

    # Pattern 3: Update inputs domain to be conditional
    input_stream_pattern = r"(inputs:\n- domain: )stream"
    input_replacement = r"\1${'stream' if interface == 'sync' else 'message'}"
    content = re.sub(input_stream_pattern, input_replacement, content)

    # Pattern 4: Update outputs domain to be conditional
    output_stream_pattern = r"(outputs:\n- domain: )stream"
    output_replacement = r"\1${'stream' if interface == 'sync' else 'message'}"
    content = re.sub(output_stream_pattern, output_replacement, content)

    # Pattern 5: Handle multi-output blocks
    multi_output_pattern = r"(\n- domain: )stream"
    multi_output_replacement = r"\1${'stream' if interface == 'sync' else 'message'}"
    content = re.sub(multi_output_pattern, multi_output_replacement, content)

    # Pattern 6: Update vlen to be conditional for inputs
    input_vlen_pattern = r"(  vlen: \$\{[^}]+\})"

    def vlen_replacement(match):
        original_vlen = match.group(1).replace("  vlen: ${", "").replace("}", "")
        return f"  vlen: ${{{original_vlen} if interface == 'sync' else 1}}"

    content = re.sub(input_vlen_pattern, vlen_replacement, content)

    with open(filepath, "w") as f:
        f.write(content)

    return f"Updated {filepath.name}"


def main():
    grc_dir = Path("grc")
    updated_count = 0

    for block_name in DUAL_INTERFACE_BLOCKS:
        filepath = grc_dir / f"linalg_{block_name}.block.yml"
        if filepath.exists():
            result = update_block_with_interface_selector(filepath)
            print(result)
            if "Updated" in result:
                updated_count += 1
        else:
            print(f"File not found: {filepath}")

    print(f"\nUpdated {updated_count} GRC blocks with interface selectors")


if __name__ == "__main__":
    main()
