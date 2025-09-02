#!/usr/bin/env python3
"""
Script to compute MD5 hashes of header files in ./include/gnuradio/linalg/
and update the hashes in the Python bindings.

This script:
1. Computes MD5 hashes of all .h files in include/gnuradio/linalg/
2. Updates the BINDTOOL_HEADER_FILE_HASH comments in corresponding Python binding files
"""

import hashlib
import os
import re
import glob


def compute_md5_hash(file_path):
    """Compute MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except IOError as e:
        print(f"Error reading {file_path}: {e}")
        return None


def find_header_files():
    """Find all header files in include/gnuradio/linalg/."""
    header_dir = "./include/gnuradio/linalg/"
    if not os.path.exists(header_dir):
        print(f"Header directory {header_dir} not found!")
        return {}

    header_files = {}
    pattern = os.path.join(header_dir, "*.h")

    for header_path in glob.glob(pattern):
        basename = os.path.basename(header_path)
        md5_hash = compute_md5_hash(header_path)
        if md5_hash:
            header_files[basename] = md5_hash
            print(f"{basename}: {md5_hash}")

    return header_files


def update_python_binding_hash(binding_file, header_name, new_hash):
    """Update the BINDTOOL_HEADER_FILE_HASH in a Python binding file."""
    try:
        with open(binding_file, "r") as f:
            content = f.read()

        # Pattern to match BINDTOOL_HEADER_FILE_HASH line
        pattern = r"(\/\* BINDTOOL_HEADER_FILE_HASH\()([^)]+)(\).*\*\/)"

        def replace_hash(match):
            return f"{match.group(1)}{new_hash}{match.group(3)}"

        # Check if this binding file is for the correct header
        header_pattern = r"\/\* BINDTOOL_HEADER_FILE\(([^)]+)\).*\*\/"
        header_match = re.search(header_pattern, content)

        if header_match and header_match.group(1) == header_name:
            new_content = re.sub(pattern, replace_hash, content)

            if new_content != content:
                with open(binding_file, "w") as f:
                    f.write(new_content)
                print(f"Updated {binding_file} with hash {new_hash}")
                return True
            else:
                print(f"No hash update needed for {binding_file}")
                return False
        else:
            # File doesn't match this header
            return False

    except IOError as e:
        print(f"Error updating {binding_file}: {e}")
        return False


def find_python_binding_files():
    """Find all Python binding files."""
    bindings_dir = "./python/gnuradio/linalg/bindings/"
    if not os.path.exists(bindings_dir):
        print(f"Bindings directory {bindings_dir} not found!")
        return []

    pattern = os.path.join(bindings_dir, "*_python.cc")
    return glob.glob(pattern)


def main():
    """Main function."""
    print("Computing MD5 hashes for header files...")
    header_hashes = find_header_files()

    if not header_hashes:
        print("No header files found!")
        return

    print(f"\nFound {len(header_hashes)} header files")

    print("\nUpdating Python binding files...")
    binding_files = find_python_binding_files()

    if not binding_files:
        print("No Python binding files found!")
        return

    print(f"Found {len(binding_files)} binding files")

    updated_count = 0
    for binding_file in binding_files:
        for header_name, header_hash in header_hashes.items():
            if update_python_binding_hash(binding_file, header_name, header_hash):
                updated_count += 1
                break  # Each binding file should only match one header

    print(f"\nUpdated {updated_count} Python binding files")
    print("Hash update complete!")


if __name__ == "__main__":
    main()
