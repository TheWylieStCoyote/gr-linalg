#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNU Radio Linear Algebra Performance Profiler - Main Entry Point

This script provides the main entry point for the modular performance
profiling system. It imports and runs the CLI interface.
"""

import sys
from pathlib import Path

# Add the profiling module to the path
sys.path.insert(0, str(Path(__file__).parent))

try:
    # Try absolute import first
    from profiling.cli import main
except ImportError:
    try:
        # Try relative import
        from .cli import main
    except ImportError:
        try:
            # Try direct import from same directory
            import cli

            main = cli.main
        except ImportError as e:
            print(f"Error: Cannot import CLI module: {e}", file=sys.stderr)
            print(
                "Please ensure the profiling module is properly installed.",
                file=sys.stderr,
            )
            sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())
