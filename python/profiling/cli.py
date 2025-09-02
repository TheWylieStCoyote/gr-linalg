#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-line interface for GNU Radio Linear Algebra Performance Profiler.

Main entry point that provides command-line argument parsing and coordinates
the various profiling components.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

from .config import create_test_configurations, get_configuration_descriptions
from .test_runner import TestRunner
from .report_generator import ReportGenerator
from .block_manager import BlockManager
from .models import TestConfiguration, TestSummary


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser"""

    parser = argparse.ArgumentParser(
        prog="linalg_profiler",
        description="GNU Radio Linear Algebra Performance Profiler - Test and benchmark linear algebra blocks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --test quick --format json
  %(prog)s --test comprehensive --output-dir ./results --verbose
  %(prog)s --test custom --blocks matrix_multiply,vector_norm --sizes 8x8,16x16
  %(prog)s --list-blocks
  %(prog)s --list-configs

Block Categories:
  matrix_basic      - Basic matrix operations (add, multiply, transpose, etc.)
  matrix_advanced   - Advanced matrix operations (inverse, solve, decompose)
  vector_operations - Vector operations (norm, cross product, dot product)
  decompositions    - Matrix decompositions (SVD, LU, QR, Cholesky, etc.)
  solvers          - Linear system solvers (direct and iterative)
  matrix_generators - Matrix generator blocks (eye, ones, zeros)

Output Formats:
  console  - Human-readable console output (default)
  json     - Structured JSON format for programmatic use
  csv      - Comma-separated values for spreadsheet analysis
  html     - Rich HTML report with charts and visualizations
        """,
    )

    # Main operation modes
    group_mode = parser.add_mutually_exclusive_group(required=True)
    group_mode.add_argument(
        "--test",
        "-t",
        choices=list(create_test_configurations().keys()),
        help="Run performance tests using predefined configuration",
    )
    group_mode.add_argument(
        "--list-blocks",
        action="store_true",
        help="List all available blocks by category",
    )
    group_mode.add_argument(
        "--list-configs",
        action="store_true",
        help="List all available test configurations",
    )

    # Test customization
    parser.add_argument(
        "--blocks",
        help="Comma-separated list of specific blocks to test (overrides config)",
    )
    parser.add_argument(
        "--sizes",
        help="Comma-separated list of matrix sizes in format NxM or vector sizes N (overrides config)",
    )
    parser.add_argument(
        "--precisions",
        choices=["f", "c", "f,c"],
        default="f",
        help="Precision types to test: f=float, c=complex (default: f)",
    )
    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        help="Number of test iterations (overrides config)",
    )

    # Output options
    parser.add_argument(
        "--format",
        "-f",
        choices=["console", "json", "csv", "html"],
        default="console",
        help="Output format (default: console)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        help="Output directory for generated files (default: current directory)",
    )
    parser.add_argument(
        "--output-file",
        help="Specific output filename (auto-generated if not provided)",
    )

    # Behavior options
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output with detailed progress information",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress all output except results and errors",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        help="Timeout in seconds for individual tests (overrides config)",
    )
    parser.add_argument(
        "--no-memory-profiling",
        action="store_true",
        help="Disable memory usage profiling for faster execution",
    )

    return parser


def parse_sizes_argument(sizes_str: str) -> tuple[list, list]:
    """Parse sizes argument into matrix and vector sizes"""
    matrix_sizes = []
    vector_sizes = []

    for size_str in sizes_str.split(","):
        size_str = size_str.strip()
        if "x" in size_str:
            # Matrix size (e.g., "8x8", "16x32")
            try:
                rows, cols = map(int, size_str.split("x"))
                matrix_sizes.append((rows, cols))
            except ValueError:
                print(f"Warning: Invalid matrix size format '{size_str}', skipping")
        else:
            # Vector size (e.g., "16", "64")
            try:
                size = int(size_str)
                vector_sizes.append(size)
            except ValueError:
                print(f"Warning: Invalid vector size format '{size_str}', skipping")

    return matrix_sizes, vector_sizes


def apply_argument_overrides(
    config: TestConfiguration, args: argparse.Namespace
) -> TestConfiguration:
    """Apply command-line argument overrides to test configuration"""

    # Override block selection
    if args.blocks:
        config.block_type = args.blocks

    # Override sizes
    if args.sizes:
        matrix_sizes, vector_sizes = parse_sizes_argument(args.sizes)
        if matrix_sizes:
            config.matrix_sizes = matrix_sizes
        if vector_sizes:
            config.vector_sizes = vector_sizes

    # Override precision types
    if args.precisions:
        config.precision_types = args.precisions.split(",")

    # Override iterations
    if args.iterations:
        config.num_iterations = args.iterations

    # Override timeout
    if args.timeout:
        config.timeout_seconds = args.timeout

    # Override memory profiling
    if args.no_memory_profiling:
        config.enable_memory_profiling = False

    return config


def list_available_blocks():
    """List all available blocks organized by category"""
    block_manager = BlockManager()

    if not block_manager.bindings_available:
        print("Error: GNU Radio linalg bindings are not available")
        print(
            "Please ensure the module is built and PYTHONPATH is configured correctly"
        )
        return False

    print("Available GNU Radio Linear Algebra Blocks:")
    print("=" * 50)

    available_blocks = block_manager.get_available_blocks()

    for category, blocks in available_blocks.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        for block in blocks:
            print(f"  • {block}")

    print(f"\nTotal: {len(block_manager.get_all_block_names())} unique blocks")
    return True


def list_available_configurations():
    """List all available test configurations"""
    configs = create_test_configurations()
    descriptions = get_configuration_descriptions()

    print("Available Test Configurations:")
    print("=" * 40)

    for config_name in configs.keys():
        config = configs[config_name]
        description = descriptions.get(config_name, "No description available")

        print(f"\n{config_name.upper()}:")
        print(f"  Description: {description}")
        print(f"  Block Types: {config.block_type}")
        print(f"  Matrix Sizes: {config.matrix_sizes}")
        print(f"  Vector Sizes: {config.vector_sizes}")
        print(f"  Precisions: {', '.join(config.precision_types)}")
        print(f"  Iterations: {config.num_iterations}")
        print(f"  Timeout: {config.timeout_seconds}s")


def run_performance_tests(config: TestConfiguration, args: argparse.Namespace) -> bool:
    """Run the performance test suite with given configuration"""

    # Create test runner
    test_runner = TestRunner(verbose=args.verbose and not args.quiet)

    if not args.quiet:
        print(f"Starting performance tests: {config.test_name}")
        print(f"Block type: {config.block_type}")
        print(f"Precision types: {', '.join(config.precision_types)}")
        if config.matrix_sizes:
            print(f"Matrix sizes: {config.matrix_sizes}")
        if config.vector_sizes:
            print(f"Vector sizes: {config.vector_sizes}")
        print(f"Iterations: {config.num_iterations}")
        print(
            f"Memory profiling: {'enabled' if config.enable_memory_profiling else 'disabled'}"
        )
        print("-" * 60)

    # Run tests
    try:
        summary = test_runner.run_test_suite(config)
    except Exception as e:
        print(f"Error during test execution: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return False

    # Generate and output results
    report_generator = ReportGenerator(args.output_dir)

    try:
        if args.format == "console" or args.format == "html":
            # Always show console summary unless quiet
            if not args.quiet:
                report_generator.print_console_summary(summary, verbose=args.verbose)

        if args.format == "json":
            output_path = report_generator.export_json(summary, args.output_file)
            if not args.quiet:
                print(f"\nJSON report saved to: {output_path}")

        elif args.format == "csv":
            output_path = report_generator.export_csv(summary, args.output_file)
            if not args.quiet:
                print(f"\nCSV report saved to: {output_path}")

        elif args.format == "html":
            output_path = report_generator.export_html(summary, args.output_file)
            if not args.quiet:
                print(f"\nHTML report saved to: {output_path}")

    except Exception as e:
        print(f"Error generating {args.format} report: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return False

    # Return success status based on test results
    return summary.failed_tests == 0


def main() -> int:
    """Main entry point for the CLI application"""

    parser = create_argument_parser()
    args = parser.parse_args()

    # Handle list operations
    if args.list_blocks:
        return 0 if list_available_blocks() else 1

    if args.list_configs:
        list_available_configurations()
        return 0

    # Validate output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(
                f"Error: Cannot create output directory '{output_dir}': {e}",
                file=sys.stderr,
            )
            return 1

    # Handle test execution
    if args.test:
        # Load base configuration
        configs = create_test_configurations()
        config = configs[args.test]

        # Apply command-line overrides
        config = apply_argument_overrides(config, args)

        # Run tests
        success = run_performance_tests(config, args)
        return 0 if success else 1

    # Should not reach here due to mutually_exclusive_group
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
