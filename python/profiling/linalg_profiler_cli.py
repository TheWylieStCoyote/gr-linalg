#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNU Radio Linear Algebra Performance Profiler CLI

A comprehensive command-line tool for performance testing and benchmarking
linear algebra blocks in GNU Radio. Provides detailed timing analysis,
throughput measurements, and comparative benchmarking across different
block types, sizes, and precision types.

Features:
- Automated performance testing for all linear algebra block categories
- Multiple test modes: quick, comprehensive, stress, and custom
- Memory usage monitoring and profiling
- Multi-threaded performance testing
- Detailed reporting with statistics and visualizations
- Export capabilities (JSON, CSV, HTML reports)
- Integration with GNU Radio's performance profiler infrastructure
"""

import argparse
import sys
import os
import time
import traceback
import json
import csv
import statistics
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
from gnuradio import gr, blocks
import psutil
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    sns = None

# Set up path to find gnuradio.linalg
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "build" / "test_modules"))


@dataclass
class TestConfiguration:
    """Configuration for performance tests"""

    test_name: str
    block_type: str
    matrix_sizes: List[Tuple[int, int]]
    vector_sizes: List[int]
    precision_types: List[str]
    num_iterations: int
    warmup_iterations: int
    timeout_seconds: float
    enable_memory_profiling: bool
    enable_multithreading: bool


@dataclass
class PerformanceResult:
    """Results from a single performance test"""

    test_name: str
    block_type: str
    precision_type: str
    data_size: Tuple[int, ...]
    execution_time_ms: float
    throughput_mops: float  # Million operations per second
    memory_usage_mb: float
    cpu_usage_percent: float
    success: bool
    error_message: Optional[str] = None
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class TestSummary:
    """Summary statistics for a test suite"""

    total_tests: int
    successful_tests: int
    failed_tests: int
    total_time_seconds: float
    average_throughput_mops: float
    peak_memory_mb: float
    results: List[PerformanceResult]


class MemoryProfiler:
    """Memory usage monitoring for performance tests"""

    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.get_memory_usage()
        self.peak_memory = self.initial_memory
        self.monitoring = False
        self.monitor_thread = None

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024

    def start_monitoring(self):
        """Start continuous memory monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()

    def stop_monitoring(self) -> float:
        """Stop monitoring and return peak memory usage"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        return self.peak_memory

    def _monitor_loop(self):
        """Monitoring loop running in separate thread"""
        while self.monitoring:
            current_memory = self.get_memory_usage()
            self.peak_memory = max(self.peak_memory, current_memory)
            time.sleep(0.01)  # 10ms sampling rate


class LinearAlgebraProfiler:
    """Main profiler class for linear algebra blocks"""

    def __init__(self):
        self.results: List[PerformanceResult] = []
        self.memory_profiler = MemoryProfiler()
        self.verbose = False

        # Block categories and their available blocks
        self.block_categories = {
            "matrix_basic": [
                "matrix_add",
                "matrix_subtract",
                "matrix_multiply",
                "matrix_transpose",
                "matrix_determinant",
                "matrix_trace",
            ],
            "matrix_advanced": [
                "matrix_inverse",
                "matrix_norm",
                "matrix_condition_number",
                "matrix_pseudo_inverse",
                "matrix_reshape",
                "matrix_solve",
                "matrix_exp",
                "matrix_hermitian",
                "matrix_kronecker_product",
            ],
            "matrix_elementwise": [
                "matrix_elementwise_multiply",
                "matrix_elementwise_divide",
                "matrix_diag",
                "matrix_power",
            ],
            "matrix_generators": ["eye", "ones", "zeros", "matrix_source_const"],
            "vector_operations": [
                "vector_norm",
                "vector_normalize",
                "vector_cross_product",
                "vector_angle",
                "vector_outer_product",
                "dot_product",
                "vector_correlate",
            ],
            "decompositions": [
                "decomp_svd",
                "decomp_eigen",
                "decomp_lu",
                "decomp_qr",
                "decomp_cholesky",
                "decomp_schur",
                "decomp_hessenberg",
            ],
            "solvers": ["matrix_solve", "solve_least_squares", "solve_iterative"],
        }

        self.precision_types = {
            "f": ("float", np.float32, blocks.vector_source_f, blocks.vector_sink_f),
            "c": (
                "complex_float",
                np.complex64,
                blocks.vector_source_c,
                blocks.vector_sink_c,
            ),
        }

    def set_verbose(self, verbose: bool):
        """Enable/disable verbose output"""
        self.verbose = verbose

    def log(self, message: str, level: str = "info"):
        """Log message with optional level"""
        if self.verbose or level == "error":
            timestamp = time.strftime("%H:%M:%S")
            prefix = (
                f"[{timestamp}] {level.upper()}: "
                if level != "info"
                else f"[{timestamp}] "
            )
            print(f"{prefix}{message}")

    @contextmanager
    def performance_timer(self, operation_name: str):
        """Context manager for timing operations"""
        start_time = time.perf_counter()
        self.memory_profiler.start_monitoring()

        try:
            yield
        finally:
            end_time = time.perf_counter()
            peak_memory = self.memory_profiler.stop_monitoring()
            execution_time = (end_time - start_time) * 1000  # Convert to milliseconds

            if self.verbose:
                self.log(
                    f"{operation_name}: {execution_time:.2f}ms, Peak Memory: {peak_memory:.2f}MB"
                )

    def create_test_data(
        self, shape: Tuple[int, ...], precision: str, complex_data: bool = False
    ) -> np.ndarray:
        """Generate test data for given shape and precision"""
        np_type = self.precision_types[precision][1]

        if complex_data or precision in ["c", "cd"]:
            # Generate complex data
            real_part = np.random.randn(*shape).astype(
                np.float32 if precision == "c" else np.float64
            )
            imag_part = np.random.randn(*shape).astype(
                np.float32 if precision == "c" else np.float64
            )
            return (real_part + 1j * imag_part).astype(np_type)
        else:
            return np.random.randn(*shape).astype(np_type)

    def get_block_function(self, block_name: str, precision: str):
        """Get the block factory function for given block and precision"""
        try:
            # Import the C++ bindings directly since high-level functions aren't available
            import sys
            import os

            # Add path to linalg_python C++ bindings
            build_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "build",
                "test_modules",
                "gnuradio",
                "linalg",
            )
            if build_path not in sys.path:
                sys.path.insert(0, build_path)

            import linalg_python

            # Try sync version first (most common)
            sync_name = f"{block_name}_sync_{precision}"
            if hasattr(linalg_python, sync_name):
                return getattr(linalg_python, sync_name)

            # Try PDU version
            pdu_name = f"{block_name}_pdu_{precision}"
            if hasattr(linalg_python, pdu_name):
                return getattr(linalg_python, pdu_name)

            # Try without sync/pdu for special cases
            simple_name = f"{block_name}_{precision}"
            if hasattr(linalg_python, simple_name):
                return getattr(linalg_python, simple_name)

            return None

        except ImportError as e:
            self.log(f"Failed to import linalg_python module: {e}", "error")
            return None

    def test_matrix_operation(
        self,
        block_name: str,
        precision: str,
        matrix_size: Tuple[int, int],
        config: TestConfiguration,
    ) -> PerformanceResult:
        """Test a matrix operation block"""
        rows, cols = matrix_size
        data_size = (rows, cols)

        try:
            # Get block factory function
            block_func = self.get_block_function(block_name, precision)
            if block_func is None:
                return PerformanceResult(
                    test_name=config.test_name,
                    block_type=block_name,
                    precision_type=precision,
                    data_size=data_size,
                    execution_time_ms=0.0,
                    throughput_mops=0.0,
                    memory_usage_mb=0.0,
                    cpu_usage_percent=0.0,
                    success=False,
                    error_message=f"Block function not found: {block_name}_{precision}",
                )

            # Create test flowgraph
            tb = gr.top_block(f"test_{block_name}_{precision}")

            # Generate test data
            test_data_a = self.create_test_data((rows * cols,), precision)
            # Create second input for operations that require two matrices
            needs_two_inputs = block_name in [
                "matrix_add",
                "matrix_subtract",
                "matrix_multiply",
            ]
            test_data_b = (
                self.create_test_data((rows * cols,), precision)
                if needs_two_inputs
                else None
            )

            # Get appropriate source/sink types
            _, _, source_type, sink_type = self.precision_types[precision]

            # Create sources
            src_a = source_type(test_data_a.tolist(), False, rows * cols)
            sources = [src_a]

            if test_data_b is not None:
                src_b = source_type(test_data_b.tolist(), False, rows * cols)
                sources.append(src_b)

            # Create block using appropriate method and parameters
            try:
                if block_name in ["matrix_add", "matrix_subtract"]:
                    # These use .make() with shape and number of inputs
                    block = block_func.make([rows, cols], len(sources))
                elif block_name in [
                    "matrix_transpose",
                    "matrix_determinant",
                    "matrix_trace",
                    "matrix_rank",
                ]:
                    # These use constructor directly with shape
                    block = block_func([rows, cols])
                elif block_name == "matrix_multiply":
                    # Matrix multiply needs two shapes
                    block = block_func.make([rows, cols], [cols, rows])
                elif block_name in ["matrix_inverse", "matrix_norm"]:
                    # These use .make() with just shape
                    block = block_func.make([rows, cols])
                else:
                    # Generic case - try common parameter patterns
                    try:
                        # Try .make() with shape first
                        block = block_func.make([rows, cols])
                    except AttributeError:
                        # Try direct constructor with shape
                        block = block_func([rows, cols])
                    except:
                        try:
                            # Try .make() with two shapes
                            block = block_func.make([rows, cols], [rows, cols])
                        except:
                            # Try no parameters
                            block = block_func()
            except Exception as e:
                return PerformanceResult(
                    test_name=config.test_name,
                    block_type=block_name,
                    precision_type=precision,
                    data_size=data_size,
                    execution_time_ms=0.0,
                    throughput_mops=0.0,
                    memory_usage_mb=0.0,
                    cpu_usage_percent=0.0,
                    success=False,
                    error_message=f"Failed to create block: {e}",
                )

            # Create sink
            output_size = rows * cols  # Default assumption
            if block_name == "matrix_determinant":
                output_size = 1
            elif block_name == "matrix_trace":
                output_size = 1

            sink = sink_type(output_size)

            # Connect flowgraph
            for i, src in enumerate(sources):
                tb.connect(src, (block, i))
            tb.connect(block, sink)

            # Run performance test
            cpu_percent_start = psutil.cpu_percent()

            with self.performance_timer(f"{block_name}_{precision}_{rows}x{cols}"):
                start_time = time.perf_counter()
                tb.start()
                time.sleep(0.1)  # Let it run briefly
                tb.stop()
                tb.wait()
                end_time = time.perf_counter()

            cpu_percent_end = psutil.cpu_percent()
            execution_time = (end_time - start_time) * 1000

            # Calculate throughput (operations per second)
            num_elements = rows * cols
            throughput_mops = (
                (num_elements / 1e6) / (execution_time / 1000)
                if execution_time > 0
                else 0
            )

            # Get results
            result_data = sink.data()
            success = len(result_data) > 0

            return PerformanceResult(
                test_name=config.test_name,
                block_type=block_name,
                precision_type=precision,
                data_size=data_size,
                execution_time_ms=execution_time,
                throughput_mops=throughput_mops,
                memory_usage_mb=self.memory_profiler.peak_memory,
                cpu_usage_percent=(cpu_percent_start + cpu_percent_end) / 2,
                success=success,
            )

        except Exception as e:
            self.log(
                f"Error testing {block_name}_{precision} {rows}x{cols}: {e}", "error"
            )
            return PerformanceResult(
                test_name=config.test_name,
                block_type=block_name,
                precision_type=precision,
                data_size=data_size,
                execution_time_ms=0.0,
                throughput_mops=0.0,
                memory_usage_mb=0.0,
                cpu_usage_percent=0.0,
                success=False,
                error_message=str(e),
            )

    def test_vector_operation(
        self,
        block_name: str,
        precision: str,
        vector_size: int,
        config: TestConfiguration,
    ) -> PerformanceResult:
        """Test a vector operation block"""
        data_size = (vector_size,)

        try:
            # Get block factory function
            block_func = self.get_block_function(block_name, precision)
            if block_func is None:
                return PerformanceResult(
                    test_name=config.test_name,
                    block_type=block_name,
                    precision_type=precision,
                    data_size=data_size,
                    execution_time_ms=0.0,
                    throughput_mops=0.0,
                    memory_usage_mb=0.0,
                    cpu_usage_percent=0.0,
                    success=False,
                    error_message=f"Block function not found: {block_name}_{precision}",
                )

            # Create test flowgraph
            tb = gr.top_block(f"test_{block_name}_{precision}")

            # Generate test data
            test_data_a = self.create_test_data((vector_size,), precision)
            test_data_b = (
                self.create_test_data((vector_size,), precision)
                if "cross" in block_name or "dot" in block_name
                else None
            )

            # Get appropriate source/sink types
            _, _, source_type, sink_type = self.precision_types[precision]

            # Create sources
            src_a = source_type(test_data_a.tolist(), False, vector_size)
            sources = [src_a]

            if test_data_b is not None:
                src_b = source_type(test_data_b.tolist(), False, vector_size)
                sources.append(src_b)

            # Create block with appropriate parameters
            if block_name in ["vector_norm", "vector_normalize"]:
                block = block_func(2, [vector_size])  # L2 norm
            elif block_name == "vector_cross_product":
                block = block_func(True)  # 3D vectors only
                if vector_size != 3:
                    raise ValueError("Cross product requires 3D vectors")
            elif block_name == "dot_product":
                block = block_func([vector_size], [vector_size])
            else:
                try:
                    block = block_func([vector_size])
                except:
                    block = block_func()

            # Create sink
            output_size = vector_size
            if block_name in ["vector_norm", "dot_product"]:
                output_size = 1
            elif block_name == "vector_outer_product":
                output_size = vector_size * vector_size

            sink = sink_type(output_size)

            # Connect flowgraph
            for i, src in enumerate(sources):
                tb.connect(src, (block, i))
            tb.connect(block, sink)

            # Run performance test
            cpu_percent_start = psutil.cpu_percent()

            with self.performance_timer(f"{block_name}_{precision}_{vector_size}"):
                start_time = time.perf_counter()
                tb.start()
                time.sleep(0.1)  # Let it run briefly
                tb.stop()
                tb.wait()
                end_time = time.perf_counter()

            cpu_percent_end = psutil.cpu_percent()
            execution_time = (end_time - start_time) * 1000

            # Calculate throughput
            throughput_mops = (
                (vector_size / 1e6) / (execution_time / 1000)
                if execution_time > 0
                else 0
            )

            # Get results
            result_data = sink.data()
            success = len(result_data) > 0

            return PerformanceResult(
                test_name=config.test_name,
                block_type=block_name,
                precision_type=precision,
                data_size=data_size,
                execution_time_ms=execution_time,
                throughput_mops=throughput_mops,
                memory_usage_mb=self.memory_profiler.peak_memory,
                cpu_usage_percent=(cpu_percent_start + cpu_percent_end) / 2,
                success=success,
            )

        except Exception as e:
            self.log(
                f"Error testing {block_name}_{precision} size {vector_size}: {e}",
                "error",
            )
            return PerformanceResult(
                test_name=config.test_name,
                block_type=block_name,
                precision_type=precision,
                data_size=data_size,
                execution_time_ms=0.0,
                throughput_mops=0.0,
                memory_usage_mb=0.0,
                cpu_usage_percent=0.0,
                success=False,
                error_message=str(e),
            )

    def run_test_suite(self, config: TestConfiguration) -> TestSummary:
        """Run a complete test suite based on configuration"""
        self.log(f"Starting test suite: {config.test_name}")
        start_time = time.time()
        test_results = []

        # Determine which blocks to test
        blocks_to_test = []
        if config.block_type == "all":
            for category_blocks in self.block_categories.values():
                blocks_to_test.extend(category_blocks)
        elif config.block_type in self.block_categories:
            blocks_to_test = self.block_categories[config.block_type]
        else:
            blocks_to_test = [config.block_type]

        # Calculate total tests more accurately
        total_tests = 0
        for block_name in blocks_to_test:
            is_vector_op = block_name.startswith("vector_") or block_name in [
                "dot_product"
            ]
            if is_vector_op:
                total_tests += len(config.precision_types) * len(config.vector_sizes)
            else:
                total_tests += len(config.precision_types) * len(config.matrix_sizes)
        current_test = 0

        for block_name in blocks_to_test:
            for precision in config.precision_types:
                # Determine if this is a vector or matrix operation
                is_vector_op = block_name.startswith("vector_") or block_name in [
                    "dot_product"
                ]

                if is_vector_op and config.vector_sizes:
                    # Test vector operations with vector sizes
                    for vector_size in config.vector_sizes:
                        current_test += 1
                        self.log(
                            f"Running test {current_test}/{total_tests}: {block_name}_{precision} vector size {vector_size}"
                        )

                        result = self.test_vector_operation(
                            block_name, precision, vector_size, config
                        )
                        test_results.append(result)
                        self.results.append(result)
                elif not is_vector_op and config.matrix_sizes:
                    # Test matrix operations with matrix sizes
                    for matrix_size in config.matrix_sizes:
                        current_test += 1
                        self.log(
                            f"Running test {current_test}/{total_tests}: {block_name}_{precision} {matrix_size[0]}x{matrix_size[1]}"
                        )

                        result = self.test_matrix_operation(
                            block_name, precision, matrix_size, config
                        )
                        test_results.append(result)
                        self.results.append(result)

        end_time = time.time()
        total_time = end_time - start_time

        # Calculate summary statistics
        successful_tests = sum(1 for r in test_results if r.success)
        failed_tests = len(test_results) - successful_tests
        successful_results = [
            r for r in test_results if r.success and r.throughput_mops > 0
        ]

        avg_throughput = (
            statistics.mean([r.throughput_mops for r in successful_results])
            if successful_results
            else 0.0
        )
        peak_memory = max([r.memory_usage_mb for r in test_results], default=0.0)

        summary = TestSummary(
            total_tests=len(test_results),
            successful_tests=successful_tests,
            failed_tests=failed_tests,
            total_time_seconds=total_time,
            average_throughput_mops=avg_throughput,
            peak_memory_mb=peak_memory,
            results=test_results,
        )

        self.log(
            f"Test suite completed: {successful_tests}/{len(test_results)} tests passed in {total_time:.2f}s"
        )
        return summary

    def generate_report(self, summary: TestSummary, output_format: str = "console"):
        """Generate performance report in specified format"""
        if output_format == "console":
            self._print_console_report(summary)
        elif output_format == "json":
            self._export_json_report(summary)
        elif output_format == "csv":
            self._export_csv_report(summary)
        elif output_format == "html":
            self._generate_html_report(summary)

    def _print_console_report(self, summary: TestSummary):
        """Print detailed console report"""
        print("\n" + "=" * 80)
        print("GNU RADIO LINEAR ALGEBRA PERFORMANCE REPORT")
        print("=" * 80)
        print(
            f"Test Suite: {summary.results[0].test_name if summary.results else 'Unknown'}"
        )
        print(f"Total Tests: {summary.total_tests}")
        print(f"Successful: {summary.successful_tests}")
        print(f"Failed: {summary.failed_tests}")
        print(
            f"Success Rate: {(summary.successful_tests/summary.total_tests)*100:.1f}%"
        )
        print(f"Total Time: {summary.total_time_seconds:.2f} seconds")
        print(f"Average Throughput: {summary.average_throughput_mops:.2f} MOPS")
        print(f"Peak Memory Usage: {summary.peak_memory_mb:.2f} MB")
        print()

        # Group results by block type
        by_block = defaultdict(list)
        for result in summary.results:
            if result.success:
                by_block[result.block_type].append(result)

        print("PERFORMANCE BY BLOCK TYPE:")
        print("-" * 80)
        print(
            f"{'Block Name':<25} {'Tests':<8} {'Avg Time (ms)':<15} {'Avg Throughput':<15} {'Best Size':<15}"
        )
        print("-" * 80)

        for block_name, results in sorted(by_block.items()):
            if not results:
                continue

            avg_time = statistics.mean([r.execution_time_ms for r in results])
            avg_throughput = statistics.mean([r.throughput_mops for r in results])
            best_result = max(results, key=lambda r: r.throughput_mops)
            best_size = "x".join(map(str, best_result.data_size))

            print(
                f"{block_name:<25} {len(results):<8} {avg_time:<15.2f} {avg_throughput:<15.2f} {best_size:<15}"
            )

        print()

        # Show top performers
        successful_results = [r for r in summary.results if r.success]
        if successful_results:
            print("TOP PERFORMERS (by throughput):")
            print("-" * 80)
            top_performers = sorted(
                successful_results, key=lambda r: r.throughput_mops, reverse=True
            )[:10]

            print(
                f"{'Block':<20} {'Precision':<10} {'Size':<12} {'Time (ms)':<12} {'Throughput':<12}"
            )
            print("-" * 80)
            for result in top_performers:
                size_str = "x".join(map(str, result.data_size))
                print(
                    f"{result.block_type:<20} {result.precision_type:<10} {size_str:<12} "
                    f"{result.execution_time_ms:<12.2f} {result.throughput_mops:<12.2f}"
                )

        # Show failures if any
        failed_results = [r for r in summary.results if not r.success]
        if failed_results:
            print("\nFAILED TESTS:")
            print("-" * 80)
            for result in failed_results[:10]:  # Show first 10 failures
                size_str = "x".join(map(str, result.data_size))
                error = (
                    result.error_message[:50] + "..."
                    if result.error_message and len(result.error_message) > 50
                    else result.error_message
                )
                print(
                    f"{result.block_type}_{result.precision_type} {size_str}: {error}"
                )

    def _export_json_report(self, summary: TestSummary):
        """Export results to JSON file"""
        output_file = f"linalg_performance_report_{int(time.time())}.json"

        report_data = {
            "summary": asdict(summary),
            "results": [asdict(result) for result in summary.results],
            "timestamp": time.time(),
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_gb": psutil.virtual_memory().total / (1024**3),
                "platform": sys.platform,
            },
        }

        with open(output_file, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        print(f"JSON report exported to: {output_file}")

    def _export_csv_report(self, summary: TestSummary):
        """Export results to CSV file"""
        output_file = f"linalg_performance_report_{int(time.time())}.csv"

        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(
                [
                    "test_name",
                    "block_type",
                    "precision_type",
                    "data_size",
                    "execution_time_ms",
                    "throughput_mops",
                    "memory_usage_mb",
                    "cpu_usage_percent",
                    "success",
                    "error_message",
                    "timestamp",
                ]
            )

            # Write results
            for result in summary.results:
                writer.writerow(
                    [
                        result.test_name,
                        result.block_type,
                        result.precision_type,
                        "x".join(map(str, result.data_size)),
                        result.execution_time_ms,
                        result.throughput_mops,
                        result.memory_usage_mb,
                        result.cpu_usage_percent,
                        result.success,
                        result.error_message or "",
                        result.timestamp,
                    ]
                )

        print(f"CSV report exported to: {output_file}")

    def _generate_html_report(self, summary: TestSummary):
        """Generate HTML report with visualizations"""
        output_file = f"linalg_performance_report_{int(time.time())}.html"

        # Create plots
        successful_results = [r for r in summary.results if r.success]

        if successful_results:
            # Performance by block type
            plt.figure(figsize=(12, 8))

            by_block = defaultdict(list)
            for result in successful_results:
                by_block[result.block_type].append(result.throughput_mops)

            block_names = list(by_block.keys())
            avg_throughput = [statistics.mean(by_block[name]) for name in block_names]

            plt.subplot(2, 2, 1)
            bars = plt.bar(range(len(block_names)), avg_throughput)
            plt.xlabel("Block Type")
            plt.ylabel("Average Throughput (MOPS)")
            plt.title("Average Throughput by Block Type")
            plt.xticks(range(len(block_names)), block_names, rotation=45, ha="right")
            plt.tight_layout()

            # Performance vs matrix size
            plt.subplot(2, 2, 2)
            matrix_results = [r for r in successful_results if len(r.data_size) == 2]
            if matrix_results:
                sizes = [r.data_size[0] * r.data_size[1] for r in matrix_results]
                throughputs = [r.throughput_mops for r in matrix_results]
                plt.scatter(sizes, throughputs, alpha=0.6)
                plt.xlabel("Matrix Size (elements)")
                plt.ylabel("Throughput (MOPS)")
                plt.title("Throughput vs Matrix Size")
                plt.xscale("log")

            # Memory usage
            plt.subplot(2, 2, 3)
            memory_usage = [
                r.memory_usage_mb for r in successful_results if r.memory_usage_mb > 0
            ]
            if memory_usage:
                plt.hist(memory_usage, bins=20, alpha=0.7)
                plt.xlabel("Memory Usage (MB)")
                plt.ylabel("Frequency")
                plt.title("Memory Usage Distribution")

            # Success rate by precision
            plt.subplot(2, 2, 4)
            by_precision = defaultdict(lambda: {"success": 0, "total": 0})
            for result in summary.results:
                by_precision[result.precision_type]["total"] += 1
                if result.success:
                    by_precision[result.precision_type]["success"] += 1

            precisions = list(by_precision.keys())
            success_rates = [
                (by_precision[p]["success"] / by_precision[p]["total"]) * 100
                for p in precisions
            ]

            plt.bar(precisions, success_rates)
            plt.xlabel("Precision Type")
            plt.ylabel("Success Rate (%)")
            plt.title("Success Rate by Precision Type")

            plt.tight_layout()
            plot_file = f"linalg_performance_plots_{int(time.time())}.png"
            plt.savefig(plot_file, dpi=150, bbox_inches="tight")
            plt.close()

        # Generate HTML
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>GNU Radio Linear Algebra Performance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1, h2 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .summary {{ background-color: #f9f9f9; padding: 20px; border-radius: 5px; }}
        .plot {{ text-align: center; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>GNU Radio Linear Algebra Performance Report</h1>
    
    <div class="summary">
        <h2>Test Summary</h2>
        <p><strong>Total Tests:</strong> {summary.total_tests}</p>
        <p><strong>Successful Tests:</strong> {summary.successful_tests}</p>
        <p><strong>Failed Tests:</strong> {summary.failed_tests}</p>
        <p><strong>Success Rate:</strong> {(summary.successful_tests/summary.total_tests)*100:.1f}%</p>
        <p><strong>Total Time:</strong> {summary.total_time_seconds:.2f} seconds</p>
        <p><strong>Average Throughput:</strong> {summary.average_throughput_mops:.2f} MOPS</p>
        <p><strong>Peak Memory Usage:</strong> {summary.peak_memory_mb:.2f} MB</p>
    </div>
    
    {"<div class='plot'><img src='" + plot_file + "' alt='Performance Plots'></div>" if successful_results else ""}
    
    <h2>Detailed Results</h2>
    <table>
        <tr>
            <th>Block Type</th>
            <th>Precision</th>
            <th>Data Size</th>
            <th>Execution Time (ms)</th>
            <th>Throughput (MOPS)</th>
            <th>Memory Usage (MB)</th>
            <th>Status</th>
        </tr>
"""

        for result in summary.results:
            size_str = "x".join(map(str, result.data_size))
            status = "✓" if result.success else "✗"
            status_class = "success" if result.success else "failure"

            html_content += f"""
        <tr class="{status_class}">
            <td>{result.block_type}</td>
            <td>{result.precision_type}</td>
            <td>{size_str}</td>
            <td>{result.execution_time_ms:.2f}</td>
            <td>{result.throughput_mops:.2f}</td>
            <td>{result.memory_usage_mb:.2f}</td>
            <td>{status}</td>
        </tr>"""

        html_content += """
    </table>
    
    <footer>
        <p><em>Generated by GNU Radio Linear Algebra Profiler CLI</em></p>
    </footer>
</body>
</html>"""

        with open(output_file, "w") as f:
            f.write(html_content)

        print(f"HTML report generated: {output_file}")


def create_test_configurations() -> Dict[str, TestConfiguration]:
    """Create predefined test configurations"""
    configs = {}

    # Quick test - small sizes, basic blocks
    configs["quick"] = TestConfiguration(
        test_name="Quick Performance Test",
        block_type="matrix_basic",
        matrix_sizes=[(2, 2), (4, 4), (8, 8)],
        vector_sizes=[4, 8, 16],
        precision_types=["f"],
        num_iterations=5,
        warmup_iterations=2,
        timeout_seconds=30.0,
        enable_memory_profiling=True,
        enable_multithreading=False,
    )

    # Comprehensive test - all blocks, multiple sizes
    configs["comprehensive"] = TestConfiguration(
        test_name="Comprehensive Performance Test",
        block_type="all",
        matrix_sizes=[(2, 2), (4, 4), (8, 8), (16, 16), (32, 32)],
        vector_sizes=[4, 8, 16, 32, 64],
        precision_types=["f", "c"],
        num_iterations=10,
        warmup_iterations=3,
        timeout_seconds=300.0,
        enable_memory_profiling=True,
        enable_multithreading=True,
    )

    # Stress test - large sizes, extended runtime
    configs["stress"] = TestConfiguration(
        test_name="Stress Performance Test",
        block_type="matrix_advanced",
        matrix_sizes=[(64, 64), (128, 128), (256, 256)],
        vector_sizes=[128, 256, 512],
        precision_types=["f"],
        num_iterations=20,
        warmup_iterations=5,
        timeout_seconds=600.0,
        enable_memory_profiling=True,
        enable_multithreading=True,
    )

    # Matrix generators test
    configs["generators"] = TestConfiguration(
        test_name="Matrix Generators Test",
        block_type="matrix_generators",
        matrix_sizes=[(4, 4), (8, 8), (16, 16), (32, 32)],
        vector_sizes=[],
        precision_types=["f"],
        num_iterations=10,
        warmup_iterations=2,
        timeout_seconds=120.0,
        enable_memory_profiling=True,
        enable_multithreading=False,
    )

    return configs


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="GNU Radio Linear Algebra Performance Profiler CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --test quick --format console
  %(prog)s --test comprehensive --format json --verbose
  %(prog)s --block matrix_multiply --sizes 32x32,64x64 --precision f,d
  %(prog)s --test stress --format html --output results/
        """,
    )

    # Test configuration
    parser.add_argument(
        "--test",
        "-t",
        choices=["quick", "comprehensive", "stress", "generators", "custom"],
        default="quick",
        help="Test configuration to run (default: quick)",
    )

    parser.add_argument(
        "--block",
        "-b",
        help="Specific block to test (e.g., matrix_multiply, vector_norm)",
    )

    parser.add_argument(
        "--category",
        "-c",
        choices=[
            "matrix_basic",
            "matrix_advanced",
            "matrix_elementwise",
            "matrix_generators",
            "vector_operations",
            "decompositions",
            "solvers",
        ],
        help="Block category to test",
    )

    parser.add_argument(
        "--sizes",
        help='Matrix/vector sizes to test (e.g., "4x4,8x8,16x16" or "8,16,32")',
    )

    parser.add_argument(
        "--precision",
        "-p",
        default="f,d",
        help="Precision types to test: f,d,c,cd (default: f,d)",
    )

    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=10,
        help="Number of iterations per test (default: 10)",
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
        "--output",
        "-o",
        help="Output directory for reports (default: current directory)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    # Advanced options
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Test timeout in seconds (default: 300)",
    )

    parser.add_argument(
        "--memory-profiling",
        action="store_true",
        default=True,
        help="Enable memory profiling (default: enabled)",
    )

    parser.add_argument(
        "--list-blocks", action="store_true", help="List all available blocks and exit"
    )

    args = parser.parse_args()

    # Initialize profiler
    profiler = LinearAlgebraProfiler()
    profiler.set_verbose(args.verbose)

    # Handle list blocks request
    if args.list_blocks:
        print("Available Linear Algebra Blocks:")
        print("=" * 40)
        for category, blocks in profiler.block_categories.items():
            print(f"\n{category.upper()}:")
            for block in sorted(blocks):
                print(f"  {block}")
        return 0

    try:
        # Create test configuration
        configs = create_test_configurations()

        if args.test == "custom":
            # Build custom configuration from command line arguments
            matrix_sizes = []
            vector_sizes = []

            if args.sizes:
                for size_spec in args.sizes.split(","):
                    if "x" in size_spec:
                        rows, cols = map(int, size_spec.split("x"))
                        matrix_sizes.append((rows, cols))
                    else:
                        vector_sizes.append(int(size_spec))

            if not matrix_sizes and not vector_sizes:
                matrix_sizes = [(4, 4), (8, 8)]
                vector_sizes = [8, 16]

            block_type = args.block or args.category or "matrix_basic"
            precision_types = args.precision.split(",")

            config = TestConfiguration(
                test_name="Custom Performance Test",
                block_type=block_type,
                matrix_sizes=matrix_sizes,
                vector_sizes=vector_sizes,
                precision_types=precision_types,
                num_iterations=args.iterations,
                warmup_iterations=max(1, args.iterations // 3),
                timeout_seconds=args.timeout,
                enable_memory_profiling=args.memory_profiling,
                enable_multithreading=True,
            )
        else:
            config = configs[args.test]

        # Run test suite
        profiler.log(f"Starting {config.test_name}")
        summary = profiler.run_test_suite(config)

        # Change to output directory if specified
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            os.chdir(args.output)

        # Generate report
        profiler.generate_report(summary, args.format)

        # Return appropriate exit code
        success_rate = (
            summary.successful_tests / summary.total_tests
            if summary.total_tests > 0
            else 0
        )
        return 0 if success_rate > 0.8 else 1

    except KeyboardInterrupt:
        profiler.log("Test interrupted by user", "error")
        return 130
    except Exception as e:
        profiler.log(f"Test failed with error: {e}", "error")
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
