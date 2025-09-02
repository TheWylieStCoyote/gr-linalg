#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance test runner for GNU Radio Linear Algebra Performance Profiler.

Handles the execution of performance tests on linear algebra blocks,
including timing measurements and flowgraph management.
"""

import time
import traceback
from typing import List, Tuple, Optional
from contextlib import contextmanager

from gnuradio import gr
import numpy as np

from .models import TestConfiguration, PerformanceResult, TestSummary
from .block_manager import BlockManager
from .memory_profiler import MemoryProfiler


class TestRunner:
    """Executes performance tests on GNU Radio linear algebra blocks"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.block_manager = BlockManager()
        self.memory_profiler = MemoryProfiler()
        self.results: List[PerformanceResult] = []

    def log(self, message: str, level: str = "info"):
        """Log message if verbose mode enabled"""
        if self.verbose:
            timestamp = time.strftime("[%H:%M:%S]")
            if level == "error":
                print(f"{timestamp} ERROR: {message}")
            elif level == "warning":
                print(f"{timestamp} WARNING: {message}")
            else:
                print(f"{timestamp} {message}")

    @contextmanager
    def timer(self):
        """Context manager for high-precision timing"""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            self.last_execution_time = (
                end_time - start_time
            ) * 1000  # Convert to milliseconds

    def test_matrix_operation(
        self,
        block_name: str,
        precision: str,
        matrix_size: Tuple[int, int],
        config: TestConfiguration,
    ) -> PerformanceResult:
        """Test a matrix operation block"""
        rows, cols = matrix_size
        self.log(f"Running test: {block_name}_{precision} {rows}x{cols}")

        try:
            # Get vector source and sink classes
            source_class, sink_class, dtype = self.block_manager.get_vector_source_sink(
                precision, rows * cols
            )

            # Create test data
            if self.block_manager.requires_two_inputs(block_name):
                data_a = self.block_manager.create_test_data(precision, (rows * cols,))
                data_b = self.block_manager.create_test_data(precision, (rows * cols,))
            else:
                data_a = self.block_manager.create_test_data(precision, (rows * cols,))
                data_b = None

            # Create GNU Radio flowgraph
            tb = gr.top_block()

            # Create sources
            src_a = source_class(data_a.flatten(), False, rows * cols)
            if data_b is not None:
                src_b = source_class(data_b.flatten(), False, rows * cols)

            # Create the block under test
            block = self.block_manager.create_block(block_name, precision, rows, cols)

            # Create sink (output size depends on block type)
            if block_name in ["matrix_determinant", "matrix_trace"]:
                # Scalar output
                sink = sink_class(1)
            elif block_name in [
                "matrix_solve",
                "solve_least_squares",
                "solve_iterative",
            ]:
                # Solution vector output
                sink = sink_class(rows)
            else:
                # Matrix output (same size as input)
                sink = sink_class(rows * cols)

            # Connect blocks
            tb.connect(src_a, (block, 0))
            if data_b is not None:
                tb.connect(src_b, (block, 1))
            tb.connect(block, sink)

            # Run performance test with timing and memory profiling
            with self.memory_profiler.profile():
                with self.timer():
                    tb.run()

            execution_time = self.last_execution_time
            peak_memory = self.memory_profiler.stop_monitoring()
            cpu_usage = self.memory_profiler.get_cpu_usage()

            # Calculate throughput (operations per second)
            num_elements = rows * cols
            throughput_mops = (num_elements / (execution_time / 1000.0)) / 1_000_000

            self.log(
                f"{block_name}_{precision}_{rows}x{cols}: {execution_time:.2f}ms, "
                f"Peak Memory: {peak_memory:.2f}MB"
            )

            return PerformanceResult(
                test_name=config.test_name,
                block_type=block_name,
                precision_type=precision,
                data_size=(rows, cols),
                execution_time_ms=execution_time,
                throughput_mops=throughput_mops,
                memory_usage_mb=peak_memory,
                cpu_usage_percent=cpu_usage,
                success=True,
            )

        except Exception as e:
            error_msg = str(e)
            self.log(
                f"Error testing {block_name}_{precision} {rows}x{cols}: {error_msg}",
                "error",
            )

            return PerformanceResult(
                test_name=config.test_name,
                block_type=block_name,
                precision_type=precision,
                data_size=(rows, cols),
                execution_time_ms=0.0,
                throughput_mops=0.0,
                memory_usage_mb=0.0,
                cpu_usage_percent=0.0,
                success=False,
                error_message=error_msg,
            )

    def test_vector_operation(
        self,
        block_name: str,
        precision: str,
        vector_size: int,
        config: TestConfiguration,
    ) -> PerformanceResult:
        """Test a vector operation block"""
        self.log(f"Running test: {block_name}_{precision} size={vector_size}")

        try:
            # Get vector source and sink classes
            source_class, sink_class, dtype = self.block_manager.get_vector_source_sink(
                precision, vector_size
            )

            # Create test data
            if self.block_manager.requires_two_inputs(block_name):
                if block_name == "vector_cross_product":
                    # Cross product requires 3D vectors
                    data_a = self.block_manager.create_test_data(precision, (3,))
                    data_b = self.block_manager.create_test_data(precision, (3,))
                    vector_size = 3
                else:
                    data_a = self.block_manager.create_test_data(
                        precision, (vector_size,)
                    )
                    data_b = self.block_manager.create_test_data(
                        precision, (vector_size,)
                    )
            else:
                data_a = self.block_manager.create_test_data(precision, (vector_size,))
                data_b = None

            # Create GNU Radio flowgraph
            tb = gr.top_block()

            # Create sources
            src_a = source_class(data_a.flatten(), False, len(data_a))
            if data_b is not None:
                src_b = source_class(data_b.flatten(), False, len(data_b))

            # Create the block under test
            if block_name == "vector_cross_product":
                # Cross product uses different constructor
                rows, cols = 3, 1
            else:
                rows, cols = vector_size, 1

            block = self.block_manager.create_block(block_name, precision, rows, cols)

            # Create sink (output size depends on block type)
            if block_name in ["vector_norm", "vector_angle", "dot_product"]:
                # Scalar output
                sink = sink_class(1)
            elif block_name == "vector_outer_product":
                # Matrix output (vector_size x vector_size)
                sink = sink_class(vector_size * vector_size)
            else:
                # Vector output (same size as input)
                sink = sink_class(vector_size)

            # Connect blocks
            tb.connect(src_a, (block, 0))
            if data_b is not None:
                tb.connect(src_b, (block, 1))
            tb.connect(block, sink)

            # Run performance test with timing and memory profiling
            with self.memory_profiler.profile():
                with self.timer():
                    tb.run()

            execution_time = self.last_execution_time
            peak_memory = self.memory_profiler.stop_monitoring()
            cpu_usage = self.memory_profiler.get_cpu_usage()

            # Calculate throughput
            throughput_mops = (vector_size / (execution_time / 1000.0)) / 1_000_000

            self.log(
                f"{block_name}_{precision}_{vector_size}: {execution_time:.2f}ms, "
                f"Peak Memory: {peak_memory:.2f}MB"
            )

            return PerformanceResult(
                test_name=config.test_name,
                block_type=block_name,
                precision_type=precision,
                data_size=(vector_size,),
                execution_time_ms=execution_time,
                throughput_mops=throughput_mops,
                memory_usage_mb=peak_memory,
                cpu_usage_percent=cpu_usage,
                success=True,
            )

        except Exception as e:
            error_msg = str(e)
            self.log(
                f"Error testing {block_name}_{precision} size={vector_size}: {error_msg}",
                "error",
            )

            return PerformanceResult(
                test_name=config.test_name,
                block_type=block_name,
                precision_type=precision,
                data_size=(vector_size,),
                execution_time_ms=0.0,
                throughput_mops=0.0,
                memory_usage_mb=0.0,
                cpu_usage_percent=0.0,
                success=False,
                error_message=error_msg,
            )

    def run_test_suite(self, config: TestConfiguration) -> TestSummary:
        """Run a complete test suite"""
        self.log(f"Starting test suite: {config.test_name}")
        self.results = []
        start_time = time.time()

        # Determine which blocks to test
        if config.block_type == "all":
            blocks_to_test = self.block_manager.get_all_block_names()
        elif config.block_type in self.block_manager.get_available_blocks():
            blocks_to_test = self.block_manager.get_blocks_by_category(
                config.block_type
            )
        else:
            # Single block or comma-separated list
            blocks_to_test = [b.strip() for b in config.block_type.split(",")]

        test_count = 0
        total_tests = (
            len(blocks_to_test)
            * len(config.precision_types)
            * (len(config.matrix_sizes) + len(config.vector_sizes))
        )

        for block_name in blocks_to_test:
            for precision in config.precision_types:
                # Test matrix operations
                if (
                    not self.block_manager.is_vector_operation(block_name)
                    and config.matrix_sizes
                ):
                    for matrix_size in config.matrix_sizes:
                        test_count += 1
                        self.log(
                            f"Running test {test_count}/{total_tests}: {block_name}_{precision} {matrix_size[0]}x{matrix_size[1]}"
                        )
                        result = self.test_matrix_operation(
                            block_name, precision, matrix_size, config
                        )
                        self.results.append(result)

                # Test vector operations
                if (
                    self.block_manager.is_vector_operation(block_name)
                    and config.vector_sizes
                ):
                    for vector_size in config.vector_sizes:
                        test_count += 1
                        self.log(
                            f"Running test {test_count}/{total_tests}: {block_name}_{precision} size={vector_size}"
                        )
                        result = self.test_vector_operation(
                            block_name, precision, vector_size, config
                        )
                        self.results.append(result)

        end_time = time.time()
        total_time = end_time - start_time

        # Calculate summary statistics
        successful_tests = sum(1 for r in self.results if r.success)
        failed_tests = len(self.results) - successful_tests

        if successful_tests > 0:
            avg_throughput = (
                sum(r.throughput_mops for r in self.results if r.success)
                / successful_tests
            )
            peak_memory = max(
                (r.memory_usage_mb for r in self.results if r.success), default=0.0
            )
        else:
            avg_throughput = 0.0
            peak_memory = 0.0

        self.log(
            f"Test suite completed: {successful_tests}/{len(self.results)} tests passed in {total_time:.2f}s"
        )

        return TestSummary(
            total_tests=len(self.results),
            successful_tests=successful_tests,
            failed_tests=failed_tests,
            total_time_seconds=total_time,
            average_throughput_mops=avg_throughput,
            peak_memory_mb=peak_memory,
            results=self.results,
        )
