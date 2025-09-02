#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration definitions for GNU Radio Linear Algebra Performance Profiler.

Defines standard test configurations for different testing scenarios.
"""

from typing import Dict
from .models import TestConfiguration


def create_test_configurations() -> Dict[str, TestConfiguration]:
    """Create predefined test configurations"""

    configurations = {}

    # Quick test configuration - basic blocks, small sizes, few iterations
    configurations["quick"] = TestConfiguration(
        test_name="Quick Performance Test",
        block_type="matrix_basic",
        matrix_sizes=[(2, 2), (4, 4), (8, 8)],
        vector_sizes=[4, 8, 16],
        precision_types=["f"],
        num_iterations=1,
        warmup_iterations=0,
        timeout_seconds=10.0,
        enable_memory_profiling=True,
        enable_multithreading=False,
    )

    # Comprehensive test configuration - all blocks, multiple sizes
    configurations["comprehensive"] = TestConfiguration(
        test_name="Comprehensive Performance Test",
        block_type="all",
        matrix_sizes=[(2, 2), (4, 4), (8, 8), (16, 16), (32, 32)],
        vector_sizes=[4, 8, 16, 32, 64],
        precision_types=["f", "c"],
        num_iterations=10,
        warmup_iterations=2,
        timeout_seconds=60.0,
        enable_memory_profiling=True,
        enable_multithreading=True,
    )

    # Stress test configuration - large matrices, advanced blocks
    configurations["stress"] = TestConfiguration(
        test_name="Stress Test",
        block_type="all",
        matrix_sizes=[(32, 32), (64, 64), (128, 128), (256, 256)],
        vector_sizes=[64, 128, 256, 512, 1024],
        precision_types=["f", "c"],
        num_iterations=20,
        warmup_iterations=5,
        timeout_seconds=300.0,
        enable_memory_profiling=True,
        enable_multithreading=True,
    )

    # Matrix generators test configuration
    configurations["generators"] = TestConfiguration(
        test_name="Matrix Generators Test",
        block_type="matrix_generators",
        matrix_sizes=[(4, 4), (8, 8), (16, 16), (32, 32)],
        vector_sizes=[],
        precision_types=["f"],
        num_iterations=5,
        warmup_iterations=1,
        timeout_seconds=30.0,
        enable_memory_profiling=True,
        enable_multithreading=False,
    )

    # Vector operations test configuration
    configurations["vectors"] = TestConfiguration(
        test_name="Vector Operations Test",
        block_type="vector_operations",
        matrix_sizes=[],
        vector_sizes=[8, 16, 32, 64, 128],
        precision_types=["f", "c"],
        num_iterations=10,
        warmup_iterations=2,
        timeout_seconds=60.0,
        enable_memory_profiling=True,
        enable_multithreading=False,
    )

    # Decomposition test configuration
    configurations["decompositions"] = TestConfiguration(
        test_name="Matrix Decompositions Test",
        block_type="decompositions",
        matrix_sizes=[(4, 4), (8, 8), (16, 16), (32, 32)],
        vector_sizes=[],
        precision_types=["f", "c"],
        num_iterations=5,
        warmup_iterations=1,
        timeout_seconds=120.0,
        enable_memory_profiling=True,
        enable_multithreading=False,
    )

    # Solver test configuration
    configurations["solvers"] = TestConfiguration(
        test_name="Linear Solvers Test",
        block_type="solvers",
        matrix_sizes=[(4, 4), (8, 8), (16, 16), (32, 32)],
        vector_sizes=[],
        precision_types=["f", "c"],
        num_iterations=5,
        warmup_iterations=1,
        timeout_seconds=90.0,
        enable_memory_profiling=True,
        enable_multithreading=False,
    )

    # Custom test configuration (template)
    configurations["custom"] = TestConfiguration(
        test_name="Custom Performance Test",
        block_type="matrix_basic",
        matrix_sizes=[(8, 8), (16, 16)],
        vector_sizes=[16, 32],
        precision_types=["f"],
        num_iterations=5,
        warmup_iterations=1,
        timeout_seconds=30.0,
        enable_memory_profiling=True,
        enable_multithreading=False,
    )

    return configurations


def get_configuration_descriptions() -> Dict[str, str]:
    """Get descriptions of available test configurations"""
    return {
        "quick": "Basic blocks, small sizes, 1 iteration (~30 seconds)",
        "comprehensive": "All blocks, multiple sizes, 10 iterations (~5 minutes)",
        "stress": "Large matrices, advanced blocks, 20 iterations (~10+ minutes)",
        "generators": "Matrix generator blocks only (~2 minutes)",
        "vectors": "Vector operations with various sizes (~3 minutes)",
        "decompositions": "Matrix decomposition blocks (~5 minutes)",
        "solvers": "Linear system solvers (~4 minutes)",
        "custom": "User-defined configuration",
    }
