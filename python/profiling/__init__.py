#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNU Radio Linear Algebra Performance Profiling Module

A comprehensive performance testing and profiling system for GNU Radio
linear algebra blocks, providing detailed performance metrics, memory
analysis, and visualization capabilities.
"""

from .models import (
    TestConfiguration,
    PerformanceResult,
    TestSummary,
    BlockInfo,
    TestStatus,
)
from .block_manager import BlockManager
from .memory_profiler import MemoryProfiler, SystemResourceMonitor
from .test_runner import TestRunner
from .report_generator import ReportGenerator
from .config import create_test_configurations, get_configuration_descriptions
from .cli import main as cli_main

__version__ = "1.0.0"
__author__ = "GNU Radio Linear Algebra Team"

# Public API
__all__ = [
    # Data models
    "TestConfiguration",
    "PerformanceResult",
    "TestSummary",
    "BlockInfo",
    "TestStatus",
    # Core components
    "BlockManager",
    "MemoryProfiler",
    "SystemResourceMonitor",
    "TestRunner",
    "ReportGenerator",
    # Configuration
    "create_test_configurations",
    "get_configuration_descriptions",
    # CLI entry point
    "cli_main",
]
