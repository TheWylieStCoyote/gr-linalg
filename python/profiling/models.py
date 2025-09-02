#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data models and structures for GNU Radio Linear Algebra Performance Profiler.

Contains all dataclass definitions and core data structures used throughout
the profiling system.
"""

import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any, Union


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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


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

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage"""
        return (
            (self.successful_tests / self.total_tests * 100.0)
            if self.total_tests > 0
            else 0.0
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "total_tests": self.total_tests,
            "successful_tests": self.successful_tests,
            "failed_tests": self.failed_tests,
            "total_time_seconds": self.total_time_seconds,
            "average_throughput_mops": self.average_throughput_mops,
            "peak_memory_mb": self.peak_memory_mb,
            "results": [result.to_dict() for result in self.results],
        }


@dataclass
class BlockInfo:
    """Information about a GNU Radio block"""

    name: str
    category: str
    precision_types: List[str]
    input_ports: int
    output_ports: int
    is_source: bool = False
    is_sink: bool = False
    requires_special_setup: bool = False


class TestStatus:
    """Enumeration of test statuses"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
