#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory profiling utilities for GNU Radio Linear Algebra Performance Profiler.

Provides memory usage monitoring and tracking capabilities during block
execution to identify memory leaks and optimize performance.
"""

import threading
import time
import psutil
from typing import Optional, List
from contextlib import contextmanager


class MemoryProfiler:
    """Profile memory usage during block execution"""

    def __init__(self):
        self.process = psutil.Process()
        self.monitoring = False
        self.monitor_thread = None
        self.peak_memory = 0.0
        self.memory_samples = []
        self.sample_interval = 0.01  # 10ms sampling interval

    def start_monitoring(self):
        """Start memory monitoring in background thread"""
        if self.monitoring:
            return

        self.monitoring = True
        self.peak_memory = 0.0
        self.memory_samples = []

        def monitor_memory():
            while self.monitoring:
                try:
                    memory_mb = self.process.memory_info().rss / (1024 * 1024)
                    self.memory_samples.append(memory_mb)
                    self.peak_memory = max(self.peak_memory, memory_mb)
                    time.sleep(self.sample_interval)
                except psutil.Error:
                    break

        self.monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self) -> float:
        """Stop memory monitoring and return peak memory usage"""
        if not self.monitoring:
            return 0.0

        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)

        return self.peak_memory

    def get_current_memory_mb(self) -> float:
        """Get current memory usage in MB"""
        try:
            return self.process.memory_info().rss / (1024 * 1024)
        except psutil.Error:
            return 0.0

    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            return self.process.cpu_percent()
        except psutil.Error:
            return 0.0

    def reset(self):
        """Reset profiling state"""
        self.peak_memory = 0.0
        self.memory_samples = []

    @contextmanager
    def profile(self):
        """Context manager for automatic memory profiling"""
        self.start_monitoring()
        try:
            yield self
        finally:
            self.stop_monitoring()

    def get_memory_statistics(self) -> dict:
        """Get comprehensive memory statistics"""
        if not self.memory_samples:
            return {
                "peak_mb": 0.0,
                "average_mb": 0.0,
                "min_mb": 0.0,
                "max_mb": 0.0,
                "sample_count": 0,
            }

        return {
            "peak_mb": self.peak_memory,
            "average_mb": sum(self.memory_samples) / len(self.memory_samples),
            "min_mb": min(self.memory_samples),
            "max_mb": max(self.memory_samples),
            "sample_count": len(self.memory_samples),
        }


class SystemResourceMonitor:
    """Monitor system-wide resource usage"""

    def __init__(self):
        self.initial_cpu_count = psutil.cpu_count()
        self.initial_memory_total = psutil.virtual_memory().total / (
            1024 * 1024 * 1024
        )  # GB

    def get_system_info(self) -> dict:
        """Get basic system information"""
        return {
            "cpu_count": self.initial_cpu_count,
            "total_memory_gb": self.initial_memory_total,
            "platform": psutil.platform.platform(),
            "python_version": psutil.PYTHON,
        }

    def get_system_load(self) -> dict:
        """Get current system load information"""
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        return {
            "cpu_usage_percent": cpu_usage,
            "memory_usage_percent": memory.percent,
            "memory_available_gb": memory.available / (1024 * 1024 * 1024),
            "memory_used_gb": memory.used / (1024 * 1024 * 1024),
        }
