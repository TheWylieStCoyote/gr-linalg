#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Report generation and visualization for GNU Radio Linear Algebra Performance Profiler.

Handles output formatting, statistical analysis, and visualization of performance
test results in various formats (console, JSON, CSV, HTML).
"""

import json
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from .models import TestSummary, PerformanceResult


class ReportGenerator:
    """Generates performance test reports in various formats"""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def print_console_summary(self, summary: TestSummary, verbose: bool = False):
        """Print formatted summary to console"""
        print("\n" + "=" * 80)
        print("GNU RADIO LINEAR ALGEBRA PERFORMANCE PROFILER - RESULTS")
        print("=" * 80)

        # Test summary
        print(f"\nTest Summary:")
        print(f"  Total Tests:     {summary.total_tests}")
        print(
            f"  Successful:      {summary.successful_tests} ({summary.success_rate:.1f}%)"
        )
        print(f"  Failed:          {summary.failed_tests}")
        print(f"  Total Time:      {summary.total_time_seconds:.2f} seconds")
        print(f"  Avg Throughput:  {summary.average_throughput_mops:.2f} MOPS")
        print(f"  Peak Memory:     {summary.peak_memory_mb:.2f} MB")

        if verbose and summary.results:
            self._print_detailed_results(summary.results)

        # Performance analysis
        self._print_performance_analysis(summary.results)

        print("\n" + "=" * 80)

    def _print_detailed_results(self, results: List[PerformanceResult]):
        """Print detailed test results"""
        print(f"\nDetailed Results:")
        print(
            f"{'Block':<25} {'Precision':<10} {'Size':<12} {'Time (ms)':<12} {'MOPS':<12} {'Memory (MB)':<12} {'Status':<8}"
        )
        print("-" * 100)

        for result in results:
            size_str = (
                f"{result.data_size[0]}x{result.data_size[1]}"
                if len(result.data_size) == 2
                else f"{result.data_size[0]}"
            )
            status = "✓" if result.success else "✗"

            print(
                f"{result.block_type:<25} {result.precision_type:<10} {size_str:<12} "
                f"{result.execution_time_ms:<12.3f} {result.throughput_mops:<12.2f} "
                f"{result.memory_usage_mb:<12.2f} {status:<8}"
            )

    def _print_performance_analysis(self, results: List[PerformanceResult]):
        """Print performance analysis and insights"""
        if not results:
            return

        successful_results = [r for r in results if r.success]
        if not successful_results:
            return

        print(f"\nPerformance Analysis:")

        # Fastest and slowest operations
        fastest = min(successful_results, key=lambda x: x.execution_time_ms)
        slowest = max(successful_results, key=lambda x: x.execution_time_ms)

        print(
            f"  Fastest Operation: {fastest.block_type}_{fastest.precision_type} "
            f"({fastest.execution_time_ms:.3f}ms)"
        )
        print(
            f"  Slowest Operation: {slowest.block_type}_{slowest.precision_type} "
            f"({slowest.execution_time_ms:.3f}ms)"
        )

        # Highest throughput
        highest_throughput = max(successful_results, key=lambda x: x.throughput_mops)
        print(
            f"  Highest Throughput: {highest_throughput.block_type}_{highest_throughput.precision_type} "
            f"({highest_throughput.throughput_mops:.2f} MOPS)"
        )

        # Memory usage stats
        avg_memory = sum(r.memory_usage_mb for r in successful_results) / len(
            successful_results
        )
        max_memory = max(r.memory_usage_mb for r in successful_results)
        print(f"  Average Memory: {avg_memory:.2f} MB, Peak: {max_memory:.2f} MB")

        # Failed operations
        failed_results = [r for r in results if not r.success]
        if failed_results:
            print(f"\nFailed Operations ({len(failed_results)}):")
            for result in failed_results[:5]:  # Show first 5 failures
                print(
                    f"  ✗ {result.block_type}_{result.precision_type}: {result.error_message}"
                )
            if len(failed_results) > 5:
                print(f"  ... and {len(failed_results) - 5} more failures")

    def export_json(self, summary: TestSummary, filename: Optional[str] = None) -> str:
        """Export results to JSON format"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"linalg_performance_{timestamp}.json"

        output_path = self.output_dir / filename

        # Add metadata
        export_data = {
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "profiler_version": "1.0.0",
                "gnu_radio_version": "unknown",
            },
            "summary": summary.to_dict(),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        return str(output_path)

    def export_csv(self, summary: TestSummary, filename: Optional[str] = None) -> str:
        """Export results to CSV format"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"linalg_performance_{timestamp}.csv"

        output_path = self.output_dir / filename

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(
                [
                    "Block Type",
                    "Precision",
                    "Data Size",
                    "Execution Time (ms)",
                    "Throughput (MOPS)",
                    "Memory Usage (MB)",
                    "CPU Usage (%)",
                    "Success",
                    "Error Message",
                    "Timestamp",
                ]
            )

            # Write data rows
            for result in summary.results:
                size_str = "x".join(map(str, result.data_size))
                writer.writerow(
                    [
                        result.block_type,
                        result.precision_type,
                        size_str,
                        f"{result.execution_time_ms:.3f}",
                        f"{result.throughput_mops:.3f}",
                        f"{result.memory_usage_mb:.3f}",
                        f"{result.cpu_usage_percent:.1f}",
                        result.success,
                        result.error_message or "",
                        datetime.fromtimestamp(result.timestamp).isoformat(),
                    ]
                )

        return str(output_path)

    def export_html(self, summary: TestSummary, filename: Optional[str] = None) -> str:
        """Export results to HTML format with embedded charts"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"linalg_performance_{timestamp}.html"

        output_path = self.output_dir / filename

        # Generate charts
        charts_html = self._generate_charts_html(summary.results)

        # Create HTML content
        html_content = self._create_html_template(summary, charts_html)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return str(output_path)

    def _generate_charts_html(self, results: List[PerformanceResult]) -> str:
        """Generate HTML with embedded charts using matplotlib and base64 encoding"""
        if not results:
            return "<p>No results to visualize</p>"

        try:
            import matplotlib

            matplotlib.use("Agg")  # Use non-interactive backend
            import matplotlib.pyplot as plt
            import base64
            from io import BytesIO

            charts_html = []

            # Chart 1: Execution Time by Block Type
            successful_results = [r for r in results if r.success]
            if successful_results:
                charts_html.append(
                    self._create_execution_time_chart(successful_results)
                )
                charts_html.append(self._create_throughput_chart(successful_results))
                charts_html.append(self._create_memory_usage_chart(successful_results))

            return "\n".join(charts_html)

        except ImportError:
            return "<p>Matplotlib not available - charts disabled</p>"

    def _create_execution_time_chart(self, results: List[PerformanceResult]) -> str:
        """Create execution time comparison chart"""
        import matplotlib.pyplot as plt
        import base64
        from io import BytesIO

        # Group results by block type
        block_times = {}
        for result in results:
            if result.block_type not in block_times:
                block_times[result.block_type] = []
            block_times[result.block_type].append(result.execution_time_ms)

        # Calculate averages
        block_names = list(block_times.keys())[:15]  # Limit to top 15
        avg_times = [
            sum(block_times[name]) / len(block_times[name]) for name in block_names
        ]

        # Create chart
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(block_names)), avg_times)
        plt.xlabel("Block Type")
        plt.ylabel("Average Execution Time (ms)")
        plt.title("Average Execution Time by Block Type")
        plt.xticks(range(len(block_names)), block_names, rotation=45, ha="right")

        # Color bars by performance
        colors = [
            "green" if t < 10 else "yellow" if t < 100 else "red" for t in avg_times
        ]
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        plt.tight_layout()

        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        buffer.seek(0)
        chart_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return f"""
        <div class="chart-container">
            <h3>Execution Time Analysis</h3>
            <img src="data:image/png;base64,{chart_data}" alt="Execution Time Chart" style="max-width: 100%; height: auto;">
        </div>
        """

    def _create_throughput_chart(self, results: List[PerformanceResult]) -> str:
        """Create throughput comparison chart"""
        import matplotlib.pyplot as plt
        import base64
        from io import BytesIO

        # Top 15 highest throughput operations
        sorted_results = sorted(results, key=lambda x: x.throughput_mops, reverse=True)[
            :15
        ]

        block_labels = [f"{r.block_type}_{r.precision_type}" for r in sorted_results]
        throughputs = [r.throughput_mops for r in sorted_results]

        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(block_labels)), throughputs)
        plt.xlabel("Block Operation")
        plt.ylabel("Throughput (MOPS)")
        plt.title("Highest Throughput Operations")
        plt.xticks(range(len(block_labels)), block_labels, rotation=45, ha="right")

        # Color bars by throughput level
        colors = [
            "darkgreen" if t > 100 else "green" if t > 10 else "orange"
            for t in throughputs
        ]
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        buffer.seek(0)
        chart_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return f"""
        <div class="chart-container">
            <h3>Throughput Analysis</h3>
            <img src="data:image/png;base64,{chart_data}" alt="Throughput Chart" style="max-width: 100%; height: auto;">
        </div>
        """

    def _create_memory_usage_chart(self, results: List[PerformanceResult]) -> str:
        """Create memory usage distribution chart"""
        import matplotlib.pyplot as plt
        import base64
        from io import BytesIO

        memory_values = [r.memory_usage_mb for r in results if r.memory_usage_mb > 0]

        if not memory_values:
            return "<p>No memory usage data available</p>"

        plt.figure(figsize=(10, 6))
        plt.hist(memory_values, bins=20, edgecolor="black", alpha=0.7)
        plt.xlabel("Memory Usage (MB)")
        plt.ylabel("Number of Tests")
        plt.title("Memory Usage Distribution")
        plt.grid(True, alpha=0.3)

        # Add statistics
        avg_memory = sum(memory_values) / len(memory_values)
        max_memory = max(memory_values)
        plt.axvline(
            avg_memory,
            color="red",
            linestyle="--",
            label=f"Average: {avg_memory:.1f} MB",
        )
        plt.axvline(
            max_memory,
            color="orange",
            linestyle="--",
            label=f"Peak: {max_memory:.1f} MB",
        )
        plt.legend()

        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        buffer.seek(0)
        chart_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return f"""
        <div class="chart-container">
            <h3>Memory Usage Distribution</h3>
            <img src="data:image/png;base64,{chart_data}" alt="Memory Usage Chart" style="max-width: 100%; height: auto;">
        </div>
        """

    def _create_html_template(self, summary: TestSummary, charts_html: str) -> str:
        """Create complete HTML report template"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GNU Radio Linear Algebra Performance Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 3px solid #007acc;
        }}
        .header h1 {{
            color: #333;
            margin-bottom: 10px;
        }}
        .header .timestamp {{
            color: #666;
            font-size: 14px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #007acc;
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            color: #333;
            font-size: 14px;
            text-transform: uppercase;
        }}
        .summary-card .value {{
            font-size: 24px;
            font-weight: bold;
            color: #007acc;
        }}
        .summary-card .unit {{
            font-size: 12px;
            color: #666;
        }}
        .success {{ border-left-color: #28a745; }}
        .success .value {{ color: #28a745; }}
        .warning {{ border-left-color: #ffc107; }}
        .warning .value {{ color: #ffc107; }}
        .error {{ border-left-color: #dc3545; }}
        .error .value {{ color: #dc3545; }}
        .chart-container {{
            margin: 30px 0;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
        }}
        .chart-container h3 {{
            margin: 0 0 15px 0;
            color: #333;
        }}
        .results-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 30px;
            font-size: 12px;
        }}
        .results-table th {{
            background-color: #007acc;
            color: white;
            padding: 12px 8px;
            text-align: left;
        }}
        .results-table td {{
            padding: 8px;
            border-bottom: 1px solid #ddd;
        }}
        .results-table tr:hover {{
            background-color: #f5f5f5;
        }}
        .status-success {{ color: #28a745; font-weight: bold; }}
        .status-error {{ color: #dc3545; font-weight: bold; }}
        .footer {{
            margin-top: 40px;
            text-align: center;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>GNU Radio Linear Algebra Performance Report</h1>
            <div class="timestamp">Generated on {timestamp}</div>
        </div>

        <div class="summary-grid">
            <div class="summary-card">
                <h3>Total Tests</h3>
                <div class="value">{summary.total_tests}</div>
            </div>
            <div class="summary-card success">
                <h3>Successful Tests</h3>
                <div class="value">{summary.successful_tests}</div>
                <div class="unit">({summary.success_rate:.1f}%)</div>
            </div>
            <div class="summary-card {'error' if summary.failed_tests > 0 else ''}">
                <h3>Failed Tests</h3>
                <div class="value">{summary.failed_tests}</div>
            </div>
            <div class="summary-card">
                <h3>Total Time</h3>
                <div class="value">{summary.total_time_seconds:.1f}</div>
                <div class="unit">seconds</div>
            </div>
            <div class="summary-card">
                <h3>Average Throughput</h3>
                <div class="value">{summary.average_throughput_mops:.2f}</div>
                <div class="unit">MOPS</div>
            </div>
            <div class="summary-card">
                <h3>Peak Memory</h3>
                <div class="value">{summary.peak_memory_mb:.1f}</div>
                <div class="unit">MB</div>
            </div>
        </div>

        {charts_html}

        <h2>Detailed Results</h2>
        <table class="results-table">
            <thead>
                <tr>
                    <th>Block Type</th>
                    <th>Precision</th>
                    <th>Data Size</th>
                    <th>Time (ms)</th>
                    <th>MOPS</th>
                    <th>Memory (MB)</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {"".join(self._create_result_row(result) for result in summary.results)}
            </tbody>
        </table>

        <div class="footer">
            GNU Radio Linear Algebra Performance Profiler v1.0.0
        </div>
    </div>
</body>
</html>"""

    def _create_result_row(self, result: PerformanceResult) -> str:
        """Create HTML table row for a single result"""
        size_str = "x".join(map(str, result.data_size))
        status_class = "status-success" if result.success else "status-error"
        status_symbol = "✓" if result.success else "✗"

        return f"""
                <tr>
                    <td>{result.block_type}</td>
                    <td>{result.precision_type}</td>
                    <td>{size_str}</td>
                    <td>{result.execution_time_ms:.3f}</td>
                    <td>{result.throughput_mops:.2f}</td>
                    <td>{result.memory_usage_mb:.2f}</td>
                    <td class="{status_class}">{status_symbol}</td>
                </tr>"""

    def generate_statistical_summary(
        self, results: List[PerformanceResult]
    ) -> Dict[str, Any]:
        """Generate comprehensive statistical analysis"""
        if not results:
            return {}

        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]

        stats = {
            "test_counts": {
                "total": len(results),
                "successful": len(successful_results),
                "failed": len(failed_results),
                "success_rate": (
                    (len(successful_results) / len(results)) * 100 if results else 0
                ),
            },
            "execution_time": self._calculate_stats(
                [r.execution_time_ms for r in successful_results]
            ),
            "throughput": self._calculate_stats(
                [r.throughput_mops for r in successful_results]
            ),
            "memory_usage": self._calculate_stats(
                [r.memory_usage_mb for r in successful_results]
            ),
            "by_block_type": self._analyze_by_block_type(successful_results),
            "by_precision": self._analyze_by_precision(successful_results),
            "failure_analysis": self._analyze_failures(failed_results),
        }

        return stats

    def _calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate basic statistical measures"""
        if not values:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0}

        import statistics

        return {
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }

    def _analyze_by_block_type(
        self, results: List[PerformanceResult]
    ) -> Dict[str, Dict[str, float]]:
        """Analyze performance by block type"""
        block_groups = {}
        for result in results:
            if result.block_type not in block_groups:
                block_groups[result.block_type] = []
            block_groups[result.block_type].append(result)

        analysis = {}
        for block_type, group_results in block_groups.items():
            execution_times = [r.execution_time_ms for r in group_results]
            throughputs = [r.throughput_mops for r in group_results]

            analysis[block_type] = {
                "count": len(group_results),
                "avg_time_ms": sum(execution_times) / len(execution_times),
                "avg_throughput_mops": sum(throughputs) / len(throughputs),
                "min_time_ms": min(execution_times),
                "max_time_ms": max(execution_times),
            }

        return analysis

    def _analyze_by_precision(
        self, results: List[PerformanceResult]
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze performance by precision type"""
        precision_groups = {}
        for result in results:
            if result.precision_type not in precision_groups:
                precision_groups[result.precision_type] = []
            precision_groups[result.precision_type].append(result)

        analysis = {}
        for precision, group_results in precision_groups.items():
            execution_times = [r.execution_time_ms for r in group_results]
            throughputs = [r.throughput_mops for r in group_results]

            analysis[precision] = {
                "count": len(group_results),
                "avg_time_ms": sum(execution_times) / len(execution_times),
                "avg_throughput_mops": sum(throughputs) / len(throughputs),
            }

        return analysis

    def _analyze_failures(
        self, failed_results: List[PerformanceResult]
    ) -> Dict[str, Any]:
        """Analyze failure patterns"""
        if not failed_results:
            return {"total_failures": 0}

        # Group failures by error message
        error_groups = {}
        for result in failed_results:
            error = result.error_message or "Unknown error"
            if error not in error_groups:
                error_groups[error] = []
            error_groups[error].append(result)

        # Most common failure types
        common_failures = sorted(
            error_groups.items(), key=lambda x: len(x[1]), reverse=True
        )

        return {
            "total_failures": len(failed_results),
            "unique_error_types": len(error_groups),
            "most_common_failures": [
                {
                    "error_message": error,
                    "count": len(results),
                    "affected_blocks": list(set(r.block_type for r in results)),
                }
                for error, results in common_failures[:5]
            ],
        }
