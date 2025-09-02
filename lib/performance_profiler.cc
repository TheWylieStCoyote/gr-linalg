/* -*- c++ -*- */
/*
 * Copyright 2024 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include <algorithm>
#include <cmath>
#include <fstream>
#include <gnuradio/linalg/performance_profiler.h>
#include <numeric>
#include <sstream>

namespace gr {
namespace linalg {

// Static member definitions
std::mutex PerformanceProfiler::metrics_mutex_;
std::unordered_map<std::string, PerformanceProfiler::OperationMetrics>
    PerformanceProfiler::metrics_;
bool PerformanceProfiler::enabled_ = true;
size_t PerformanceProfiler::max_recent_samples_ = 100;
int PerformanceProfiler::next_timer_id_ = 1;
std::unordered_map<
    int, std::pair<std::string, std::chrono::high_resolution_clock::time_point>>
    PerformanceProfiler::active_timers_;

//==============================================================================
// Timer Implementation
//==============================================================================

PerformanceProfiler::Timer::Timer(const std::string &operation_name,
                                  size_t data_size)
    : operation_name_(operation_name), data_size_(data_size), is_active_(true) {
  if (PerformanceProfiler::is_enabled()) {
    start_time_ = std::chrono::high_resolution_clock::now();
  }
}

PerformanceProfiler::Timer::~Timer() {
  if (is_active_ && PerformanceProfiler::is_enabled()) {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time_);
    double duration_ms = duration.count() / 1000.0;
    PerformanceProfiler::record_operation(operation_name_, duration_ms,
                                          data_size_);
  }
}

PerformanceProfiler::Timer::Timer(Timer &&other) noexcept
    : operation_name_(std::move(other.operation_name_)),
      data_size_(other.data_size_), start_time_(other.start_time_),
      is_active_(other.is_active_) {
  other.is_active_ = false;
}

PerformanceProfiler::Timer &
PerformanceProfiler::Timer::operator=(Timer &&other) noexcept {
  if (this != &other) {
    // Complete current timing if active
    if (is_active_ && PerformanceProfiler::is_enabled()) {
      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - start_time_);
      double duration_ms = duration.count() / 1000.0;
      PerformanceProfiler::record_operation(operation_name_, duration_ms,
                                            data_size_);
    }

    // Move from other
    operation_name_ = std::move(other.operation_name_);
    data_size_ = other.data_size_;
    start_time_ = other.start_time_;
    is_active_ = other.is_active_;
    other.is_active_ = false;
  }
  return *this;
}

//==============================================================================
// OperationMetrics Implementation
//==============================================================================

void PerformanceProfiler::OperationMetrics::update(double time_ms,
                                                   size_t data_size) {
  call_count++;
  total_time_ms += time_ms;
  min_time_ms = std::min(min_time_ms, time_ms);
  max_time_ms = std::max(max_time_ms, time_ms);
  total_data_size += data_size;

  // Store recent timing for trend analysis
  recent_times.push_back(time_ms);
  if (recent_times.size() > PerformanceProfiler::max_recent_samples_) {
    recent_times.erase(recent_times.begin());
  }

  calculate_statistics();
}

void PerformanceProfiler::OperationMetrics::calculate_statistics() {
  if (call_count > 0) {
    average_time_ms = total_time_ms / call_count;

    // Calculate throughput in MB/s if we have data size information
    if (total_data_size > 0 && total_time_ms > 0) {
      double total_mb = total_data_size / (1024.0 * 1024.0);
      double total_sec = total_time_ms / 1000.0;
      throughput_mb_per_sec = total_mb / total_sec;
    }
  }
}

//==============================================================================
// PerformanceProfiler Implementation
//==============================================================================

std::unique_ptr<PerformanceProfiler::Timer>
PerformanceProfiler::create_timer(const std::string &operation_name,
                                  size_t data_size) {
  return std::make_unique<Timer>(operation_name, data_size);
}

int PerformanceProfiler::start_timer(const std::string &operation_name) {
  if (!enabled_) {
    return -1;
  }

  std::lock_guard<std::mutex> lock(metrics_mutex_);
  int timer_id = next_timer_id_++;
  auto start_time = std::chrono::high_resolution_clock::now();
  active_timers_[timer_id] = std::make_pair(operation_name, start_time);
  return timer_id;
}

void PerformanceProfiler::end_timer(int timer_id, size_t data_size) {
  if (!enabled_ || timer_id < 0) {
    return;
  }

  auto end_time = std::chrono::high_resolution_clock::now();

  std::lock_guard<std::mutex> lock(metrics_mutex_);
  auto it = active_timers_.find(timer_id);
  if (it != active_timers_.end()) {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - it->second.second);
    double duration_ms = duration.count() / 1000.0;

    // Record the operation (internal call, already locked)
    auto &metrics = metrics_[it->second.first];
    metrics.operation_name = it->second.first;
    metrics.update(duration_ms, data_size);

    active_timers_.erase(it);
  }
}

void PerformanceProfiler::record_operation(const std::string &operation_name,
                                           double duration_ms,
                                           size_t data_size) {
  if (!enabled_) {
    return;
  }

  std::lock_guard<std::mutex> lock(metrics_mutex_);
  auto &metrics = metrics_[operation_name];
  metrics.operation_name = operation_name;
  metrics.update(duration_ms, data_size);
}

PerformanceProfiler::OperationMetrics
PerformanceProfiler::get_metrics(const std::string &operation_name) {
  std::lock_guard<std::mutex> lock(metrics_mutex_);
  auto it = metrics_.find(operation_name);
  if (it != metrics_.end()) {
    return it->second;
  }
  return OperationMetrics{};
}

std::unordered_map<std::string, PerformanceProfiler::OperationMetrics>
PerformanceProfiler::get_all_metrics() {
  std::lock_guard<std::mutex> lock(metrics_mutex_);
  return metrics_;
}

void PerformanceProfiler::clear_metrics() {
  std::lock_guard<std::mutex> lock(metrics_mutex_);
  metrics_.clear();
  active_timers_.clear();
}

void PerformanceProfiler::print_report(bool sort_by_time) {
  std::lock_guard<std::mutex> lock(metrics_mutex_);

  if (metrics_.empty()) {
    std::cout << "No performance metrics collected." << std::endl;
    return;
  }

  // Convert to vector for sorting
  std::vector<std::pair<std::string, OperationMetrics>> sorted_metrics;
  for (const auto &pair : metrics_) {
    sorted_metrics.push_back(pair);
  }

  // Sort by total time or call count
  if (sort_by_time) {
    std::sort(sorted_metrics.begin(), sorted_metrics.end(),
              [](const auto &a, const auto &b) {
                return a.second.total_time_ms > b.second.total_time_ms;
              });
  } else {
    std::sort(sorted_metrics.begin(), sorted_metrics.end(),
              [](const auto &a, const auto &b) {
                return a.second.call_count > b.second.call_count;
              });
  }

  // Print header
  std::cout << "\n" << std::string(80, '=') << std::endl;
  std::cout << "                    GNU Radio Linear Algebra Performance Report"
            << std::endl;
  std::cout << std::string(80, '=') << std::endl;
  std::cout << std::left << std::setw(25) << "Operation" << std::right
            << std::setw(8) << "Calls" << std::setw(12) << "Total(ms)"
            << std::setw(10) << "Avg(ms)" << std::setw(10) << "Min(ms)"
            << std::setw(10) << "Max(ms)" << std::setw(12) << "MB/s"
            << std::endl;
  std::cout << std::string(80, '-') << std::endl;

  // Print metrics
  for (const auto &pair : sorted_metrics) {
    const auto &metrics = pair.second;
    std::cout << std::left << std::setw(25) << metrics.operation_name
              << std::right << std::setw(8) << metrics.call_count
              << std::setw(12) << std::fixed << std::setprecision(3)
              << metrics.total_time_ms << std::setw(10) << std::fixed
              << std::setprecision(3) << metrics.average_time_ms
              << std::setw(10) << std::fixed << std::setprecision(3)
              << metrics.min_time_ms << std::setw(10) << std::fixed
              << std::setprecision(3) << metrics.max_time_ms << std::setw(12)
              << std::fixed << std::setprecision(2)
              << metrics.throughput_mb_per_sec << std::endl;
  }

  std::cout << std::string(80, '=') << std::endl;

  // Calculate total statistics
  double total_time = 0.0;
  size_t total_calls = 0;
  for (const auto &pair : metrics_) {
    total_time += pair.second.total_time_ms;
    total_calls += pair.second.call_count;
  }

  std::cout << "Total operations: " << total_calls << std::endl;
  std::cout << "Total time: " << std::fixed << std::setprecision(3)
            << total_time << " ms" << std::endl;
  std::cout << "Average per operation: " << std::fixed << std::setprecision(3)
            << (total_calls > 0 ? total_time / total_calls : 0.0) << " ms"
            << std::endl;
  std::cout << std::string(80, '=') << std::endl << std::endl;
}

void PerformanceProfiler::export_json(const std::string &filename) {
  std::lock_guard<std::mutex> lock(metrics_mutex_);

  std::ostringstream json;
  json << "{\n";
  json << "  \"performance_metrics\": {\n";
  json << "    \"timestamp\": \""
       << std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count()
       << "\",\n";
  json << "    \"operations\": [\n";

  bool first = true;
  for (const auto &pair : metrics_) {
    if (!first)
      json << ",\n";
    first = false;

    const auto &metrics = pair.second;
    json << "      {\n";
    json << "        \"name\": \"" << metrics.operation_name << "\",\n";
    json << "        \"call_count\": " << metrics.call_count << ",\n";
    json << "        \"total_time_ms\": " << metrics.total_time_ms << ",\n";
    json << "        \"average_time_ms\": " << metrics.average_time_ms << ",\n";
    json << "        \"min_time_ms\": " << metrics.min_time_ms << ",\n";
    json << "        \"max_time_ms\": " << metrics.max_time_ms << ",\n";
    json << "        \"throughput_mb_per_sec\": "
         << metrics.throughput_mb_per_sec << ",\n";
    json << "        \"total_data_size\": " << metrics.total_data_size << "\n";
    json << "      }";
  }

  json << "\n    ]\n";
  json << "  }\n";
  json << "}\n";

  if (filename.empty()) {
    std::cout << json.str();
  } else {
    std::ofstream file(filename);
    file << json.str();
  }
}

void PerformanceProfiler::export_csv(const std::string &filename) {
  std::lock_guard<std::mutex> lock(metrics_mutex_);

  std::ostringstream csv;
  csv << "Operation,Calls,Total_Time_ms,Average_Time_ms,Min_Time_ms,Max_Time_"
         "ms,Throughput_MB_per_s,Total_Data_Size\n";

  for (const auto &pair : metrics_) {
    const auto &metrics = pair.second;
    csv << metrics.operation_name << "," << metrics.call_count << ","
        << metrics.total_time_ms << "," << metrics.average_time_ms << ","
        << metrics.min_time_ms << "," << metrics.max_time_ms << ","
        << metrics.throughput_mb_per_sec << "," << metrics.total_data_size
        << "\n";
  }

  if (filename.empty()) {
    std::cout << csv.str();
  } else {
    std::ofstream file(filename);
    file << csv.str();
  }
}

void PerformanceProfiler::set_enabled(bool enabled) { enabled_ = enabled; }

bool PerformanceProfiler::is_enabled() { return enabled_; }

void PerformanceProfiler::set_max_recent_samples(size_t max_recent) {
  max_recent_samples_ = max_recent;

  // Trim existing recent samples if necessary
  std::lock_guard<std::mutex> lock(metrics_mutex_);
  for (auto &pair : metrics_) {
    auto &recent = pair.second.recent_times;
    if (recent.size() > max_recent_samples_) {
      recent.erase(recent.begin(), recent.end() - max_recent_samples_);
    }
  }
}

} // namespace linalg
} // namespace gr