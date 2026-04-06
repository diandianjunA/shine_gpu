#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "common/statistics.hh"
#include "common/types.hh"
#include "nlohmann/json.hh"

namespace service::breakdown {

using Clock = std::chrono::steady_clock;
using Nanoseconds = std::chrono::nanoseconds;

enum class Operation : u8 { query = 0, insert = 1 };

enum class Phase : u8 {
  query_medoid_fetch = 0,
  query_neighbor_fetch,
  query_vector_fetch,
  query_rabitq_fetch,
  query_gpu_prepare,
  query_gpu_distance,
  query_gpu_rerank,
  query_cpu_merge_sort,
  query_result_materialize,
  insert_medoid_fetch,
  insert_candidate_search,
  insert_candidate_vector_fetch,
  insert_gpu_distance,
  insert_gpu_prune,
  insert_quantize,
  insert_remote_alloc,
  insert_new_node_write,
  insert_neighbor_update,
  insert_medoid_update,
  count
};

constexpr size_t kPhaseCount = static_cast<size_t>(Phase::count);

inline constexpr std::array<std::string_view, kPhaseCount> kPhaseNames = {
  "medoid_fetch_ns",
  "neighbor_fetch_ns",
  "vector_fetch_ns",
  "rabitq_fetch_ns",
  "gpu_prepare_ns",
  "gpu_distance_ns",
  "gpu_rerank_ns",
  "cpu_merge_sort_ns",
  "result_materialize_ns",
  "medoid_fetch_ns",
  "candidate_search_ns",
  "candidate_vector_fetch_ns",
  "gpu_distance_ns",
  "gpu_prune_ns",
  "quantize_ns",
  "remote_alloc_ns",
  "new_node_write_ns",
  "neighbor_update_ns",
  "medoid_update_ns",
};

inline constexpr bool phase_matches_operation(const Phase phase, const Operation operation) {
  const size_t idx = static_cast<size_t>(phase);
  if (operation == Operation::query) {
    return idx <= static_cast<size_t>(Phase::query_result_materialize);
  }
  return idx >= static_cast<size_t>(Phase::insert_medoid_fetch);
}

inline constexpr std::string_view operation_name(const Operation operation) {
  return operation == Operation::query ? "query" : "insert";
}

struct ThreadCounterDelta {
  u64 rdma_read_bytes{};
  u64 rdma_write_bytes{};
  u64 neighbor_rdma_bytes{};
  u64 vector_rdma_bytes{};
  u64 rabitq_rdma_bytes{};
  u64 h2d_bytes{};
  u64 d2h_bytes{};
  u64 l2_kernels{};
  u64 prune_kernels{};
  u64 rabitq_kernels{};
  u64 exact_reranks{};
  u64 visited_nodes{};
  u64 visited_neighborlists{};
  u64 remote_allocations{};
  u64 overflow_prunes{};
  u64 cache_hits{};
  u64 cache_misses{};
};

inline ThreadCounterDelta diff_thread_counters(const statistics::ThreadStatistics& end,
                                               const statistics::ThreadStatistics& start,
                                               const Operation operation) {
  ThreadCounterDelta out{};

  if (operation == Operation::query) {
    out.rdma_read_bytes = end.query_rdma_reads_in_bytes - start.query_rdma_reads_in_bytes;
    out.rdma_write_bytes = end.query_rdma_writes_in_bytes - start.query_rdma_writes_in_bytes;
    out.neighbor_rdma_bytes = end.query_neighbor_rdma_reads_in_bytes - start.query_neighbor_rdma_reads_in_bytes;
    out.vector_rdma_bytes = end.query_vector_rdma_reads_in_bytes - start.query_vector_rdma_reads_in_bytes;
    out.rabitq_rdma_bytes = end.query_rabitq_rdma_reads_in_bytes - start.query_rabitq_rdma_reads_in_bytes;
    out.h2d_bytes = end.query_h2d_bytes - start.query_h2d_bytes;
    out.d2h_bytes = end.query_d2h_bytes - start.query_d2h_bytes;
    out.rabitq_kernels = end.query_rabitq_kernels - start.query_rabitq_kernels;
    out.exact_reranks = end.query_exact_reranks - start.query_exact_reranks;
    out.visited_nodes =
      (end.visited_nodes - start.visited_nodes) + (end.visited_nodes_l0 - start.visited_nodes_l0);
    out.visited_neighborlists = end.visited_neighborlists - start.visited_neighborlists;
    out.cache_hits = end.cache_hits - start.cache_hits;
    out.cache_misses = end.cache_misses - start.cache_misses;
    return out;
  }

  out.rdma_read_bytes = end.build_rdma_reads_in_bytes - start.build_rdma_reads_in_bytes;
  out.rdma_write_bytes = end.build_rdma_writes_in_bytes - start.build_rdma_writes_in_bytes;
  out.neighbor_rdma_bytes = end.build_neighbor_rdma_reads_in_bytes - start.build_neighbor_rdma_reads_in_bytes;
  out.vector_rdma_bytes = end.build_vector_rdma_reads_in_bytes - start.build_vector_rdma_reads_in_bytes;
  out.rabitq_rdma_bytes = end.build_rabitq_rdma_reads_in_bytes - start.build_rabitq_rdma_reads_in_bytes;
  out.h2d_bytes = end.build_h2d_bytes - start.build_h2d_bytes;
  out.d2h_bytes = end.build_d2h_bytes - start.build_d2h_bytes;
  out.l2_kernels = end.build_l2_kernels - start.build_l2_kernels;
  out.prune_kernels = end.build_prune_kernels - start.build_prune_kernels;
  out.remote_allocations = end.remote_allocations - start.remote_allocations;
  out.overflow_prunes = end.build_overflow_prunes - start.build_overflow_prunes;
  return out;
}

struct Sample {
  explicit Sample(Operation op) : operation(op) {}

  Operation operation;
  Clock::time_point enqueued_at{};
  Clock::time_point dequeued_at{};
  Clock::time_point started_at{};
  Clock::time_point finished_at{};
  std::array<u64, kPhaseCount> phase_ns{};
  statistics::ThreadStatistics start_counters{};
  statistics::ThreadStatistics end_counters{};
  u64 queue_wait_ns{};
  u64 service_ns{};
  u64 end_to_end_ns{};
  u64 lock_attempts{};
  u64 lock_retries{};
  u64 cas_failures{};
  bool started_flag{};
  bool finished_flag{};

  void mark_started(const Clock::time_point dequeued,
                    const Clock::time_point started,
                    const statistics::ThreadStatistics& counters) {
    dequeued_at = dequeued;
    started_at = started;
    start_counters = counters;
    started_flag = true;
    queue_wait_ns =
      static_cast<u64>(std::chrono::duration_cast<Nanoseconds>(dequeued_at - enqueued_at).count());
  }

  void mark_finished(const Clock::time_point finished, const statistics::ThreadStatistics& counters) {
    finished_at = finished;
    end_counters = counters;
    finished_flag = true;
    service_ns = static_cast<u64>(std::chrono::duration_cast<Nanoseconds>(finished_at - started_at).count());
    end_to_end_ns = static_cast<u64>(std::chrono::duration_cast<Nanoseconds>(finished_at - enqueued_at).count());
  }

  void add_phase(const Phase phase, const u64 ns) { phase_ns[static_cast<size_t>(phase)] += ns; }

  ThreadCounterDelta counters() const { return diff_thread_counters(end_counters, start_counters, operation); }
};

struct Aggregate {
  Operation operation{Operation::query};
  size_t count{};
  u64 total_queue_wait_ns{};
  u64 total_service_ns{};
  u64 total_end_to_end_ns{};
  std::vector<u64> end_to_end_latencies_ns{};
  std::vector<u64> service_latencies_ns{};
  std::array<u64, kPhaseCount> phase_ns{};
  ThreadCounterDelta counters{};
  u64 lock_attempts{};
  u64 lock_retries{};
  u64 cas_failures{};

  [[nodiscard]] u64 phase_sum() const {
    u64 total = 0;
    for (size_t i = 0; i < phase_ns.size(); ++i) {
      if (phase_matches_operation(static_cast<Phase>(i), operation)) {
        total += phase_ns[i];
      }
    }
    return total;
  }

  [[nodiscard]] u64 unattributed_ns() const {
    const u64 total = phase_sum();
    return total_service_ns > total ? total_service_ns - total : 0;
  }
};

struct Report {
  Aggregate query{};
  Aggregate insert{};

  [[nodiscard]] bool has_query() const { return query.count > 0; }
  [[nodiscard]] bool has_insert() const { return insert.count > 0; }
};

inline void add_sample(Aggregate& aggregate, const Sample& sample) {
  if (!sample.finished_flag) {
    return;
  }

  aggregate.operation = sample.operation;
  ++aggregate.count;
  aggregate.total_queue_wait_ns += sample.queue_wait_ns;
  aggregate.total_service_ns += sample.service_ns;
  aggregate.total_end_to_end_ns += sample.end_to_end_ns;
  aggregate.end_to_end_latencies_ns.push_back(sample.end_to_end_ns);
  aggregate.service_latencies_ns.push_back(sample.service_ns);
  for (size_t i = 0; i < aggregate.phase_ns.size(); ++i) {
    aggregate.phase_ns[i] += sample.phase_ns[i];
  }

  const ThreadCounterDelta delta = sample.counters();
  aggregate.counters.rdma_read_bytes += delta.rdma_read_bytes;
  aggregate.counters.rdma_write_bytes += delta.rdma_write_bytes;
  aggregate.counters.neighbor_rdma_bytes += delta.neighbor_rdma_bytes;
  aggregate.counters.vector_rdma_bytes += delta.vector_rdma_bytes;
  aggregate.counters.rabitq_rdma_bytes += delta.rabitq_rdma_bytes;
  aggregate.counters.h2d_bytes += delta.h2d_bytes;
  aggregate.counters.d2h_bytes += delta.d2h_bytes;
  aggregate.counters.l2_kernels += delta.l2_kernels;
  aggregate.counters.prune_kernels += delta.prune_kernels;
  aggregate.counters.rabitq_kernels += delta.rabitq_kernels;
  aggregate.counters.exact_reranks += delta.exact_reranks;
  aggregate.counters.visited_nodes += delta.visited_nodes;
  aggregate.counters.visited_neighborlists += delta.visited_neighborlists;
  aggregate.counters.remote_allocations += delta.remote_allocations;
  aggregate.counters.overflow_prunes += delta.overflow_prunes;
  aggregate.counters.cache_hits += delta.cache_hits;
  aggregate.counters.cache_misses += delta.cache_misses;
  aggregate.lock_attempts += sample.lock_attempts;
  aggregate.lock_retries += sample.lock_retries;
  aggregate.cas_failures += sample.cas_failures;
}

inline double ns_to_ms(const u64 ns) { return static_cast<double>(ns) / 1'000'000.0; }

inline u64 percentile_ns(std::vector<u64> values, const double percentile) {
  if (values.empty()) {
    return 0;
  }
  std::sort(values.begin(), values.end());
  const double idx = percentile * static_cast<double>(values.size() - 1);
  return values[static_cast<size_t>(idx)];
}

inline nlohmann::json aggregate_to_json(const Aggregate& aggregate) {
  using json = nlohmann::json;
  json out;
  out["operation"] = operation_name(aggregate.operation);
  out["count"] = aggregate.count;
  out["latency"] = {
    {"queue_wait_ns", aggregate.total_queue_wait_ns},
    {"service_ns", aggregate.total_service_ns},
    {"end_to_end_ns", aggregate.total_end_to_end_ns},
    {"mean_queue_wait_ns", aggregate.count == 0 ? 0 : aggregate.total_queue_wait_ns / aggregate.count},
    {"mean_service_ns", aggregate.count == 0 ? 0 : aggregate.total_service_ns / aggregate.count},
    {"mean_end_to_end_ns", aggregate.count == 0 ? 0 : aggregate.total_end_to_end_ns / aggregate.count},
    {"p50_end_to_end_ns", percentile_ns(aggregate.end_to_end_latencies_ns, 0.50)},
    {"p95_end_to_end_ns", percentile_ns(aggregate.end_to_end_latencies_ns, 0.95)},
    {"p99_end_to_end_ns", percentile_ns(aggregate.end_to_end_latencies_ns, 0.99)},
    {"p50_service_ns", percentile_ns(aggregate.service_latencies_ns, 0.50)},
    {"p95_service_ns", percentile_ns(aggregate.service_latencies_ns, 0.95)},
    {"p99_service_ns", percentile_ns(aggregate.service_latencies_ns, 0.99)},
  };

  json phases = json::object();
  for (size_t i = 0; i < aggregate.phase_ns.size(); ++i) {
    const auto phase = static_cast<Phase>(i);
    if (phase_matches_operation(phase, aggregate.operation)) {
      phases[std::string{kPhaseNames[i]}] = aggregate.phase_ns[i];
    }
  }
  phases["unattributed_ns"] = aggregate.unattributed_ns();
  out["phases"] = std::move(phases);

  out["counters"] = {
    {"rdma_read_bytes", aggregate.counters.rdma_read_bytes},
    {"rdma_write_bytes", aggregate.counters.rdma_write_bytes},
    {"neighbor_rdma_bytes", aggregate.counters.neighbor_rdma_bytes},
    {"vector_rdma_bytes", aggregate.counters.vector_rdma_bytes},
    {"rabitq_rdma_bytes", aggregate.counters.rabitq_rdma_bytes},
    {"h2d_bytes", aggregate.counters.h2d_bytes},
    {"d2h_bytes", aggregate.counters.d2h_bytes},
    {"l2_kernels", aggregate.counters.l2_kernels},
    {"prune_kernels", aggregate.counters.prune_kernels},
    {"rabitq_kernels", aggregate.counters.rabitq_kernels},
    {"exact_reranks", aggregate.counters.exact_reranks},
    {"visited_nodes", aggregate.counters.visited_nodes},
    {"visited_neighborlists", aggregate.counters.visited_neighborlists},
    {"remote_allocations", aggregate.counters.remote_allocations},
    {"overflow_prunes", aggregate.counters.overflow_prunes},
    {"cache_hits", aggregate.counters.cache_hits},
    {"cache_misses", aggregate.counters.cache_misses},
    {"lock_attempts", aggregate.lock_attempts},
    {"lock_retries", aggregate.lock_retries},
    {"cas_failures", aggregate.cas_failures},
  };
  return out;
}

inline std::string aggregate_text_summary(const Aggregate& aggregate) {
  std::ostringstream os;
  os << operation_name(aggregate.operation) << " breakdown\n";
  os << "  count: " << aggregate.count << '\n';
  os << "  latency_ms: mean=" << ns_to_ms(aggregate.count == 0 ? 0 : aggregate.total_end_to_end_ns / aggregate.count)
     << " p50=" << ns_to_ms(percentile_ns(aggregate.end_to_end_latencies_ns, 0.50))
     << " p95=" << ns_to_ms(percentile_ns(aggregate.end_to_end_latencies_ns, 0.95))
     << " p99=" << ns_to_ms(percentile_ns(aggregate.end_to_end_latencies_ns, 0.99)) << '\n';

  std::vector<std::pair<std::string, u64>> ranked;
  ranked.reserve(kPhaseCount + 1);
  for (size_t i = 0; i < aggregate.phase_ns.size(); ++i) {
    const auto phase = static_cast<Phase>(i);
    if (phase_matches_operation(phase, aggregate.operation)) {
      ranked.emplace_back(std::string{kPhaseNames[i]}, aggregate.phase_ns[i]);
    }
  }
  ranked.emplace_back("unattributed_ns", aggregate.unattributed_ns());
  std::sort(ranked.begin(), ranked.end(), [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });

  os << "  top_phases:\n";
  const size_t top = std::min<size_t>(3, ranked.size());
  for (size_t i = 0; i < top; ++i) {
    const double ratio = aggregate.total_service_ns == 0
                           ? 0.0
                           : static_cast<double>(ranked[i].second) / static_cast<double>(aggregate.total_service_ns);
    os << "    " << ranked[i].first << ": " << ns_to_ms(ranked[i].second) << " ms ("
       << ratio * 100.0 << "%)\n";
  }
  return os.str();
}

inline nlohmann::json report_to_json(const Report& report) {
  nlohmann::json out;
  if (report.has_query()) {
    out["query_breakdown"] = aggregate_to_json(report.query);
  }
  if (report.has_insert()) {
    out["insert_breakdown"] = aggregate_to_json(report.insert);
  }
  return out;
}

}  // namespace service::breakdown
