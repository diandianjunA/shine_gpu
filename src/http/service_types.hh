#pragma once

#include <chrono>
#include <future>

#include "common/types.hh"

namespace service {

struct InsertRequest {
  node_t id;
  vec<element_t> components;
  std::promise<bool> result;
  std::chrono::steady_clock::time_point enqueued_at{std::chrono::steady_clock::now()};
};

struct QueryRequest {
  vec<element_t> components;
  u32 k;
  std::promise<vec<node_t>> result;
  std::chrono::steady_clock::time_point enqueued_at{std::chrono::steady_clock::now()};
};

using InsertQueue = concurrent_queue<InsertRequest*>;
using QueryQueue = concurrent_queue<QueryRequest*>;

}  // namespace service

// Expose commonly used types at global scope
using service::InsertQueue;
using service::QueryQueue;
