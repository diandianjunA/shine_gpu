#pragma once

#include "common/constants.hh"
#include "vamana/vamana_node.hh"

/**
 * Manages the memory-registered buffer globally per compute node.
 * Each thread has its own freelist which however may be accessed concurrently.
 */
class BufferAllocator {
public:
  explicit BufferAllocator(u32 num_threads, u64 buffer_size_bytes = COMPUTE_NODE_MAX_MEMORY) {
    // allocate a contiguous buffer for local memory
    local_buffer_.allocate(buffer_size_bytes);
    local_buffer_.touch_memory();

    buffer_ptr_ = local_buffer_.get_full_buffer();

    freelists_vamana_node_.resize(num_threads);
  }

  HugePage<byte_t>& get_raw_buffer() { return local_buffer_; }

  // Allocate a buffer suitable for a full VamanaNode (including rabitq + neighbors)
  [[nodiscard]] byte_t* allocate_vamana_node(u32 thread_id) {
    return get_free_space(VamanaNode::total_size(), freelists_vamana_node_[thread_id]);
  }

  // General-purpose allocation for buffers of arbitrary size
  [[nodiscard]] byte_t* allocate_buffer(size_t size) {
    return allocate(size);
  }

  [[nodiscard]] u64* allocate_pointer() { return reinterpret_cast<u64*>(allocate(sizeof(u64))); }

  size_t allocated_memory() const { return bump_pointer_; }

private:
  static size_t align(size_t size) {
    while (size % CACHELINE_SIZE != 0) {
      ++size;
    }

    return size;
  }

  byte_t* get_free_space(size_t size, concurrent_queue<byte_t*>& freelist) {
    byte_t* ptr;

    if (!freelist.try_dequeue(ptr)) {
      ptr = allocate(size);
    }

    return ptr;
  }

  byte_t* allocate(size_t size) {
    lib_assert(size > 0, "unable to allocate 0 bytes");

    byte_t* ptr = buffer_ptr_ + bump_pointer_.fetch_add(align(size));
    lib_assert(bump_pointer_ <= local_buffer_.buffer_size, "out of local memory");

    // do not track 8B pointers
    if (size > sizeof(u64)) {
      allocated_buffers_.push_back(ptr);
    }

    return ptr;
  }

private:
  byte_t* buffer_ptr_;
  std::atomic<idx_t> bump_pointer_{0};  // points to free space
  HugePage<byte_t> local_buffer_;

  // freelists per thread (but other threads may append to them)
  // this is significantly faster than having single global freelists
  vec<concurrent_queue<byte_t*>> freelists_vamana_node_;

  concurrent_vec<byte_t*> allocated_buffers_;  // track valid pointers (for cache eviction)
};