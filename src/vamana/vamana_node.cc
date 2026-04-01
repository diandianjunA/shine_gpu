#include "vamana/vamana_node.hh"

#include "compute_thread.hh"

VamanaNode::~VamanaNode() {
  // Buffer memory is managed by BufferAllocator's bump allocator.
  // Individual deallocation is not supported; memory is reclaimed in bulk.
}
