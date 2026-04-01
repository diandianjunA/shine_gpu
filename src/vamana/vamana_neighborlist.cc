#include "vamana_neighborlist.hh"

#include "compute_thread.hh"

VamanaNeighborlist::~VamanaNeighborlist() {
  // Buffer memory is managed by BufferAllocator's bump allocator.
  // Individual deallocation is not supported; memory is reclaimed in bulk.
}
