#pragma once

/**
 * RDMA read operations for the Vamana index.
 * Follows the same patterns as rdma_reads.hh for HNSW but adapted
 * for the fixed-size VamanaNode layout.
 */

#include "compute_thread.hh"
#include "coroutine.hh"
#include "remote_pointer.hh"
#include "vamana/vamana_neighborlist.hh"
#include "vamana/vamana_node.hh"

namespace rdma::vamana {

/**
 * Read a complete VamanaNode from remote memory.
 * Reads header + id + edge_count + pad + vector (not rabitq or neighbors).
 */
inline auto read_vamana_node(RemotePtr rptr, const u_ptr<ComputeThread>& thread) {
    const size_t read_size = VamanaNode::size_until_vector_end();
    byte_t* node_ptr = thread->buffer_allocator.allocate_buffer(read_size);

    thread->stats.rdma_reads_in_bytes += read_size;
    thread->track_post();

    const QP& qp = thread->ctx->qps[rptr.memory_node()]->qp;
    qp->post_send(reinterpret_cast<u64>(node_ptr),
                  read_size,
                  thread->ctx->get_lkey(),
                  IBV_WR_RDMA_READ,
                  true,
                  false,
                  thread->ctx->get_remote_mrt(rptr.memory_node()),
                  rptr.byte_offset(),
                  0,
                  thread->create_wr_id());

    struct awaitable {
        RemotePtr rptr;
        byte_t* node_ptr;
        size_t read_size;
        const u_ptr<ComputeThread>& thread;

        static bool await_ready() { return false; }
        static void await_suspend(std::coroutine_handle<>) {}
        s_ptr<VamanaNode> await_resume() {
            return std::make_shared<VamanaNode>(node_ptr, read_size, rptr, thread.get());
        }
    };

    return awaitable{rptr, node_ptr, read_size, thread};
}

/**
 * Read the full VamanaNode including rabitq data and neighbor list.
 */
inline auto read_vamana_node_full(RemotePtr rptr, const u_ptr<ComputeThread>& thread) {
    const size_t read_size = VamanaNode::total_size();
    byte_t* node_ptr = thread->buffer_allocator.allocate_vamana_node(thread->get_id());

    thread->stats.rdma_reads_in_bytes += read_size;
    thread->track_post();

    const QP& qp = thread->ctx->qps[rptr.memory_node()]->qp;
    qp->post_send(reinterpret_cast<u64>(node_ptr),
                  read_size,
                  thread->ctx->get_lkey(),
                  IBV_WR_RDMA_READ,
                  true,
                  false,
                  thread->ctx->get_remote_mrt(rptr.memory_node()),
                  rptr.byte_offset(),
                  0,
                  thread->create_wr_id());

    struct awaitable {
        RemotePtr rptr;
        byte_t* node_ptr;
        size_t read_size;
        const u_ptr<ComputeThread>& thread;

        static bool await_ready() { return false; }
        static void await_suspend(std::coroutine_handle<>) {}
        s_ptr<VamanaNode> await_resume() {
            return std::make_shared<VamanaNode>(node_ptr, read_size, rptr, thread.get());
        }
    };

    return awaitable{rptr, node_ptr, read_size, thread};
}

/**
 * Read the neighbor list portion of a VamanaNode.
 * Reads: edge_count(1B) + R * RemotePtr(8B) from the neighbors offset.
 */
inline auto read_vamana_neighbors(RemotePtr node_rptr, const u_ptr<ComputeThread>& thread) {
    // Read from edge_count offset through end of neighbors (includes 3B padding in between)
    const size_t read_offset = VamanaNode::offset_edge_count();
    const size_t read_end = VamanaNode::offset_neighbors() + VamanaNode::NEIGHBORS_SIZE;
    const size_t read_size = read_end - read_offset;
    byte_t* local_buffer = thread->buffer_allocator.allocate_buffer(read_size);

    thread->stats.rdma_reads_in_bytes += read_size;
    thread->track_post();

    const QP& qp = thread->ctx->qps[node_rptr.memory_node()]->qp;
    qp->post_send(reinterpret_cast<u64>(local_buffer),
                  read_size,
                  thread->ctx->get_lkey(),
                  IBV_WR_RDMA_READ,
                  true,
                  false,
                  thread->ctx->get_remote_mrt(node_rptr.memory_node()),
                  node_rptr.byte_offset() + read_offset,
                  0,
                  thread->create_wr_id());

    struct awaitable {
        byte_t* local_buffer;
        size_t read_size;
        const u_ptr<ComputeThread>& thread;

        static bool await_ready() { return false; }
        static void await_suspend(std::coroutine_handle<>) {}
        s_ptr<VamanaNeighborlist> await_resume() {
            // local_buffer has: [edge_count(1B) | pad(3B) | neighbors(R*8B)]
            // Compact: move neighbors from offset 4 to offset 1
            // edge_count already at position 0
            // Move neighbors from offset 4 (1B count + 3B pad) to offset 1
            std::memmove(local_buffer + sizeof(u8),
                        local_buffer + (VamanaNode::offset_neighbors() - VamanaNode::offset_edge_count()),
                        VamanaNode::NEIGHBORS_SIZE);
            return std::make_shared<VamanaNeighborlist>(local_buffer, read_size, thread.get());
        }
    };

    return awaitable{local_buffer, read_size, thread};
}

/**
 * Read just the RaBitQ data portion of a VamanaNode.
 * Returns raw bytes: packed_bits + add(4B) + rescale(4B).
 */
inline auto read_rabitq_vector(RemotePtr node_rptr, const u_ptr<ComputeThread>& thread) {
    const size_t rabitq_size = VamanaNode::RABITQ_SIZE;
    byte_t* local_buffer = thread->buffer_allocator.allocate_buffer(rabitq_size);

    thread->stats.rdma_reads_in_bytes += rabitq_size;
    thread->track_post();

    const QP& qp = thread->ctx->qps[node_rptr.memory_node()]->qp;
    qp->post_send(reinterpret_cast<u64>(local_buffer),
                  rabitq_size,
                  thread->ctx->get_lkey(),
                  IBV_WR_RDMA_READ,
                  true,
                  false,
                  thread->ctx->get_remote_mrt(node_rptr.memory_node()),
                  node_rptr.byte_offset() + VamanaNode::offset_rabitq(),
                  0,
                  thread->create_wr_id());

    struct awaitable {
        byte_t* local_buffer;
        size_t rabitq_size;
        const u_ptr<ComputeThread>& thread;

        static bool await_ready() { return false; }
        static void await_suspend(std::coroutine_handle<>) {}
        std::pair<byte_t*, size_t> await_resume() {
            return {local_buffer, rabitq_size};
        }
    };

    return awaitable{local_buffer, rabitq_size, thread};
}

/**
 * Batch read RaBitQ vectors for multiple nodes.
 * Posts multiple RDMA reads, one co_await for all.
 * Returns a vector of (buffer, size) pairs.
 */
inline auto batch_read_rabitq(const vec<RemotePtr>& node_rptrs, const u_ptr<ComputeThread>& thread) {
    const size_t rabitq_size = VamanaNode::RABITQ_SIZE;
    vec<byte_t*> buffers;
    buffers.reserve(node_rptrs.size());

    for (const auto& rptr : node_rptrs) {
        byte_t* local_buffer = thread->buffer_allocator.allocate_buffer(rabitq_size);
        buffers.push_back(local_buffer);

        thread->stats.rdma_reads_in_bytes += rabitq_size;
        thread->track_post();

        const QP& qp = thread->ctx->qps[rptr.memory_node()]->qp;
        qp->post_send(reinterpret_cast<u64>(local_buffer),
                      rabitq_size,
                      thread->ctx->get_lkey(),
                      IBV_WR_RDMA_READ,
                      true,
                      false,
                      thread->ctx->get_remote_mrt(rptr.memory_node()),
                      rptr.byte_offset() + VamanaNode::offset_rabitq(),
                      0,
                      thread->create_wr_id());
    }

    struct awaitable {
        vec<byte_t*> buffers;

        static bool await_ready() { return false; }
        static void await_suspend(std::coroutine_handle<>) {}
        vec<byte_t*> await_resume() { return std::move(buffers); }
    };

    return awaitable{std::move(buffers)};
}

/**
 * Batch read full vectors (float components) for multiple nodes.
 * Used during insert for RobustPrune which needs full-precision vectors.
 */
inline auto batch_read_vectors(const vec<RemotePtr>& node_rptrs, const u_ptr<ComputeThread>& thread) {
    const size_t vec_size = VamanaNode::DIM * sizeof(element_t);
    vec<byte_t*> buffers;
    buffers.reserve(node_rptrs.size());

    for (const auto& rptr : node_rptrs) {
        byte_t* local_buffer = thread->buffer_allocator.allocate_buffer(vec_size);
        buffers.push_back(local_buffer);

        thread->stats.rdma_reads_in_bytes += vec_size;
        thread->track_post();

        const QP& qp = thread->ctx->qps[rptr.memory_node()]->qp;
        qp->post_send(reinterpret_cast<u64>(local_buffer),
                      vec_size,
                      thread->ctx->get_lkey(),
                      IBV_WR_RDMA_READ,
                      true,
                      false,
                      thread->ctx->get_remote_mrt(rptr.memory_node()),
                      rptr.byte_offset() + VamanaNode::offset_vector(),
                      0,
                      thread->create_wr_id());
    }

    struct awaitable {
        vec<byte_t*> buffers;

        static bool await_ready() { return false; }
        static void await_suspend(std::coroutine_handle<>) {}
        vec<byte_t*> await_resume() { return std::move(buffers); }
    };

    return awaitable{std::move(buffers)};
}

/**
 * Read multiple full VamanaNodes in batch.
 */
inline auto read_vamana_nodes(const span<RemotePtr> remote_ptrs, const u_ptr<ComputeThread>& thread) {
    vec<s_ptr<VamanaNode>> nodes;
    nodes.reserve(remote_ptrs.size());

    const size_t read_size = VamanaNode::size_until_vector_end();

    for (auto& rptr : remote_ptrs) {
        byte_t* node_ptr = thread->buffer_allocator.allocate_buffer(read_size);
        nodes.emplace_back(std::make_shared<VamanaNode>(node_ptr, read_size, rptr, thread.get()));

        thread->stats.rdma_reads_in_bytes += read_size;
        thread->track_post();

        const QP& qp = thread->ctx->qps[rptr.memory_node()]->qp;
        qp->post_send(reinterpret_cast<u64>(node_ptr),
                      read_size,
                      thread->ctx->get_lkey(),
                      IBV_WR_RDMA_READ,
                      true,
                      false,
                      thread->ctx->get_remote_mrt(rptr.memory_node()),
                      rptr.byte_offset(),
                      0,
                      thread->create_wr_id());
    }

    struct awaitable {
        vec<s_ptr<VamanaNode>> nodes;

        static bool await_ready() { return false; }
        static void await_suspend(std::coroutine_handle<>) {}
        vec<s_ptr<VamanaNode>> await_resume() { return std::move(nodes); }
    };

    return awaitable{std::move(nodes)};
}

/**
 * Read the medoid pointer from memory node 0 (stored at offset 8, same as entry_point).
 */
inline auto read_medoid_ptr(const u_ptr<ComputeThread>& thread) {
    thread->stats.rdma_reads_in_bytes += sizeof(u64);
    thread->track_post();

    const QP& qp = thread->ctx->qps[0]->qp;
    qp->post_send(reinterpret_cast<u64>(thread->coros_pointer_slot()),
                  sizeof(u64),
                  thread->ctx->get_lkey(),
                  IBV_WR_RDMA_READ,
                  true,
                  false,
                  thread->ctx->get_remote_mrt(0),
                  8,  // medoid_ptr at offset 8 (after free_ptr)
                  0,
                  thread->create_wr_id());

    struct awaitable {
        const u_ptr<ComputeThread>& thread;

        static bool await_ready() { return false; }
        static void await_suspend(std::coroutine_handle<>) {}
        RemotePtr await_resume() const { return RemotePtr{*thread->coros_pointer_slot()}; }
    };

    return awaitable{thread};
}

}  // namespace rdma::vamana
