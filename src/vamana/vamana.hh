#pragma once

/**
 * Vamana Index: GPU-accelerated beam-search graph index with RDMA disaggregated memory.
 *
 * Replaces HNSW with a single-layer directed graph using:
 *  - Beam search (instead of multi-layer greedy descent)
 *  - RobustPrune (alpha-based diversity pruning instead of HNSW heuristic)
 *  - Exact L2 distances for search (GPU-accelerated)
 *  - Full-precision L2 distances for insert/prune (GPU-accelerated)
 *  - Coroutine-based RDMA overlap (from SHINE)
 */

#include <algorithm>
#include <cuda_runtime.h>

#include "cache/cache.hh"
#include "common/constants.hh"
#include "common/debug.hh"
#include "common/types.hh"
#include "compute_thread.hh"
#include "coroutine.hh"
#include "gpu/gpu_awaitable.hh"
#include "gpu/gpu_kernel_launcher.hh"
#include "rdma/vamana_rdma_operations.hh"
#include "remote_pointer.hh"
#include "vamana/vamana_neighborlist.hh"
#include "vamana/vamana_node.hh"

namespace vamana {

template <class Distance>
class Vamana {
public:
    Vamana(u32 R, u32 beam_width, u32 beam_width_construction, f64 alpha,
           u32 k, u32 rabitq_bits, u32 dim, bool use_cache)
        : R_(R),
          beam_width_(beam_width),
          beam_width_construction_(beam_width_construction),
          alpha_(static_cast<f32>(alpha)),
          k_(k),
          rabitq_bits_(rabitq_bits),
          dim_(dim),
          use_cache_(use_cache) {
        lib_assert(beam_width_ >= k_, "beam_width must be >= k");
        VamanaNode::init_static_storage(dim, R, rabitq_bits);
    }

    // =========================================================================
    // Search (knn)
    // =========================================================================

    VamanaCoroutine knn(node_t q_id, const span<element_t> components,
                        const u_ptr<ComputeThread>& thread) const {
        dbg::print(dbg::stream{} << "T" << thread->get_id() << " queries " << q_id << "\n");
        ++thread->stats.processed;

        auto& coro_state = thread->current_vamana_coroutine();
        auto& beam = coro_state.beam;
        auto& visited = coro_state.visited_nodes;
        auto& gpu = thread->gpu_buffers;
        const u32 coro_id = thread->current_coroutine_id();  // current coroutine id managed by scheduler

        // Read medoid
        RemotePtr medoid_ptr = co_await rdma::vamana::read_medoid_ptr(thread);

        s_ptr<VamanaNode> medoid_node;
        {
            auto coro = cache_lookup(medoid_ptr, medoid_node, thread, true);
            while (!coro.handle.done()) {
                co_await std::suspend_always{};
                coro.handle.resume();
            }
        }

        // Upload query vector to GPU once.
        auto& gs = gpu.state(coro_id);
        std::memcpy(gs.h_query, components.data(), dim_ * sizeof(float));
        cudaMemcpyAsync(gs.d_query, gs.h_query, dim_ * sizeof(float),
                        cudaMemcpyHostToDevice, gs.stream);

        // Initialize beam with medoid (exact L2 distance)
        distance_t medoid_dist = Distance::dist(components, medoid_node->components(), VamanaNode::DIM);
        ++thread->stats.distcomps;

        beam.clear();
        beam.push_back({medoid_ptr, medoid_dist, false});
        visited.clear();
        visited.insert(medoid_ptr);

        // Beam search loop using exact L2 distances on GPU.
        while (true) {
            // Find closest unexpanded candidate
            i32 best_idx = -1;
            distance_t best_dist = std::numeric_limits<distance_t>::max();
            for (i32 i = 0; i < static_cast<i32>(beam.size()); ++i) {
                if (!beam[i].expanded && beam[i].distance < best_dist) {
                    best_dist = beam[i].distance;
                    best_idx = i;
                }
            }
            if (best_idx < 0) break;  // all expanded

            beam[best_idx].expanded = true;

            // Read neighbor list of best candidate
            s_ptr<VamanaNeighborlist> nlist =
                co_await rdma::vamana::read_vamana_neighbors(beam[best_idx].rptr, thread);
            ++thread->stats.visited_neighborlists;

            // Filter unvisited neighbors
            vec<RemotePtr> unvisited;
            for (const RemotePtr& n_ptr : nlist->view()) {
                if (n_ptr.is_null()) continue;
                if (!visited.contains(n_ptr)) {
                    visited.insert(n_ptr);
                    unvisited.push_back(n_ptr);
                }
            }

            if (unvisited.empty()) continue;

            // Batch RDMA read full vectors for exact distance computation.
            vec<byte_t*> vec_bufs = co_await rdma::vamana::batch_read_vectors(unvisited, thread);
            const u32 n_batch = unvisited.size();
            for (u32 i = 0; i < n_batch; ++i) {
                std::memcpy(gs.h_candidate_vecs + i * dim_,
                           reinterpret_cast<float*>(vec_bufs[i]),
                           dim_ * sizeof(float));
                thread->buffer_allocator.free_buffer(vec_bufs[i], dim_ * sizeof(element_t));
            }

            cudaMemcpyAsync(gs.d_candidate_vecs, gs.h_candidate_vecs,
                           n_batch * dim_ * sizeof(float),
                           cudaMemcpyHostToDevice, gs.stream);

            gpu::launch_batch_l2_distances(
                gs.stream, gs.event,
                gs.d_query, gs.d_candidate_vecs,
                gs.d_distances, n_batch, dim_);
            co_await gpu::GpuAwaitable{thread.get()};

            // Copy distances back
            cudaMemcpyAsync(gs.h_distances, gs.d_distances,
                           n_batch * sizeof(float),
                           cudaMemcpyDeviceToHost, gs.stream);
            cudaStreamSynchronize(gs.stream);

            // Update beam
            for (u32 i = 0; i < n_batch; ++i) {
                distance_t dist = gs.h_distances[i];
                ++thread->stats.distcomps;
                insert_into_beam(beam, unvisited[i], dist, beam_width_);
            }
        }

        // Extract top-k results
        // Beam is sorted by distance (best first after insert_into_beam)
        std::sort(beam.begin(), beam.end(),
                  [](const auto& a, const auto& b) { return a.distance < b.distance; });

        auto& results = thread->query_results[q_id];
        u32 count = std::min(k_, static_cast<u32>(beam.size()));

        // We need to resolve node IDs — read the nodes for top-k
        for (u32 i = 0; i < count; ++i) {
            s_ptr<VamanaNode> node;
            auto coro = cache_lookup(beam[i].rptr, node, thread, true);
            while (!coro.handle.done()) {
                co_await std::suspend_always{};
                coro.handle.resume();
            }
            results.push_back(node->id());
        }

        beam.clear();
        visited.clear();
    }

    // =========================================================================
    // Insert
    // =========================================================================

    VamanaCoroutine insert(node_t id, const span<element_t> components,
                           const u_ptr<ComputeThread>& thread) {
        dbg::print(dbg::stream{} << "T" << thread->get_id() << " inserts " << id << "\n");
        ++thread->stats.processed;

        auto& coro_state = thread->current_vamana_coroutine();
        auto& beam = coro_state.beam;
        auto& visited = coro_state.visited_nodes;
        auto& gpu = thread->gpu_buffers;
        const u32 coro_id = thread->current_coroutine_id();        auto& gs = gpu.state(coro_id);

        // Read medoid
        RemotePtr medoid_ptr = co_await rdma::vamana::read_medoid_ptr(thread);

        // Handle first insert (empty index)
        if (medoid_ptr.is_null()) {
            RemotePtr new_ptr = co_await rdma::vamana::allocate_vamana_node(thread);

            // Quantize vector with GPU
            std::memcpy(gs.h_query, components.data(), dim_ * sizeof(float));
            cudaMemcpyAsync(gs.d_query, gs.h_query, dim_ * sizeof(float),
                           cudaMemcpyHostToDevice, gs.stream);

            // Allocate temp device buffer for RaBitQ output
            u32 rabitq_data_size = VamanaNode::RABITQ_SIZE;
            uint8_t* d_rabitq_out = reinterpret_cast<uint8_t*>(gs.d_rabitq_vecs);  // reuse buffer

            gpu::launch_rabitq_quantize_single(
                gs.stream, gs.event,
                gs.d_query,
                gpu.d_rotation_matrix(),
                gpu.d_centroid(),
                d_rabitq_out,
                dim_, rabitq_bits_, gpu.t_const());
            co_await gpu::GpuAwaitable{thread.get()};

            // Copy RaBitQ data back
            cudaMemcpyAsync(gs.h_rabitq_vecs, d_rabitq_out, rabitq_data_size,
                           cudaMemcpyDeviceToHost, gs.stream);
            cudaStreamSynchronize(gs.stream);

            // Write node with no neighbors
            vec<RemotePtr> empty_neighbors;
            s_ptr<VamanaNode> new_node = co_await rdma::vamana::write_vamana_node(
                new_ptr, id, components, gs.h_rabitq_vecs, span<RemotePtr>{},
                0, false, false, thread);

            // Try to set as medoid
            RemotePtr old_medoid = co_await rdma::vamana::swap_medoid_ptr(
                RemotePtr{}, new_ptr, thread);
            if (old_medoid.is_null()) {
                // Success — we set the medoid
                co_await rdma::vamana::write_vamana_header(new_ptr, true, false, false, thread);
                co_return;
            }
            // Another thread won the race — fall through to normal insert
            medoid_ptr = old_medoid;
        }

        // Phase 1: Beam search for candidate neighbors
        s_ptr<VamanaNode> medoid_node;
        {
            auto coro = cache_lookup(medoid_ptr, medoid_node, thread, true);
            while (!coro.handle.done()) {
                co_await std::suspend_always{};
                coro.handle.resume();
            }
        }

        distance_t medoid_dist = Distance::dist(components, medoid_node->components(), VamanaNode::DIM);
        ++thread->stats.distcomps;

        beam.clear();
        beam.push_back({medoid_ptr, medoid_dist, false});
        visited.clear();
        visited.insert(medoid_ptr);

        // Beam search using full L2 distances (exact, not RaBitQ)
        // Upload query vector once
        std::memcpy(gs.h_query, components.data(), dim_ * sizeof(float));
        cudaMemcpyAsync(gs.d_query, gs.h_query, dim_ * sizeof(float),
                       cudaMemcpyHostToDevice, gs.stream);

        while (true) {
            i32 best_idx = -1;
            distance_t best_dist = std::numeric_limits<distance_t>::max();
            for (i32 i = 0; i < static_cast<i32>(beam.size()); ++i) {
                if (!beam[i].expanded && beam[i].distance < best_dist) {
                    best_dist = beam[i].distance;
                    best_idx = i;
                }
            }
            if (best_idx < 0) break;

            beam[best_idx].expanded = true;

            s_ptr<VamanaNeighborlist> nlist =
                co_await rdma::vamana::read_vamana_neighbors(beam[best_idx].rptr, thread);
            ++thread->stats.visited_neighborlists;

            vec<RemotePtr> unvisited;
            for (const RemotePtr& n_ptr : nlist->view()) {
                if (n_ptr.is_null()) continue;
                if (!visited.contains(n_ptr)) {
                    visited.insert(n_ptr);
                    unvisited.push_back(n_ptr);
                }
            }

            if (unvisited.empty()) continue;

            // Batch read full vectors for exact distance
            vec<byte_t*> vec_bufs = co_await rdma::vamana::batch_read_vectors(unvisited, thread);

            // Stage to GPU
            const u32 n_batch = unvisited.size();
            for (u32 i = 0; i < n_batch; ++i) {
                std::memcpy(gs.h_candidate_vecs + i * dim_,
                           reinterpret_cast<float*>(vec_bufs[i]),
                           dim_ * sizeof(float));
                thread->buffer_allocator.free_buffer(vec_bufs[i], dim_ * sizeof(element_t));
            }

            cudaMemcpyAsync(gs.d_candidate_vecs, gs.h_candidate_vecs,
                           n_batch * dim_ * sizeof(float),
                           cudaMemcpyHostToDevice, gs.stream);

            gpu::launch_batch_l2_distances(
                gs.stream, gs.event,
                gs.d_query, gs.d_candidate_vecs,
                gs.d_distances, n_batch, dim_);
            co_await gpu::GpuAwaitable{thread.get()};

            cudaMemcpyAsync(gs.h_distances, gs.d_distances,
                           n_batch * sizeof(float),
                           cudaMemcpyDeviceToHost, gs.stream);
            cudaStreamSynchronize(gs.stream);

            for (u32 i = 0; i < n_batch; ++i) {
                ++thread->stats.distcomps;
                insert_into_beam(beam, unvisited[i], gs.h_distances[i], beam_width_construction_);
            }

            // (bump allocator; no individual free)
        }

        // Phase 2: GPU RobustPrune to select R neighbors
        std::sort(beam.begin(), beam.end(),
                  [](const auto& a, const auto& b) { return a.distance < b.distance; });

        const u32 n_candidates = beam.size();

        // Collect candidate RemotePtrs and distances (already sorted)
        vec<RemotePtr> candidate_rptrs;
        candidate_rptrs.reserve(n_candidates);
        for (auto& entry : beam) {
            candidate_rptrs.push_back(entry.rptr);
        }

        // Batch read full vectors for all candidates (for RobustPrune)
        vec<byte_t*> cand_vec_bufs = co_await rdma::vamana::batch_read_vectors(candidate_rptrs, thread);

        // Stage to GPU: candidate vectors + distances
        for (u32 i = 0; i < n_candidates; ++i) {
            std::memcpy(gs.h_candidate_vecs + i * dim_,
                       reinterpret_cast<float*>(cand_vec_bufs[i]),
                       dim_ * sizeof(float));
            gs.h_candidate_dists[i] = beam[i].distance;
            thread->buffer_allocator.free_buffer(cand_vec_bufs[i], dim_ * sizeof(element_t));
        }

        cudaMemcpyAsync(gs.d_candidate_vecs, gs.h_candidate_vecs,
                       n_candidates * dim_ * sizeof(float),
                       cudaMemcpyHostToDevice, gs.stream);
        cudaMemcpyAsync(gs.d_candidate_dists, gs.h_candidate_dists,
                       n_candidates * sizeof(float),
                       cudaMemcpyHostToDevice, gs.stream);

        gpu::launch_robust_prune(
            gs.stream, gs.event,
            gs.d_query,  // source vector (the new node being inserted)
            gs.d_candidate_vecs,
            gs.d_candidate_dists,
            nullptr,
            n_candidates, dim_, alpha_, R_,
            gs.d_pruned_indices, gs.d_pruned_count);
        co_await gpu::GpuAwaitable{thread.get()};

        cudaMemcpyAsync(gs.h_pruned_indices, gs.d_pruned_indices,
                       R_ * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost, gs.stream);
        cudaMemcpyAsync(gs.h_pruned_count, gs.d_pruned_count,
                       sizeof(uint32_t),
                       cudaMemcpyDeviceToHost, gs.stream);
        cudaStreamSynchronize(gs.stream);

        const u32 pruned_count = *gs.h_pruned_count;

        // Map pruned indices back to RemotePtrs
        vec<RemotePtr> selected_neighbors;
        selected_neighbors.reserve(pruned_count);
        for (u32 i = 0; i < pruned_count; ++i) {
            u32 idx = gs.h_pruned_indices[i];
            selected_neighbors.push_back(candidate_rptrs[idx]);
        }

        // Free candidate vector buffers (bump allocator; no individual free)

        // Phase 3: Quantize new vector with RaBitQ
        uint8_t* d_rabitq_out = reinterpret_cast<uint8_t*>(gs.d_rabitq_vecs);
        gpu::launch_rabitq_quantize_single(
            gs.stream, gs.event,
            gs.d_query,
            gpu.d_rotation_matrix(),
            gpu.d_centroid(),
            d_rabitq_out,
            dim_, rabitq_bits_, gpu.t_const());
        co_await gpu::GpuAwaitable{thread.get()};

        u32 rabitq_data_size = VamanaNode::RABITQ_SIZE;
        cudaMemcpyAsync(gs.h_rabitq_vecs, d_rabitq_out, rabitq_data_size,
                       cudaMemcpyDeviceToHost, gs.stream);
        cudaStreamSynchronize(gs.stream);

        // Phase 4: Allocate and write new node
        RemotePtr new_ptr = co_await rdma::vamana::allocate_vamana_node(thread);

        s_ptr<VamanaNode> new_node = co_await rdma::vamana::write_vamana_node(
            new_ptr, id, components, gs.h_rabitq_vecs,
            span<RemotePtr>{selected_neighbors.data(), selected_neighbors.size()},
            static_cast<u8>(pruned_count), false, false, thread);

        // Phase 5: Update reverse edges (bidirectional connectivity)
        for (const RemotePtr& neighbor_ptr : selected_neighbors) {
            // Lock the neighbor
            s_ptr<VamanaNode> neighbor_node =
                co_await rdma::vamana::read_vamana_node(neighbor_ptr, thread);
            {
                auto coro = rdma::vamana::spinlock_vamana_node(neighbor_node, thread);
                while (!coro.handle.done()) {
                    co_await std::suspend_always{};
                    coro.handle.resume();
                }
            }

            // Read neighbor's neighbor list
            s_ptr<VamanaNeighborlist> neighbor_nlist =
                co_await rdma::vamana::read_vamana_neighbors(neighbor_ptr, thread);

            if (neighbor_nlist->num_neighbors() < R_) {
                // Room available — just append
                neighbor_nlist->add(new_ptr);

                // Write updated neighbor list
                co_await rdma::vamana::write_vamana_neighbors(
                    neighbor_node,
                    neighbor_nlist->view(),
                    neighbor_nlist->num_neighbors(),
                    thread);
            } else {
                // Need to prune: gather all candidate neighbors + new node
                vec<RemotePtr> all_candidate_ptrs;
                all_candidate_ptrs.reserve(neighbor_nlist->num_neighbors() + 1);
                for (const auto& n : neighbor_nlist->view()) {
                    all_candidate_ptrs.push_back(n);
                }
                all_candidate_ptrs.push_back(new_ptr);

                u32 n_all = all_candidate_ptrs.size();
                vec<byte_t*> all_vec_bufs(n_all, nullptr);

                // Read remote vectors for the existing neighbors only.
                vec<RemotePtr> remote_candidate_ptrs;
                remote_candidate_ptrs.reserve(n_all - 1);
                for (u32 i = 0; i + 1 < n_all; ++i) {
                    remote_candidate_ptrs.push_back(all_candidate_ptrs[i]);
                }
                vec<byte_t*> remote_vec_bufs =
                    co_await rdma::vamana::batch_read_vectors(remote_candidate_ptrs, thread);
                for (u32 i = 0; i + 1 < n_all; ++i) {
                    all_vec_bufs[i] = remote_vec_bufs[i];
                }

                // Upload neighbor's vector as query for prune
                std::memcpy(gs.h_query, neighbor_node->components().data(), dim_ * sizeof(float));
                cudaMemcpyAsync(gs.d_query, gs.h_query, dim_ * sizeof(float),
                               cudaMemcpyHostToDevice, gs.stream);

                // Compute distances from neighbor to all candidates
                for (u32 i = 0; i < n_all; ++i) {
                    if (i + 1 == n_all) {
                        std::memcpy(gs.h_candidate_vecs + i * dim_,
                                   components.data(),
                                   dim_ * sizeof(float));
                    } else {
                        std::memcpy(gs.h_candidate_vecs + i * dim_,
                                   reinterpret_cast<float*>(all_vec_bufs[i]),
                                   dim_ * sizeof(float));
                    }
                }

                cudaMemcpyAsync(gs.d_candidate_vecs, gs.h_candidate_vecs,
                               n_all * dim_ * sizeof(float),
                               cudaMemcpyHostToDevice, gs.stream);

                gpu::launch_batch_l2_distances(
                    gs.stream, gs.event,
                    gs.d_query, gs.d_candidate_vecs,
                    gs.d_distances, n_all, dim_);
                co_await gpu::GpuAwaitable{thread.get()};

                cudaMemcpyAsync(gs.h_distances, gs.d_distances,
                               n_all * sizeof(float),
                               cudaMemcpyDeviceToHost, gs.stream);
                cudaStreamSynchronize(gs.stream);

                // Sort candidates by distance
                vec<std::pair<u32, float>> idx_dist;
                idx_dist.reserve(n_all);
                for (u32 i = 0; i < n_all; ++i) {
                    idx_dist.push_back({i, gs.h_distances[i]});
                }
                std::sort(idx_dist.begin(), idx_dist.end(),
                          [](const auto& a, const auto& b) { return a.second < b.second; });

                // Reorder candidate vecs and distances for prune kernel
                for (u32 i = 0; i < n_all; ++i) {
                    // Copy sorted distances
                    // We can reuse h_candidate_dists as temp
                    gs.h_candidate_dists[i] = idx_dist[i].second;
                    gs.h_candidate_order[i] = idx_dist[i].first;
                }

                // Upload sorted distances and the corresponding original candidate order.
                cudaMemcpyAsync(gs.d_candidate_dists, gs.h_candidate_dists,
                               n_all * sizeof(float),
                               cudaMemcpyHostToDevice, gs.stream);
                cudaMemcpyAsync(gs.d_candidate_order, gs.h_candidate_order,
                               n_all * sizeof(uint32_t),
                               cudaMemcpyHostToDevice, gs.stream);

                for (u32 i = 0; i + 1 < n_all; ++i) {
                    thread->buffer_allocator.free_buffer(all_vec_bufs[i], dim_ * sizeof(element_t));
                }

                gpu::launch_robust_prune(
                    gs.stream, gs.event,
                    gs.d_query,
                    gs.d_candidate_vecs,
                    gs.d_candidate_dists,
                    gs.d_candidate_order,
                    n_all, dim_, alpha_, R_,
                    gs.d_pruned_indices, gs.d_pruned_count);
                co_await gpu::GpuAwaitable{thread.get()};

                cudaMemcpyAsync(gs.h_pruned_indices, gs.d_pruned_indices,
                               R_ * sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, gs.stream);
                cudaMemcpyAsync(gs.h_pruned_count, gs.d_pruned_count,
                               sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, gs.stream);
                cudaStreamSynchronize(gs.stream);

                u32 new_count = *gs.h_pruned_count;
                vec<RemotePtr> pruned_neighbors;
                pruned_neighbors.reserve(new_count);
                for (u32 i = 0; i < new_count; ++i) {
                    pruned_neighbors.push_back(all_candidate_ptrs[gs.h_pruned_indices[i]]);
                }

                co_await rdma::vamana::write_vamana_neighbors(
                    neighbor_node,
                    span<RemotePtr>{pruned_neighbors.data(), pruned_neighbors.size()},
                    static_cast<u8>(new_count),
                    thread);

                // (bump allocator; no individual free)
            }

            // Unlock neighbor
            co_await rdma::vamana::unlock_vamana_node(neighbor_node, thread);
        }

        beam.clear();
        visited.clear();
    }

    // =========================================================================
    // Index size estimation
    // =========================================================================

    size_t estimate_index_size(size_t num_nodes) const {
        return num_nodes * VamanaNode::total_size();
    }

private:
    // =========================================================================
    // Beam management
    // =========================================================================

    static void insert_into_beam(vec<VamanaCoroutine::BeamEntry>& beam,
                                 const RemotePtr& rptr, distance_t dist,
                                 u32 max_beam_width) {
        // Find insertion position (beam is maintained sorted by distance ascending)
        auto it = std::lower_bound(beam.begin(), beam.end(), dist,
            [](const VamanaCoroutine::BeamEntry& e, distance_t d) {
                return e.distance < d;
            });

        // Insert
        beam.insert(it, {rptr, dist, false});

        // Trim if over capacity
        if (beam.size() > max_beam_width) {
            beam.resize(max_beam_width);
        }
    }

    // =========================================================================
    // Cache lookup (currently bypasses cache; VamanaNode caching TBD)
    // =========================================================================

    MinorCoroutine cache_lookup(RemotePtr rptr,
                                s_ptr<VamanaNode>& value,
                                const u_ptr<ComputeThread>& thread,
                                bool admit) const {
        if (!use_cache_) {
            value = co_await rdma::vamana::read_vamana_node(rptr, thread);
            co_return;
        }

        auto cache_entry = thread->cache.get<VamanaNode>(rptr);
        if (cache_entry.has_value()) {
            value = *cache_entry;
            ++thread->stats.cache_hits;
        } else {
            value = co_await rdma::vamana::read_vamana_node(rptr, thread);
            if (admit) {
                thread->cache.insert(rptr, value, thread->get_id());
            }
            ++thread->stats.cache_misses;
        }
    }

private:
    const u32 R_;
    const u32 beam_width_;
    const u32 beam_width_construction_;
    const f32 alpha_;
    const u32 k_;
    const u32 rabitq_bits_;
    const u32 dim_;
    const bool use_cache_;
};

}  // namespace vamana
