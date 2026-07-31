#pragma once

// Per-layer heatmap rollup from node_timings (placement-grade heat).
// Synthesizes true per-layer timings from raw node_timing entries
// (l_out-N.* / blk.N.*) emitted by sched-trace and RPC server telemetry.
// Pure: no GPU, no llama deps beyond std. Unit-tested via llama-common.

#include <cstdint>
#include <string>
#include <vector>

struct heatmap_node_timing {
    std::string name;   // raw tensor name, e.g. "l_out-14.attn_q"
    double      us = 0; // running-average microseconds
    int         samples = 0;
};

struct heatmap_layer_rollup {
    int         idx = 0;    // transformer layer index
    double      ms = 0;     // sum of node us in this layer, in ms
    uint64_t    us = 0;     // sum of node us in this layer
    int         n_nodes = 0;
    std::vector<std::string> nodes; // member tensor names
};

struct heatmap_op_category {
    std::string category; // attn | ffn | norm | other
    double      ms = 0;
    uint64_t    us = 0;
    int         n_nodes = 0;
    std::vector<std::string> ops; // op suffixes in this category
};

// Parse node_timings from sched-trace.jsonl graph_compute_async events.
// Merges repeated nodes by name with a running average.
std::vector<heatmap_node_timing> heatmap_parse_node_timings_sched_trace(const std::string & path);

// Parse node_timings from server-telemetry.jsonl node_timings events.
// Handles both raw and aggregated:true forms. Merges by name (running avg).
std::vector<heatmap_node_timing> heatmap_parse_node_timings_server_telemetry(const std::string & path);

// Merge two node-timing vectors by name (running-average weighted by samples).
std::vector<heatmap_node_timing> heatmap_merge_node_timings(
    const std::vector<heatmap_node_timing> & a,
    const std::vector<heatmap_node_timing> & b);

// Compute per-layer rollup + op categories from node timings.
// Returns true when rollup covers all n_layers (full heat).
bool heatmap_compute_layer_rollup(
    const std::vector<heatmap_node_timing> & nodes,
    int n_layers,
    std::vector<heatmap_layer_rollup> & rollup,
    std::vector<heatmap_op_category> & categories);

// Extract layer index from a tensor name. Returns true if it names a layer.
// Supports l_out-N.* (decoder) and blk.N.* (encoder).
bool heatmap_layer_index_from_name(const std::string & name, int & out_idx);

// Map an op suffix (the part after the last '.') to a category.
// Precedence: *norm* > attn* > ffn* > other. Matches TELEMETRY.md.
std::string heatmap_op_category_from_op(const std::string & op);

// Honest heat status from rollup coverage.
// full     -> rollup.size() == n_layers
// partial  -> some layers present
// none     -> no node timings at all
// estimated-> no node timings, but device-level telemetry exists
std::string heatmap_heat_status(size_t rollup_size, int n_layers, bool has_device_telemetry);
