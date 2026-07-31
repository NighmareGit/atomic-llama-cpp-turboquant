// Per-layer heatmap rollup from node_timings (placement-grade heat).
// See heatmap-rollup.h. Minimal key-scanning JSON extraction (no JSON lib),
// consistent with tools/llama-gpipe-profiler/llama-gpipe-profiler.cpp.

#include "heatmap-rollup.h"

#include "common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal JSON helpers (key scanning). Fixed, simple formats only.
// ---------------------------------------------------------------------------

static bool extract_json_string(const std::string & line, const std::string & key, std::string & out) {
    std::string search = "\"" + key + "\"";
    size_t pos = line.find(search);
    if (pos == std::string::npos) return false;
    pos = line.find(':', pos + search.size());
    if (pos == std::string::npos) return false;
    pos = line.find('"', pos + 1);
    if (pos == std::string::npos) return false;
    size_t end = line.find('"', pos + 1);
    if (end == std::string::npos) return false;
    out = line.substr(pos + 1, end - pos - 1);
    return true;
}

static bool extract_json_uint64(const std::string & line, const std::string & key, uint64_t & out) {
    std::string search = "\"" + key + "\"";
    size_t pos = line.find(search);
    if (pos == std::string::npos) return false;
    size_t colon = line.find(':', pos + search.size());
    if (colon == std::string::npos) return false;
    size_t end = line.find_first_of(",}]", colon + 1);
    if (end == std::string::npos) return false;
    out = (uint64_t) std::stoull(line.substr(colon + 1, end - colon - 1));
    return true;
}

// Parse a single {name,us} object from a node_timing entry substring.
static bool parse_node_timing_obj(const std::string & obj, heatmap_node_timing & out) {
    if (!extract_json_string(obj, "name", out.name)) return false;
    uint64_t us = 0;
    // Prefer avg_us (aggregated form); fall back to us.
    if (!extract_json_uint64(obj, "avg_us", us)) {
        if (!extract_json_uint64(obj, "us", us)) return false;
    }
    out.us = (double) us;
    out.samples = 1;
    return true;
}

// Parse a [...] array of {name,us} objects at key in line. Returns the entries.
static std::vector<heatmap_node_timing> extract_node_timing_array(const std::string & line, const std::string & key) {
    std::vector<heatmap_node_timing> out;
    std::string search = "\"" + key + "\"";
    size_t pos = line.find(search);
    if (pos == std::string::npos) return out;
    size_t start = line.find('[', pos + search.size());
    if (start == std::string::npos) return out;
    size_t cursor = start + 1;
    while (cursor < line.size()) {
        size_t ob = line.find('{', cursor);
        if (ob == std::string::npos) break;
        size_t cb = line.find('}', ob + 1);
        if (cb == std::string::npos) break;
        heatmap_node_timing nt;
        if (parse_node_timing_obj(line.substr(ob, cb - ob + 1), nt)) {
            out.push_back(nt);
        }
        cursor = cb + 1;
        size_t close = line.find(']', cursor);
        if (close != std::string::npos && line.find('{', cursor) > close) break;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Running-average merge by name
// ---------------------------------------------------------------------------

static std::vector<heatmap_node_timing> merge_by_name(std::vector<heatmap_node_timing> nodes) {
    // Accumulate by name: weighted running average. Preserves first-seen order.
    struct acc { std::string name; double us; int samples; };
    std::vector<acc> out;
    std::map<std::string, size_t> index; // name -> position in out
    for (auto & n : nodes) {
        auto it = index.find(n.name);
        if (it == index.end()) {
            index[n.name] = out.size();
            out.push_back({n.name, n.us, n.samples});
        } else {
            acc & a = out[it->second];
            double total = a.us * a.samples + n.us * n.samples;
            a.samples += n.samples;
            a.us = total / a.samples;
        }
    }
    std::vector<heatmap_node_timing> result;
    result.reserve(out.size());
    for (auto & a : out) {
        result.push_back({a.name, a.us, a.samples});
    }
    return result;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::vector<heatmap_node_timing> heatmap_parse_node_timings_sched_trace(const std::string & path) {
    std::vector<heatmap_node_timing> accum;
    std::ifstream in(path);
    if (!in) return accum;
    std::string line;
    while (std::getline(in, line)) {
        if (line.find("\"graph_compute_async\"") == std::string::npos) continue;
        auto arr = extract_node_timing_array(line, "node_timings");
        accum.insert(accum.end(), arr.begin(), arr.end());
    }
    return merge_by_name(accum);
}

std::vector<heatmap_node_timing> heatmap_parse_node_timings_server_telemetry(const std::string & path) {
    std::vector<heatmap_node_timing> accum;
    std::ifstream in(path);
    if (!in) return accum;
    std::string line;
    while (std::getline(in, line)) {
        if (line.find("\"event\":\"node_timings\"") == std::string::npos &&
            line.find("\"event\": \"node_timings\"") == std::string::npos) continue;
        auto arr = extract_node_timing_array(line, "entries");
        accum.insert(accum.end(), arr.begin(), arr.end());
    }
    return merge_by_name(accum);
}

std::vector<heatmap_node_timing> heatmap_merge_node_timings(
    const std::vector<heatmap_node_timing> & a,
    const std::vector<heatmap_node_timing> & b) {
    std::vector<heatmap_node_timing> all = a;
    all.insert(all.end(), b.begin(), b.end());
    return merge_by_name(all);
}

// ---------------------------------------------------------------------------
// Layer index + op category
// ---------------------------------------------------------------------------

bool heatmap_layer_index_from_name(const std::string & name, int & out_idx) {
    // l_out-N.* (decoder layers)
    const std::string prefix = "l_out-";
    if (name.rfind(prefix, 0) == 0) {
        size_t dash = prefix.size();
        size_t end = name.find('.', dash);
        if (end == std::string::npos) end = name.size();
        out_idx = std::stoi(name.substr(dash, end - dash));
        return true;
    }
    // blk.N.* (encoder blocks)
    const std::string bprefix = "blk.";
    if (name.rfind(bprefix, 0) == 0) {
        size_t dot = bprefix.size();
        size_t end = name.find('.', dot);
        if (end == std::string::npos) end = name.size();
        out_idx = std::stoi(name.substr(dot, end - dot));
        return true;
    }
    return false;
}

std::string heatmap_op_category_from_op(const std::string & op) {
    if (op.find("norm") != std::string::npos) return "norm";
    if (op.rfind("attn", 0) == 0) return "attn";
    if (op.rfind("ffn", 0) == 0) return "ffn";
    return "other";
}

// ---------------------------------------------------------------------------
// Rollup
// ---------------------------------------------------------------------------

bool heatmap_compute_layer_rollup(
    const std::vector<heatmap_node_timing> & nodes,
    int n_layers,
    std::vector<heatmap_layer_rollup> & rollup,
    std::vector<heatmap_op_category> & categories) {
    rollup.clear();
    categories.clear();

    // Group nodes by layer index.
    struct layer_acc { double us = 0; std::vector<std::string> names; };
    std::map<int, layer_acc> layers;
    // op category accumulators.
    struct cat_acc { double us = 0; int n = 0; std::vector<std::string> ops; };
    std::map<std::string, cat_acc> cats;

    for (const auto & nt : nodes) {
        int idx = -1;
        if (heatmap_layer_index_from_name(nt.name, idx)) {
            auto & la = layers[idx];
            la.us += nt.us;
            la.names.push_back(nt.name);
        }
        // Op category from suffix after last '.'.
        std::string op = nt.name;
        size_t dot = nt.name.rfind('.');
        if (dot != std::string::npos) op = nt.name.substr(dot + 1);
        std::string cat = heatmap_op_category_from_op(op);
        auto & ca = cats[cat];
        ca.us += nt.us;
        ca.n += 1;
        ca.ops.push_back(op);
    }

    for (auto & kv : layers) {
        heatmap_layer_rollup r;
        r.idx = kv.first;
        r.us = (uint64_t) std::round(kv.second.us);
        r.ms = kv.second.us / 1000.0;
        r.n_nodes = (int) kv.second.names.size();
        r.nodes = std::move(kv.second.names);
        rollup.push_back(r);
    }

    // Emit categories in fixed order: attn, ffn, norm, other.
    const std::string k_cats[] = {"attn", "ffn", "norm", "other"};
    for (const auto & cat : k_cats) {
        auto it = cats.find(cat);
        if (it == cats.end()) continue;
        heatmap_op_category c;
        c.category = it->first;
        c.us = (uint64_t) std::round(it->second.us);
        c.ms = it->second.us / 1000.0;
        c.n_nodes = it->second.n;
        c.ops = std::move(it->second.ops);
        categories.push_back(c);
    }

    return n_layers > 0 && rollup.size() == (size_t) n_layers;
}

std::string heatmap_heat_status(size_t rollup_size, int n_layers, bool has_device_telemetry) {
    if (n_layers > 0 && rollup_size == (size_t) n_layers) return "full";
    if (rollup_size > 0) return "partial";
    if (has_device_telemetry) return "estimated";
    return "none";
}
