#include "pipeline-diagnose.h"
#include "pipeline-jsonl.h"
#include "pipeline-trace-parse.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <numeric>
#include <regex>
#include <set>
#include <sstream>

namespace {

double kv_double(const std::map<std::string, std::string> & kv, const char * key, double def = 0.0) {
    auto it = kv.find(key);
    if (it == kv.end() || it->second.empty()) {
        return def;
    }
    try {
        return std::stod(it->second);
    } catch (...) {
        return def;
    }
}

int kv_int(const std::map<std::string, std::string> & kv, const char * key, int def = 0) {
    return (int) kv_double(kv, key, def);
}

std::map<std::string, std::string> parse_summary_kv(const std::string & text) {
    std::map<std::string, std::string> kv;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        const auto b = line.find_first_not_of(" \t\r\n");
        if (b == std::string::npos) {
            continue;
        }
        const auto e = line.find_last_not_of(" \t\r\n");
        const std::string stripped = line.substr(b, e - b + 1);

        std::smatch m;
        if (std::regex_search(stripped, m, std::regex(R"(split_total_ms_sum=([\d.]+))"))) {
            kv["split_total_ms_sum"] = m[1].str();
        }
        for (const char * key : {
                "assembly_overlap_count", "overlap_pct", "drain_flush_ms",
                "blocking_ms", "input_wait_copy_ms", "graph_compute_async_ms",
                "event_record_ms", "split_total_count",
            }) {
            const std::string pat = std::string(key) + R"(=([\d.]+))";
            if (std::regex_search(stripped, m, std::regex(pat))) {
                kv[key] = m[1].str();
            }
        }
        if (std::regex_match(stripped, m, std::regex(R"(backend(\d+)\s+splits=(\d+)\s+ms=([\d.]+))"))) {
            kv["backend" + m[1].str() + "_ms"] = m[3].str();
        }
    }
    return kv;
}

int estimate_gen_tokens(
        const std::vector<nlohmann::ordered_json> & sched_rows,
        const std::map<std::string, std::string> & summary_kv,
        int override_tokens,
        const fs::path & meta_file) {
    if (override_tokens > 0) {
        return std::max(1, override_tokens);
    }
    if (fs::exists(meta_file)) {
        std::ifstream in(meta_file);
        std::string line;
        while (std::getline(in, line)) {
            std::smatch m;
            if (std::regex_search(line, m, std::regex(R"((?:gen[_-]?tokens|n_gen)\s*[=:]\s*(\d+))", std::regex::icase))) {
                return std::max(1, std::stoi(m[1].str()));
            }
        }
    }
    std::vector<nlohmann::ordered_json> splits;
    for (const auto & row : sched_rows) {
        if (row.contains("phase") && row["phase"].is_string() && row["phase"] == "split_total") {
            splits.push_back(row);
        }
    }
    if (splits.empty()) {
        return 128;
    }
    std::set<int> split_ids;
    for (const auto & row : splits) {
        if (row.contains("split")) {
            split_ids.insert(row["split"].get<int>());
        }
    }
    const int n_splits    = split_ids.empty() ? 3 : (int) split_ids.size();
    const int split_count = kv_int(summary_kv, "split_total_count", (int) splits.size());
    return std::max(1, split_count / std::max(1, n_splits));
}

static int64_t json_ts_us(const nlohmann::ordered_json & row) {
    if (!row.contains("ts_us")) {
        return 0;
    }
    return row["ts_us"].get<long long>();
}

int64_t gen_window_start_us(
        const std::vector<nlohmann::ordered_json> & sched_rows,
        const std::vector<nlohmann::ordered_json> & rpc_rows) {
    int64_t last_hash = 0;
    bool found_hash   = false;
    for (const auto & row : rpc_rows) {
        if (row.contains("cmd") && row["cmd"].get<int>() == 7) {
            const int64_t ts = json_ts_us(row);
            if (ts > last_hash) {
                last_hash  = ts;
                found_hash = true;
            }
        }
    }
    if (found_hash) {
        return last_hash;
    }
    std::vector<nlohmann::ordered_json> splits = sched_rows;
    std::sort(splits.begin(), splits.end(), [](const auto & a, const auto & b) {
        return json_ts_us(a) < json_ts_us(b);
    });
    if (splits.size() > 20) {
        return json_ts_us(splits[10]);
    }
    return 0;
}

std::vector<nlohmann::ordered_json> filter_gen(
        const std::vector<nlohmann::ordered_json> & rows,
        int64_t start_us) {
    if (start_us <= 0) {
        return rows;
    }
    std::vector<nlohmann::ordered_json> out;
    out.reserve(rows.size());
    for (const auto & row : rows) {
        const int64_t ts = row.contains("ts_us") ? row["ts_us"].get<int64_t>() : 0;
        if (ts >= start_us) {
            out.push_back(row);
        }
    }
    return out;
}

bool is_blocking(const nlohmann::ordered_json & row) {
    if (!row.contains("blocking")) {
        return false;
    }
    if (row["blocking"].is_boolean()) {
        return row["blocking"].get<bool>();
    }
    if (row["blocking"].is_string()) {
        const auto s = row["blocking"].get<std::string>();
        return s == "true" || s == "True";
    }
    return false;
}

double parse_g_tps(const fs::path & trace_dir) {
    const std::vector<fs::path> candidates = {
        trace_dir / "result.jsonl",
        trace_dir.parent_path() / "result.jsonl",
        trace_dir / ".." / "result.jsonl",
    };
    for (const auto & p : candidates) {
        if (!fs::exists(p)) {
            continue;
        }
        std::ifstream in(p);
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) {
                continue;
            }
            try {
                const auto row = nlohmann::ordered_json::parse(line);
                const double g = pipeline_result_g_tps(row);
                if (g > 0) {
                    return g;
                }
            } catch (...) {
                continue;
            }
        }
    }
    return 0.0;
}

std::vector<std::string> gpu_smells(const fs::path & gpu_dir) {
    std::vector<std::string> smells;
    const fs::path csv_path = gpu_dir / "nvidia-local.csv";
    if (!fs::exists(csv_path)) {
        return smells;
    }
    std::ifstream in(csv_path);
    std::string header_line;
    if (!std::getline(in, header_line)) {
        return smells;
    }
    std::vector<std::string> headers;
    {
        std::istringstream hs(header_line);
        std::string col;
        while (std::getline(hs, col, ',')) {
            headers.push_back(col);
        }
    }
    auto col_idx = [&](const char * name) -> int {
        for (size_t i = 0; i < headers.size(); ++i) {
            if (headers[i] == name) {
                return (int) i;
            }
        }
        return -1;
    };
    const int pi = col_idx("power_w");
    const int ui = col_idx("util_gpu_pct");
    const int ci = col_idx("sm_clock_mhz");
    if (pi < 0 || ui < 0) {
        return smells;
    }

    std::vector<double> powers;
    std::vector<double> utils;
    std::vector<double> clocks;
    std::string line;
    while (std::getline(in, line)) {
        std::vector<std::string> cols;
        std::istringstream ls(line);
        std::string col;
        while (std::getline(ls, col, ',')) {
            cols.push_back(col);
        }
        if ((size_t) std::max({pi, ui, ci}) >= cols.size()) {
            continue;
        }
        try {
            const double p = std::stod(cols[pi]);
            const double u = std::stod(cols[ui]);
            if (p > 0) {
                powers.push_back(p);
            }
            utils.push_back(u);
            if (ci >= 0 && (size_t) ci < cols.size()) {
                const double c = std::stod(cols[ci]);
                if (c > 0) {
                    clocks.push_back(c);
                }
            }
        } catch (...) {
            continue;
        }
    }
    if (powers.empty() || utils.empty()) {
        return smells;
    }
    constexpr double tdp = 300.0;
    const double max_u   = *std::max_element(utils.begin(), utils.end());
    const double max_p   = *std::max_element(powers.begin(), powers.end());
    const double max_p_pct = 100.0 * max_p / tdp;
    const double avg_u = std::accumulate(utils.begin(), utils.end(), 0.0) / utils.size();
    if (max_u >= 40 && max_p_pct < 10) {
        smells.push_back("GPU_METRIC_MISMATCH");
    }
    if (max_u < 15 && max_p_pct < 10) {
        smells.push_back("ORCHESTRATION_STALL");
    }
    if (!clocks.empty()) {
        const double cmax = *std::max_element(clocks.begin(), clocks.end());
        const double cmin = *std::min_element(clocks.begin(), clocks.end());
        if (cmax > 0 && cmin < 0.85 * cmax) {
            smells.push_back("CLOCK_THROTTLE");
        }
    }
    if (avg_u < 5 && max_p_pct < 8) {
        smells.push_back("LOW_DUTY_CYCLE");
    }
    return smells;
}

} // namespace

bool pipeline_diagnose_run(
        const std::string & telemetry_dir,
        const pipeline_diagnose_options & opts,
        pipeline_diagnose_result & out) {
    const fs::path trace_dir(telemetry_dir);
    pipeline_trace_parse(telemetry_dir);

    auto sched_rows = pipeline_load_jsonl(trace_dir / "sched-trace.jsonl");
    auto rpc_rows   = pipeline_load_jsonl(trace_dir / "rpc-trace.jsonl");

    std::string summary_text;
    const fs::path summary_file = trace_dir / "trace-summary.txt";
    if (fs::exists(summary_file)) {
        std::ifstream in(summary_file);
        std::ostringstream ss;
        ss << in.rdbuf();
        summary_text = ss.str();
    }
    auto summary_kv = parse_summary_kv(summary_text);

    fs::path meta_file = trace_dir.parent_path() / "meta.txt";
    if (!fs::exists(meta_file)) {
        meta_file = trace_dir / "meta.txt";
    }

    if (opts.gen_only && !sched_rows.empty()) {
        const int64_t start_us = gen_window_start_us(sched_rows, rpc_rows);
        sched_rows = filter_gen(sched_rows, start_us);
        rpc_rows   = filter_gen(rpc_rows, start_us);
    }

    const int gen_tokens = estimate_gen_tokens(sched_rows, summary_kv, opts.gen_tokens_override, meta_file);
    const double g_tps   = parse_g_tps(trace_dir);

    const double split_ms   = kv_double(summary_kv, "split_total_ms_sum");
    const double wait_ms    = kv_double(summary_kv, "input_wait_copy_ms");
    const double event_ms   = kv_double(summary_kv, "event_record_ms");
    const double compute_ms = kv_double(summary_kv, "graph_compute_async_ms");
    const double stall_ms   = wait_ms + event_ms;
    const double per_tok    = gen_tokens > 0 ? split_ms / gen_tokens : 0.0;
    const double ratio      = split_ms > 0 ? stall_ms / split_ms : 0.0;

    std::map<std::string, double> backends;
    std::string straggler_id;
    double straggler_ms = 0.0;
    for (const auto & [k, v] : summary_kv) {
        std::smatch m;
        if (std::regex_match(k, m, std::regex(R"(backend(\d+)_ms)"))) {
            const double ms  = std::stod(v);
            const double mpt = gen_tokens > 0 ? ms / gen_tokens : 0.0;
            backends[m[1].str()] = std::round(mpt * 1000.0) / 1000.0;
            if (mpt > straggler_ms) {
                straggler_ms  = mpt;
                straggler_id  = m[1].str();
            }
        }
    }

    const int overlap_count = kv_int(summary_kv, "assembly_overlap_count");
    const double overlap_pct = kv_double(summary_kv, "overlap_pct");
    const bool s5 = overlap_count > 0;
    const bool b6 = overlap_pct >= opts.overlap_target_pct;

    int blocking_count = 0;
    int fire_count     = 0;
    int graph_count    = 0;
    for (const auto & row : rpc_rows) {
        if (is_blocking(row)) {
            ++blocking_count;
        } else if (row.contains("cmd") && !row["cmd"].is_null()) {
            ++fire_count;
        }
        if (row.contains("cmd")) {
            const int cmd = row["cmd"].get<int>();
            if (cmd == 10 || cmd == 16) {
                ++graph_count;
            }
        }
    }

    double eff = 0.0;
    bool has_eff = false;
    if (g_tps > 0 && gen_tokens > 0) {
        const double wall_ms_per_token = 1000.0 / g_tps;
        const double serial_ms         = split_ms / gen_tokens;
        if (wall_ms_per_token > 0) {
            eff     = std::round(10000.0 * serial_ms / wall_ms_per_token) / 10000.0;
            has_eff = true;
        }
    }

    auto smells = gpu_smells(trace_dir / "gpu");

    out.diagnose = nlohmann::ordered_json::object();
    out.diagnose["version"]       = 1;
    out.diagnose["trace_dir"]     = trace_dir.string();
    out.diagnose["gen_only"]      = opts.gen_only;
    out.diagnose["gen_tokens_est"] = gen_tokens;
    out.diagnose["G_tps"]         = g_tps > 0 ? nlohmann::ordered_json(g_tps) : nlohmann::ordered_json(nullptr);
    out.diagnose["overlap_efficiency"] = has_eff ? nlohmann::ordered_json(eff) : nlohmann::ordered_json(nullptr);
    out.diagnose["drain_flush_ms"] = kv_double(summary_kv, "drain_flush_ms");
    out.diagnose["blocking_ms"]    = kv_double(summary_kv, "blocking_ms");
    out.diagnose["gate_s5"]        = s5 ? "PASS" : "FAIL";
    out.diagnose["gate_b6"]        = b6 ? "PASS" : "FAIL";
    out.diagnose["assembly_overlap_count"] = overlap_count;
    out.diagnose["overlap_pct"]          = overlap_pct;
    out.diagnose["overlap_target_pct"]   = opts.overlap_target_pct;
    out.diagnose["split_ms_per_token"]   = std::round(per_tok * 1000.0) / 1000.0;
    out.diagnose["stall_ratio"]           = std::round(ratio * 10000.0) / 10000.0;
    out.diagnose["stall_ms"]              = std::round(stall_ms * 100.0) / 100.0;
    out.diagnose["compute_ms"]            = std::round(compute_ms * 100.0) / 100.0;
    out.diagnose["backends_ms_per_token"] = backends;
    out.diagnose["straggler_backend"]     = straggler_id.empty() ? nullptr : nlohmann::ordered_json(straggler_id);
    out.diagnose["straggler_ms_per_token"] = std::round(straggler_ms * 1000.0) / 1000.0;
    out.diagnose["topology_class"]        = "client_split";
    out.diagnose["rpc_rtt_per_token"]     = gen_tokens > 0 ? std::round(100.0 * blocking_count / gen_tokens) / 100.0 : 0.0;
    out.diagnose["graph_submit_count"]      = graph_count;
    out.diagnose["blocking_rpc_count"]      = blocking_count;
    out.diagnose["fire_and_forget_count"]   = fire_count;
    out.diagnose["gpu_smell_flags"]         = smells;
    out.diagnose["path_c_reserved"] = {
        {"cross_endpoint_copy_count", nullptr},
        {"server_local_copy_count", nullptr},
    };

    if (!opts.baseline_dir.empty()) {
        const fs::path base_path = fs::path(opts.baseline_dir) / "diagnose.json";
        if (fs::exists(base_path)) {
            try {
                std::ifstream in(base_path);
                auto base = nlohmann::ordered_json::parse(in);
                out.diagnose["baseline_delta"] = {
                    {"G_tps", g_tps - base.value("G_tps", 0.0)},
                    {"overlap_pct", overlap_pct - base.value("overlap_pct", 0.0)},
                    {"stall_ratio", out.diagnose["stall_ratio"].get<double>() - base.value("stall_ratio", 0.0)},
                };
            } catch (...) {
            }
        }
    }

    const fs::path diagnose_out = trace_dir / "diagnose.json";
    std::ofstream diag_file(diagnose_out);
    if (!diag_file) {
        return false;
    }
    diag_file << out.diagnose.dump(2) << "\n";

    std::ostringstream human;
    human << "=== llama-pipeline-diagnose ===\n";
    human << "diagnose.json -> " << diagnose_out.string() << "\n";
    human << "gate_s5=" << out.diagnose["gate_s5"].get<std::string>()
          << " gate_b6=" << out.diagnose["gate_b6"].get<std::string>()
          << " overlap_pct=" << overlap_pct << "\n";
    human << "stall_ratio=" << out.diagnose["stall_ratio"].get<double>()
          << " straggler=backend" << straggler_id
          << " (" << out.diagnose["straggler_ms_per_token"].get<double>() << " ms/tok)\n";
    human << "overlap_efficiency=" << (has_eff ? std::to_string(eff) : "null") << "\n";
    human << "gpu_smells=" << (smells.empty() ? "none" : "");
    for (size_t i = 0; i < smells.size(); ++i) {
        if (i > 0) {
            human << ",";
        }
        human << smells[i];
    }
    human << "\n";
    out.human_summary = human.str();

    const fs::path human_out = trace_dir / "diagnose-summary.txt";
    std::ofstream human_file(human_out);
    human_file << out.human_summary;
    fprintf(stderr, "%s", out.human_summary.c_str());
    return true;
}