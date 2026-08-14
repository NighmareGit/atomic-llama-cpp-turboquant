#include "pipeline-trace-sample.h"
#include "pipeline-jsonl.h"

#include <cstdio>
#include <fstream>
#include <set>
#include <unordered_set>
#include <vector>

namespace {

int write_jsonl(const fs::path & path, const std::vector<nlohmann::ordered_json> & rows) {
    std::ofstream out(path);
    if (!out) {
        return 0;
    }
    for (const auto & row : rows) {
        out << row.dump() << "\n";
    }
    return (int) rows.size();
}

std::vector<nlohmann::ordered_json> sample_rows(
        const std::vector<nlohmann::ordered_json> & rows,
        int every,
        int max_lines,
        const std::unordered_set<std::string> & keep_phase,
        bool keep_blocking) {
    if (rows.empty()) {
        return {};
    }
    every = std::max(1, every);
    std::vector<nlohmann::ordered_json> out;
    std::set<int> seen_decode;
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto & row = rows[i];
        bool keep = false;
        if (row.contains("phase") && row["phase"].is_string()) {
            const auto phase = row["phase"].get<std::string>();
            if (keep_phase.count(phase)) {
                keep = true;
            }
        }
        if (keep_blocking && row.contains("blocking")) {
            if (row["blocking"].is_boolean() && row["blocking"].get<bool>()) {
                keep = true;
            } else if (row["blocking"].is_string() && row["blocking"] == "true") {
                keep = true;
            }
        }
        if (row.contains("decode_id") && row["decode_id"].is_number_integer()) {
            const int decode_id = row["decode_id"].get<int>();
            if (!seen_decode.count(decode_id) && decode_id % every == 0) {
                keep = true;
                seen_decode.insert(decode_id);
            }
        }
        if ((int) i % every == 0) {
            keep = true;
        }
        if (keep) {
            out.push_back(row);
        }
    }
    if (max_lines > 0 && (int) out.size() > max_lines) {
        const int step = std::max(1, (int) out.size() / max_lines);
        std::vector<nlohmann::ordered_json> trimmed;
        for (size_t i = 0; i < out.size() && (int) trimmed.size() < max_lines; i += (size_t) step) {
            trimmed.push_back(out[i]);
        }
        out = std::move(trimmed);
    }
    return out;
}

} // namespace

bool pipeline_trace_sample_run(
        const std::string & telemetry_dir,
        const pipeline_trace_sample_options & opts) {
    const fs::path trace_dir(telemetry_dir);
    const int every     = std::max(1, opts.every);
    const int max_lines = opts.max_lines;

    auto sched_rows = pipeline_load_jsonl(trace_dir / "sched-trace.jsonl");
    auto rpc_rows   = pipeline_load_jsonl(trace_dir / "rpc-trace.jsonl");
    auto pipe_rows  = pipeline_load_jsonl(trace_dir / "pipeline-trace.jsonl");

    const std::unordered_set<std::string> sched_keep = {"split_total"};
    auto sched_sample = sample_rows(sched_rows, every, max_lines, sched_keep, false);
    auto rpc_sample   = sample_rows(rpc_rows, every, max_lines, {}, true);

    nlohmann::ordered_json meta;
    meta["version"]   = 1;
    meta["every"]     = every;
    meta["max_lines"] = max_lines;
    meta["files"]     = nlohmann::ordered_json::object();

    const int sched_n = write_jsonl(trace_dir / "sched-trace.sample.jsonl", sched_sample);
    meta["files"]["sched-trace.jsonl"] = {
        {"source_lines", (int) sched_rows.size()},
        {"sample_lines", sched_n},
    };

    const int rpc_n = write_jsonl(trace_dir / "rpc-trace.sample.jsonl", rpc_sample);
    meta["files"]["rpc-trace.jsonl"] = {
        {"source_lines", (int) rpc_rows.size()},
        {"sample_lines", rpc_n},
    };

    if (!pipe_rows.empty()) {
        const int pipe_n = write_jsonl(trace_dir / "pipeline-trace.sample.jsonl", pipe_rows);
        meta["files"]["pipeline-trace.jsonl"] = {
            {"source_lines", (int) pipe_rows.size()},
            {"sample_lines", pipe_n},
        };
    } else if (fs::exists(trace_dir / "pipeline-trace.jsonl")) {
        fs::copy_file(
            trace_dir / "pipeline-trace.jsonl",
            trace_dir / "pipeline-trace.sample.jsonl",
            fs::copy_options::overwrite_existing);
        meta["files"]["pipeline-trace.jsonl"] = {
            {"source_lines", "copied"},
            {"sample_lines", "copied"},
        };
    }

    std::ofstream meta_out(trace_dir / "sample-meta.json");
    meta_out << meta.dump(2) << "\n";

    fprintf(stderr, "trace sample -> %s\n", trace_dir.string().c_str());
    for (const auto & [name, info] : meta["files"].items()) {
        fprintf(stderr, "  %s: %s -> %s\n",
                name.c_str(),
                info["source_lines"].dump().c_str(),
                info["sample_lines"].dump().c_str());
    }
    return true;
}