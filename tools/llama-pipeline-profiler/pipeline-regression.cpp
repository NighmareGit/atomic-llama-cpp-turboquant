#include "pipeline-regression.h"
#include "pipeline-jsonl.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <regex>
#include <sstream>

namespace fs = std::filesystem;

namespace {

std::map<std::string, std::string> read_env_txt(const fs::path & path) {
    std::map<std::string, std::string> out;
    if (!fs::exists(path)) {
        return out;
    }
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        const auto pos = line.find('=');
        if (pos == std::string::npos) {
            continue;
        }
        out[line.substr(0, pos)] = line.substr(pos + 1);
    }
    return out;
}

double read_result_g(const fs::path & out_dir) {
    const fs::path result_path = out_dir / "result.jsonl";
    if (fs::exists(result_path)) {
        std::ifstream in(result_path);
        std::string line;
        if (std::getline(in, line) && !line.empty()) {
            try {
                const auto row = nlohmann::ordered_json::parse(line);
                const double g = pipeline_result_g_tps(row);
                if (g > 0) {
                    return g;
                }
            } catch (...) {
            }
        }
    }
    for (const char * name : {"result.meta", "bench.result", "profile-summary.txt"}) {
        const fs::path meta_path = out_dir / name;
        if (!fs::exists(meta_path)) {
            continue;
        }
        std::ifstream in(meta_path);
        std::ostringstream ss;
        ss << in.rdbuf();
        const std::string text = ss.str();
        std::vector<double> vals;
        std::regex re(R"(G=([\d.]+))");
        for (std::sregex_iterator it(text.begin(), text.end(), re), end; it != end; ++it) {
            vals.push_back(std::stod((*it)[1].str()));
        }
        if (!vals.empty()) {
            return *std::max_element(vals.begin(), vals.end());
        }
    }
    return 0.0;
}

std::string utc_now() {
    const auto now = std::chrono::system_clock::now();
    const auto t   = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    std::ostringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return ss.str();
}

} // namespace

bool pipeline_regression_append(
        const std::string & telemetry_dir,
        const pipeline_regression_options & opts) {
    const fs::path trace_dir(telemetry_dir);
    const fs::path diagnose_path = trace_dir / "diagnose.json";
    if (!fs::exists(diagnose_path)) {
        fprintf(stderr, "regression: missing %s\n", diagnose_path.string().c_str());
        return false;
    }

    fs::path out_dir = opts.out_dir.empty() ? trace_dir.parent_path() : fs::path(opts.out_dir);
    std::ifstream diag_in(diagnose_path);
    nlohmann::ordered_json diagnose;
    try {
        diagnose = nlohmann::ordered_json::parse(diag_in);
    } catch (...) {
        fprintf(stderr, "regression: failed to parse %s\n", diagnose_path.string().c_str());
        return false;
    }

    const auto env = read_env_txt(out_dir / "env.txt");
    double g_tps   = read_result_g(out_dir);
    if (g_tps <= 0 && diagnose.contains("G_tps") && !diagnose["G_tps"].is_null()) {
        g_tps = diagnose["G_tps"].get<double>();
    }

    int plus = opts.plus;
    if (env.count("GGML_PIPELINE_PLUS")) {
        plus = std::stoi(env.at("GGML_PIPELINE_PLUS"));
    }

    nlohmann::ordered_json record;
    record["version"]     = 1;
    record["ts_utc"]      = utc_now();
    record["label"]       = opts.label;
    record["git_sha"]     = env.count("GIT_SHA") ? env.at("GIT_SHA") : opts.git_sha;
    record["client_kind"] = env.count("client_kind") ? env.at("client_kind") : opts.client_kind;
    record["trace_dir"]   = trace_dir.string();
    record["out_dir"]     = out_dir.string();
    record["rpc"]  = opts.rpc_endpoints.empty() && env.count("RPC") ? env.at("RPC") : opts.rpc_endpoints;
    record["mode"] = opts.mode.empty() && env.count("MODE") ? env.at("MODE") : opts.mode;
    record["GGML_PIPELINE_PLUS"] = plus;
    record["G_tps"]       = g_tps > 0 ? nlohmann::ordered_json(g_tps) : nlohmann::ordered_json(nullptr);

    for (const char * key : {
            "gate_s5", "gate_b6", "overlap_pct", "assembly_overlap_count",
            "stall_ratio", "straggler_backend", "straggler_ms_per_token",
            "drain_flush_ms", "blocking_ms", "rpc_rtt_per_token",
            "overlap_efficiency", "gen_tokens_est",
        }) {
        if (diagnose.contains(key)) {
            record[key] = diagnose[key];
        }
    }
    record["gpu_smell_flags"] = diagnose.value("gpu_smell_flags", nlohmann::ordered_json::array());
    record["diagnose_version"] = diagnose.value("version", 1);

    const fs::path regression_file(opts.regression_file);
    fs::create_directories(regression_file.parent_path());
    std::ofstream out(regression_file, std::ios::app);
    if (!out) {
        return false;
    }
    out << record.dump() << "\n";

    const std::string g_str   = record["G_tps"].dump();
    const std::string ov_str  = record.contains("overlap_pct") ? record["overlap_pct"].dump() : "null";
    const std::string b6_str  = record.contains("gate_b6") ? record["gate_b6"].get<std::string>() : "?";
    fprintf(stderr, "regression append -> %s\n", regression_file.string().c_str());
    fprintf(stderr, "  label=%s G_tps=%s overlap_pct=%s gate_b6=%s\n",
            opts.label.c_str(), g_str.c_str(), ov_str.c_str(), b6_str.c_str());
    return true;
}