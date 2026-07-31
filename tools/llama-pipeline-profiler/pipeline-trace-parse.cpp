#include "pipeline-trace-parse.h"
#include "pipeline-jsonl.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace {

double round_ms(int64_t us_sum) {
    return std::round(us_sum / 10.0) / 100.0;
}

double round_avg(const std::vector<int64_t> & us_values) {
    if (us_values.empty()) {
        return 0.0;
    }
    int64_t sum = 0;
    for (int64_t v : us_values) {
        sum += v;
    }
    return std::round((double) sum / (double) us_values.size());
}

const char * cmd_name(int cmd) {
    static const std::unordered_map<int, const char *> names = {
        {0, "ALLOC_BUFFER"}, {1, "GET_ALIGNMENT"}, {2, "GET_MAX_SIZE"},
        {3, "BUFFER_GET_BASE"}, {4, "FREE_BUFFER"}, {5, "BUFFER_CLEAR"},
        {6, "SET_TENSOR"}, {7, "SET_TENSOR_HASH"}, {8, "GET_TENSOR"},
        {9, "COPY_TENSOR"}, {10, "GRAPH_COMPUTE"}, {11, "GET_DEVICE_MEMORY"},
        {12, "INIT_TENSOR"}, {13, "GET_ALLOC_SIZE"}, {14, "HELLO"},
        {15, "DEVICE_COUNT"}, {16, "GRAPH_RECOMPUTE"}, {17, "SET_TENSOR_BATCH"},
        {18, "EVENT_RECORD"}, {19, "COPY_TENSOR_PEER"},
    };
    auto it = names.find(cmd);
    if (it != names.end()) {
        return it->second;
    }
    return nullptr;
}

bool is_true(const nlohmann::ordered_json & v) {
    if (v.is_boolean()) {
        return v.get<bool>();
    }
    if (v.is_string()) {
        const auto s = v.get<std::string>();
        return s == "true" || s == "True" || s == "1";
    }
    return false;
}

int64_t json_int(const nlohmann::ordered_json & row, const char * key, int64_t def = 0) {
    if (!row.contains(key) || row[key].is_null()) {
        return def;
    }
    if (row[key].is_number_integer()) {
        return row[key].get<int64_t>();
    }
    if (row[key].is_number_float()) {
        return (int64_t) row[key].get<double>();
    }
    return def;
}

std::string json_str(const nlohmann::ordered_json & row, const char * key, const std::string & def = "") {
    if (!row.contains(key) || row[key].is_null()) {
        return def;
    }
    if (row[key].is_string()) {
        return row[key].get<std::string>();
    }
    if (row[key].is_number()) {
        return std::to_string(row[key].get<int64_t>());
    }
    return def;
}

} // namespace

bool pipeline_trace_parse(const std::string & telemetry_dir) {
    const fs::path dir(telemetry_dir);
    const fs::path rpc_file   = dir / "rpc-trace.jsonl";
    const fs::path sched_file = dir / "sched-trace.jsonl";
    const fs::path out_file   = dir / "trace-summary.txt";

    std::ostringstream lines;
    lines << "=== trace summary ===\n";
    lines << "dir=" << dir.string() << "\n";

    auto rpc_rows = pipeline_load_jsonl(rpc_file);
    if (fs::exists(rpc_file)) {
        lines << "\n[rpc] events=" << rpc_rows.size() << "\n";
        if (!rpc_rows.empty()) {
            std::map<int, std::vector<nlohmann::ordered_json>> by_cmd;
            for (const auto & row : rpc_rows) {
                if (!row.contains("cmd") || row["cmd"].is_null()) {
                    continue;
                }
                by_cmd[json_int(row, "cmd")].push_back(row);
            }
            for (const auto & [cmd, group] : by_cmd) {
                std::vector<int64_t> us;
                us.reserve(group.size());
                for (const auto & row : group) {
                    us.push_back(json_int(row, "elapsed_us"));
                }
                const char * name = cmd_name(cmd);
                std::string label = name ? name : ("cmd" + std::to_string(cmd));
                int64_t sum = 0;
                for (int64_t v : us) {
                    sum += v;
                }
                lines << "  " << label << " count=" << group.size()
                      << " total_ms=" << round_ms(sum)
                      << " avg_us=" << round_avg(us) << "\n";
            }

            std::vector<nlohmann::ordered_json> blocking;
            for (const auto & row : rpc_rows) {
                if (row.contains("blocking") && is_true(row["blocking"])) {
                    blocking.push_back(row);
                }
            }
            int64_t bus_sum = 0;
            for (const auto & row : blocking) {
                bus_sum += json_int(row, "elapsed_us");
            }
            lines << "  blocking_events=" << blocking.size()
                  << " blocking_ms=" << round_ms(bus_sum) << "\n";

            std::vector<nlohmann::ordered_json> drain;
            for (const auto & row : rpc_rows) {
                if (!row.contains("fn")) {
                    continue;
                }
                const auto fn = json_str(row, "fn");
                std::string lower = fn;
                std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                if (lower.find("drain") != std::string::npos || lower.find("flush") != std::string::npos) {
                    drain.push_back(row);
                }
            }
            if (!drain.empty()) {
                int64_t dus = 0;
                for (const auto & row : drain) {
                    dus += json_int(row, "elapsed_us");
                }
                lines << "  drain_flush_ms=" << round_ms(dus) << "\n";
            }

            std::vector<nlohmann::ordered_json> drain_copy;
            for (const auto & row : rpc_rows) {
                if (json_str(row, "phase") == "drain_copy") {
                    drain_copy.push_back(row);
                }
            }
            if (!drain_copy.empty()) {
                int64_t dcus = 0;
                for (const auto & row : drain_copy) {
                    dcus += json_int(row, "elapsed_us");
                }
                lines << "  drain_copy_count=" << drain_copy.size()
                      << " drain_copy_ms=" << round_ms(dcus) << "\n";
            }

            for (const auto & row : rpc_rows) {
                if (json_str(row, "phase") == "hello") {
                    lines << "  hello endpoint=" << json_str(row, "endpoint", "?")
                          << " minor=" << json_str(row, "minor")
                          << " peer_copy=" << json_str(row, "peer_copy") << "\n";
                }
            }

            std::vector<nlohmann::ordered_json> copy_issue;
            for (const auto & row : rpc_rows) {
                if (json_str(row, "phase") == "copy_issue") {
                    copy_issue.push_back(row);
                }
            }
            if (!copy_issue.empty()) {
                int defer = 0;
                int peer  = 0;
                for (const auto & row : copy_issue) {
                    if (row.contains("defer") && is_true(row["defer"])) {
                        ++defer;
                    }
                    if (row.contains("peer_copy") && is_true(row["peer_copy"])) {
                        ++peer;
                    }
                }
                lines << "  copy_issue_count=" << copy_issue.size()
                      << " defer_count=" << defer
                      << " peer_copy_count=" << peer << "\n";
            }
        }
    } else {
        lines << "\n[rpc] missing " << rpc_file.string() << "\n";
    }

    auto sched_rows = pipeline_load_jsonl(sched_file);
    if (fs::exists(sched_file)) {
        lines << "\n[sched] events=" << sched_rows.size() << "\n";
        if (!sched_rows.empty()) {
            std::vector<nlohmann::ordered_json> splits;
            for (const auto & row : sched_rows) {
                if (json_str(row, "phase") == "split_total") {
                    splits.push_back(row);
                }
            }
            lines << "  split_total_count=" << splits.size() << "\n";
            if (!splits.empty()) {
                std::vector<int64_t> sus;
                sus.reserve(splits.size());
                for (const auto & row : splits) {
                    sus.push_back(json_int(row, "elapsed_us"));
                }
                int64_t sus_sum = 0;
                for (int64_t v : sus) {
                    sus_sum += v;
                }
                lines << "  split_total_ms_sum=" << round_ms(sus_sum)
                      << " avg_us=" << round_avg(sus) << "\n";

                std::map<std::string, std::vector<nlohmann::ordered_json>> by_backend;
                std::vector<std::string> backend_order;
                for (const auto & row : splits) {
                    const auto backend = json_str(row, "backend");
                    if (by_backend.find(backend) == by_backend.end()) {
                        backend_order.push_back(backend);
                    }
                    by_backend[backend].push_back(row);
                }
                for (const auto & backend : backend_order) {
                    const auto & group = by_backend[backend];
                    int64_t bus = 0;
                    for (const auto & row : group) {
                        bus += json_int(row, "elapsed_us");
                    }
                    lines << "    backend" << backend << " splits=" << group.size()
                          << " ms=" << round_ms(bus) << "\n";
                }
            }

            for (const char * phase : {"input_wait_copy", "graph_compute_async", "event_record"}) {
                std::vector<nlohmann::ordered_json> phase_rows;
                for (const auto & row : sched_rows) {
                    if (json_str(row, "phase") == phase) {
                        phase_rows.push_back(row);
                    }
                }
                if (!phase_rows.empty()) {
                    int64_t pus = 0;
                    for (const auto & row : phase_rows) {
                        pus += json_int(row, "elapsed_us");
                    }
                    lines << "  " << phase << "_ms=" << round_ms(pus) << "\n";
                }
            }

            if (splits.size() >= 2) {
                int overlap = 0;
                int pairs   = 0;
                for (size_t i = 0; i < splits.size(); ++i) {
                    const int64_t a_start = json_int(splits[i], "ts_us");
                    const int64_t a_end   = a_start + json_int(splits[i], "elapsed_us");
                    const auto a_backend  = json_str(splits[i], "backend");
                    for (size_t j = i + 1; j < splits.size(); ++j) {
                        const auto b_backend = json_str(splits[j], "backend");
                        if (a_backend == b_backend) {
                            continue;
                        }
                        const int64_t b_start = json_int(splits[j], "ts_us");
                        if (a_start <= b_start && b_start < a_end) {
                            ++overlap;
                        }
                        ++pairs;
                    }
                }
                const double pct = pairs > 0 ? std::round(1000.0 * overlap / pairs) / 10.0 : 0.0;
                lines << "  assembly_overlap_count=" << overlap
                      << " pairs=" << pairs
                      << " overlap_pct=" << pct << "\n";
            }

            std::map<std::string, std::vector<nlohmann::ordered_json>> by_copy;
            std::vector<std::string> copy_order;
            for (const auto & row : splits) {
                const auto copy = json_str(row, "copy");
                if (by_copy.find(copy) == by_copy.end()) {
                    copy_order.push_back(copy);
                }
                by_copy[copy].push_back(row);
            }
            if (by_copy.size() > 1) {
                for (const auto & copy : copy_order) {
                    lines << "    copy" << copy << " splits=" << by_copy[copy].size() << "\n";
                }
            }
        }
    } else {
        lines << "\n[sched] missing " << sched_file.string() << "\n";
    }

    const std::string text = lines.str();
    std::ofstream out(out_file);
    if (!out) {
        return false;
    }
    out << text;
    fprintf(stderr, "%s", text.c_str());
    fprintf(stderr, "summary -> %s\n", out_file.string().c_str());
    return true;
}