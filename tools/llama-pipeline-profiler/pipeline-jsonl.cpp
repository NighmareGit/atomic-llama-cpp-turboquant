#include "pipeline-jsonl.h"

#include <fstream>

std::vector<nlohmann::ordered_json> pipeline_load_jsonl(const fs::path & path) {
    std::vector<nlohmann::ordered_json> rows;
    if (!fs::exists(path)) {
        return rows;
    }
    std::ifstream in(path, std::ios::binary);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line[0] == '\xef') {
            if (line.size() >= 3 &&
                (unsigned char) line[0] == 0xef &&
                (unsigned char) line[1] == 0xbb &&
                (unsigned char) line[2] == 0xbf) {
                line.erase(0, 3);
            }
        }
        if (line.empty()) {
            continue;
        }
        try {
            auto row = nlohmann::ordered_json::parse(line);
            if (row.is_object()) {
                rows.push_back(std::move(row));
            }
        } catch (const nlohmann::json::exception &) {
            continue;
        }
    }
    return rows;
}

double pipeline_result_g_tps(const nlohmann::ordered_json & row) {
    if (row.contains("G_tps") && !row["G_tps"].is_null()) {
        return row["G_tps"].get<double>();
    }
    if (row.contains("avg_ts") && !row["avg_ts"].is_null()) {
        return row["avg_ts"].get<double>();
    }
    const uint64_t elapsed_ns = row.contains("elapsed_ns") ? row["elapsed_ns"].get<uint64_t>() : 0;
    if (elapsed_ns == 0) {
        return 0.0;
    }
    int n_gen = 0;
    if (row.contains("n_gen_tokens") && !row["n_gen_tokens"].is_null()) {
        n_gen = row["n_gen_tokens"].get<int>();
    }
    if (n_gen > 0) {
        return 1e9 * n_gen / (double) elapsed_ns;
    }
    if (row.contains("n_eval") && !row["n_eval"].is_null()) {
        const int n_eval = row["n_eval"].get<int>();
        if (n_eval > 0) {
            return 1e9 * n_eval / (double) elapsed_ns;
        }
    }
    return 0.0;
}