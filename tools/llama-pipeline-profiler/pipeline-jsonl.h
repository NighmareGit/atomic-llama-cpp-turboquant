#pragma once

#include <nlohmann/json.hpp>

#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

std::vector<nlohmann::ordered_json> pipeline_load_jsonl(const fs::path & path);

// Decode throughput from a profiler result.jsonl row.
// Prefers explicit G_tps, else wall n_gen_tokens/elapsed_ns (topology-agnostic).
double pipeline_result_g_tps(const nlohmann::ordered_json & row);