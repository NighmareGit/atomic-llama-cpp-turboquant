#pragma once

#include <nlohmann/json.hpp>

#include <string>

struct pipeline_rpc_validate_result {
    bool                    ok = false;
    nlohmann::ordered_json  report;
    std::string             error;
};

// Load RPC backend plugin only (skip CUDA/ROCm init). For --validate-rpc preflight.
bool pipeline_rpc_validate_prepare();

// Probe each RPC endpoint (HELLO + device memory). No model load.
bool pipeline_rpc_validate(
        const std::string & rpc_endpoints,
        const std::string & tensor_split_str,
        pipeline_rpc_validate_result & out);