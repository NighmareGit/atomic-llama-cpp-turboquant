#include "pipeline-rpc-validate.h"

#include "common.h"
#include "ggml-backend.h"

#include <cstdio>
#include <vector>

namespace {

std::vector<std::string> split_csv(const std::string & s) {
    return string_split<std::string>(s, ',');
}

int count_tensor_split_devices(const std::string & ts) {
    if (ts.empty()) {
        return 0;
    }
    int count = 0;
    for (const auto & part : split_csv(ts)) {
        try {
            if (std::stof(part) > 0.0f) {
                ++count;
            }
        } catch (...) {
        }
    }
    return count;
}

} // namespace

bool pipeline_rpc_validate(
        const std::string & rpc_endpoints,
        const std::string & tensor_split_str,
        pipeline_rpc_validate_result & out) {
    out.report = nlohmann::ordered_json::object();
    out.report["version"] = 1;
    out.ok                = false;
    out.error.clear();

    const auto endpoints = split_csv(rpc_endpoints);
    if (endpoints.empty()) {
        out.error = "no RPC endpoints";
        out.report["error"] = out.error;
        return false;
    }

    auto * rpc_reg = ggml_backend_reg_by_name("RPC");
    if (!rpc_reg) {
        out.error = "RPC backend not available";
        out.report["error"] = out.error;
        return false;
    }

    using add_rpc_server_fn = ggml_backend_reg_t (*)(const char * endpoint);
    auto * add_fn = (add_rpc_server_fn) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_add_server");
    if (!add_fn) {
        out.error = "ggml_backend_rpc_add_server missing";
        out.report["error"] = out.error;
        return false;
    }

    nlohmann::ordered_json rows = nlohmann::ordered_json::array();
    bool all_ok                 = true;

    for (const auto & endpoint : endpoints) {
        nlohmann::ordered_json row;
        row["endpoint"] = endpoint;
        try {
            const auto reg = add_fn(endpoint.c_str());
            if (!reg) {
                row["ok"]    = false;
                row["error"] = "register failed";
                all_ok       = false;
            } else {
                ggml_backend_register(reg);
                size_t mem_free = 0;
                size_t mem_total = 0;
                const size_t n_devs = ggml_backend_reg_dev_count(reg);
                if (n_devs == 0) {
                    row["ok"]    = false;
                    row["error"] = "no devices on RPC reg";
                    all_ok       = false;
                    rows.push_back(row);
                    continue;
                }
                ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
                ggml_backend_dev_memory(dev, &mem_free, &mem_total);
                const bool reachable = mem_total > 0;
                row["ok"]           = reachable;
                row["mem_free_mb"]  = (int) (mem_free / (1024 * 1024));
                row["mem_total_mb"] = (int) (mem_total / (1024 * 1024));
                if (!reachable) {
                    row["error"] = "HELLO/device memory failed";
                    all_ok       = false;
                }
            }
        } catch (const std::exception & e) {
            row["ok"]    = false;
            row["error"] = e.what();
            all_ok       = false;
        } catch (...) {
            row["ok"]    = false;
            row["error"] = "unknown error";
            all_ok       = false;
        }
        rows.push_back(row);
    }

    const int ts_devices = count_tensor_split_devices(tensor_split_str);
    out.report["endpoints"]       = rows;
    out.report["endpoint_count"]  = (int) endpoints.size();
    out.report["tensor_split"]    = tensor_split_str;
    out.report["ts_device_count"] = ts_devices;
    if (ts_devices > 0 && (int) endpoints.size() > ts_devices) {
        out.report["ts_warning"] = "more RPC endpoints than non-zero tensor-split entries";
        fprintf(stderr, "warning: -rpc has %zu endpoints but -ts lists %d non-zero devices\n",
                endpoints.size(), ts_devices);
    } else if (ts_devices > (int) endpoints.size()) {
        out.report["ts_note"] = "tensor-split has more devices than RPC endpoints (multi-GPU per server)";
    }

    out.ok = all_ok;
    if (!all_ok) {
        out.error = "one or more RPC endpoints unreachable";
        out.report["error"] = out.error;
    }

    fprintf(stderr, "=== rpc validate ===\n%s\n", out.report.dump(2).c_str());
    return out.ok;
}