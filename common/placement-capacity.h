#pragma once

// Opt-in placement capacity discovery (P0).
// Design: docs/research/placement-capacity-discovery-design.md
//
// Classic argv without placement discover flags/env has no discover side effects.

#include "ggml-backend.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

// reserve_model_version for client table mode
#define PLACEMENT_RESERVE_MODEL_TABLE_V1 "table-v1"

enum placement_reserve_mode {
    PLACEMENT_RESERVE_MODE_TABLE = 0,
    PLACEMENT_RESERVE_MODE_AUTO  = 1, // falls back to table in P0 if projection unavailable
};

enum placement_backend_kind {
    PLACEMENT_KIND_LOCAL_GPU  = 0,
    PLACEMENT_KIND_RPC_DEVICE = 1,
};

struct placement_static_pads {
    uint64_t fragmentation_mib    = 0;
    uint64_t runtime_overhead_mib = 0;
    uint64_t process_reserve_mib  = 0;
};

struct placement_caps {
    bool multi_device         = false;
    bool telemetry            = false;
    bool legacy_memory_only   = false; // free/total only; pads zero + warning
};

// Inputs for client table reserves (no model load required).
struct placement_reserve_params {
    int32_t n_ctx      = 0;
    int32_t n_parallel = 1;
    bool    flash_attn = false;
    // KV type byte sizes (approx); defaults match F16
    int     kv_type_k_bytes = 2;
    int     kv_type_v_bytes = 2;
    placement_reserve_mode mode = PLACEMENT_RESERVE_MODE_TABLE;
};

struct placement_capacity_record {
    int         schema_version = 1;
    std::string backend_id;
    placement_backend_kind kind = PLACEMENT_KIND_LOCAL_GPU;
    uint64_t    total_mib = 0;
    uint64_t    free_mib  = 0;
    placement_static_pads static_pads;
    placement_caps        caps;
    std::string reported_at; // ISO-8601 UTC

    // optional metadata
    std::string display_name;
    std::string endpoint;
    int32_t     device_index = -1;
    std::string backend_family;
    std::string device_id; // PCI when known
    // perf_class omitted when unknown

    // client-only after reserves
    uint64_t    client_reserves_mib = 0;
    uint64_t    usable_weight_mib   = 0;
    std::string reserve_mode;
    std::string reserve_model_version;

    std::vector<std::string> warnings;
};

struct placement_discover_error {
    std::string target;  // endpoint or device label
    std::string message;
};

struct placement_inventory {
    int  schema_version = 1;
    bool topology_complete = false; // false if any configured endpoint missing / error
    std::string reserve_mode;
    std::string reserve_model_version;
    std::vector<placement_capacity_record> records;
    std::vector<placement_discover_error>  discover_errors;
    // Documented wire path for RPC multi-device: "batch_opcode" | "legacy_n_get_device_memory"
    std::string rpc_fetch_mode;
};

// Injected raw device snapshot for unit tests and live discover.
struct placement_raw_device {
    placement_backend_kind kind = PLACEMENT_KIND_LOCAL_GPU;
    std::string name;           // e.g. ROCm0, RPC0
    std::string description;    // product name (local) or host:port (RPC)
    std::string device_id;      // PCI bus id when known
    std::string backend_family; // CUDA / HIP / RPC / ...
    uint64_t free_bytes  = 0;
    uint64_t total_bytes = 0;
    int32_t  device_index = -1; // server device index for RPC
    std::string endpoint;       // host:port for RPC (canonical)
    bool multi_device_endpoint = false;
    bool telemetry = false;
    // When true, static pads unknown (legacy free/total only).
    bool legacy_memory_only = false;
    placement_static_pads static_pads; // server-reported; zero for legacy
};

// --- pure helpers (test seams) ---

std::string placement_kind_str(placement_backend_kind kind);

// local:<devname> or local:pci:<busid> when PCI known (prefer name form always as primary id)
std::string placement_make_local_backend_id(const std::string & dev_name, const std::string & pci = {});

// rpc://host:port#index
std::string placement_make_rpc_backend_id(const std::string & endpoint, int32_t device_index);

uint64_t placement_static_pads_sum_mib(const placement_static_pads & pads);

// table-v1 client reserves (MiB). Independent of free; floors for small VRAM.
uint64_t placement_table_reserves_mib(const placement_reserve_params & rp, uint64_t total_mib);

// usable = max(0, free - pads - client_reserves); never negative.
uint64_t placement_usable_weight_mib(uint64_t free_mib, const placement_static_pads & pads, uint64_t client_reserves_mib);

// Expand multi-device endpoint device list into N records (batch body already fetched).
// Each entry is one device on the same endpoint.
std::vector<placement_capacity_record> placement_expand_rpc_endpoint_batch(
    const std::string & endpoint,
    const std::vector<placement_raw_device> & devices_on_endpoint,
    const placement_reserve_params & rp,
    const std::string & reported_at);

// Build inventory from raw snapshots + configured RPC list (for partial-fail).
// configured_rpc_endpoints: host:port strings from --rpc (may include unreachable).
placement_inventory placement_build_inventory(
    const std::vector<placement_raw_device> & devices,
    const std::vector<std::string> & configured_rpc_endpoints,
    const placement_reserve_params & rp,
    const std::string & reported_at = {});

// ISO-8601 UTC now (or fixed clock for tests when non-empty override).
std::string placement_now_iso8601();

// JSON dump (nlohmann ordered).
std::string placement_inventory_to_json(const placement_inventory & inv, int indent = 2);

// Write inventory to path; returns false on I/O error.
bool placement_inventory_write_file(const placement_inventory & inv, const std::string & path);

// Live discover: walk registered backends (local GPU + RPC).
// configured_rpc_endpoints used for topology completeness.
// RPC multi-device: groups by endpoint, fetches free/total per device via
// legacy N x GET_DEVICE_MEMORY (documented fallback; batch opcode not required for P0).
placement_inventory placement_discover_live(
    const std::vector<std::string> & configured_rpc_endpoints,
    const placement_reserve_params & rp);

// Gate: true only when CLI/env discover is enabled.
bool placement_discover_is_enabled(bool cli_flag, const char * env_value);

// Collect free/total for one RPC endpoint into raw devices (legacy N x memory query).
// Returns empty + error message on total failure (connect / zero devices).
bool placement_rpc_query_endpoint_legacy(
    const std::string & endpoint,
    std::vector<placement_raw_device> & out_devices,
    std::string & err_msg);
