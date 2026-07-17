#pragma once

// Placement plan IR parse + validate + apply preparation (P1 layer ranges).
// Design: docs/research/placement-plan-ir-apply-design.md
// Issue 09: empty overrides only; non-empty => not implemented error.

#include "placement-capacity.h"

#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

struct placement_plan_assignment {
    int32_t     layer_start = 0; // inclusive
    int32_t     layer_end   = 0; // exclusive
    std::string backend_id;      // "cpu" or canonical backend_id
};

struct placement_plan_override {
    std::string match;
    std::string backend_id;
};

struct placement_plan_heat {
    std::string status = "none"; // full | partial | none
    std::string task;            // tg | pp | empty
    int         schema_version = 1;
};

struct placement_plan {
    int         schema_version = 1;
    std::string created_at;
    std::string model_path;
    int32_t     n_layer = 0; // required for validate; may come from plan or caller
    std::string split_mode = "layer"; // layer | tensor
    std::string capacity_snapshot_at;
    std::string reserve_mode = "table";
    std::string reserve_model_version = PLACEMENT_RESERVE_MODEL_TABLE_V1;
    std::vector<placement_plan_assignment> assignments;
    std::vector<placement_plan_override>   overrides;
    placement_plan_heat heat;
    // optional backends snapshot from generator (usable_weight_mib for budget checks)
    struct backend_snapshot {
        std::string backend_id;
        uint64_t    usable_weight_mib = 0;
        uint64_t    free_mib = 0;
        uint64_t    total_mib = 0;
    };
    std::vector<backend_snapshot> backends;
};

struct placement_plan_error {
    std::string message;
};

// Per-layer map after expand: length n_layer; "cpu" or backend_id
using placement_layer_map = std::vector<std::string>;

// Parse plan JSON text. Returns false + errors on hard parse failure.
bool placement_plan_parse_json(const std::string & json_text, placement_plan & out, std::vector<placement_plan_error> & errors);

bool placement_plan_load_file(const std::string & path, placement_plan & out, std::vector<placement_plan_error> & errors);

// Validate structure + full partition of [0, n_layer). Uses plan.n_layer or n_layer_override if > 0.
// Rejects non-empty overrides (issue 09 / slice 10).
// Rejects illegal tensor split_mode for multi-backend mixed topologies (caller passes mixed flag).
bool placement_plan_validate(
    const placement_plan & plan,
    int32_t n_layer,
    bool topology_has_mixed_rpc_and_local,
    std::vector<placement_plan_error> & errors);

// Expand assignments to per-layer backend_id strings (length n_layer). Requires successful validate.
bool placement_plan_expand_layers(
    const placement_plan & plan,
    int32_t n_layer,
    placement_layer_map & out_layers,
    std::vector<placement_plan_error> & errors);

// Resolve backend_id -> device using discover inventory.
// "cpu" resolves to nullptr device (CPU path).
// Fills missing_ids for any backend_id not present (except cpu).
// Also matches local:NAME when inventory has local:pci:... for same live device name via aliases map.
struct placement_backend_resolve {
    std::string         backend_id;
    ggml_backend_dev_t  dev = nullptr; // null => CPU
};

// Build map of inventory backend_id -> dev by walking registered devices (live).
// Uses placement_make_* ids consistent with discover.
std::vector<placement_backend_resolve> placement_resolve_backends_live(
    const placement_inventory & inv);

// Match plan backend_id against inventory records. Returns true if all non-cpu ids found.
bool placement_plan_match_backends(
    const placement_plan & plan,
    const placement_inventory & inv,
    std::vector<std::string> & missing_ids);

// Apply preparation result: devices list (null-terminated storage) + per-layer devices.
struct placement_apply_result {
    std::vector<ggml_backend_dev_t> devices;       // unique GPUs used, null-terminated later
    std::vector<ggml_backend_dev_t> layer_devices; // length n_layer; nullptr = CPU
    placement_layer_map             layer_backend_ids;
    std::string                     debug_dump;    // layer->backend assignment text
};

// Build apply result from plan + live inventory. Fail-loud errors listed.
// usable_check: if true, refuse backends with usable_weight_mib==0 that have layers.
bool placement_plan_prepare_apply(
    const placement_plan & plan,
    int32_t n_layer,
    const placement_inventory & inv,
    bool topology_has_mixed_rpc_and_local,
    placement_apply_result & out,
    std::vector<placement_plan_error> & errors);

// Dump layer map as multi-line string for logs / determinism checks.
std::string placement_layer_map_to_string(const placement_layer_map & layers);
