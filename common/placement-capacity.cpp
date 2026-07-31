#include "placement-capacity.h"

#include "ggml-backend.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>

using json = nlohmann::ordered_json;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

std::string placement_kind_str(placement_backend_kind kind) {
    switch (kind) {
        case PLACEMENT_KIND_LOCAL_GPU:   return "local_gpu";
        case PLACEMENT_KIND_RPC_DEVICE:  return "rpc_device";
        case PLACEMENT_KIND_RPC_TP_UNIT: return "rpc_tp_unit";
    }
    return "unknown";
}

std::string placement_make_local_backend_id(const std::string & dev_name, const std::string & pci) {
    if (!pci.empty()) {
        return "local:pci:" + pci;
    }
    return "local:" + dev_name;
}

std::string placement_make_rpc_backend_id(const std::string & endpoint, int32_t device_index) {
    return "rpc://" + endpoint + "#" + std::to_string(device_index);
}

std::string placement_make_rpc_tp_backend_id(const std::string & endpoint) {
    return "rpc-tp://" + endpoint;
}

bool placement_backend_id_is_tp_unit(const std::string & backend_id) {
    return backend_id.rfind("rpc-tp://", 0) == 0;
}

std::string placement_endpoint_from_backend_id(const std::string & backend_id) {
    if (backend_id.rfind("rpc-tp://", 0) == 0) {
        return backend_id.substr(9);
    }
    if (backend_id.rfind("rpc://", 0) == 0) {
        const size_t hash = backend_id.rfind('#');
        if (hash == std::string::npos) {
            return backend_id.substr(6);
        }
        return backend_id.substr(6, hash - 6);
    }
    return {};
}

bool placement_tp_vram_equal(const std::vector<uint64_t> & total_mib, uint64_t tol_mib) {
    if (total_mib.empty()) {
        return false;
    }
    uint64_t lo = total_mib[0];
    uint64_t hi = total_mib[0];
    for (uint64_t t : total_mib) {
        if (t < lo) {
            lo = t;
        }
        if (t > hi) {
            hi = t;
        }
    }
    return (hi - lo) <= tol_mib;
}

placement_tp_unit_gate_result placement_tp_unit_eval_gates(const placement_tp_unit_gate_input & in) {
    placement_tp_unit_gate_result r;
    if (!in.opt_in) {
        r.eligible = false;
        r.reason = "opt-in required (default Shape B)";
        return r;
    }
    if (in.n_devices != PLACEMENT_TP_UNIT_N_DEVICES) {
        r.eligible = false;
        r.reason = "N=" + std::to_string(in.n_devices) + " not supported (Shape A requires N=2)";
        return r;
    }
    if ((int) in.total_mib.size() != in.n_devices) {
        r.eligible = false;
        r.reason = "total_mib size mismatch vs n_devices";
        return r;
    }
    if (!placement_tp_vram_equal(in.total_mib)) {
        r.eligible = false;
        r.reason = "mixed VRAM (totals not equal within tolerance)";
        return r;
    }
    if (!in.families.empty()) {
        if ((int) in.families.size() != in.n_devices) {
            r.eligible = false;
            r.reason = "families size mismatch vs n_devices";
            return r;
        }
        const std::string & fam0 = in.families[0];
        for (int i = 1; i < in.n_devices; ++i) {
            // empty family is unknown; require match when both non-empty
            if (!fam0.empty() && !in.families[i].empty() && fam0 != in.families[i]) {
                r.eligible = false;
                r.reason = "heterogeneous backend family";
                return r;
            }
        }
    }
    if (!in.specialized_ar_ok) {
        r.eligible = false;
        r.reason = "specialized AllReduce not available (butterfly-only or AR disabled)";
        return r;
    }
    r.eligible = true;
    r.reason.clear();
    return r;
}

bool placement_plan_ids_tp_double_count(
        const std::vector<std::string> & plan_backend_ids,
        std::vector<std::string> & error_messages) {
    std::map<std::string, bool> has_tp;
    std::map<std::string, bool> has_phys;
    for (const auto & id : plan_backend_ids) {
        if (placement_backend_id_is_tp_unit(id)) {
            has_tp[placement_endpoint_from_backend_id(id)] = true;
        } else if (id.rfind("rpc://", 0) == 0) {
            has_phys[placement_endpoint_from_backend_id(id)] = true;
        }
    }
    bool ok = true;
    for (const auto & kv : has_tp) {
        if (has_phys[kv.first]) {
            error_messages.push_back(
                "TP-unit double-count: plan uses both rpc-tp://" + kv.first +
                " and physical rpc://" + kv.first + "#* for the same endpoint");
            ok = false;
        }
    }
    return ok;
}

void placement_inventory_apply_tp_units(
        placement_inventory & inv,
        const placement_tp_unit_options & opts,
        std::vector<std::string> * out_logs) {
    auto log = [&](const std::string & line) {
        if (out_logs) {
            out_logs->push_back(line);
        }
    };

    std::vector<placement_capacity_record> non_rpc;
    std::map<std::string, std::vector<placement_capacity_record>> by_ep;
    for (const auto & r : inv.records) {
        if (r.kind == PLACEMENT_KIND_RPC_DEVICE && !r.endpoint.empty()) {
            by_ep[r.endpoint].push_back(r);
        } else {
            non_rpc.push_back(r);
        }
    }

    std::vector<placement_capacity_record> out = std::move(non_rpc);

    for (auto & kv : by_ep) {
        const std::string & ep = kv.first;
        auto & members = kv.second;
        std::sort(members.begin(), members.end(),
            [](const placement_capacity_record & a, const placement_capacity_record & b) {
                return a.device_index < b.device_index;
            });

        if (members.size() < 2) {
            for (auto & m : members) {
                m.tp_unit_eligible = false;
                m.tp_unit_ineligible_reason = "single-device endpoint";
                out.push_back(m);
            }
            continue;
        }

        placement_tp_unit_gate_input gin;
        gin.n_devices = (int) members.size();
        gin.opt_in = opts.opt_in;
        for (const auto & m : members) {
            gin.total_mib.push_back(m.total_mib);
            gin.families.push_back(m.backend_family);
        }
        auto ar_it = opts.specialized_ar_ok_by_endpoint.find(ep);
        gin.specialized_ar_ok = (ar_it != opts.specialized_ar_ok_by_endpoint.end())
            ? ar_it->second
            : opts.specialized_ar_ok_default;

        // Structural gates (N, VRAM, family) without opt-in/AR
        placement_tp_unit_gate_input g_struct = gin;
        g_struct.opt_in = true;
        g_struct.specialized_ar_ok = true;
        const auto structural = placement_tp_unit_eval_gates(g_struct);

        // Hardware-ready: structural + specialized AR (opt-in still required to collapse)
        placement_tp_unit_gate_input g_hw = gin;
        g_hw.opt_in = true;
        const auto hw_ready = placement_tp_unit_eval_gates(g_hw);

        const auto full = placement_tp_unit_eval_gates(gin);

        if (full.eligible) {
            placement_capacity_record unit;
            unit.schema_version = 1;
            unit.kind = PLACEMENT_KIND_RPC_TP_UNIT;
            unit.backend_id = placement_make_rpc_tp_backend_id(ep);
            unit.endpoint = ep;
            unit.device_index = -1;
            unit.backend_family = members[0].backend_family;
            unit.reported_at = members[0].reported_at;
            unit.reserve_mode = members[0].reserve_mode;
            unit.reserve_model_version = members[0].reserve_model_version;
            unit.caps.multi_device = true;
            unit.caps.telemetry = members[0].caps.telemetry;
            unit.caps.legacy_memory_only = members[0].caps.legacy_memory_only;
            unit.display_name = "tp-unit " + ep;
            unit.tp_unit_eligible = true;

            uint64_t sum_usable = 0;
            uint64_t sum_free = 0;
            uint64_t sum_total = 0;
            for (const auto & m : members) {
                placement_tp_member tm;
                tm.backend_id = m.backend_id;
                tm.device_index = m.device_index;
                tm.total_mib = m.total_mib;
                tm.free_mib = m.free_mib;
                tm.usable_weight_mib = m.usable_weight_mib;
                unit.members.push_back(tm);
                sum_usable += m.usable_weight_mib;
                sum_free += m.free_mib;
                sum_total += m.total_mib;
            }
            unit.total_mib = sum_total;
            unit.free_mib = sum_free;
            const uint64_t pad = opts.tp_workspace_pad_mib;
            unit.usable_weight_mib = (sum_usable > pad) ? (sum_usable - pad) : 0;
            unit.static_pads.process_reserve_mib = pad;
            unit.warnings.push_back(
                "usable_weight_mib = sum(member usable) - tp_workspace_pad (" +
                std::to_string(pad) + " MiB); KV/activations do not scale with N");

            out.push_back(unit);
            log("endpoint " + ep + ": Shape A logical " + unit.backend_id +
                " usable_weight_mib=" + std::to_string(unit.usable_weight_mib));
            continue;
        }

        // Shape B path: keep physical devices, annotate eligibility
        for (auto & m : members) {
            // tp_unit_eligible means "can collapse now" (all gates including opt-in).
            m.tp_unit_eligible = false;
            if (!opts.opt_in) {
                if (hw_ready.eligible) {
                    m.tp_unit_ineligible_reason = "opt-in required (default Shape B)";
                    m.warnings.push_back(
                        "tp_unit available: pass --placement-tp-unit to enable Shape A");
                } else if (!structural.eligible) {
                    m.tp_unit_ineligible_reason = structural.reason;
                } else {
                    m.tp_unit_ineligible_reason = hw_ready.reason;
                }
            } else {
                m.tp_unit_ineligible_reason = full.reason;
                m.warnings.push_back("Shape A refused: " + full.reason);
            }
            out.push_back(m);
        }
        log("endpoint " + ep + ": Shape B (" + full.reason + ")");
    }

    inv.records = std::move(out);
}

uint64_t placement_static_pads_sum_mib(const placement_static_pads & pads) {
    return pads.fragmentation_mib + pads.runtime_overhead_mib + pads.process_reserve_mib;
}

uint64_t placement_table_reserves_mib(const placement_reserve_params & rp, uint64_t total_mib) {
    // table-v1: inspectable floors + coarse KV pad from ctx/parallel.
    // Not a full fit projection (that is reserve_mode=auto, later).
    const uint64_t graph_pad_mib = 256;
    const int32_t  n_ctx = rp.n_ctx > 0 ? rp.n_ctx : 4096;
    const int32_t  n_par = rp.n_parallel > 0 ? rp.n_parallel : 1;
    // Rough KV: 2 * n_layer_est * n_embd_est is unknown without model; use
    // bytes-per-token floor scaled by ctx*parallel only (conservative small pad).
    // 0.25 KiB per token per sequence as a minimal table placeholder (256 B).
    const uint64_t kv_bytes =
        (uint64_t) n_ctx * (uint64_t) n_par * 256ull;
    uint64_t kv_pad_mib = (kv_bytes + 1024ull * 1024ull - 1) / (1024ull * 1024ull);
    if (kv_pad_mib < 64) {
        kv_pad_mib = 64;
    }
    if (rp.flash_attn) {
        // FA reduces KV pressure slightly in table mode
        kv_pad_mib = (kv_pad_mib * 3) / 4;
        if (kv_pad_mib < 48) {
            kv_pad_mib = 48;
        }
    }
    uint64_t small_floor = 0;
    if (total_mib > 0 && total_mib < 10240) {
        // 8 GB class: keep extra headroom so cards stay plannable but not overfilled
        small_floor = 256;
    }
    // kv type scale (F16=2 baseline)
    const int kv_b = std::max(1, (rp.kv_type_k_bytes + rp.kv_type_v_bytes) / 2);
    if (kv_b > 2) {
        kv_pad_mib = kv_pad_mib * (uint64_t) kv_b / 2ull;
    }
    return graph_pad_mib + kv_pad_mib + small_floor;
}

uint64_t placement_usable_weight_mib(
        uint64_t free_mib,
        const placement_static_pads & pads,
        uint64_t client_reserves_mib) {
    const uint64_t sub = placement_static_pads_sum_mib(pads) + client_reserves_mib;
    if (sub >= free_mib) {
        return 0;
    }
    return free_mib - sub;
}

std::string placement_now_iso8601() {
    using clock = std::chrono::system_clock;
    const auto now = clock::now();
    const std::time_t t = clock::to_time_t(now);
    std::tm tm_buf{};
#if defined(_WIN32)
    gmtime_s(&tm_buf, &t);
#else
    gmtime_r(&t, &tm_buf);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_buf);
    return std::string(buf);
}

static void apply_client_reserves(placement_capacity_record & rec, const placement_reserve_params & rp) {
    rec.reserve_mode = (rp.mode == PLACEMENT_RESERVE_MODE_AUTO) ? "auto" : "table";
    // P0: auto falls back to table
    if (rp.mode == PLACEMENT_RESERVE_MODE_AUTO) {
        rec.warnings.push_back("reserve_mode=auto not available; fell back to table");
        rec.reserve_mode = "table";
    }
    rec.reserve_model_version = PLACEMENT_RESERVE_MODEL_TABLE_V1;
    rec.client_reserves_mib = placement_table_reserves_mib(rp, rec.total_mib);
    rec.usable_weight_mib = placement_usable_weight_mib(
        rec.free_mib, rec.static_pads, rec.client_reserves_mib);
}

static placement_capacity_record record_from_raw(
        const placement_raw_device & d,
        const placement_reserve_params & rp,
        const std::string & reported_at) {
    placement_capacity_record rec;
    rec.schema_version = 1;
    rec.kind = d.kind;
    rec.total_mib = d.total_bytes / (1024ull * 1024ull);
    rec.free_mib  = d.free_bytes  / (1024ull * 1024ull);
    rec.static_pads = d.static_pads;
    rec.caps.multi_device = d.multi_device_endpoint;
    rec.caps.telemetry = d.telemetry;
    rec.caps.legacy_memory_only = d.legacy_memory_only;
    rec.reported_at = reported_at.empty() ? placement_now_iso8601() : reported_at;
    rec.display_name = d.description.empty() ? d.name : d.description;
    rec.backend_family = d.backend_family;
    rec.device_id = d.device_id;

    if (d.kind == PLACEMENT_KIND_RPC_DEVICE) {
        rec.endpoint = d.endpoint.empty() ? d.description : d.endpoint;
        rec.device_index = d.device_index >= 0 ? d.device_index : 0;
        rec.backend_id = placement_make_rpc_backend_id(rec.endpoint, rec.device_index);
        if (d.legacy_memory_only) {
            rec.static_pads = {};
            rec.warnings.push_back("legacy GET_DEVICE_MEMORY only; static_pads empty/zero");
        }
    } else {
        rec.backend_id = placement_make_local_backend_id(d.name, d.device_id);
        rec.device_index = d.device_index;
    }

    apply_client_reserves(rec, rp);
    return rec;
}

std::vector<placement_capacity_record> placement_expand_rpc_endpoint_batch(
        const std::string & endpoint,
        const std::vector<placement_raw_device> & devices_on_endpoint,
        const placement_reserve_params & rp,
        const std::string & reported_at) {
    std::vector<placement_capacity_record> out;
    out.reserve(devices_on_endpoint.size());
    const bool multi = devices_on_endpoint.size() > 1;
    for (size_t i = 0; i < devices_on_endpoint.size(); ++i) {
        placement_raw_device d = devices_on_endpoint[i];
        d.kind = PLACEMENT_KIND_RPC_DEVICE;
        if (d.endpoint.empty()) {
            d.endpoint = endpoint;
        }
        if (d.device_index < 0) {
            d.device_index = (int32_t) i;
        }
        d.multi_device_endpoint = multi || d.multi_device_endpoint;
        out.push_back(record_from_raw(d, rp, reported_at));
    }
    return out;
}

static std::string normalize_endpoint(std::string ep) {
    // trim whitespace
    while (!ep.empty() && (ep.front() == ' ' || ep.front() == '\t')) {
        ep.erase(ep.begin());
    }
    while (!ep.empty() && (ep.back() == ' ' || ep.back() == '\t')) {
        ep.pop_back();
    }
    return ep;
}

placement_inventory placement_build_inventory(
        const std::vector<placement_raw_device> & devices,
        const std::vector<std::string> & configured_rpc_endpoints,
        const placement_reserve_params & rp,
        const std::string & reported_at) {
    placement_inventory inv;
    inv.schema_version = 1;
    inv.reserve_mode = (rp.mode == PLACEMENT_RESERVE_MODE_AUTO) ? "table" : "table";
    inv.reserve_model_version = PLACEMENT_RESERVE_MODEL_TABLE_V1;
    inv.rpc_fetch_mode = "legacy_n_get_device_memory";

    const std::string ts = reported_at.empty() ? placement_now_iso8601() : reported_at;

    // Group RPC devices by endpoint for batch expansion semantics
    std::map<std::string, std::vector<placement_raw_device>> rpc_by_ep;
    std::vector<placement_raw_device> local_devs;

    for (const auto & d : devices) {
        if (d.kind == PLACEMENT_KIND_RPC_DEVICE) {
            const std::string ep = normalize_endpoint(d.endpoint.empty() ? d.description : d.endpoint);
            rpc_by_ep[ep].push_back(d);
        } else {
            local_devs.push_back(d);
        }
    }

    for (const auto & d : local_devs) {
        inv.records.push_back(record_from_raw(d, rp, ts));
    }

    for (auto & kv : rpc_by_ep) {
        // stable order by device_index
        std::sort(kv.second.begin(), kv.second.end(),
            [](const placement_raw_device & a, const placement_raw_device & b) {
                return a.device_index < b.device_index;
            });
        auto expanded = placement_expand_rpc_endpoint_batch(kv.first, kv.second, rp, ts);
        inv.records.insert(inv.records.end(), expanded.begin(), expanded.end());
    }

    // Topology completeness vs configured endpoints
    std::map<std::string, bool> seen_ep;
    for (const auto & kv : rpc_by_ep) {
        seen_ep[normalize_endpoint(kv.first)] = true;
    }

    for (const auto & raw_ep : configured_rpc_endpoints) {
        const std::string ep = normalize_endpoint(raw_ep);
        if (ep.empty()) {
            continue;
        }
        if (!seen_ep[ep]) {
            placement_discover_error err;
            err.target = ep;
            err.message = "configured RPC endpoint missing from discover (unreachable or zero devices)";
            inv.discover_errors.push_back(err);
        }
    }

    inv.topology_complete = inv.discover_errors.empty();
    // If configured list empty, complete only when we have at least one record or no RPC expected
    if (configured_rpc_endpoints.empty()) {
        inv.topology_complete = inv.discover_errors.empty();
    }

    return inv;
}

static json pads_to_json(const placement_static_pads & p) {
    return json{
        {"fragmentation_mib", p.fragmentation_mib},
        {"runtime_overhead_mib", p.runtime_overhead_mib},
        {"process_reserve_mib", p.process_reserve_mib},
        {"sum_mib", placement_static_pads_sum_mib(p)},
    };
}

static json caps_to_json(const placement_caps & c) {
    return json{
        {"multi_device", c.multi_device},
        {"telemetry", c.telemetry},
        {"legacy_memory_only", c.legacy_memory_only},
    };
}

static json record_to_json(const placement_capacity_record & r) {
    json j = {
        {"schema_version", r.schema_version},
        {"backend_id", r.backend_id},
        {"kind", placement_kind_str(r.kind)},
        {"total_mib", r.total_mib},
        {"free_mib", r.free_mib},
        {"static_pads", pads_to_json(r.static_pads)},
        {"caps", caps_to_json(r.caps)},
        {"reported_at", r.reported_at},
        {"client_reserves_mib", r.client_reserves_mib},
        {"usable_weight_mib", r.usable_weight_mib},
        {"reserve_mode", r.reserve_mode},
        {"reserve_model_version", r.reserve_model_version},
    };
    if (!r.display_name.empty()) {
        j["display_name"] = r.display_name;
    }
    if (!r.endpoint.empty()) {
        j["endpoint"] = r.endpoint;
    }
    if (r.device_index >= 0) {
        j["device_index"] = r.device_index;
    }
    if (!r.backend_family.empty()) {
        j["backend_family"] = r.backend_family;
    }
    if (!r.device_id.empty()) {
        j["device_id"] = r.device_id;
    }
    if (!r.warnings.empty()) {
        j["warnings"] = r.warnings;
    }
    if (r.kind == PLACEMENT_KIND_RPC_TP_UNIT || r.tp_unit_eligible || !r.tp_unit_ineligible_reason.empty()) {
        j["tp_unit_eligible"] = r.tp_unit_eligible;
        if (!r.tp_unit_ineligible_reason.empty()) {
            j["tp_unit_ineligible_reason"] = r.tp_unit_ineligible_reason;
        }
    }
    if (!r.members.empty()) {
        j["members"] = json::array();
        for (const auto & m : r.members) {
            j["members"].push_back(json{
                {"backend_id", m.backend_id},
                {"device_index", m.device_index},
                {"total_mib", m.total_mib},
                {"free_mib", m.free_mib},
                {"usable_weight_mib", m.usable_weight_mib},
            });
        }
    }
    return j;
}

std::string placement_inventory_to_json(const placement_inventory & inv, int indent) {
    json j;
    j["schema_version"] = inv.schema_version;
    j["topology_complete"] = inv.topology_complete;
    j["reserve_mode"] = inv.reserve_mode;
    j["reserve_model_version"] = inv.reserve_model_version;
    j["rpc_fetch_mode"] = inv.rpc_fetch_mode;
    j["records"] = json::array();
    for (const auto & r : inv.records) {
        j["records"].push_back(record_to_json(r));
    }
    j["discover_errors"] = json::array();
    for (const auto & e : inv.discover_errors) {
        j["discover_errors"].push_back(json{{"target", e.target}, {"message", e.message}});
    }
    return j.dump(indent);
}

bool placement_inventory_write_file(const placement_inventory & inv, const std::string & path) {
    std::ofstream ofs(path);
    if (!ofs) {
        return false;
    }
    ofs << placement_inventory_to_json(inv, 2);
    ofs << "\n";
    return (bool) ofs;
}

bool placement_discover_is_enabled(bool cli_flag, const char * env_value) {
    if (cli_flag) {
        return true;
    }
    if (env_value == nullptr || env_value[0] == '\0') {
        return false;
    }
    // truthy: 1, true, yes, on (case-insensitive first char)
    if (env_value[0] == '0') {
        return false;
    }
    if (env_value[0] == 'f' || env_value[0] == 'F' ||
        env_value[0] == 'n' || env_value[0] == 'N') {
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Live discover (backends + optional RPC dlsym helpers)
// ---------------------------------------------------------------------------

typedef void (*ggml_backend_rpc_get_device_memory_t)(const char * endpoint, uint32_t device, size_t * free, size_t * total);

static ggml_backend_rpc_get_device_memory_t resolve_rpc_get_device_memory(void) {
    ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
    if (!rpc_reg) {
        return nullptr;
    }
    return (ggml_backend_rpc_get_device_memory_t)
        ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_get_device_memory");
}

bool placement_rpc_query_endpoint_legacy(
        const std::string & endpoint,
        std::vector<placement_raw_device> & out_devices,
        std::string & err_msg) {
    out_devices.clear();
    // Prefer walking already-registered devices for this endpoint so device
    // indices match client registration. Also re-query memory via RPC API when
    // available (legacy N x GET_DEVICE_MEMORY).
    auto get_mem = resolve_rpc_get_device_memory();

    // Count registered RPC devices with matching description/endpoint
    std::vector<std::pair<int32_t, ggml_backend_dev_t>> matched;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char * name = ggml_backend_dev_name(dev);
        if (name == nullptr) {
            continue;
        }
        // RPC devices are named RPCn
        if (std::string(name).rfind("RPC", 0) != 0) {
            continue;
        }
        const char * desc = ggml_backend_dev_description(dev);
        if (desc == nullptr || endpoint != desc) {
            continue;
        }
        // device index is order among same endpoint in registration (0..n-1)
        matched.push_back({(int32_t) matched.size(), dev});
    }

    if (matched.empty() && get_mem) {
        // Endpoint not registered: try device 0 only to detect reachability
        size_t free_b = 0, total_b = 0;
        get_mem(endpoint.c_str(), 0, &free_b, &total_b);
        if (free_b == 0 && total_b == 0) {
            err_msg = "RPC endpoint unreachable or returned zero memory";
            return false;
        }
        // Unknown device count without DEVICE_COUNT export; single-device fallback
        placement_raw_device d;
        d.kind = PLACEMENT_KIND_RPC_DEVICE;
        d.name = "RPC?";
        d.description = endpoint;
        d.endpoint = endpoint;
        d.device_index = 0;
        d.free_bytes = free_b;
        d.total_bytes = total_b;
        d.backend_family = "RPC";
        d.legacy_memory_only = true;
        out_devices.push_back(d);
        return true;
    }

    if (matched.empty()) {
        err_msg = "no registered RPC devices for endpoint";
        return false;
    }

    for (size_t i = 0; i < matched.size(); ++i) {
        ggml_backend_dev_t dev = matched[i].second;
        size_t free_b = 0, total_b = 0;
        if (get_mem) {
            get_mem(endpoint.c_str(), (uint32_t) i, &free_b, &total_b);
        } else {
            ggml_backend_dev_memory(dev, &free_b, &total_b);
        }
        placement_raw_device d;
        d.kind = PLACEMENT_KIND_RPC_DEVICE;
        d.name = ggml_backend_dev_name(dev);
        d.description = endpoint;
        d.endpoint = endpoint;
        d.device_index = (int32_t) i;
        d.free_bytes = free_b;
        d.total_bytes = total_b;
        d.backend_family = "RPC";
        d.legacy_memory_only = true; // no batch capacity opcode pads yet
        d.multi_device_endpoint = matched.size() > 1;
        out_devices.push_back(d);
    }
    return true;
}

placement_inventory placement_discover_live(
        const std::vector<std::string> & configured_rpc_endpoints,
        const placement_reserve_params & rp) {
    std::vector<placement_raw_device> raw;

    // Local non-CPU devices
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }
        const char * name = ggml_backend_dev_name(dev);
        if (name && std::string(name).rfind("RPC", 0) == 0) {
            continue; // handled via endpoint batch below
        }
        size_t free_b = 0, total_b = 0;
        ggml_backend_dev_memory(dev, &free_b, &total_b);

        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(dev, &props);

        placement_raw_device d;
        d.kind = PLACEMENT_KIND_LOCAL_GPU;
        d.name = name ? name : "unknown";
        d.description = props.description ? props.description : "";
        d.device_id = props.device_id ? props.device_id : "";
        d.free_bytes = free_b;
        d.total_bytes = total_b;
        d.legacy_memory_only = true; // local static pads helper not yet server-side
        // Local pad table: small fixed process reserve for planner honesty
        d.static_pads.process_reserve_mib = 64;
        d.legacy_memory_only = false;

        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        if (reg) {
            const char * rname = ggml_backend_reg_name(reg);
            d.backend_family = rname ? rname : "";
        }
        raw.push_back(d);
    }

    // RPC: batch per configured endpoint (legacy N x GET_DEVICE_MEMORY)
    placement_inventory inv_partial;
    inv_partial.rpc_fetch_mode = "legacy_n_get_device_memory";
    std::vector<placement_discover_error> rpc_errors;

    std::vector<std::string> endpoints = configured_rpc_endpoints;
    if (endpoints.empty()) {
        // Infer from registered RPC device descriptions
        std::map<std::string, bool> seen;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            const char * name = ggml_backend_dev_name(dev);
            if (!name || std::string(name).rfind("RPC", 0) != 0) {
                continue;
            }
            const char * desc = ggml_backend_dev_description(dev);
            if (desc && !seen[desc]) {
                seen[desc] = true;
                endpoints.push_back(desc);
            }
        }
    }

    for (const auto & ep_raw : endpoints) {
        const std::string ep = normalize_endpoint(ep_raw);
        if (ep.empty()) {
            continue;
        }
        std::vector<placement_raw_device> ep_devs;
        std::string err;
        if (!placement_rpc_query_endpoint_legacy(ep, ep_devs, err)) {
            placement_discover_error de;
            de.target = ep;
            de.message = err;
            rpc_errors.push_back(de);
            continue;
        }
        raw.insert(raw.end(), ep_devs.begin(), ep_devs.end());
    }

    placement_inventory inv = placement_build_inventory(raw, configured_rpc_endpoints, rp);
    inv.rpc_fetch_mode = "legacy_n_get_device_memory";
    // Merge any live query errors not already present
    for (const auto & e : rpc_errors) {
        bool found = false;
        for (const auto & existing : inv.discover_errors) {
            if (existing.target == e.target) {
                found = true;
                break;
            }
        }
        if (!found) {
            inv.discover_errors.push_back(e);
        }
    }
    inv.topology_complete = inv.discover_errors.empty();
    return inv;
}
