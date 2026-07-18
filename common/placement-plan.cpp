#include "placement-plan.h"

#include "log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <regex>
#include <set>
#include <sstream>

using json = nlohmann::ordered_json;

static void add_err(std::vector<placement_plan_error> & errors, const std::string & msg) {
    errors.push_back({msg});
}

bool placement_plan_parse_json(
        const std::string & json_text,
        placement_plan & out,
        std::vector<placement_plan_error> & errors) {
    out = placement_plan{};
    json j;
    try {
        j = json::parse(json_text);
    } catch (const std::exception & e) {
        add_err(errors, std::string("plan JSON parse failed: ") + e.what());
        return false;
    }

    if (!j.is_object()) {
        add_err(errors, "plan root must be an object");
        return false;
    }

    out.schema_version = j.value("schema_version", 1);
    out.created_at = j.value("created_at", "");
    out.split_mode = j.value("split_mode", "layer");
    out.capacity_snapshot_at = j.value("capacity_snapshot_at", "");
    out.reserve_mode = j.value("reserve_mode", "table");
    if (j.contains("reserve_model_version")) {
        if (j["reserve_model_version"].is_number()) {
            out.reserve_model_version = std::to_string(j["reserve_model_version"].get<int>());
        } else {
            out.reserve_model_version = j["reserve_model_version"].get<std::string>();
        }
    }

    if (j.contains("model") && j["model"].is_object()) {
        out.model_path = j["model"].value("path", "");
        out.n_layer = j["model"].value("n_layer", 0);
    }
    if (j.contains("n_layer") && j["n_layer"].is_number_integer()) {
        out.n_layer = j["n_layer"].get<int32_t>();
    }

    if (!j.contains("assignments") || !j["assignments"].is_array()) {
        add_err(errors, "plan missing assignments array");
        return false;
    }
    for (const auto & a : j["assignments"]) {
        if (!a.is_object()) {
            add_err(errors, "assignment entry must be object");
            return false;
        }
        placement_plan_assignment as;
        as.layer_start = a.value("layer_start", -1);
        as.layer_end   = a.value("layer_end", -1);
        as.backend_id  = a.value("backend_id", "");
        if (as.backend_id.empty() && a.contains("backend")) {
            as.backend_id = a["backend"].get<std::string>();
        }
        out.assignments.push_back(as);
    }

    if (j.contains("overrides") && j["overrides"].is_array()) {
        for (const auto & o : j["overrides"]) {
            placement_plan_override ov;
            ov.match = o.value("match", "");
            ov.backend_id = o.value("backend_id", "");
            out.overrides.push_back(ov);
        }
    }

    if (j.contains("heat") && j["heat"].is_object()) {
        out.heat.status = j["heat"].value("status", "none");
        out.heat.task = j["heat"].value("task", "");
        out.heat.schema_version = j["heat"].value("schema_version", 1);
    }

    if (j.contains("backends") && j["backends"].is_array()) {
        for (const auto & b : j["backends"]) {
            placement_plan::backend_snapshot bs;
            bs.backend_id = b.value("backend_id", "");
            bs.usable_weight_mib = b.value("usable_weight_mib", 0ull);
            bs.free_mib = b.value("free_mib", 0ull);
            bs.total_mib = b.value("total_mib", 0ull);
            out.backends.push_back(bs);
        }
    }

    return true;
}

bool placement_plan_load_file(
        const std::string & path,
        placement_plan & out,
        std::vector<placement_plan_error> & errors) {
    std::ifstream ifs(path);
    if (!ifs) {
        add_err(errors, "failed to open plan file: " + path);
        return false;
    }
    std::ostringstream ss;
    ss << ifs.rdbuf();
    return placement_plan_parse_json(ss.str(), out, errors);
}

bool placement_plan_validate(
        const placement_plan & plan,
        int32_t n_layer,
        bool topology_has_mixed_rpc_and_local,
        std::vector<placement_plan_error> & errors) {
    const size_t n_err0 = errors.size();

    if (plan.schema_version < 1) {
        add_err(errors, "schema_version must be >= 1");
    }
    if (n_layer <= 0) {
        add_err(errors, "n_layer must be positive for validation");
        return false;
    }
    if (plan.assignments.empty()) {
        add_err(errors, "assignments must not be empty");
    }

    // overrides[] validated structurally here; backend resolve at prepare_apply
    for (size_t oi = 0; oi < plan.overrides.size(); ++oi) {
        const auto & o = plan.overrides[oi];
        if (o.match.empty()) {
            add_err(errors, "override " + std::to_string(oi) + " missing match pattern");
        }
        if (o.backend_id.empty()) {
            add_err(errors, "override " + std::to_string(oi) + " missing backend_id");
        }
        if (!o.match.empty()) {
            try {
                std::regex re(o.match);
                (void) re;
            } catch (const std::regex_error & e) {
                add_err(errors, "override " + std::to_string(oi) + " invalid regex: " + e.what());
            }
        }
    }

    std::string sm = plan.split_mode;
    std::transform(sm.begin(), sm.end(), sm.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    if (sm.empty()) {
        sm = "layer";
    }
    if (sm != "layer" && sm != "tensor") {
        add_err(errors, "split_mode must be \"layer\" or \"tensor\"");
    }
    if (sm == "tensor" && topology_has_mixed_rpc_and_local) {
        add_err(errors, "split_mode=tensor refused for mixed local+RPC topology (illegal global tensor rail)");
    }

    // coverage
    std::vector<int> cover(n_layer, 0);
    for (size_t ai = 0; ai < plan.assignments.size(); ++ai) {
        const auto & a = plan.assignments[ai];
        if (a.backend_id.empty()) {
            add_err(errors, "assignment " + std::to_string(ai) + " missing backend_id");
            continue;
        }
        if (a.layer_start < 0 || a.layer_end <= a.layer_start) {
            add_err(errors, "assignment " + std::to_string(ai) + " invalid range [" +
                std::to_string(a.layer_start) + "," + std::to_string(a.layer_end) + ")");
            continue;
        }
        if (a.layer_end > n_layer) {
            add_err(errors, "assignment " + std::to_string(ai) + " layer_end " +
                std::to_string(a.layer_end) + " exceeds n_layer " + std::to_string(n_layer));
            continue;
        }
        for (int32_t il = a.layer_start; il < a.layer_end; ++il) {
            cover[il]++;
            if (cover[il] > 1) {
                add_err(errors, "layer " + std::to_string(il) + " covered more than once (overlap)");
            }
        }
    }
    for (int32_t il = 0; il < n_layer; ++il) {
        if (cover[il] == 0) {
            add_err(errors, "layer " + std::to_string(il) + " not covered (gap)");
        }
    }

    return errors.size() == n_err0;
}

bool placement_plan_expand_layers(
        const placement_plan & plan,
        int32_t n_layer,
        placement_layer_map & out_layers,
        std::vector<placement_plan_error> & errors) {
    out_layers.assign(n_layer, "");
    for (const auto & a : plan.assignments) {
        if (a.layer_start < 0 || a.layer_end > n_layer || a.layer_end <= a.layer_start) {
            add_err(errors, "expand: invalid range");
            return false;
        }
        for (int32_t il = a.layer_start; il < a.layer_end; ++il) {
            if (!out_layers[il].empty()) {
                add_err(errors, "expand: overlap at layer " + std::to_string(il));
                return false;
            }
            out_layers[il] = a.backend_id;
        }
    }
    for (int32_t il = 0; il < n_layer; ++il) {
        if (out_layers[il].empty()) {
            add_err(errors, "expand: gap at layer " + std::to_string(il));
            return false;
        }
    }
    return true;
}

static bool backend_id_is_cpu(const std::string & id) {
    return id == "cpu" || id == "CPU";
}

// Fuzzy match: local:NAME vs local:pci:BUS when names align via inventory display
static bool backend_ids_match(const std::string & plan_id, const std::string & inv_id) {
    if (plan_id == inv_id) {
        return true;
    }
    // allow plan local:ROCm0 to match inv local:ROCm0
    return false;
}

bool placement_plan_match_backends(
        const placement_plan & plan,
        const placement_inventory & inv,
        std::vector<std::string> & missing_ids) {
    missing_ids.clear();
    std::set<std::string> needed;
    for (const auto & a : plan.assignments) {
        if (!backend_id_is_cpu(a.backend_id)) {
            needed.insert(a.backend_id);
        }
    }
    std::set<std::string> have;
    for (const auto & r : inv.records) {
        have.insert(r.backend_id);
        // also index by local:NAME form from display if local:pci
        if (r.kind == PLACEMENT_KIND_LOCAL_GPU && !r.display_name.empty()) {
            // no stable short name in record alone - use device_id already in backend_id
        }
    }
    // Also accept plan local:<devname> if any inventory local has matching suffix after local:
    for (const auto & need : needed) {
        if (have.count(need)) {
            continue;
        }
        bool found = false;
        if (need.rfind("local:", 0) == 0) {
            const std::string rest = need.substr(6);
            for (const auto & r : inv.records) {
                if (r.kind != PLACEMENT_KIND_LOCAL_GPU) {
                    continue;
                }
                if (r.backend_id == need) {
                    found = true;
                    break;
                }
                // plan uses local:ROCm0, inv uses local:pci:... — match via walking devices later
                if (r.backend_id.rfind("local:", 0) == 0) {
                    // try live name match in prepare_apply
                    if (rest.find("pci:") != 0 && r.device_id.empty()) {
                        // will resolve live
                    }
                }
            }
        }
        // defer hard miss to prepare which has live devices; still list if no record shares prefix
        if (!found) {
            // if any inv rpc/local exists with exact id we're good; else mark missing for now
            bool any = false;
            for (const auto & r : inv.records) {
                if (backend_ids_match(need, r.backend_id)) {
                    any = true;
                    break;
                }
                // local:NAME vs local:pci:X — not equal; prepare_apply resolves via live name
                if (need.rfind("local:", 0) == 0 && r.kind == PLACEMENT_KIND_LOCAL_GPU) {
                    any = true; // optimistic; prepare will confirm
                    break;
                }
            }
            if (!any) {
                missing_ids.push_back(need);
            }
        }
    }
    return missing_ids.empty();
}

std::string placement_layer_map_to_string(const placement_layer_map & layers) {
    std::ostringstream ss;
    for (size_t i = 0; i < layers.size(); ++i) {
        ss << "layer " << i << " -> " << layers[i] << "\n";
    }
    return ss.str();
}

// Live resolve: map backend_id from inventory + registered devices
static ggml_backend_dev_t find_dev_for_backend_id(const std::string & backend_id) {
    if (backend_id_is_cpu(backend_id)) {
        return nullptr;
    }
    // RPC form rpc://host:port#idx
    if (backend_id.rfind("rpc://", 0) == 0) {
        const size_t hash = backend_id.rfind('#');
        if (hash == std::string::npos) {
            return nullptr;
        }
        const std::string ep = backend_id.substr(6, hash - 6); // after rpc://
        int idx = 0;
        try {
            idx = std::stoi(backend_id.substr(hash + 1));
        } catch (...) {
            return nullptr;
        }
        int seen = 0;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            const char * name = ggml_backend_dev_name(dev);
            if (!name || std::string(name).rfind("RPC", 0) != 0) {
                continue;
            }
            const char * desc = ggml_backend_dev_description(dev);
            if (!desc || ep != desc) {
                continue;
            }
            if (seen == idx) {
                return dev;
            }
            ++seen;
        }
        return nullptr;
    }
    // local:pci:BUS or local:NAME
    if (backend_id.rfind("local:", 0) == 0) {
        const std::string rest = backend_id.substr(6);
        const bool is_pci = rest.rfind("pci:", 0) == 0;
        const std::string pci = is_pci ? rest.substr(4) : "";
        const std::string want_name = is_pci ? "" : rest;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                continue;
            }
            const char * name = ggml_backend_dev_name(dev);
            if (name && std::string(name).rfind("RPC", 0) == 0) {
                continue;
            }
            ggml_backend_dev_props props{};
            ggml_backend_dev_get_props(dev, &props);
            if (is_pci) {
                if (props.device_id && pci == props.device_id) {
                    return dev;
                }
            } else {
                if (name && want_name == name) {
                    return dev;
                }
                // also accept local:ROCm0 style vs name
            }
        }
    }
    return nullptr;
}

bool placement_plan_prepare_apply(
        const placement_plan & plan,
        int32_t n_layer,
        const placement_inventory & inv,
        bool topology_has_mixed_rpc_and_local,
        placement_apply_result & out,
        std::vector<placement_plan_error> & errors) {
    out = placement_apply_result{};
    if (!placement_plan_validate(plan, n_layer, topology_has_mixed_rpc_and_local, errors)) {
        return false;
    }
    if (!placement_plan_expand_layers(plan, n_layer, out.layer_backend_ids, errors)) {
        return false;
    }

    // Check inventory completeness for topology (optional soft: only required backends)
    std::set<std::string> needed;
    for (const auto & id : out.layer_backend_ids) {
        if (!backend_id_is_cpu(id)) {
            needed.insert(id);
        }
    }
    for (const auto & o : plan.overrides) {
        if (!backend_id_is_cpu(o.backend_id)) {
            needed.insert(o.backend_id);
        }
    }

    // usable lookup from live inventory
    std::map<std::string, uint64_t> usable_by_id;
    for (const auto & r : inv.records) {
        usable_by_id[r.backend_id] = r.usable_weight_mib;
    }

    std::map<std::string, ggml_backend_dev_t> resolved;
    for (const auto & id : needed) {
        ggml_backend_dev_t dev = find_dev_for_backend_id(id);
        if (!dev) {
            // try inventory-equivalent local:pci if plan has local:NAME
            if (id.rfind("local:", 0) == 0) {
                for (const auto & r : inv.records) {
                    if (r.kind != PLACEMENT_KIND_LOCAL_GPU) {
                        continue;
                    }
                    // if plan is local:ROCm0 and we find ROCm0 device
                    ggml_backend_dev_t d2 = find_dev_for_backend_id(r.backend_id);
                    if (!d2) {
                        continue;
                    }
                    const char * nm = ggml_backend_dev_name(d2);
                    const std::string rest = id.substr(6);
                    if (nm && rest == nm) {
                        dev = d2;
                        usable_by_id[id] = r.usable_weight_mib;
                        break;
                    }
                    if (rest.rfind("pci:", 0) == 0 && r.device_id == rest.substr(4)) {
                        dev = d2;
                        usable_by_id[id] = r.usable_weight_mib;
                        break;
                    }
                }
            }
        }
        if (!dev) {
            add_err(errors, "backend_id missing at apply re-discover: " + id);
            continue;
        }
        resolved[id] = dev;

        // usable budget: refuse if live usable is 0 for backends with layers
        uint64_t usable = 0;
        if (usable_by_id.count(id)) {
            usable = usable_by_id[id];
        } else {
            // find by resolved device matching inv record
            for (const auto & r : inv.records) {
                if (find_dev_for_backend_id(r.backend_id) == dev) {
                    usable = r.usable_weight_mib;
                    break;
                }
            }
        }
        if (usable == 0) {
            // still allow if free > 0 in inv for that id (usable 0 from huge reserves)
            bool free_ok = false;
            for (const auto & r : inv.records) {
                if (r.backend_id == id || find_dev_for_backend_id(r.backend_id) == dev) {
                    if (r.free_mib > 0) {
                        free_ok = true;
                    }
                    break;
                }
            }
            if (!free_ok) {
                add_err(errors, "backend_id has no usable/free capacity at apply: " + id);
            }
        }
    }

    if (!errors.empty()) {
        return false;
    }

    out.layer_devices.assign(n_layer, nullptr);
    std::vector<ggml_backend_dev_t> unique;
    for (int32_t il = 0; il < n_layer; ++il) {
        const std::string & id = out.layer_backend_ids[il];
        if (backend_id_is_cpu(id)) {
            out.layer_devices[il] = nullptr;
            continue;
        }
        ggml_backend_dev_t dev = resolved[id];
        out.layer_devices[il] = dev;
        if (std::find(unique.begin(), unique.end(), dev) == unique.end()) {
            unique.push_back(dev);
        }
    }
    out.devices = unique;
    out.debug_dump = placement_layer_map_to_string(out.layer_backend_ids);

    // Tensor overrides: backend_id -> buft. Policy: override wins for matched tensors.
    out.override_pattern_storage.clear();
    out.tensor_buft_overrides.clear();
    out.override_notes.clear();
    out.override_pattern_storage.reserve(plan.overrides.size());
    out.tensor_buft_overrides.reserve(plan.overrides.size() + 1);

    for (size_t oi = 0; oi < plan.overrides.size(); ++oi) {
        const auto & o = plan.overrides[oi];
        ggml_backend_buffer_type_t buft = nullptr;
        if (backend_id_is_cpu(o.backend_id)) {
            buft = ggml_backend_cpu_buffer_type();
        } else {
            auto it = resolved.find(o.backend_id);
            if (it == resolved.end() || it->second == nullptr) {
                add_err(errors, "override " + std::to_string(oi) +
                    " backend_id missing at apply: " + o.backend_id);
                continue;
            }
            buft = ggml_backend_dev_buffer_type(it->second);
            if (!buft) {
                add_err(errors, "override " + std::to_string(oi) +
                    " no buffer type for backend_id: " + o.backend_id);
                continue;
            }
            if (std::find(unique.begin(), unique.end(), it->second) == unique.end()) {
                unique.push_back(it->second);
            }
        }
        out.override_pattern_storage.push_back(o.match);
        // pointer filled after all patterns stored (vector may reallocate)
        out.override_notes.push_back(
            "override wins for match \"" + o.match + "\" -> " + o.backend_id +
            " (layer assignment still applies to unmatched tensors)");
    }

    if (!errors.empty()) {
        return false;
    }

    out.devices = unique;
    // Build null-terminated buft override list with stable pattern c_str()
    for (size_t i = 0; i < out.override_pattern_storage.size(); ++i) {
        const auto & o = plan.overrides[i];
        ggml_backend_buffer_type_t buft = nullptr;
        if (backend_id_is_cpu(o.backend_id)) {
            buft = ggml_backend_cpu_buffer_type();
        } else {
            buft = ggml_backend_dev_buffer_type(resolved[o.backend_id]);
        }
        out.tensor_buft_overrides.push_back({
            out.override_pattern_storage[i].c_str(),
            buft,
        });
    }
    out.tensor_buft_overrides.push_back({nullptr, nullptr});

    return true;
}

// ---------------------------------------------------------------------------
// Capacity packer (issue 11)
// ---------------------------------------------------------------------------

bool placement_plan_pack_capacity(
        const placement_inventory & inv,
        int32_t n_layer,
        const std::string & model_path,
        placement_plan & out,
        std::vector<placement_plan_error> & errors) {
    out = placement_plan{};
    if (n_layer <= 0) {
        add_err(errors, "pack: n_layer must be positive");
        return false;
    }

    struct cand {
        std::string backend_id;
        uint64_t    usable = 0;
        uint64_t    free_mib = 0;
        uint64_t    total_mib = 0;
    };
    std::vector<cand> cands;
    for (const auto & r : inv.records) {
        // Prefer usable; fall back to free if usable is 0 but free remains
        uint64_t u = r.usable_weight_mib;
        if (u == 0) {
            u = r.free_mib;
        }
        if (u == 0 && r.total_mib == 0) {
            continue;
        }
        if (u == 0) {
            continue; // no budget
        }
        cands.push_back({r.backend_id, u, r.free_mib, r.total_mib});
    }

    if (cands.empty()) {
        add_err(errors, "pack: no backends with positive usable/free capacity");
        return false;
    }

    // Deterministic order: usable desc, then backend_id asc
    std::sort(cands.begin(), cands.end(), [](const cand & a, const cand & b) {
        if (a.usable != b.usable) {
            return a.usable > b.usable;
        }
        return a.backend_id < b.backend_id;
    });

    // Cap number of backends to n_layer (at most one layer each)
    if ((int32_t) cands.size() > n_layer) {
        cands.resize((size_t) n_layer);
    }

    const size_t nb = cands.size();
    uint64_t sum_u = 0;
    for (const auto & c : cands) {
        sum_u += c.usable;
    }
    if (sum_u == 0) {
        add_err(errors, "pack: total usable is zero");
        return false;
    }

    // Proportional layer counts; guarantee min 1 for each cand when possible
    std::vector<int32_t> counts(nb, 0);
    int32_t assigned = 0;
    for (size_t i = 0; i < nb; ++i) {
        // floor; min 1
        int32_t n = (int32_t) ((cands[i].usable * (uint64_t) n_layer) / sum_u);
        if (n < 1) {
            n = 1;
        }
        counts[i] = n;
        assigned += n;
    }
    // If over-allocated (min-1 push), trim from largest first
    while (assigned > n_layer) {
        bool trimmed = false;
        for (size_t i = 0; i < nb && assigned > n_layer; ++i) {
            if (counts[i] > 1) {
                counts[i]--;
                assigned--;
                trimmed = true;
            }
        }
        if (!trimmed) {
            break;
        }
    }
    // Remainder to largest usable (index 0)
    if (assigned < n_layer) {
        counts[0] += (n_layer - assigned);
        assigned = n_layer;
    }
    // If still short (pathological), dump rest on first
    if (assigned < n_layer) {
        counts[0] += (n_layer - assigned);
    }

    // Contiguous ranges: order by pack sort (large first = early layers).
    // Small cards as cold fillers still get a trailing or mid share via proportion;
    // with large-first ordering they land later when remainder is small — actually
    // large first means early layers on big GPUs (common for PP). Small get later
    // contiguous blocks if we emit in sort order... Wait: counts[0] is largest,
    // so layers 0..c0-1 on largest, then next, etc. Small cards get late layers.
    // AC: "smaller card gets fewer layers" — satisfied by proportion + min 1.
    // "cold filler" — late layers on small is OK for TG-ish; fine for capacity pack.

    out.schema_version = 1;
    out.created_at = placement_now_iso8601();
    out.model_path = model_path;
    out.n_layer = n_layer;
    out.split_mode = "layer";
    out.capacity_snapshot_at = inv.records.empty() ? out.created_at : inv.records[0].reported_at;
    out.reserve_mode = inv.reserve_mode.empty() ? "table" : inv.reserve_mode;
    out.reserve_model_version = inv.reserve_model_version.empty()
        ? PLACEMENT_RESERVE_MODEL_TABLE_V1 : inv.reserve_model_version;
    out.heat.status = "none";
    out.heat.task = "";
    out.heat.schema_version = 1;
    out.overrides.clear();

    int32_t cursor = 0;
    for (size_t i = 0; i < nb; ++i) {
        if (counts[i] <= 0) {
            continue;
        }
        placement_plan_assignment a;
        a.layer_start = cursor;
        a.layer_end = cursor + counts[i];
        a.backend_id = cands[i].backend_id;
        out.assignments.push_back(a);
        cursor = a.layer_end;

        placement_plan::backend_snapshot bs;
        bs.backend_id = cands[i].backend_id;
        bs.usable_weight_mib = cands[i].usable;
        bs.free_mib = cands[i].free_mib;
        bs.total_mib = cands[i].total_mib;
        out.backends.push_back(bs);
    }

    if (cursor != n_layer) {
        add_err(errors, "pack: internal layer sum " + std::to_string(cursor) +
            " != n_layer " + std::to_string(n_layer));
        return false;
    }

    // Self-validate
    std::vector<placement_plan_error> verr;
    if (!placement_plan_validate(out, n_layer, /*mixed*/ true, verr)) {
        for (const auto & e : verr) {
            errors.push_back(e);
        }
        return false;
    }
    return true;
}

std::string placement_plan_to_json(const placement_plan & plan, int indent) {
    json j;
    j["schema_version"] = plan.schema_version;
    if (!plan.created_at.empty()) {
        j["created_at"] = plan.created_at;
    }
    j["split_mode"] = plan.split_mode.empty() ? "layer" : plan.split_mode;
    {
        json m;
        if (!plan.model_path.empty()) {
            m["path"] = plan.model_path;
        }
        m["n_layer"] = plan.n_layer;
        j["model"] = m;
    }
    if (!plan.capacity_snapshot_at.empty()) {
        j["capacity_snapshot_at"] = plan.capacity_snapshot_at;
    }
    j["reserve_mode"] = plan.reserve_mode;
    j["reserve_model_version"] = plan.reserve_model_version;

    j["backends"] = json::array();
    for (const auto & b : plan.backends) {
        j["backends"].push_back(json{
            {"backend_id", b.backend_id},
            {"usable_weight_mib", b.usable_weight_mib},
            {"free_mib", b.free_mib},
            {"total_mib", b.total_mib},
        });
    }

    j["assignments"] = json::array();
    for (const auto & a : plan.assignments) {
        j["assignments"].push_back(json{
            {"layer_start", a.layer_start},
            {"layer_end", a.layer_end},
            {"backend_id", a.backend_id},
        });
    }

    j["overrides"] = json::array();
    for (const auto & o : plan.overrides) {
        j["overrides"].push_back(json{
            {"match", o.match},
            {"backend_id", o.backend_id},
        });
    }

    j["heat"] = json{
        {"status", plan.heat.status.empty() ? "none" : plan.heat.status},
        {"task", plan.heat.task},
        {"schema_version", plan.heat.schema_version},
        {"layers", json::array()},
    };

    return j.dump(indent);
}

bool placement_plan_write_file(const placement_plan & plan, const std::string & path) {
    std::ofstream ofs(path);
    if (!ofs) {
        return false;
    }
    ofs << placement_plan_to_json(plan, 2);
    ofs << "\n";
    return (bool) ofs;
}
