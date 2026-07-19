// Unit tests for Shape A RPC TP-unit gates and inventory collapse (issue 14).
// Seams: placement_tp_unit_eval_gates, placement_inventory_apply_tp_units,
//        placement_plan_ids_tp_double_count, JSON kind/members.

#include "placement-capacity.h"
#include "placement-plan.h"

#include <cstdio>
#include <string>
#include <vector>

static int g_fails = 0;

#define EXPECT_TRUE(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_fails; \
    } \
} while (0)

#define EXPECT_EQ_STR(a, b) do { \
    const std::string _a = (a); \
    const std::string _b = (b); \
    if (_a != _b) { \
        std::fprintf(stderr, "FAIL %s:%d: '%s' != '%s'\n", __FILE__, __LINE__, _a.c_str(), _b.c_str()); \
        ++g_fails; \
    } \
} while (0)

#define EXPECT_EQ_U64(a, b) do { \
    const uint64_t _a = (a); \
    const uint64_t _b = (b); \
    if (_a != _b) { \
        std::fprintf(stderr, "FAIL %s:%d: %llu != %llu\n", __FILE__, __LINE__, \
            (unsigned long long)_a, (unsigned long long)_b); \
        ++g_fails; \
    } \
} while (0)

static placement_capacity_record make_rpc(
        const std::string & ep, int idx, uint64_t total_mib, uint64_t free_mib,
        uint64_t usable, const std::string & fam = "CUDA") {
    placement_capacity_record r;
    r.kind = PLACEMENT_KIND_RPC_DEVICE;
    r.endpoint = ep;
    r.device_index = idx;
    r.backend_id = placement_make_rpc_backend_id(ep, idx);
    r.total_mib = total_mib;
    r.free_mib = free_mib;
    r.usable_weight_mib = usable;
    r.backend_family = fam;
    r.reported_at = "2026-07-19T00:00:00Z";
    r.reserve_mode = "table";
    r.reserve_model_version = PLACEMENT_RESERVE_MODEL_TABLE_V1;
    r.caps.multi_device = true;
    return r;
}

static void test_ids_and_vram() {
    EXPECT_EQ_STR(placement_make_rpc_tp_backend_id("host:50051"), "rpc-tp://host:50051");
    EXPECT_TRUE(placement_backend_id_is_tp_unit("rpc-tp://a:1"));
    EXPECT_TRUE(!placement_backend_id_is_tp_unit("rpc://a:1#0"));
    EXPECT_EQ_STR(placement_endpoint_from_backend_id("rpc-tp://a:1"), "a:1");
    EXPECT_EQ_STR(placement_endpoint_from_backend_id("rpc://a:1#0"), "a:1");

    EXPECT_TRUE(placement_tp_vram_equal({24576, 24576}));
    EXPECT_TRUE(placement_tp_vram_equal({24576, 24576 - 100}));
    EXPECT_TRUE(!placement_tp_vram_equal({24576, 8192})); // 24 vs 8
}

static void test_gates() {
    placement_tp_unit_gate_input in;
    in.n_devices = 2;
    in.total_mib = {24576, 24576};
    in.families = {"CUDA", "CUDA"};
    in.specialized_ar_ok = true;
    in.opt_in = true;
    auto ok = placement_tp_unit_eval_gates(in);
    EXPECT_TRUE(ok.eligible);
    EXPECT_TRUE(ok.reason.empty());

    // default no opt-in
    in.opt_in = false;
    auto no = placement_tp_unit_eval_gates(in);
    EXPECT_TRUE(!no.eligible);
    EXPECT_TRUE(no.reason.find("opt-in") != std::string::npos);

    // mixed VRAM
    in.opt_in = true;
    in.total_mib = {24576, 8192};
    auto mix = placement_tp_unit_eval_gates(in);
    EXPECT_TRUE(!mix.eligible);
    EXPECT_TRUE(mix.reason.find("mixed") != std::string::npos ||
                mix.reason.find("VRAM") != std::string::npos);

    // butterfly / no specialized AR
    in.total_mib = {24576, 24576};
    in.specialized_ar_ok = false;
    auto ar = placement_tp_unit_eval_gates(in);
    EXPECT_TRUE(!ar.eligible);
    EXPECT_TRUE(ar.reason.find("AllReduce") != std::string::npos);

    // N=3
    in.specialized_ar_ok = true;
    in.n_devices = 3;
    in.total_mib = {8, 8, 8};
    in.families = {"CUDA", "CUDA", "CUDA"};
    auto n3 = placement_tp_unit_eval_gates(in);
    EXPECT_TRUE(!n3.eligible);
    EXPECT_TRUE(n3.reason.find("N=") != std::string::npos);

    // heterogeneous family
    in.n_devices = 2;
    in.total_mib = {24576, 24576};
    in.families = {"CUDA", "HIP"};
    auto fam = placement_tp_unit_eval_gates(in);
    EXPECT_TRUE(!fam.eligible);
    EXPECT_TRUE(fam.reason.find("family") != std::string::npos);
}

static void test_default_shape_b() {
    placement_inventory inv;
    inv.records.push_back(make_rpc("node:50052", 0, 24576, 20000, 18000));
    inv.records.push_back(make_rpc("node:50052", 1, 24576, 20000, 18000));

    placement_tp_unit_options opts; // opt_in false
    opts.specialized_ar_ok_default = true;
    placement_inventory_apply_tp_units(inv, opts, nullptr);

    EXPECT_EQ_U64((uint64_t) inv.records.size(), 2ull);
    EXPECT_EQ_STR(placement_kind_str(inv.records[0].kind), "rpc_device");
    EXPECT_EQ_STR(placement_kind_str(inv.records[1].kind), "rpc_device");
    // hardware ready but not opted in
    EXPECT_TRUE(!inv.records[0].tp_unit_eligible);
    EXPECT_TRUE(inv.records[0].tp_unit_ineligible_reason.find("opt-in") != std::string::npos);
}

static void test_collapse_equal_vram() {
    placement_inventory inv;
    inv.records.push_back(make_rpc("node:50052", 0, 24576, 20000, 18000));
    inv.records.push_back(make_rpc("node:50052", 1, 24576, 20000, 18000));
    placement_capacity_record local;
    local.kind = PLACEMENT_KIND_LOCAL_GPU;
    local.backend_id = "local:ROCm0";
    local.usable_weight_mib = 20000;
    local.total_mib = 24576;
    inv.records.push_back(local);

    placement_tp_unit_options opts;
    opts.opt_in = true;
    opts.specialized_ar_ok_default = true;
    opts.tp_workspace_pad_mib = 512;
    std::vector<std::string> logs;
    placement_inventory_apply_tp_units(inv, opts, &logs);

    EXPECT_EQ_U64((uint64_t) inv.records.size(), 2ull); // local + unit
    bool found_unit = false;
    for (const auto & r : inv.records) {
        if (r.kind == PLACEMENT_KIND_RPC_TP_UNIT) {
            found_unit = true;
            EXPECT_EQ_STR(r.backend_id, "rpc-tp://node:50052");
            EXPECT_EQ_U64((uint64_t) r.members.size(), 2ull);
            // sum usable 36000 - 512
            EXPECT_EQ_U64(r.usable_weight_mib, 36000ull - 512ull);
            EXPECT_EQ_STR(placement_kind_str(r.kind), "rpc_tp_unit");
        }
    }
    EXPECT_TRUE(found_unit);
    EXPECT_TRUE(!logs.empty());

    const std::string js = placement_inventory_to_json(inv, 2);
    EXPECT_TRUE(js.find("rpc_tp_unit") != std::string::npos);
    EXPECT_TRUE(js.find("members") != std::string::npos);
    EXPECT_TRUE(js.find("rpc-tp://node:50052") != std::string::npos);
}

static void test_mixed_vram_stays_b() {
    placement_inventory inv;
    inv.records.push_back(make_rpc("mix:1", 0, 24576, 20000, 18000));
    inv.records.push_back(make_rpc("mix:1", 1, 8192, 7000, 6000));

    placement_tp_unit_options opts;
    opts.opt_in = true;
    opts.specialized_ar_ok_default = true;
    placement_inventory_apply_tp_units(inv, opts, nullptr);

    EXPECT_EQ_U64((uint64_t) inv.records.size(), 2ull);
    EXPECT_EQ_STR(placement_kind_str(inv.records[0].kind), "rpc_device");
    EXPECT_TRUE(!inv.records[0].tp_unit_eligible);
    EXPECT_TRUE(inv.records[0].tp_unit_ineligible_reason.find("mixed") != std::string::npos ||
                inv.records[0].tp_unit_ineligible_reason.find("VRAM") != std::string::npos);
}

static void test_butterfly_stays_b() {
    placement_inventory inv;
    inv.records.push_back(make_rpc("ar:1", 0, 24576, 20000, 18000));
    inv.records.push_back(make_rpc("ar:1", 1, 24576, 20000, 18000));

    placement_tp_unit_options opts;
    opts.opt_in = true;
    opts.specialized_ar_ok_default = false; // butterfly-only
    placement_inventory_apply_tp_units(inv, opts, nullptr);

    EXPECT_EQ_U64((uint64_t) inv.records.size(), 2ull);
    EXPECT_TRUE(inv.records[0].tp_unit_ineligible_reason.find("AllReduce") != std::string::npos);
}

static void test_double_count_guard() {
    std::vector<std::string> ids = {
        "local:ROCm0",
        "rpc-tp://node:50052",
        "rpc://node:50052#0",
    };
    std::vector<std::string> errs;
    EXPECT_TRUE(!placement_plan_ids_tp_double_count(ids, errs));
    EXPECT_TRUE(!errs.empty());

    errs.clear();
    ids = {"local:ROCm0", "rpc-tp://node:50052"};
    EXPECT_TRUE(placement_plan_ids_tp_double_count(ids, errs));
    EXPECT_TRUE(errs.empty());

    // plan validate path
    placement_plan plan;
    plan.schema_version = 1;
    plan.split_mode = "layer";
    plan.n_layer = 4;
    {
        placement_plan_assignment a;
        a.layer_start = 0; a.layer_end = 2; a.backend_id = "rpc-tp://ep:1";
        plan.assignments.push_back(a);
    }
    {
        placement_plan_assignment a;
        a.layer_start = 2; a.layer_end = 4; a.backend_id = "rpc://ep:1#0";
        plan.assignments.push_back(a);
    }
    std::vector<placement_plan_error> perrs;
    EXPECT_TRUE(!placement_plan_validate(plan, 4, true, perrs));
    bool found_dc = false;
    for (const auto & e : perrs) {
        if (e.message.find("double-count") != std::string::npos) {
            found_dc = true;
        }
    }
    EXPECT_TRUE(found_dc);
}

static void test_outer_layer_rail_ok() {
    // local + logical unit is legal (no double-count)
    placement_plan plan;
    plan.schema_version = 1;
    plan.split_mode = "layer";
    plan.n_layer = 4;
    {
        placement_plan_assignment a;
        a.layer_start = 0; a.layer_end = 2; a.backend_id = "local:ROCm0";
        plan.assignments.push_back(a);
    }
    {
        placement_plan_assignment a;
        a.layer_start = 2; a.layer_end = 4; a.backend_id = "rpc-tp://node:50052";
        plan.assignments.push_back(a);
    }
    std::vector<placement_plan_error> perrs;
    EXPECT_TRUE(placement_plan_validate(plan, 4, true, perrs));
}

int main() {
    test_ids_and_vram();
    test_gates();
    test_default_shape_b();
    test_collapse_equal_vram();
    test_mixed_vram_stays_b();
    test_butterfly_stays_b();
    test_double_count_guard();
    test_outer_layer_rail_ok();

    if (g_fails != 0) {
        std::fprintf(stderr, "%d test assertion(s) failed\n", g_fails);
        return 1;
    }
    std::printf("test-placement-tp-unit: all passed\n");
    return 0;
}
