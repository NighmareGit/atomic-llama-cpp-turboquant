// Unit tests for capacity packer (issue 11).
// Seams: pack proportions, determinism, validate, heat none, JSON round-trip.

#include "placement-plan.h"
#include "placement-capacity.h"

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

#define EXPECT_EQ_I(a, b) do { \
    if ((a) != (b)) { \
        std::fprintf(stderr, "FAIL %s:%d: %d != %d\n", __FILE__, __LINE__, (int)(a), (int)(b)); \
        ++g_fails; \
    } \
} while (0)

static placement_inventory make_hetero_inv() {
    placement_inventory inv;
    inv.topology_complete = true;
    inv.reserve_mode = "table";
    inv.reserve_model_version = PLACEMENT_RESERVE_MODEL_TABLE_V1;
    {
        placement_capacity_record r;
        r.backend_id = "local:pci:0000:03:00.0";
        r.kind = PLACEMENT_KIND_LOCAL_GPU;
        r.total_mib = 24560;
        r.free_mib = 24000;
        r.usable_weight_mib = 23000;
        r.reported_at = "2026-07-18T00:00:00Z";
        inv.records.push_back(r);
    }
    {
        placement_capacity_record r;
        r.backend_id = "rpc://127.0.0.1:50051#0";
        r.kind = PLACEMENT_KIND_RPC_DEVICE;
        r.total_mib = 8192;
        r.free_mib = 7500;
        r.usable_weight_mib = 7000; // 8 GB class
        r.reported_at = "2026-07-18T00:00:00Z";
        inv.records.push_back(r);
    }
    {
        placement_capacity_record r;
        r.backend_id = "rpc://192.168.8.23:50054#0";
        r.kind = PLACEMENT_KIND_RPC_DEVICE;
        r.total_mib = 24123;
        r.free_mib = 23000;
        r.usable_weight_mib = 22000;
        r.reported_at = "2026-07-18T00:00:00Z";
        inv.records.push_back(r);
    }
    return inv;
}

static void test_pack_hetero_proportions() {
    auto inv = make_hetero_inv();
    placement_plan plan;
    std::vector<placement_plan_error> err;
    EXPECT_TRUE(placement_plan_pack_capacity(inv, 33, "model.gguf", plan, err));
    EXPECT_TRUE(err.empty());
    EXPECT_EQ_I(plan.n_layer, 33);
    EXPECT_TRUE(plan.heat.status == "none");
    EXPECT_TRUE(plan.overrides.empty());
    EXPECT_TRUE(plan.split_mode == "layer");
    EXPECT_TRUE(plan.assignments.size() >= 2);

    // Full coverage
    err.clear();
    EXPECT_TRUE(placement_plan_validate(plan, 33, true, err));

    // Count layers per backend
    int n_local = 0, n_8g = 0, n_3090 = 0;
    for (const auto & a : plan.assignments) {
        const int n = a.layer_end - a.layer_start;
        if (a.backend_id.find("0000:03:00.0") != std::string::npos) {
            n_local = n;
        } else if (a.backend_id.find("50051") != std::string::npos) {
            n_8g = n;
        } else if (a.backend_id.find("50054") != std::string::npos) {
            n_3090 = n;
        }
    }
    EXPECT_TRUE(n_local > 0);
    EXPECT_TRUE(n_8g > 0);
    EXPECT_TRUE(n_3090 > 0);
    // 8 GB fewer than 24 GB class
    EXPECT_TRUE(n_8g < n_local);
    EXPECT_TRUE(n_8g < n_3090);
    EXPECT_EQ_I(n_local + n_8g + n_3090, 33);
}

static void test_pack_deterministic() {
    auto inv = make_hetero_inv();
    placement_plan p1, p2;
    std::vector<placement_plan_error> err;
    EXPECT_TRUE(placement_plan_pack_capacity(inv, 33, "m.gguf", p1, err));
    err.clear();
    EXPECT_TRUE(placement_plan_pack_capacity(inv, 33, "m.gguf", p2, err));
    EXPECT_EQ_I((int) p1.assignments.size(), (int) p2.assignments.size());
    for (size_t i = 0; i < p1.assignments.size(); ++i) {
        EXPECT_EQ_I(p1.assignments[i].layer_start, p2.assignments[i].layer_start);
        EXPECT_EQ_I(p1.assignments[i].layer_end, p2.assignments[i].layer_end);
        EXPECT_TRUE(p1.assignments[i].backend_id == p2.assignments[i].backend_id);
    }
    // JSON stable enough for same structure
    EXPECT_TRUE(placement_plan_to_json(p1) == placement_plan_to_json(p2));
}

static void test_pack_single_backend() {
    placement_inventory inv;
    inv.topology_complete = true;
    placement_capacity_record r;
    r.backend_id = "local:ROCm0";
    r.usable_weight_mib = 20000;
    r.free_mib = 20000;
    r.total_mib = 24000;
    inv.records.push_back(r);
    placement_plan plan;
    std::vector<placement_plan_error> err;
    EXPECT_TRUE(placement_plan_pack_capacity(inv, 10, "", plan, err));
    EXPECT_EQ_I((int) plan.assignments.size(), 1);
    EXPECT_EQ_I(plan.assignments[0].layer_start, 0);
    EXPECT_EQ_I(plan.assignments[0].layer_end, 10);
}

static void test_pack_empty_fails() {
    placement_inventory inv;
    placement_plan plan;
    std::vector<placement_plan_error> err;
    EXPECT_TRUE(!placement_plan_pack_capacity(inv, 10, "", plan, err));
    EXPECT_TRUE(!err.empty());
}

static void test_json_roundtrip() {
    auto inv = make_hetero_inv();
    placement_plan plan;
    std::vector<placement_plan_error> err;
    EXPECT_TRUE(placement_plan_pack_capacity(inv, 16, "toy.gguf", plan, err));
    const std::string js = placement_plan_to_json(plan);
    EXPECT_TRUE(js.find("\"heat\"") != std::string::npos);
    EXPECT_TRUE(js.find("\"none\"") != std::string::npos);
    EXPECT_TRUE(js.find("assignments") != std::string::npos);
    placement_plan plan2;
    err.clear();
    EXPECT_TRUE(placement_plan_parse_json(js, plan2, err));
    EXPECT_TRUE(placement_plan_validate(plan2, 16, true, err));
}

int main() {
    test_pack_hetero_proportions();
    test_pack_deterministic();
    test_pack_single_backend();
    test_pack_empty_fails();
    test_json_roundtrip();
    if (g_fails) {
        std::fprintf(stderr, "%d assertion(s) failed\n", g_fails);
        return 1;
    }
    std::printf("test-placement-pack: all passed\n");
    return 0;
}
