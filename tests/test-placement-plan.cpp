// Unit tests for placement plan IR (P1 layer ranges).
// Seams: parse, validate coverage, expand, illegal tensor, empty overrides.

#include "placement-plan.h"
#include "placement-capacity.h"

#include <cstdio>
#include <cstring>
#include <string>

static int g_fails = 0;

#define EXPECT_TRUE(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_fails; \
    } \
} while (0)

#define EXPECT_EQ_STR(a, b) do { \
    if (std::string(a) != std::string(b)) { \
        std::fprintf(stderr, "FAIL %s:%d: '%s' != '%s'\n", __FILE__, __LINE__, \
            std::string(a).c_str(), std::string(b).c_str()); \
        ++g_fails; \
    } \
} while (0)

static const char * k_good_plan = R"json({
  "schema_version": 1,
  "split_mode": "layer",
  "model": { "n_layer": 8, "path": "toy.gguf" },
  "assignments": [
    { "layer_start": 0, "layer_end": 2, "backend_id": "rpc://192.168.8.23:50055#0" },
    { "layer_start": 2, "layer_end": 8, "backend_id": "local:ROCm0" }
  ],
  "overrides": [],
  "heat": { "status": "none", "layers": [] }
})json";

static void test_parse_and_validate_ok() {
    placement_plan plan;
    std::vector<placement_plan_error> err;
    EXPECT_TRUE(placement_plan_parse_json(k_good_plan, plan, err));
    EXPECT_TRUE(err.empty());
    EXPECT_TRUE(plan.schema_version == 1);
    EXPECT_TRUE(plan.n_layer == 8);
    EXPECT_EQ_STR(plan.split_mode, "layer");
    EXPECT_TRUE(plan.assignments.size() == 2);
    EXPECT_TRUE(plan.overrides.empty());
    EXPECT_EQ_STR(plan.heat.status, "none");

    err.clear();
    EXPECT_TRUE(placement_plan_validate(plan, 8, true, err));
    EXPECT_TRUE(err.empty());

    placement_layer_map layers;
    err.clear();
    EXPECT_TRUE(placement_plan_expand_layers(plan, 8, layers, err));
    EXPECT_TRUE(layers.size() == 8);
    EXPECT_EQ_STR(layers[0], "rpc://192.168.8.23:50055#0");
    EXPECT_EQ_STR(layers[1], "rpc://192.168.8.23:50055#0");
    EXPECT_EQ_STR(layers[2], "local:ROCm0");
    EXPECT_EQ_STR(layers[7], "local:ROCm0");

    const std::string dump = placement_layer_map_to_string(layers);
    EXPECT_TRUE(dump.find("layer 0 ->") != std::string::npos);
    // determinism: same expand twice
    placement_layer_map layers2;
    placement_plan_expand_layers(plan, 8, layers2, err);
    EXPECT_TRUE(layers == layers2);
}

static void test_gap_and_overlap() {
    const char * gap = R"json({
      "schema_version": 1,
      "model": { "n_layer": 4 },
      "assignments": [
        { "layer_start": 0, "layer_end": 2, "backend_id": "local:ROCm0" }
      ],
      "overrides": []
    })json";
    placement_plan plan;
    std::vector<placement_plan_error> err;
    EXPECT_TRUE(placement_plan_parse_json(gap, plan, err));
    err.clear();
    EXPECT_TRUE(!placement_plan_validate(plan, 4, false, err));
    EXPECT_TRUE(!err.empty());

    const char * overlap = R"json({
      "schema_version": 1,
      "model": { "n_layer": 4 },
      "assignments": [
        { "layer_start": 0, "layer_end": 3, "backend_id": "local:A" },
        { "layer_start": 2, "layer_end": 4, "backend_id": "local:B" }
      ],
      "overrides": []
    })json";
    err.clear();
    EXPECT_TRUE(placement_plan_parse_json(overlap, plan, err));
    err.clear();
    EXPECT_TRUE(!placement_plan_validate(plan, 4, false, err));
}

static void test_illegal_tensor_mixed() {
    const char * tplan = R"json({
      "schema_version": 1,
      "split_mode": "tensor",
      "model": { "n_layer": 2 },
      "assignments": [
        { "layer_start": 0, "layer_end": 1, "backend_id": "local:ROCm0" },
        { "layer_start": 1, "layer_end": 2, "backend_id": "rpc://h:1#0" }
      ],
      "overrides": []
    })json";
    placement_plan plan;
    std::vector<placement_plan_error> err;
    EXPECT_TRUE(placement_plan_parse_json(tplan, plan, err));
    err.clear();
    EXPECT_TRUE(!placement_plan_validate(plan, 2, true, err));
    bool found = false;
    for (const auto & e : err) {
        if (e.message.find("tensor") != std::string::npos) {
            found = true;
        }
    }
    EXPECT_TRUE(found);
}

static void test_nonempty_overrides_validate_ok() {
    const char * p = R"json({
      "schema_version": 1,
      "model": { "n_layer": 2 },
      "assignments": [
        { "layer_start": 0, "layer_end": 2, "backend_id": "local:ROCm0" }
      ],
      "overrides": [ { "match": "blk\\..*\\.ffn", "backend_id": "cpu" } ]
    })json";
    placement_plan plan;
    std::vector<placement_plan_error> err;
    EXPECT_TRUE(placement_plan_parse_json(p, plan, err));
    err.clear();
    EXPECT_TRUE(placement_plan_validate(plan, 2, false, err));
    EXPECT_TRUE(plan.overrides.size() == 1);
    EXPECT_EQ_STR(plan.overrides[0].backend_id, "cpu");
}

static void test_override_bad_regex() {
    const char * p = R"json({
      "schema_version": 1,
      "model": { "n_layer": 1 },
      "assignments": [
        { "layer_start": 0, "layer_end": 1, "backend_id": "cpu" }
      ],
      "overrides": [ { "match": "[invalid", "backend_id": "cpu" } ]
    })json";
    placement_plan plan;
    std::vector<placement_plan_error> err;
    EXPECT_TRUE(placement_plan_parse_json(p, plan, err));
    err.clear();
    EXPECT_TRUE(!placement_plan_validate(plan, 1, false, err));
}

static void test_override_missing_backend() {
    const char * p = R"json({
      "schema_version": 1,
      "model": { "n_layer": 2 },
      "assignments": [
        { "layer_start": 0, "layer_end": 2, "backend_id": "cpu" }
      ],
      "overrides": [ { "match": "blk\\..*", "backend_id": "rpc://missing:1#0" } ]
    })json";
    placement_plan plan;
    std::vector<placement_plan_error> err;
    EXPECT_TRUE(placement_plan_parse_json(p, plan, err));
    placement_inventory inv;
    inv.topology_complete = true;
    placement_apply_result apply;
    err.clear();
    EXPECT_TRUE(!placement_plan_prepare_apply(plan, 2, inv, false, apply, err));
    bool found = false;
    for (const auto & e : err) {
        if (e.message.find("override") != std::string::npos ||
            e.message.find("missing") != std::string::npos) {
            found = true;
        }
    }
    EXPECT_TRUE(found);
}

static void test_override_cpu_prepare() {
    // All layers on "cpu" + override to cpu: no live GPU required
    const char * p = R"json({
      "schema_version": 1,
      "model": { "n_layer": 2 },
      "assignments": [
        { "layer_start": 0, "layer_end": 2, "backend_id": "cpu" }
      ],
      "overrides": [ { "match": "token_embd", "backend_id": "cpu" } ]
    })json";
    placement_plan plan;
    std::vector<placement_plan_error> err;
    EXPECT_TRUE(placement_plan_parse_json(p, plan, err));
    placement_inventory inv;
    inv.topology_complete = true;
    placement_apply_result apply;
    err.clear();
    EXPECT_TRUE(placement_plan_prepare_apply(plan, 2, inv, false, apply, err));
    EXPECT_TRUE(apply.tensor_buft_overrides.size() >= 2); // one + null term
    EXPECT_TRUE(apply.tensor_buft_overrides[0].pattern != nullptr);
    EXPECT_TRUE(apply.tensor_buft_overrides.back().pattern == nullptr);
    EXPECT_TRUE(!apply.override_notes.empty());
    // override wins policy note present
    EXPECT_TRUE(apply.override_notes[0].find("override wins") != std::string::npos);
}

static void test_cpu_range_ok() {
    const char * p = R"json({
      "schema_version": 1,
      "model": { "n_layer": 3 },
      "assignments": [
        { "layer_start": 0, "layer_end": 1, "backend_id": "cpu" },
        { "layer_start": 1, "layer_end": 3, "backend_id": "local:ROCm0" }
      ],
      "overrides": []
    })json";
    placement_plan plan;
    std::vector<placement_plan_error> err;
    EXPECT_TRUE(placement_plan_parse_json(p, plan, err));
    err.clear();
    EXPECT_TRUE(placement_plan_validate(plan, 3, false, err));
}

static void test_prepare_missing_backend() {
    placement_plan plan;
    std::vector<placement_plan_error> err;
    EXPECT_TRUE(placement_plan_parse_json(k_good_plan, plan, err));
    placement_inventory inv;
    inv.topology_complete = true;
    // only local, missing RPC
    placement_capacity_record r;
    r.backend_id = "local:ROCm0";
    r.kind = PLACEMENT_KIND_LOCAL_GPU;
    r.free_mib = 20000;
    r.total_mib = 24000;
    r.usable_weight_mib = 19000;
    inv.records.push_back(r);

    placement_apply_result apply;
    err.clear();
    // without live RPC device, prepare should fail on missing backend
    EXPECT_TRUE(!placement_plan_prepare_apply(plan, 8, inv, true, apply, err));
    EXPECT_TRUE(!err.empty());
}

int main() {
    test_parse_and_validate_ok();
    test_gap_and_overlap();
    test_illegal_tensor_mixed();
    test_nonempty_overrides_validate_ok();
    test_override_bad_regex();
    test_override_missing_backend();
    test_override_cpu_prepare();
    test_cpu_range_ok();
    test_prepare_missing_backend();

    if (g_fails != 0) {
        std::fprintf(stderr, "%d assertion(s) failed\n", g_fails);
        return 1;
    }
    std::printf("test-placement-plan: all passed\n");
    return 0;
}
