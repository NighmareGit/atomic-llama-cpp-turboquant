// Unit tests for per-layer heatmap rollup (placement-grade heat).
// Seams: parse node_timings (sched-trace + server-telemetry), merge,
// layer_rollup, op_categories, heat_status. Synthetic JSONL, no GPU.

#include "heatmap-rollup.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

static int g_fails = 0;

#define EXPECT_TRUE(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_fails; \
    } \
} while (0)

#define EXPECT_EQ_INT(a, b) do { \
    if ((a) != (b)) { \
        std::fprintf(stderr, "FAIL %s:%d: %d != %d\n", __FILE__, __LINE__, (int)(a), (int)(b)); \
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

#define EXPECT_NEAR(a, b, tol) do { \
    if (std::fabs((double)(a) - (double)(b)) > (tol)) { \
        std::fprintf(stderr, "FAIL %s:%d: %f != %f (tol %f)\n", __FILE__, __LINE__, \
            (double)(a), (double)(b), (double)(tol)); \
        ++g_fails; \
    } \
} while (0)

// ---------------------------------------------------------------------------
// Helpers to write synthetic JSONL
// ---------------------------------------------------------------------------

static void write_lines(const std::string & path, const std::vector<std::string> & lines) {
    std::ofstream f(path);
    for (const auto & l : lines) f << l << "\n";
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

static void test_layer_index_from_name() {
    int idx = -1;
    EXPECT_TRUE(heatmap_layer_index_from_name("l_out-0.attn_q", idx));
    EXPECT_EQ_INT(idx, 0);
    EXPECT_TRUE(heatmap_layer_index_from_name("l_out-14.ffn_gate", idx));
    EXPECT_EQ_INT(idx, 14);
    EXPECT_TRUE(heatmap_layer_index_from_name("blk.3.attn_norm", idx));
    EXPECT_EQ_INT(idx, 3);
    EXPECT_TRUE(heatmap_layer_index_from_name("blk.0.proj", idx));
    EXPECT_EQ_INT(idx, 0);
    // Non-layer names return false.
    EXPECT_TRUE(!heatmap_layer_index_from_name("inp_embd", idx));
    EXPECT_TRUE(!heatmap_layer_index_from_name("output_norm", idx));
    EXPECT_TRUE(!heatmap_layer_index_from_name("output", idx));
}

static void test_op_category_from_op() {
    EXPECT_EQ_STR(heatmap_op_category_from_op("attn_q"), "attn");
    EXPECT_EQ_STR(heatmap_op_category_from_op("attn_k"), "attn");
    EXPECT_EQ_STR(heatmap_op_category_from_op("attn_o"), "attn");
    EXPECT_EQ_STR(heatmap_op_category_from_op("ffn_gate"), "ffn");
    EXPECT_EQ_STR(heatmap_op_category_from_op("ffn_up"), "ffn");
    EXPECT_EQ_STR(heatmap_op_category_from_op("ffn_down"), "ffn");
    // norm precedence: attn_norm / ffn_norm are norm, not attn/ffn.
    EXPECT_EQ_STR(heatmap_op_category_from_op("attn_norm"), "norm");
    EXPECT_EQ_STR(heatmap_op_category_from_op("ffn_norm"), "norm");
    EXPECT_EQ_STR(heatmap_op_category_from_op("output_norm"), "norm");
    EXPECT_EQ_STR(heatmap_op_category_from_op("inp_embd"), "other");
    EXPECT_EQ_STR(heatmap_op_category_from_op("output"), "other");
}

static void test_parse_sched_trace_node_timings() {
    const std::string path = "/tmp/test-sched-trace.jsonl";
    write_lines(path, {
        "{\"ts_us\":1,\"trace_id\":1,\"split\":0,\"backend\":0,\"copy\":0,"
        "\"phase\":\"graph_compute_async\",\"node_timings\":["
        "{\"name\":\"l_out-0.attn_q\",\"us\":100},"
        "{\"name\":\"l_out-0.attn_k\",\"us\":80}],"
        "\"elapsed_us\":180}",
        "{\"ts_us\":2,\"trace_id\":2,\"split\":0,\"backend\":0,\"copy\":0,"
        "\"phase\":\"graph_compute_async\",\"node_timings\":["
        "{\"name\":\"l_out-0.attn_q\",\"us\":120},"
        "{\"name\":\"l_out-0.attn_k\",\"us\":90}],"
        "\"elapsed_us\":210}",
        // A non-graph_compute_async line should be ignored.
        "{\"ts_us\":3,\"trace_id\":3,\"split\":0,\"backend\":0,\"copy\":0,"
        "\"phase\":\"event_record\",\"elapsed_us\":5}"
    });

    auto nodes = heatmap_parse_node_timings_sched_trace(path);
    EXPECT_EQ_INT((int)nodes.size(), 2);
    // Running avg: attn_q = (100 + 120)/2 = 110; attn_k = (80 + 90)/2 = 85.
    EXPECT_EQ_STR(nodes[0].name, "l_out-0.attn_q");
    EXPECT_NEAR(nodes[0].us, 110.0, 0.001);
    EXPECT_EQ_INT(nodes[0].samples, 2);
    EXPECT_EQ_STR(nodes[1].name, "l_out-0.attn_k");
    EXPECT_NEAR(nodes[1].us, 85.0, 0.001);
}

static void test_parse_server_telemetry_node_timings() {
    const std::string path = "/tmp/test-server-telemetry.jsonl";
    write_lines(path, {
        // Raw form.
        "{\"event\":\"node_timings\",\"ts_us\":1,\"trace_id\":1,\"entries\":["
        "{\"name\":\"l_out-0.attn_q\",\"us\":89},"
        "{\"name\":\"l_out-0.attn_k\",\"us\":72}]}",
        // Aggregated form (avg_us present).
        "{\"event\":\"node_timings\",\"ts_us\":2,\"trace_id\":1,\"aggregated\":true,\"entries\":["
        "{\"name\":\"l_out-0.attn_q\",\"us\":91,\"avg_us\":91,\"min_us\":87,\"max_us\":95,\"count\":16},"
        "{\"name\":\"l_out-0.attn_k\",\"us\":73,\"avg_us\":73,\"min_us\":69,\"max_us\":77,\"count\":16}]}",
        // A server_telemetry frame (device-level) should be ignored by node parser.
        "{\"event\":\"server_telemetry\",\"ts_us\":3,\"trace_id\":1,"
        "\"device_timings_us\":[1082],\"layer_assignments\":[0]}"
    });

    auto nodes = heatmap_parse_node_timings_server_telemetry(path);
    EXPECT_EQ_INT((int)nodes.size(), 2);
    EXPECT_EQ_STR(nodes[0].name, "l_out-0.attn_q");
    EXPECT_EQ_INT(nodes[0].samples, 2);
    EXPECT_EQ_STR(nodes[1].name, "l_out-0.attn_k");
}

static void test_merge_node_timings() {
    std::vector<heatmap_node_timing> a = {{"l_out-0.attn_q", 100.0, 1}};
    std::vector<heatmap_node_timing> b = {{"l_out-0.attn_q", 200.0, 1}};
    auto m = heatmap_merge_node_timings(a, b);
    EXPECT_EQ_INT((int)m.size(), 1);
    EXPECT_NEAR(m[0].us, 150.0, 0.001);
    EXPECT_EQ_INT(m[0].samples, 2);

    // Disjoint names: union.
    std::vector<heatmap_node_timing> c = {{"l_out-0.attn_q", 100.0, 1}};
    std::vector<heatmap_node_timing> d = {{"l_out-1.attn_q", 300.0, 2}};
    auto m2 = heatmap_merge_node_timings(c, d);
    EXPECT_EQ_INT((int)m2.size(), 2);
}

static void test_layer_rollup_full() {
    // 8 layers, 7 nodes each (attn_q/k/v/o + ffn_gate/up/down).
    std::vector<heatmap_node_timing> nodes;
    for (int i = 0; i < 8; ++i) {
        nodes.push_back({"l_out-" + std::to_string(i) + ".attn_q", 10.0, 1});
        nodes.push_back({"l_out-" + std::to_string(i) + ".attn_k", 10.0, 1});
        nodes.push_back({"l_out-" + std::to_string(i) + ".attn_v", 10.0, 1});
        nodes.push_back({"l_out-" + std::to_string(i) + ".attn_o", 10.0, 1});
        nodes.push_back({"l_out-" + std::to_string(i) + ".ffn_gate", 20.0, 1});
        nodes.push_back({"l_out-" + std::to_string(i) + ".ffn_up", 20.0, 1});
        nodes.push_back({"l_out-" + std::to_string(i) + ".ffn_down", 20.0, 1});
    }
    std::vector<heatmap_layer_rollup> rollup;
    std::vector<heatmap_op_category> cats;
    bool full = heatmap_compute_layer_rollup(nodes, 8, rollup, cats);
    EXPECT_TRUE(full);
    EXPECT_EQ_INT((int)rollup.size(), 8);
    // Each layer: 4*10 + 3*20 = 100 us = 0.1 ms.
    EXPECT_NEAR(rollup[0].ms, 0.1, 1e-6);
    EXPECT_EQ_INT(rollup[0].n_nodes, 7);
    EXPECT_EQ_INT(rollup[3].idx, 3);

    // op_categories are cross-layer sums (TELEMETRY.md "across the entire model").
    // attn: 4 ops * 10 us * 8 layers = 320 us; ffn: 3 ops * 20 us * 8 = 480 us.
    EXPECT_EQ_INT((int)cats.size(), 2); // attn + ffn only
    EXPECT_EQ_STR(cats[0].category, "attn");
    EXPECT_NEAR(cats[0].us, 320.0, 1e-6);
    EXPECT_EQ_STR(cats[1].category, "ffn");
    EXPECT_NEAR(cats[1].us, 480.0, 1e-6);
}

static void test_layer_rollup_partial() {
    std::vector<heatmap_node_timing> nodes;
    // Only layers 0, 2, 5 present.
    for (int i : {0, 2, 5}) {
        nodes.push_back({"l_out-" + std::to_string(i) + ".attn_q", 10.0, 1});
    }
    std::vector<heatmap_layer_rollup> rollup;
    std::vector<heatmap_op_category> cats;
    bool full = heatmap_compute_layer_rollup(nodes, 8, rollup, cats);
    EXPECT_TRUE(!full);
    EXPECT_EQ_INT((int)rollup.size(), 3);
}

static void test_op_categories_with_norm() {
    std::vector<heatmap_node_timing> nodes = {
        {"l_out-0.attn_q", 10.0, 1},
        {"l_out-0.attn_norm", 5.0, 1},
        {"l_out-0.ffn_gate", 20.0, 1},
        {"l_out-0.ffn_norm", 5.0, 1},
        {"l_out-0.inp_embd", 1.0, 1}, // other
    };
    std::vector<heatmap_layer_rollup> rollup;
    std::vector<heatmap_op_category> cats;
    heatmap_compute_layer_rollup(nodes, 1, rollup, cats);
    // Expect 4 categories: attn, ffn, norm, other.
    EXPECT_EQ_INT((int)cats.size(), 4);
    EXPECT_EQ_STR(cats[0].category, "attn");
    EXPECT_EQ_STR(cats[1].category, "ffn");
    EXPECT_EQ_STR(cats[2].category, "norm");
    EXPECT_NEAR(cats[2].us, 10.0, 1e-6); // attn_norm + ffn_norm
    EXPECT_EQ_STR(cats[3].category, "other");
    EXPECT_NEAR(cats[3].us, 1.0, 1e-6);
}

static void test_heat_status() {
    EXPECT_EQ_STR(heatmap_heat_status(8, 8, false), "full");
    EXPECT_EQ_STR(heatmap_heat_status(8, 8, true), "full");
    EXPECT_EQ_STR(heatmap_heat_status(3, 8, false), "partial");
    EXPECT_EQ_STR(heatmap_heat_status(0, 8, false), "none");
    EXPECT_EQ_STR(heatmap_heat_status(0, 8, true), "estimated");
    EXPECT_EQ_STR(heatmap_heat_status(0, 0, false), "none");
}

int main() {
    test_layer_index_from_name();
    test_op_category_from_op();
    test_parse_sched_trace_node_timings();
    test_parse_server_telemetry_node_timings();
    test_merge_node_timings();
    test_layer_rollup_full();
    test_layer_rollup_partial();
    test_op_categories_with_norm();
    test_heat_status();

    if (g_fails == 0) {
        std::fprintf(stderr, "ALL HEATMAP ROLLUP TESTS PASSED\n");
        return 0;
    }
    std::fprintf(stderr, "%d HEATMAP ROLLUP TEST(S) FAILED\n", g_fails);
    return 1;
}
