// Unit tests for placement capacity discovery (P0).
// Seams: backend_id, usable_weight, batch expand, topology_complete, gate, JSON fields.

#include "placement-capacity.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static void test_backend_ids() {
    EXPECT_EQ_STR(placement_make_local_backend_id("ROCm0", ""), "local:ROCm0");
    EXPECT_EQ_STR(placement_make_local_backend_id("CUDA0", "0000:0a:00.0"), "local:pci:0000:0a:00.0");
    EXPECT_EQ_STR(placement_make_rpc_backend_id("192.168.1.10:50051", 0), "rpc://192.168.1.10:50051#0");
    EXPECT_EQ_STR(placement_make_rpc_backend_id("host:9", 1), "rpc://host:9#1");
    // not bare RPC0
    const std::string id = placement_make_rpc_backend_id("e:1", 0);
    EXPECT_TRUE(id.find("RPC0") == std::string::npos);
    EXPECT_TRUE(id.rfind("rpc://", 0) == 0);
}

static void test_usable_weight() {
    placement_static_pads pads;
    pads.fragmentation_mib = 100;
    pads.runtime_overhead_mib = 50;
    pads.process_reserve_mib = 50;
    EXPECT_EQ_U64(placement_static_pads_sum_mib(pads), 200ull);

    // free 1000, pads 200, reserves 300 -> usable 500
    EXPECT_EQ_U64(placement_usable_weight_mib(1000, pads, 300), 500ull);
    // clamp at zero
    EXPECT_EQ_U64(placement_usable_weight_mib(100, pads, 300), 0ull);

    placement_reserve_params rp;
    rp.n_ctx = 4096;
    rp.n_parallel = 1;
    const uint64_t table = placement_table_reserves_mib(rp, 24576);
    EXPECT_TRUE(table > 0);
    // small VRAM gets extra floor
    const uint64_t table_8g = placement_table_reserves_mib(rp, 8192);
    EXPECT_TRUE(table_8g >= table);
}

static void test_batch_expand_multi_device() {
    placement_reserve_params rp;
    rp.n_ctx = 2048;
    rp.n_parallel = 1;

    std::vector<placement_raw_device> batch;
    {
        placement_raw_device d;
        d.kind = PLACEMENT_KIND_RPC_DEVICE;
        d.endpoint = "node:50052";
        d.device_index = 0;
        d.free_bytes = 20ull * 1024 * 1024 * 1024;
        d.total_bytes = 24ull * 1024 * 1024 * 1024;
        d.legacy_memory_only = true;
        batch.push_back(d);
    }
    {
        placement_raw_device d;
        d.kind = PLACEMENT_KIND_RPC_DEVICE;
        d.endpoint = "node:50052";
        d.device_index = 1;
        d.free_bytes = 6ull * 1024 * 1024 * 1024;
        d.total_bytes = 8ull * 1024 * 1024 * 1024;
        d.legacy_memory_only = true;
        batch.push_back(d);
    }

    auto recs = placement_expand_rpc_endpoint_batch("node:50052", batch, rp, "2026-07-17T00:00:00Z");
    EXPECT_EQ_U64((uint64_t) recs.size(), 2ull);
    EXPECT_EQ_STR(recs[0].backend_id, "rpc://node:50052#0");
    EXPECT_EQ_STR(recs[1].backend_id, "rpc://node:50052#1");
    EXPECT_EQ_STR(placement_kind_str(recs[0].kind), "rpc_device");
    EXPECT_TRUE(recs[0].caps.multi_device);
    EXPECT_TRUE(recs[0].usable_weight_mib < recs[0].free_mib);
    EXPECT_EQ_STR(recs[0].reserve_mode, "table");
    EXPECT_EQ_STR(recs[0].reserve_model_version, PLACEMENT_RESERVE_MODEL_TABLE_V1);
    EXPECT_TRUE(!recs[0].warnings.empty()); // legacy warning
}

static void test_topology_incomplete() {
    placement_reserve_params rp;
    std::vector<placement_raw_device> devices;
    {
        placement_raw_device d;
        d.kind = PLACEMENT_KIND_LOCAL_GPU;
        d.name = "ROCm0";
        d.free_bytes = 20ull << 30;
        d.total_bytes = 24ull << 30;
        devices.push_back(d);
    }
    {
        placement_raw_device d;
        d.kind = PLACEMENT_KIND_RPC_DEVICE;
        d.endpoint = "alive:50051";
        d.device_index = 0;
        d.free_bytes = 7ull << 30;
        d.total_bytes = 8ull << 30;
        d.legacy_memory_only = true;
        devices.push_back(d);
    }

    std::vector<std::string> configured = {"alive:50051", "dead:50052"};
    auto inv = placement_build_inventory(devices, configured, rp, "2026-07-17T00:00:00Z");
    EXPECT_TRUE(!inv.topology_complete);
    EXPECT_TRUE(inv.discover_errors.size() == 1);
    EXPECT_EQ_STR(inv.discover_errors[0].target, "dead:50052");
    EXPECT_TRUE(inv.records.size() == 2);

    // complete when all configured present
    auto inv2 = placement_build_inventory(devices, {"alive:50051"}, rp, "2026-07-17T00:00:00Z");
    EXPECT_TRUE(inv2.topology_complete);
    EXPECT_TRUE(inv2.discover_errors.empty());
}

static void test_schema_json_fields() {
    placement_reserve_params rp;
    rp.n_ctx = 4096;
    std::vector<placement_raw_device> devices;
    {
        placement_raw_device d;
        d.kind = PLACEMENT_KIND_LOCAL_GPU;
        d.name = "ROCm0";
        d.device_id = "0000:03:00.0";
        d.free_bytes = 22ull << 30;
        d.total_bytes = 24ull << 30;
        d.static_pads.process_reserve_mib = 64;
        devices.push_back(d);
    }
    {
        placement_raw_device d;
        d.kind = PLACEMENT_KIND_RPC_DEVICE;
        d.endpoint = "192.168.1.20:50051";
        d.device_index = 0;
        d.free_bytes = 7ull << 30;
        d.total_bytes = 8ull << 30;
        d.legacy_memory_only = true;
        devices.push_back(d);
    }
    auto inv = placement_build_inventory(devices, {"192.168.1.20:50051"}, rp, "2026-07-17T12:00:00Z");
    const std::string js = placement_inventory_to_json(inv, 2);

    EXPECT_TRUE(js.find("\"schema_version\"") != std::string::npos);
    EXPECT_TRUE(js.find("\"backend_id\"") != std::string::npos);
    EXPECT_TRUE(js.find("local:pci:0000:03:00.0") != std::string::npos ||
                js.find("local:ROCm0") != std::string::npos);
    EXPECT_TRUE(js.find("rpc://192.168.1.20:50051#0") != std::string::npos);
    EXPECT_TRUE(js.find("\"local_gpu\"") != std::string::npos);
    EXPECT_TRUE(js.find("\"rpc_device\"") != std::string::npos);
    EXPECT_TRUE(js.find("\"total_mib\"") != std::string::npos);
    EXPECT_TRUE(js.find("\"free_mib\"") != std::string::npos);
    EXPECT_TRUE(js.find("\"static_pads\"") != std::string::npos);
    EXPECT_TRUE(js.find("\"caps\"") != std::string::npos);
    EXPECT_TRUE(js.find("\"reported_at\"") != std::string::npos);
    EXPECT_TRUE(js.find("\"usable_weight_mib\"") != std::string::npos);
    EXPECT_TRUE(js.find("\"reserve_mode\"") != std::string::npos);
    EXPECT_TRUE(js.find("\"reserve_model_version\"") != std::string::npos);
    EXPECT_TRUE(js.find("table-v1") != std::string::npos);
    EXPECT_TRUE(js.find("\"topology_complete\"") != std::string::npos);
    EXPECT_TRUE(js.find("legacy_n_get_device_memory") != std::string::npos);

    // usable < free when reserves apply
    EXPECT_TRUE(inv.records[0].usable_weight_mib < inv.records[0].free_mib);
}

static void test_discover_gate() {
    EXPECT_TRUE(!placement_discover_is_enabled(false, nullptr));
    EXPECT_TRUE(!placement_discover_is_enabled(false, ""));
    EXPECT_TRUE(!placement_discover_is_enabled(false, "0"));
    EXPECT_TRUE(!placement_discover_is_enabled(false, "false"));
    EXPECT_TRUE(placement_discover_is_enabled(true, nullptr));
    EXPECT_TRUE(placement_discover_is_enabled(false, "1"));
    EXPECT_TRUE(placement_discover_is_enabled(false, "true"));
    EXPECT_TRUE(placement_discover_is_enabled(false, "yes"));
}

int main() {
    test_backend_ids();
    test_usable_weight();
    test_batch_expand_multi_device();
    test_topology_incomplete();
    test_schema_json_fields();
    test_discover_gate();

    if (g_fails != 0) {
        std::fprintf(stderr, "%d test assertion(s) failed\n", g_fails);
        return 1;
    }
    std::printf("test-placement-capacity: all passed\n");
    return 0;
}
