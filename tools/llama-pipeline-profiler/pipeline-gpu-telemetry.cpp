#include "pipeline-gpu-telemetry.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <thread>

#ifdef _WIN32
#include <io.h>
#define popen  _popen
#define pclose _pclose
#else
#include <unistd.h>
#endif

namespace {

bool command_exists(const char * cmd) {
#ifdef _WIN32
    const char * suffix = " 1>nul 2>nul";
#else
    const char * suffix = " >/dev/null 2>&1";
#endif
    const std::string probe = std::string(cmd) + " --version" + suffix;
    return std::system(probe.c_str()) == 0;
}

void write_schema(const fs::path & gpu_dir) {
    std::ofstream out(gpu_dir / "schema.json");
    out << R"({
  "version": 1,
  "nvidia_csv": {
    "file_pattern": "nvidia-*.csv",
    "columns": [
      "timestamp_utc", "power_w", "util_gpu_pct", "util_mem_pct",
      "sm_clock_mhz", "mem_clock_mhz", "mem_used_mib", "mem_total_mib",
      "pcie_rx_mbs", "pcie_tx_mbs"
    ],
    "interval_ms": 500
  },
  "tdp_w": { "5070": 300, "5060": 175, "6600": 140 }
}
)";
}

std::string utc_timestamp() {
    const auto now = std::chrono::system_clock::now();
    const auto t   = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

} // namespace

struct pipeline_gpu_collector::impl {
    std::thread       worker;
    std::atomic<bool> stop{false};
    std::atomic<bool> running{false};

    void collect_loop(const fs::path csv_path, int duration_sec, int interval_ms) {
        std::ofstream csv(csv_path);
        if (!csv) {
            fprintf(stderr, "gpu-telemetry: cannot write %s\n", csv_path.string().c_str());
            return;
        }
        csv << "timestamp_utc,power_w,util_gpu_pct,util_mem_pct,sm_clock_mhz,mem_clock_mhz,"
               "mem_used_mib,mem_total_mib,pcie_rx_mbs,pcie_tx_mbs\n";

        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(duration_sec);
        while (!stop.load() && std::chrono::steady_clock::now() < deadline) {
#ifdef _WIN32
            const char * smi_cmd =
                "nvidia-smi --query-gpu=power.draw,utilization.gpu,utilization.memory,"
                "clocks.sm,clocks.mem,memory.used,memory.total "
                "--format=csv,noheader,nounits 2>nul";
#else
            const char * smi_cmd =
                "nvidia-smi --query-gpu=power.draw,utilization.gpu,utilization.memory,"
                "clocks.sm,clocks.mem,memory.used,memory.total "
                "--format=csv,noheader,nounits 2>/dev/null";
#endif
            FILE * pipe = popen(smi_cmd, "r");
            if (pipe) {
                char line[512];
                if (fgets(line, sizeof(line), pipe)) {
                    std::string vals = line;
                    while (!vals.empty() && (vals.back() == '\n' || vals.back() == '\r')) {
                        vals.pop_back();
                    }
                    vals.erase(std::remove(vals.begin(), vals.end(), ' '), vals.end());
                    if (!vals.empty()) {
                        csv << utc_timestamp() << "," << vals << ",0,0\n";
                    }
                }
                pclose(pipe);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
        }
        fprintf(stderr, "gpu-telemetry done -> %s\n", csv_path.string().c_str());
    }
};

pipeline_gpu_collector::pipeline_gpu_collector() : pimpl(std::make_unique<impl>()) {}

pipeline_gpu_collector::~pipeline_gpu_collector() {
    stop();
}

void pipeline_gpu_collector::start(
        const fs::path & gpu_dir,
        int duration_sec,
        const pipeline_gpu_telemetry_options & opts) {
    stop();
    if (!command_exists("nvidia-smi")) {
        fprintf(stderr, "gpu-telemetry: nvidia-smi not found, skipping local collect\n");
        return;
    }
    fs::create_directories(gpu_dir);
    write_schema(gpu_dir);
    const fs::path csv_path = gpu_dir / "nvidia-local.csv";
    pimpl->stop.store(false);
    pimpl->running.store(true);
    fprintf(stderr, ">>> gpu-telemetry: %s duration=%ds\n", gpu_dir.string().c_str(), duration_sec);
    pimpl->worker = std::thread([this, csv_path, duration_sec, opts]() {
        pimpl->collect_loop(csv_path, duration_sec, opts.interval_ms);
        pimpl->running.store(false);
    });
}

void pipeline_gpu_collector::stop() {
    pimpl->stop.store(true);
    if (pimpl->worker.joinable()) {
        pimpl->worker.join();
    }
    pimpl->running.store(false);
}