#pragma once

#include <filesystem>
#include <memory>

namespace fs = std::filesystem;

struct pipeline_gpu_telemetry_options {
    int interval_ms = 500;
};

// Background local NVIDIA sampler (nvidia-smi). No-op if nvidia-smi unavailable.
class pipeline_gpu_collector {
public:
    pipeline_gpu_collector();
    ~pipeline_gpu_collector();

    void start(const fs::path & gpu_dir, int duration_sec, const pipeline_gpu_telemetry_options & opts);
    void stop();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};