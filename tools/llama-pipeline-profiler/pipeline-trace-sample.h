#pragma once

#include <string>

struct pipeline_trace_sample_options {
    int every     = 10;
    int max_lines = 0;
};

bool pipeline_trace_sample_run(
        const std::string & telemetry_dir,
        const pipeline_trace_sample_options & opts);