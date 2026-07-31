#pragma once

#include <nlohmann/json.hpp>

#include <string>

struct pipeline_diagnose_options {
    bool        gen_only            = true;
    double      overlap_target_pct  = 5.0;
    std::string baseline_dir;
    int         gen_tokens_override = 0;
};

struct pipeline_diagnose_result {
    nlohmann::ordered_json diagnose;
    std::string          human_summary;
};

// Parse traces (if needed), compute diagnose.json fields, write artifacts.
bool pipeline_diagnose_run(
        const std::string & telemetry_dir,
        const pipeline_diagnose_options & opts,
        pipeline_diagnose_result & out);