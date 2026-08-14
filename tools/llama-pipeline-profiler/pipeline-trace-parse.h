#pragma once

#include <string>

// Build trace-summary.txt from sched/rpc jsonl in telemetry_dir.
bool pipeline_trace_parse(const std::string & telemetry_dir);