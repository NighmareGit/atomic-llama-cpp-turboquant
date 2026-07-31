#pragma once

#include <string>

struct pipeline_regression_options {
    std::string label;
    std::string regression_file;
    std::string out_dir;
    std::string rpc_endpoints;
    std::string mode;
    int         plus        = 1;
    std::string client_kind = "native";
    std::string git_sha;
};

bool pipeline_regression_append(
        const std::string & telemetry_dir,
        const pipeline_regression_options & opts);