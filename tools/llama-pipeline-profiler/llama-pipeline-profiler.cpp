#include "build-info.h"
#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "pipeline-diagnose.h"
#include "pipeline-gpu-telemetry.h"
#include "pipeline-regression.h"
#include "pipeline-rpc-validate.h"
#include "pipeline-trace-sample.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace fs = std::filesystem;

static uint64_t time_ns() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::nanoseconds(clock::now().time_since_epoch()).count();
}

enum profiler_mode {
    MODE_THROUGHPUT,
    MODE_TRACE,
    MODE_PROFILE,
    MODE_AB_PLUS,
    MODE_SPIKE_CHECK,
    MODE_TRACE_OBSERVER,
};

enum output_format {
    OUTPUT_NONE,
    OUTPUT_JSONL,
};

struct profiler_config {
    std::string model_path;
    std::string rpc_endpoints;
    std::string tensor_split_str;
    std::string out_dir;
    std::string trace_dir;
    std::string mode_str = "throughput";
    profiler_mode mode   = MODE_THROUGHPUT;
    int n_gen            = 128;
    int n_prompt         = 0;
    std::string prompt_file;
    int n_gpu_layers     = 99;
    int n_batch          = 512;
    int n_ubatch         = 512;
    int n_threads        = 0;
    int reps             = 1;
    int ctx_size         = 4096;
    llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER;
    ggml_type type_k     = GGML_TYPE_Q4_0;
    ggml_type type_v     = GGML_TYPE_Q4_0;
    bool sync_per_token  = false;
    bool no_warmup       = false;
    bool with_gpu_telemetry = false;
    bool enable_trace    = false;
    int pipeline_plus    = 1;
    int overlap_target      = 5;
    int watchdog_sec        = 0;
    int trace_sample_every  = 0;
    output_format output_fmt = OUTPUT_JSONL;
    std::string regression_file;
    std::string diagnose_only_dir;
    bool validate_rpc_only   = false;
    bool skip_rpc_validate   = false;
    std::vector<float> tensor_split;
};

static void register_rpc_servers(const std::string & servers) {
    auto rpc_servers = string_split<std::string>(servers, ',');
    if (rpc_servers.empty()) {
        throw std::invalid_argument("no RPC servers specified");
    }
    auto * rpc_reg = ggml_backend_reg_by_name("RPC");
    if (!rpc_reg) {
        throw std::invalid_argument("failed to find RPC backend");
    }
    using add_rpc_server_fn = ggml_backend_reg_t (*)(const char * endpoint);
    auto * add_fn = (add_rpc_server_fn) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_add_server");
    if (!add_fn) {
        throw std::invalid_argument("failed to find RPC add server function");
    }
    for (const auto & server : rpc_servers) {
        auto reg = add_fn(server.c_str());
        ggml_backend_register(reg);
    }
}

static std::vector<float> parse_tensor_split(const std::string & ts, int n_max) {
    std::vector<float> out(n_max, 0.0f);
    auto parts = string_split<std::string>(ts, ',');
    for (size_t i = 0; i < parts.size() && i < (size_t) n_max; ++i) {
        out[i] = std::stof(parts[i]);
    }
    return out;
}

static void apply_tensor_split(profiler_config & cfg) {
    if (!cfg.tensor_split_str.empty()) {
        cfg.tensor_split = parse_tensor_split(cfg.tensor_split_str, llama_max_devices());
    }
}

static ggml_type parse_cache_type(const std::string & s) {
    if (s == "f16")  return GGML_TYPE_F16;
    if (s == "bf16") return GGML_TYPE_BF16;
    if (s == "q8_0") return GGML_TYPE_Q8_0;
    if (s == "q4_0") return GGML_TYPE_Q4_0;
    if (s == "q4_1") return GGML_TYPE_Q4_1;
    throw std::invalid_argument("unknown cache type: " + s);
}

static llama_split_mode parse_split_mode(const std::string & s) {
    if (s == "none")   return LLAMA_SPLIT_MODE_NONE;
    if (s == "layer")  return LLAMA_SPLIT_MODE_LAYER;
    if (s == "row")    return LLAMA_SPLIT_MODE_ROW;
    if (s == "tensor") return LLAMA_SPLIT_MODE_TENSOR;
    throw std::invalid_argument("unknown split mode: " + s);
}

static void setup_trace_env(const std::string & trace_dir, bool enable) {
    if (!enable || trace_dir.empty()) {
        return;
    }
    fs::create_directories(trace_dir);
    const std::string sched    = trace_dir + "/sched-trace.jsonl";
    const std::string rpc      = trace_dir + "/rpc-trace.jsonl";
    const std::string pipeline = trace_dir + "/pipeline-trace.jsonl";
#ifdef _WIN32
    _putenv_s("GGML_SCHED_TRACE", "1");
    _putenv_s("GGML_RPC_TRACE", "1");
    _putenv_s("GGML_PIPELINE_TRACE", "1");
    _putenv_s("GGML_SCHED_TRACE_FILE", sched.c_str());
    _putenv_s("GGML_RPC_TRACE_FILE", rpc.c_str());
    _putenv_s("GGML_PIPELINE_TRACE_FILE", pipeline.c_str());
#else
    setenv("GGML_SCHED_TRACE", "1", 1);
    setenv("GGML_RPC_TRACE", "1", 1);
    setenv("GGML_PIPELINE_TRACE", "1", 1);
    setenv("GGML_SCHED_TRACE_FILE", sched.c_str(), 1);
    setenv("GGML_RPC_TRACE_FILE", rpc.c_str(), 1);
    setenv("GGML_PIPELINE_TRACE_FILE", pipeline.c_str(), 1);
#endif
}

static void setup_plus_env(int plus) {
    const std::string v = std::to_string(plus);
#ifdef _WIN32
    _putenv_s("GGML_PIPELINE_PLUS", v.c_str());
#else
    setenv("GGML_PIPELINE_PLUS", v.c_str(), 1);
#endif
}

struct run_result {
    double avg_tps = 0.0;
    int    n_gen_tokens = 0;
    int    n_eval  = 0;
    int    n_reused = 0;
    uint64_t elapsed_ns = 0;
    int    rep     = 1;
};

static double compute_gen_tps(int n_gen_tokens, uint64_t elapsed_ns, int perf_n_eval, int perf_n_reused) {
    if (n_gen_tokens > 0 && elapsed_ns > 0) {
        if (perf_n_reused > 0 && perf_n_eval > 0 && perf_n_eval < n_gen_tokens / 2) {
            fprintf(stderr,
                "note: llama_perf n_eval=%d n_reused=%d; G uses wall n_gen=%d (not perf counter)\n",
                perf_n_eval, perf_n_reused, n_gen_tokens);
        }
        return 1e9 * n_gen_tokens / (double) elapsed_ns;
    }
    if (perf_n_eval > 0 && elapsed_ns > 0) {
        return 1e9 * perf_n_eval / (double) elapsed_ns;
    }
    return 0.0;
}

static bool run_prompt(llama_context * ctx, const std::vector<llama_token> & tokens, int n_batch) {
    if (tokens.empty()) {
        return true;
    }
    int n_processed = 0;
    while (n_processed < (int) tokens.size()) {
        const int n_tokens = std::min((int) tokens.size() - n_processed, n_batch);
        llama_token * batch_ptr = const_cast<llama_token *>(tokens.data()) + n_processed;
        if (llama_decode(ctx, llama_batch_get_one(batch_ptr, n_tokens)) != 0) {
            return false;
        }
        n_processed += n_tokens;
    }
    llama_synchronize(ctx);
    return true;
}

static bool run_gen(llama_context * ctx, int n_gen, bool sync_per_token) {
    const llama_model * model   = llama_get_model(ctx);
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);
    llama_token token = llama_vocab_get_add_bos(vocab) ? llama_vocab_bos(vocab) : 0;
    for (int i = 0; i < n_gen; ++i) {
        if (llama_decode(ctx, llama_batch_get_one(&token, 1)) != 0) {
            return false;
        }
        if (sync_per_token) {
            llama_synchronize(ctx);
        }
        token = (llama_token) (i + 1) % n_vocab;
    }
    if (!sync_per_token) {
        llama_synchronize(ctx);
    }
    return true;
}

static run_result run_session(const profiler_config & cfg, int plus_val) {
    setup_plus_env(plus_val);
    setup_trace_env(cfg.trace_dir, cfg.enable_trace);

    if (!cfg.rpc_endpoints.empty()) {
        register_rpc_servers(cfg.rpc_endpoints);
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = cfg.n_gpu_layers;
    mparams.split_mode   = cfg.split_mode;
    if (!cfg.tensor_split.empty()) {
        mparams.tensor_split = cfg.tensor_split.data();
    }

    llama_model * model = llama_model_load_from_file(cfg.model_path.c_str(), mparams);
    if (!model) {
        throw std::runtime_error("failed to load model: " + cfg.model_path);
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_batch  = cfg.n_batch;
    cparams.n_ubatch = cfg.n_ubatch;
    cparams.type_k   = cfg.type_k;
    cparams.type_v   = cfg.type_v;
    cparams.offload_kqv = true;

    const llama_vocab * vocab_pre = llama_model_get_vocab(model);
    std::vector<llama_token> prompt_tokens;
    if (!cfg.prompt_file.empty()) {
        std::ifstream in(cfg.prompt_file);
        if (!in) {
            llama_model_free(model);
            throw std::runtime_error("failed to open prompt file: " + cfg.prompt_file);
        }
        std::ostringstream ss;
        ss << in.rdbuf();
        prompt_tokens = common_tokenize(vocab_pre, ss.str(), true, true);
        if (prompt_tokens.empty()) {
            llama_model_free(model);
            throw std::runtime_error("prompt file tokenized to empty sequence: " + cfg.prompt_file);
        }
    } else if (cfg.n_prompt > 0) {
        prompt_tokens.reserve((size_t) cfg.n_prompt);
        for (int i = 0; i < cfg.n_prompt; ++i) {
            if (i == 0 && llama_vocab_get_add_bos(vocab_pre)) {
                prompt_tokens.push_back(llama_vocab_bos(vocab_pre));
            } else {
                prompt_tokens.push_back((llama_token) (i % llama_vocab_n_tokens(vocab_pre)));
            }
        }
    }
    const int prompt_len = (int) prompt_tokens.size();
    cparams.n_ctx        = std::max(cfg.ctx_size, prompt_len + cfg.n_gen + 32);

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        llama_model_free(model);
        throw std::runtime_error("failed to create context");
    }

    const int n_threads = cfg.n_threads > 0 ? cfg.n_threads : common_cpu_get_num_math();
    llama_set_n_threads(ctx, n_threads, n_threads);
    llama_perf_context_reset(ctx);
    llama_memory_clear(llama_get_memory(ctx), false);

    if (!cfg.no_warmup) {
        run_gen(ctx, 1, true);
        llama_perf_context_reset(ctx);
        llama_memory_clear(llama_get_memory(ctx), false);
    }

    if (!prompt_tokens.empty()) {
        fprintf(stderr, ">>> prefill tokens=%d\n", prompt_len);
        if (!run_prompt(ctx, prompt_tokens, cfg.n_batch)) {
            llama_free(ctx);
            llama_model_free(model);
            throw std::runtime_error("prefill failed");
        }
        llama_perf_context_reset(ctx);
    }

    const uint64_t t0 = time_ns();
    if (!run_gen(ctx, cfg.n_gen, cfg.sync_per_token)) {
        llama_free(ctx);
        llama_model_free(model);
        throw std::runtime_error("generation failed");
    }
    const uint64_t elapsed = time_ns() - t0;

    const auto perf = llama_perf_context(ctx);
    run_result res;
    res.elapsed_ns    = elapsed;
    res.n_gen_tokens  = cfg.n_gen;
    res.n_eval        = perf.n_eval;
    res.n_reused      = perf.n_reused;
    res.avg_tps       = compute_gen_tps(cfg.n_gen, elapsed, perf.n_eval, perf.n_reused);

    llama_free(ctx);
    llama_model_free(model);
    return res;
}

static void write_env_txt(const fs::path & dir, const profiler_config & cfg, const std::string & label, int plus) {
    std::ofstream f(dir / "env.txt");
    f << "LABEL=" << label << "\n";
    f << "GIT_SHA=" << llama_commit() << "\n";
    f << "client_kind=native\n";
    f << "MODE=" << cfg.mode_str << "\n";
    f << "RPC=" << cfg.rpc_endpoints << "\n";
    f << "TS=" << cfg.tensor_split_str << "\n";
    f << "GGML_PIPELINE_PLUS=" << plus << "\n";
    f << "N_GEN=" << cfg.n_gen << "\n";
    f << "N_PROMPT=" << (cfg.prompt_file.empty() ? std::to_string(cfg.n_prompt) : cfg.prompt_file) << "\n";
    f << "SYNC_PER_TOKEN=" << (cfg.sync_per_token ? 1 : 0) << "\n";
    f << "TRACE=" << (cfg.enable_trace ? 1 : 0) << "\n";
    f << "TRACE_DIR=" << cfg.trace_dir << "\n";
}

static void write_result_jsonl(const fs::path & dir, const run_result & res, int plus, int reps) {
    std::ofstream f(dir / "result.jsonl", std::ios::app);
    f << "{"
      << "\"avg_ts\":" << res.avg_tps << ","
      << "\"G_tps\":" << res.avg_tps << ","
      << "\"g_method\":\"wall_n_gen\","
      << "\"n_gen_tokens\":" << res.n_gen_tokens << ","
      << "\"n_eval\":" << res.n_eval << ","
      << "\"n_reused\":" << res.n_reused << ","
      << "\"elapsed_ns\":" << res.elapsed_ns << ","
      << "\"rep\":" << res.rep << ","
      << "\"reps\":" << reps << ","
      << "\"GGML_PIPELINE_PLUS\":" << plus
      << "}\n";
}

static void write_summary_md(const fs::path & dir, const run_result & res, const profiler_config & cfg, int plus) {
    std::ofstream f(dir / "summary.md");
    f << "# llama-pipeline-profiler run\n\n";
    f << "| Field | Value |\n|-------|-------|\n";
    f << "| G (t/s) | " << res.avg_tps << " |\n";
    f << "| n_gen | " << res.n_gen_tokens << " |\n";
    f << "| n_eval | " << res.n_eval << " |\n";
    f << "| n_reused | " << res.n_reused << " |\n";
    f << "| GGML_PIPELINE_PLUS | " << plus << " |\n";
    f << "| rpc | " << cfg.rpc_endpoints << " |\n";
    f << "| trace | " << (cfg.enable_trace ? "yes" : "no") << " |\n";
}

static void emit_stdout_jsonl(
        const profiler_config & cfg,
        const std::string & label,
        int plus,
        const run_result & res) {
    if (cfg.output_fmt != OUTPUT_JSONL) {
        return;
    }
    std::printf(
        "{\"label\":\"%s\",\"mode\":\"%s\",\"GGML_PIPELINE_PLUS\":%d,"
        "\"rep\":%d,\"reps\":%d,"
        "\"avg_ts\":%.4f,\"G_tps\":%.4f,\"g_method\":\"wall_n_gen\","
        "\"n_gen_tokens\":%d,\"n_eval\":%d,\"n_reused\":%d,"
        "\"elapsed_ns\":%llu,\"rpc\":\"%s\",\"tensor_split\":\"%s\"}\n",
        label.c_str(),
        cfg.mode_str.c_str(),
        plus,
        res.rep,
        cfg.reps,
        res.avg_tps,
        res.avg_tps,
        res.n_gen_tokens,
        res.n_eval,
        res.n_reused,
        (unsigned long long) res.elapsed_ns,
        cfg.rpc_endpoints.c_str(),
        cfg.tensor_split_str.c_str());
}

static bool path_looks_like_repo_root(const fs::path & p) {
    return fs::exists(p / "benches" / "path-b-plus") || fs::exists(p / ".git");
}

static std::string repo_root_from(const fs::path & seed) {
    fs::path p = fs::absolute(seed);
    for (int i = 0; i < 12 && !p.empty(); ++i) {
        if (path_looks_like_repo_root(p)) {
            return p.string();
        }
        const fs::path parent = p.parent_path();
        if (parent == p) {
            break;
        }
        p = parent;
    }
    return {};
}

static std::string repo_root() {
    std::vector<fs::path> seeds;
    seeds.push_back(fs::current_path());
#ifdef _WIN32
    wchar_t buf[MAX_PATH];
    const DWORD len = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (len > 0) {
        seeds.push_back(fs::path(buf).parent_path());
    }
#endif
    for (const auto & seed : seeds) {
        const std::string found = repo_root_from(seed);
        if (!found.empty()) {
            return found;
        }
    }
    return fs::current_path().string();
}

static void run_diagnose(const profiler_config & cfg) {
    if (!cfg.enable_trace || cfg.trace_dir.empty()) {
        return;
    }
    fprintf(stderr, ">>> diagnose: %s\n", cfg.trace_dir.c_str());
    pipeline_diagnose_options opts;
    opts.gen_only           = true;
    opts.overlap_target_pct = cfg.overlap_target;
    pipeline_diagnose_result result;
    pipeline_diagnose_run(cfg.trace_dir, opts, result);
}

static void run_trace_sample(const profiler_config & cfg) {
    if (cfg.trace_sample_every <= 0 || cfg.trace_dir.empty()) {
        return;
    }
    fprintf(stderr, ">>> trace-sample: %s every=%d\n", cfg.trace_dir.c_str(), cfg.trace_sample_every);
    pipeline_trace_sample_options opts;
    opts.every = cfg.trace_sample_every;
    pipeline_trace_sample_run(cfg.trace_dir, opts);
}

static void run_regression_append(
        const profiler_config & cfg,
        const std::string & label,
        int plus) {
    if (cfg.regression_file.empty() || cfg.trace_dir.empty()) {
        return;
    }
    const fs::path diag = fs::path(cfg.trace_dir) / "diagnose.json";
    if (!fs::exists(diag)) {
        return;
    }
    fprintf(stderr, ">>> regression: label=%s\n", label.c_str());
    pipeline_regression_options opts;
    opts.label            = label;
    opts.regression_file  = cfg.regression_file;
    opts.out_dir          = cfg.out_dir;
    opts.rpc_endpoints    = cfg.rpc_endpoints;
    opts.mode             = cfg.mode_str;
    opts.plus             = plus;
    opts.client_kind      = "native";
    opts.git_sha          = llama_commit();
    pipeline_regression_append(cfg.trace_dir, opts);
}

static void write_trace_observer_json(
        const fs::path & dir,
        double g_throughput,
        double g_trace,
        int n_reused_trace) {
    std::ofstream f(dir / "trace-observer.json");
    double delta_pct = 0.0;
    if (g_throughput > 0.0) {
        delta_pct = 100.0 * (g_trace / g_throughput - 1.0);
    }
    f << "{"
      << "\"g_throughput\":" << g_throughput << ","
      << "\"g_trace\":" << g_trace << ","
      << "\"delta_pct\":" << delta_pct << ","
      << "\"n_reused_trace\":" << n_reused_trace
      << "}\n";
    std::ofstream md(dir / "trace-observer.md");
    md << "# trace observer\n\n";
    md << "| Mode | G (t/s) |\n|------|--------|\n";
    md << "| throughput | " << g_throughput << " |\n";
    md << "| trace | " << g_trace << " |\n\n";
    md << "Delta (trace vs throughput): " << delta_pct << "%\n";
}

static bool run_rpc_validate(const profiler_config & cfg) {
    if (cfg.rpc_endpoints.empty()) {
        fprintf(stderr, "error: --validate-rpc requires -rpc\n");
        return false;
    }
    pipeline_rpc_validate_result result;
    return pipeline_rpc_validate(cfg.rpc_endpoints, cfg.tensor_split_str, result);
}

static void usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s -m <model.gguf> [options]\n"
        "  --mode <throughput|trace|profile|ab-plus|spike-check|trace-observer>\n"
        "  --trace-dir <path>       enable GGML sched/rpc trace\n"
        "  --out-dir <path>         artifacts (default: ./profiler-out)\n"
        "  -o, --output <jsonl|none> stdout format (default: jsonl)\n"
        "  -rpc, --rpc <endpoints>  comma-separated RPC servers\n"
        "  -ts, --tensor-split <ts> e.g. 50,50\n"
        "  -ngl, --n-gpu-layers <n> default 99\n"
        "  -sm, --split-mode <none|layer|row|tensor> default layer\n"
        "  -ctk, --cache-type-k <t>  default q4_0\n"
        "  -ctv, --cache-type-v <t>  default q4_0\n"
        "  -r, --repetitions <n>     default 1\n"
        "  -p, --n-prompt <n>        synthetic prefill tokens (ignored if -f set)\n"
        "  -f, --file <path>         prefill from prompt text file\n"
        "  -n, --n-gen <n>           default 128\n"
        "  -b, --batch-size <n>      default 512\n"
        "  -ub, --ubatch-size <n>    default 512\n"
        "  -t, --threads <n>         default: auto\n"
        "  --sync-per-token          llama-bench style (legacy compare)\n"
        "  --trace                   enable GGML sched/rpc trace (default dir: <out>/telemetry)\n"
        "  --with-gpu-telemetry      local nvidia-smi sampler during gen (C++)\n"
        "  --overlap-target <pct>    B+6 gate (default 5)\n"
        "  --trace-sample <n>        downsample trace jsonl every n lines (0=off)\n"
        "  --regression-file <path>  append diagnose row (default: benches/path-b-plus/regression.jsonl)\n"
        "  --diagnose-only <dir>     offline diagnose on existing telemetry (no model run)\n"
        "  --validate-rpc            probe -rpc endpoints and exit (R5 preflight)\n"
        "  --skip-rpc-validate       skip automatic RPC probe before model load\n"
        "  --no-warmup\n"
        "  -h, --help\n",
        argv0);
}

static profiler_mode parse_mode(const std::string & s) {
    if (s == "trace")        return MODE_TRACE;
    if (s == "profile")      return MODE_PROFILE;
    if (s == "ab-plus")      return MODE_AB_PLUS;
    if (s == "spike-check")     return MODE_SPIKE_CHECK;
    if (s == "trace-observer")  return MODE_TRACE_OBSERVER;
    return MODE_THROUGHPUT;
}

int llama_pipeline_profiler(int argc, char ** argv) {
    profiler_config cfg;
    cfg.tensor_split.assign(llama_max_devices(), 0.0f);
    cfg.out_dir = "profiler-out";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need = [&](const char * flag) {
            if (++i >= argc) {
                die_fmt("missing value for %s", flag);
            }
            return std::string(argv[i]);
        };
        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            return 0;
        } else if (arg == "-m" || arg == "--model") {
            cfg.model_path = need(arg.c_str());
        } else if (arg == "-o" || arg == "--output") {
            const auto fmt = need(arg.c_str());
            if (fmt == "jsonl") {
                cfg.output_fmt = OUTPUT_JSONL;
            } else if (fmt == "none") {
                cfg.output_fmt = OUTPUT_NONE;
            } else {
                die_fmt("unknown output format: %s", fmt.c_str());
            }
        } else if (arg == "--mode") {
            cfg.mode_str = need(arg.c_str());
            cfg.mode     = parse_mode(cfg.mode_str);
        } else if (arg == "--trace-dir") {
            cfg.trace_dir = need(arg.c_str());
        } else if (arg == "--out-dir") {
            cfg.out_dir = need(arg.c_str());
        } else if (arg == "-rpc" || arg == "--rpc") {
            cfg.rpc_endpoints = need(arg.c_str());
        } else if (arg == "-ts" || arg == "--tensor-split") {
            cfg.tensor_split_str = need(arg.c_str());
        } else if (arg == "-ngl" || arg == "--n-gpu-layers") {
            cfg.n_gpu_layers = std::stoi(need(arg.c_str()));
        } else if (arg == "-sm" || arg == "--split-mode") {
            cfg.split_mode = parse_split_mode(need(arg.c_str()));
        } else if (arg == "-ctk" || arg == "--cache-type-k") {
            cfg.type_k = parse_cache_type(need(arg.c_str()));
        } else if (arg == "-ctv" || arg == "--cache-type-v") {
            cfg.type_v = parse_cache_type(need(arg.c_str()));
        } else if (arg == "-r" || arg == "--repetitions") {
            cfg.reps = std::stoi(need(arg.c_str()));
        } else if (arg == "-p" || arg == "--n-prompt") {
            cfg.n_prompt = std::stoi(need(arg.c_str()));
        } else if (arg == "-f" || arg == "--file") {
            cfg.prompt_file = need(arg.c_str());
        } else if (arg == "-n" || arg == "--n-gen") {
            cfg.n_gen = std::stoi(need(arg.c_str()));
        } else if (arg == "-b" || arg == "--batch-size") {
            cfg.n_batch = std::stoi(need(arg.c_str()));
        } else if (arg == "-ub" || arg == "--ubatch-size") {
            cfg.n_ubatch = std::stoi(need(arg.c_str()));
        } else if (arg == "-t" || arg == "--threads") {
            cfg.n_threads = std::stoi(need(arg.c_str()));
        } else if (arg == "--sync-per-token") {
            cfg.sync_per_token = true;
        } else if (arg == "--trace") {
            cfg.enable_trace = true;
        } else if (arg == "--with-gpu-telemetry") {
            cfg.with_gpu_telemetry = true;
        } else if (arg == "--overlap-target") {
            cfg.overlap_target = std::stoi(need(arg.c_str()));
        } else if (arg == "--trace-sample") {
            cfg.trace_sample_every = std::stoi(need(arg.c_str()));
        } else if (arg == "--regression-file") {
            cfg.regression_file = need(arg.c_str());
        } else if (arg == "--diagnose-only") {
            cfg.diagnose_only_dir = need(arg.c_str());
        } else if (arg == "--validate-rpc") {
            cfg.validate_rpc_only = true;
        } else if (arg == "--skip-rpc-validate") {
            cfg.skip_rpc_validate = true;
        } else if (arg == "--no-warmup") {
            cfg.no_warmup = true;
        } else {
            fprintf(stderr, "error: unknown arg %s\n", arg.c_str());
            usage(argv[0]);
            return 1;
        }
    }

    apply_tensor_split(cfg);

    if (cfg.validate_rpc_only) {
        llama_backend_init();
        ggml_backend_load_all();
        const bool ok = run_rpc_validate(cfg);
        llama_backend_free();
        return ok ? 0 : 1;
    }

    if (!cfg.diagnose_only_dir.empty()) {
        pipeline_diagnose_options dopts;
        dopts.gen_only           = true;
        dopts.overlap_target_pct = cfg.overlap_target;
        pipeline_diagnose_result result;
        if (!pipeline_diagnose_run(cfg.diagnose_only_dir, dopts, result)) {
            fprintf(stderr, "error: diagnose failed for %s\n", cfg.diagnose_only_dir.c_str());
            return 1;
        }
        if (cfg.trace_sample_every > 0) {
            pipeline_trace_sample_options sopts;
            sopts.every = cfg.trace_sample_every;
            pipeline_trace_sample_run(cfg.diagnose_only_dir, sopts);
        }
        return 0;
    }

    if (cfg.model_path.empty()) {
        usage(argv[0]);
        return 1;
    }

    if (cfg.mode == MODE_TRACE || cfg.mode == MODE_PROFILE || cfg.mode == MODE_SPIKE_CHECK) {
        cfg.enable_trace = true;
    }
    if (cfg.trace_dir.empty() && cfg.enable_trace) {
        cfg.trace_dir = cfg.out_dir + "/telemetry";
    }
    if (cfg.mode == MODE_PROFILE) {
        cfg.with_gpu_telemetry = true;
        if (cfg.trace_sample_every <= 0) {
            cfg.trace_sample_every = 10;
        }
    }
    if (cfg.regression_file.empty()) {
        cfg.regression_file = repo_root() + "/benches/path-b-plus/regression.jsonl";
    }

    fs::create_directories(cfg.out_dir);
    if (cfg.enable_trace) {
        fs::create_directories(cfg.trace_dir);
        setup_trace_env(cfg.trace_dir, true);
    }

    llama_backend_init();
    ggml_backend_load_all();

    if (!cfg.rpc_endpoints.empty() && !cfg.skip_rpc_validate) {
        if (!run_rpc_validate(cfg)) {
            fprintf(stderr, "error: RPC preflight failed (use --skip-rpc-validate to override)\n");
            llama_backend_free();
            return 1;
        }
    }

    const auto run_cell = [&](const profiler_config & cell, const std::string & label, int plus) -> run_result {
        fprintf(stderr, ">>> run label=%s GGML_PIPELINE_PLUS=%d reps=%d\n", label.c_str(), plus, cell.reps);
        if (cell.enable_trace && !cell.trace_dir.empty()) {
            fs::create_directories(cell.trace_dir);
        }

        std::unique_ptr<pipeline_gpu_collector> gpu;
        if (cell.with_gpu_telemetry) {
            const fs::path gpu_root = cell.trace_dir.empty()
                ? fs::path(cell.out_dir) / "gpu"
                : fs::path(cell.trace_dir) / "gpu";
            gpu = std::make_unique<pipeline_gpu_collector>();
            pipeline_gpu_telemetry_options gopts;
            gpu->start(gpu_root, cell.n_gen / 2 + 30, gopts);
        }

        const int reps = std::max(1, cell.reps);
        run_result last;
        double tps_sum = 0.0;

        for (int rep = 1; rep <= reps; ++rep) {
            profiler_config rep_cfg = cell;
            rep_cfg.reps = reps;
            if (cell.enable_trace && rep < reps) {
                rep_cfg.enable_trace = false;
            }
            fprintf(stderr, ">>> rep %d/%d\n", rep, reps);
            auto res     = run_session(rep_cfg, plus);
            res.rep      = rep;
            last         = res;
            tps_sum     += res.avg_tps;
            write_result_jsonl(fs::path(cell.out_dir), res, plus, reps);
            emit_stdout_jsonl(rep_cfg, label, plus, res);
            fprintf(stderr, ">>> G=%.2f t/s n_gen=%d n_reused=%d (rep %d)\n",
                    res.avg_tps, res.n_gen_tokens, res.n_reused, rep);
        }

        if (gpu) {
            gpu->stop();
        }

        last.avg_tps = tps_sum / reps;
        write_env_txt(fs::path(cell.out_dir), cell, label, plus);
        write_summary_md(fs::path(cell.out_dir), last, cell, plus);
        fprintf(stderr, ">>> mean G=%.2f t/s over %d reps\n", last.avg_tps, reps);

        if (cell.enable_trace) {
            run_diagnose(cell);
            run_trace_sample(cell);
            run_regression_append(cell, label, plus);
        }
        return last;
    };

    try {
        if (cfg.mode == MODE_TRACE_OBSERVER) {
            const std::string base = cfg.out_dir;
            profiler_config tp_cfg   = cfg;
            tp_cfg.enable_trace      = false;
            tp_cfg.trace_dir.clear();
            tp_cfg.mode_str          = "throughput";
            tp_cfg.out_dir           = base + "/throughput";
            fs::create_directories(tp_cfg.out_dir);
            auto res_tp = run_cell(tp_cfg, "throughput", cfg.pipeline_plus);

            profiler_config tr_cfg  = cfg;
            tr_cfg.enable_trace     = true;
            tr_cfg.out_dir          = base + "/trace";
            tr_cfg.trace_dir        = tr_cfg.out_dir + "/telemetry";
            tr_cfg.mode_str         = "trace";
            fs::create_directories(tr_cfg.out_dir);
            if (tr_cfg.trace_sample_every <= 0) {
                tr_cfg.trace_sample_every = cfg.trace_sample_every > 0 ? cfg.trace_sample_every : 10;
            }
            auto res_tr = run_cell(tr_cfg, "trace", cfg.pipeline_plus);
            write_trace_observer_json(fs::path(base), res_tp.avg_tps, res_tr.avg_tps, res_tr.n_reused);
            cfg.out_dir = base;
        } else if (cfg.mode == MODE_AB_PLUS) {
            const bool trace_ab = cfg.enable_trace;
            const std::string base = cfg.out_dir;
            profiler_config on_cfg = cfg;
            on_cfg.enable_trace = trace_ab;
            on_cfg.out_dir      = base + "/plus-on";
            on_cfg.trace_dir    = on_cfg.out_dir + "/telemetry";
            fs::create_directories(on_cfg.out_dir);
            run_cell(on_cfg, "plus-on", 1);

            profiler_config off_cfg = cfg;
            off_cfg.enable_trace = trace_ab;
            off_cfg.out_dir      = base + "/plus-off";
            off_cfg.trace_dir    = off_cfg.out_dir + "/telemetry";
            fs::create_directories(off_cfg.out_dir);
            run_cell(off_cfg, "plus-off", 0);
            cfg.out_dir = base;
        } else {
            run_cell(cfg, cfg.mode_str, cfg.pipeline_plus);
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        llama_backend_free();
        return 1;
    }

    llama_backend_free();
    fprintf(stderr, "llama-pipeline-profiler done out=%s\n", cfg.out_dir.c_str());
    return 0;
}