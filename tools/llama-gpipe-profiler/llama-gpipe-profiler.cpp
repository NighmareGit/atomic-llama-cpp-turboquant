// llama-gpipe-profiler: task-stratified multi-GPU profiler.
// Drives pp (prompt processing) and tg (token generation) tasks across
// local + RPC endpoints, captures client-side traces and optional server
// telemetry, and synthesizes a task-stratified heatmap JSON.
//
// Patterned after tools/llama-bench and tools/llama-pipeline-profiler.
// RPC registration and trace-env helpers are borrowed from llama-pipeline-profiler.

#include "build-info.h"
#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"

#include <chrono>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
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

int llama_gpipe_profiler(int argc, char ** argv);

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

static uint64_t time_ns() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::nanoseconds(clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

enum task_type {
    TASK_PP = 0,
    TASK_TG = 1,
};

struct task_config {
    task_type type;
    int n_prompt;   // pp: prompt tokens
    int n_gen;      // tg: generation tokens
};

struct profiler_config {
    std::string model_path;
    std::string rpc_endpoints;
    std::string tensor_split_str;
    std::string out_dir  = "./profiler-out";
    std::string trace_dir;
    std::string output_path = "heatmap.json";

    std::vector<task_config> tasks;
    std::vector<float> tensor_split;

    int n_gpu_layers  = 99;
    int n_batch       = 512;
    int n_ubatch      = 512;
    int n_threads     = 0;
    int ctx_size      = 4096;
    int repeat        = 5;
    int overlap_target = 5;
    bool warmup       = true;
    bool enable_trace = false;
    bool server_telemetry = false;
    int  gpipe_stages = 0;        // D6.9: 0=disabled, 2+=enable GPipe with N stages

    // MTP / NextN speculative decoding (TG-only)
    bool mtp_enabled         = false;
    std::string model_draft_path;  // separate draft model (e.g., Gemma assistant)
    int  spec_draft_n_max    = 2;
    int  spec_draft_n_min    = 1;
    float spec_draft_p_min   = 0.0f;
    std::string prompt_file;       // real prompt text for TG (forces diverse output)
    int  n_parallel          = 1;
    bool sample              = false; // use real sampling instead of synthetic token cycling

    llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER;
    ggml_type type_k = GGML_TYPE_Q4_0;
    ggml_type type_v = GGML_TYPE_Q4_0;
};

// ---------------------------------------------------------------------------
// RPC registration (borrowed from llama-pipeline-profiler)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Trace environment setup
// ---------------------------------------------------------------------------

static void setup_trace_env(const std::string & trace_dir, bool enable, bool server_telemetry) {
    if (!enable || trace_dir.empty()) {
        return;
    }
    fs::create_directories(trace_dir);
    const std::string sched    = trace_dir + "/sched-trace.jsonl";
    const std::string rpc      = trace_dir + "/rpc-trace.jsonl";
    const std::string pipeline = trace_dir + "/pipeline-trace.jsonl";
    const std::string srv_tel  = trace_dir + "/server-telemetry.jsonl";
    // Truncate so repeated runs do not accumulate rows.
    for (const auto & p : { sched, rpc, pipeline, srv_tel }) {
        std::ofstream(p).close();
    }
#ifdef _WIN32
    _putenv_s("GGML_SCHED_TRACE", "1");
    _putenv_s("GGML_RPC_TRACE", "1");
    _putenv_s("GGML_PIPELINE_TRACE", "1");
    _putenv_s("GGML_SCHED_TRACE_FILE", sched.c_str());
    _putenv_s("GGML_RPC_TRACE_FILE", rpc.c_str());
    _putenv_s("GGML_PIPELINE_TRACE_FILE", pipeline.c_str());
    if (server_telemetry) {
        _putenv_s("GGML_RPC_SERVER_TELEMETRY", "1");
        _putenv_s("GGML_RPC_SERVER_TELEMETRY_FILE", srv_tel.c_str());
    }
#else
    setenv("GGML_SCHED_TRACE", "1", 1);
    setenv("GGML_RPC_TRACE", "1", 1);
    setenv("GGML_PIPELINE_TRACE", "1", 1);
    setenv("GGML_SCHED_TRACE_FILE", sched.c_str(), 1);
    setenv("GGML_RPC_TRACE_FILE", rpc.c_str(), 1);
    setenv("GGML_PIPELINE_TRACE_FILE", pipeline.c_str(), 1);
    if (server_telemetry) {
        setenv("GGML_RPC_SERVER_TELEMETRY", "1", 1);
        setenv("GGML_RPC_SERVER_TELEMETRY_FILE", srv_tel.c_str(), 1);
    }
#endif
}

// ---------------------------------------------------------------------------
// Argument parsing helpers
// ---------------------------------------------------------------------------

static std::vector<float> parse_tensor_split(const std::string & ts, int n_max) {
    std::vector<float> out(n_max, 0.0f);
    auto parts = string_split<std::string>(ts, ',');
    for (size_t i = 0; i < parts.size() && i < (size_t) n_max; ++i) {
        out[i] = std::stof(parts[i]);
    }
    return out;
}

static ggml_type parse_cache_type(const std::string & s) {
    if (s == "f16")    return GGML_TYPE_F16;
    if (s == "bf16")   return GGML_TYPE_BF16;
    if (s == "q8_0")   return GGML_TYPE_Q8_0;
    if (s == "q4_0")   return GGML_TYPE_Q4_0;
    if (s == "q4_1")   return GGML_TYPE_Q4_1;
    if (s == "turbo2") return GGML_TYPE_TURBO2_0;
    if (s == "turbo3") return GGML_TYPE_TURBO3_0;
    if (s == "turbo4") return GGML_TYPE_TURBO4_0;
    throw std::invalid_argument("unknown cache type: " + s);
}

static llama_split_mode parse_split_mode(const std::string & s) {
    if (s == "none")   return LLAMA_SPLIT_MODE_NONE;
    if (s == "layer")  return LLAMA_SPLIT_MODE_LAYER;
    if (s == "row")    return LLAMA_SPLIT_MODE_ROW;
    if (s == "tensor") return LLAMA_SPLIT_MODE_TENSOR;
    throw std::invalid_argument("unknown split mode: " + s);
}

static std::vector<task_type> parse_tasks(const std::string & s) {
    std::vector<task_type> out;
    auto parts = string_split<std::string>(s, ',');
    for (const auto & p : parts) {
        if (p == "pp") {
            out.push_back(TASK_PP);
        } else if (p == "tg") {
            out.push_back(TASK_TG);
        } else {
            throw std::invalid_argument("unknown task: " + p);
        }
    }
    if (out.empty()) {
        throw std::invalid_argument("no tasks specified");
    }
    return out;
}

// ---------------------------------------------------------------------------
// Inference helpers
// ---------------------------------------------------------------------------

static bool run_prompt(llama_context * ctx, const std::vector<llama_token> & tokens, int n_batch) {
    if (tokens.empty()) {
        return true;
    }
    int n_processed = 0;
    const int n_total = (int) tokens.size();
    while (n_processed < n_total) {
        const int n_tokens = std::min(n_total - n_processed, n_batch);
        llama_token * batch_ptr = const_cast<llama_token *>(tokens.data()) + n_processed;
        if (llama_decode(ctx, llama_batch_get_one(batch_ptr, n_tokens)) != 0) {
            return false;
        }
        n_processed += n_tokens;
    }
    llama_synchronize(ctx);
    return true;
}

static bool run_gen(llama_context * ctx, int n_gen, bool use_sampling = false) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    common_sampler_ptr smpl;
    if (use_sampling) {
        common_params_sampling sparams;
        sparams.no_perf  = true;
        sparams.temp     = 0.7f;
        sparams.top_k    = 40;
        sparams.top_p    = 0.95f;
        sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TOP_P, COMMON_SAMPLER_TYPE_TEMPERATURE };
        smpl.reset(common_sampler_init(model, sparams));
    }

    llama_token token;
    if (use_sampling && smpl) {
        // After run_prompt, the context is positioned past the prompt.
        // Sample the first token from the logits at the last prompt position.
        common_sampler_reset(smpl.get());
        token = common_sampler_sample(smpl.get(), ctx, -1);
        char buf[256];
        int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
        if (n > 0) {
            fwrite(buf, 1, n, stdout);
            fflush(stdout);
        }
    } else {
        token = llama_vocab_get_add_bos(vocab) ? llama_vocab_bos(vocab) : 0;
    }

    for (int i = 0; i < n_gen; ++i) {
        if (llama_decode(ctx, llama_batch_get_one(&token, 1)) != 0) {
            return false;
        }
        if (use_sampling && smpl) {
            common_sampler_reset(smpl.get());
            common_sampler_accept(smpl.get(), token, false);
            token = common_sampler_sample(smpl.get(), ctx, 0);
            char buf[256];
            int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
            if (n > 0) {
                fwrite(buf, 1, n, stdout);
                fflush(stdout);
            }
        } else {
            token = (llama_token) (i + 1) % n_vocab;
        }
    }
    llama_synchronize(ctx);
    return true;
}

// ---------------------------------------------------------------------------
// Per-task result
// ---------------------------------------------------------------------------

struct task_result {
    task_type type;
    double wall_ms = 0.0;
    double tps     = 0.0;
    int    n_tokens = 0;
    int    n_eval   = 0;
    int    n_reused = 0;
    // MTP draft acceptance
    int    n_draft          = 0;
    int    n_draft_accepted = 0;
};

// ---------------------------------------------------------------------------
// Session: load model, warmup, run a single task
// ---------------------------------------------------------------------------

static task_result run_session(
        const profiler_config & cfg,
        const task_config & task,
        bool capture_trace) {
    setup_trace_env(cfg.trace_dir, capture_trace, cfg.server_telemetry);

    if (!cfg.rpc_endpoints.empty()) {
        register_rpc_servers(cfg.rpc_endpoints);
    }

    // D6.9: enable GPipe with configurable stage count for per-stage profiling
    if (cfg.gpipe_stages >= 2) {
        setenv("GGML_SCHED_GPIPE", "1", 1);
        setenv("GGML_SCHED_GPIPE_DEPTH", std::to_string(cfg.gpipe_stages).c_str(), 1);
        fprintf(stderr, ">>> GPipe enabled: n_stages=%d\n", cfg.gpipe_stages);
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
    cparams.n_ctx    = cfg.ctx_size;
    cparams.offload_kqv = true;

    const llama_vocab * vocab_pre = llama_model_get_vocab(model);
    std::vector<llama_token> prompt_tokens;
    if (task.type == TASK_PP && task.n_prompt > 0) {
        prompt_tokens.reserve((size_t) task.n_prompt);
        for (int i = 0; i < task.n_prompt; ++i) {
            if (i == 0 && llama_vocab_get_add_bos(vocab_pre)) {
                prompt_tokens.push_back(llama_vocab_bos(vocab_pre));
            } else {
                prompt_tokens.push_back((llama_token) (i % llama_vocab_n_tokens(vocab_pre)));
            }
        }
    }

    // Real prompt for TG — processed once to seed KV cache, then generation starts
    std::vector<llama_token> tg_prompt_tokens;
    if (task.type == TASK_TG) {
        if (!cfg.prompt_file.empty()) {
            std::ifstream pf(cfg.prompt_file);
            if (!pf) {
                throw std::runtime_error("failed to open prompt file: " + cfg.prompt_file);
            }
            std::string text((std::istreambuf_iterator<char>(pf)),
                             std::istreambuf_iterator<char>());
            tg_prompt_tokens = common_tokenize(vocab_pre, text, true, false);
            fprintf(stderr, ">>> tg prompt: %zu tokens from %s\n",
                    tg_prompt_tokens.size(), cfg.prompt_file.c_str());
        } else {
            // Fallback: single BOS token (minimal prompt, high MTP acceptance)
            if (llama_vocab_get_add_bos(vocab_pre)) {
                tg_prompt_tokens.push_back(llama_vocab_bos(vocab_pre));
            }
        }
    }

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        llama_model_free(model);
        throw std::runtime_error("failed to create context");
    }

    const int n_threads = cfg.n_threads > 0 ? cfg.n_threads : common_cpu_get_num_math();
    llama_set_n_threads(ctx, n_threads, n_threads);

    llama_perf_context_reset(ctx);
    llama_memory_clear(llama_get_memory(ctx), false);

    // MTP draft context (TG-only)
    llama_context * ctx_dft = nullptr;
    llama_model   * model_dft = nullptr;
    common_speculative_ptr spec;

    if (cfg.mtp_enabled && task.type == TASK_TG) {
        fprintf(stderr, ">>> MTP: creating draft context (n_max=%d, n_min=%d)%s\n",
                cfg.spec_draft_n_max, cfg.spec_draft_n_min,
                cfg.model_draft_path.empty() ? "" : " [separate draft]");

        auto cparams_mtp = cparams;
        cparams_mtp.ctx_type      = LLAMA_CONTEXT_TYPE_MTP;
        cparams_mtp.type_k        = cfg.type_k;
        cparams_mtp.type_v        = cfg.type_v;
        cparams_mtp.n_rs_seq      = 0;
        cparams_mtp.n_outputs_max = cfg.n_parallel;
        cparams_mtp.ctx_other     = ctx;

        if (!cfg.model_draft_path.empty()) {
            // Load separate draft model (e.g., Gemma assistant for MTP)
            fprintf(stderr, ">>> MTP: loading draft model: %s\n", cfg.model_draft_path.c_str());
            auto mparams_dft = mparams;
            mparams_dft.n_gpu_layers = 99; // draft is small, offload entirely
            model_dft = llama_model_load_from_file(cfg.model_draft_path.c_str(), mparams_dft);
            if (!model_dft) {
                throw std::runtime_error("failed to load draft model: " + cfg.model_draft_path);
            }
            ctx_dft = llama_init_from_model(model_dft, cparams_mtp);
        } else {
            // Fused MTP (e.g., Qwen NextN)
            ctx_dft = llama_init_from_model(model, cparams_mtp);
        }

        if (!ctx_dft) {
            if (model_dft) llama_model_free(model_dft);
            llama_free(ctx);
            llama_model_free(model);
            throw std::runtime_error("failed to create MTP draft context");
        }

        llama_set_n_threads(ctx_dft, n_threads, n_threads);

        // Wire speculative driver
        common_params_speculative sparams;
        sparams.types           = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        sparams.draft.n_max     = cfg.spec_draft_n_max;
        sparams.draft.n_min     = cfg.spec_draft_n_min;
        sparams.draft.p_min     = cfg.spec_draft_p_min;
        sparams.draft.ctx_tgt   = ctx;
        sparams.draft.ctx_dft   = ctx_dft;
        sparams.draft.backend_sampling = true;

        spec.reset(common_speculative_init(sparams, cfg.n_parallel));
        if (!spec) {
            if (model_dft) llama_model_free(model_dft);
            llama_free(ctx_dft);
            llama_free(ctx);
            llama_model_free(model);
            throw std::runtime_error("failed to init speculative driver");
        }
    }

    llama_perf_context_reset(ctx);
    llama_memory_clear(llama_get_memory(ctx), false);

    // Warmup: single gen pass, excluded from timing and traces.
    if (cfg.warmup) {
        run_gen(ctx, 1);
        llama_perf_context_reset(ctx);
        llama_memory_clear(llama_get_memory(ctx), false);
    }

    task_result res;
    res.type = task.type;

    if (task.type == TASK_PP) {
        // Prompt processing: batched decode of n_prompt tokens.
        fprintf(stderr, ">>> pp prompt_tokens=%d\n", (int) prompt_tokens.size());
        if (!prompt_tokens.empty()) {
            if (!run_prompt(ctx, prompt_tokens, cfg.n_batch)) {
                llama_free(ctx);
                llama_model_free(model);
                throw std::runtime_error("pp prompt processing failed");
            }
        }
        const uint64_t t0 = time_ns();
        // Re-run prompt for timing (kv now warm, but we measure decode wall).
        llama_memory_clear(llama_get_memory(ctx), false);
        if (!prompt_tokens.empty()) {
            if (!run_prompt(ctx, prompt_tokens, cfg.n_batch)) {
                llama_free(ctx);
                llama_model_free(model);
                throw std::runtime_error("pp timed prompt processing failed");
            }
        }
        const uint64_t elapsed = time_ns() - t0;
        res.n_tokens = (int) prompt_tokens.size();
        res.wall_ms  = elapsed / 1e6;
        res.tps      = res.n_tokens > 0 && elapsed > 0
                       ? 1e9 * res.n_tokens / (double) elapsed : 0.0;
    } else if (cfg.mtp_enabled && spec) {
        // Prompt processing — use tg_prompt_tokens for TG (real prompt or BOS)
        if (!tg_prompt_tokens.empty()) {
            if (!run_prompt(ctx, tg_prompt_tokens, cfg.n_batch)) {
                llama_free(ctx);
                llama_model_free(model);
                throw std::runtime_error("tg prefill failed");
            }
        }

        llama_perf_context_reset(ctx);
        const uint64_t t0 = time_ns();

        if (!run_gen(ctx, task.n_gen, cfg.sample)) {
            llama_free(ctx);
            llama_model_free(model);
            throw std::runtime_error("tg generation failed");
        }

        const uint64_t elapsed = time_ns() - t0;
        res.n_tokens  = task.n_gen;
        res.wall_ms   = elapsed / 1e6;
        res.tps       = task.n_gen > 0 && elapsed > 0
                        ? 1e9 * task.n_gen / (double) elapsed : 0.0;
    } else {
        // Token generation: serial decode of n_gen tokens.
        if (!run_prompt(ctx, prompt_tokens, cfg.n_batch)) {
            llama_free(ctx);
            llama_model_free(model);
            throw std::runtime_error("tg prefill failed");
        }
        llama_perf_context_reset(ctx);
        const uint64_t t0 = time_ns();
        if (!run_gen(ctx, task.n_gen, cfg.sample)) {
            llama_free(ctx);
            llama_model_free(model);
            throw std::runtime_error("tg generation failed");
        }
        const uint64_t elapsed = time_ns() - t0;
        res.n_tokens = task.n_gen;
        res.wall_ms  = elapsed / 1e6;
        res.tps      = task.n_gen > 0 && elapsed > 0
                       ? 1e9 * task.n_gen / (double) elapsed : 0.0;
    }

    const auto perf = llama_perf_context(ctx);
    res.n_eval   = perf.n_eval;
    res.n_reused = perf.n_reused;

    // Free draft before target (shares KV cache via ctx_other)
    if (ctx_dft) {
        spec.reset();  // release speculative driver first
        llama_free(ctx_dft);
    }
    if (model_dft) {
        llama_model_free(model_dft);
    }
    llama_free(ctx);
    llama_model_free(model);
    return res;
}

// ---------------------------------------------------------------------------
// Minimal JSON extraction for server-telemetry.jsonl
// The telemetry format is fixed and simple; we extract by key scanning
// rather than pulling in a full JSON library.
// ---------------------------------------------------------------------------

static bool extract_json_string(const std::string & line, const std::string & key, std::string & out) {
    std::string search = "\"" + key + "\"";
    size_t pos = line.find(search);
    if (pos == std::string::npos) return false;
    pos = line.find(':', pos + search.size());
    if (pos == std::string::npos) return false;
    pos = line.find('"', pos + 1);
    if (pos == std::string::npos) return false;
    size_t end = line.find('"', pos + 1);
    if (end == std::string::npos) return false;
    out = line.substr(pos + 1, end - pos - 1);
    return true;
}

static bool extract_json_array_uint64(const std::string & line, const std::string & key, std::vector<uint64_t> & out) {
    out.clear();
    std::string search = "\"" + key + "\"";
    size_t pos = line.find(search);
    if (pos == std::string::npos) return false;
    size_t start = line.find('[', pos + search.size());
    if (start == std::string::npos) return false;
    size_t end = line.find(']', start + 1);
    if (end == std::string::npos) return false;
    std::string arr = line.substr(start + 1, end - start - 1);
    if (arr.empty()) return true;
    auto parts = string_split<std::string>(arr, ',');
    for (const auto & p : parts) {
        out.push_back((uint64_t) std::stoull(p));
    }
    return true;
}

static bool extract_json_array_int32(const std::string & line, const std::string & key, std::vector<int32_t> & out) {
    out.clear();
    std::string search = "\"" + key + "\"";
    size_t pos = line.find(search);
    if (pos == std::string::npos) return false;
    size_t start = line.find('[', pos + search.size());
    if (start == std::string::npos) return false;
    size_t end = line.find(']', start + 1);
    if (end == std::string::npos) return false;
    std::string arr = line.substr(start + 1, end - start - 1);
    if (arr.empty()) return true;
    auto parts = string_split<std::string>(arr, ',');
    for (const auto & p : parts) {
        out.push_back((int32_t) std::stoi(p));
    }
    return true;
}

static bool extract_json_double(const std::string & line, const std::string & key, double & out) {
    std::string search = "\"" + key + "\"";
    size_t pos = line.find(search);
    if (pos == std::string::npos) return false;
    size_t colon = line.find(':', pos + search.size());
    if (colon == std::string::npos) return false;
    // Skip whitespace.
    size_t val = colon + 1;
    while (val < line.size() && (line[val] == ' ' || line[val] == '\t')) val++;
    out = std::stod(line.substr(val));
    return true;
}

struct gpu_device_meta {
    std::string name;
    uint64_t vram_mib = 0;
    std::string backend;
    int pcie_gen  = 0;
    int pcie_width = 0;
};

struct client_backend_timing {
    int backend_id;
    uint64_t avg_us;
    uint64_t min_us;
    uint64_t max_us;
    int count;
};

struct telemetry_data {
    bool valid = false;
    std::vector<uint64_t> device_timings_us;
    std::vector<int32_t>  layer_assignments;
    std::vector<uint64_t> copy_times_us;
    std::vector<gpu_device_meta> devices;
    std::vector<uint64_t> kv_read_times_us;
    std::vector<uint64_t> kv_write_times_us;
    // Client-side GPU data (enumerated from ggml devices + sched-trace)
    std::vector<gpu_device_meta> client_devices;
    std::vector<client_backend_timing> client_timings;
};

// Parse one device_meta object from a substring like:
// {"name":"NVIDIA RTX 3090","vram_mib":24576,"backend":"CUDA","pcie_gen":4,"pcie_width":16}
static gpu_device_meta parse_device_meta(const std::string & obj) {
    gpu_device_meta d;
    extract_json_string(obj, "name", d.name);
    extract_json_string(obj, "backend", d.backend);
    std::string vram_str;
    // vram_mib may be int or float; extract manually.
    {
        std::string key = "\"vram_mib\"";
        size_t pos = obj.find(key);
        if (pos != std::string::npos) {
            size_t colon = obj.find(':', pos + key.size());
            if (colon != std::string::npos) {
                size_t comma = obj.find_first_of(",}", colon + 1);
                std::string val = obj.substr(colon + 1, comma - colon - 1);
                d.vram_mib = (uint64_t) std::stoull(val);
            }
        }
    }
    {
        double g = 0;
        if (extract_json_double(obj, "pcie_gen", g)) d.pcie_gen = (int) g;
    }
    {
        double w = 0;
        if (extract_json_double(obj, "pcie_width", w)) d.pcie_width = (int) w;
    }
    return d;
}

static telemetry_data parse_server_telemetry(const fs::path & path) {
    telemetry_data tel;
    std::ifstream in(path);
    if (!in) {
        fprintf(stderr, "warning: server telemetry file not found: %s\n", path.c_str());
        return tel;
    }

    std::string line;
    while (std::getline(in, line)) {
        if (line.find("\"event\":\"server_telemetry\"") == std::string::npos &&
            line.find("\"event\": \"server_telemetry\"") == std::string::npos) {
            continue;
        }
        tel.valid = true;

        std::vector<uint64_t> u64;
        std::vector<int32_t> i32;

        if (extract_json_array_uint64(line, "device_timings_us", u64)) {
            tel.device_timings_us = u64;
        }
        if (extract_json_array_int32(line, "layer_assignments", i32)) {
            tel.layer_assignments = i32;
        }
        if (extract_json_array_uint64(line, "copy_times_us", u64)) {
            tel.copy_times_us = u64;
        }
        if (extract_json_array_uint64(line, "kv_read_times_us", u64)) {
            tel.kv_read_times_us = u64;
        }
        if (extract_json_array_uint64(line, "kv_write_times_us", u64)) {
            tel.kv_write_times_us = u64;
        }

        // device_meta is an array of objects: [{...},{...}]
        {
            std::string key = "\"device_meta\"";
            size_t pos = line.find(key);
            if (pos != std::string::npos) {
                size_t start = line.find('[', pos + key.size());
                if (start != std::string::npos) {
                    // Find each {...} within the array.
                    size_t cursor = start + 1;
                    while (cursor < line.size()) {
                        size_t ob = line.find('{', cursor);
                        if (ob == std::string::npos) break;
                        size_t cb = line.find('}', ob + 1);
                        if (cb == std::string::npos) break;
                        tel.devices.push_back(parse_device_meta(line.substr(ob, cb - ob + 1)));
                        cursor = cb + 1;
                        // Break at closing ].
                        size_t close = line.find(']', cursor);
                        if (close != std::string::npos && line.find('{', cursor) > close) break;
                    }
                }
            }
        }
    }
    return tel;
}

// Collect GPU-type devices from ggml registry for client-side metadata.
static std::vector<gpu_device_meta> collect_client_devices() {
    std::vector<gpu_device_meta> out;
    size_t n_dev = ggml_backend_dev_count();
    fprintf(stderr, ">>> collect_client_devices: %zu total devices registered\n", n_dev);
    for (size_t i = 0; i < n_dev; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        enum ggml_backend_dev_type dtype = ggml_backend_dev_type(dev);
        const char * _name = ggml_backend_dev_name(dev);
        const char * _desc = ggml_backend_dev_description(dev);
        fprintf(stderr, ">>>   dev[%zu]: name=%s desc=%s type=%d\n", i, _name ? _name : "null", _desc ? _desc : "null", (int)dtype);
        if (dtype != GGML_BACKEND_DEVICE_TYPE_GPU && dtype != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
        gpu_device_meta d;
        const char * name = ggml_backend_dev_name(dev);
        d.name = name ? name : "unknown";
        const char * desc = ggml_backend_dev_description(dev);
        d.backend = desc ? desc : "unknown";
        size_t free_mem = 0, total_mem = 0;
        ggml_backend_dev_memory(dev, &free_mem, &total_mem);
        d.vram_mib = total_mem / (1024 * 1024);
        // PCIe info not directly available from props; use 0 as default.
        d.pcie_gen = 0;
        d.pcie_width = 0;
        out.push_back(d);
    }
    return out;
}

// Parse sched-trace to extract per-backend compute timing (graph_compute_async aggregates).
static std::vector<client_backend_timing> parse_sched_trace_for_client_timing(const fs::path & trace_dir) {
    std::vector<client_backend_timing> out;
    fs::path sched_path = trace_dir / "sched-trace.jsonl";
    std::ifstream in(sched_path);
    if (!in) return out;

    // Aggregate per backend_id: sum, min, max, count
    struct agg { uint64_t sum; uint64_t min_val; uint64_t max_val; int count; };
    std::map<int, agg> backends;

    std::string line;
    while (std::getline(in, line)) {
        // Only look at graph_compute_async phases
        if (line.find("\"graph_compute_async\"") == std::string::npos) continue;
        // Extract backend
        size_t pos = line.find("\"backend\"");
        if (pos == std::string::npos) continue;
        size_t colon = line.find(':', pos);
        if (colon == std::string::npos) continue;
        size_t comma = line.find_first_of(",}", colon + 1);
        std::string bstr = line.substr(colon + 1, comma - colon - 1);
        int bid = std::stoi(bstr);
        // Extract elapsed_us
        pos = line.find("\"elapsed_us\"");
        if (pos == std::string::npos) continue;
        colon = line.find(':', pos);
        if (colon == std::string::npos) continue;
        comma = line.find_first_of(",}", colon + 1);
        std::string estr = line.substr(colon + 1, comma - colon - 1);
        uint64_t eus = (uint64_t)std::stoull(estr);

        auto & a = backends[bid];
        a.sum += eus;
        if (a.count == 0 || eus < a.min_val) a.min_val = eus;
        if (a.count == 0 || eus > a.max_val) a.max_val = eus;
        a.count++;
    }

    for (const auto & kv : backends) {
        client_backend_timing t;
        t.backend_id = kv.first;
        t.avg_us = kv.second.sum / kv.second.count;
        t.min_us = kv.second.min_val;
        t.max_us = kv.second.max_val;
        t.count = kv.second.count;
        out.push_back(t);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Heatmap synthesis
// ---------------------------------------------------------------------------

static std::string escape_json_string(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '"')  out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out += c;
    }
    return out;
}

static std::string current_timestamp_iso() {
    std::time_t now = std::time(nullptr);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    return std::string(buf);
}

struct heatmap_data {
    std::string git_sha;
    std::string generated_at;
    std::string model_path;
    int n_layers = 0;
    double param_count_b = 0.0;
    int ctx_size = 0;

    // Task results.
    bool has_pp = false;
    task_result pp_result;
    bool has_tg = false;
    task_result tg_result;

    // MTP draft stats (TG-only)
    bool has_mtp = false;
    int  mtp_draft_total    = 0;
    int  mtp_accept_total   = 0;

    // Server telemetry.
    bool has_telemetry = false;
    telemetry_data telemetry;
};

static void write_heatmap_json(const fs::path & path, const heatmap_data & hm) {
    std::ofstream f(path);
    if (!f) {
        throw std::runtime_error("failed to open output: " + path.string());
    }

    f << "{\n";
    f << "  \"schema_version\": 1,\n";
    f << "  \"generated_at\": \"" << hm.generated_at << "\",\n";
    f << "  \"git_sha\": \"" << hm.git_sha << "\",\n";

    f << "  \"model\": {\n";
    f << "    \"path\": \"" << escape_json_string(hm.model_path) << "\",\n";
    f << "    \"n_layers\": " << hm.n_layers << ",\n";
    f << "    \"param_count_b\": " << hm.param_count_b << ",\n";
    f << "    \"ctx_size\": " << hm.ctx_size << "\n";
    f << "  },\n";

    // Tasks.
    f << "  \"tasks\": {\n";
    bool first_task = true;

    auto write_task = [&](const std::string & name, const task_result & tr, int n_tokens_field, bool is_pp) {
        if (!first_task) f << ",\n";
        first_task = false;
        f << "    \"" << name << "\": {\n";
        if (is_pp) {
            f << "      \"n_prompt_tokens\": " << n_tokens_field << ",\n";
        } else {
            f << "      \"n_gen_tokens\": " << n_tokens_field << ",\n";
        }
        f << "      \"wall_ms\": " << tr.wall_ms << ",\n";
        f << "      \"tps\": " << tr.tps << ",\n";

        // MTP draft acceptance (TG-only)
        if (!is_pp && tr.n_draft > 0) {
            double accept_pct = tr.n_draft > 0
                ? 100.0 * tr.n_draft_accepted / tr.n_draft : 0.0;
            f << "      \"draft_accepted\": " << tr.n_draft_accepted << ",\n";
            f << "      \"draft_total\": " << tr.n_draft << ",\n";
            f << "      \"draft_accept_pct\": " << accept_pct << ",\n";
        }

        // Per-layer timing from server telemetry.
        f << "      \"layers\": [";
        // Use layer_assignments + device_timings if available.
        if (hm.has_telemetry && !hm.telemetry.layer_assignments.empty()) {
            for (size_t i = 0; i < hm.telemetry.layer_assignments.size(); ++i) {
                if (i > 0) f << ", ";
                int gpu_id = hm.telemetry.layer_assignments[i];
                double ms = 0.0;
                if (i < hm.telemetry.device_timings_us.size()) {
                    ms = hm.telemetry.device_timings_us[i] / 1000.0;
                }
                f << "\n        { \"idx\": " << i << ", \"ms\": " << ms << ", \"gpu_id\": " << gpu_id << " }";
            }
            f << "\n      ";
        }
        f << "]";

        // KV cache timing if available.
        if (hm.has_telemetry && !hm.telemetry.kv_read_times_us.empty()) {
            double total_read_ms = 0.0;
            for (auto v : hm.telemetry.kv_read_times_us) total_read_ms += v / 1000.0;
            double total_write_ms = 0.0;
            for (auto v : hm.telemetry.kv_write_times_us) total_write_ms += v / 1000.0;
            f << ",\n      \"kv\": { \"read_ms\": " << total_read_ms
              << ", \"write_ms\": " << total_write_ms << ", \"evict_count\": 0 }";
        } else if (!hm.has_telemetry) {
            f << ",\n      \"kv_source\": \"estimated\"";
        }

        f << "\n    }";
    };

    if (hm.has_pp) {
        write_task("pp", hm.pp_result, hm.pp_result.n_tokens, true);
    }
    if (hm.has_tg) {
        write_task("tg", hm.tg_result, hm.tg_result.n_tokens, false);
    }

    f << "\n  },\n";

    // GPU metadata: combine server telemetry devices + client-side GPU devices.
    f << "  \"gpu_metadata\": [\n";
    bool first_dev = true;
    // Server-side (RPC) devices
    if (hm.has_telemetry && !hm.telemetry.devices.empty()) {
        for (size_t i = 0; i < hm.telemetry.devices.size(); ++i) {
            if (!first_dev) f << ",\n";
            first_dev = false;
            const auto & d = hm.telemetry.devices[i];
            f << "    { \"id\": " << (int)hm.telemetry.devices.size() + (int)hm.telemetry.client_devices.size() + i
              << ", \"name\": \"" << escape_json_string(d.name) << "\""
              << ", \"backend\": \"" << escape_json_string(d.backend) << "\""
              << ", \"vram_total_mib\": " << d.vram_mib
              << ", \"pci_link_gen\": " << d.pcie_gen
              << ", \"pci_link_width\": " << d.pcie_width
              << ", \"source\": \"rpc_server\" }";
        }
    }
    // Client-side (local) GPU devices
    for (size_t i = 0; i < hm.telemetry.client_devices.size(); ++i) {
        if (!first_dev) f << ",\n";
        first_dev = false;
        const auto & d = hm.telemetry.client_devices[i];
        // Match client timing by backend_id (where backend_id == device index)
        uint64_t avg_us = 0, max_us = 0, min_us = 0;
        for (const auto & t : hm.telemetry.client_timings) {
            if (t.backend_id == (int)i) {
                avg_us = t.avg_us;
                max_us = t.max_us;
                min_us = t.min_us;
                break;
            }
        }
        f << "    { \"id\": " << i
          << ", \"name\": \"" << escape_json_string(d.name) << "\""
          << ", \"backend\": \"" << escape_json_string(d.backend) << "\""
          << ", \"vram_total_mib\": " << d.vram_mib
          << ", \"pci_link_gen\": " << d.pcie_gen
          << ", \"pci_link_width\": " << d.pcie_width
          << ", \"source\": \"client\""
          << ", \"compute_avg_us\": " << avg_us
          << ", \"compute_max_us\": " << max_us
          << ", \"compute_min_us\": " << min_us
          << " }";
    }
    f << "\n  ],\n";

    // Summary.
    f << "  \"summary\": {\n";
    bool has_any_timing = (hm.has_telemetry && !hm.telemetry.device_timings_us.empty())
                       || !hm.telemetry.client_timings.empty();
    if (has_any_timing) {
        // Straggler: GPU with highest avg compute time
        uint64_t max_time = 0;
        int straggler = 0;
        // Check server telemetry timings
        for (size_t i = 0; i < hm.telemetry.device_timings_us.size(); ++i) {
            if (hm.telemetry.device_timings_us[i] > max_time) {
                max_time = hm.telemetry.device_timings_us[i];
                straggler = (int)(hm.telemetry.client_devices.size() + i);
            }
        }
        // Check client timings
        for (const auto & t : hm.telemetry.client_timings) {
            if (t.avg_us > max_time) {
                max_time = t.avg_us;
                straggler = t.backend_id;
            }
        }
        f << "    \"straggler_gpu\": " << straggler << ",\n";
        f << "    \"kv_source\": \"server\"\n";
    } else {
        f << "    \"kv_source\": \"estimated\"\n";
    }
    f << "  }\n";

    f << "}\n";
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

static void usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s -m <model.gguf> [options]\n"
        "\n"
        "  -m, --model PATH           GGUF model path (required)\n"
        "  -rpc, --rpc HOST:PORT      RPC endpoint (comma-separated for multiple)\n"
        "  -ts, --tensor-split LIST   Per-device weight split (comma-separated floats)\n"
        "  --tasks LIST               Tasks: pp, tg (comma-separated, default: pp,tg)\n"
        "  -p, --n-prompt N           Prompt tokens for pp task (default: 512)\n"
        "  -n, --n-gen N              Generation tokens for tg task (default: 128)\n"
        "  -r, --repeat N             Repetitions per task (default: 5)\n"
        "  --warmup, --no-warmup      Warmup run before profiling (default: enabled)\n"
        "  -o, --output PATH          Heatmap output path (default: heatmap.json)\n"
        "  --out-dir PATH             Artifact directory (default: ./profiler-out)\n"
        "  --trace                    Enable sched/rpc/pipeline traces\n"
        "  --server-telemetry         Request server telemetry (sets GGML_RPC_SERVER_TELEMETRY=1)\n"
        "  -ngl, --n-gpu-layers N     GPU layers (default: 99)\n"
        "  -sm, --split-mode MODE     Split strategy: none, layer, row, tensor (default: layer)\n"
        "  -ctk, --cache-type-k TYPE  KV cache K type (default: q4_0)\n"
        "  -ctv, --cache-type-v TYPE  KV cache V type (default: q4_0)\n"
        "  -b, --batch-size N         Batch size (default: 512)\n"
        "  -ub, --ubatch-size N       Micro-batch size (default: 512)\n"
        "  -t, --threads N            Thread count (default: auto)\n"
        "  --ctx-size N               Context size (default: 4096)\n"
        "  --overlap-target N         B+6 gate threshold percent (default: 5)\n"
        "  --gpipe-stages N           Enable GPipe with N stages for per-stage profiling\n"
        "  --spec-type draft-mtp       Enable MTP speculative decoding (TG-only)\n"
        "  --spec-draft-n-max N        Max draft tokens per MTP step (default: 2)\n"
        "  --spec-draft-n-min N        Min draft tokens per MTP step (default: 1)\n"
        "  --spec-draft-p-min N        Min probability threshold for draft (default: 0.0)\n"
        "  --prompt-file PATH          Text file for TG prompt (default: BOS-only)\n"
        "  --model-draft PATH          Separate draft model for MTP (e.g., Gemma assistant)\n"
        "  --sample                   Use real sampling instead of token cycling (TG quality test)\n"
        "  -h, --help                 Usage\n",
        argv0);
}

// ---------------------------------------------------------------------------
// Main entry
// ---------------------------------------------------------------------------

int llama_gpipe_profiler(int argc, char ** argv) {
    profiler_config cfg;
    std::string tasks_str = "pp,tg";

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
        } else if (arg == "-rpc" || arg == "--rpc") {
            cfg.rpc_endpoints = need(arg.c_str());
        } else if (arg == "-ts" || arg == "--tensor-split") {
            cfg.tensor_split_str = need(arg.c_str());
        } else if (arg == "--tasks") {
            tasks_str = need(arg.c_str());
        } else if (arg == "-p" || arg == "--n-prompt") {
            // Applied to all tasks; only meaningful for pp.
            if (cfg.tasks.empty()) {
                // Store temporarily via a default task config.
                task_config tc;
                tc.type = TASK_PP;
                tc.n_prompt = std::stoi(need(arg.c_str()));
                tc.n_gen = 0;
                cfg.tasks.push_back(tc);
            } else {
                need(arg.c_str());
            }
        } else if (arg == "-n" || arg == "--n-gen") {
            if (cfg.tasks.empty()) {
                task_config tc;
                tc.type = TASK_TG;
                tc.n_prompt = 0;
                tc.n_gen = std::stoi(need(arg.c_str()));
                cfg.tasks.push_back(tc);
            } else {
                need(arg.c_str());
            }
        } else if (arg == "-r" || arg == "--repeat") {
            cfg.repeat = std::stoi(need(arg.c_str()));
        } else if (arg == "--warmup") {
            cfg.warmup = true;
        } else if (arg == "--no-warmup") {
            cfg.warmup = false;
        } else if (arg == "-o" || arg == "--output") {
            cfg.output_path = need(arg.c_str());
        } else if (arg == "--out-dir") {
            cfg.out_dir = need(arg.c_str());
        } else if (arg == "--trace") {
            cfg.enable_trace = true;
        } else if (arg == "--server-telemetry") {
            cfg.server_telemetry = true;
        } else if (arg == "-ngl" || arg == "--n-gpu-layers") {
            cfg.n_gpu_layers = std::stoi(need(arg.c_str()));
        } else if (arg == "-sm" || arg == "--split-mode") {
            cfg.split_mode = parse_split_mode(need(arg.c_str()));
        } else if (arg == "-ctk" || arg == "--cache-type-k") {
            cfg.type_k = parse_cache_type(need(arg.c_str()));
        } else if (arg == "-ctv" || arg == "--cache-type-v") {
            cfg.type_v = parse_cache_type(need(arg.c_str()));
        } else if (arg == "-b" || arg == "--batch-size") {
            cfg.n_batch = std::stoi(need(arg.c_str()));
        } else if (arg == "-ub" || arg == "--ubatch-size") {
            cfg.n_ubatch = std::stoi(need(arg.c_str()));
        } else if (arg == "-t" || arg == "--threads") {
            cfg.n_threads = std::stoi(need(arg.c_str()));
        } else if (arg == "--ctx-size") {
            cfg.ctx_size = std::stoi(need(arg.c_str()));
        } else if (arg == "--overlap-target") {
            cfg.overlap_target = std::stoi(need(arg.c_str()));
        } else if (arg == "--gpipe-stages") {
            // D6.9: enable GPipe with N stages for stage-granularity profiling
            cfg.gpipe_stages = std::stoi(need(arg.c_str()));
        } else if (arg == "--spec-type") {
            std::string spec_type = need(arg.c_str());
            if (spec_type == "draft-mtp") {
                cfg.mtp_enabled = true;
            } else {
                die_fmt("unknown spec type: %s (supported: draft-mtp)", spec_type.c_str());
            }
        } else if (arg == "--spec-draft-n-max") {
            cfg.spec_draft_n_max = std::stoi(need(arg.c_str()));
        } else if (arg == "--spec-draft-n-min") {
            cfg.spec_draft_n_min = std::stoi(need(arg.c_str()));
        } else if (arg == "--spec-draft-p-min") {
            cfg.spec_draft_p_min = std::stof(need(arg.c_str()));
        } else if (arg == "--prompt-file") {
            cfg.prompt_file = need(arg.c_str());
        } else if (arg == "--sample") {
            cfg.sample = true;
        } else if (arg == "--model-draft") {
            cfg.model_draft_path = need(arg.c_str());
        } else {
            fprintf(stderr, "error: unknown arg %s\n", arg.c_str());
            usage(argv[0]);
            return 1;
        }
    }

    if (cfg.model_path.empty()) {
        usage(argv[0]);
        return 1;
    }

    // Parse tasks.
    std::vector<task_type> task_types = parse_tasks(tasks_str);

    // Parse tensor split.
    if (!cfg.tensor_split_str.empty()) {
        cfg.tensor_split = parse_tensor_split(cfg.tensor_split_str, llama_max_devices());
    }

    // Build task configs.
    // -p and -n-gen apply globally; each task picks what it needs.
    int n_prompt_default = 512;
    int n_gen_default = 128;
    // Re-scan for -p and -n-gen defaults if they were provided.
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-p" || arg == "--n-prompt") && i + 1 < argc) {
            n_prompt_default = std::atoi(argv[++i]);
        } else if ((arg == "-n" || arg == "--n-gen") && i + 1 < argc) {
            n_gen_default = std::atoi(argv[++i]);
        } else if (arg == "-p" || arg == "--n-prompt" || arg == "-n" || arg == "--n-gen") {
            ++i; // skip value even if not matched above
        }
    }

    for (auto tt : task_types) {
        task_config tc;
        tc.type = tt;
        tc.n_prompt = (tt == TASK_PP) ? n_prompt_default : 0;
        tc.n_gen    = (tt == TASK_TG) ? n_gen_default  : 0;
        cfg.tasks.push_back(tc);
    }

    // Trace dir default.
    if (cfg.trace_dir.empty() && cfg.enable_trace) {
        cfg.trace_dir = cfg.out_dir + "/telemetry";
    }

    fs::create_directories(cfg.out_dir);
    if (cfg.enable_trace) {
        fs::create_directories(cfg.trace_dir);
    }

    llama_backend_init();
    ggml_backend_load_all();

    // Collect client-side GPU device metadata before model load.
    heatmap_data hm;
    hm.telemetry.client_devices = collect_client_devices();
    hm.git_sha      = llama_commit();
    hm.generated_at = current_timestamp_iso();
    hm.model_path   = cfg.model_path;
    hm.ctx_size     = cfg.ctx_size;

    // Load model once to extract metadata (n_layers, params).
    {
        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = cfg.n_gpu_layers;
        mparams.split_mode   = cfg.split_mode;
        if (!cfg.tensor_split.empty()) {
            mparams.tensor_split = cfg.tensor_split.data();
        }
        llama_model * model = llama_model_load_from_file(cfg.model_path.c_str(), mparams);
        if (!model) {
            llama_backend_free();
            throw std::runtime_error("failed to load model for metadata: " + cfg.model_path);
        }
        hm.n_layers = llama_model_n_layer(model);
        // param_count via desc if available; approximate via n_params.
        uint64_t n_params = llama_model_n_params(model);
        hm.param_count_b = (double) n_params / 1e9;
        llama_model_free(model);
    }

    try {
        for (const auto & task : cfg.tasks) {
            const int reps = std::max(1, cfg.repeat);
            task_result last;

            fprintf(stderr, ">>> task=%s reps=%d\n",
                    task.type == TASK_PP ? "pp" : "tg", reps);

            for (int rep = 1; rep <= reps; ++rep) {
                // Trace capture on last rep only.
                bool capture = cfg.enable_trace && (rep == reps);
                fprintf(stderr, ">>> rep %d/%d trace=%d\n", rep, reps, capture ? 1 : 0);
                auto res = run_session(cfg, task, capture);
                last = res;
                fprintf(stderr, ">>> wall_ms=%.2f tps=%.2f n_eval=%d n_reused=%d",
                        res.wall_ms, res.tps, res.n_eval, res.n_reused);
                if (res.n_draft > 0) {
                    double accept_pct = 100.0 * res.n_draft_accepted / res.n_draft;
                    fprintf(stderr, " draft=%d accept=%.1f%%",
                            res.n_draft, accept_pct);
                }
                fprintf(stderr, "\n");
            }

            if (task.type == TASK_PP) {
                hm.has_pp = true;
                hm.pp_result = last;
            } else {
                hm.has_tg = true;
                hm.tg_result = last;
                if (last.n_draft > 0) {
                    hm.has_mtp = true;
                    hm.mtp_draft_total  = last.n_draft;
                    hm.mtp_accept_total = last.n_draft_accepted;
                }
            }
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        llama_backend_free();
        return 1;
    }

    // Ingest server telemetry if requested.
    if (cfg.server_telemetry && cfg.enable_trace) {
        fs::path tel_path = fs::path(cfg.trace_dir) / "server-telemetry.jsonl";
        // Preserve client-side data that was collected earlier.
        auto client_devices_saved = std::move(hm.telemetry.client_devices);
        hm.telemetry = parse_server_telemetry(tel_path);
        hm.telemetry.client_devices = std::move(client_devices_saved);
        hm.has_telemetry = hm.telemetry.valid;
        if (!hm.has_telemetry) {
            fprintf(stderr, "warning: server telemetry unavailable; heatmap kv fields omitted\n");
        }
        // Also parse sched-trace for client-side per-backend compute timing.
        hm.telemetry.client_timings = parse_sched_trace_for_client_timing(cfg.trace_dir);
    }

    // Write heatmap.
    fs::path out_path = fs::path(cfg.out_dir) / cfg.output_path;
    write_heatmap_json(out_path, hm);
    fprintf(stderr, ">>> heatmap written: %s\n", out_path.c_str());

    llama_backend_free();
    fprintf(stderr, "llama-gpipe-profiler done out=%s\n", cfg.out_dir.c_str());
    return 0;
}
