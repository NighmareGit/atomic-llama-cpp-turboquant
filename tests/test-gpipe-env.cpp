#include "testing.h"
#include "../src/llama-context.h"
#include "llama.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sys/wait.h>
#include <unistd.h>

// Test accessor declared in llama-context.h, defined in llama-context.cpp
extern "C" bool llama_gpipe_enabled_accessor();
extern "C" bool llama_gpipe_context_enabled(struct llama_context * ctx);

static const char * g_model_path = nullptr;
static bool g_skip_model_tests = false;

// Helper: run a function in a child process with a specific env var set.
// Returns 0 if the child reported success.
static int run_with_env(const char * name, const char * value, int (*fn)(void)) {
    pid_t pid = fork();
    if (pid == 0) {
        if (value == nullptr) {
            unsetenv(name);
        } else {
            setenv(name, value, 1);
        }
        _exit(fn());
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 0 : 1;
}

// Return 0 when the test condition holds (success).
static int check_context_enabled_is_true(void) {
    if (!llama_gpipe_enabled_accessor()) {
        return 1;
    }

    struct llama_model_params mparams = llama_model_default_params();
    struct llama_model * model = llama_model_load_from_file(g_model_path, mparams);
    if (!model) {
        std::cerr << "FAIL: could not load model" << std::endl;
        return 1;
    }

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 512;

    struct llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        llama_model_free(model);
        return 1;
    }

    bool enabled = llama_gpipe_context_enabled(ctx);
    llama_free(ctx);
    llama_model_free(model);

    return enabled ? 0 : 1;
}

static int check_context_enabled_is_false(void) {
    if (llama_gpipe_enabled_accessor()) {
        return 1;
    }

    struct llama_model_params mparams = llama_model_default_params();
    struct llama_model * model = llama_model_load_from_file(g_model_path, mparams);
    if (!model) {
        std::cerr << "FAIL: could not load model" << std::endl;
        return 1;
    }

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 512;

    struct llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        llama_model_free(model);
        return 1;
    }

    bool enabled = llama_gpipe_context_enabled(ctx);
    llama_free(ctx);
    llama_model_free(model);

    return enabled ? 1 : 0;
}

static void test_env_unset(testing & t) {
    int rc = run_with_env("GGML_SCHED_GPIPE", nullptr, check_context_enabled_is_false);
    t.assert_equal(0, rc);
}

static void test_env_zero(testing & t) {
    int rc = run_with_env("GGML_SCHED_GPIPE", "0", check_context_enabled_is_false);
    t.assert_equal(0, rc);
}

static void test_env_one(testing & t) {
    int rc = run_with_env("GGML_SCHED_GPIPE", "1", check_context_enabled_is_true);
    t.assert_equal(0, rc);
}

int main(int argc, char * argv[]) {
    g_model_path = getenv("LLAMACPP_TEST_MODELFILE");

    if (argc >= 2 && std::string(argv[1]) != std::string("--help")) {
        g_model_path = argv[1];
    }

    if (!g_model_path || std::strlen(g_model_path) == 0) {
        std::cerr << "WARNING: No model file provided. Skipping model-dependent tests."
                  << " Set LLAMACPP_TEST_MODELFILE=<gguf_model_path> or pass model path as arg." << std::endl;
        g_skip_model_tests = true;
    }

    testing t(std::cout);
    if (argc >= 2) {
        t.set_filter(argv[argc - 1]);
    }

    if (g_skip_model_tests) {
        std::cerr << "Skipping all tests (no model available)." << std::endl;
        return 0;
    }

    t.test("env_unset_gpipe_disabled", test_env_unset);
    t.test("env_zero_gpipe_disabled", test_env_zero);
    t.test("env_one_gpipe_enabled",  test_env_one);

    return t.summary();
}
