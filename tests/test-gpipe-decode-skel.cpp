#include "testing.h"
#include "../src/llama-context.h"
#include "llama.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sys/wait.h>
#include <unistd.h>

// Accessors declared in llama-context.h, defined in llama-context.cpp
extern "C" bool llama_gpipe_enabled_accessor();
extern "C" bool llama_gpipe_context_enabled(struct llama_context * ctx);

// Skeleton function under test
int32_t llama_decode_gpipe_impl(llama_context * ctx, llama_batch batch, const float * logits);

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

// Verify llama_decode_gpipe_impl returns -1 when GPipe is disabled
static int check_returns_minus_one_when_disabled(void) {
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

    // Verify GPipe is disabled
    if (llama_gpipe_context_enabled(ctx)) {
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    // Call the skeleton function - should return -1
    llama_batch batch = {};
    int32_t result = llama_decode_gpipe_impl(ctx, batch, nullptr);

    llama_free(ctx);
    llama_model_free(model);

    return (result == -1) ? 0 : 1;
}

static void test_returns_minus_one_when_disabled(testing & t) {
    int rc = run_with_env("GGML_SCHED_GPIPE", nullptr, check_returns_minus_one_when_disabled);
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

    t.test("returns_minus_one_when_disabled", test_returns_minus_one_when_disabled);

    return t.summary();
}
