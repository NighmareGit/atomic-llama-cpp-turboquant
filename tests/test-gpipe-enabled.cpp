#include "testing.h"
#include "../src/llama-context.h"

#include <cstdlib>
#include <iostream>
#include <sys/wait.h>
#include <unistd.h>

// Test accessor declared in llama-context.h, defined in llama-context.cpp
extern "C" bool llama_gpipe_enabled_accessor();

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
static int check_enabled_is_true(void) {
    return llama_gpipe_enabled_accessor() ? 0 : 1;
}

static int check_enabled_is_false(void) {
    return llama_gpipe_enabled_accessor() ? 1 : 0;
}

static void test_env_unset(testing & t) {
    int rc = run_with_env("GGML_SCHED_GPIPE", nullptr, check_enabled_is_false);
    t.assert_equal(0, rc);
}

static void test_env_zero(testing & t) {
    int rc = run_with_env("GGML_SCHED_GPIPE", "0", check_enabled_is_false);
    t.assert_equal(0, rc);
}

static void test_env_one(testing & t) {
    int rc = run_with_env("GGML_SCHED_GPIPE", "1", check_enabled_is_true);
    t.assert_equal(0, rc);
}

static void test_env_nonzero(testing & t) {
    int rc = run_with_env("GGML_SCHED_GPIPE", "42", check_enabled_is_true);
    t.assert_equal(0, rc);
}

int main(int argc, char * argv[]) {
    testing t(std::cout);
    if (argc >= 2) {
        t.set_filter(argv[1]);
    }

    t.test("env_unset_returns_false",  test_env_unset);
    t.test("env_zero_returns_false",  test_env_zero);
    t.test("env_one_returns_true",    test_env_one);
    t.test("env_nonzero_returns_true", test_env_nonzero);

    return t.summary();
}
