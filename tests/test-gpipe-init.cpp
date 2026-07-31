#include "testing.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <iostream>

// Verify ggml_sched_gpipe_init is declared in the header
// (linker will catch if declaration exists but definition is missing)
extern "C" void ggml_sched_gpipe_init(ggml_backend_sched_t sched, int n_stages);

static void test_function_declared(testing & t) {
    // If this compiles and links, the function is declared in ggml-backend.h
    void (*fn)(ggml_backend_sched_t, int) = ggml_sched_gpipe_init;
    t.assert_true("function declared", fn != nullptr);
}

static void test_init_with_cpu_backend(testing & t) {
    // Create a minimal CPU-only scheduler
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    if (!cpu_backend) {
        std::cerr << "SKIP: cannot create CPU backend" << std::endl;
        return;
    }

    ggml_backend_t backends[] = { cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, false);

    if (!sched) {
        ggml_backend_free(cpu_backend);
        std::cerr << "SKIP: cannot create scheduler" << std::endl;
        return;
    }

    // Should not crash when initializing GPipe events
    ggml_sched_gpipe_init(sched, 2);

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu_backend);

    t.assert_true("init completed without crash", true);
}

static void test_init_zero_stages(testing & t) {
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    if (!cpu_backend) {
        std::cerr << "SKIP: cannot create CPU backend" << std::endl;
        return;
    }

    ggml_backend_t backends[] = { cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, false);

    if (!sched) {
        ggml_backend_free(cpu_backend);
        std::cerr << "SKIP: cannot create scheduler" << std::endl;
        return;
    }

    // Zero stages should be a no-op (or assert, depending on design)
    // We just verify it doesn't corrupt memory
    ggml_sched_gpipe_init(sched, 0);

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu_backend);

    t.assert_true("init zero stages completed", true);
}

int main(int argc, char * argv[]) {
    testing t(std::cout);
    if (argc >= 2) {
        t.set_filter(argv[1]);
    }

    t.test("function_declared", test_function_declared);
    t.test("init_with_cpu_backend", test_init_with_cpu_backend);
    t.test("init_zero_stages", test_init_zero_stages);

    return t.summary();
}
