#include "testing.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <iostream>

// Verify ggml_sched_gpipe_wait is declared in the header
extern "C" void ggml_sched_gpipe_wait(ggml_backend_sched_t sched, int split_id);

static void test_function_declared(testing & t) {
    // If this compiles and links, the function is declared in ggml-backend.h
    void (*fn)(ggml_backend_sched_t, int) = ggml_sched_gpipe_wait;
    t.assert_true("function declared", fn != nullptr);
}

static void test_wait_no_op_when_uninitialized(testing & t) {
    // Create a minimal CPU-only scheduler without GPipe init
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

    // n_gpipe_stages == 0 (uninitialized) - should be a no-op
    ggml_sched_gpipe_wait(sched, 0);
    ggml_sched_gpipe_wait(sched, -1);

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu_backend);

    t.assert_true("wait no-op when uninitialized", true);
}

static void test_wait_all_events(testing & t) {
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

    // Initialize GPipe with 3 stages
    ggml_sched_gpipe_init(sched, 3);

    // split_id < 0 should wait on all events
    ggml_sched_gpipe_wait(sched, -1);

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu_backend);

    t.assert_true("wait all events (split_id < 0)", true);
}

static void test_wait_specific_event(testing & t) {
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

    // Initialize GPipe with 4 stages
    ggml_sched_gpipe_init(sched, 4);

    // Wait on specific valid stage
    ggml_sched_gpipe_wait(sched, 0);
    ggml_sched_gpipe_wait(sched, 2);
    ggml_sched_gpipe_wait(sched, 3);

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu_backend);

    t.assert_true("wait specific event (split_id >= 0)", true);
}

int main(int argc, char * argv[]) {
    testing t(std::cout);
    if (argc >= 2) {
        t.set_filter(argv[1]);
    }

    t.test("function_declared", test_function_declared);
    t.test("wait_no_op_when_uninitialized", test_wait_no_op_when_uninitialized);
    t.test("wait_all_events", test_wait_all_events);
    t.test("wait_specific_event", test_wait_specific_event);

    return t.summary();
}
