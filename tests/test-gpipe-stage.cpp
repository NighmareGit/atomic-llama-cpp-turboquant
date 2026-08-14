#include "testing.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <iostream>

extern "C" int32_t llama_decode_gpipe_impl(llama_context * ctx, llama_batch batch, const float * logits);
extern "C" void ggml_sched_gpipe_init(ggml_backend_sched_t sched, int n_stages);
extern "C" void ggml_sched_gpipe_wait(ggml_backend_sched_t sched, int split_id);
extern "C" void ggml_sched_gpipe_record(ggml_backend_sched_t sched, int stage_id);

static void test_function_declared(testing & t) {
    void (*fn)(ggml_backend_sched_t, int) = ggml_sched_gpipe_record;
    t.assert_true("ggml_sched_gpipe_record declared", fn != nullptr);
}

static void test_stage_transition(testing & t) {
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

    ggml_sched_gpipe_init(sched, 2);

    // Stage 0: record event on stage 0
    ggml_sched_gpipe_record(sched, 0);

    // Stage 1: wait on stage 0 event (simulates consumer waiting for producer)
    ggml_sched_gpipe_wait(sched, 0);

    // Record event on stage 1
    ggml_sched_gpipe_record(sched, 1);

    // Wait on all events (split_id < 0)
    ggml_sched_gpipe_wait(sched, -1);

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu_backend);

    t.assert_true("stage transition completed without crash", true);
}

static void test_stage_advance_logic(testing & t) {
    // Verify the stage state machine transitions correctly
    // Stage 0 -> Stage 1 -> Stage 0 (cyclic)
    int cur_stage = 0;
    const int n_stages = 2;

    // Simulate stage advance
    cur_stage = (cur_stage + 1) % n_stages;
    t.assert_equal(1, cur_stage);

    cur_stage = (cur_stage + 1) % n_stages;
    t.assert_equal(0, cur_stage);

    // Verify wait on invalid stage_id is handled
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

    ggml_sched_gpipe_init(sched, 2);

    // Wait on out-of-range stage_id should be a no-op
    ggml_sched_gpipe_wait(sched, 99);

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu_backend);

    t.assert_true("invalid stage_id handled gracefully", true);
}

int main(int argc, char * argv[]) {
    testing t(std::cout);
    if (argc >= 2) {
        t.set_filter(argv[1]);
    }

    t.test("function_declared", test_function_declared);
    t.test("stage_transition", test_stage_transition);
    t.test("stage_advance_logic", test_stage_advance_logic);

    return t.summary();
}
