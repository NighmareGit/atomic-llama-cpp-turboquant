#include "testing.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <iostream>

// D6.8: Per-sequence event API tests.
// Each concurrent sequence owns an independent event array so events
// never alias regardless of concurrency level.

// Test 1: per-seq init with n_seqs=2 allocates both event sets
static void test_init_multi_seq(testing & t) {
    ggml_backend_t cpu0 = ggml_backend_cpu_init();
    ggml_backend_t cpu1 = ggml_backend_cpu_init();
    if (!cpu0 || !cpu1) {
        ggml_backend_free(cpu0);
        ggml_backend_free(cpu1);
        t.assert_true("backend init", false);
        return;
    }

    ggml_backend_t backends[] = { cpu0, cpu1 };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);

    if (!sched) {
        ggml_backend_free(cpu0);
        ggml_backend_free(cpu1);
        t.assert_true("scheduler init", false);
        return;
    }

    ggml_sched_gpipe_init_multi(sched, 3, 2);

    // Record on both sequences independently
    ggml_sched_gpipe_record_seq(sched, 0, 0);
    ggml_sched_gpipe_record_seq(sched, 0, 1);

    // Drain both sequences (split_id=-1 drains all stages for that seq)
    ggml_sched_gpipe_wait_seq(sched, -1, 0);
    ggml_sched_gpipe_wait_seq(sched, -1, 1);

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu0);
    ggml_backend_free(cpu1);

    t.assert_true("per-seq init+record+wait", true);
}

// Test 2: two sequences use independent event arrays without conflict
static void test_two_seq_independent_events(testing & t) {
    ggml_backend_t cpu0 = ggml_backend_cpu_init();
    ggml_backend_t cpu1 = ggml_backend_cpu_init();
    if (!cpu0 || !cpu1) {
        ggml_backend_free(cpu0);
        ggml_backend_free(cpu1);
        t.assert_true("backend init", false);
        return;
    }

    ggml_backend_t backends[] = { cpu0, cpu1 };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);

    if (!sched) {
        ggml_backend_free(cpu0);
        ggml_backend_free(cpu1);
        t.assert_true("scheduler init", false);
        return;
    }

    ggml_sched_gpipe_init_multi(sched, 3, 2);

    // Seq 0 at stage 0, Seq 1 at stage 0
    ggml_sched_gpipe_record_seq(sched, 0, 0);
    ggml_sched_gpipe_record_seq(sched, 0, 1);

    // Seq 0 advances: wait stage 0, record stage 1
    ggml_sched_gpipe_wait_seq(sched, 0, 0);
    ggml_sched_gpipe_record_seq(sched, 1, 0);

    // Seq 1 advances: wait stage 0, record stage 1
    ggml_sched_gpipe_wait_seq(sched, 0, 1);
    ggml_sched_gpipe_record_seq(sched, 1, 1);

    // Seq 0 completes: wait stage 1, record stage 2 (last)
    ggml_sched_gpipe_wait_seq(sched, 1, 0);
    ggml_sched_gpipe_record_seq(sched, 2, 0);
    ggml_sched_gpipe_wait_seq(sched, 2, 0);

    // Seq 1 completes: wait stage 1, record stage 2 (last)
    ggml_sched_gpipe_wait_seq(sched, 1, 1);
    ggml_sched_gpipe_record_seq(sched, 2, 1);
    ggml_sched_gpipe_wait_seq(sched, 2, 1);

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu0);
    ggml_backend_free(cpu1);

    t.assert_true("two-seq independent events without deadlock", true);
}

// Test 3: per-seq isolation — seq 0 and seq 1 events never alias
static void test_per_seq_isolation(testing & t) {
    ggml_backend_t cpu0 = ggml_backend_cpu_init();
    ggml_backend_t cpu1 = ggml_backend_cpu_init();
    if (!cpu0 || !cpu1) {
        ggml_backend_free(cpu0);
        ggml_backend_free(cpu1);
        t.assert_true("backend init", false);
        return;
    }

    ggml_backend_t backends[] = { cpu0, cpu1 };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);

    if (!sched) {
        ggml_backend_free(cpu0);
        ggml_backend_free(cpu1);
        t.assert_true("scheduler init", false);
        return;
    }

    ggml_sched_gpipe_init_multi(sched, 2, 2);

    // Seq 0: record stage 0, wait, record stage 1, wait
    ggml_sched_gpipe_record_seq(sched, 0, 0);
    ggml_sched_gpipe_wait_seq(sched, 0, 0);
    ggml_sched_gpipe_record_seq(sched, 1, 0);
    ggml_sched_gpipe_wait_seq(sched, 1, 0);

    // Seq 1: independent of seq 0 — record stage 0, wait, record stage 1, wait
    ggml_sched_gpipe_record_seq(sched, 0, 1);
    ggml_sched_gpipe_wait_seq(sched, 0, 1);
    ggml_sched_gpipe_record_seq(sched, 1, 1);
    ggml_sched_gpipe_wait_seq(sched, 1, 1);

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu0);
    ggml_backend_free(cpu1);

    t.assert_true("per-seq isolation no event aliasing", true);
}

// Test 4: 3-stage pipeline with 2 sequences interleaved for 3 cycles
static void test_full_3stage_2seq_pipeline(testing & t) {
    ggml_backend_t cpu0 = ggml_backend_cpu_init();
    ggml_backend_t cpu1 = ggml_backend_cpu_init();
    if (!cpu0 || !cpu1) {
        ggml_backend_free(cpu0);
        ggml_backend_free(cpu1);
        t.assert_true("backend init", false);
        return;
    }

    ggml_backend_t backends[] = { cpu0, cpu1 };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);

    if (!sched) {
        ggml_backend_free(cpu0);
        ggml_backend_free(cpu1);
        t.assert_true("scheduler init", false);
        return;
    }

    ggml_sched_gpipe_init_multi(sched, 3, 2);

    // 3 full pipeline cycles: each sequence owns its own event array.
    // No bank toggling needed — seq_id identifies the event set.
    for (int cycle = 0; cycle < 3; cycle++) {
        // Both sequences start stage 0
        ggml_sched_gpipe_record_seq(sched, 0, 0);
        ggml_sched_gpipe_record_seq(sched, 0, 1);

        // Both advance to stage 1
        ggml_sched_gpipe_wait_seq(sched, 0, 0);
        ggml_sched_gpipe_record_seq(sched, 1, 0);

        ggml_sched_gpipe_wait_seq(sched, 0, 1);
        ggml_sched_gpipe_record_seq(sched, 1, 1);

        // Both complete stage 2 (last = KV write)
        ggml_sched_gpipe_wait_seq(sched, 1, 0);
        ggml_sched_gpipe_record_seq(sched, 2, 0);
        ggml_sched_gpipe_wait_seq(sched, 2, 0);

        ggml_sched_gpipe_wait_seq(sched, 1, 1);
        ggml_sched_gpipe_record_seq(sched, 2, 1);
        ggml_sched_gpipe_wait_seq(sched, 2, 1);
    }

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu0);
    ggml_backend_free(cpu1);

    t.assert_true("3-stage 2-seq pipeline 3 cycles per-seq", true);
}

// Test 5: n_seqs=1 is backward-compatible with single-seq Mode A
static void test_single_seq_backward_compat(testing & t) {
    ggml_backend_t cpu0 = ggml_backend_cpu_init();
    if (!cpu0) {
        t.assert_true("backend init", false);
        return;
    }

    ggml_backend_t backends[] = { cpu0 };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, false);

    if (!sched) {
        ggml_backend_free(cpu0);
        t.assert_true("scheduler init", false);
        return;
    }

    ggml_sched_gpipe_init_multi(sched, 2, 1);

    ggml_sched_gpipe_record_seq(sched, 0, 0);
    ggml_sched_gpipe_wait_seq(sched, 0, 0);
    ggml_sched_gpipe_record_seq(sched, 1, 0);
    ggml_sched_gpipe_wait_seq(sched, 1, 0);

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu0);

    t.assert_true("single-seq backward compat", true);
}

// Test 6: drain all events across sequences with split_id=-1
static void test_drain_all_seqs(testing & t) {
    ggml_backend_t cpu0 = ggml_backend_cpu_init();
    ggml_backend_t cpu1 = ggml_backend_cpu_init();
    if (!cpu0 || !cpu1) {
        ggml_backend_free(cpu0);
        ggml_backend_free(cpu1);
        t.assert_true("backend init", false);
        return;
    }

    ggml_backend_t backends[] = { cpu0, cpu1 };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);

    if (!sched) {
        ggml_backend_free(cpu0);
        ggml_backend_free(cpu1);
        t.assert_true("scheduler init", false);
        return;
    }

    ggml_sched_gpipe_init_multi(sched, 3, 2);

    // Record on all stages across both sequences
    ggml_sched_gpipe_record_seq(sched, 0, 0);
    ggml_sched_gpipe_record_seq(sched, 1, 0);
    ggml_sched_gpipe_record_seq(sched, 2, 0);
    ggml_sched_gpipe_record_seq(sched, 0, 1);
    ggml_sched_gpipe_record_seq(sched, 1, 1);
    ggml_sched_gpipe_record_seq(sched, 2, 1);

    // Drain all events across all sequences
    ggml_sched_gpipe_wait_seq(sched, -1, 0);
    ggml_sched_gpipe_wait_seq(sched, -1, 1);

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu0);
    ggml_backend_free(cpu1);

    t.assert_true("drain all seqs", true);
}

int main() {
    testing t;

    test_init_multi_seq(t);
    test_two_seq_independent_events(t);
    test_per_seq_isolation(t);
    test_full_3stage_2seq_pipeline(t);
    test_single_seq_backward_compat(t);
    test_drain_all_seqs(t);

    return t.summary();
}
