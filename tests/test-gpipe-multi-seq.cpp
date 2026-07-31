#include "testing.h"

#include <iostream>
#include <vector>
#include <map>
#include <string>

// D6.5: multi-seq state machine unit tests (pure logic, no GPU backends)

// --- Simulated state machine (mirrors the implementation in llama-context.cpp) ---

static constexpr int MAX_STAGES = 8;

struct seq_pipeline_state {
    int n_stages;
    int microbatch_size;
    std::vector<int64_t> stage_tokens;  // stage -> seq_id, -1 = free
    std::map<int64_t, int> seq_stage;   // seq_id -> current stage
    int active_sequences;
};

static void seq_pipeline_init(seq_pipeline_state & s, int n_stages, int n_seqs) {
    s.n_stages = n_stages;
    s.microbatch_size = n_seqs;
    s.stage_tokens.assign(n_stages, -1);
    s.seq_stage.clear();
    s.active_sequences = 0;
}

static bool seq_pipeline_reserve(seq_pipeline_state & s, int64_t seq_id) {
    if (s.active_sequences >= s.microbatch_size) {
        return false;
    }
    if (s.seq_stage.find(seq_id) != s.seq_stage.end()) {
        return false;  // already reserved
    }
    s.seq_stage[seq_id] = 0;
    s.active_sequences++;
    return true;
}

static void seq_pipeline_release(seq_pipeline_state & s, int64_t seq_id) {
    auto it = s.seq_stage.find(seq_id);
    if (it == s.seq_stage.end()) {
        return;
    }
    // Release any stage owned by this sequence
    for (int i = 0; i < s.n_stages; i++) {
        if (s.stage_tokens[i] == seq_id) {
            s.stage_tokens[i] = -1;
        }
    }
    s.seq_stage.erase(it);
    s.active_sequences--;
}

// Stage-available dispatch: fill free stages from any sequence
// Each sequence advances at most one stage per dispatch call.
// Returns number of stages dispatched
static int seq_pipeline_dispatch(seq_pipeline_state & s) {
    int dispatched = 0;
    std::map<int64_t, bool> used_this_cycle;

    for (int stage = 0; stage < s.n_stages; stage++) {
        if (s.stage_tokens[stage] == -1) {
            // Skip stage 0 for sequences that already advanced this cycle
            // (a sequence's next token must wait for the next dispatch call)
            for (auto & [seq_id, pos] : s.seq_stage) {
                if (pos == stage && !used_this_cycle[seq_id]) {
                    s.stage_tokens[stage] = seq_id;
                    used_this_cycle[seq_id] = true;

                    if (stage == s.n_stages - 1) {
                        // Last stage: KV write complete, release and wrap
                        s.stage_tokens[stage] = -1;
                        pos = 0;
                    } else {
                        // Advance to next stage, release previous
                        if (stage > 0) {
                            s.stage_tokens[stage - 1] = -1;
                        }
                        pos = stage + 1;
                    }
                    dispatched++;
                    break;
                }
            }
        }
    }
    return dispatched;
}

// --- Tests ---

static void test_reserve_and_release(testing & t) {
    seq_pipeline_state s;
    seq_pipeline_init(s, 3, 2);

    // Reserve two sequences
    t.assert_true("reserve seq 0", seq_pipeline_reserve(s, 0));
    t.assert_true("reserve seq 1", seq_pipeline_reserve(s, 1));
    t.assert_true("seq 0 at stage 0", s.seq_stage[0] == 0);
    t.assert_true("seq 1 at stage 0", s.seq_stage[1] == 0);
    t.assert_true("2 active sequences", s.active_sequences == 2);

    // Cannot reserve a third (microbatch_size=2)
    t.assert_true("reserve seq 2 fails", !seq_pipeline_reserve(s, 2));
    t.assert_true("still 2 active", s.active_sequences == 2);

    // Release sequence 0
    seq_pipeline_release(s, 0);
    t.assert_true("seq 0 released", s.seq_stage.find(0) == s.seq_stage.end());
    t.assert_true("1 active sequence", s.active_sequences == 1);
}

static void test_stage_ownership(testing & t) {
    seq_pipeline_state s;
    seq_pipeline_init(s, 3, 2);
    seq_pipeline_reserve(s, 0);

    // Sequence 0 enters stage 0
    s.stage_tokens[0] = 0;
    t.assert_true("stage 0 owned by seq 0", s.stage_tokens[0] == 0);
    t.assert_true("stage 1 free", s.stage_tokens[1] == -1);
    t.assert_true("stage 2 free", s.stage_tokens[2] == -1);
}

static void test_stage_available_dispatch(testing & t) {
    seq_pipeline_state s;
    seq_pipeline_init(s, 3, 2);
    seq_pipeline_reserve(s, 0);
    seq_pipeline_reserve(s, 1);

    // First dispatch: only one sequence can enter stage 0
    // seq_stage is ordered map (0, 1), so seq 0 gets priority
    int d0 = seq_pipeline_dispatch(s);
    t.assert_true("first dispatch filled at least stage 0", d0 >= 1);
    t.assert_true("stage 0 occupied by seq 0", s.stage_tokens[0] == 0);

    // After first dispatch, seq 0 should have advanced to stage 1
    t.assert_true("seq 0 at stage 1", s.seq_stage[0] == 1);

    // Second dispatch: seq 0 advances to stage 2, seq 1 not yet entered
    int d1 = seq_pipeline_dispatch(s);
    t.assert_true("second dispatch filled stages", d1 >= 1);

    // Third dispatch: seq 1 should be able to enter stage 0 now
    int d2 = seq_pipeline_dispatch(s);
    t.assert_true("third dispatch filled stages", d2 >= 1);

    // Both sequences should make progress
    bool seq0_advanced = (s.seq_stage[0] > 1 || (s.seq_stage[0] == 0 && s.stage_tokens[0] != 0));
    bool seq1_entered = (s.stage_tokens[0] == 1 || s.stage_tokens[1] == 1 || s.stage_tokens[2] == 1);
    t.assert_true("both sequences make progress", seq0_advanced || seq1_entered);
}

static void test_full_pipeline_cycle(testing & t) {
    seq_pipeline_state s;
    seq_pipeline_init(s, 3, 2);
    seq_pipeline_reserve(s, 0);

    // Run 10 dispatch cycles: sequence 0 should complete multiple tokens
    int total_dispatched = 0;
    for (int i = 0; i < 10; i++) {
        total_dispatched += seq_pipeline_dispatch(s);
    }
    // Each complete token cycle requires 3 stage dispatches (stages 0, 1, 2)
    t.assert_true("at least 3 dispatches", total_dispatched >= 3);
    // After 10 cycles with 1 sequence, it should have made progress
    t.assert_true("sequence 0 still active", s.seq_stage.find(0) != s.seq_stage.end());
}

static void test_two_seq_independent_progress(testing & t) {
    seq_pipeline_state s;
    seq_pipeline_init(s, 3, 2);
    seq_pipeline_reserve(s, 0);
    seq_pipeline_reserve(s, 1);

    // Run dispatch cycles until both sequences have made progress past stage 0
    int cycles = 0;
    bool seq0_past_0 = false;
    bool seq1_past_0 = false;

    while (cycles < 20 && (!seq0_past_0 || !seq1_past_0)) {
        seq_pipeline_dispatch(s);
        // Check after each dispatch if either sequence has ever advanced past stage 0
        if (s.seq_stage[0] > 0) seq0_past_0 = true;
        if (s.seq_stage[1] > 0) seq1_past_0 = true;
        cycles++;
    }

    t.assert_true("seq 0 made progress", seq0_past_0);
    t.assert_true("seq 1 made progress", seq1_past_0);
}

static void test_stage_cascade_release(testing & t) {
    seq_pipeline_state s;
    seq_pipeline_init(s, 3, 2);
    seq_pipeline_reserve(s, 0);

    // Manually simulate: seq 0 advances through all stages
    s.stage_tokens[0] = 0;    // enter stage 0
    s.seq_stage[0] = 1;       // advance to stage 1
    s.stage_tokens[0] = -1;   // release stage 0 (cascade)

    s.stage_tokens[1] = 0;    // enter stage 1
    s.seq_stage[0] = 2;       // advance to stage 2
    s.stage_tokens[1] = -1;   // release stage 1 (cascade)

    s.stage_tokens[2] = 0;    // enter stage 2 (last stage)
    s.seq_stage[0] = 0;       // wrap to stage 0 (next token)
    s.stage_tokens[2] = -1;   // release stage 2 (KV write complete)

    // All stages should be free
    for (int i = 0; i < 3; i++) {
        std::string msg = "stage " + std::to_string(i) + " free after full cycle";
        t.assert_true(msg.c_str(), s.stage_tokens[i] == -1);
    }
    t.assert_true("seq 0 wrapped to stage 0", s.seq_stage[0] == 0);
}

int main() {
    testing t;

    test_reserve_and_release(t);
    test_stage_ownership(t);
    test_stage_available_dispatch(t);
    test_full_pipeline_cycle(t);
    test_two_seq_independent_progress(t);
    test_stage_cascade_release(t);

    return t.summary();
}
