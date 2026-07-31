# D2 Implementation Agent Plan — Code Loop

**Phase:** D2 (Implementation Loop)  
**Goal:** Execute D1.1-D1.7 tickets with /tdd and /code-review  
**Agent persona:** `/implement` (drives /tdd internally)

---

## Mission Statement

Implement GPipe client scheduler following the ticket structure in `docs/tickets/path-d-tickets.md`. Each ticket uses test-driven development with:
1. Red test (fails)
2. Green implementation (passes)
3. Code review
4. If issues: hand back for fix

---

## Ticket Execution Order

| Order | Ticket | Skill | Test File |
|-------|--------|-------|-----------|
| 1 | D1.1 | /implement + /tdd | `tests/test-gpipe-state.cpp` |
| 2 | D1.2 | /implement + /tdd | `tests/test-gpipe-enabled.cpp` |
| 3 | D1.3 | /implement + /tdd | `tests/test-gpipe-env.cpp` |
| 4 | D1.4 | /implement + /tdd | `tests/test-gpipe-decode-skel.cpp` |
| 5 | D1.5 | /implement + /tdd | `tests/test-gpipe-init.cpp` |
| 6 | D1.6 | /implement + /tdd | `tests/test-gpipe-wait.cpp` |
| 7 | D1.7 | /implement + /tdd | `tests/test-gpipe-stage.cpp` |

---

## Per-Ticket Workflow

### For Each Implementation Ticket:

```
/handoff "<Ticket> Implementation"
→ New session

/implement
→ Drives /tdd internally:
  1. Create failing test in tests/
  2. Implement code to pass test
  3. Run test locally
  4. Call /code-review
→ If review passes:
  - Update TRACKING.md
  - Proceed to next ticket
→ If review fails:
  - Hand back to /implement for fix
```

---

## Test File Templates

### D1.1 Test Template (`tests/test-gpipe-state.cpp`)
```cpp
// Test llama_gpipe_state struct exists and has correct fields
TEST(gpipe_state, struct_exists) {
    llama_gpipe_state state = {};
    EXPECT_EQ(state.n_stages, 0);
    EXPECT_EQ(state.cur_stage, 0);
    EXPECT_EQ(state.microbatch_size, 0);
    EXPECT_EQ(state.enabled, false);
}
```

### D1.7 Test Template (`tests/test-gpipe-stage.cpp`)
```cpp
// Test stage state machine transitions
TEST(gpipe_stage, state_machine) {
    // Stage 0 -> Stage 1 -> Stage 0 transition
    // Verify event signaling
    // Verify KV-ready release
}
```

---

## Code Review Checklist

After each ticket:

- [ ] Code compiles with `cmake --build build-rocm-docker`
- [ ] Test passes
- [ ] No regressions on existing tests
- [ ] Follows existing code style
- [ ] Comments are minimal (per AGENTS.md)

---

## Handoff Pattern

Each ticket uses `/handoff` to preserve context:

```
/handoff "Implement D1.1: Add llama_gpipe_state struct"
→ Reads docs/tickets/path-d-tickets.md
→ Reads docs/path-d-spec.md section 3.1
→ Implements
→ Calls /code-review
```

---

## Output Deliverables

| File | Purpose |
|------|---------|
| `tests/test-gpipe-*.cpp` | Unit tests per ticket |
| Updated `src/*.cpp` files | Implementation |
| Updated `TRACKING.md` | Progress tracking |

---

## Success Criteria

- [ ] All D1.1-D1.7 tickets implemented
- [ ] All tests pass
- [ ] All code reviews pass
- [ ] No regressions on Path-B+ baseline

---

*Plan prepared for implementation agent invocation — 2026-07-10*