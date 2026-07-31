# D1 To-Tickets Agent Plan — Work Breakdown

**Ticket:** D1 (Work Breakdown Phase)  
**Type:** task  
**Goal:** Split specification into agent-ready tickets with blocking edges  
**Agent persona:** `/to-tickets` (turn spec into independently-grabbable issues)

---

## Mission Statement

Convert `docs/path-d-spec.md` into a set of GitHub-ready local tickets (`docs/tickets/path-d-tickets.md`) where each ticket is:
- Fully specified with acceptance criteria
- Has blocking edges to prior tickets
- Can be implemented by an agent without human context

---

## Input Materials

| File | Purpose |
|------|---------|
| `docs/path-d-spec.md` | Master specification |
| `docs/wayfinder/D0.5-implementation-seam.md` | Implementation details |
| `docs/adr/0002-gpipe-kv-ordering.md` | Architecture decisions |
| `docs/wayfinder/D0.2-split-topology-map.md` | Split boundaries |

---

## Execution Steps

### Step 1: Parse Specification
Read `docs/path-d-spec.md` to extract:
- Implementation requirements (sections 4.1-4.2)
- Test requirements (section 5)
- API contracts (section 3)

### Step 2: Create Ticket Structure

Each ticket follows this format:

```markdown
### D1.X -- <Ticket Title>

**Type:** implementation | test
**Blocks:** D1.Y, D1.Z (or "Review" for final)
**Blocked by:** D1.W

**Goal:** <One sentence description>

**Acceptance Criteria:**
- [ ] Criterion 1
- [ ] Criterion 2

**Implementation Notes:**
- File: `src/llama-context.cpp:XXXX`
- Reference: D0.5 section X.Y

**Test Requirements:**
- Test: `scripts/test-path-d-d1x.sh`
- Metric: `global_3bk_pct >= 1%`
```

### Step 3: Write Output

Create `docs/tickets/path-d-tickets.md` with all tickets in order.

---

## Expected Ticket Structure

Based on the specification, create these tickets:

### Implementation Tickets

| ID | Title | Blocks | Blocked By |
|----|-------|--------|------------|
| D1.1 | Add llama_gpipe_state struct | D1.2 | none |
| D1.2 | Implement llama_gpipe_enabled() helper | D1.3 | D1.1 |
| D1.3 | Add GGML_SCHED_GPIPE env var handling | D1.4 | D1.2 |
| D1.4 | Implement llama_decode_gpipe_impl() skeleton | D1.5 | D1.3 |
| D1.5 | Implement ggml_sched_gpipe_init() | D1.6 | D1.4 |
| D1.6 | Implement ggml_sched_gpipe_wait() | D1.7 | D1.5 |
| D1.7 | Implement stage state machine | D2.1 | D1.6 |

### Test Tickets

| ID | Title | Blocks | Blocked By |
|----|-------|--------|------------|
| D2.1 | Correctness tests (logits, KV) | Review | D1.7 |
| D2.2 | Performance tests (overlap, G) | Review | D2.1 |
| D2.3 | Regression tests (Path-B+ compat) | Review | D2.2 |

---

## Output Deliverables

| File | Purpose |
|------|---------|
| `docs/tickets/path-d-tickets.md` | Work breakdown with blocking edges |
| `docs/tickets/D1.X-*.md` | Individual ticket files (optional) |

---

## Success Criteria

- [ ] All spec requirements mapped to tickets
- [ ] Blocking edges correctly specified
- [ ] Acceptance criteria for each ticket
- [ ] Test requirements linked to tickets
- [ ] Tickets are agent-implementable

---

*Plan prepared for `/to-tickets` agent invocation — 2026-07-10*