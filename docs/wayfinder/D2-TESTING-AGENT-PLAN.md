# D2 Testing Agent Plan

**Phase:** D2 (Testing)  
**Goal:** Verify GPipe implementation correctness and performance  
**Agent persona:** `/research` (test planning and execution)

---

## Mission Statement

Execute the test plan for GPipe implementation:
1. Run unit tests for all GPipe functions
2. Execute correctness tests (logits, KV fill)
3. Run performance benchmarks
4. Verify no regressions on Path-B+ baseline

---

## Input Materials

| File | Purpose |
|------|---------|
| `docs/wayfinder/HANDOFF-TESTING-PHASE.md` | This handoff file |
| `docs/path-d-spec.md` | Specification (test requirements) |
| `tests/test-gpipe-*.cpp` | Test files to run |
| `docs/wayfinder/TRACKING.md` | Current status |

---

## Execution Steps

### Step 1: Build Tests

```bash
cd /home/hunter/projects/path-d-gpipeline-assembly-line

# Configure
cmake -B build-rocm-docker -S . \
  -DGGML_BACKEND_HIP=ON \
  -DGGML_HIP_DYNAMIC_LINK=ON \
  -DCMAKE_BUILD_TYPE=Release

# Build test targets
cmake --build build-rocm-docker --target test-gpipe-state test-gpipe-enabled test-gpipe-env test-gpipe-init test-gpipe-wait test-gpipe-stage-full -j$(nproc)
```

### Step 2: Run Unit Tests

```bash
cd build-rocm-docker

# Run all GPipe tests
ctest -R test-gpipe --output-on-failure

# Run individual tests
./test-gpipe-state
./test-gpipe-enabled
./test-gpipe-env
./test-gpipe-init
./test-gpipe-wait
./test-gpipe-stage-full
```

### Step 3: Correctness Tests

Verify:
- [ ] All tests pass
- [ ] No crashes on invalid inputs
- [ ] Struct initialization correct
- [ ] Env var handling correct

### Step 4: Performance Tests

If unit tests pass, run romulus-local benchmark:

```bash
# Start RPC worker
bash scripts/romulus-local-up.sh --rpc-only

# Run server with GPipe
export GGML_SCHED_GPIPE=1
export GGML_PIPELINE_PLUS=1
export GGML_SCHED_TRACE=1
export GGML_RPC_TRACE=1

bash scripts/romulus-local-up.sh \
  --model /mnt/models/Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf \
  --ts 50,50 \
  --ctx 8192 \
  --mtp on
```

Collect metrics:
- `global_3bk_pct` from scheduler trace
- `overlap_pct` from assembly overlap count
- G (tokens/sec) from server output

### Step 5: Regression Tests

Run comparison with GPipe OFF:

```bash
# GPipe OFF (baseline)
GGML_SCHED_GPIPE=0 bash scripts/romulus-local-up.sh --model /mnt/models/Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf --ts 50,50 --ctx 8192 --mtp on

# Compare G values
```

---

## Expected Outputs

| File | Purpose |
|------|---------|
| `benches/path-b-plus/gpipe-test-results/` | Test output and metrics |
| Updated `docs/wayfinder/TRACKING.md` | Status after testing |

---

## Success Criteria

- [ ] All `test-gpipe-*` tests pass
- [ ] No regressions on Path-B+ baseline
- [ ] `global_3bk_pct >= 25%` achieved
- [ ] `overlap_pct >= 5%` achieved

---

## Handoff Command

```
/handoff "Testing Phase: Run GPipe tests and verify performance"
→ Read docs/wayfinder/HANDOFF-TESTING-PHASE.md
→ Follow this plan
→ Update docs/wayfinder/TRACKING.md
```

---

*Plan prepared for `/research` agent invocation — 2026-07-10*