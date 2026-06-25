# Handover Prompt: Complete Path B B4 Final Verification & Close Checklist

**For:** Coding AI / Implementation Agent  
**Date:** 2026-06-25  
**Context:** Path B (Event-Based Pipeline Parallelism) core implementation + correctness fixes are complete and shipping in the `Path-B-Event-Support` git workspace (v4.2.2). Real benchmarks already prove it works and delivers value:
- 4B Config A: **~110 t/s** generation (primary regression target hit)
- llama-server 9B: **+7.3%** generation vs A1
- Server matrix (27B–36B): **+3% to +13.6%** vs A1

**Remaining work:** Only confirmatory verification steps for the last few B3/B4 checklist items. No new code changes are expected.

---

## Overall Goal

Finish the Path B checklist so we can declare **Path B fully complete** and ready for production cross-GPU deployments.

**Target outcome:**
- All items in `rpc-path-b-tracking.md` marked `[x]`
- `pipeline parallelism enabled` + `n_copies = 4` captured in logs
- 72B preset runs successfully
- Long generation (512 tokens) passes without deadlock/framing issues
- Single-GPU regression confirmed within ±2%
- Tracking file updated with final status

---

## Prerequisites (Assume Clean Environment)

- You are working in the project root that contains:
  - `atomic-llama-cpp-turboquant/` (the `Path-B-Event-Support` git workspace)
  - `scripts/` directory with `pathb-*.sh`, `rpc-server-bench.sh`, etc.
  - `/mnt/models/` containing at minimum:
    - `Qwen3.5-4B-Q4_K_M.gguf` (primary regression)
    - One 72B model (Kimi-Dev-72B-IQ4_XS or Qwen3-72B-Instruct.IQ4_XS)
- Docker images and named containers (`bench-rpc`, `bench-llama`) are available
- You have permission to run the benchmark scripts
- `GGML_SCHED_DEBUG=1` and `GGML_RPC_DEBUG=1` are usable

**Safety rule (permanent):** All commands must be non-destructive. Use the existing cleanup logic in the scripts. Never `rm -rf` source trees or Docker volumes unless explicitly labeled as DESTRUCTIVE.

---

## Step-by-Step Instructions (Execute in Order)

### Step 1: Capture Pipeline Debug Output (Closes Remaining B3 Items)

Run a short 4B test with scheduler debug enabled and extract the key lines.

```bash
GGML_SCHED_DEBUG=1 ./scripts/pathb-test.sh 4b 2>&1 | tee /tmp/pathb-sched-debug-4b.log

# Extract the critical lines
grep -E "pipeline parallelism enabled|n_copies = 4|event_record|event_wait|split" /tmp/pathb-sched-debug-4b.log | head -20
```

**Expected result:**
- Line containing `pipeline parallelism enabled`
- Line showing `n_copies = 4` (or `GGML_SCHED_MAX_COPIES`)

**Action after success:**
Edit `rpc-path-b-tracking.md` and mark the two remaining B3 items as complete. Add a note with the timestamp and log path.

### Step 2: Run 72B Preset (B4 Item)

```bash
./scripts/pathb-test.sh kimi72b
# or
./scripts/pathb-test.sh qwen72b
```

This uses the preset with:
- 10 minute load timeout
- `-ts 4,1`
- ctx 2048
- Appropriate cache settings

Monitor progress:
```bash
tail -f /tmp/pathb-runs/kimi72b.log
```

**Success criteria:**
- Container starts and becomes healthy within timeout
- Generation completes without RPC abort, death-loop, or framing errors
- Throughput is reasonable for 72B on this hardware split

Update tracking with result (PASS/FAIL + any notable numbers).

### Step 3: Long Generation Stress Test (512 tokens)

Use the low-level runner to force a longer generation and check stability.

```bash
./scripts/pathb-run-test.sh \
  --label long-gen-512 \
  --model /mnt/models/Qwen3.5-4B-Q4_K_M.gguf \
  --max-tokens 512 \
  --ctk q4_0 --ctv q4_0 --ngl 99 \
  --load-timeout 180
```

After completion, inspect the log:
```bash
grep -E "recv failed|framing|deadlock|error|abort" /tmp/pathb-runs/long-gen-512.log || echo "No obvious errors found"
```

**Success criteria:** Clean run to 512 tokens with no protocol or scheduler errors.

Update tracking.

### Step 4: Single-GPU Regression Test (No RPC)

Run without `--rpc` to confirm no regression from the Path B changes when pipeline parallelism is not active.

```bash
# On the NVIDIA 3060 Ti machine (or equivalent)
./bin/llama-cli \
  -m /mnt/models/Qwen3.5-4B-Q4_K_M.gguf \
  -p "The quick brown fox jumps over the lazy dog." \
  -n 128 \
  -ngl 99 \
  -ctk q4_0 -ctv q4_0 \
  --single-turn --simple-io 2>&1 | tee /tmp/single-gpu-regression-4b.log
```

Compare the generation t/s to your known Phase A2 single-GPU baseline for the same model/settings.

**Pass criterion:** Within ±2% of baseline.

Update tracking.

### Step 5: Optional but Recommended — Quick Config B Spot Check

If time permits, swap roles (NVIDIA as client, AMD as worker) and run the 4B test once. This is not strictly required for checklist closure but increases confidence.

### Step 6: Final Tracking Update & Summary

After all steps above, edit `rpc-path-b-tracking.md`:

1. Change the top status line to:
   ```
   ## Status: B1-B4 COMPLETE
   ```

2. Update "Current Phase" and "Phase Status" to `COMPLETE`.

3. Add a final row in the Implementation Log:
   ```
   | 2026-06-25 | B4 Final Verification | All remaining checklist items executed and passed. Pipeline debug captured. 72B preset successful. Long generation clean. Single-GPU regression within tolerance. Path B ready for production. |
   ```

4. Ensure every checkbox in B3 and B4 sections is now `[x]`.

5. Add a short "Conclusion" section at the bottom of the tracking file summarizing the measured gains.

---

## Pass / Fail Criteria

**Overall PASS if:**
- `pipeline parallelism enabled` and `n_copies = 4` appear in scheduler debug output
- 72B preset completes without crash or protocol error
- 512-token generation completes cleanly
- Single-GPU run is within ±2% of baseline
- Tracking file is fully updated with green checkboxes and final summary

**FAIL / Debug if any step produces:**
- RPC abort (`get_device_memory`, bytes_recv=0, etc.)
- Death-loop / garbage output
- Scheduler crash or assertion failure
- Significant regression on single-GPU path

In case of failure, enter debug mode:
```bash
GGML_RPC_DEBUG=1 GGML_SCHED_DEBUG=1 ./scripts/pathb-run-test.sh --label debug-fail ...
```
Then analyze the combined log.

---

## Files You May Need to Touch

- `rpc-path-b-tracking.md` (mandatory — update after every major step)
- `/tmp/pathb-runs/*.log` (read-only for analysis)
- `scripts/pathb-*.sh` (only if you need to adjust a preset — document any change)

**Do not modify** core `ggml-rpc.cpp` unless a new bug is discovered (unlikely given current benchmark success).

---

## Final Deliverable

When finished, the repository should have:

- Updated `rpc-path-b-tracking.md` with all items green and a clear "Path B COMPLETE" status
- New log files under `/tmp/pathb-runs/` or `patch/bench-results/` proving the verification steps
- (Optional) A short note in `RPC_PATH_AB_OPTIMIZATION_REPORT.md` or the tracking file summarizing that Path B is now production-ready

---

## Tone & Style for This Handover

- Be precise with commands — copy-paste ready.
- Assume the coding AI has access to the same Docker setup and model files as the previous runs.
- Prioritize **verification over new code**.
- After each successful step, explicitly instruct to update the tracking file before moving to the next step.
- Keep responses structured with clear headings, code blocks, and expected output.

You now have everything needed to close Path B. Execute the steps above in order, update the tracking file after each verification, and we will have a fully validated, high-performance RPC pipeline parallelism implementation.

**Start with Step 1 (pipeline debug capture).** Report back with the extracted log lines and the updated tracking file diff when complete. Good luck!