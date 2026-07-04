# rpc-multi-backend-pipeline-plus

Mission vocabulary for Path-B Plus multi-backend RPC orchestration, B+6 overlap gate, and profiler-led mitigation ladder.

## Language

**Bench gate**:
A fixed profiler preset (label, topology, n=384 canonical) whose `diagnose.json` is compared against M1/M3 overlap milestones.
_Avoid_: test case, benchmark run

**Bisect**:
A controlled rollback run that sets one mitigation env flag to `0` while holding topology and Plus=1 constant, to measure that flag's marginal effect on overlap.
_Avoid_: A/B test, experiment

**Romulus**:
Linux dual-GPU host at `192.168.8.108`: RX **7900 XTX** (ROCm gate **client**) and RTX **3060 Ti** (CUDA RPC docker `127.0.0.1:50051`, container `pathb-rpc-romulus`). Both GPUs live on romulus — the 3060 is **not** on remus.
_Avoid_: cluster, server

**Canonical cluster repo path**:
Git checkout location on every Linux node: `~/projects/atomic-llama-cpp-turboquant`. Docker images build from gitea clone inside `~/docker/Atomic-Llama-*-PathB/build.sh` (independent of host checkout path). See `rpc-patch/patch/CLUSTER-NODE-LAYOUT.md`.
_Avoid_: turboquant root, repo dir

**Romulus-local stack**:
Single-host 2-GPU path: native ROCm `llama-server` on 7900 XTX + docker `pathb-rpc-romulus` on 3060 Ti (`127.0.0.1:50051`). No remus/triton/jupiter. Launchers: `scripts/romulus-local-up.sh`, `scripts/romulus-local-build.sh`. Default model: Qwen3.6 35B **MTP** GGUF under `/mnt/models`. Default listen `0.0.0.0:8080`. Seq-repair on by default.
_Avoid_: local cluster, single-node bench

**Romulus-local failure composite**:
Observed symptom mix on dirty/stale romulus checkouts: input echo (`this is a test` repeated), slash garbage (`/////...`), TSC morpheme stutter, and intermittent RPC docker crash — not a single hang class. Treat as env + checkout hygiene before new code changes.
_Avoid_: random garbage, model broken

**Seq-repair knob**:
`GGML_PIPELINE_MULTI_BACKEND_SEQ=1` — TSC repair bundle (R1+R2). Must be set on **both** client and RPC worker processes in multi-backend topologies; not default-on yet.
_Avoid_: MULTI_BACKEND_SEQ env, pipeline fix flag

**Remus**:
Linux dual-GPU host at `192.168.8.176`: RTX **5060 Ti** CUDA RPC on `:50051` (active worker) and RX **6600** on `:50052` (present per `rocm-smi`, **excluded** from production 4-GPU and 35B+ MoE paths).
_Avoid_: worker, remote GPU

**Triton**:
Ubuntu 24.04 dual-NVIDIA host at `192.168.8.23`: RTX **3090** RPC on `:50054` (**4-GPU RPC2**); RTX **3070** on `:50055` (**5-GPU Linux RPC3**). Both cards count toward the all-Linux 5-GPU path (`b6-5gpu-g`). Ops: `scripts/b6-gate-triton-*.sh`.
_Avoid_: worker, remote GPU

**Mitigation ladder**:
Ordered B+8 through B+13 code changes tested via OFF-bisects before declaring a structural overlap ceiling.
_Avoid_: fix list, optimization pass

**Structural ceiling**:
Documented verdict that M3 overlap cannot be reached within Path-B+ scope after the mitigation ladder is exhausted on 2-GPU and 4-GPU topologies. Recorded in TRACKING 2026-07-01.
_Avoid_: giving up, hard limit

**Path C (out of scope)**:
Server-side scheduling / distributed orchestration track. **Not implemented** on path-b-plus; may be referenced for ideas only.
_Avoid_: next step, fallback plan

**Gate client**:
The host that runs `llama-pipeline-profiler` for a bench gate preset. Today usually romulus (7900XTX); any synced node may serve this role.
_Avoid_: master, primary server

**Topology-agnostic bench**:
A profiler run defined only by label, RPC endpoints, and env flags — not by which physical host executes the client binary.
_Avoid_: portable test, host-independent

**Layer A (depth-2)**:
Speculative draft preparation layer. Enables preparing the draft for iteration k+1 concurrently with target decode of iteration k inside the llama-server loop (MTP/NextN). Primary knob: `LLAMA_PIPELINE_DEPTH2`. Orthogonal to Layer B.
_Avoid_: MTP layer, speculative pipeline (ambiguous)

**Layer B (Path-B-Plus)**:
Multi-backend graph-split orchestration and copy-slot pipelining across local + RPC devices using events. Primary knob: `GGML_PIPELINE_PLUS` + the B+ mitigation ladder. This is the primary focus of the current mission.
_Avoid_: RPC layer (too narrow)

**trace_id**:
Monotonically increasing uint64 correlation identifier (one per `llama_decode` or micro-batch). Propagated through sched, RPC wire (EVENT_RECORD), pipeline barrier, and Layer A traces. Enables exact joins in `pathb-hotpath-summary.sh` without timestamp heuristics.
_Avoid_: decode_id (resets on perf_reset), session id, correlation key

**Blessed configuration**:
The minimal set of env flags documented as production defaults for a given topology (see CONFIGURATION.md and the 4-GPU baseline env.txt). Changes outside this set should be explicitly justified.
_Avoid_: default flags, recommended flags (too vague)

**Deprecation policy (this workstream)**:
When a flag is superseded, it emits a one-time warning naming the flag, the superseding work, and the planned removal phase. Flags remain functional until removal phase. Removal happens only after the successor is validated on the gate.
_Avoid_: just remove it, soft delete

**Lateral addition**:
A supporting change (observability, hygiene, API surface, or new mitigation) that improves the ability to attack the core mission (overlap gate / straggler diagnosis) or reduces future friction, without itself moving the primary metric.
_Avoid_: side quest, unrelated feature

### Gated DeltaNet / TSC diagnostics

**Total Semantic Collapse (TSC)**:
Generation degenerates into a stable short token cycle (infinite repetition loop) with collapsed output diversity.
_Avoid_: repetition penalty failure, EOS miss, bad prompt

**State divergence witness**:
Evidence that recurrent Gated DeltaNet state (`GGML_OP_GATED_DELTA_NET` I/O, especially the `s` state tensor) differs across backends at the same decode step.
_Avoid_: tensor hash telemetry, SET_TENSOR_HASH bucket

**TSC symptom**:
The observable output cycle that triggers investigation; diagnosed from the token stream, not from scheduler split timing alone.
_Avoid_: semantic collapse, model collapse

**Fox bench**:
The default rpc-server-bench prompt (`The quick brown fox jumps over the lazy dog.`) labeled `fox` in `.result` previews.
_Avoid_: default-fox file, pangram test

**Output stutter**:
Repetitive morphemes in a completed generation preview (e.g. `pangramramramram`, `known-known-known`) while throughput gates still pass and the run finishes `n_predict` tokens.
_Avoid_: TSC, infinite loop (unless generation actually fails to terminate)

**Reproduction vector**:
Pinned launcher + env + prompt + artifact paths that reproduce a failure class; for cluster fox benches see `trace-g-4gpu-primary-r3` meta in `rpc-patch/patch/bench-results/`.
_Avoid_: repro steps, test vector

**TSC validation matrix**:
Controlled A/B runs that falsify misdiagnosis: same fox prompt across topology (2-GPU / 3-GPU / 4-GPU) and Plus on/off, scoring output stutter vs true non-termination before any GDB witness work.
_Avoid_: retest plan, bug hunt checklist

**Plus=1 TSC bugfix track**:
Mitigation bisect (strategy D) before any GDN pin or selective-sync patch. See [BUGFIX-plus1-tsc-semantic-collapse.md](BUGFIX-plus1-tsc-semantic-collapse.md).

**TSC composite gate**:
Pass/fail uses hang detection (A), automated stutter metrics (B), and one human spot-check per topology class (C); no single signal alone is sufficient.
_Avoid_: coherence check, preview lint

**Cluster inventory**:
Live node layout, deploy dirs, and bring-up commands live in gitignored `.scratch/cluster-inventory.md`; credentials in `.scratch/cluster-access.env`.
_Avoid_: cluster map, SSH cheatsheet (in repo root)

**Cluster ops handover**:
Versioned runbook for docker lifecycle, build paths, bench launchers, and artifact locations (`rpc-patch/patch/HANDOVER-CLUSTER-OPS.md`).
_Avoid_: setup guide, infra README