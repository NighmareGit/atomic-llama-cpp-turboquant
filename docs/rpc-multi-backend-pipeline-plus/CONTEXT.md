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