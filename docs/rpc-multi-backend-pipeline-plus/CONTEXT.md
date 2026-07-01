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
Linux ROCm cluster host at `192.168.8.108`; primary gate **client** (7900XTX) with optional local RPC worker (3060 Ti on `:50051`) and access to `/mnt/models`.
_Avoid_: cluster, server

**Triton**:
Windows CUDA spike worker at `192.168.8.23`; RTX 3090 RPC on `:50054` (3070 on `:50055`, parked for 35B MoE).
_Avoid_: worker, remote GPU

**Mitigation ladder**:
Ordered B+8 through B+13 code changes tested via OFF-bisects before declaring a structural overlap ceiling.
_Avoid_: fix list, optimization pass

**Structural ceiling**:
Documented verdict that M3 overlap cannot be reached within Path-B+ scope after the mitigation ladder is exhausted on 2-GPU and 4-GPU topologies.
_Avoid_: giving up, hard limit

**Gate client**:
The host that runs `llama-pipeline-profiler` for a bench gate preset. Today usually romulus (7900XTX); any synced node may serve this role.
_Avoid_: master, primary server

**Topology-agnostic bench**:
A profiler run defined only by label, RPC endpoints, and env flags — not by which physical host executes the client binary.
_Avoid_: portable test, host-independent