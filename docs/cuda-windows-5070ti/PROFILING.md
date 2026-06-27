# GPU Underutilization Profiling (Config F)

Investigation of why Qwen3.6-35B-A3B MoE (IQ4_NL_XL) shows high VRAM use but low `nvidia-smi` utilization during multi-RPC inference.

## Executive summary

**The donkey is limping because of serial RPC orchestration, not because PCIe, NIC, or system RAM are saturated.**

The RX6600 third hop is actively harmful: dropping it raised throughput **37 -> 49 t/s** (+32%) on the same model.

Low `utilization.gpu` is partly a **measurement blind spot** (1s `nvidia-smi` average vs bursty 500ms samples). During gen bursts, the 5070 Ti briefly hits **78 W (26% TDP)** and **25% SM util** before returning to idle between RPC round-trips.

## Profile runs (2026-06-27)

| Run | Topology | ts | G (t/s) | 5070 peak TDP% | 5070 max util% | 5070 duty@20%TDP |
|-----|----------|-----|---------|----------------|----------------|------------------|
| `profile-f-36b-nl-base` | 5070 + 5060 + 6600 | 30,12,58 | **37.0** | n/a* | n/a* | n/a* |
| `profile-e-36b-nl` | 5070 + 5060 (Config E) | 50,50 | **42.4** | 18.7% | 5% | 0% |
| `profile-f-36b-nl-no6600` | 5070 + 5060 only | 50,50 | **48.9** | 26.1% | 25% | 3.8% |

\*First base run lacked Windows GPU CSV (monitor job flushed late); fixed in later runs.

Prior matrix reference: `config-f-36b-nl-gpu` averaged **~40 t/s** (64-token runs).

## Bottleneck classification

### 1. Serial RPC critical path (PRIMARY)

Evidence:
- Pipeline parallelism enabled, `sched copies = 4` on all runs
- Config F: `graph splits = 4` (3 RPC/CUDA backends + boundaries)
- Config E / no-6600: `graph splits = 3`
- Prior docs: 7-22+ blocking RPC RTTs per token remain even with Path B events
- 500ms telemetry shows **burst-then-idle** power/util pattern (orchestration stall signature)

Verdict: **Dominant term.** GPUs compute in short bursts; client waits on TCP RPC between graph segments.

### 2. RX6600 straggler / third hop (SECONDARY, actionable)

Evidence:
- F (3 devices): G=37.0
- E (2 devices): G=42.4 (+15%)
- F-no-6600 (2 devices, ts=50,50): G=48.9 (+32% vs F)

Verdict: **6600 adds a serial hop without enough compute payoff** for this model. Config F should not use `:50052` for 35B/36B A3B MoE when 5070+5060 VRAM suffices.

### 3. Measurement artifact (CONTRIBUTING)

Evidence:
- Legacy `pathb-gpu-monitor-win.sh` polls every **2s**; missed bursts
- 1s `dmon` minimum on Windows GeForce
- 500ms `nvidia-smi --query-gpu` during no-6600 gen: max **25% util**, **78 W** vs avg **26 W**, **1% util**

Verdict: **Do not trust 0-2% util alone.** Use power duty cycle at >=500ms and TDP-normalized thresholds.

### 4. PCIe / NIC / RAM data-path (RULED OUT for GEN)

| Path | LOAD | GEN | Verdict |
|------|------|-----|---------|
| System RAM | pages/sec spikes to 75k-131k (mmap) | pages/sec <3k | **GEN ruled out** |
| NIC | - | peak ~1.6 Gbps counter sum | **Not wire-saturated** (RTT-bound RPC) |
| PCIe | idle ~6-14 MB/s | not measured on base | **Unlikely** at 40-50 t/s tensor volume |

Cross-path matrix during GEN: low CPU/GPU/PCIe most samples -> **orchestration stall**, not upstream DMA/NIC saturation.

## Scheduler trace (profile-f-36b-nl-base)

```
pipeline parallelism enabled
sched_reserve: graph splits = 4
sched_reserve: reserve took 1468.19 ms, sched copies = 4
```

Tensor placement (full GPU, ngl=99):
- CUDA0: 10186 MiB weights
- RPC0 (5060): 5737 MiB
- RPC1 (6600): 2148 MiB

## Recommendations (ranked)

1. **Drop 6600 from Config F for 35B/36B A3B MoE** -- use `--rpc 192.168.8.176:50051` + CUDA0, `ts=50,50` or VRAM-calc optimal 2-device split. Measured **+32% G** vs 3-device F.

2. **Prefer Config E topology** for MoE models that fit 5070+5060 VRAM (38.5 GB combined without 6600).

3. **Do not tune PCIe/NIC** -- data-path headroom is ample; effort belongs in RPC latency (Path A/C) or topology.

4. **Telemetry**: use `-Profile` bench mode (`rpc-server-bench.ps1 -Profile`) with 500ms Windows GPU query + 45s post-gen flush. Compare **power duty cycle % of TDP**, not raw util%.

5. **Optional engineering** (if more speed needed): Path C server-side multi-GPU aggregation on remus (5060+6600 as one RPC worker), or Path A RTT batching.

## Artifacts

| Path | Contents |
|------|----------|
| `benchmarks/profile-f-36b-nl-base/` | Base F run + phase.log + RAM/NIC |
| `benchmarks/profile-e-36b-nl/` | Config E isolate + win-5070-dmon.csv |
| `benchmarks/profile-f-36b-nl-no6600/` | 2-device F + best telemetry |
| `benchmarks/telemetry/probe-notes.txt` | PCIe counter probe results |

## Commands

```powershell
# Single profile run
.\scripts\cuda-windows-5070ti\rpc-server-bench.ps1 `
  -Label profile-f-36b-nl-no6600 -Config config-f `
  -RpcEndpoint 192.168.8.176:50051 `
  -ModelPath D:\models\Qwen3.6-35B-A3B-UD-IQ4_NL_XL.gguf `
  -TensorSplit 50,50 -Profile -GenTokens 256 -Runs 1 -EnsurePathbRpc

# Parse summary
.\scripts\cuda-windows-5070ti\pathb-profile-parse.ps1 -ProfileDir docs\cuda-windows-5070ti\benchmarks\profile-f-36b-nl-no6600
```

## Open items (not run this session)

- `profile-f-36b-nl-ts1080` (ts=12,8,80) tensor-split sweep
- MTP decomposition run
- VS2022/Nsight native SDK session (CPU thread hotspots on `llama-server.exe`)
- remus `nvidia-smi dmon` + `rocm-smi` CSV during gen (WSL job reliability improved with 45s flush)