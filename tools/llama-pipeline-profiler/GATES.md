# Validation gates and smell detectors

## Spike gates (from rpc-path-b-plus-spikes.md)

| ID | Criterion | diagnose.json field |
|----|-----------|---------------------|
| S4 | No corruption; pipeline + sched copies=4 | Manual / server log |
| S5 | `assembly_overlap_count > 0` during GEN | `gate_s5` |
| S1 | GEN `COPY_TENSOR` budget documented | rpc trace summary |
| S3 | `drain_flush_ms` down vs legacy | `drain_flush_ms` |
| B+6 | `overlap_pct >= overlap_target` (default 5%) | `gate_b6` |

S5 can PASS while B+6 FAILs (observed: 0.1-0.5% overlap on production topologies).

## GPU smell flags

Detected by cross-referencing NVIDIA CSV power vs utilization during GEN:

| Flag | Condition |
|------|-----------|
| `GPU_METRIC_MISMATCH` | High util, low power % TDP |
| `ORCHESTRATION_STALL` | Low util and low power (burst-then-idle) |
| `CLOCK_THROTTLE` | SM clock drops during run |
| `LOW_DUTY_CYCLE` | Sustained low util and power |

Trust sched `stall_ratio` when `GPU_METRIC_MISMATCH` is set.

## Idle / hang detection

Profiler `--watchdog-sec` (future): no decode progress within window.

Generic stall at load/slot-init: check last RPC op in trace before timeout.

## Parked outlier

RX6600 topologies are not production targets. Document only; do not optimize profiler defaults for `:50052`.