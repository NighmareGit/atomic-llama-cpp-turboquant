# B+6 ladder execution after B+9 null bisect (2026-07-01)

Grill-with-docs session locked the next 1-2 week plan after n=384 triton data showed B+9 OFF (`GGML_RPC_EVENT_DEFER_BARRIER=0`) produced no overlap delta (0.2% canonical vs 0.2% no-defer).

**Decisions:**

1. **Next bisect:** B+8 OFF (`GGML_PIPELINE_BARRIER_PARTIAL=0`) on 2-GPU triton n=384, then B+10 OFF — before 4-GPU work.
2. **Execution host:** Romulus native primary (sync branch first); remus-docker fallback. Future runs are topology-agnostic (label + RPC + flags define the bench, not the client host).
3. **Phase 1.1 instrumentation:** Parallel, non-blocking — re-parse existing traces first; new bench only if gaps appear.
4. **validate-rpc:** Parallel hygiene track; bisects use `PROFILER_SKIP_VALIDATE=1` until strace fix lands.
5. **After 2-GPU B+8+B+10 OFF if M1 still FAIL:** Write partial 2-GPU verdict in TRACKING, then run 4-GPU (`b6-4gpu-g` on romulus). Structural ceiling doc only after 4-GPU ladder also fails M1.

**Considered:** Skip to ceiling doc after 2-GPU null bisects; jump to Path C. Rejected — PLAN order requires B+7a' on 4-GPU; Path C needs explicit scope approval.