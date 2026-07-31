# Ticket A Result: TCP Transport — H1 Investigation

**Date:** 2026-07-24
**Test:** Disable UDP transport (`GGML_RPC_UDP=0`) on Qwen3.5-122B-A10B 5-GPU layer-split
**Script:** `run-122b-tcp.sh` (copy of `run-122b-5gpu.sh`)
**Config:** `-sm layer`, `-ngl 43`, `-tensor-split 13,3,9,15,3`, PPLUS + WAVEFRONT_CROSS + RPC_GET_DEFER + W2 active

## Output Samples

| # | Prompt | max_tokens | Output |
|---|--------|-----------|--------|
| 1 | "Once upon a time," | 50 | `//////////////////////////////////////////////////` |
| 2 | "The capital of France is" | 50 | `//////////////////////////////////////////////////` |
| 3 | "Write a haiku about programming:" | 50 | `//////////////////////////////////////////////////` |

**Verdict: All 3 outputs are GARBLED** — identical slash characters, zero coherent English.

## UDP Error Count

```
grep -c "send_udp" /tmp/llama-server-122b-tcp.log
→ 0
```

Zero UDP errors confirmed. TCP transport is clean.

## Throughput

| Test | prompt eval | eval (tg) |
|------|-------------|-----------|
| "Once upon a time," | 4.59 t/s | 13.68 t/s |
| "The capital of France is" | 6.87 t/s | 12.19 t/s |
| "Write a haiku about programming:" | 8.78 t/s | 12.51 t/s |

Average generation throughput: **~12.8 tg t/s** (consistent with the ~16.5 t/s measured earlier with UDP — slight drop expected from TCP overhead).

## Verdict

**H1 FALSIFIED.** Disabling UDP transport (switching to TCP) eliminated all `send_udp` errors but did NOT fix garbled output. The model still produces identical `//////////////////////////////////////////////////` on every prompt, regardless of transport protocol.

The garbled output pattern is interesting: it's not random unicode but a consistent, repetitive character (`/` = ASCII 0x2F). This suggests a systematic corruption rather than random transport loss. Possible mechanisms:
- A specific tensor/weight offset is wrong (e.g., logits buffer pointing to token ID 47)
- Pipeline flag interaction corrupting compute graph execution (consistent across runs)
- KV-cache state corruption (PPLUS multi-GPU known issue)

## Next Step

Proceed to **Ticket B: Strip Pipeline Flags** — disable PPLUS, WAVEFRONT_CROSS, RPC_GET_DEFER, W2 to test H2 (flag incompatibility at 5-GPU scale).
