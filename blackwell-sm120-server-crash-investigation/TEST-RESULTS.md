# Test Results Log

**Status: CLOSED (2026-07-09)** - post-fix verification complete. Pre-fix section is historical.

## Post-Fix Verification (2026-07-09)

All configs that previously crashed now pass on RTX 5070 Ti with `build-cuda-b-bin/bin/Release/`:

| Test | Config | Result | Log |
|------|--------|--------|-----|
| smoke | `-c 2048 -ctk turbo3` default np4 | PASS | `smoke-test.err` |
| inv-4096 | `-c 4096 --flash-attn off --spec-type none` | PASS | `cfg-inv-4096.err` |
| inv-16384 | `-c 16384` fit on | PASS | `cfg-inv-16384.err` |
| gen-500 | np4, 500 tokens, math prompt | PASS | `gen-np4-500.err` |
| maxverb | `-lv 2147483647` fit on np4 | PASS | `repro-maxverb.err` |
| official | `smoke-llama-server.ps1` | SMOKE_OK | `docs/cuda-windows-5070ti/benchmarks/20260709-100701/` |
| mtp-20k | `-c 20480 -np 1 -ctk turbo3 -ctv q8_0 --spec-type draft-mtp` | PASS | `mtp-20k-test/server-v2.err` |

**MTP 20k metrics** (`mtp-20k-test/response-v2.json`): draft_n=269, draft_n_accepted=246 (91.4%), 129 t/s, n_slots=1.

---

## llama-server Tests (Pre-Fix - All Failed)

### Test 1: Default Configuration
**Date**: 2026-07-09
**Args**: `--ctx-size 16384 -ngl 99 -t 4`
**Model**: Qwen3.5-9B-MTP-Q4_K_M.gguf
**Result**: CRASH - during startup, before any request
**Log**: `models/server-16384.log`

### Test 2: Long Context (16k)
**Date**: 2026-07-09
**Args**: `--ctx-size 16384`
**Model**: Qwen3.5-9B-MTP-Q4_K_M.gguf
**Result**: CRASH - during startup
**Log**: `models/server-longctx.log`

### Test 3: Medium Context (8k)
**Date**: 2026-07-09
**Args**: `--ctx-size 8192`
**Model**: Qwen3.5-9B-MTP-Q4_K_M.gguf
**Result**: CRASH - during startup
**Log**: `models/server-8192.log`

### Test 4: Small Context + No MTP
**Date**: 2026-07-09
**Args**: `--ctx-size 4096 --spec-type none`
**Model**: Qwen3.5-9B-MTP-Q4_K_M.gguf
**Result**: CRASH at n_decoded=100
**Log**: `models/server-4096-nomtp.log`
**Notes**: First response partially captured

### Test 5: Debug Build Style
**Date**: 2026-07-09
**Args**: `--ctx-size 4096`
**Model**: Qwen3.5-9B-MTP-Q4_K_M.gguf
**Result**: CRASH at n_decoded=100
**Log**: `models/server-crash-debug.log`

### Test 6: No mmap
**Date**: 2026-07-09
**Args**: `--ctx-size 4096 --no-mmap`
**Model**: Qwen3.5-9B-MTP-Q4_K_M.gguf
**Result**: CRASH at n_decoded=100
**Log**: `models/server-nocheck.log`

### Test 7: Flash Attention Off
**Date**: 2026-07-09
**Args**: `--ctx-size 4096 --flash-attn off`
**Model**: Qwen3.5-9B-MTP-Q4_K_M.gguf
**Result**: CRASH at n_decoded=100
**Log**: `models/server-faoff.log`
**Response**: `models/resp-faoff.json` (1442 bytes) - successful response before crash

### Test 8: Speculative Decoding Off
**Date**: 2026-07-09
**Args**: `--ctx-size 4096 --spec-type none`
**Model**: Qwen3.5-9B-MTP-Q4_K_M.gguf
**Result**: CRASH at n_decoded=100
**Log**: `models/server-specnone.log`

### Test 9: Single Thread
**Date**: 2026-07-09
**Args**: `--ctx-size 4096 -t 1`
**Model**: Qwen3.5-9B-MTP-Q4_K_M.gguf
**Result**: CRASH at n_decoded=100
**Log**: `models/server-1t.log`

### Test 10: Four Threads
**Date**: 2026-07-09
**Args**: `--ctx-size 4096 -t 4`
**Model**: Qwen3.5-9B-MTP-Q4_K_M.gguf
**Result**: CRASH at n_decoded=100
**Log**: `models/server-4t.log`

### Test 11: Eight Threads
**Date**: 2026-07-09
**Args**: `--ctx-size 4096 -t 8`
**Model**: Qwen3.5-9B-MTP-Q4_K_M.gguf
**Result**: CRASH at n_decoded=100

### Test 12: Invalid Flag -concurrency
**Date**: 2026-07-09
**Args**: `--concurrency 1`
**Result**: exit 1 (flag not recognized)

### Test 13: Invalid Flag -slot-flash-prefill
**Date**: 2026-07-09
**Args**: `--slot-flash-prefill off`
**Result**: exit 1 (flag not recognized)

### Test 14: Invalid Flag -log-err-std
**Date**: 2026-07-09
**Args**: `--log-err-std`
**Result**: exit 1 (flag not recognized)

### Test 15: Invalid Flag -verbose
**Date**: 2026-07-09
**Args**: `--verbose`
**Result**: exit 1 (flag not recognized)

---

## llama-cli Tests (All Passed)

### Test 1: Math/Logic/Coding Combined
**Date**: 2026-07-09
**Args**: `-m D:\models\Qwen3.5-9B-MTP-Q4_K_M.gguf -n 512 -ngl 99 -c 16384 -t 4 --flash-attn off --no-warmup`
**Prompt**: Train catch-up math + 5-house logic + Python palindrome
**Result**: PASS
- Prompt tokens: 660.6 t/s
- Generation: 118.6 t/s
- Output: Correct math answer (4 hours), correct logic (yellow-green-red-blue-white), valid Python code
**Log**: `models/test-qwen-mtp-stdout.txt` (20 MB)

### Test 2: Long Context Stability
**Date**: 2026-07-09
**Args**: `-c 16384 --flash-attn off`
**Model**: Qwen3.5-9B-MTP-Q4_K_M.gguf
**Result**: PASS
- No NaN corruption in output
- KV cache integrity maintained

### Test 3: Ornith 9B Long Context
**Date**: 2026-07-09
**Args**: `-c 65536`
**Model**: D:\models\ornith-1.0-9b-Q5_K_M.gguf
**Result**: PASS
- No NaN corruption
- Output quality maintained at 64k context

---

## Response Captures

### resp-noc.json (1441 bytes)
- Config: Default (no special flags)
- Speed: 121.6 t/s
- Content: Math problem solution
- Status: Successful capture before crash

### resp-faoff.json (1442 bytes)
- Config: --flash-attn off
- Speed: ~121 t/s
- Content: Same math problem solution
- Status: Successful capture before crash

**Note**: resp-specnone.json, resp-1t.json, resp-conc1.json do NOT exist (crash before response)

---

## Build Verification

### Full Rebuild
**Date**: 2026-07-09 08:17-8:18 AM
**Steps**: 672/672
**Warnings**: 0 (#221-D)
**Errors**: 0
**Artifacts**: All built successfully in `build-cuda-b-bin/bin/Release/`

### Clean Rebuild Test
**Purpose**: Verify BEX64 crash was from stale artifacts
**Method**: Delete build dir, full rebuild
**Result**: BEX64 crashes resolved. Server crash also resolved after `-INFINITY` CUDA fix (see Post-Fix Verification above).
