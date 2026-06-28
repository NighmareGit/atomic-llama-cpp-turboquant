# T3: composition matrix snapshots

Full-stack cells: speculative mode x TurboQuant KV x pipeline layers x topology.

## Matrix C (partial)

| Stack | depth-2 | Plus | spec | KV | mmproj | G (t/s) | Source |
|-------|---------|------|------|-----|--------|---------|--------|
| qwen-35B turbo3-nextn | on | n/a | nextn | turbo3 | no | 82.7 short | [NEXTN.md](../../../NEXTN.md) |
| 2gpu Config F | n/a | 1 | none | q4_0 | no | 48.9 | trace-f-2gpu-plus |
| 4gpu Config G | n/a | 1 | none | q8_0 | no | 38-43 | cluster-4gpu-primary |
| gemma-26B turbo3-mtp | on | n/a | mtp | turbo3 | no | 80.5 short | [MTP.md](../../../MTP.md) |
| qwen-35B nextn + mmproj text | on | n/a | nextn | turbo3 | yes | ~69 | [NEXTN.md](../../../NEXTN.md) S10 |

Rows with explicit Plus on/off **and** depth-2 on/off on the same host are TBD.