# V2 CUDA validation — llama-server + triton RPC

| Field | Value |
|-------|-------|
| Topology | remus CUDA docker `llama-server` (5060 Ti) + triton `:50054` (3090) |
| Model | `meta-llama-3-70b-instruct.Q4_K_M.gguf` |
| ts | 50,50 |
| fit | on (`--fit-target 512,1024`, `-ngl 0`, `-np 1`) |
| GGML_PIPELINE_PLUS | 1 |
| Load | ~130s to `model loaded` |
| Gen smoke | 32 max_tokens req, 3 tokens (`Hello!`), ~162s wall |
| Verdict | **PASS** (correctness; profiler OOM was topology, not B+15) |

Note: `-ngl 99` OOMs on 5060 Ti (~20GB cudaMalloc). Use `--fit on` for dense 70B on this client class.