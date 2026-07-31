# --tensor-split Semantics (Reverse-Engineered)

> **TL;DR**: `--tensor-split` is a proportional split of 42 layer-group slots across N devices. MTP layers are force-pinned to device[0]. The mapping from percentage to actual layer indices is opaque — always verify with `--verbose`.

## Device Order

```
Position 0: local GPU (ROCm on romulus)
Position 1: --rpc first entry  (3060Ti :50051)
Position 2: --rpc second entry (3070   :50055)
Position 3: --rpc third entry  (3090   :50054)
```

In `--rpc 127.0.0.1:50051,192.168.8.23:50055,192.168.8.23:50054`:
- `splits[0]` = local GPU (ROCm 7900 XTX)
- `splits[1]` = 127.0.0.1:50051 (3060 Ti, Docker, 8 GB)
- `splits[2]` = 192.168.8.23:50055 (RTX 3070, 8 GB)
- `splits[3]` = 192.168.8.23:50054 (RTX 3090, 24 GB)

## Algorithm (from `src/llama-model.cpp:1270-1324`)

```cpp
// 1. Populate splits from --tensor-split or free memory
splits = [raw_values from CLI]   // e.g. [25, 1, 1, 73]

// 2. Accumulate into prefix sums, then normalize to [0, 1]
split_sum = sum(splits)
for each i: splits[i] = prefix_sum(splits, i) / split_sum
// [0.25, 0.26, 0.27, 1.00]

// 3. Layer assignment via upper_bound on 42 slots
// act_gpu_layers = n_layer_all + 1 = 40 + 1 = 41
// But model has 42 entries (0-41) in practice:
//   0-25: 26 main transformer layers
//   26-39: 14 MTP head layers (FORCE-PINNED to device[0], skipped by upper_bound)
//   40:    nextn prediction head (NOT caught by MTP pinning, goes through upper_bound)
//   41:    output/embeddings layer
for each layer il in [0, 41]:
    if il >= 26 && il < 40:  // MTP head layers
        device = devices[0]  // FORCE-PINNED to local GPU
    else:
        device = upper_bound(splits, il / 42)
```

## Key gotchas

1. **MTP layers are invisible but count in the denominator.** 14 of 42 slots (33%) are force-pinned and don't participate in the split. This makes the percentages misleading — a 25% split for ROCm doesn't mean 25% of VRAM, it means 25% of the *non-MTP* layers plus all MTP layers.

2. **Layer 40 (nextn head) is NOT MTP-pinned.** It goes through upper_bound. In the default split it lands on the 3060Ti alongside layers 0-5, pushing it to 7 layers (4719 MiB).

3. **Early layers cost more compute AND more VRAM.** Layers 0-5 on the 3060Ti weigh 674 MiB/layer. Layers 11-25 on the 3090 weigh 438 MiB/layer. The front layers are denser (MoE with full attention interval, shared experts, etc.).

4. **`-fit off` is required.** The default `-fit on` conflicts with `--tensor-split` (assertion failure in `common/fit.cpp:378-387`).

## Layer counts and VRAM per GPU

| GPU | Free VRAM | Max safe layers | MiB/layer (observed) |
|-----|-----------|----------------|---------------------|
| ROCm 7900 XTX | 24484 MiB | ~20 (with MTP) | ~535 MiB |
| 3060 Ti | 7659 MiB | ~5 | ~674 MiB |
| RTX 3070 | 7623 MiB | ~5 | ~535 MiB |
| RTX 3090 | 23788 MiB | ~20 | ~438 MiB |

The 3060Ti has the tightest constraint: larger per-layer VRAM (early layers) + only 8 GB.

## How to check actual placement

Always run with `--verbose` and grep the output:

```bash
./build-rocm/bin/llama-server ... --verbose -lv 3 2>&1 | grep "layer.*assigned"
```

This produces lines like:
```
load_tensors: layer   0 assigned to device RPC0, is_swa = 0
load_tensors: layer  40 assigned to device RPC0 (MTP head), is_swa = 0
```

Where RPC0/RPC1/RPC2 map to `--rpc` entries in order, and ROCm0 is the local GPU.

## Per-layer device assignment (the better way)

The `llama_model_params` struct already has a `layer_devices` array and the `load_tensors()` code handles it (line 1294-1307). This is exposed via the placement plan system (`--placement-plan`, `--placement-heatmap`), which generates a JSON file with per-layer backend assignments.

**What doesn't exist yet:** A simple CLI flag like `--layer-split 0,0,0,0,0,1,1,1,1,1,...` to directly specify per-layer device indices. This would be a moderate change:
- Parse CLI flag in `common/common.cpp` (~30 lines)
- Fill `params.placement_layer_devices` vector
- Wire into `mparams.layer_devices` (already done for placement plans)

**Estimated effort:** ~50 lines of code, primarily CLI parsing + validation. The runtime path already works — it's the same code path the placement plan system uses.

## Experiment results summary

| Experiment | -ts values | 3060Ti layers | Status | Throughput |
|-----------|-----------|--------------|--------|-----------|
| Baseline (default) | proportional | 6 main + 1 head | OK | 0.313 tok/s |
| Exp D | 45,5,5,45 | ? | FAIL (13.7 GB) | - |
| Exp E | 55,7.5,7.5,30 | ? | FAIL (13.7 GB) | - |
| Exp F | 50,5,5,40 | ? | FAIL (12.3 GB) | - |
| Exp G | 35,5,5,55 | ? | FAIL (9.6 GB) | - |
| Exp H | 25,1,1,73 | ~1 main | OK | **1.804 tok/s** |
| Exp I | 33,6,7,54 | 14 main + 1 head | FAIL (9.1 GB) | - |

**Key insight:** The 3060Ti (8 GB, slowest GPU) was the pipeline bottleneck at 80% of compute in the baseline. Reducing it to near-zero with `-ts 25,1,1,73` shifted work to ROCm (82%) and boosted throughput 5.8x.
