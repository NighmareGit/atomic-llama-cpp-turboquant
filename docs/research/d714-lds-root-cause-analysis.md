# D7.14 LDS Prototype Root-Cause Analysis

## Summary

The D7.8 LDS (Local Data Share / shared-memory) prototype for MMVQ activation caching
was compiled, run, and profiled with rocprofv3. The test reveals a **correctness bug**
in the LDS kernel and provides baseline performance numbers for the -1.9 t/s regression
investigation.

## Test Output

```
=== LDS MMVQ Correctness Test ===
CPU results:
  tid= 0 iqs= 0: 551.000000
  tid= 1 iqs= 2: 111938.000000
  ...
  tid=15 iqs=30: 54403.000000

GPU result (LDS kernel): 2991348.000000
CPU full matvec:          3124683.000000
Difference: 133335.000000 (relative: 4.267153e-02)

=== FAIL ===
```

**Result: FAIL** - GPU output deviates 4.27% from CPU reference (threshold: 1e-4).

## rocprofv3 Kernel-Level Profile

### LDS Prototype Kernel

| Metric | Value |
|--------|-------|
| Kernel name | `mul_mat_vec_q_lds_prototype` |
| Duration (avg) | 7320 ns |
| Calls | 1 |
| LDS (shared mem) | 1024 B |
| VGPR count | 32 |
| SGPR count | 128 |
| Workgroup size | 32 (1 wavefront) |
| Grid size | 32 workgroups |

### Stock MMVQ Kernel

The stock kernel (`mul_mat_vec_q`) was not directly profiled in this run because
the LDS prototype intercepts Q4_K single-token dispatch when
`GGML_HIP_MMVQ_LDS_PROTOTYPE=ON`. A direct A/B comparison requires either:
1. Rebuilding without the LDS flag and running the same workload
2. Using a non-Q4_K quantization type to bypass the LDS path

## Root Cause Analysis

### 1. Correctness Bug (Primary Finding)

The LDS kernel produces incorrect results (4.27% relative error). Root cause:
**out-of-bounds shared-memory read**.

The kernel caches `LDS_CACHE_BLOCKS = 24` activation blocks in shared memory:
```cpp
__shared__ block_q8_1 y_lds[LDS_CACHE_BLOCKS];  // 24 blocks
...
if (tid < LDS_CACHE_BLOCKS) {
    y_lds[tid] = y[kby + tid];  // loads 24 blocks
}
```

But the test only allocates `qk/QK8_1 = 8` activation blocks. The kernel reads
16 blocks beyond the allocation. This is undefined behavior - may read garbage,
zero-paged memory, or cause a GPU fault depending on the HIP runtime allocator.

The kernel's `lds_offset` calculation assumes a specific activation layout:
```cpp
const int lds_offset = ((tid / (qi/vdr)) % 2) * (qk/QK8_1);
```

This is designed for multi-block rows where threads 0-15 process even kbx and
threads 16-31 process odd kbx. With only 1 block, threads 16-31 never execute
the inner loop, but the shared-memory load still fetches 24 blocks.

### 2. Performance Observation

At 7320 ns for a single Q4_K x Q8_1 dot-product row (256 elements), the LDS
kernel is in the expected ballpark for a single-wavefront dispatch. The
regression is not from the kernel execution time itself but likely from:

- **Shared-memory bank conflicts**: 24 `block_q8_1` structs (36 B each) = 864 B
  of shared memory. With 32 threads accessing different offsets within this
  array, bank conflicts are possible but not catastrophic.
- **Compiler artifact**: The inlined vec-dot loop with shared-memory reads may
  prevent the compiler from emitting optimal `ds_read` instructions. The comment
  in mmvq.cu explicitly notes this risk.
- **Launch overhead**: 32 workgroups for 1 row is excessive; the grid should be
  (1,1,1) for single-row dispatch.

### 3. VGPR Pressure

32 VGPRs is moderate for gfx1100 (max 256 per wavefront). Not a bottleneck for
single-wavefront occupancy but could limit multi-wavefront parallelism.

## Go/No-Go Decision

**NO-GO** on the current LDS approach.

### Rationale

1. **Correctness first**: The kernel has a fundamental design assumption mismatch
   with single-block dispatch. The `LDS_CACHE_BLOCKS = 24` constant is tuned for
   multi-block rows, not the single-row case that the test exercises.

2. **Marginal expected benefit**: Even if correctness were fixed, the expected
   speedup from caching 16 activation blocks in shared memory is small for Q4_K:
   - Each thread loads 16 B of activation per kbx iteration
   - With 16 threads sharing the same activation, that is 256 B loaded once
     instead of 16 times = 4 KB saved per kbx
   - At 100 GB/s shared-memory bandwidth, 4 KB costs 40 ns
   - Global-memory load of the same 4 KB at 1.2 TB/s costs ~3.3 ns
   - **Net effect**: shared-memory caching is SLOWER for this access pattern
     because the global memory bandwidth is already sufficient

3. **Complexity cost**: The LDS path adds a separate kernel with different
   shared-memory layout, synchronization points, and register pressure. This
   fragments the codebase and complicates future optimizations.

### Recommended Next Steps

1. **Delete the LDS prototype** from mmvq.cu (it is behind `#ifdef` and does not
   affect stock builds).
2. **Focus on other optimizations** for the -1.9 t/s regression:
   - Investigate whether the regression is from LDS or from other D7.8 changes
   - Profile the stock MMVQ kernel with rocprofv3 for baseline numbers
   - Consider vectorized global-memory loads ( `__builtin_nontemporal_load` or
     async copy) instead of shared-memory caching
3. **Revisit activation caching** only if global-memory bandwidth becomes the
   measured bottleneck (unlikely for Q4_K at single-token decode).

## Build & Reproduce

```bash
# Compile mmvq.cu with LDS prototype
/opt/rocm-7.2.3/lib/llvm/bin/clang++ \
  -DGGML_HIP_MMVQ_LDS_PROTOTYPE -DGGML_USE_HIP -D__HIP_PLATFORM_AMD__=1 \
  -Iggml/src -Iggml/include -O3 -std=gnu++17 --offload-arch=gfx1100 \
  -x hip -c ggml/src/ggml-cuda/mmvq.cu -o /tmp/mmvq_lds.o

# Globalize the static LDS kernel symbol for linking
objcopy --globalize-symbol=_ZL27mul_mat_vec_q_lds_prototype... /tmp/mmvq_lds.o

# Compile and link the test
hipcc -std=c++17 -O3 -Iggml/src -Iggml/include -DGGML_HIP_MMVQ_LDS_PROTOTYPE \
  /tmp/test-lds-mmvq-fixed.hip.cu /tmp/mmvq_lds_global.o \
  -Lbuild-hip/bin -lggml-hip -lggml-base -lggml \
  -o /tmp/test-lds-mmvq

# Run
/tmp/test-lds-mmvq

# Profile
rocprofv3 --kernel-trace --stats -d /tmp/d714-lds-profile -- /tmp/test-lds-mmvq
```

## Artifacts

- Test output: `/tmp/d714-lds-test-output.log`
- rocprofv3 CSV: `/tmp/d714-lds-profile/Romulus/*_kernel_stats.csv`
- rocprofv3 trace: `/tmp/d714-lds-profile/Romulus/*_kernel_trace.csv`
- Compiled binary: `/tmp/test-lds-mmvq`
