// PROTOTYPE: D7.13-Idea6
// Throwaway prototype: test if V_WMMA_I32_16X16X16_IU4 can replace
// the dp4a path in MMQ kernels for ncols_dst>=16 on RDNA3 (gfx1100).
//
// KEY FINDING: The __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32 builtin has
// its A and B parameters SWAPPED compared to the ISA encoding.
//   ISA:  D = A * B + C  (SRC0=A, SRC1=B)
//   Builtin: __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(signB, B, signA, A, C, signC)
// So to compute D = A * B + C, you must pass B first, then A.
//
// This file is NOT for production. Delete after benchmarking.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>

// Native vector types for WMMA builtins (required by clang builtins)
typedef int __attribute__((ext_vector_type(2))) int2_native;
typedef int __attribute__((ext_vector_type(8))) int8_native;

// ============================================================
// Correctness test: verify IU4 WMMA with known pattern
// ============================================================
__global__ void test_iu4_correctness(int32_t *result) {
    int lane = threadIdx.x;
    int row = lane % 16;

    // A matrix (weights): A[row][k] = ((row % 15) + 1) for all k
    // Values 1..15 fit in nibbles (16 would overflow to 0)
    int a_nibble = (row % 15) + 1;
    int a_val = 0;
    for (int k = 0; k < 8; k++) {
        a_val |= (a_nibble & 0xF) << (4 * k);
    }
    int2_native A = {a_val, a_val};

    // B matrix (activations): B[k][j] = 1 for all k, j
    int2_native B = {0x11111111, 0x11111111};

    // Zero accumulator
    int8_native C = {0, 0, 0, 0, 0, 0, 0, 0};

    // WMMA: D = A * B + C
    // CRITICAL: builtin parameter order is (signB, B, signA, A, C, signC)
    auto D = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
        false, B, false, A, C, false
    );

    // Store result
    for (int i = 0; i < 8; i++) {
        result[lane * 8 + i] = D[i];
    }
}

// ============================================================
// CPU reference for IU4 matrix multiply
// ============================================================
void cpu_reference_iu4(const uint8_t *a_packed, const uint8_t *b_packed,
                        int32_t *c, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++) {
                // Extract nibble from packed format (low nibble first)
                uint8_t a_byte = a_packed[i * (K/2) + k/2];
                uint8_t a_nib = (k % 2 == 0) ? (a_byte & 0x0F) : ((a_byte >> 4) & 0x0F);
                uint8_t b_byte = b_packed[j * (K/2) + k/2];
                uint8_t b_nib = (k % 2 == 0) ? (b_byte & 0x0F) : ((b_byte >> 4) & 0x0F);
                sum += (int32_t)a_nib * (int32_t)b_nib;
            }
            c[i * N + j] = sum;
        }
    }
}

// ============================================================
// Full Q4_K + Q8_1 tile test
// Maps Q4_K block_q4_K.qs nibbles and Q8_1 block_q8_1.qs bytes
// into WMMA registers for a 16x16 tile.
// ============================================================
// ggml_half is uint16_t (IEEE 754 half-precision)
typedef uint16_t ggml_half;

struct block_q4_K {
    ggml_half d;
    ggml_half dmin;
    uint8_t scales[12];
    uint8_t qs[128];  // 256 nibbles
};

struct block_q8_1 {
    ggml_half d;
    ggml_half s;
    int8_t qs[32];
};

// Q4_K dequantization reference for a single 16-element sub-block
// Q4_K layout: 8 sub-blocks of 32 elements each
// Each sub-block has its own scale/min from the scales[] array
// For simplicity, we use the first 16 elements of qs[] as a test tile
__host__ __device__ uint8_t get_q4_K_nibble(const block_q4_K *block, int element_idx) {
    // qs[] has 128 bytes = 256 nibbles
    // element_idx 0..255 maps to qs[element_idx/2], low/high nibble
    int byte_idx = element_idx / 2;
    int is_high = element_idx % 2;
    uint8_t byte = block->qs[byte_idx];
    return is_high ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
}

__host__ __device__ int8_t get_q8_1_value(const block_q8_1 *block, int element_idx) {
    // qs[] has 32 int8_t values
    return block->qs[element_idx % 32];
}

// Load a 16x16 IU4 tile from Q4_K weights (16 weight values per row)
// Each row of the tile corresponds to one output row (M dimension)
// Each column corresponds to one K dimension element
__device__ int2_native load_q4_K_tile_row(const block_q4_K *weight, int row) {
    // For a 16x16 tile, we take 16 consecutive nibbles per row
    // Weight layout: row i uses nibbles [i*16 .. i*16+15]
    // But Q4_K has 256 nibbles per block, and we need to map them
    // to the WMMA A matrix layout.
    //
    // Simplified test: use nibbles [row*16 + k] for k=0..15
    // This maps to qs bytes [row*8 + k/2], alternating low/high nibble

    int a_val = 0;
    for (int k = 0; k < 8; k++) {
        int nibble_idx = row * 16 + k;
        uint8_t nib = get_q4_K_nibble(weight, nibble_idx);
        a_val |= (nib & 0xF) << (4 * k);
    }
    int a_val2 = 0;
    for (int k = 8; k < 16; k++) {
        int nibble_idx = row * 16 + k;
        uint8_t nib = get_q4_K_nibble(weight, nibble_idx);
        a_val2 |= (nib & 0xF) << (4 * (k - 8));
    }
    return {a_val, a_val2};
}

// Load a 16x16 IU4 tile from Q8_1 activations
// Q8_1 has int8 values, but WMMA IU4 needs nibbles.
// We split each int8 into low/high nibbles, requiring 2 WMMA calls.
// This function loads the low nibble tile.
__device__ int2_native load_q8_1_tile_row_low(const block_q8_1 *act, int row) {
    int a_val = 0;
    for (int k = 0; k < 8; k++) {
        int8_t val = get_q8_1_value(act, row * 16 + k);
        uint8_t nib = val & 0x0F;  // Low nibble
        a_val |= (nib & 0xF) << (4 * k);
    }
    int a_val2 = 0;
    for (int k = 8; k < 16; k++) {
        int8_t val = get_q8_1_value(act, row * 16 + k);
        uint8_t nib = val & 0x0F;  // Low nibble
        a_val2 |= (nib & 0xF) << (4 * (k - 8));
    }
    return {a_val, a_val2};
}

// Full Q4_K x Q8_1 WMMA kernel (single 16x16 tile)
__global__ void test_q4K_q8_1_wmma(
    const block_q4_K * __restrict__ weight,
    const block_q8_1 * __restrict__ activ,
    int32_t * __restrict__ result)
{
    int lane = threadIdx.x;
    int row = lane % 16;

    // Load weight tile (A matrix) from Q4_K
    int2_native A = load_q4_K_tile_row(weight, row);

    // Load activation tile (B matrix) from Q8_1 - low nibbles
    int2_native B = load_q8_1_tile_row_low(activ, row);

    // Zero accumulator
    int8_native C = {0, 0, 0, 0, 0, 0, 0, 0};

    // WMMA: D = A * B + C (builtin: B first, then A)
    auto D = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
        false, B, false, A, C, false
    );

    // Store result
    for (int i = 0; i < 8; i++) {
        result[lane * 8 + i] = D[i];
    }
}

// ============================================================
// CPU reference for Q4_K x Q8_1 dot product (single tile)
// ============================================================
void cpu_reference_q4K_q8_1(
    const block_q4_K *weight,
    const block_q8_1 *activ,
    int32_t *c)
{
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            int32_t sum = 0;
            for (int k = 0; k < 16; k++) {
                uint8_t a_nib = get_q4_K_nibble(weight, i * 16 + k);
                int8_t b_val = get_q8_1_value(activ, j * 16 + k);
                uint8_t b_nib = b_val & 0x0F;  // Low nibble only
                sum += (int32_t)a_nib * (int32_t)b_nib;
            }
            c[i * 16 + j] = sum;
        }
    }
}

// ============================================================
// Test runner
// ============================================================
void run_correctness_test() {
    printf("=== IU4 WMMA Correctness Test ===\n\n");

    int32_t *d_result;
    hipMalloc(&d_result, 32 * 8 * sizeof(int32_t));

    hipLaunchKernelGGL(test_iu4_correctness, dim3(1), dim3(32), 0, 0, d_result);
    hipDeviceSynchronize();

    int32_t h_result[32 * 8];
    hipMemcpy(h_result, d_result, 32 * 8 * sizeof(int32_t), hipMemcpyDeviceToHost);

    printf("Test: A[row][k] = ((row %% 15) + 1), B[k][j] = 1\n");
    printf("Expected: C[row][*] = 16 * ((row %% 15) + 1)\n\n");

    bool pass = true;
    for (int row = 0; row < 16; row++) {
        int expected = 16 * ((row % 15) + 1);
        printf("Row %2d: ", row);
        for (int i = 0; i < 4; i++) {
            printf("%3d ", h_result[row * 8 + i]);
        }
        printf("... (expected %d)", expected);

        bool row_pass = true;
        for (int i = 0; i < 8; i++) {
            if (h_result[row * 8 + i] != expected) row_pass = false;
        }
        printf(" %s\n", row_pass ? "OK" : "FAIL");
        if (!row_pass) pass = false;
    }

    // Check replication
    printf("\nReplication check (lanes 16-31 vs 0-15):\n");
    for (int row = 0; row < 16; row++) {
        for (int i = 0; i < 8; i++) {
            if (h_result[row * 8 + i] != h_result[(row + 16) * 8 + i]) {
                printf("  MISMATCH at row %d, col %d: lane %d = %d, lane %d = %d\n",
                       row, i, row, h_result[row * 8 + i],
                       row + 16, h_result[(row + 16) * 8 + i]);
                pass = false;
            }
        }
    }

    printf("\nCorrectness: %s\n\n", pass ? "PASS" : "FAIL");
    hipFree(d_result);
}

void run_q4K_q8_1_test() {
    printf("=== Q4_K x Q8_1 WMMA Test ===\n\n");

    // Create test data
    block_q4_K h_weight;
    block_q8_1 h_activ;

    // Fill weight with deterministic pattern
    for (int i = 0; i < 128; i++) {
        h_weight.qs[i] = (uint8_t)((i * 17 + 13) & 0xFF);
    }
    h_weight.d = 0x3C00;  // 1.0f in half
    h_weight.dmin = 0x0000;  // 0.0f in half

    // Fill activation with deterministic pattern
    for (int i = 0; i < 32; i++) {
        h_activ.qs[i] = (int8_t)((i * 11 + 7) & 0xFF);
    }
    h_activ.d = 0x3C00;  // 1.0f in half
    h_activ.s = 0x0000;

    // CPU reference
    int32_t h_cpu[16 * 16];
    cpu_reference_q4K_q8_1(&h_weight, &h_activ, h_cpu);

    printf("CPU reference (first 4x4):\n");
    for (int i = 0; i < 4; i++) {
        printf("  ");
        for (int j = 0; j < 4; j++) {
            printf("%4d ", h_cpu[i * 16 + j]);
        }
        printf("\n");
    }

    // GPU test
    block_q4_K *d_weight;
    block_q8_1 *d_activ;
    int32_t *d_result;
    hipMalloc(&d_weight, sizeof(block_q4_K));
    hipMalloc(&d_activ, sizeof(block_q8_1));
    hipMalloc(&d_result, 32 * 8 * sizeof(int32_t));

    hipMemcpy(d_weight, &h_weight, sizeof(block_q4_K), hipMemcpyHostToDevice);
    hipMemcpy(d_activ, &h_activ, sizeof(block_q8_1), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(test_q4K_q8_1_wmma, dim3(1), dim3(32), 0, 0,
                       d_weight, d_activ, d_result);
    hipDeviceSynchronize();

    int32_t h_result[32 * 8];
    hipMemcpy(h_result, d_result, 32 * 8 * sizeof(int32_t), hipMemcpyDeviceToHost);

    printf("\nGPU result (first 4x4):\n");
    for (int i = 0; i < 4; i++) {
        printf("  ");
        for (int j = 0; j < 4; j++) {
            printf("%4d ", h_result[i * 8 + j]);
        }
        printf("\n");
    }

    // Compare
    bool pass = true;
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 8; j++) {
            if (h_result[i * 8 + j] != h_cpu[i * 16 + j]) {
                printf("MISMATCH at [%d][%d]: GPU=%d, CPU=%d\n",
                       i, j, h_result[i * 8 + j], h_cpu[i * 16 + j]);
                pass = false;
            }
        }
    }

    printf("\nQ4_K x Q8_1: %s\n\n", pass ? "PASS" : "FAIL");

    hipFree(d_weight);
    hipFree(d_activ);
    hipFree(d_result);
}

// ============================================================
// Performance benchmark
// ============================================================
void benchmark() {
    printf("=== Performance Benchmark ===\n\n");

    const int iterations = 10000;

    // Allocate device memory for simple benchmark
    int32_t *d_result;
    hipMalloc(&d_result, 32 * 8 * sizeof(int32_t));

    // Warmup
    for (int i = 0; i < 100; i++) {
        hipLaunchKernelGGL(test_iu4_correctness, dim3(1), dim3(32), 0, 0, d_result);
    }
    hipDeviceSynchronize();

    // Benchmark
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        hipLaunchKernelGGL(test_iu4_correctness, dim3(1), dim3(32), 0, 0, d_result);
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float ms = 0;
    hipEventElapsedTime(&ms, start, stop);

    double total_ops = (double)iterations * 16 * 16 * 16;  // M*N*K MACs
    double gops = total_ops / (ms / 1000.0) / 1e9;

    printf("WMMA IU4 kernel: %.3f ms for %d iterations (%.3f us/kernel)\n",
           ms, iterations, ms * 1000.0f / iterations);
    printf("Throughput: %.2f Gint4-MACs/s\n", gops);
    printf("Peak theoretical: %.0f Gint4-MACs/s (1024 MACs/clock * 2.48 GHz)\n",
           1024.0 * 2.482);
    printf("Efficiency: %.2f%%\n", gops / (1024.0 * 2.482) * 100.0);

    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipFree(d_result);
}

// ============================================================
// Main
// ============================================================
int main() {
    printf("========================================\n");
    printf("PROTOTYPE: D7.13-Idea6 -- IU4 WMMA MMQ Prototype (CORRECTED)\n");
    printf("Tests V_WMMA_I32_16X16X16_IU4 on RDNA3\n");
    printf("========================================\n\n");

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Arch: %d.%d, Warp: %d, Clock: %d MHz\n",
           prop.major, prop.minor, prop.warpSize, prop.clockRate / 1000);
    printf("\n");

    run_correctness_test();
    run_q4K_q8_1_test();
    benchmark();

    printf("========================================\n");
    printf("PROTOTYPE: D7.13-Idea6 -- Prototype complete\n");
    printf("========================================\n");

    return 0;
}
