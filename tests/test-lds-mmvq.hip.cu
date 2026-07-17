// Minimal correctness test for the LDS-cached MMVQ prototype.
// Compares LDS kernel output against CPU reference for a single row.
// Build: see test-lds-mmvq.hip.cmake

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>

// Pull in ggml types and the MMVQ kernel code
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml/src/ggml-common.h"

// Need to pull in vec_dot and warp_reduce
#define GGML_HIP_MMVQ_LDS_PROTOTYPE
#include "ggml/src/ggml-cuda/vecdotq.cuh"
#include "ggml/src/ggml-cuda/mmvq.cuh"
#include "ggml/src/ggml-cuda/common.cuh"

// We need a host-side reference implementation of vec_dot_q4_K_q8_1
// Compute dot product on CPU for verification
static float cpu_vec_dot_q4_K_q8_1(
    const block_q4_K & bq4_K, const block_q8_1 * bq8_1, int iqs) {

    const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));

    // Extract scales and mins from block_q4_K
    const uint16_t * scales_ptr = (const uint16_t *)bq4_K.scales;
    uint16_t aux[2];
    const int j = bq8_offset/2;
    if (j < 2) {
        aux[0] = scales_ptr[j+0] & 0x3f3f;
        aux[1] = scales_ptr[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales_ptr[j+2] >> 0) & 0x0f0f) | ((scales_ptr[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales_ptr[j+2] >> 4) & 0x0f0f) | ((scales_ptr[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

    float sumf = 0.0f;
    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1 & bq8i = bq8_1[bq8_offset + i];

        // Extract q4 values
        const int q4_offset = 16 * bq8_offset + 4 * ((iqs/2)%4);
        const int * q4_ptr = (const int *)(bq4_K.qs + q4_offset);
        int v0 = q4_ptr[0];
        int v1 = q4_ptr[4];

        int vi0 = (v0 >> (4*i)) & 0x0F0F0F0F;
        int vi1 = (v1 >> (4*i)) & 0x0F0F0F0F;

        // Extract q8 values
        const int * q8_ptr = (const int *)bq8i.qs + ((iqs/2)%4);
        int u0 = q8_ptr[0];
        int u1 = q8_ptr[4];

        // dp4a equivalent on CPU
        auto dp4a = [](int a, int b, int c) -> int {
            int sum = c;
            sum += ((a >>  0) & 0xFF) * ((b >>  0) & 0xFF);
            sum += ((a >>  8) & 0xFF) * ((b >>  8) & 0xFF);
            sum += ((a >> 16) & 0xFF) * ((b >> 16) & 0xFF);
            sum += ((a >> 24) & 0xFF) * ((b >> 24) & 0xFF);
            return sum;
        };

        int dot1 = dp4a(vi1, u1, dp4a(vi0, u0, 0));
        int dot2 = dp4a(0x01010101, u1, dp4a(0x01010101, u0, 0));

        float d8_val = __half2float(bq8i.ds.x);
        sumf += d8_val * (dot1 * sc[i]);
        sumf -= d8_val * (dot2 * m[i]);
    }

    float2 dm4f = __half22float2(bq4_K.dm);
    return dm4f.x * sumf;
}

int main() {
    printf("=== LDS MMVQ Correctness Test ===\n");

    // Allocate test data: one q4_K block and 8 q8_1 blocks (one row)
    const int qk = 256;
    const int blocks_per_row = 1; // single Q4_K block for simplicity
    const int nrows = 1;

    block_q4_K * h_weight = new block_q4_K[blocks_per_row * nrows];
    block_q8_1 * h_activ  = new block_q8_1[qk / QK8_1]; // 8 blocks

    // Fill with deterministic test data
    for (int i = 0; i < blocks_per_row * nrows; i++) {
        h_weight[i].dm.x = __float2half(1.0f);
        h_weight[i].dm.y = __float2half(0.0f);
        // Simple deterministic quants: all zeros -> all values = -dmin = 0
        // Actually use a simple pattern for non-trivial test
        for (int j = 0; j < QK_K/2; j++) {
            h_weight[i].qs[j] = (uint8_t)((j * 17 + i * 13) & 0xFF);
        }
        for (int j = 0; j < K_SCALE_SIZE; j++) {
            h_weight[i].scales[j] = (uint8_t)((j * 31 + i * 7 + 1) & 0xFF);
        }
    }

    for (int i = 0; i < qk / QK8_1; i++) {
        h_activ[i].ds.x = __float2half(1.0f);
        h_activ[i].ds.y = __float2half(0.0f);
        for (int j = 0; j < QI8_1; j++) {
            h_activ[i].qs[j] = (uint8_t)((i * 41 + j * 19 + 3) & 0xFF);
        }
    }

    // CPU reference: compute dot product for all iqs values (0,2,4,...,30)
    float cpu_results[16];
    for (int t = 0; t < 16; t++) {
        int iqs = 2 * t;
        cpu_results[t] = cpu_vec_dot_q4_K_q8_1(h_weight[0], h_activ, iqs);
    }

    printf("CPU results:\n");
    for (int t = 0; t < 16; t++) {
        printf("  tid=%2d iqs=%2d: %.6f\n", t, 2*t, cpu_results[t]);
    }

    // GPU: launch the LDS kernel on this data
    block_q4_K * d_weight;
    block_q8_1 * d_activ;
    float * d_result;
    hipMalloc(&d_weight, sizeof(block_q4_K) * blocks_per_row * nrows);
    hipMalloc(&d_activ,  sizeof(block_q8_1) * (qk / QK8_1));
    hipMalloc(&d_result, sizeof(float) * nrows);

    hipMemcpy(d_weight, h_weight, sizeof(block_q4_K) * blocks_per_row * nrows, hipMemcpyHostToDevice);
    hipMemcpy(d_activ,  h_activ,  sizeof(block_q8_1) * (qk / QK8_1), hipMemcpyHostToDevice);

    // Launch LDS prototype kernel
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    // Pre-compute strides: ncols_x=qk=256, nrows_x=1, ncols_dst=1, nchannels=1, nsamples=1
    dim3 block_nums(1, 1, 1);
    dim3 block_dims(warp_size, 1, 1);

    ggml_cuda_mm_fusion_args_device fusion = {};
    uint3 nchannels_y = {1,1,1};
    uint3 channel_ratio = {1,1,1};
    uint3 sample_ratio = {1,1,1};

    // We need to call mul_mat_vec_q_lds_prototype through a launch macro
    // Since we can't easily use ggml_cuda_kernel_launch without the full build,
    // let's just use standard hipLaunchKernelGGL
    const int nbytes_shared = LDS_CACHE_BLOCKS * sizeof(block_q8_1);

    hipLaunchKernelGGL(
        mul_mat_vec_q_lds_prototype,
        block_nums, block_dims, nbytes_shared, 0,
        d_weight, d_activ, nullptr, fusion, d_result,
        (uint32_t)qk, nchannels_y, (uint32_t)1,  // ncols_x, nchannels_y, stride_row_x
        (uint32_t)(qk/QK8_1), (uint32_t)1,        // stride_col_y, stride_col_dst
        channel_ratio,
        (uint32_t)1, (uint32_t)(qk/QK8_1), (uint32_t)1,  // stride_channel_x,y,dst
        sample_ratio,
        (uint32_t)1, (uint32_t)1, (uint32_t)1,  // stride_sample_x,y,dst
        (uint32_t)1  // ids_stride
    );

    hipDeviceSynchronize();

    float h_result;
    hipMemcpy(&h_result, d_result, sizeof(float), hipMemcpyDeviceToHost);

    printf("\nGPU result (LDS kernel): %.6f\n", h_result);

    // The GPU kernel computes the full matvec (all kbx iterations, warp-reduced).
    // For a single Q4_K block, the result should be the sum of all 16 thread contributions.
    float cpu_full = 0.0f;
    for (int t = 0; t < 16; t++) {
        cpu_full += cpu_results[t];
    }
    printf("CPU full matvec:          %.6f\n", cpu_full);

    float diff = fabsf(h_result - cpu_full);
    float rel_diff = cpu_full != 0.0f ? diff / fabsf(cpu_full) : diff;
    printf("Difference: %.6f (relative: %.6e)\n", diff, rel_diff);

    // Also compute the non-LDS GPU result to see if the kernel dispatch itself
    // has an issue vs pure CPU reference
    // We can approximate this by using the known-good original kernel
    // but for now, just compare LDS CPU vs GPU.

    bool pass = rel_diff < 1e-4f;
    printf("\n=== %s ===\n", pass ? "PASS" : "FAIL");

    hipFree(d_weight);
    hipFree(d_activ);
    hipFree(d_result);
    delete[] h_weight;
    delete[] h_activ;

    return pass ? 0 : 1;
}
