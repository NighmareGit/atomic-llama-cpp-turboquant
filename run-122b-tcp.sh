#!/bin/bash
# Qwen3.5-122B-A10B-Q4_K_S — 5-GPU layer-split on romulus
# Model: 70 GiB, 49 layers, qwen35moe (122B/10B MoE)
# Context: 65K, ctk=q8_0, ctv=turbo3
# Layer distribution: 7900XTX=13, 3090=15, 5060Ti=9, 3060Ti=3, 3070=3, CPU=6

set -e

cd /home/hunter/scratch/prototype-auto/atomic-llama-cpp-turboquant

# === Best performance flags (from AGENTS.md + research docs) ===
export GGML_RPC_UDP=0              # DISABLED: UDP causing send_udp errors, H1 investigation
export GGML_PIPELINE_PLUS=1        # Master switch for pipeline optimizations
export GGML_RPC_GET_TENSOR_DEFER=1 # Defer RPC GET_TENSOR downloads
export GGML_SCHED_WAVEFRONT_CROSS=1 # Cross-decode wavefront dispatch

# === Required for sm_86 GPUs (3090, 3070, 3060 Ti) ===
export GGML_CUDA_PDL=0

# === Optional: enable for debugging ===
# export GGML_RPC_DEBUG=1

echo "=== Starting llama-server: Qwen3.5-122B-A10B 5-GPU ==="
echo "Model:  /mnt/980pro/models/Qwen3.5-122B-A10B-Q4_K_S.gguf"
echo "Split:  layer (stable path)"
echo "Tensor split: 13 3 9 15 3 (7900XTX 3060Ti 5060Ti 3090 3070)"
echo "Context: 65536"
echo "ctk: q8_0, ctv: turbo3"
echo "GPUs: 7900XTX(local) + 3060Ti + 5060Ti + 3090 + 3070"
echo ""

# Device order: gpu,rpc → 7900XTX is device 0, then RPC in order:
# RPC0=3060Ti(127.0.0.1:50051), RPC1=5060Ti(192.168.8.22:50051),
# RPC2=3090(192.168.8.23:50051), RPC3=3070(192.168.8.23:50052)
# Tensor split order matches device order: 7900XTX, 3060Ti, 5060Ti, 3090, 3070

./build-rocm-rpc-split/bin/llama-server \
  --model /mnt/980pro/models/Qwen3.5-122B-A10B-Q4_K_S.gguf \
  --split-mode layer \
  --device-order gpu,rpc \
  --rpc 127.0.0.1:50051,192.168.8.22:50051,192.168.8.23:50051,192.168.8.23:50052 \
  --tensor-split 13,3,9,15,3 \
  -ngl 43 \
  -ctk q8_0 \
  -ctv turbo3 \
  -c 65536 \
  -t 18 \
  --no-mmap \
  --jinja \
  --host 0.0.0.0 --port 8080
