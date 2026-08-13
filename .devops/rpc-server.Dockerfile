# Multi-stage Dockerfile for rpc-server with UDP transport support.
#
# Builds the rpc-server binary + libggml-rpc.so (the UDP transport stack) from
# current source on an ubuntu22.04 builder so the resulting binaries link
# against glibc 2.35 (matching the runtime base). This is required because the
# host's newer glibc (2.39, Ubuntu 24.04) binaries are NOT backward-compatible
# with the container's older glibc.
#
# BUG-010: the previous ad-hoc image (built Jul 23) shipped a libggml-rpc.so
# lacking UDP transport support. Since GGML_RPC_UDP=1 became the default
# (07-26), clients targeting this server fail with "send_udp failed" and
# produce empty output. Rebuilding the rpc-server binary and libggml-rpc.so
# from current source (good-prototype HEAD) pulls in the UDP transport.
#
# WHY ONLY THE RPC TRANSPORT STACK IS REBUILT:
# UDP is purely an RPC-transport concern — it lives entirely in libggml-rpc.so
# and does not touch the CUDA compute backend (libggml-cuda.so). The CUDA
# backend in this fork's current code fails CUDA device enumeration against the
# host driver (580.x) in this container, while the Jul 23 CUDA build enumerates
# correctly. Since rebuilding CUDA is not required for UDP and would regress
# GPU enumeration, this image rebuilds only the RPC transport stack (verified
# to link cleanly against the existing libggml.so / libggml-base.so) and
# reuses the known-good CUDA + CPU + core libs from the previous image.
#
# The result: GPU enumeration works (old CUDA lib) AND UDP transport works
# (new rpc-lib). Verified: device enumerates as "RTX 3060 Ti, compute 8.6".

ARG CUDA_VERSION=12.8.0
ARG UBUNTU_VERSION=22.04

# ---- Build stage: compile the RPC transport stack from current source ----
# GGML_CUDA=OFF: we do not rebuild the CUDA backend (see note above).
# GGML_CPU_ALL_VARIANTS=ON: required for CMake 3.22 CUDA arch detection to
# succeed even though CUDA is off (the flag influences shared cmake logic).
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build

RUN apt-get update && apt-get install -y \
        gcc g++ build-essential cmake python3 git \
        libssl-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN cmake -B /tmp/build \
        -DGGML_NATIVE=OFF \
        -DGGML_CUDA=OFF \
        -DGGML_BACKEND_DL=ON \
        -DGGML_RPC=ON \
        -DGGML_CPU_ALL_VARIANTS=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_SERVER=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DCMAKE_BUILD_TYPE=Release . \
 && cmake --build /tmp/build --target rpc-server -j$(nproc)

# ---- Runtime stage: known-good CUDA/CPU/core libs + new UDP transport ----
# Inherit the working CUDA + CPU + core ggml libs from the previous image and
# overlay the newly-built rpc-server binary + UDP-enabled libggml-rpc.so.
FROM atomic-llama-gpu-host-pathd-rpc:stale-20260723

# Remove the old (UDP-less) RPC transport library; replaced by the COPY below.
RUN rm -f /usr/local/lib/libggml-rpc.so* /usr/local/bin/rpc-server

COPY --from=build /tmp/build/bin/rpc-server      /usr/local/bin/rpc-server
# The builder (GGML_CUDA=OFF) emits libggml-rpc.so without a version suffix,
# but the running rpc-server (via the core ggml backend loader) opens
# "libggml-rpc.so.0". Copy with an explicit rename to the versioned filename
# and create the soname symlinks so the dynamic loader resolves it.
COPY --from=build /tmp/build/bin/libggml-rpc.so /usr/local/lib/libggml-rpc.so.0.15.1

# Re-create the soname symlinks and refresh the linker cache.
RUN cd /usr/local/lib \
 && ln -sf libggml-rpc.so.0.15.1 libggml-rpc.so.0 \
 && ln -sf libggml-rpc.so.0 libggml-rpc.so \
 && ldconfig \
 && chmod +x /usr/local/bin/rpc-server
