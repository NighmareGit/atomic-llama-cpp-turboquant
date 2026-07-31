# Multi-stage Dockerfile for rpc-server with UDP transport support.
#
# Builds rpc-server + ggml shared libraries from current source on an
# ubuntu22.04 builder so the resulting binaries link against glibc 2.35
# (matching the runtime base). This is required because the host's newer
# glibc (2.39, Ubuntu 24.04) binaries are NOT backward-compatible with
# the container's older glibc.
#
# BUG-010: the previous ad-hoc image (built Jul 23) shipped a libggml-rpc.so
# lacking UDP transport support. Since GGML_RPC_UDP=1 became the default
# (07-26), clients targeting this server fail with "send_udp failed" and
# produce empty output. Rebuilding libggml-rpc.so (and the rpc-server binary
# + libggml + libggml-base that link it) from current source pulls in the UDP
# transport and makes the image reproducible.
#
# NOTE on the CUDA backend (libggml-cuda.so): the CUDA compute kernels are
# independent of the RPC transport. The current ggml-cuda code in this fork
# fails CUDA device enumeration against the host driver (driver 580.x) in this
# container, while the Jul 23 CUDA build enumerates correctly. Since UDP is
# purely an RPC-transport concern and does not touch the CUDA backend, this
# image reuses the known-good CUDA lib from the previous image and rebuilds
# only the RPC transport stack. This keeps GPU enumeration working while
# adding UDP support.
#
# CUDA arch for the builder stage (used only if rebuilding CUDA is ever needed):
# bare numbers (no -real/-virtual suffix). CMake 3.22 (Ubuntu 22.04) rejects the
# suffixed forms that ggml-cuda's auto-detection emits.

ARG CUDA_VERSION=12.8.0
ARG UBUNTU_VERSION=22.04

# ---- Build stage: compile the RPC transport stack from current source ----
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build

RUN apt-get update && apt-get install -y \
        gcc g++ build-essential cmake python3 git \
        libssl-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

# Build rpc-server + libggml + libggml-base + libggml-rpc (the UDP transport
# stack). The CUDA backend is intentionally NOT rebuilt here (see NOTE above);
# GGML_CUDA=OFF keeps the build fast and avoids the device-enumeration
# regression in the current ggml-cuda code. GGML_CPU_ALL_VARIANTS=ON is
# required for CUDA arch detection to succeed in this CMake 3.22 environment.
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

# Stage the newly-built UDP transport libs + binary.
RUN mkdir -p /app/staging/bin /app/staging/lib \
 && cp /tmp/build/bin/rpc-server                 /app/staging/bin/ \
 && cp /tmp/build/bin/libggml.so*               /app/staging/lib/ \
 && cp /tmp/build/bin/libggml-base.so*          /app/staging/lib/ \
 && cp /tmp/build/bin/libggml-rpc.so*           /app/staging/lib/

# ---- Runtime stage: known-good CUDA lib + new UDP transport stack ----
# Base on the previous image to inherit its working libggml-cuda.so (the
# Jul 23 CUDA build enumerates devices correctly against the host driver).
FROM atomic-llama-romulus-pathd-rpc:stale-20260723 AS runtime_base

# Remove the old (UDP-less) RPC transport libs; they are replaced below.
RUN rm -f /usr/local/lib/libggml-rpc.so* \
          /usr/local/lib/libggml-base.so* \
          /usr/local/lib/libggml.so*

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

RUN apt-get update && apt-get install -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CUDA compute backend from the known-good previous image (enumerates GPU).
COPY --from=runtime_base /usr/local/lib/libggml-cuda.so*  /usr/local/lib/
COPY --from=runtime_base /usr/local/lib/libggml-cpu*.so*  /usr/local/lib/

# New UDP transport stack built from current source.
COPY --from=build /app/staging/bin/rpc-server        /usr/local/bin/rpc-server
COPY --from=build /app/staging/lib/libggml.so*       /usr/local/lib/
COPY --from=build /app/staging/lib/libggml-base.so*  /usr/local/lib/
COPY --from=build /app/staging/lib/libggml-rpc.so*   /usr/local/lib/

RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/llama.conf \
 && ldconfig \
 && chmod +x /usr/local/bin/rpc-server

ENV NVIDIA_VISIBLE_DEVICES=0
ENV CUDA_VISIBLE_DEVICES=0

EXPOSE 50051

ENTRYPOINT ["/usr/local/bin/rpc-server"]
CMD ["-H", "0.0.0.0", "-p", "50051", "-d", "CUDA0"]
