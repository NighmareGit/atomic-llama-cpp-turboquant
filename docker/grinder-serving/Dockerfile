# =============================================================================
# Grinder Serving Hub — llama-server with CUDA, pinned to good-prototype HEAD
# =============================================================================
#
# PURPOSE
#   Permanent baseline serving image for the grinder's models on the 3060 Ti.
#   Builds the fork's llama-server + the full libggml backend stack
#   (base/cpu/cuda) from the STABLE main branch (good-prototype), so the image
#   always carries the latest stable patches. This supersedes the temporary
#   build-cuda-exp patch — the patch retires once this hub verifies.
#
# WHY A CONTAINER (not a host build dir)
#   The host build-cuda-b-bin got clobbered (now CPU-only). A pinned,
#   reproducible image is the durable baseline: the same commit → the same
#   binary, rebuilt on demand via the documented procedure below.
#
# PIN
#   The image is tagged with the good-prototype commit it was built from, e.g.
#   grinder-serving:855fdcca4. "Always the latest" = the rebuild procedure.
#
# REBUILD / PIN PROCEDURE  (when good-prototype advances)
#   cd atomic-llama-cpp-turboquant
#   COMMIT=$(git rev-parse good-prototype)
#   docker build --build-arg SOURCE_COMMIT=$COMMIT -t grinder-serving:$COMMIT \
#       -f docker/grinder-serving/Dockerfile .
#   docker tag grinder-serving:$COMMIT grinder-serving:latest
#   # then re-verify on the 3060 Ti (see RUN COMMAND below)
#
# RUN COMMAND (3060 Ti) — verified working
#   docker run -d --name grinder-hub --gpus all -p 8094:8080 \
#       -v <model-mount>/models/prototype-auto:/models \
#       grinder-serving:latest \
#       -m /models/Qwen3.5-9B-MTP-Q4_K_M.gguf \
#       --host 0.0.0.0 --port 8080 -ngl 999 -fa auto \
#       -ctk turbo3 -ctv turbo3 --ctx-size 8192 \
#       --spec-type draft-mtp --completion-auto-template --parallel 2
#
#   Notes:
#   - --gpus all (not device=0) so the runtime injects the GPU.
#   - -fa auto (explicit value) so the optional-value flag doesn't eat -ctk.
#   - MTP/TPQ flags are run-time (not baked in) — the image is the stable
#     transport, the run command is the model-specific tuning.
# =============================================================================

ARG CUDA_VERSION=12.8.0
ARG UBUNTU_VERSION=22.04

# ---- Build stage: compile llama-server + libggml stack from good-prototype ----
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build

# Pinned source commit (passed by build arg; falls back to "unknown").
ARG SOURCE_COMMIT=unknown
# CUDA architectures to compile for. Default covers the fleet's CUDA GPUs:
#   sm_86 = Ampere  (RTX 3060 Ti, 3090, 3070)
#   sm_120 = Blackwell (RTX 5060 Ti)
# Override with --build-arg CUDA_DOCKER_ARCH=<arch> (or "default" for all).
ARG CUDA_DOCKER_ARCH=86;120

LABEL org.opencontainers.image.description="Grinder serving hub — llama-server CUDA (good-prototype ${SOURCE_COMMIT})"

RUN apt-get update && apt-get install -y \
        gcc g++ build-essential cmake python3 git \
        libssl-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the fork source (good-prototype HEAD). .dockerignore excludes build*/,
# models*/, binaries — only the source tree + .git (for the build number) land
# here. The fork's tracked files are NOT modified (K1 — read-only source).
COPY . .

RUN if [ "${CUDA_DOCKER_ARCH}" != "default" ]; then \
        export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_DOCKER_ARCH}"; \
    fi && \
    cmake -B build \
        -DGGML_NATIVE=OFF \
        -DGGML_CUDA=ON \
        -DGGML_CUDA_F16=ON \
        -DGGML_BACKEND_DL=ON \
        -DGGML_CPU_ALL_VARIANTS=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_UI=OFF \
        -DLLAMA_USE_PREBUILT_UI=OFF \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_BUILD_TYPE=Release \
        ${CMAKE_ARGS} . && \
    cmake --build build --target llama-server -j$(nproc)

# Stage the libs + binary into clean layout dirs (mirrors .devops/cuda.Dockerfile:
# cp -P preserves the libggml.so -> libggml.so.0 -> libggml.so.0.15.1 chain).
# Copy ALL shared libs: the ggml backend stack (base/cpu/cuda) AND the llama
# libs the server binary links against (libllama-server-impl, libllama,
# libllama-common, libmtmd) — each with their versioned variants + symlinks.
RUN mkdir -p /app/lib && find build/bin -name "lib*.so*" -exec cp -P {} /app/lib \;
RUN mkdir -p /app/server && cp build/bin/llama-server /app/server/llama-server

# ---- Runtime stage: nvidia runtime + llama-server + libggml libs ----
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

ARG SOURCE_COMMIT=unknown

LABEL org.opencontainers.image.description="Grinder serving hub — llama-server CUDA (good-prototype ${SOURCE_COMMIT})"

RUN apt-get update && apt-get install -y \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# llama-server binary
COPY --from=build /app/server/llama-server /usr/local/bin/llama-server

# The full libggml backend stack (base + cpu + cuda) AND the llama libs the
# server binary links against (libllama-server-impl, libllama, libllama-common,
# libmtmd). Staged with symlinks preserved, so the sonames resolve after
# ldconfig.
# They live in /usr/local/lib/ for ldconfig, AND are symlinked into
# /usr/local/bin/ (the executable's directory) because ggml's dynamic backend
# loader (GGML_BACKEND_DL) searches the executable's own directory for
# libggml-cuda.so / libggml-cpu-*.so at runtime.
COPY --from=build /app/lib/ /usr/local/lib/

RUN ldconfig \
 && for f in /usr/local/lib/libggml*.so*; do ln -sf "$f" /usr/local/bin/; done \
 && for f in /usr/local/lib/libllama*.so* /usr/local/lib/libmtmd.so*; do ln -sf "$f" /usr/local/bin/; done

ENV LLAMA_ARG_HOST=0.0.0.0
EXPOSE 8080

HEALTHCHECK CMD ["curl", "-f", "http://localhost:8080/health"]

ENTRYPOINT ["/usr/local/bin/llama-server"]
