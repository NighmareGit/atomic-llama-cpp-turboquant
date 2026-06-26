# Full multi-stage Dockerfile (requires: docker build --gpus=all on hosts that support it).
# On remus use ./build.sh instead (GPU compile via docker run + Dockerfile.runtime).
ARG PROXY_IP=192.168.8.108

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS build-stage

RUN bash -c 'timeout 1 bash -c "cat < /dev/null > /dev/tcp/'${PROXY_IP}'/3128" 2>/dev/null && (echo "Acquire::http::Proxy \"http://'${PROXY_IP}':3128\";" > /etc/apt/apt.conf.d/01proxy) || echo "Squid proxy offline"'

RUN apt-get update && apt-get install -y \
    git cmake ninja-build build-essential \
    libopenblas-dev libomp-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /build

ARG GIT_URL=http://192.168.8.108:3005/hunter/atomic-llama-cpp-turboquant.git
ARG GIT_BRANCH=Path-B-Event-Support

RUN git clone "${GIT_URL}" . && git checkout "${GIT_BRANCH}"

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
RUN cmake -S . -B build -G Ninja \
    -DGGML_CUDA=ON \
    -DGGML_RPC=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="120a-real" \
    -DGGML_CUDA_F16=ON \
    -DGGML_CUDA_K_QUANTS=ON \
    -DGGML_CUDA_MMQ=ON \
    -DGGML_SCHED_MAX_COPIES=4 \
    && cmake --build build --config Release --target rpc-server llama-server -- -j$(nproc)

RUN mkdir -p /staging/bin /staging/lib \
    && cp build/bin/rpc-server build/bin/llama-server /staging/bin/ \
    && find build -name "*.so*" -exec cp -v {} /staging/lib/ \;

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=build-stage /staging/bin/rpc-server /usr/local/bin/rpc-server
COPY --from=build-stage /staging/bin/llama-server /usr/local/bin/llama-server
COPY --from=build-stage /staging/lib/ /usr/local/lib/

RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/llama.conf && ldconfig

ENV NVIDIA_VISIBLE_DEVICES=0
ENV CUDA_VISIBLE_DEVICES=0

EXPOSE 50051

ENTRYPOINT ["/usr/local/bin/rpc-server"]
CMD ["-H", "0.0.0.0", "-p", "50051", "-d", "CUDA0"]