FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*
COPY staging86/bin/rpc-server /usr/local/bin/rpc-server
COPY staging86/lib/ /usr/local/lib/
RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/llama.conf && ldconfig && chmod +x /usr/local/bin/rpc-server
ENV NVIDIA_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
EXPOSE 50051
ENTRYPOINT ["/usr/local/bin/rpc-server"]
CMD ["-H", "0.0.0.0", "-p", "50051", "-d", "CUDA0"]
