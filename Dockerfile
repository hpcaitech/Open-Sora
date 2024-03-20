FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel
WORKDIR /workspace
ENV CUDA_HOME=/usr/local/cuda
COPY . /workspace/Open-Sora

RUN apt-get update && apt-get install -y git \ 
    && pip3 install torch torchvision \
    && pip install opencv-python-headless \
    && pip install packaging ninja \
    && pip install flash-attn --no-build-isolation \
    && pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git \
    && pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121 \
    && cd Open-Sora \
    && pip install -v .
