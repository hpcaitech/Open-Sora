FROM hpcaitech/pytorch-cuda:2.1.0-12.1.0

# metainformation
LABEL org.opencontainers.image.source = "https://github.com/hpcaitech/Open-Sora"
LABEL org.opencontainers.image.licenses = "Apache License 2.0"
LABEL org.opencontainers.image.base.name = "docker.io/library/hpcaitech/pytorch-cuda:2.1.0-12.1.0"

# Set the working directory
WORKDIR /workspace/Open-Sora
# Copy the current directory contents into the container at /workspace/Open-Sora
COPY . .

# inatall library dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# install flash attention
RUN pip install flash-attn --no-build-isolation

# install apex
RUN pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git

# install xformers
RUN pip install xformers --index-url https://download.pytorch.org/whl/cu121

# install this project
RUN pip install -v .
