FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    python3-pip \
    ffmpeg \
    git \
    wget \
    build-essential \
    libsndfile1 \
    sox \
    python3-dev \
    unzip \
    tar \
    cmake \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3.8 /usr/bin/python

# Install basic Python dependencies first
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir numpy typing_extensions pydub sentencepiece gdown

# Install fairseq2
RUN pip install --no-cache-dir fairseq2==0.2.1

# GPU対応版PyTorchをインストール (CUDA 11.3 対応版)
RUN pip install --no-cache-dir torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# Copy application code for this component only
COPY seamless_communication/ /app/

# Install from GitHub directly as shown in the notebook
RUN pip install --no-cache-dir git+https://github.com/nawta/seamless_communication.git

# Create model directories
RUN mkdir -p /app/models

# ------------------------------
# (NEW) SeamlessExpressive modelのキャッシュ利用
# ------------------------------
COPY models_cache/ /tmp/models_cache/

RUN set -eux; \
    FILES="m2m_expressive_unity.pt pretssel_melhifigan_wm.pt pretssel_melhifigan_wm-16khz.pt"; \
    for f in $FILES; do \
        if [ -f "/tmp/models_cache/$f" ]; then \
            echo "[CACHE] copying $f"; \
            cp "/tmp/models_cache/$f" "/app/models/$f"; \
        fi; \
    done && \
    # いずれか欠けていれば一括ダウンロード
    if [ ! -f /app/models/m2m_expressive_unity.pt ] || [ ! -f /app/models/pretssel_melhifigan_wm.pt ] || [ ! -f /app/models/pretssel_melhifigan_wm-16khz.pt ]; then \
        echo "[DOWNLOAD] SeamlessExpressive model tar"; \
        cd /app/models && \
        gdown https://drive.google.com/uc?id=1oQzAtQiDPP0QVYM-UAVgE-_-LhDQxzcV -O SeamlessExpressive.tar && \
        tar -xzvf SeamlessExpressive.tar && \
        rm SeamlessExpressive.tar && \
        mv SeamlessExpressive/* . && \
        rm -r SeamlessExpressive; \
    fi && \
    # シンボリックリンクを再作成
    ln -sf /app/models/m2m_expressive_unity.pt /app/m2m_expressive_unity.pt && \
    ln -sf /app/models/pretssel_melhifigan_wm.pt /app/pretssel_melhifigan_wm.pt && \
    ln -sf /app/models/pretssel_melhifigan_wm-16khz.pt /app/pretssel_melhifigan_wm-16khz.pt

# Create directory for shared data
RUN mkdir -p /shared_data

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/app/models


# Default command
CMD ["tail", "-f", "/dev/null"]
