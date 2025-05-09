FROM python:3.8-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
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
    && rm -rf /var/lib/apt/lists/*

# Install basic Python dependencies first
RUN pip install --upgrade pip
RUN pip install --no-cache-dir numpy typing_extensions pydub sentencepiece gdown

# Install fairseq2 first
RUN pip install --no-cache-dir fairseq2==0.2.1

# Copy application code
COPY . /app/

# Install from GitHub directly as shown in the notebook
RUN pip install --no-cache-dir git+https://github.com/nawta/seamless_communication.git

# Create model directories
RUN mkdir -p /app/models

# Download SeamlessExpressive model from Google Drive and place it in the models directory
RUN cd /app/models && \
    gdown https://drive.google.com/uc?id=1oQzAtQiDPP0QVYM-UAVgE-_-LhDQxzcV -O SeamlessExpressive.tar && \
    tar -xzvf SeamlessExpressive.tar && \
    rm SeamlessExpressive.tar && \
    mv SeamlessExpressive/* . && \
    rm -r SeamlessExpressive

# Create a symbolic link to the model file
RUN ln -s /app/models/m2m_expressive_unity.pt /app/m2m_expressive_unity.pt
RUN ln -s /app/models/pretssel_melhifigan_wm.pt /app/pretssel_melhifigan_wm.pt
RUN ln -s /app/models/pretssel_melhifigan_wm-16khz.pt /app/pretssel_melhifigan_wm-16khz.pt

# Create directory for shared data
RUN mkdir -p /shared_data

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/app/models


# Default command
CMD ["tail", "-f", "/dev/null"]
