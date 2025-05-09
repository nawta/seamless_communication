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
    cmake \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install basic Python dependencies first
RUN pip install --upgrade pip
RUN pip install --no-cache-dir numpy typing_extensions pydub sentencepiece

# Install fairseq2 first
RUN pip install --no-cache-dir fairseq2==0.2.1

# Copy application code
COPY . /app/

# Install from GitHub directly as shown in the notebook
RUN pip install --no-cache-dir git+https://github.com/nawta/seamless_communication.git

# Create model directories
RUN mkdir -p /app/models/SeamlessExpressive

# Download SeamlessExpressive model from Google Drive
RUN cd /app/models && \\
    gdown https://drive.google.com/uc?id=1oQzAtQiDPP0QVYM-UAVgE-_-LhDQxzcV -O SeamlessExpressive.tar && \\
    tar -xzvf SeamlessExpressive.tar && \\
    rm SeamlessExpressive.tar

# Create directory for shared data
RUN mkdir -p /shared_data

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/app/models/SeamlessExpressive


# Default command
CMD ["tail", "-f", "/dev/null"]
