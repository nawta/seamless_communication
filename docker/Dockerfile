FROM python:3.8-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    build-essential \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir audiocraft

# Create directory for shared data
RUN mkdir -p /shared_data

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "-c", "import time; print('seamless_communication container is running'); time.sleep(infinity)"]
