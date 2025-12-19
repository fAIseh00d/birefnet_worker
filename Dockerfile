FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python-is-python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt test_input.json ./
RUN python3 -m pip install -r requirements.txt --no-cache-dir --break-system-packages && python3 -m pip cache purge
# Pre-download model from Hugging Face
COPY models/lite.onnx models/lite.onnx
COPY models/lite_2k.onnx models/lite_2k.onnx

COPY src/ ./
RUN python -m compileall /app

# Suppress NVIDIA CUDA banner in logs
ENTRYPOINT []

# Run the handler
CMD python3 -u /app/rp_handler.py
