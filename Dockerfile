FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python-is-python3 python3-pip wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt test_input.json ./
RUN python3 -m pip install -r requirements.txt --no-cache-dir --break-system-packages && python3 -m pip cache purge
# Pre-download model from Hugging Face
RUN mkdir -p models/BiRefNet_lite/onnx && \
    wget -q -O models/BiRefNet_lite/onnx/model_fp16.onnx \
    https://huggingface.co/onnx-community/BiRefNet_lite-ONNX/resolve/main/onnx/model_fp16.onnx

COPY src/ ./
# Run the handler
ENV MODEL_TYPE="BiRefNet_lite"
CMD python3 -u /app/rp_handler.py
