FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install -r requirements.txt --no-cache-dir && pip cache purge
# Pre-download and cache the model
RUN python -c "from rembg import new_session; session = new_session('birefnet-general-lite')"

COPY handler.py schemas.py test_input.json ./

ARG CUDA_LIB_ROOT=/usr/local/lib/python3.11/dist-packages/nvidia/
ENV LD_LIBRARY_PATH=${CUDA_LIB_ROOT}cuda_runtime/lib:${CUDA_LIB_ROOT}cufft/lib:${CUDA_LIB_ROOT}cudnn/lib:${CUDA_LIB_ROOT}cuda_nvrtc/lib/:${CUDA_LIB_ROOT}cublas/lib/:${CUDA_LIB_ROOT}curand/lib/:${LD_LIBRARY_PATH}

# Run the handler
CMD python -u /app/handler.py
