FROM runpod/pytorch:0.7.0-dev-cu1241-torch260-ubuntu2204

WORKDIR /app

COPY requirements.txt test_input.json ./
RUN python3 -m pip install -r requirements.txt --no-cache-dir && python3 -m pip cache purge
# Pre-download and cache the model
RUN huggingface-cli download ZhengPeng7/BiRefNet_HR --cache-dir models/BiRefNet_HR && \
    huggingface-cli download ZhengPeng7/BiRefNet_lite --cache-dir models/BiRefNet_lite && \
    huggingface-cli download ZhengPeng7/BiRefNet_lite-2K --cache-dir models/BiRefNet_lite-2K && \
    huggingface-cli download ZhengPeng7/BiRefNet --cache-dir models/BiRefNet

COPY src/ ./
# Run the handler
ENV MODEL_TYPE="ZhengPeng7/BiRefNet_lite"
CMD python3 -u /app/rp_handler.py
