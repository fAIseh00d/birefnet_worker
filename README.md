![BiRefNet Background Removal Service](/public/banner.jpg)

---
A fast, GPU-accelerated background removal service using BiRefNet neural network, optimized for RunPod serverless deployment.

## Features

- **High Quality**: Uses BiRefNet for accurate foreground-background segmentation
- **GPU Accelerated**: Optimized for NVIDIA GPUs with CUDA
- **Serverless Ready**: Built for RunPod with proper logging and error handling
- **Minimal Dependencies**: Only essential Python packages
- **Fast Startup**: Pre-compiled ONNX model with optimized inference

## Quick Start

### Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Run test
python test.py
```

### Docker Build

```bash
docker build -t birefnet_worker .
```

### RunPod Deployment

Upload the project files to RunPod and use the included Dockerfile.

## API

### Input Format

```json
{
  "input": {
    "filename": "image.jpg",
    "image_b64": "<base64_encoded_image>",
    "only_mask": false,
    "bgcolor": [255, 255, 255, 255]
  }
}
```

### Parameters

- `filename` (string, required): Image filename
- `image_b64` (string, required): Base64 encoded image data
- `only_mask` (boolean, optional): Return only binary mask instead of cutout (default: false)
- `bgcolor` (array, optional): Background color as [R, G, B, A] (0-255) (default: transparent)

### Output Format

```json
{
  "filename": "image.jpg",
  "image_b64": "<base64_encoded_result>"
}
```

### Supported Image Formats

- JPEG
- PNG
- BMP
- TIFF
- WebP

## Architecture

```
Input Image → BiRefNet ONNX → Binary Mask → Alpha Composite → Output Image
```

### Model Details

- **Model**: BiRefNet Lite (FP16 ONNX)
- **Input**: RGB images, any size (auto-resized to 1024x1024 internally)
- **Output**: Soft mask (0-255 values)
- **Backend**: ONNX Runtime with CUDA acceleration

## Performance

- **Inference Time**: ~200-500ms per image (depending on image size)
- **Memory**: ~2GB GPU memory usage
- **Startup**: < 5 seconds (model pre-loaded)

## Dependencies

- `runpod~=1.8.1` - RunPod serverless framework
- `Pillow>=10.2.0` - Image processing
- `onnxruntime-gpu` - GPU-accelerated ONNX inference

## Development

### Project Structure

```
├── src/
│   ├── rembg_onnx/
│   │   ├── __init__.py
│   │   ├── bg.py           # Background removal logic
│   │   └── birefnet_onnx.py # BiRefNet ONNX session
│   ├── rp_handler.py       # RunPod handler
│   └── rp_schemas.py       # Input validation schemas
├── models/
│   └── BiRefNet_lite/
│       └── onnx/
│           └── model_fp16.onnx
├── Dockerfile
├── requirements.txt
└── test.py
```

### Environment Variables

- `BIREFNET_MODEL_PATH`: Path to BiRefNet ONNX model (default: "models/BiRefNet_lite/onnx/model_fp16.onnx")
- `OMP_NUM_THREADS`: Number of CPU threads for ONNX (auto-detected)

## License

See LICENSE file for details.

## Credits

- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) - Background removal model
- [ONNX Community](https://huggingface.co/onnx-community/BiRefNet_lite-ONNX) - Converted ONNX model
- [RunPod](https://runpod.io) - Serverless deployment platform