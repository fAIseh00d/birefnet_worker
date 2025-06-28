![BiRefNet Background Removal Service](/public/banner.jpg)

---

# BiRefNet Background Removal Service

A high-performance background removal service built for RunPod Serverless using the BiRefNet model. This service automatically removes backgrounds from images using advanced AI technology optimized for production use.

[![Runpod](https://api.runpod.io/badge/fAIseh00d/birefnet_worker)](https://console.runpod.io/hub/fAIseh00d/birefnet_worker)

## Features

- **High-Quality Background Removal**: Uses BiRefNet models with configurable model selection (BiRefNet, BiRefNet_lite, BiRefNet_HR, BiRefNet_lite-2K)
- **Memory Optimized**: Automatic garbage collection and context managers for efficient processing
- **Multiple Format Support**: Supports JPEG, PNG, BMP, TIFF, and WEBP formats
- **Comprehensive Logging**: Structured JSON logging with RunPod Logger for debugging and monitoring
- **Error Handling**: Robust validation and error handling with detailed error messages
- **Production Ready**: Built for RunPod Serverless with optimized Docker container and pre-cached models
- **Modular Architecture**: Clean separation of concerns with dedicated modules for BiRefNet processing and utilities

## API Reference

### Input Schema

```json
{
  "input": {
    "filename": "string (required)",
    "image_b64": "string (required, base64-encoded image)"
  }
}
```

### Response Schema

**Success Response:**
```json
{
  "filename": "string",
  "image_b64": "string (base64-encoded image with background removed)"
}
```

**Error Response:**
```json
{
  "error": "string (error description)"
}
```

### Example Usage

```json
{
  "input": {
    "filename": "my_photo.jpg",
    "image_b64": "iVBORw0KGgoAAAANSUhEUgAAAIAAAACSCAIAAAACQlUMAAAA..."
  }
}
```

## Dependencies

- **runpod**: Serverless framework with logging utilities
- **transformers**: Hugging Face transformers library for BiRefNet model loading
- **Pillow**: Image processing library
- **einops**: Tensor operations for flexible array manipulations
- **kornia**: Computer vision library for geometric transformations
- **timm**: PyTorch image models library
- **PyTorch**: Deep learning framework (via base image)

## Local Development

### Prerequisites

1. Python 3.11+
2. CUDA-compatible GPU (for optimal performance)
3. Docker (for containerized testing)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Testing Locally

1. **Prepare test input:**
   - Modify `test_input.json` with your base64-encoded image
   - Ensure the image is in a supported format (JPEG, PNG, BMP, TIFF, WEBP)

2. **Run the handler:**
   ```bash
   python src/rp_handler.py
   ```

   This will process the image in `test_input.json` and output the result.

3. **Test with custom input:**
   ```bash
   python src/rp_handler.py --test_input '{"input": {"filename": "test.jpg", "image_b64": "your_base64_image_data"}}'
   ```

4. **Run local test server:**
   ```bash
   python src/rp_handler.py --rp_serve_api
   ```
   
   Then test with curl:
   ```bash
   curl -X POST http://localhost:8000/run \
        -H "Content-Type: application/json" \
        -d '{"input": {"filename": "test.jpg", "image_b64": "your_base64_image_data"}}'
   ```

   Note: The Docker container runs `python -u /app/rp_handler.py` by default with `MODEL_TYPE=ZhengPeng7/BiRefNet_lite`.

## Deployment

### Option 1: GitHub Integration (Recommended)

1. **Connect to RunPod:**
   - Go to [RunPod Serverless Console](https://www.runpod.io/console/serverless)
   - Create a new template
   - Connect your GitHub repository
   - RunPod will automatically build and deploy on git push

2. **Configure deployment:**
   - Set appropriate GPU resources (T4, RTX A4000, RTX 4090, etc.)
   - Configure scaling settings based on expected load
   - Set timeout values appropriately for image processing
   - Set environment variable `MODEL_TYPE` to choose model variant:
     - `ZhengPeng7/BiRefNet_lite` (default, fastest)
     - `ZhengPeng7/BiRefNet` (balanced)
     - `ZhengPeng7/BiRefNet_HR` (high resolution)
     - `ZhengPeng7/BiRefNet_lite-2K` (optimized for 2K images)

### Option 2: Manual Docker Build

1. **Build the image:**
   ```bash
   docker build -t birefnet-worker .
   ```

2. **Test locally:**
   ```bash
   docker run --gpus all -p 8000:8000 birefnet-worker
   ```

3. **Push to registry:**
   ```bash
   docker tag birefnet-worker your-registry/birefnet-worker:latest
   docker push your-registry/birefnet-worker:latest
   ```

## Performance Optimization

### Image Processing
- **Memory Management**: Aggressive garbage collection and context managers for optimal memory usage
- **Alpha Channel Handling**: Proper RGBA processing for transparent backgrounds
- **Input Validation**: Comprehensive format and data validation before processing

### Model Optimization
- **Pre-cached Models**: All BiRefNet model variants are downloaded during Docker build for faster cold starts
- **GPU Acceleration**: Optimized for CUDA with automatic device detection and tensor operations
- **Custom Implementation**: BirefnetHandler class with optimized transformations and post-processing pipeline

## Logging and Monitoring

### Structured Logging
The service uses RunPod's structured JSON logging system with detailed information:

- **Job Processing**: Start/end of jobs with job IDs and filenames
- **Image Validation**: Format validation, size information, and processing steps
- **Memory Management**: Context managers and garbage collection events
- **Error Handling**: Detailed error messages with context and error types
- **Performance Metrics**: Processing pipeline steps and timing information

### Log Levels
- **INFO**: Major processing steps, job status, successful operations
- **DEBUG**: Detailed processing information, image dimensions, internal states
- **ERROR**: Validation failures, processing errors, system issues

### Monitoring in RunPod Console
Monitor these logs in the RunPod console for:
- Job success/failure rates
- Processing performance
- Memory usage patterns
- Error patterns and debugging

## Technical Details

### Architecture
- **Base Image**: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- **Model**: Configurable BiRefNet models (default: ZhengPeng7/BiRefNet_lite) loaded via transformers
- **Processing Pipeline**: 
  1. Input validation and base64 decoding
  2. Image format validation and loading (auto-convert to RGB)
  3. BiRefNet segmentation processing with 1024x1024 resizing and normalization
  4. Post-processing with bilinear interpolation and alpha channel composition
  5. Base64 encoding and response

### Module Structure
- **`src/rp_handler.py`**: Main serverless handler with RunPod integration
- **`src/modules/birefnet.py`**: Custom BiRefNet implementation with BirefnetHandler class
- **`src/modules/utils.py`**: Utility functions for tensor/image transformations and device configuration
- **`src/rp_schemas.py`**: Input validation schema definitions

### Error Handling
- **Input Validation**: Comprehensive base64 and image format validation
- **Memory Protection**: Context managers and automatic cleanup on errors
- **Graceful Degradation**: Structured error responses with detailed messages
- **Exception Handling**: Proper categorization of validation vs. system errors

### Memory Management
- **Context Managers**: Isolated scopes for different processing stages with automatic cleanup
- **Garbage Collection**: Automatic cleanup after processing and on errors
- **GPU Memory**: Efficient tensor operations with proper device management

## Troubleshooting

### Common Issues

1. **Unsupported Format**: Ensure image is JPEG, PNG, BMP, TIFF, or WEBP
2. **Invalid Base64**: Verify base64 encoding is correct and complete
3. **Memory Errors**: Service includes automatic garbage collection and context management
4. **GPU Issues**: Check CUDA availability and library paths

### Debugging

Check the structured logs for:
- Job ID to trace specific requests
- Image processing pipeline steps
- Error types and detailed messages
- Memory usage and optimization steps

## Support

- [RunPod Documentation](https://docs.runpod.io/serverless/overview)
- [RunPod Handler Functions Guide](https://docs.runpod.io/serverless/workers/handler-functions)
- [BiRefNet Repo](https://github.com/ZhengPeng7/BiRefNet)
- [Transformers Library](https://huggingface.co/docs/transformers)
- [RunPod Discord Community](https://discord.gg/cUpRmau42V)

## License

See the [LICENSE](LICENSE) file for details.
