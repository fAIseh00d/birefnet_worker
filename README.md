# BiRefNet Background Removal Service

A high-performance background removal service built for RunPod Serverless using the BiRefNet model. This service automatically removes backgrounds from images using advanced AI technology optimized for production use.

![RunPod](https://api.runpod.io/badge/runpod-workers/worker-template)

## Features

- **High-Quality Background Removal**: Uses BiRefNet General Lite model for superior results
- **Memory Optimized**: Automatic image resizing and garbage collection for efficient processing
- **Multiple Format Support**: Supports JPEG, PNG, BMP, TIFF, and WEBP formats
- **Comprehensive Logging**: Structured JSON logging with RunPod Logger for debugging and monitoring
- **Error Handling**: Robust validation and error handling with detailed error messages
- **Production Ready**: Built for RunPod Serverless with optimized Docker container

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
- **rembg[gpu]**: Background removal with GPU acceleration
- **Pillow**: Image processing library
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
   python handler.py
   ```

   This will process the image in `test_input.json` and output the result.

3. **Test with custom input:**
   ```bash
   python handler.py --test_input '{"input": {"filename": "test.jpg", "image_b64": "your_base64_image_data"}}'
   ```

4. **Run local test server:**
   ```bash
   python handler.py --rp_serve_api
   ```
   
   Then test with curl:
   ```bash
   curl -X POST http://localhost:8000/run \
        -H "Content-Type: application/json" \
        -d '{"input": {"filename": "test.jpg", "image_b64": "your_base64_image_data"}}'
   ```

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

### Option 2: Manual Docker Build

1. **Build the image:**
   ```bash
   docker build -t birefnet-worker .
   ```

2. **Test locally:**
   ```bash
   docker run --gpus all -p 8080:8080 birefnet-worker
   ```

3. **Push to registry:**
   ```bash
   docker tag birefnet-worker your-registry/birefnet-worker:latest
   docker push your-registry/birefnet-worker:latest
   ```

## Performance Optimization

### Image Processing
- **Automatic Resizing**: Images larger than 1024px are automatically resized to prevent memory issues
- **Memory Management**: Aggressive garbage collection and context managers for optimal memory usage
- **Alpha Channel Handling**: Preserves original image quality by processing alpha channels separately when resizing is required

### Model Optimization
- **Pre-cached Model**: BiRefNet model is downloaded during Docker build for faster cold starts
- **GPU Acceleration**: Optimized for CUDA with proper library path configuration

## Logging and Monitoring

### Structured Logging
The service uses RunPod's structured JSON logging system with detailed information:

- **Job Processing**: Start/end of jobs with job IDs and filenames
- **Image Validation**: Format validation, size information, and processing steps
- **Memory Management**: Resize operations and garbage collection events
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
- **Model**: BiRefNet General Lite (optimized for speed/quality balance)
- **Processing Pipeline**: 
  1. Input validation and base64 decoding
  2. Image format validation and loading
  3. Memory-optimized preprocessing (resize if needed)
  4. Background removal with BiRefNet
  5. Alpha channel processing (if resizing was performed)
  6. Base64 encoding and response

### Error Handling
- **Input Validation**: Comprehensive base64 and image format validation
- **Memory Protection**: Automatic cleanup on errors with garbage collection
- **Graceful Degradation**: Structured error responses with detailed messages
- **Exception Handling**: Proper categorization of validation vs. system errors

### Memory Management
- **Context Managers**: Isolated scopes for different processing stages
- **Garbage Collection**: Automatic cleanup after processing and on errors
- **Image Resizing**: Automatic downsizing of large images to prevent OOM errors
- **Alpha Channel Optimization**: Efficient processing of transparency data

## Troubleshooting

### Common Issues

1. **Image Too Large**: The service automatically resizes images > 1024px
2. **Unsupported Format**: Ensure image is JPEG, PNG, BMP, TIFF, or WEBP
3. **Invalid Base64**: Verify base64 encoding is correct and complete
4. **Memory Errors**: Service includes automatic garbage collection

### Debugging

Check the structured logs for:
- Job ID to trace specific requests
- Image processing pipeline steps
- Error types and detailed messages
- Memory usage and optimization steps

## Support

- [RunPod Documentation](https://docs.runpod.io/serverless/overview)
- [RunPod Handler Functions Guide](https://docs.runpod.io/serverless/workers/handler-functions)
- [BiRefNet Paper](https://arxiv.org/abs/2401.15883)
- [rembg Library](https://github.com/danielgatis/rembg)
- [RunPod Discord Community](https://discord.gg/cUpRmau42V)

## License

See the [LICENSE](LICENSE) file for details.
