"""BiRefNet Background Removal Handler with proper logging."""

import runpod
from runpod.serverless.modules.rp_logger import RunPodLogger
from runpod.serverless.utils.rp_validator import validate

import base64
import io
import gc
from contextlib import contextmanager
from PIL import Image

from schemas import INPUT_SCHEMA
from modules.birefnet import BirefnetHandler
from modules.utils import tform_to_pil, tform_to_tensor, device
# Initialize RunPod logger for structured logging
log = RunPodLogger()

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.
log.info("Loading BiRefNet model session...")
birefnet_handler = BirefnetHandler()
log.info("BiRefNet model session loaded successfully")

@contextmanager
def image_processing_scope():
    """Context manager for image processing that ensures cleanup."""
    log.debug("Entering image processing scope")
    try:
        yield
    finally:
        log.debug("Cleaning up image processing scope")
        gc.collect()  # Clean up when exiting scope

def load_and_validate_image(image_string):
    """Load and validate image from base64 string."""
    log.debug("Starting image validation and loading")
    
    # Validate and decode base64
    try:
        image_bytes = base64.b64decode(image_string)
        log.debug("Base64 image decoded successfully")
    except Exception as e:
        log.error(f"Base64 decoding failed: {str(e)}")
        raise ValueError(f"Invalid base64 image data: {str(e)}")
    
    # Validate image bytes are not empty
    if not image_bytes:
        log.error("Decoded image data is empty")
        raise ValueError("Decoded image data is empty")
    
    # Validate and open PIL image
    try:
        image_pil = Image.open(io.BytesIO(image_bytes))
        # Verify image can be loaded by accessing its properties
        _ = image_pil.size  # This will raise an exception if image is corrupted
        image_pil.verify()  # Additional verification
        # Reopen image since verify() closes it
        image_pil = Image.open(io.BytesIO(image_bytes))
        # Convert to RGB if not already
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
            log.debug(f"Image converted to RGB mode")
        log.debug(f"Image loaded and validated successfully - Format: {image_pil.format}, Size: {image_pil.size}")
    except Exception as e:
        log.error(f"Image validation failed: {str(e)}")
        raise ValueError(f"Invalid or corrupted image data: {str(e)}")
    
    # Validate image format
    if image_pil.format not in ['JPEG', 'PNG', 'BMP', 'TIFF', 'WEBP']:
        log.error(f"Unsupported image format: {image_pil.format}")
        raise ValueError(f"Unsupported image format: {image_pil.format}. Supported formats: JPEG, PNG, BMP, TIFF, WEBP")
    
    return image_pil

def save_image_to_base64(image, format="PNG", optimize=True):
    """Save PIL image to base64 string."""
    log.debug(f"Converting image to base64 - Format: {format}, Optimize: {optimize}")
    buffered = io.BytesIO()
    image.save(buffered, format=format, optimize=optimize)
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    log.debug(f"Image converted to base64 successfully - Size: {len(image_base64)} characters")
    return image_base64

def process_birefnet_image(processed_image):
    """Isolated scope for BiRefNet processing."""
    log.debug("Starting BiRefNet background removal processing")
    with image_processing_scope():
        image = tform_to_tensor(processed_image).to(device)
        image = birefnet_handler.process_imgs([image])[0]
        image = tform_to_pil(image)
        log.debug("BiRefNet background removal completed successfully")
        return image

def handler(job):
    """Handler function that will be used to process jobs."""
    job_id = job.get("id", "unknown")
    log.info(f"Starting BiRefNet background removal job - Job ID: {job_id}")
    
    try:
        validate(job, INPUT_SCHEMA)
        log.debug("Job input validation passed")

        file_name = job["input"]["filename"]
        image_string = job["input"]["image_b64"]
        
        log.debug(f"Processing image: {file_name}")
        
        # Load and validate image from base64
        image_pil = load_and_validate_image(image_string)
        
        # Main processing in isolated scopes
        final_image = process_birefnet_image(image_pil)
        
        # Convert final image to base64
        image_base64 = save_image_to_base64(final_image)
        
        log.info(f"Job completed successfully for {file_name} - Job ID: {job_id}")
        
        return {"filename": file_name, "image_b64": image_base64}
    
    except ValueError as ve:
        # Handle validation errors specifically
        log.error(f"Validation error in job {job_id}: {str(ve)}")
        gc.collect()
        return {"error": f"Validation Error: {str(ve)}"}
    
    except Exception as e:
        # Handle unexpected errors
        log.error(f"Unexpected error in job {job_id} ({type(e).__name__}): {str(e)}")
        # Force garbage collection on error
        gc.collect()
        raise e


runpod.serverless.start({"handler": handler})
