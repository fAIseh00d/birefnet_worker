"""BiRefNet Background Removal Handler with proper logging."""

import runpod
from runpod import RunpodLogger
from runpod.serverless.utils.rp_validator import validate

from schemas import INPUT_SCHEMA
from rembg import remove, new_session
from PIL import Image
import base64
import io
import os
import gc
from contextlib import contextmanager

# Initialize RunPod logger for structured logging
log = RunpodLogger()

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.
log.info("Loading BiRefNet model session...")
session = new_session("birefnet-general-lite")
log.info("BiRefNet model session loaded successfully")

def preprocess_image(image_pil, max_size=1024):
    """Preprocess image to ensure it's within memory limits."""
    # Get original dimensions
    width, height = image_pil.size
    original_size = (width, height)
    was_resized = False
    
    log.debug(f"Original image dimensions: {width}x{height}")
    
    # Calculate if resizing is needed
    if max(width, height) > max_size:
        # Calculate new dimensions maintaining aspect ratio
        if width > height:
            new_width = max_size
            new_height = int((height * max_size) / width)
        else:
            new_height = max_size
            new_width = int((width * max_size) / height)
        
        # Resize image
        resized_image = image_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
        log.info(f"Resized image from {width}x{height} to {new_width}x{new_height} for memory optimization")
        was_resized = True
        return resized_image, original_size, was_resized
    
    log.debug("No resizing needed, image within memory limits")
    return image_pil, original_size, was_resized

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
        log.info(f"Image loaded and validated successfully - Format: {image_pil.format}, Size: {image_pil.size}")
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
    log.info("Starting BiRefNet background removal processing")
    with image_processing_scope():
        result = remove(processed_image, session=session)
        log.info("BiRefNet background removal completed successfully")
        return result

def resize_alpha_and_composite(image_rembg, original_image, original_size):
    """Isolated scope for alpha resizing and compositing."""
    log.info(f"Starting alpha channel processing for original size: {original_size}")
    
    # Extract alpha channel from the processed result
    alpha_channel = image_rembg.split()[-1]
    
    # Resize alpha channel back to original size
    alpha_resized = alpha_channel.resize(original_size, Image.Resampling.LANCZOS)
    
    # Convert original image to RGBA if it isn't already
    if original_image.mode != 'RGBA':
        original_image = original_image.convert('RGBA')
    
    # Apply the resized alpha channel to the original image
    original_image.putalpha(alpha_resized)
    
    log.info(f"Applied resized alpha channel back to original {original_size} image")
    return original_image

def handler(job):
    """Handler function that will be used to process jobs."""
    job_id = job.get("id", "unknown")
    log.info(f"Starting BiRefNet background removal job", extra={"job_id": job_id})
    
    try:
        validate(job, INPUT_SCHEMA)
        log.debug("Job input validation passed")

        file_name = job["input"]["filename"]
        image_string = job["input"]["image_b64"]
        
        log.info(f"Processing image: {file_name}")
        
        # Load and validate image from base64
        image_pil = load_and_validate_image(image_string)
        
        # Main processing in isolated scopes
        def process_image():
            log.info("Starting main image processing pipeline")
            
            # Keep original image for final compositing
            original_image = image_pil.copy()
            
            # Preprocess image to prevent memory issues
            processed_image, original_size, was_resized = preprocess_image(image_pil)

            # Process through BiRefNet in isolated scope
            image_rembg = process_birefnet_image(processed_image)
            
            # Handle resizing in isolated scope
            if was_resized:
                log.info("Applying alpha channel processing due to image resizing")
                final_image = resize_alpha_and_composite(image_rembg, original_image, original_size)
            else:
                log.info("Using BiRefNet output directly, no resizing was performed")
                final_image = image_rembg
            
            return final_image
        
        # Execute processing in isolated scope
        final_image = process_image()
        
        # Convert final image to base64
        image_base64 = save_image_to_base64(final_image)
        
        log.info(f"Job completed successfully for {file_name}", extra={
            "job_id": job_id,
            "filename": file_name,
            "success": True
        })
        
        return {"filename": file_name, "image_b64": image_base64}
    
    except ValueError as ve:
        # Handle validation errors specifically
        log.error(f"Validation error in job {job_id}: {str(ve)}", extra={
            "job_id": job_id,
            "error_type": "ValidationError",
            "error_message": str(ve)
        })
        gc.collect()
        return {"error": f"Validation Error: {str(ve)}"}
    
    except Exception as e:
        # Handle unexpected errors
        log.error(f"Unexpected error in job {job_id}: {str(e)}", extra={
            "job_id": job_id,
            "error_type": type(e).__name__,
            "error_message": str(e)
        })
        # Force garbage collection on error
        gc.collect()
        raise e


runpod.serverless.start({"handler": handler})
