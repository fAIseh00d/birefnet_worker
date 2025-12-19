"""BiRefNet Background Removal Handler."""

import runpod
from runpod.serverless.modules.rp_logger import RunPodLogger
from runpod.serverless.utils.rp_validator import validate

import base64
import io
import gc
import os
from contextlib import contextmanager
from PIL import Image
import onnxruntime as ort

# Suppress ONNX Runtime warnings (only show errors)
ort.set_default_logger_severity(3)

from rembg_onnx import remove, BiRefNetSessionONNX, MODEL_CONFIGS

from rp_schemas import INPUT_SCHEMA

model_dict = {model: f"models/{model}.onnx" for model in MODEL_CONFIGS.keys()}

# Initialize RunPod logger for structured logging
log = RunPodLogger()

# Configure session options
sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

if "OMP_NUM_THREADS" in os.environ:
    threads = int(os.environ["OMP_NUM_THREADS"])
    sess_opts.inter_op_num_threads = threads
    sess_opts.intra_op_num_threads = threads

# Load BiRefNet model
log.info("Loading BiRefNet model session...")
birefnet_model_path = model_dict[os.getenv("MODEL_TYPE", "lite")]
birefnet_session = BiRefNetSessionONNX(birefnet_model_path, sess_opts)
log.info("BiRefNet model session loaded successfully")


@contextmanager
def image_processing_scope():
    """Context manager for image processing that ensures cleanup."""
    log.debug("Entering image processing scope")
    try:
        yield
    finally:
        log.debug("Cleaning up image processing scope")
        gc.collect()


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
        _ = image_pil.size
        image_pil.verify()
        # Reopen image since verify() closes it
        image_pil = Image.open(io.BytesIO(image_bytes))
        # Convert to RGB if not already
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
            log.debug("Image converted to RGB mode")
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


def process_image(image, options):
    """Process image with BiRefNet."""
    log.debug("Starting background removal processing")
    with image_processing_scope():
        result = remove(
            image,
            session=birefnet_session,
            only_mask=options.get("only_mask", False),
            bgcolor=tuple(options["bgcolor"]) if options.get("bgcolor") else None,
        )
        log.debug("Background removal completed successfully")
        return result


def handler(job):
    """Handler function that will be used to process jobs."""
    job_id = job.get("id", "unknown")
    log.info(f"Starting background removal job - Job ID: {job_id}")
    
    try:
        validate(job, INPUT_SCHEMA)
        log.debug("Job input validation passed")

        job_input = job["input"]
        file_name = job_input["filename"]
        image_string = job_input["image_b64"]
        
        # Extract processing options
        options = {
            "only_mask": job_input.get("only_mask", False),
            "bgcolor": job_input.get("bgcolor"),
        }
        
        log.debug(f"Processing image: {file_name}, options: {options}")
        
        # Load and validate image from base64
        image_pil = load_and_validate_image(image_string)
        
        # Main processing
        final_image = process_image(image_pil, options)
        
        # Convert final image to base64
        image_base64 = save_image_to_base64(final_image)
        
        log.info(f"Job completed successfully for {file_name} - Job ID: {job_id}")
        
        return {"filename": file_name, "image_b64": image_base64}
    
    except ValueError as ve:
        log.error(f"Validation error in job {job_id}: {str(ve)}")
        gc.collect()
        return {"error": f"Validation Error: {str(ve)}"}
    
    except Exception as e:
        log.error(f"Unexpected error in job {job_id} ({type(e).__name__}): {str(e)}")
        gc.collect()
        raise e


runpod.serverless.start({"handler": handler})
