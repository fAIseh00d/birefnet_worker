"""Example handler file."""

import runpod
from runpod.serverless.utils.rp_validator import validate

from schemas import INPUT_SCHEMA
from rembg import remove, new_session
from PIL import Image
import base64
import io
import os

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.
session = new_session("birefnet-general-lite")

def handler(job):
    """Handler function that will be used to process jobs."""
    validate(job, INPUT_SCHEMA)

    file_name = job["input"]["filename"]
    image_string = job["input"]["image_b64"]
    
    image_bytes = base64.b64decode(image_string)
    image_pil = Image.open(io.BytesIO(image_bytes))

    image_rembg = remove(image_pil, session=session)

    buffered = io.BytesIO()
    image_rembg.save(buffered, format="PNG", optimize=True)
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {"filename": file_name, "image_b64": image_base64}


runpod.serverless.start({"handler": handler})
