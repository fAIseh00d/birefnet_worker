"""Test script for BiRefNet background removal."""

import sys
sys.path.insert(0, "src")

import onnxruntime as ort
from PIL import Image
from rembg_onnx import BiRefNetSessionONNX

# Configure session options
sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Load BiRefNet model
print("Loading BiRefNet model...")
birefnet = BiRefNetSessionONNX("models/BiRefNet_lite/onnx/model_fp16.onnx", sess_opts)
print("BiRefNet loaded.")

# Load test image
print("Loading test image...")
image = Image.open("00006_00.jpg").convert("RGB")
print(f"Image size: {image.size}")

# Get mask from BiRefNet
print("Running BiRefNet inference...")
masks = birefnet.predict(image)
mask = masks[0]

# Create cutout
birefnet_cutout = image.convert("RGBA")
birefnet_cutout.putalpha(mask)
birefnet_cutout.save("output_birefnet.png")
print("Saved: output_birefnet.png")

# Save mask
mask.save("output_mask.png")
print("Saved: output_mask.png")

print("\nDone!")

