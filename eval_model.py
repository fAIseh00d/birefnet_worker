"""Test script for BiRefNet background removal with timing and batch support."""

import sys
import argparse
import time
sys.path.insert(0, "src")

import numpy as np
import onnxruntime as ort
from PIL import Image
from rembg_onnx import BiRefNetSessionONNX


def main():
    parser = argparse.ArgumentParser(description="Test BiRefNet ONNX model")
    parser.add_argument("--model", "-m", type=str, 
                        default="models/BiRefNet_lite/onnx/model_fp16_dynamic.onnx",
                        help="Path to ONNX model")
    parser.add_argument("--image", "-i", type=str, default="00006_00.jpg",
                        help="Input image path")
    parser.add_argument("--batch", "-b", type=int, default=1,
                        help="Batch size for testing (requires dynamic batch model)")
    parser.add_argument("--warmup", "-w", type=int, default=3,
                        help="Number of warmup runs")
    parser.add_argument("--runs", "-r", type=int, default=10,
                        help="Number of timed runs")
    parser.add_argument("--output", "-o", type=str, default="output_birefnet.png",
                        help="Output image path")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"BiRefNet ONNX Test")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch}")
    
    # Configure session options
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Load BiRefNet model
    print("\nLoading model...")
    t0 = time.perf_counter()
    birefnet = BiRefNetSessionONNX(args.model, sess_opts)
    load_time = time.perf_counter() - t0
    print(f"  Load time: {load_time*1000:.1f}ms")
    
    # Get model info
    input_info = birefnet.inner_session.get_inputs()[0]
    print(f"  Input: {input_info.name}, shape: {input_info.shape}, type: {input_info.type}")
    print(f"  Model size (HxW): {birefnet.input_height}x{birefnet.input_width}")
    print(f"  Providers: {birefnet.inner_session.get_providers()}")

    # Load and preprocess test image
    print(f"\nLoading image: {args.image}")
    image = Image.open(args.image).convert("RGB")
    print(f"  Size: {image.size}")

    # Prepare batch input (uses model's auto-detected input size)
    if args.batch > 1:
        print(f"\nPreparing batch of {args.batch} images...")
        # Get normalized input for single image
        single_input = birefnet.normalize(
            image, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), birefnet.input_size_pil
        )
        input_name = list(single_input.keys())[0]
        single_array = single_input[input_name]
        
        # Stack to create batch
        batch_input = {input_name: np.repeat(single_array, args.batch, axis=0)}
        print(f"  Batch shape: {batch_input[input_name].shape}")
    else:
        batch_input = birefnet.normalize(
            image, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), birefnet.input_size_pil
        )

    # Warmup runs
    print(f"\nWarmup ({args.warmup} runs)...")
    for i in range(args.warmup):
        _ = birefnet.inner_session.run(None, batch_input)
    print("  Done")

    # Timed runs
    print(f"\nTimed runs ({args.runs} runs)...")
    times = []
    for i in range(args.runs):
        t0 = time.perf_counter()
        outputs = birefnet.inner_session.run(None, batch_input)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        
    times = np.array(times) * 1000  # Convert to ms
    
    print(f"\n{'='*60}")
    print(f"Results (batch={args.batch})")
    print(f"{'='*60}")
    print(f"  Total time:    {times.mean():.2f}ms ± {times.std():.2f}ms")
    print(f"  Per image:     {times.mean()/args.batch:.2f}ms")
    print(f"  Throughput:    {1000*args.batch/times.mean():.1f} images/sec")
    print(f"  Min:           {times.min():.2f}ms")
    print(f"  Max:           {times.max():.2f}ms")

    # Save output from first image in batch
    if args.output:
        pred = birefnet.sigmoid(outputs[0][0:1, 0, :, :])
        ma = np.max(pred)
        mi = np.min(pred)
        pred = (pred - mi) / (ma - mi)
        pred = np.squeeze(pred)
        pred = np.where(pred >= 0.98, 1.0, pred)
        pred = np.where(pred <= 0.02, 0.0, pred)
        
        mask = Image.fromarray((pred * 255).astype("uint8"))
        mask = mask.resize(image.size, Image.Resampling.LANCZOS)
        
        # Create cutout
        cutout = image.convert("RGBA")
        cutout.putalpha(mask)
        cutout.save(args.output)
        print(f"\nSaved: {args.output}")
        
        # Save mask
        mask_path = args.output.replace('.png', '_mask.png')
        mask.save(mask_path)
        print(f"Saved: {mask_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
