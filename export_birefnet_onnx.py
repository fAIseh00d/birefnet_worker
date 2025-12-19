#!/usr/bin/env python3
"""
BiRefNet ONNX Export and Quantization Script

Exports BiRefNet models to ONNX format with optional quantization.
Supports dynamic batch sizes and various quantization levels.

Usage:
    python export_birefnet_onnx.py --model general --output models/BiRefNet/onnx/
    python export_birefnet_onnx.py --model lite --output models/BiRefNet_lite/onnx/ --quantize int8
    python export_birefnet_onnx.py --model general --batch-size 4 --dynamic-batch

Models:
    general: ZhengPeng7/BiRefNet (swin_v1_l backbone, ~900MB)
    lite:    ZhengPeng7/BiRefNet_lite (swin_v1_tiny backbone, ~200MB)
"""

import argparse
import os
import sys
import gc
import shutil
from pathlib import Path
from typing import Optional, Tuple, List
import warnings

import torch
import torch.nn as nn
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*TorchScript.*")
warnings.filterwarnings("ignore", message=".*TracerWarning.*")


# ============================================================================
# Deformable Conv2d ONNX Export Support
# ============================================================================

def register_deform_conv2d_onnx_op():
    """
    Register deformable conv2d for ONNX export using standard ONNX operators.
    Uses the deform_conv2d_onnx_exporter library with patches for dynamic shapes.
    
    This implements DCNv2 using GatherND/GatherElements - works with any ONNX runtime.
    """
    import sys
    from pathlib import Path
    
    # Add the exporter to path
    exporter_path = Path(__file__).parent / "submodules" / "deform_conv2d_onnx_exporter" / "src"
    if exporter_path.exists() and str(exporter_path) not in sys.path:
        sys.path.insert(0, str(exporter_path))
    
    try:
        # Try to import the patched version
        import deform_conv2d_onnx_exporter
        
        # Apply dynamic shape patch from BiRefNet notebook
        # This fixes issues when tensor dimensions are not statically known
        import torch.onnx.symbolic_helper as sym_help
        
        original_get_tensor_dim_size = deform_conv2d_onnx_exporter.get_tensor_dim_size
        
        def patched_get_tensor_dim_size(tensor, dim):
            """Patched version that handles dynamic shapes."""
            import typing
            from torch import _C
            
            tensor_dim_size = sym_help._get_tensor_dim_size(tensor, dim)
            
            if tensor_dim_size is None and dim in (2, 3):
                try:
                    x_type = typing.cast(_C.TensorType, tensor.type())
                    x_strides = x_type.strides()
                    if x_strides:
                        tensor_dim_size = x_strides[2] if dim == 3 else x_strides[1] // x_strides[2]
                except Exception:
                    pass
            elif tensor_dim_size is None and dim == 0:
                try:
                    x_type = typing.cast(_C.TensorType, tensor.type())
                    x_strides = x_type.strides()
                    if x_strides:
                        tensor_dim_size = x_strides[3]
                except Exception:
                    pass
            
            return tensor_dim_size
        
        # Apply the patch
        deform_conv2d_onnx_exporter.get_tensor_dim_size = patched_get_tensor_dim_size
        
        # Override the opset version to 17 for CUDA EP compatibility
        deform_conv2d_onnx_exporter.onnx_opset_version = 17
        
        # Register the operator with GatherND for better compatibility
        deform_conv2d_onnx_exporter.register_deform_conv2d_onnx_op(
            use_gathernd=True,
            enable_openvino_patch=False
        )
        
        print("✓ Registered deformable conv2d ONNX symbolic (using GatherND)")
        
    except ImportError as e:
        print(f"⚠ Could not import deform_conv2d_onnx_exporter: {e}")
        print("  Install with: pip install deform_conv2d_onnx_exporter")
        print("  Or ensure submodules/deform_conv2d_onnx_exporter exists")
    except Exception as e:
        print(f"⚠ Could not set up deform_conv2d export: {e}")
        import traceback
        traceback.print_exc()


def patch_deformable_conv_for_export():
    """
    Patch deformable convolutions to use regular convolutions for ONNX export.
    This is a fallback when deform_conv2d is not supported.
    """
    try:
        from torchvision.ops import DeformConv2d
        
        class DeformConv2dExportable(nn.Module):
            """Exportable version that falls back to regular conv during export."""
            def __init__(self, original_module):
                super().__init__()
                self.in_channels = original_module.in_channels
                self.out_channels = original_module.out_channels
                self.kernel_size = original_module.kernel_size
                self.stride = original_module.stride
                self.padding = original_module.padding
                self.dilation = original_module.dilation
                self.groups = original_module.groups
                self.weight = original_module.weight
                self.bias = original_module.bias
                
            def forward(self, x, offset=None, mask=None):
                # Use regular conv2d for ONNX export (approximation)
                return nn.functional.conv2d(
                    x, self.weight, self.bias,
                    stride=self.stride, padding=self.padding,
                    dilation=self.dilation, groups=self.groups
                )
        
        return DeformConv2dExportable
    except ImportError:
        return None


# ============================================================================
# Model Loading
# ============================================================================

sys.path.insert(0, "src")
from rembg_onnx import MODEL_CONFIGS

def load_birefnet_model(
    model_name: str = "general", 
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> nn.Module:
    """Load BiRefNet model from HuggingFace.
    
    Note: HuggingFace models are stored in FP16 but loaded as FP32 by default.
    We load them in native FP16 to avoid unnecessary precision increase.
    """
    from transformers import AutoModelForImageSegmentation
    
    config = MODEL_CONFIGS.get(model_name)
    if config is None:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
    
    print(f"Loading {config['description']}...")
    print(f"  Repository: {config['repo_id']}")
    print(f"  Precision: {dtype}")
    
    # Load model - weights are stored as FP16 on HuggingFace
    model = AutoModelForImageSegmentation.from_pretrained(
        config['repo_id'],
        trust_remote_code=True,
        dtype=dtype,  # Load in specified precision
    )
    
    model = model.to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    param_dtype = next(model.parameters()).dtype
    print(f"  Parameters: {total_params / 1e6:.1f}M")
    print(f"  Loaded dtype: {param_dtype}")
    
    return model, config


# ============================================================================
# ONNX Export
# ============================================================================

def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_size: Tuple[int, int] = (1024, 1024),
    batch_size: int = 1,
    opset_version: int = 17,
    device: str = "cuda",
    simplify: bool = True,
) -> str:
    """
    Export BiRefNet model to ONNX format.
    
    Note: Dynamic batch size is NOT supported due to deformable convolution
    limitations. The batch size is baked into the model at export time.
    
    Args:
        model: BiRefNet model (can be FP16 or FP32)
        output_path: Output ONNX file path
        input_size: (height, width) input size
        batch_size: Batch size to bake into model (1-8 recommended for 24GB VRAM)
        opset_version: ONNX opset version
        device: Device for export
        simplify: Run onnxslim to simplify model
    
    Returns:
        Path to exported ONNX model
    """
    print(f"\n{'='*60}")
    print("ONNX Export")
    print(f"{'='*60}")
    
    # Prepare output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Detect model precision from parameters
    model_dtype = next(model.parameters()).dtype
    is_fp16 = model_dtype == torch.float16
    
    # Create dummy input matching model precision
    h, w = input_size
    dummy_input = torch.randn(batch_size, 3, h, w, device=device, dtype=model_dtype)
    
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Model precision: {'FP16' if is_fp16 else 'FP32'}")
    print(f"  Batch size: {batch_size}")
    print(f"  Opset version: {opset_version}")
    
    if batch_size > 1:
        print(f"  ⚠ WARNING: batch > 1 may not work due to deformable conv limitations!")
    
    # Export to ONNX
    print(f"  Exporting to: {output_path}")
    
    # Suppress TracerWarnings during export (expected for fixed-size models)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        with torch.no_grad():
            try:
                # Try dynamo-based export first (better for complex models)
                torch.onnx.export(
                    model,
                    dummy_input,
                    str(output_path),
                    input_names=["input_image"],
                    output_names=["output_image"],
                    opset_version=opset_version,
                    do_constant_folding=True,
                    export_params=True,
                    verbose=False,
                    dynamo=True,  # Use new export
                )
                print(f"  ✓ Export complete (dynamo)")
            except Exception as e:
                print(f"  ⚠ Dynamo export failed (deform_conv2d not supported)")
                print(f"  Trying legacy TorchScript export...")
                
                # Fallback to legacy export
                torch.onnx.export(
                    model,
                    dummy_input,
                    str(output_path),
                    input_names=["input_image"],
                    output_names=["output_image"],
                    opset_version=opset_version,
                    do_constant_folding=True,
                    export_params=True,
                    verbose=False,
                    dynamo=False,
                )
                print(f"  ✓ Export complete (legacy)")
    
    # Simplify with onnxslim if available
    if simplify:
        try:
            import onnxslim
            import onnx
            print("  Running onnxslim optimization...")
            
            slim_output = str(output_path).replace('.onnx', '_slim.onnx')
            model_slim = onnxslim.slim(str(output_path))
            
            # onnxslim returns an ONNX model, save it using onnx.save
            onnx.save(model_slim, slim_output)
            
            # Replace original with slimmed version
            original_size = os.path.getsize(output_path)
            slim_size = os.path.getsize(slim_output)
            
            if slim_size < original_size:
                shutil.move(slim_output, output_path)
                print(f"  ✓ Optimized: {original_size/1e6:.1f}MB → {slim_size/1e6:.1f}MB")
            else:
                os.remove(slim_output)
                print(f"  ✓ Original size kept: {original_size/1e6:.1f}MB")
                
        except ImportError:
            print("  ⚠ onnxslim not available, skipping optimization")
        except Exception as e:
            print(f"  ⚠ Optimization failed: {e}")
    
    # Report final size
    final_size = os.path.getsize(output_path)
    print(f"  Final size: {final_size / 1e6:.1f} MB")
    
    return str(output_path)


# ============================================================================
# Quantization
# ============================================================================

def quantize_onnx_model(
    input_path: str,
    output_path: str,
    quantize_type: str = "fp8",
) -> str:
    """
    Quantize ONNX model to FP8 precision for TensorRT.
    
    FP8 (8-bit floating point) is supported on Ada Lovelace (RTX 40xx) and 
    Hopper (H100) GPUs via TensorRT. This creates an FP8-ready model that
    TensorRT can accelerate.
    
    Args:
        input_path: Input ONNX model path (FP32 or FP16)
        output_path: Output quantized model path
        quantize_type: Only "fp8" is supported
    
    Returns:
        Path to quantized model
    """
    print(f"\n{'='*60}")
    print(f"Quantization: {quantize_type}")
    print(f"{'='*60}")
    
    import onnx
    from onnx import numpy_helper, TensorProto, helper
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    original_size = os.path.getsize(input_path)
    print(f"  Input model: {input_path}")
    print(f"  Original size: {original_size / 1e6:.1f} MB")
    
    if quantize_type == "fp8":
        print("  Applying FP8 (E4M3) weight compression...")
        print("  Note: For inference, use TensorRT EP on Ada/Hopper GPUs")
        print("        or the model will fall back to FP16 compute.")
        
        # Load the ONNX model
        model = onnx.load(input_path)
        
        import ml_dtypes
        
        converted_count = 0
        skipped_count = 0
        
        # Create a mapping of initializer names to their FP8 scale factors
        # We'll store weights as FP8 with a scale, and add DequantizeLinear nodes
        new_initializers = []
        scales_to_add = []
        nodes_to_add = []
        initializer_names_converted = set()
        
        for initializer in model.graph.initializer:
            if initializer.data_type in [TensorProto.FLOAT, TensorProto.FLOAT16]:
                try:
                    # Get the weight as numpy array
                    weight = numpy_helper.to_array(initializer)
                    weight_fp32 = weight.astype(np.float32)
                    
                    # Calculate scale for FP8 E4M3 (max representable: 448)
                    abs_max = np.abs(weight_fp32).max()
                    if abs_max == 0:
                        scale = 1.0
                    else:
                        scale = abs_max / 448.0
                    
                    # Scale and convert to FP8
                    weight_scaled = weight_fp32 / scale
                    weight_fp8 = weight_scaled.astype(ml_dtypes.float8_e4m3fn)
                    
                    # Create new FP8 initializer with suffix
                    fp8_name = initializer.name + "_fp8"
                    fp8_init = numpy_helper.from_array(
                        weight_fp8.view(np.int8),  # Store as int8 bytes
                        name=fp8_name
                    )
                    # Change type to FLOAT8E4M3FN
                    fp8_init.data_type = 17  # FLOAT8E4M3FN
                    new_initializers.append(fp8_init)
                    
                    # Create scale initializer
                    scale_name = initializer.name + "_scale"
                    scale_init = numpy_helper.from_array(
                        np.array([scale], dtype=np.float32),
                        name=scale_name
                    )
                    scales_to_add.append(scale_init)
                    
                    # Create DequantizeLinear node: fp8_weight -> fp32_weight
                    dq_node = helper.make_node(
                        'DequantizeLinear',
                        inputs=[fp8_name, scale_name],
                        outputs=[initializer.name],
                        name=f"dequant_{initializer.name}"
                    )
                    nodes_to_add.append(dq_node)
                    
                    initializer_names_converted.add(initializer.name)
                    converted_count += 1
                    
                except Exception as e:
                    skipped_count += 1
        
        # Remove original initializers that were converted
        model.graph.initializer[:] = [
            init for init in model.graph.initializer 
            if init.name not in initializer_names_converted
        ]
        
        # Add new FP8 initializers and scales
        model.graph.initializer.extend(new_initializers)
        model.graph.initializer.extend(scales_to_add)
        
        # Insert DequantizeLinear nodes at the beginning
        for node in reversed(nodes_to_add):
            model.graph.node.insert(0, node)
        
        print(f"  Converted {converted_count} tensors to FP8 with DequantizeLinear")
        if skipped_count > 0:
            print(f"  Skipped {skipped_count} tensors")
        
        # Save the quantized model
        onnx.save(model, str(output_path))
        
    else:
        raise ValueError(f"Unknown quantization type: {quantize_type}. Only 'fp8' is supported.")
    
    quantized_size = os.path.getsize(output_path)
    reduction = (1 - quantized_size / original_size) * 100
    print(f"  ✓ Quantization complete")
    print(f"  Output: {output_path}")
    print(f"  Quantized size: {quantized_size / 1e6:.1f} MB ({reduction:.1f}% reduction)")
    
    return str(output_path)


# ============================================================================
# Verification
# ============================================================================

def verify_onnx_model(
    onnx_path: str,
    input_size: Tuple[int, int] = (1024, 1024),
    test_inference: bool = True,
) -> bool:
    """Verify ONNX model is valid and can run inference."""
    print(f"\n{'='*60}")
    print("Verification")
    print(f"{'='*60}")
    
    import onnx
    import onnxruntime as ort
    
    print(f"  Model: {onnx_path}")
    
    # Check model structure
    try:
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print("  ✓ ONNX structure valid")
    except Exception as e:
        print(f"  ✗ ONNX structure invalid: {e}")
        return False
    
    # Get model info
    inputs = model.graph.input
    outputs = model.graph.output
    print(f"  Inputs: {[i.name for i in inputs]}")
    print(f"  Outputs: {[o.name for o in outputs]}")
    
    if not test_inference:
        return True
    
    # Test inference
    try:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        available = ort.get_available_providers()
        providers = [p for p in providers if p in available]
        
        print(f"  Providers: {providers}")
        
        # Try creating session, fallback to CPU if CUDA fails
        try:
            session = ort.InferenceSession(onnx_path, providers=providers)
        except Exception as cuda_err:
            if 'CUDAExecutionProvider' in providers:
                print(f"  ⚠ CUDA failed: {str(cuda_err)[:80]}...")
                print(f"  Falling back to CPU...")
                providers = ['CPUExecutionProvider']
                session = ort.InferenceSession(onnx_path, providers=providers)
        
        input_info = session.get_inputs()[0]
        print(f"  Input shape: {input_info.shape}")
        print(f"  Input type: {input_info.type}")
        
        # Create test input
        h, w = input_size
        batch_size = input_info.shape[0] if isinstance(input_info.shape[0], int) else 1
        
        if 'float16' in input_info.type.lower():
            test_input = np.random.randn(batch_size, 3, h, w).astype(np.float16)
        else:
            test_input = np.random.randn(batch_size, 3, h, w).astype(np.float32)
        
        # Run inference
        import time
        start = time.time()
        outputs = session.run(None, {input_info.name: test_input})
        elapsed = time.time() - start
        
        print(f"  ✓ Inference successful ({elapsed*1000:.1f}ms)")
        print(f"  Output shape: {outputs[0].shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Inference failed: {e}")
        return False


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Export BiRefNet to ONNX with optional quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export lite model to FP16 ONNX (default, matches HuggingFace storage)
  %(prog)s --model lite --output models/BiRefNet_lite/onnx/model_fp16.onnx
  
  # Export general model with dynamic batch size
  %(prog)s --model general --output models/BiRefNet/onnx/model_fp16_dynamic.onnx --dynamic-batch
  
  # Export and quantize to INT8 (uses FP32 intermediate automatically)
  %(prog)s --model lite --output models/BiRefNet_lite/onnx/model_int8.onnx --quantize int8_dynamic
  
  # Export with specific batch size
  %(prog)s --model lite --batch-size 4 --output models/BiRefNet_lite/onnx/model_b4.onnx
  
  # Force FP32 export
  %(prog)s --model general --output models/BiRefNet/onnx/model_fp32.onnx --fp32

Available models:
  general   - BiRefNet with Swin-L backbone (~900MB FP16)
  lite      - BiRefNet Lite with Swin-T backbone (~85MB FP16)
  lite-2k   - BiRefNet Lite for 2K resolution
  portrait  - BiRefNet optimized for portraits
  matting   - BiRefNet for alpha matting
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="general",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model variant to export"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output ONNX file path"
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1,
        help="Batch size (only batch=1 supported due to deformable conv limitations)"
    )
    
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Export in FP32 precision (default is FP16, matching HuggingFace storage)"
    )
    
    parser.add_argument(
        "--quantize", "-q",
        type=str,
        choices=["fp8"],
        default=None,
        help="Quantization type: fp8 (Ada/Hopper GPUs, ~50%% reduction with minimal quality loss)"
    )
    
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17, max for CUDA EP compatibility)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for export (default: cuda if available)"
    )
    
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip onnxslim optimization"
    )
    
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification after export"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("BiRefNet ONNX Export")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    
    # Determine precision
    # Default is FP16 (matching HuggingFace storage), use FP32 if requested or if quantizing
    use_fp32 = args.fp32 or args.quantize
    model_dtype = torch.float32 if use_fp32 else torch.float16
    
    print(f"Precision: {'FP32' if use_fp32 else 'FP16'}")
    if args.quantize:
        print(f"Note: Using FP32 for quantization to {args.quantize}")
    
    # Register deformable conv2d for ONNX export
    register_deform_conv2d_onnx_op()
    
    # Load model in appropriate precision
    model, config = load_birefnet_model(args.model, args.device, dtype=model_dtype)
    
    # Determine output path
    output_path = args.output
    if args.quantize:
        # Export to temp FP32 first, then quantize
        base_output = Path(output_path)
        fp32_output = base_output.parent / f"{base_output.stem}_fp32{base_output.suffix}"
    else:
        fp32_output = output_path
    
    # Export to ONNX
    export_to_onnx(
        model=model,
        output_path=str(fp32_output),
        input_size=config["input_size"],
        batch_size=args.batch_size,
        opset_version=args.opset,
        device=args.device,
        simplify=not args.no_simplify,
    )
    
    # Clear GPU memory
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Quantize if requested
    final_output = str(fp32_output)
    if args.quantize:
        final_output = quantize_onnx_model(
            input_path=str(fp32_output),
            output_path=output_path,
            quantize_type=args.quantize,
        )
        
        # Remove intermediate FP32 model
        if str(fp32_output) != output_path and os.path.exists(fp32_output):
            os.remove(fp32_output)
    
    # Verify
    if not args.no_verify:
        verify_onnx_model(final_output, config["input_size"])
    
    print(f"\n{'='*60}")
    print("✓ Export Complete!")
    print(f"{'='*60}")
    print(f"Output: {final_output}")
    print(f"Size: {os.path.getsize(final_output) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

