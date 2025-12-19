from .bg import remove
from .birefnet_onnx import BiRefNetSessionONNX

MODEL_CONFIGS = {
    "general": {
        "repo_id": "ZhengPeng7/BiRefNet",
        "backbone": "swin_v1_l",
        "input_size": (1024, 1024),
        "description": "BiRefNet General - Swin-L backbone (~900MB)"
    },
    "general-hr": {
        "repo_id": "ZhengPeng7/BiRefNet_HR",
        "backbone": "swin_v1_l",
        "input_size": (2048, 2048),
        "description": "BiRefNet HR General - Swin-L backbone (~900MB)"
    },
    "lite": {
        "repo_id": "ZhengPeng7/BiRefNet_lite", 
        "backbone": "swin_v1_t",
        "input_size": (1024, 1024),
        "description": "BiRefNet Lite - Swin-T backbone (~200MB)"
    },
    "lite-2k": {
        "repo_id": "ZhengPeng7/BiRefNet_lite-2K",
        "backbone": "swin_v1_t", 
        "input_size": (1440, 2560),  # (H, W) - 2K landscape
        "description": "BiRefNet Lite 2K - Swin-T backbone, 2K resolution"
    },
    "portrait": {
        "repo_id": "ZhengPeng7/BiRefNet-portrait",
        "backbone": "swin_v1_l",
        "input_size": (1024, 1024),
        "description": "BiRefNet Portrait - optimized for portraits"
    },
    "matting": {
        "repo_id": "ZhengPeng7/BiRefNet-matting",
        "backbone": "swin_v1_l",
        "input_size": (1024, 1024),
        "description": "BiRefNet Matting - alpha matting variant"
    }
}