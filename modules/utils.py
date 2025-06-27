from typing import Union, Tuple
import numpy as np
import torch
from torchvision import transforms

MODELS_DIR = 'models'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


tform_to_pil = transforms.ToPILImage()
tform_to_tensor = transforms.ToTensor()


tform_np_to_torch = transforms.Lambda(lambda x: torch.from_numpy(
    np.array(x[..., ::-1].transpose(2, 0, 1),   # Transpose to (C, H, W) and convert BGR → RGB
             dtype=np.float32                   # Cast to float32
             ) / 255                            # Normalize to [0, 1]
))

tform_torch_to_np_BGR = transforms.Lambda(lambda x: np.array(
    (x.detach().cpu().numpy() * 255).clip(0, 255)   # Convert to uint8 range
    .astype(np.uint8)                               # Cast to uint8
    .transpose(1, 2, 0)[..., ::-1]                  # Transpose to (H, W, C) and convert RGB → BGR
))

tform_torch_to_np_RGB = transforms.Lambda(lambda x: np.array(
    (x.detach().cpu().numpy() * 255).clip(0, 255)   # Convert to uint8 range
    .astype(np.uint8)                               # Cast to uint8
    .transpose(1, 2, 0)                             # Transpose to (H, W, C)
))
