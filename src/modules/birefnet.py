import os
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from modules.utils import device

model_dict = {
    'ZhengPeng7/BiRefNet_HR': 2048,
    'ZhengPeng7/BiRefNet_lite': 1024,
    'ZhengPeng7/BiRefNet_lite-2K': 2048,
    'ZhengPeng7/BiRefNet': 1024
}

class BirefnetHandler:
    def __init__(self, model_name: str = 'ZhengPeng7/BiRefNet') -> None:
        """
        Initializes the BirefnetHandler with a pre-trained BiRefNet model and necessary transformations.
        """
        self.model_name = model_name
        self.model_path = os.path.join('models', self.model_name.split('/')[-1])
        self.model = AutoModelForImageSegmentation.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=self.model_path
        ).to(device)
        self.resolution = model_dict[self.model_name]
        self.tform_birefnet = transforms.Compose([
            transforms.Resize((self.resolution, self.resolution), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def process_imgs(self, images: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Apply segmentation masks to composite images using the provided segmentation model.

        Args:
            images (list[torch.Tensor]): A list of input image tensors, each with shape (C, H, W)

        Returns:
            list[torch.Tensor]: A list of segmentation mask tensors
        """
        out = []

        transformed_images = torch.stack([self.tform_birefnet(img) for img in images]).to(device)

        with torch.no_grad():
            mask_images = self.model(transformed_images)[-1].sigmoid().to(device)
        
        for image, mask in zip(images, mask_images):
            postprocess_mask = transforms.Compose([
                transforms.Resize(
                    size=(*image.shape[-2:],),
                    interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.Lambda(lambda x: torch.where(x < 1/255, 0.0, torch.where(x > 253/255, 1.0, x))),
                #transforms.Lambda(lambda x: x.clamp(min=0, max=1))
            ])
            mask = postprocess_mask(mask)
            image = torch.cat([image[:3], mask], dim=0)
            out.append(image)
            
        return out
