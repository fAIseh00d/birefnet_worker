import torch
from torchvision import transforms

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

tform_to_pil = transforms.ToPILImage()
tform_to_tensor = transforms.ToTensor()
