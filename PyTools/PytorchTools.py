import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import os

def torch_device_config(index:int=0) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(index))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return device

# Import and Export images to tensors
# the tensor size is [N, D, H, W], the range of value is within [0, 1]
def import_image_tensor(img_dir, size=None, dtype:torch.dtype=torch.float32, device='cpu'):
    '''
    size: (H, W), e.g. size=(768, 768)
    '''
    img = Image.open(img_dir)
    if size is not None:
        img = img.resize(size)
    img_tensor = transforms.ToTensor()(img).type(dtype)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    return img_tensor

def export_image_tensor(img_tensor: torch.Tensor, img_dir):
    img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
    img = img_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    img = np.ascontiguousarray(img)
    plt.imsave(img_dir, img)

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)