from typing import Union
import os
import random

import torch
from torchvision import transforms
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

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

def img_type_convert(img:Union[np.ndarray, PIL.Image.Image], dtype=None, device=None):
    '''    
    Support:
    Image.Image --> torch.Tensor    RGB or RGBA --> [1, C, H, W]  range: [0, 1]
    np.ndarray --> torch.Tensor    [H, W, C] --> [1, C, H, W]  range: same with input
    '''
    
    flag_correct_input_type = True
    
    if type(img) == PIL.Image.Image:
        if dtype is None:
            dtype = torch.float32
        if device is None:
            device = 'cpu'
        output = transforms.ToTensor()(img).unsqueeze(0).type(dtype).to(device)
    elif type(img) == np.ndarray:
        if dtype is None:
            dtype = torch.float32
        if device is None:
            device = 'cpu'
        output = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).type(dtype).to(device)
    else:
        flag_correct_input_type = False    
    
    assert flag_correct_input_type, f'input type {type(img)} is not supported'
    
    return output
