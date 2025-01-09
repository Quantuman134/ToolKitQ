import os
import wget
import torch
import numpy as np
import time
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
from rembg import remove
import cv2
from dataclasses import dataclass

def sam_init(device='cpu'):
    sam_checkpoint = os.path.join(os.path.dirname(__file__), "SAM_pth", "sam_vit_h.pth")
    if (os.path.isfile(sam_checkpoint) == False):
        wget.download("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", sam_checkpoint)
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=device)
    predictor = SamPredictor(sam)
    return predictor

def _sam_segment(predictor, input_image, *bbox_coords):
    bbox = np.array(bbox_coords)
    image = np.asarray(input_image)

    start_time = time.time()
    predictor.set_image(image)

    masks_bbox, scores_bbox, logits_bbox = predictor.predict(
        box=bbox,
        multimask_output=True
    )

    print(f"SAM Time: {time.time() - start_time:.3f}s")
    out_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    out_image[:, :, :3] = image
    out_image_bbox = out_image.copy()
    out_image_bbox[:, :, 3] = masks_bbox[-1].astype(np.uint8) * 255
    torch.cuda.empty_cache()
    return Image.fromarray(out_image_bbox, mode='RGBA'), masks_bbox[-1].astype(np.uint8) * 255

def _expand2square(pil_img, background_color):
    width, height = pil_img.size

    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

@dataclass
class SegmentOutput:
    image:Image
    mask_array:np.ndarray
    mask_image:Image
    mask_reverse_image:Image
    origin_size:tuple # the segment image might be expand and resized

def segment(predictor, input_image, segment=True, square_output=True, size:tuple=None, return_dict=False):
    '''
        input_image: PIL.Image
        mask: mask of input image (H, W), value range int[0, 255]
        output: PIL.Image with RGBA channel
    '''
    # RES = 1024
    # input_image.thumbnail([RES, RES], Image.Resampling.LANCZOS)
    width, height = input_image.size
    origin_size = (width, height)

    if segment:
        image_rem = input_image.convert('RGBA')
        image_nobg = remove(image_rem, alpha_matting=True)
        arr = np.asarray(image_nobg)[:,:,-1]
        x_nonzero = np.nonzero(arr.sum(axis=0))
        y_nonzero = np.nonzero(arr.sum(axis=1))
        x_min = int(x_nonzero[0].min())
        y_min = int(y_nonzero[0].min())
        x_max = int(x_nonzero[0].max())
        y_max = int(y_nonzero[0].max())
        input_image, mask = _sam_segment(predictor, input_image.convert('RGB'), x_min, y_min, x_max, y_max)
    if square_output:
        input_image = _expand2square(input_image, (127, 127, 127, 0))
    if size is not None:
        input_image = input_image.resize(size, Image.Resampling.LANCZOS)
    
    if return_dict:
        mask_image = np.stack([mask] * 3, axis=2)
        mask_reverse_image = 255 - mask_image
        mask_image = Image.fromarray(mask_image, mode='RGB')
        mask_reverse_image = Image.fromarray(mask_reverse_image, mode='RGB')
        return SegmentOutput(image=input_image, mask_array=mask, mask_image=mask_image, 
                             mask_reverse_image=mask_reverse_image, origin_size=origin_size)
    else:
        return input_image

if __name__ == '__main__':
    from ToolKitQ.PyTools.PytorchTools import torch_device_config

    device = torch_device_config()
    predictor = sam_init(device)
    image = Image.open("./ToolKitQ/ImgTools/sample/dog.jpg")
    image_after = segment(predictor, image, size=(320, 320))
    image_after.save("./ToolKitQ/ImgTools/sample/seg.png")
