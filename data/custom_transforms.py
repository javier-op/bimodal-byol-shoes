import cairocffi as cairo
import cairosvg
from io import BytesIO
import math
import numpy as np
from PIL import Image
import random
import skimage.feature
import torch
import torchvision.transforms as T

class BatchTransform(object):
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, image_batch):
        transformed_image_batch = []
        for image in image_batch:
            transformed_image_batch.append(self.transform(image))
        return transformed_image_batch


class ListToTensor(object):
    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype
    
    def __call__(self, list_batch):
        return torch.stack(list_batch, axis=0).to(self.device, dtype=self.dtype)


class PadToSquare(object):
    def __init__(self, fill=0):
        self.fill = fill
        
    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            _, height, width = image.shape
        else:
            width, height = image.size
        l_pad, t_pad, r_pad, b_pad = 0, 0, 0, 0
        if height < width:
            t_pad = int((width-height)/2)
            b_pad = (width-height)-t_pad
        elif width < height:
            l_pad = int((height-width)/2)
            r_pad = (height-width)-l_pad
        image = T.functional.pad(image, (l_pad, t_pad, r_pad, b_pad), fill=self.fill)
        return image


class SelectFromTuple(object):
    def __init__(self, index):
        self.index = index
    
    def __call__(self, data_tuple):
        return data_tuple[self.index]

