import glob
import numpy as np
import os
from PIL import Image, ImageFile
import random
import torch
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

class MultiPairsDataset(torch.utils.data.Dataset):
    
    def __init__(self, image_root_dir, sketch_root_dir, separator='_'):
        self.image_sketch_pairs = []
        
        image_name_to_sketch_paths = {}
        for class_path in glob.glob(os.path.join(sketch_root_dir, '**')):
            class_name = os.path.basename(class_path)
            for sketch_path in glob.iglob(os.path.join(class_path, '**')):
                image_name = os.path.basename(sketch_path)
                image_name = image_name.split(separator)
                if len(image_name) > 2:
                    image_name = separator.join(image_name[:-1])
                else:
                    image_name = image_name[0]
                if class_name+'/'+image_name not in image_name_to_sketch_paths:
                    image_name_to_sketch_paths[class_name+'/'+image_name] = []
                image_name_to_sketch_paths[class_name+'/'+image_name].append(sketch_path)
        
        for class_path in glob.glob(os.path.join(image_root_dir, '**')):
            class_name = os.path.basename(class_path)
            for image_path in glob.iglob(os.path.join(class_path, '**')):
                image_name = os.path.basename(image_path)
                image_name = image_name.split('.')
                if len(image_name) > 2:
                    image_name = '.'.join(image_name[:-1])
                else:
                    image_name = image_name[0]
                if class_name+'/'+image_name in image_name_to_sketch_paths:
                    for sketch_path in image_name_to_sketch_paths[class_name+'/'+image_name]:
                        self.image_sketch_pairs.append((image_path, sketch_path))
            
    def __len__(self):
        return len(self.image_sketch_pairs)
    
    def __getitem__(self, idx):
        image, sketch = self.image_sketch_pairs[idx]
        return self._parse_pair(image, sketch)
    
    def _get_pair_paths(self, idx):
        return self.image_sketch_pairs[idx]

    def _parse_pair(self, image, sketch):
        image = Image.open(image)
        image = image.convert('RGB')
        image = np.array(image, dtype=np.uint8)
        image = image.transpose(2, 0, 1)
        sketch = Image.open(sketch)
        sketch = sketch.convert('RGB')
        sketch = np.array(sketch, dtype=np.uint8)
        sketch = sketch.transpose(2, 0, 1)
        return torch.from_numpy(image), torch.from_numpy(sketch)


def pair_collate_fn(batch):
    pairs = []
    for sketch_vectors, image in batch:
        pairs.append((sketch_vectors, image))
    return pairs
