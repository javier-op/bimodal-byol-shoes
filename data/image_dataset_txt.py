import numpy as np
from PIL import Image
import torch
from torchvision import transforms

class ImageDatasetFromText(torch.utils.data.Dataset):
    
    def __init__(self, root_dir, channels, separator, transform=None):
        assert channels == 1 or channels == 3
        self.image_data = []
        self.label_data = []
        with open(root_dir, 'r') as text_file:
            filenames = text_file.readlines()
            for filename in filenames:
                filename = filename.rstrip('\n')
                if not filename or filename == '':
                    continue
                filename = filename.split(separator)
                if len(filename) > 2:
                    image_path, label = separator.join(filename[:-1]), filename[-1]
                else:
                    image_path, label = filename
                self.image_data.append(image_path)
                self.label_data.append(int(label))
        self.image_data = np.array(self.image_data, dtype=object)
        self.label_data = np.array(self.label_data, dtype=np.int32)
        self.channels = channels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_data)
    
    def __getitem__(self, idx):
        selected_images = self.image_data[idx]
        selected_labels = self.label_data[idx]
        if isinstance(selected_images, str):
            return self._parse_image_label(selected_images, selected_labels)
        images, labels = [], []
        for image_path, label in zip(selected_images, selected_labels):
            image, label = self._parse_image_label(image_path, label)
            images.append(image)
            labels.append(label)
        return torch.stack(images), torch.stack(labels)
    
    def get_path(self, idx):
        return self.image_data[idx]
    
    def number_of_classes(self):
        return np.max(self.label_data)+1

    def _parse_image_label(self, image_path, label):
        as_gray = True if self.channels == 1 else False
        image = Image.open(image_path)
        if self.channels == 1:
            image = image.convert('L')
        else:
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        image = transforms.functional.pil_to_tensor(image)
        label = np.array(label, dtype=np.int32)
        label = torch.from_numpy(label)
        return image.to('cpu', dtype=torch.float), label
