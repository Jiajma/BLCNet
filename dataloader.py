import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
class NoisyDataset(Dataset):
    def __init__(self, image_dir, noise_type=None, noise_level=25, transform=None):
        self.image_dir = image_dir
        self.image_list = os.listdir(image_dir)
        self.transform = transform
        self.noise_type = noise_type
        self.noise_level = noise_level
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        if isinstance(self.noise_level, str) and 'to' in self.noise_level:
            low, high = map(int, self.noise_level.split('to'))
            noise_level = random.randint(low, high)
        else:
            noise_level = self.noise_level
        noisy_image = image
        return noisy_image, image, self.image_list[idx]
def get_dataloader(image_dir, noise_type=None, noise_level=25, batch_size=2, shuffle=True, num_workers=0, is_train=True):
    if is_train:
        transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),transforms.RandomVerticalFlip(p=0.5),transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.ToTensor()])
    dataset = NoisyDataset(image_dir=image_dir, noise_type=noise_type, noise_level=noise_level, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader