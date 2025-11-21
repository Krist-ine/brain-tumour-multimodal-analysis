# src/dataset.py (Revised)
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd

class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform if transform else self._default_transforms()
        self.image_paths = []
        self.labels = []
        self.tabular_data = [] 

        self.class_to_idx = {}
        self.idx_to_class = {}

        self._load_data()


    def _default_transforms(self):
        """
        Defines the default image transformations.
        """
        return transforms.Compose([
            transforms.Resize((224, 224)), # Example resize, adjust as needed
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet stats
        ])

    def _load_data(self):
        """
        Loads image paths, labels, and tabular data from the dataset directory.
        """
        class_names = sorted([d for d in os.listdir(self.root_dir)
                              if os.path.isdir(os.path.join(self.root_dir, d))])

        if not class_names:
            raise RuntimeError(f"Found 0 subdirectories in {self.root_dir}. Check your dataset path and structure.")

        for i, class_name in enumerate(class_names):
            self.class_to_idx[class_name] = i
            self.idx_to_class[i] = class_name
        
        print(f"Class to index mapping: {self.class_to_idx}") # For debugging

        for class_name in class_names:
            class_path = os.path.join(self.root_dir, class_name)
            
            for filename in os.listdir(class_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    img_path = os.path.join(class_path, filename)
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[class_name]) 

                    dummy_tabular_feature = torch.rand(10)
                    self.tabular_data.append(dummy_tabular_feature)



    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB") # Ensure 3 channels

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        tabular = self.tabular_data[idx] # Already a tensor

        sample = {
            'image': image,
            'label': label,
            'tabular': tabular,
        }
        return sample

