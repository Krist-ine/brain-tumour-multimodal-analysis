import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

def load_images_as_array(root_dir):
    class_names = sorted(os.listdir(root_dir))  # Example: ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

    X = []
    y = []

    for label_index, class_name in enumerate(class_names):
        class_folder = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_folder):
            continue

        for img_name in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((224, 224))
                img = np.array(img) / 255.0
                img = img.flatten()  # Flatten to 1D
                X.append(img)
                y.append(label_index)
            except:
                continue

    X = np.array(X)
    y = np.array(y)
    return X, y, class_names

class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Get class names from folder names
        self.class_names = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}

        # Collect image paths + labels
        self.image_paths = []
        self.labels = []

        for cls_name in self.class_names:
            cls_folder = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_folder):
                self.image_paths.append(os.path.join(cls_folder, img_name))
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        # Apply transforms if given
        if self.transform:
            image = self.transform(image)

        return {"image": image, "label": label}
