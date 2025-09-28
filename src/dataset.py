import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch_geometric.data import Data
from PIL import Image
import os
import numpy as np

class BrainTumorDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {"glioma": 0, "meningioma": 1, "notumor": 2, "pituitary": 3}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Load image paths and labels
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        np.random.seed(42)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Load the real MRI image and its label
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        image_tensor = self.transform(image)

        # 2. Generate synthetic tabular data (will be used later)
        signal = label * 0.25
        noise = np.random.randn(10)
        synthetic_tabular = torch.tensor(noise + signal, dtype=torch.float32)

        # 3. Generate synthetic graph data (will be used later)
        num_nodes = 15
        node_features = torch.rand((num_nodes, 8))
        node_features[0, 0] = float(label)
        edge_index = torch.randint(0, num_nodes, (2, 30), dtype=torch.long)
        synthetic_graph = Data(x=node_features, edge_index=edge_index)

        return {
            "image": image_tensor,
            "tabular": synthetic_tabular,
            "graph": synthetic_graph,
            "label": torch.tensor(label, dtype=torch.long)
        }