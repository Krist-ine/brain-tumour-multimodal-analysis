import torch
from torch.utils.data import DataLoader
from dataset import BrainTumorDataset
from models import create_resnet_model   # adjust depending on model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print(f"Using device: {DEVICE}")

    # Load test dataset
    test_dataset = BrainTumorDataset(root_dir="data/Testing")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Load model
    model = create_resnet_model(num_classes=len(test_dataset.class_to_idx)).to(DEVICE)
    model.load_state_dict(torch.load("resnet_model.pth", map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=list(idx_to_class.values())))

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    main()
