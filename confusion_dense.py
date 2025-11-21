import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import random
from torchvision import transforms

from dataset import BrainTumorDataset
from models import create_densenet   # Same create_densenet you used in training

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

def plot_confusion_matrix(true_labels, pred_labels, class_names):
    cm = confusion_matrix(true_labels, pred_labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True Label',
        xlabel='Predicted Label'
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Write numbers on each square
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.show()

def main():
    # Same transform used during training
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_dataset = BrainTumorDataset("data/Testing", transform=transform)
    class_names = test_dataset.class_names

    random.seed(42)
    test_indices = random.sample(range(len(test_dataset)), min(300, len(test_dataset)))
    test_loader = DataLoader(Subset(test_dataset, test_indices), batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    model = create_densenet(num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(torch.load("best_densenet_model.pth", map_location=DEVICE))
    model.eval()
    print("\nModel loaded successfully.")

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

    acc = accuracy_score(all_labels, all_preds)
    print(f"\nAccuracy: {acc:.4f}\n")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    print("\nConfusion Matrix:")
    plot_confusion_matrix(all_labels, all_preds, class_names)

if __name__ == "__main__":
    main()
