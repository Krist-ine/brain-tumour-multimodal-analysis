import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import random
from torchvision import transforms
import matplotlib.pyplot as plt

from dataset import BrainTumorDataset
from models import create_densenet   # Make sure this function returns DenseNet121/169 etc.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4

def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    for batch in dataloader:
        inputs = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return acc, all_labels, all_preds

def visualize_predictions(model, test_dataset, class_names):
    model.eval()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    with torch.no_grad():
        for i in range(8):
            img_info = test_dataset[i]
            img, label = img_info["image"], img_info["label"]
            img_input = img.unsqueeze(0).to(DEVICE)
            pred = torch.argmax(model(img_input)).item()

            img_show = img.permute(1, 2, 0).cpu().numpy()

            axes[i].imshow(img_show)
            axes[i].set_title(f"Pred: {class_names[pred]}\nTrue: {class_names[label]}")
            axes[i].axis("off")

    plt.tight_layout()
    plt.show()

def main():
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = BrainTumorDataset("data/Training", transform=transform)
    test_dataset = BrainTumorDataset("data/Testing", transform=transform)

    random.seed(42)
    train_indices = random.sample(range(len(train_dataset)), min(1000, len(train_dataset)))
    test_indices = random.sample(range(len(test_dataset)), min(300, len(test_dataset)))

    train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(Subset(test_dataset, test_indices), batch_size=BATCH_SIZE, shuffle=False)

    class_names = train_dataset.class_names

    model = create_densenet(num_classes=len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nTraining started...\n")
    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, train_loader, criterion, optimizer)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f}")

    print("\nEvaluating model...")
    acc, true_labels, pred_labels = evaluate(model, test_loader)
    print(f"\nAccuracy: {acc:.4f}\n")

    report = classification_report(true_labels, pred_labels, target_names=class_names)
    print(report)

    torch.save(model.state_dict(), "best_densenet_model.pth")
    print("\nModel saved as best_densenet_model.pth")

    print("\nVisualizing sample predictions...")
    visualize_predictions(model, test_dataset, class_names)

if __name__ == "__main__":
    main()
