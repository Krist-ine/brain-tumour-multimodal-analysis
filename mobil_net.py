import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report
import pandas as pd
import random
from torchvision import transforms
import matplotlib.pyplot as plt

from dataset import BrainTumorDataset        # Your dataset class
from models import create_mobilenetv2        # Your model creator


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


def evaluate(model, dataloader, class_names):
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

    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    return report


def main():
    print(f"Using Device: {DEVICE}")

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = BrainTumorDataset("data/Training", transform=transform)
    test_dataset = BrainTumorDataset("data/Testing", transform=transform)

    # Optional: Subsample for speed
    train_indices = list(range(len(train_dataset)))
    test_indices = list(range(len(test_dataset)))
    random.shuffle(train_indices)
    random.shuffle(test_indices)
    train_subset = Subset(train_dataset, train_indices[:1000])
    test_subset = Subset(test_dataset, test_indices[:300])

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

    class_names = train_dataset.class_names

    model = create_mobilenetv2(num_classes=len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nStarting Training...\n")
    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, train_loader, criterion, optimizer)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}")

    print("\nEvaluating Test Set...")
    report = evaluate(model, test_loader, class_names)
    print(pd.DataFrame(report).transpose())
    print(f"\nAccuracy: {report['accuracy']:.4f}")

    # Save model
    torch.save(model.state_dict(), "mobilenet_brain_tumor.pth")
    print("\nModel saved as mobilenet_brain_tumor.pth")

    # ---- Show some predictions ----
    print("\nShowing Sample Predictions...")
    model.eval()
    batch = next(iter(test_loader))

    images = batch["image"].to(DEVICE)
    labels = batch["label"].to(DEVICE)

    with torch.no_grad():
        preds = torch.argmax(model(images), dim=1)

    images = images.cpu()
    preds = preds.cpu()
    labels = labels.cpu()

    plt.figure(figsize=(12,6))
    for i in range(8):
        img = images[i].permute(1,2,0)  # CHW → HWC
        img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])  # un-normalize
        img = img.clamp(0,1)

        plt.subplot(2,4,i+1)
        plt.imshow(img)
        plt.title(f"Pred: {class_names[preds[i]]}\nTrue: {class_names[labels[i]]}")
        plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()
