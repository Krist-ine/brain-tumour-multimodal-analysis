import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report
import pandas as pd
import random
from torchvision import transforms
from dataset import BrainTumorDataset
from models import create_swin_transformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4

def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    for batch in dataloader:
        inputs = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)  # Shape might be [B, C, H, W]
        # Adaptive average pool to [B, C, 1, 1]
        pooled = F.adaptive_avg_pool2d(outputs, 1)
        # Flatten to [B, C]
        logits = pooled.view(pooled.size(0), -1)
        loss = criterion(logits, labels)
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
            pooled = F.adaptive_avg_pool2d(outputs, 1)
            logits = pooled.view(pooled.size(0), -1)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    return report

def main():
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = BrainTumorDataset("data/Training", transform=transform)
    test_dataset = BrainTumorDataset("data/Testing", transform=transform)

    train_indices = list(range(len(train_dataset)))
    test_indices = list(range(len(test_dataset)))
    random.seed(42)
    random.shuffle(train_indices)
    random.shuffle(test_indices)
    train_subset = Subset(train_dataset, train_indices[:1000])
    test_subset = Subset(test_dataset, test_indices[:300])

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

    class_names = train_dataset.class_names

    model = create_swin_transformer(num_classes=len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Sanity check output shape on dummy input
    dummy_input = torch.randn(2, 3, 224, 224).to(DEVICE)
    dummy_output = model(dummy_input)
    print(f"Output shape on dummy input before pooling: {dummy_output.shape}")

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss:.4f}")

    print("\nEvaluating on test set...")
    report = evaluate(model, test_loader, class_names)
    print(pd.DataFrame(report).transpose())
    print(f"\nOverall Accuracy: {report['accuracy']:.4f}")

if __name__ == "__main__":
    main()
