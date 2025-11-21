import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import random
from torchvision import transforms
from dataset import BrainTumorDataset
from models import create_resnext_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 5
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
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    return report, all_labels, all_preds

def main():
    # Transforms suitable for ResNeXt/ResNet
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = BrainTumorDataset(root_dir="data/Training", transform=data_transforms)
    test_dataset = BrainTumorDataset(root_dir="data/Testing", transform=data_transforms)

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
    model = create_resnext_model(num_classes=len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        print(f"Epoch {epoch+1}/{EPOCHS} - Training Loss: {train_loss:.4f}")

    print("\nEvaluating model...")
    report, all_labels, all_preds = evaluate(model, test_loader, class_names)
    print(pd.DataFrame(report).transpose())

    # Print overall accuracy like XGBoost style
    accuracy = report['accuracy']
    print(f"\nOverall Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
