import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm

from .dataset import BrainTumorDataset
from models.resnet import ResNetClassifier

def run_resnet_training_and_evaluation():
    DATA_ROOT = r'C:\Users\KRISTINE\projects\brain-tumour-multimodal-analysis\data'
    MODEL_SAVE_PATH = 'resnet_brain_tumor_classifier.pth'
    NUM_CLASSES = 4
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loading
    train_root_dir = os.path.join(DATA_ROOT, 'Training')
    test_root_dir = os.path.join(DATA_ROOT, 'Testing')

    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Loading training data for ResNet...")
    train_dataset = BrainTumorDataset(root_dir=train_root_dir, transform=image_transforms)
    print(f"Found {len(train_dataset)} training images.")
    print("Loading testing data for ResNet...")
    test_dataset = BrainTumorDataset(root_dir=test_root_dir, transform=image_transforms)
    print(f"Found {len(test_dataset)} testing images.")
    print("----------------------------------------------")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    class_names = [train_dataset.idx_to_class[i] for i in sorted(train_dataset.idx_to_class.keys())]

    # Model, loss, optimizer
    model = ResNetClassifier(num_classes=NUM_CLASSES, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("\nStarting ResNet training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        for data in progress_bar:
            images = data['image'].to(device)
            labels = data['label'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{(correct_predictions / total_samples):.4f}'
            })

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct_predictions / total_samples
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    # Save model
    print("\nTraining complete. Saving ResNet model...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    print("----------------------------------------------")

    # Evaluation
    print("\nStarting ResNet evaluation...")
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data in test_loader:
            images = data['image'].to(device)
            labels = data['label'].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n--- ResNet Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    print("\n--- ResNet Confusion Matrix ---")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for ResNet Brain Tumor Classification')
    plt.show()

if __name__ == "__main__":
    run_resnet_training_and_evaluation()
