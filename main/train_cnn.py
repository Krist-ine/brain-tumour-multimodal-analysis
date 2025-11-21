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
from models.cnn import SimpleCNN

def run_cnn_training_and_evaluation():

    DATA_ROOT = r'C:\Users\KRISTINE\projects\brain-tumour-multimodal-analysis\data'
    MODEL_SAVE_PATH = 'cnn_brain_tumor_classifier.pth' 
    NUM_CLASSES = 4 
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loading
    train_root_dir = os.path.join(DATA_ROOT, 'Training')
    test_root_dir = os.path.join(DATA_ROOT, 'Testing')

    # image transformations
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Loading training data for CNN...")
    train_dataset = BrainTumorDataset(root_dir=train_root_dir, transform=image_transforms)
    print(f"Found {len(train_dataset)} training images.")
    print("Loading testing data for CNN...")
    test_dataset = BrainTumorDataset(root_dir=test_root_dir, transform=image_transforms)
    print(f"Found {len(test_dataset)} testing images.")
    print("----------------------------------------------")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    class_names = [train_dataset.idx_to_class[i] for i in sorted(train_dataset.idx_to_class.keys())]

    model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training 
    print("\nStarting CNN training...")
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

            # Update the progress bar dynamically
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{(correct_predictions/total_samples):.4f}'
            })

        epoch_loss = running_loss / len(train_dataset)
        epoch_accuracy = correct_predictions / total_samples
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    print("\nTraining complete. Saving CNN model...")
    # Model will be saved in the 'main/' directory
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    print("----------------------------------------------")


    # --- Evaluation ---
    print("\nStarting CNN evaluation...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in test_loader:
            images = data['image'].to(device)
            labels = data['label'].to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("----------------------------------------------")





    print("\n--- CNN Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    print("\n--- CNN Confusion Matrix ---")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for CNN Brain Tumor Classification')
    plt.show()




    # Display Sample Predictions 
    print("\n--- CNN Sample Test Predictions ---")
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('Sample Test Predictions (CNN)', fontsize=16)

    num_samples_to_show = 8
    sample_loader = DataLoader(test_dataset, batch_size=num_samples_to_show, shuffle=True)
    data_iter = iter(sample_loader)
    sample_data = next(data_iter)

    sample_images = sample_data['image'].to(device)
    sample_labels = sample_data['label'].to(device)

    with torch.no_grad():
        sample_outputs = model(sample_images)
        _, sample_predicted = torch.max(sample_outputs.data, 1)

    for i in range(num_samples_to_show):
        ax = fig.add_subplot(2, 4, i + 1)
        # Un-normalize the image for display
        img = sample_images[i].cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        ax.imshow(img)
        true_label = class_names[sample_labels[i].item()]
        pred_label = class_names[sample_predicted[i].item()]
        ax.set_title(f"Pred: {pred_label}\nTrue: {true_label}")
        ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    run_cnn_training_and_evaluation()