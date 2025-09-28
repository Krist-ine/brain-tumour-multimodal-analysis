# src/main.py
import torch
import torch.nn as nn
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from tqdm import tqdm
import random
import numpy as np

# Updated imports
from dataset import BrainTumorDataset
from models import ImageCNN, create_tabnet_model # <-- Changed from TabularMLP

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-4

# NOTE: The TabNet library uses its own data loaders, so our custom collate is only for the PyTorch models
def custom_collate(batch):
    filtered_batch = [{k: v for k, v in item.items() if k != 'graph'} for item in batch]
    return default_collate(filtered_batch)

# This train function is now only for the CNN (and later, GNN and Fusion)
def train_one_epoch_pytorch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc=f"Training (CNN)"):
        inputs = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# This evaluate function is also only for the PyTorch models
def evaluate_pytorch(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating (CNN)"):
            inputs = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            outputs, _ = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    class_names = list(dataloader.dataset.dataset.class_to_idx.keys())
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    return report

def main():
    print(f"Using device: {DEVICE}")

    # --- Load Data ---
    train_dataset = BrainTumorDataset(root_dir="data/Training")
    test_dataset = BrainTumorDataset(root_dir="data/Testing")
    train_indices = list(range(len(train_dataset)))
    test_indices = list(range(len(test_dataset)))
    random.seed(42)
    random.shuffle(train_indices)
    random.shuffle(test_indices)
    train_subset_indices = train_indices[:1000]
    test_subset_indices = test_indices[:300]
    train_subset = Subset(train_dataset, train_subset_indices)
    test_subset = Subset(test_dataset, test_subset_indices)

    # DataLoader for PyTorch models (CNN)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)
    
    results = {}
    
    # --- 1. Train & Evaluate the Image CNN ---
    print("\n--- Starting CNN Model Training ---")
    cnn_model = ImageCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss = train_one_epoch_pytorch(cnn_model, train_loader, criterion, optimizer)
        print(f"Average Training Loss: {train_loss:.4f}")

    print("\n--- Evaluating CNN Model ---")
    cnn_report = evaluate_pytorch(cnn_model, test_loader)
    results['CNN'] = cnn_report['accuracy']
    print(pd.DataFrame(cnn_report).transpose())
    
    # --- 2. Train & Evaluate TabNet ---
    print("\n\n--- Starting TabNet Model Training ---")
    
    # Prepare data in NumPy format, which TabNet expects
    X_train = np.array([train_subset[i]['tabular'].numpy() for i in range(len(train_subset))])
    y_train = np.array([train_subset[i]['label'].numpy() for i in range(len(train_subset))])
    X_test = np.array([test_subset[i]['tabular'].numpy() for i in range(len(test_subset))])
    y_test = np.array([test_subset[i]['label'].numpy() for i in range(len(test_subset))])

    tabnet_model = create_tabnet_model()
    
    # TabNet has its own .fit() method which is highly optimized
    tabnet_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        patience=5, # Stop training if the validation score doesn't improve for 5 epochs
        max_epochs=50,
        batch_size=128
    )

    print("\n--- Evaluating TabNet Model ---")
    # Use the .predict() method
    tabnet_preds = tabnet_model.predict(X_test)
    class_names = list(train_dataset.class_to_idx.keys())
    tabnet_report = classification_report(y_test, tabnet_preds, target_names=class_names, output_dict=True, zero_division=0)
    results['TabNet'] = tabnet_report['accuracy']
    print(pd.DataFrame(tabnet_report).transpose())


    # --- FINAL RESULTS TABLE ---
    print("\n\n--- FINAL RESULTS (Summary) ---")
    summary_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Test Accuracy': list(results.values())
    })
    print(summary_df)

if __name__ == "__main__":
    main()