# src/main.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataloader import default_collate # <--- NEW IMPORT
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from dataset import BrainTumorDataset
from models import ImageCNN

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-4

# --- NEW HELPER FUNCTION ---
# This function filters out the graph data, which the standard DataLoader can't handle.
def custom_collate(batch):
    # Create a new list of samples, but without the 'graph' key
    filtered_batch = []
    for item in batch:
        filtered_item = {k: v for k, v in item.items() if k != 'graph'}
        filtered_batch.append(filtered_item)
    
    # Now, use the default collate function on the filtered batch
    return default_collate(filtered_batch)
# -------------------------

def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        
        optimizer.zero_grad()
        outputs, _ = model(images)
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
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            outputs, _ = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return accuracy_score(all_labels, all_preds)

def main():
    print(f"Using device: {DEVICE}")

    # --- Load Data ---
    train_dataset = BrainTumorDataset(root_dir="data/Training")
    test_dataset = BrainTumorDataset(root_dir="data/Testing")
    train_subset = Subset(train_dataset, range(1000))
    test_subset = Subset(test_dataset, range(300))

    # --- UPDATED DATALOADERS ---
    # We now pass our custom_collate function to the DataLoaders
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)
    # ---------------------------
    
    # --- 1. Train & Evaluate the Image CNN ---
    print("\n--- Starting CNN Model Training ---")
    cnn_model = ImageCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss = train_one_epoch(cnn_model, train_loader, criterion, optimizer)
        print(f"Average Training Loss: {train_loss:.4f}")

    print("\n--- Evaluating CNN Model ---")
    test_accuracy = evaluate(cnn_model, test_loader)
    
    print(f"\n--- FINAL CNN TEST ACCURACY: {test_accuracy:.4f} ---")

if __name__ == "__main__":
    main()