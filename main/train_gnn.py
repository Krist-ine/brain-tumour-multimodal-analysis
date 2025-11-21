import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import os, sys
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "models"))

from gnn import GNNClassifier

# Example dataset loader
def load_graph_dataset(data_csv):
    """
    Convert CSV features into PyG graph dataset
    Assumes: columns for node features + 'tumor_type' as label
    This is a placeholder. You should replace with real connectome construction.
    """
    df = pd.read_csv(data_csv)
    graphs = []
    labels = df['tumor_type'].values
    node_features = df.drop('tumor_type', axis=1).values

    # Create one graph per patient (row), here fully connected dummy edges
    num_nodes = node_features.shape[1]  # example: 10 nodes
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().contiguous()
    
    for i in range(len(df)):
        x = torch.tensor(node_features[i], dtype=torch.float).unsqueeze(-1)  # shape [num_nodes, features]
        y = torch.tensor(labels[i], dtype=torch.long).unsqueeze(0)            # graph-level label
        data = Data(x=x, edge_index=edge_index, y=y)
        graphs.append(data)
    return graphs


def run_gnn_training_and_evaluation():
    DATA_ROOT = r"C:\Users\KRISTINE\projects\brain-tumour-multimodal-analysis\main\data"
    TRAIN_CSV = os.path.join(DATA_ROOT, "synthetic_clinical_train.csv")
    TEST_CSV = os.path.join(DATA_ROOT, "synthetic_clinical_test.csv")
    MODEL_SAVE_PATH = "gnn_brain_tumor_classifier.pth"
    NUM_CLASSES = 4
    BATCH_SIZE = 16
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load graphs
    train_graphs = load_graph_dataset(TRAIN_CSV)
    test_graphs = load_graph_dataset(TEST_CSV)
    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)

    # Infer input feature dimension
    in_features = train_graphs[0].x.shape[1]
    model = GNNClassifier(in_features=in_features, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nStarting GNN training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0
        all_preds, all_labels = [], []

        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.num_graphs
            preds = out.argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(data.y.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1} - Loss: {running_loss/len(train_graphs):.4f}, Accuracy: {acc:.4f}")

    # Save model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"GNN model saved to {MODEL_SAVE_PATH}")

    # Evaluation
    print("\nEvaluating on test set...")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            preds = out.argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(data.y.cpu().numpy())

    print("\n--- GNN Classification Report ---")
    print(classification_report(all_labels, all_preds))
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("GNN Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    run_gnn_training_and_evaluation()
import pandas as pd

df = pd.read_csv("data/synthetic_clinical_train.csv")
print(df['tumor_type'].value_counts())
