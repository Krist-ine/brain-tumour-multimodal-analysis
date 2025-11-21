import torch
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # ensures current folder is in path
from create_multimodal_dataset import ResNetFeatureExtractor

def run_tabnet_training_and_evaluation():
    MODEL_SAVE_PATH = "tabnet_brain_tumor_classifier.pth"
    CNN_MODEL_PATH = r"C:\Users\KRISTINE\projects\brain-tumour-multimodal-analysis\main\resnet_brain_tumor_classifier.pth"
    DATA_ROOT = r"C:\Users\KRISTINE\projects\brain-tumour-multimodal-analysis\data"
    NUM_CLASSES = 4
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Step 1: Load CNN feature extractor
    feature_extractor = ResNetFeatureExtractor(model_path=CNN_MODEL_PATH, device=device)

    # Step 2: Extract features
    print("Extracting features for TabNet training...")
    from dataset import BrainTumorDataset
    from torch.utils.data import DataLoader
    from torchvision import transforms

    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = BrainTumorDataset(root_dir=os.path.join(DATA_ROOT, "Training"), transform=image_transforms)
    test_dataset = BrainTumorDataset(root_dir=os.path.join(DATA_ROOT, "Testing"), transform=image_transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    X_train, y_train, _ = feature_extractor.extract_features_from_loader(train_loader, desc="Train Feature Extraction")
    X_test, y_test, _ = feature_extractor.extract_features_from_loader(test_loader, desc="Test Feature Extraction")

    # Step 3: Train TabNet
    print("\nStarting TabNet training...")
    tabnet = TabNetClassifier(
        input_dim=X_train.shape[1],
        output_dim=NUM_CLASSES,
        n_d=64, n_a=64,
        n_steps=5,
        gamma=1.5,
        n_independent=2, n_shared=2,
        lambda_sparse=1e-4,
        momentum=0.02,
        clip_value=2.0
    )

    tabnet.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_name=['valid'],
        eval_metric=['accuracy'],
        max_epochs=NUM_EPOCHS,
        patience=5,
        batch_size=BATCH_SIZE,
        virtual_batch_size=8,
        num_workers=0,
        drop_last=False,
        # disable automatic explanation
        compute_importance=False

    )

    # Save model
    print(f"\nTraining complete. Saving TabNet model to {MODEL_SAVE_PATH}")
    tabnet.save_model("tabnet_brain_tumor_classifier.pth")

    # Step 4: Evaluation
    preds = tabnet.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("\n--- TabNet Classification Report ---")
    print(classification_report(y_test, preds))
    print(f"\nOverall Accuracy: {acc * 100:.2f}%")

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - TabNet Brain Tumor Classification')
    plt.show()

if __name__ == "__main__":
    run_tabnet_training_and_evaluation()
