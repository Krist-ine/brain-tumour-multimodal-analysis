import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from dataset import BrainTumorDataset
from models.resnet import ResNetClassifier


class ResNetFeatureExtractor:
    """
    Extract deep features from a trained ResNet model.
    """
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load trained ResNet
        self.model = ResNetClassifier(num_classes=4).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Use everything except final classification layer for features
        self.feature_extractor = nn.Sequential(*list(self.model.resnet.children())[:-1])
        print("ResNet Feature Extractor initialized.")

    def extract_features_from_loader(self, data_loader, desc="Extracting features"):
        all_features = []
        all_labels = []
        all_paths = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=desc):
                images = batch['image'].to(self.device)
                labels = batch['label']

                # Temporary workaround for missing 'path'
                paths = batch.get('path', [None]*len(labels))

                features = self._extract_single_batch(images)
                all_features.append(features.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_paths.extend(paths)

        features_array = np.vstack(all_features)
        labels_array = np.array(all_labels)
        return features_array, labels_array, all_paths

    def _extract_single_batch(self, images):
        """
        Extract 2048-dim features from ResNet before final FC layer.
        """
        x = self.feature_extractor(images)  # output shape: [B, 2048, 1, 1]
        x = torch.flatten(x, 1)             # [B, 2048]
        return x

    def compute_radiomics_features(self, features):
        """
        Compute statistical/radiomics features from embeddings.
        """
        radiomics = {}
        radiomics['mean_activation'] = np.mean(features, axis=1, keepdims=True)
        radiomics['std_activation'] = np.std(features, axis=1, keepdims=True)
        radiomics['max_activation'] = np.max(features, axis=1, keepdims=True)
        radiomics['min_activation'] = np.min(features, axis=1, keepdims=True)
        radiomics['median_activation'] = np.median(features, axis=1, keepdims=True)
        radiomics['percentile_25'] = np.percentile(features, 25, axis=1, keepdims=True)
        radiomics['percentile_75'] = np.percentile(features, 75, axis=1, keepdims=True)
        radiomics['iqr'] = radiomics['percentile_75'] - radiomics['percentile_25']
        radiomics['sparsity'] = np.sum(features > 0, axis=1, keepdims=True) / features.shape[1]
        radiomics['energy'] = np.sum(features ** 2, axis=1, keepdims=True)

        radiomics_array = np.hstack([v for v in radiomics.values()])
        return radiomics_array


def create_multimodal_dataset(resnet_model_path, data_root, clinical_csv_path, save_prefix, use_full_features=False):
    print(f"\n{'='*60}")
    print(f"Creating Multimodal {save_prefix.upper()} Dataset")
    print(f"{'='*60}\n")

    # Data loader
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = BrainTumorDataset(root_dir=data_root, transform=image_transforms)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    print(f"Found {len(dataset)} images")

    # Extract features
    extractor = ResNetFeatureExtractor(resnet_model_path)
    features, labels, image_paths = extractor.extract_features_from_loader(
        data_loader, desc=f"Extracting {save_prefix} features"
    )

    if use_full_features:
        imaging_features = features
        feature_prefix = 'resnet_feat_'
    else:
        imaging_features = extractor.compute_radiomics_features(features)
        feature_prefix = 'radiomics_'

    imaging_columns = [f"{feature_prefix}{i}" for i in range(imaging_features.shape[1])]
    imaging_df = pd.DataFrame(imaging_features, columns=imaging_columns)

    # Load clinical data
    clinical_df = pd.read_csv(clinical_csv_path)
    if len(imaging_df) != len(clinical_df):
        min_len = min(len(imaging_df), len(clinical_df))
        imaging_df = imaging_df.iloc[:min_len]
        clinical_df = clinical_df.iloc[:min_len]
        labels = labels[:min_len]

    clinical_features = clinical_df.drop('tumor_type', axis=1)
    multimodal_df = pd.concat([imaging_df, clinical_features], axis=1)
    multimodal_df['tumor_type'] = labels
    multimodal_df['image_path'] = image_paths  # currently None if dataset doesn't return path

    output_path = f"multimodal_{save_prefix}.csv"
    multimodal_df.to_csv(output_path, index=False)
    print(f"Multimodal dataset created: {len(multimodal_df)} samples, {len(multimodal_df.columns)-2} features (+label & path)")
    print(f"Saved to: {output_path}")

    return multimodal_df


def main():
    RESNET_MODEL_PATH = "resnet_brain_tumor_classifier.pth"
    DATA_ROOT = "../data"

    # Training set
    train_root = os.path.join(DATA_ROOT, "Training")
    train_clinical = os.path.join(DATA_ROOT, "synthetic_clinical_train.csv")
    create_multimodal_dataset(RESNET_MODEL_PATH, train_root, train_clinical, save_prefix='train', use_full_features=False)

    # Testing set
    test_root = os.path.join(DATA_ROOT, "Testing")
    test_clinical = os.path.join(DATA_ROOT, "synthetic_clinical_test.csv")
    create_multimodal_dataset(RESNET_MODEL_PATH, test_root, test_clinical, save_prefix='test', use_full_features=False)


if __name__ == "__main__":
    main()
