import os
import random
import numpy as np
import cv2
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# ---------------------------------------------------
# LOAD IMAGES SAME WAY AS TRAINING
# ---------------------------------------------------
def load_images(root_dir):
    class_names = sorted(os.listdir(root_dir))
    images = []
    labels = []

    for label in class_names:
        class_folder = os.path.join(root_dir, label)
        for img_name in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 128))
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels), class_names


def visualize_predictions(model, images, labels, encoder, class_names, num_samples=8):
    indices = random.sample(range(len(images)), num_samples)

    plt.figure(figsize=(15, 6))

    for i, idx in enumerate(indices):
        img = images[idx]
        true_label = labels[idx]
        
        img_flat = img.flatten().reshape(1, -1)
        pred_label_encoded = model.predict(img_flat)[0]
        pred_label = encoder.inverse_transform([pred_label_encoded])[0]

        plt.subplot(2, 4, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Pred: {pred_label}\nTrue: {true_label}", color="green" if pred_label==true_label else "red")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    # Load test data
    images, labels, class_names = load_images("data/Testing")

    # Encode labels
    encoder = LabelEncoder()
    encoder.fit(class_names)
    labels_encoded = encoder.transform(labels)

    # Load model
    model = xgb.XGBClassifier()
    model.load_model("xgboost_model.json")

    print("Model loaded. Displaying predictions...\n")

    visualize_predictions(model, images, labels, encoder, class_names)


if __name__ == "__main__":
    main()
