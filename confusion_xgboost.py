import os
import numpy as np
import cv2
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------
# LOAD IMAGES THE SAME WAY YOU DID IN TRAINING
# ---------------------------------------------------
def load_images_as_array(root_dir):
    class_names = sorted(os.listdir(root_dir))
    X = []
    y = []

    for label in class_names:
        class_folder = os.path.join(root_dir, label)
        for img_name in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 128))
            X.append(img.flatten())
            y.append(label)

    return np.array(X), np.array(y), class_names


# ---------------------------------------------------
# PLOT CONFUSION MATRIX
# ---------------------------------------------------
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(6,5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()

    plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
    plt.yticks(np.arange(len(class_names)), class_names)

    thresh = cm.max() / 2
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            plt.text(j, i, cm[i, j],
                     ha="center", va="center",
                     color="white" if cm[i,j] > thresh else "black")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
def main():
    # Load Test Data
    X_test, y_test, class_names = load_images_as_array("data/Testing")

    # Encode labels (same as before)
    encoder = LabelEncoder()
    encoder.fit(class_names)
    y_test_enc = encoder.transform(y_test)

    # Load Model
    model = xgb.XGBClassifier()
    model.load_model("xgboost_model.json")
    print("Loaded xgboost_model.json successfully.\n")

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test_enc, y_pred)
    print(f"Accuracy: {acc:.3f}\n")

    print("Classification Report:")
    print(classification_report(y_test_enc, y_pred, target_names=class_names, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test_enc, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    plot_confusion_matrix(cm, class_names)

    # Save confusion matrix to CSV
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv("xgboost_confusion_matrix.csv")
    print("\nSaved: xgboost_confusion_matrix.csv")


if __name__ == "__main__":
    main()
