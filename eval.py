import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

IMG_SIZE = (256, 256)

def load_dataset(split):
    images_path = os.path.join("data", split, "images")
    masks_path = os.path.join("data", split, "masks")

    X, Y = [], []

    for img_file in os.listdir(images_path):
        image = load_img(os.path.join(images_path, img_file), target_size=IMG_SIZE)
        mask = load_img(os.path.join(masks_path, img_file), target_size=IMG_SIZE, color_mode="grayscale")

        X.append(img_to_array(image) / 255.0)
        mask_array = img_to_array(mask) / 255.0
        mask_array = (mask_array > 0.5).astype(np.float32)
        Y.append(mask_array)

    return np.array(X), np.array(Y)

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = (y_true_f * y_pred_f).sum()
    return (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)

def visualize_results(X, Y, Y_pred, idx=0):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(X[idx])
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(Y[idx].squeeze(), cmap='gray')
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(Y_pred[idx].squeeze() > 0.5, cmap='gray')
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    print("Loading dataset...")
    X_val, Y_val = load_dataset("val")

    print("Loading model...")
    model = load_model("brain_tumor_segnet.h5")  # Change to your model filename

    print("Predicting masks...")
    preds = model.predict(X_val)

    # Calculate Dice for each sample and average
    dice_scores = [dice_coef(Y_val[i], (preds[i] > 0.5).astype(np.float32)) for i in range(len(preds))]
    mean_dice = np.mean(dice_scores)
    print(f"Mean Dice Coefficient on Validation Set: {mean_dice:.4f}")

    # Visualize a few examples
    for i in range(3):
        visualize_results(X_val, Y_val, preds, idx=i)
