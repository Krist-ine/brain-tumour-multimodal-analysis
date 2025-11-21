import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

IMG_SIZE = (256, 256)

def load_mask(mask_path):
    mask = load_img(mask_path, target_size=IMG_SIZE, color_mode="grayscale")
    mask = img_to_array(mask) / 255.0
    return (mask > 0.5).astype(np.uint8)

def load_image(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    img = img_to_array(img) / 255.0
    return img

# ----------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------
model = tf.keras.models.load_model("brain_tumor_segnet.h5")

# ----------------------------------------------------
# PATHS
# ----------------------------------------------------
test_images_path = "data/val/images"
test_masks_path = "data/val/masks"

y_true_all = []
y_pred_all = []

print("Processing test dataset...")

for img_name in os.listdir(test_images_path):
    img_path = os.path.join(test_images_path, img_name)
    mask_path = os.path.join(test_masks_path, img_name)

    if not os.path.exists(mask_path):
        continue

    img = load_image(img_path)
    mask = load_mask(mask_path)

    pred = model.predict(np.expand_dims(img, axis=0))[0]
    pred = (pred > 0.5).astype(np.uint8)

    y_true_all.extend(mask.flatten())
    y_pred_all.extend(pred.flatten())

# ----------------------------------------------------
# CONFUSION MATRIX
# ----------------------------------------------------
cm = confusion_matrix(y_true_all, y_pred_all)
TN, FP, FN, TP = cm.ravel()

print("\nConfusion Matrix:")
print(cm)

# ----------------------------------------------------
# METRICS
# ----------------------------------------------------
accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-7)
precision = TP / (TP + FP + 1e-7)
recall = TP / (TP + FN + 1e-7)
dice = (2 * TP) / (2 * TP + FP + FN + 1e-7)
iou = TP / (TP + FP + FN + 1e-7)

print("\nMetrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"Dice Score:{dice:.4f}")
print(f"IoU Score: {iou:.4f}")

# ----------------------------------------------------
# VISUALIZE CONFUSION MATRIX
# ----------------------------------------------------
plt.figure(figsize=(5,4))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix (Pixel-wise)")
plt.colorbar()
plt.xticks([0,1], ["Background","Tumor"])
plt.yticks([0,1], ["Background","Tumor"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
