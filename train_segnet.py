import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

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

print("Loading dataset...")
X_train, Y_train = load_dataset("train")
X_val, Y_val = load_dataset("val")
print("✅ Loaded dataset:", X_train.shape, Y_train.shape)

print("Loading saved model...")
model = load_model('brain_tumor_segnet.h5')

# Continue training for 5 more epochs (if you previously did N epochs, set initial_epoch=N)
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    batch_size=8,
    epochs=10,  # Replace YOUR_PREVIOUS_EPOCHS with your completed epoch count
    initial_epoch=5  # e.g., if you trained 5 epochs before, set both to 5
)

model.save("brain_tumor_segnet.h5")
print("✅ Continued training complete! Model saved as brain_tumor_segnet.h5")
