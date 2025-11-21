import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

IMG_SIZE = (256, 256)

def load_sample(img_path, mask_path):
    image = load_img(img_path, target_size=IMG_SIZE)
    mask = load_img(mask_path, target_size=IMG_SIZE, color_mode="grayscale")

    return img_to_array(image)/255.0, img_to_array(mask)/255.0

model = tf.keras.models.load_model("brain_tumor_unet.h5")

test_img = "data/val/images/img_84.png"   # <-- You can change to any val sample number
test_mask = "data/val/masks/img_84.png"

X, Y = load_sample(test_img, test_mask)
pred = model.predict(np.expand_dims(X, axis=0))[0]

pred = (pred > 0.5).astype(np.uint8)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.title("MRI"); plt.imshow(X)
plt.subplot(1,3,2); plt.title("Ground Truth"); plt.imshow(Y[:,:,0], cmap="gray")
plt.subplot(1,3,3); plt.title("Predicted Mask"); plt.imshow(pred[:,:,0], cmap="gray")
plt.show()
