import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
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

# ---------------- U-NET MODEL ----------------
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x

def unet():
    inputs = layers.Input((*IMG_SIZE, 3))

    c1 = conv_block(inputs, 64)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 128)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 256)
    p3 = layers.MaxPooling2D()(c3)

    c4 = conv_block(p3, 512)
    p4 = layers.MaxPooling2D()(c4)

    c5 = conv_block(p4, 1024)

    u6 = layers.Conv2DTranspose(512, 2, strides=2, padding="same")(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = conv_block(u6, 512)

    u7 = layers.Conv2DTranspose(256, 2, strides=2, padding="same")(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = conv_block(u7, 256)

    u8 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = conv_block(u8, 128)

    u9 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = conv_block(u9, 64)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c9)

    return Model(inputs, outputs)

model = unet()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=8, epochs=3)

model.save("brain_tumor_unet.h5")
print("✅ Training Complete! Model saved as brain_tumor_unet.h5")
