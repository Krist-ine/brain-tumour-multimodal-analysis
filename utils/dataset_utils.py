import os
import numpy as np
from PIL import Image

def load_images_as_array(root_dir, size=(64,64)):
    images = []
    labels = []
    class_names = sorted(os.listdir(root_dir))
    class_to_idx = {c:i for i,c in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(root_dir, class_name)
        for file in os.listdir(class_dir):
            img = Image.open(os.path.join(class_dir, file)).convert("L")  # grayscale
            img = img.resize(size)
            img_arr = np.array(img).flatten()  # flatten to 1D vector
            images.append(img_arr)
            labels.append(class_to_idx[class_name])
    return np.array(images), np.array(labels), class_names

X_train, y_train, class_names = load_images_as_array("data/Training")
X_test, y_test, _ = load_images_as_array("data/Testing")
