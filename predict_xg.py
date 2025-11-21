import xgboost as xgb
import cv2
import numpy as np
import os

# Load trained model
model = xgb.XGBClassifier()
model.load_model("xgboost_model.json")

# Class names in SAME order as training
class_names = sorted(os.listdir("data/Training"))

def predict_image(image_path):
    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize EXACT same as training
    img = cv2.resize(img, (128, 128))

    # Flatten to 1D
    img = img.flatten().astype(float)

    # Convert to 2D array for prediction
    img = img.reshape(1, -1)

    # Predict class index
    pred_class_index = model.predict(img)[0]

    # Map index to class name
    predicted_label = class_names[pred_class_index]
    return predicted_label


# --------------------------
# TEST WITH ONE IMAGE
# --------------------------

IMAGE_PATH = "data\\Testing\\pituitary_tumor\\image(2).jpg"   # <-- change to any image you want
result = predict_image(IMAGE_PATH)
print("\nPredicted Output:", result)
