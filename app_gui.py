import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

import tensorflow as tf
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.models as models

IMG_SIZE = (256, 256)
CLASS_NAMES = ["No Tumor", "Meningioma Tumor", "Pituitary Tumor", "Giloma Tumor"]

# ----------------------------------------------------
# Load Models
# ----------------------------------------------------
# Load Keras SegNet model
seg_model = tf.keras.models.load_model("brain_tumor_segnet.h5")

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define MobileNet architecture and load weights
mobilenet_model = models.mobilenet_v2(pretrained=False, num_classes=len(CLASS_NAMES))
state_dict = torch.load("mobilenet_brain_tumor.pth", map_location=device)
mobilenet_model.load_state_dict(state_dict)
mobilenet_model.to(device)
mobilenet_model.eval()

# Preprocessing for MobileNet classification
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])

# ----------------------------------------------------
# Helper: Predict function
# ----------------------------------------------------
def predict_combined(image_path):
    # Load and preprocess image for Keras SegNet
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, IMG_SIZE) / 255.0
    img_input = np.expand_dims(img_resized, axis=0)  # add batch dim

    # Predict segmentation mask
    mask_pred = seg_model.predict(img_input)[0]
    mask = (mask_pred > 0.5).astype(np.uint8) * 255  # binary mask

    # Extract tumor region and prepare for MobileNet classification
    tumor = img_resized * (mask[..., 0] / 255)[:, :, None]
    tumor_rgb = cv2.cvtColor((tumor * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    tumor_pil = Image.fromarray(tumor_rgb)

    tumor_tensor = transform(tumor_pil).unsqueeze(0).to(device)

    # Classification with MobileNet
    with torch.no_grad():
        outputs = mobilenet_model(tumor_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)
        prediction = CLASS_NAMES[pred_idx.item()]
        confidence_score = confidence.item()

    return img, mask[:, :, 0], prediction, confidence_score

# ----------------------------------------------------
# GUI Functions
# ----------------------------------------------------
def browse_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.png *.jpeg *.bmp")]
    )
    if file_path:
        result_label.config(text="Processing...")
        root.after(100, lambda: show_result(file_path))

def show_result(image_path):
    img, mask, prediction, confidence = predict_combined(image_path)

    img_display = cv2.resize(img, (300, 300))
    mask_display = cv2.resize(mask, (300, 300))

    img_display = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)))
    mask_display = ImageTk.PhotoImage(Image.fromarray(mask_display))

    original_panel.config(image=img_display)
    original_panel.image = img_display

    mask_panel.config(image=mask_display)
    mask_panel.image = mask_display

    result_label.config(text=f"Prediction: {prediction}  |  Confidence: {confidence:.2f}")

# ----------------------------------------------------
# GUI Layout
# ----------------------------------------------------
root = tk.Tk()
root.title("Brain Tumor Segmentation + Classification")
root.geometry("700x550")
root.resizable(False, False)

btn = tk.Button(root, text="Select MRI Image", font=("Arial", 14), command=browse_image)
btn.pack(pady=10)

frame = tk.Frame(root)
frame.pack()

original_panel = tk.Label(frame)
original_panel.grid(row=0, column=0, padx=10)

mask_panel = tk.Label(frame)
mask_panel.grid(row=0, column=1, padx=10)

result_label = tk.Label(root, text="No image selected", font=("Arial", 16))
result_label.pack(pady=20)

root.mainloop()
