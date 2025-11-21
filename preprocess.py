import os
import cv2
import glob
from tqdm import tqdm

# Path to your downloaded dataset
DATASET_PATH = r"C:\Users\Chidam\Downloads\brain\lgg-mri-segmentation\kaggle_3m"

# Output folders
IMG_OUT = "data/images"
MASK_OUT = "data/masks"

os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(MASK_OUT, exist_ok=True)

# Desired image size
IMG_SIZE = 128

# Loop through each patient folder
patient_folders = glob.glob(os.path.join(DATASET_PATH, "*"))

count = 0

for patient in patient_folders:
    image_files = sorted(glob.glob(os.path.join(patient, "*.tif")))
    
    for file in image_files:
        if "_mask" in file:
            continue
        
        mask_file = file.replace(".tif", "_mask.tif")

        if not os.path.exists(mask_file):
            continue

        # Read image & mask
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

        # Resize to 128×128
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

        # Normalize
        img = img / 255.0
        mask = mask / 255.0

        # Save as PNG
        cv2.imwrite(f"{IMG_OUT}/img_{count}.png", (img * 255).astype('uint8'))
        cv2.imwrite(f"{MASK_OUT}/mask_{count}.png", (mask * 255).astype('uint8'))

        count += 1

print("✅ Preprocessing Complete")
print(f"Total samples created: {count}")
