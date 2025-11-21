import os
import random
import shutil

IMG_DIR = "data/images"
MASK_DIR = "data/masks"

TRAIN_IMG = "data/train/images"
TRAIN_MASK = "data/train/masks"
VAL_IMG = "data/val/images"
VAL_MASK = "data/val/masks"

for d in [TRAIN_IMG, TRAIN_MASK, VAL_IMG, VAL_MASK]:
    os.makedirs(d, exist_ok=True)

image_files = sorted(os.listdir(IMG_DIR))
random.shuffle(image_files)

split_ratio = 0.8
train_count = int(len(image_files) * split_ratio)

train_files = image_files[:train_count]
val_files = image_files[train_count:]

for f in train_files:
    shutil.copy(f"{IMG_DIR}/{f}", f"{TRAIN_IMG}/{f}")
    shutil.copy(f"{MASK_DIR}/{f.replace('img', 'mask')}", f"{TRAIN_MASK}/{f.replace('img', 'mask')}")

for f in val_files:
    shutil.copy(f"{IMG_DIR}/{f}", f"{VAL_IMG}/{f}")
    shutil.copy(f"{MASK_DIR}/{f.replace('img', 'mask')}", f"{VAL_MASK}/{f.replace('img', 'mask')}")

print("✅ Train/Validation Split Completed")
