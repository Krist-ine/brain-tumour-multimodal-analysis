import os
import re

# Set the folder path
folder_path = 'data/val/masks'  # Replace with your folder path

# Regular expression to match files like mask_1.png, mask_42.png, etc.
pattern = re.compile(r'^mask_(\d+)\.png$')

for filename in os.listdir(folder_path):
    match = pattern.match(filename)
    if match:
        num = match.group(1)
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, f'img_{num}.png')
        os.rename(src, dst)
        print(f'Renamed {src} to {dst}')
