import zipfile
import os
import cv2
import numpy as np
import shutil
from glob import glob

# Flatten to /images and /masks
source_root = "data"
os.makedirs("dataset/images", exist_ok=True)
os.makedirs("dataset/masks", exist_ok=True)

for patient in os.listdir(source_root):
    patient_path = os.path.join(source_root, patient)
    if os.path.isdir(patient_path):
        for file in os.listdir(patient_path):
            if "_mask" in file:
                shutil.copy(os.path.join(patient_path, file), "dataset/masks")
            else:
                shutil.copy(os.path.join(patient_path, file), "dataset/images")

# Padding + Resize only if not already 256x256
def pad_and_resize(image, target_size=(256, 256), is_mask=False):
    if image.shape[:2] == target_size:
        return image  # No need to pad or resize

    h, w = image.shape[:2]
    max_dim = max(h, w)

    top = (max_dim - h) // 2
    bottom = max_dim - h - top
    left = (max_dim - w) // 2
    right = max_dim - w - left

    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    resized = cv2.resize(padded, target_size, interpolation=interpolation)
    return resized

# Apply only when needed
image_paths = sorted(glob("dataset/images/*"))
mask_paths = sorted(glob("dataset/masks/*"))

for img_path in image_paths:
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        continue
    image_resized = pad_and_resize(image, target_size=(256, 256), is_mask=False)
    cv2.imwrite(img_path, image_resized)

for msk_path in mask_paths:
    mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        continue
    mask_resized = pad_and_resize(mask, target_size=(256, 256), is_mask=True)
    cv2.imwrite(msk_path, mask_resized)

print("✅ All images and masks padded & resized **only if not already 256×256**.")
