import streamlit as st
import numpy as np
import cv2
import json
from PIL import Image
from tensorflow.keras.models import load_model
import os

# --- Constants ---
MODEL_PATH = r'C:\CDAC\Tumor_Detector_And_Report_Generation\tumor_segmentation_model.h5'
PIXEL_SPACING_CM = 0.05  # Adjust this if you know exact value

# --- Load model once ---
@st.cache_resource
def load_unet_model():
    return load_model(MODEL_PATH, compile=False)

model = load_unet_model()

# --- Preprocessing ---
def preprocess_image(uploaded_img, target_size=(256, 256)):
    image = Image.open(uploaded_img).convert("L")
    original_size = image.size
    image_resized = image.resize(target_size)
    image_arr = np.array(image_resized).astype(np.float32) / 255.0
    return image_arr[..., np.newaxis], np.array(image), original_size

# --- Postprocessing ---
def get_tumor_info(mask, prob_map):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        area_px = cv2.contourArea(cnt)
        area_cm2 = area_px * (PIXEL_SPACING_CM**2)
        confidence = np.mean(prob_map[mask == 1])
        confidence_percent = round(float(confidence) * 100, 2) 
        return int(x), int(y), int(w), int(h), round(area_cm2, 2), confidence_percent
    return None

# --- Streamlit UI ---
st.title("🧠 Brain Tumor Segmentation App")
st.write("Upload an MRI image to detect tumor location, size, and confidence.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["tif", "tiff", "png", "jpg", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Process image
    img_tensor, original_img, orig_size = preprocess_image(uploaded_file)
    input_tensor = np.expand_dims(img_tensor, axis=0)

    # Predict
    pred_mask = model.predict(input_tensor)[0, ..., 0]
    binary_mask = (pred_mask > 0.5).astype(np.uint8)
    
    binary_mask_resized = cv2.resize(binary_mask, orig_size, interpolation=cv2.INTER_NEAREST)
    tumor_info = get_tumor_info(binary_mask_resized, cv2.resize(pred_mask, orig_size))

    # Resize back to original for visualization
   # pred_mask_resized = cv2.resize(binary_mask, orig_size)
    #tumor_info = get_tumor_info(pred_mask_resized, pred_mask)

    # Overlay mask
    overlay = np.stack([original_img]*3, axis=-1)
    overlay[binary_mask_resized == 1] = [255, 0, 0]

    st.image(overlay, caption="Tumor Segmentation (red)", use_container_width=True)

    # Display tumor info
    if tumor_info:
        x, y, w, h, area_cm2, confidence = tumor_info
        st.subheader("🔍 Tumor Details")
        st.json({
            "Location (px)": {"x": x, "y": y},
            "Size (px)": {"width": w, "height": h},
            "Area (cm²)": area_cm2,
            "Confidence of location & size": confidence
        })
    else:
        st.warning(" No tumor detected.")

   