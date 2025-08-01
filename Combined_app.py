import streamlit as st
import numpy as np
import json
import cv2
import os
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

# === Load Models ===
classifier_model = load_model(r'C:\CDAC\Tumor_Detector_And_Report_Generation\brain_tumor_classifier.keras')
segment_model = load_model(r'C:\CDAC\Tumor_Detector_And_Report_Generation\tumor_segmentation_model.h5', compile=False)  # Skip compiling for inference

# === Load class indices ===
with open(r"C:\CDAC\Tumor_Detector_And_Report_Generation\class_indices.json") as f:
    class_indices = json.load(f)
index_to_class = {v: k for k, v in class_indices.items()}

# === Grad-CAM Utility ===
def get_gradcam_heatmap(model, img_array, pred_index=None):
    grad_model = tf.keras.models.Model([
        model.inputs], [model.get_layer(index=-1).output, model.get_layer("Conv_1_bn").output])

    with tf.GradientTape() as tape:
        preds, conv_outputs = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# === Segmentation Info Utility ===
PIXEL_SPACING_CM = 0.0625  # Example: 1 pixel = 0.0625 cm

def get_tumor_info(mask, prob_map):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        area_px = cv2.contourArea(cnt)
        area_cm2 = area_px * (PIXEL_SPACING_CM ** 2)
        confidence = np.mean(prob_map[mask == 255])
        confidence_percent = round(float(confidence) * 100, 2)
        return int(x), int(y), int(w), int(h), round(area_cm2, 2), confidence_percent
    else:
        return None, None, None, None, 0.0, 0.0

# === Streamlit UI ===
st.title("🧠 Brain Tumor Classification + Segmentation")
uploaded_file = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # === Preprocess image for classification ===
    img_resized = img.resize((512, 512))
    img_array = img_to_array(img_resized) / 255.0
    img_array_exp = np.expand_dims(img_array, axis=0)

    # === Classification ===
    preds = classifier_model.predict(img_array_exp)
    pred_index = np.argmax(preds)
    pred_class = index_to_class[pred_index]
    confidence = float(preds[0][pred_index])

    st.subheader("🧠 Classification Result")
    st.success(f"Prediction: **{pred_class}**")
    st.info(f"Confidence: **{confidence * 100:.2f}%**")

    # === Grad-CAM ===
    heatmap = get_gradcam_heatmap(classifier_model, img_array_exp, pred_index)
    heatmap_resized = cv2.resize(heatmap, (512, 512))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(np.array(img_resized), 0.6, heatmap_color, 0.4, 0)
    st.image(superimposed_img, caption="Grad-CAM Heatmap", use_container_width=True)

    # === Segmentation ===
    st.subheader("✂️ Tumor Segmentation")
    seg_img = img.resize((256, 256)).convert('L')
    seg_array = np.expand_dims(img_to_array(seg_img) / 255.0, axis=0)

    prob_map = segment_model.predict(seg_array)[0]  # probability map
    seg_mask = (prob_map > 0.5).astype(np.uint8) * 255  # binary mask

    # Resize for overlay display
    seg_mask_resized = cv2.resize(seg_mask, (512, 512))
    seg_mask_rgb = cv2.cvtColor(seg_mask_resized, cv2.COLOR_GRAY2RGB)
    segmented_overlay = cv2.addWeighted(np.array(img_resized), 0.6, seg_mask_rgb, 0.4, 0)
    st.image(segmented_overlay, caption="Tumor Segmentation Overlay", use_container_width=True)

    # Get location info
    x, y, w, h, area_cm2, conf_percent = get_tumor_info(seg_mask, prob_map)
    bbox_info = {"x": x, "y": y, "width": w, "height": h} if None not in (x, y, w, h) else None

    # === JSON Output ===
    result = {
        "filename": uploaded_file.name,
        "classification": {
            "predicted_class": pred_class,
            "confidence": round(confidence, 4)
        },
        "segmentation": {
            "segmentation_mask_shape": seg_mask.shape,
            "tumor_area_pixels": int(np.sum(seg_mask > 0)),
            "bounding_box": bbox_info,
            "area_cm2": area_cm2,
            "confidence_percent": conf_percent
        }
    }

    os.makedirs("results", exist_ok=True)
    json_path = os.path.join("results", f"{os.path.splitext(uploaded_file.name)[0]}_analysis.json")
    with open(json_path, "w") as jf:
        json.dump(result, jf, indent=4)

    st.success(f"✅ JSON file saved at: {json_path}")
