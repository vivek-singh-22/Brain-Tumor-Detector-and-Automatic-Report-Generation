import streamlit as st
import numpy as np
import json
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import matplotlib.pyplot as plt

# Load model
model = load_model(r"C:\CDAC\Tumor_Detector_And_Report_Generation\brain_tumor_classifier.keras")

# Load label map
with open(r"C:\CDAC\Tumor_Detector_And_Report_Generation\class_indices.json") as f:
    class_indices = json.load(f)
index_to_class = {v: k for k, v in class_indices.items()}

# Grad-CAM utility
def get_gradcam_heatmap(model, img_array, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(index=-1).output, model.get_layer("Conv_1_bn").output]
    )

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

# Streamlit UI
st.title("🧠 Brain Tumor Classifier with Grad-CAM")
uploaded_file = st.file_uploader("Choose an image for prediction & Report...", type=["jpg", "jpeg", "png", "tiff", "tif"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_resized = img.resize((512, 512))  # match training size
    img_array = img_to_array(img_resized) / 255.0
    img_array_exp = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array_exp)
    predicted_index = np.argmax(pred)
    predicted_class = index_to_class[predicted_index]
    confidence_score = float(pred[0][predicted_index])

    st.success(f" Predicted: **{predicted_class}**")
    st.info(f" Confidence: **{confidence_score * 100:.2f}%**")

    # === Grad-CAM ===
    heatmap = get_gradcam_heatmap(model, img_array_exp, predicted_index)
    heatmap_resized = cv2.resize(heatmap, (img_resized.size[0], img_resized.size[1]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    # Superimpose on original image
    original_img_cv = np.array(img_resized)
    superimposed_img = cv2.addWeighted(original_img_cv, 0.6, heatmap_color, 0.4, 0)

    st.markdown("###  Grad-CAM Visualization")
    st.image(superimposed_img, channels="RGB", use_container_width=True)

    # Save result to JSON
    result_json = {
        "filename": uploaded_file.name,
        "predicted_class": predicted_class,
        "confidence_score": round(confidence_score, 4)
    }

    with open("prediction_result.json", "w") as f:
        json.dump(result_json, f, indent=4)

    st.download_button(
        label="📥 Download Prediction JSON",
        data=json.dumps(result_json, indent=4),
        file_name="prediction_result.json",
        mime="application/json"
    )
