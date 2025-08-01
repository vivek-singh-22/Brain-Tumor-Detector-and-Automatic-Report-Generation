import os
import json
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

# === CONFIG ===
MODEL_PATH = r"C:\CDAC\Tumor_Detector_And_Report_Generation\brain_tumor_classifier.keras"              # Change this to your model
VAL_DIR = r"C:\CDAC\Tumor_Detector_And_Report_Generation\Dataset\Val"                           # Your validation folder used in training
IMAGE_SIZE = (512, 512)                   # Must match model input
LIMIT = 5                                 # Number of images per class to test

CLASS_INDICES = {'no_tumor': 0, 'glioma_tumor': 1, 'meningioma_tumor': 2, 'pituitary_tumor': 3}
INV_CLASS_INDICES = {v: k for k, v in CLASS_INDICES.items()}

# === Load model ===
model = load_model(MODEL_PATH)

# === Predict limited images ===
results = []

for class_name in os.listdir(VAL_DIR):
    class_folder = os.path.join(VAL_DIR, class_name)
    if not os.path.isdir(class_folder):
        continue

    image_files = os.listdir(class_folder)[:LIMIT]

    for fname in image_files:
        img_path = os.path.join(class_folder, fname)
        try:
            # Load and preprocess image
            img = load_img(img_path, target_size=IMAGE_SIZE).convert("RGB")
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            preds = model.predict(img_array)
            pred_idx = np.argmax(preds)
            pred_label = INV_CLASS_INDICES[pred_idx]

            # Store result
            results.append({
                "image_path": img_path,
                "actual_class": class_name,
                "predicted_class": pred_label
            })

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# === Save as JSON ===
with open("val_predictions.json", "w") as f:
    json.dump(results, f, indent=4)

print(f"✅ Prediction done for {LIMIT} images per class. Output saved to 'val_predictions.json'")
