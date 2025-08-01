from model import build_unet
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
import os
import cv2
import numpy as np
from PIL import Image
#import segmentation_models as sm
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from segmentation_models.losses import TverskyLoss
from segmentation_models.metrics import IOUScore, FScore
import tensorflow as tf

# ------------------------------
# ⚙️ Paths
image_dir = 'dataset/images'
mask_dir = 'dataset/masks'

# ------------------------------
# 🗂 Filter valid files
all_files = sorted([
    f for f in os.listdir(image_dir)
    if f.lower().endswith(('.tif', '.tiff')) and 
       os.path.exists(os.path.join(mask_dir, f.replace('.tif', '_mask.tif')))
])

train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

# ------------------------------
# 🔁 Data Generator
class BrainTumorGenerator(Sequence):
    def __init__(self, image_dir, mask_dir, file_list, batch_size=4, target_size=(256, 256)):  # 🔄 Changed to 256
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.file_list = file_list
        self.batch_size = batch_size
        self.target_size = target_size
        print(f"Matched image-mask pairs: {len(self.file_list)}")

    def __len__(self):
        return int(np.ceil(len(self.file_list) / self.batch_size))

    def __getitem__(self, idx):
        batch_files = self.file_list[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, Y = [], []

        for img_filename in batch_files:
            img_path = os.path.join(self.image_dir, img_filename)
            mask_filename = img_filename.replace('.tif', '_mask.tif')
            mask_path = os.path.join(self.mask_dir, mask_filename)

            try:
                img = Image.open(img_path).convert("L")
                mask = Image.open(mask_path).convert("L")

                img = img.resize(self.target_size)
                mask = mask.resize(self.target_size)

                img = np.array(img).astype(np.float32) / 255.0
                mask = np.array(mask).astype(np.float32)
                mask = (mask > 127).astype(np.float32)

                X.append(img[..., np.newaxis])
                Y.append(mask[..., np.newaxis])
            except Exception as e:
                print(f"⚠️ Skipping {img_filename}: {e}")
                continue

        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

# ------------------------------
# Build model with 256×256 input
model = build_unet(input_shape=(256, 256, 1))  # 🔄 Changed to 256

# ------------------------------
# Dice loss + BCE combo
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred_bin, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

loss = TverskyLoss(alpha=0.5, beta=0.5)  # You can tune alpha/beta
# ------------------------------
# Compile model
#model.compile(optimizer=Adam(learning_rate=1e-4), loss=bce_dice_loss, metrics=['accuracy', dice_coef])
model.compile(optimizer='adam', loss=loss, metrics=[sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)])


# ------------------------------
#  Generators with 256×256
batch_size = 2
train_gen = BrainTumorGenerator(image_dir, mask_dir, train_files, batch_size=batch_size, target_size=(256, 256))
val_gen = BrainTumorGenerator(image_dir, mask_dir, val_files, batch_size=batch_size, target_size=(256, 256))

print("Images:", len(os.listdir(image_dir)))
print("Masks :", len(os.listdir(mask_dir)))

# ------------------------------
# Callbacks
callbacks = [
    EarlyStopping(monitor='val_dice_coef', mode='max',patience=7, restore_best_weights=True),
]

# ------------------------------
# Train model
model.fit(train_gen, validation_data=val_gen, epochs=70, callbacks=callbacks)

# ------------------------------
# Save model
model.save("tumor_segmentation_model.h5")
