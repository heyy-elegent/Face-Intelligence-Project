"""Facial Emotion Recognition - FER2013 CNN training pipeline.

Trains a compact CNN (VGG-style) on the FER2013 dataset and saves a
TensorFlow/Keras model (emotion_model.h5) for real-time inference in app.py.

Usage:
    python train.py --epochs 40 --batch 64
"""
import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
DATASET_URL = "https://figshare.com/ndownloader/files/33876163"  # FER2013 CSV mirror
LABEL_URL = "https://www.kaggle.com/datasets/msambare/fer2013"  # canonical source


def load_fer2013(csv_path):
    df = pd.read_csv(csv_path)
    pixels = df["pixels"].apply(lambda p: np.fromstring(p, dtype=int, sep=" "))
    X = np.vstack(pixels.values).reshape(-1, 48, 48, 1).astype("float32") / 255.0
    y = keras.utils.to_categorical(df["emotion"].values, num_classes=7)
    return X, y


def ensure_dataset(save_dir="."):
    csv_path = os.path.join(save_dir, "fer2013.csv")
    if os.path.exists(csv_path):
        print("[dataset] found:", csv_path)
        return csv_path

    print("[dataset] fer2013.csv not found locally.")
    print("[dataset] please download it from:")
    print("   ", LABEL_URL)
    print(f"  and place it as: {csv_path}")
    print("[dataset] attempting automatic download ...")
    try:
        import urllib.request

        print("[dataset] downloading ... (this may take a while)")
        urllib.request.urlretrieve(DATASET_URL, csv_path)
        print("[dataset] downloaded:", csv_path)
        return csv_path
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "Automatic download failed. Download fer2013.csv manually "
            "(see README) and place it next to train.py."
        ) from exc


def build_model(input_shape=(48, 48, 1), num_classes=7):
    inputs = keras.Input(shape=input_shape)

    def conv_block(x, filters, dropout=0.25):
        x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout)(x)
        return x

    x = conv_block(inputs, 64)
    x = conv_block(x, 128)
    x = conv_block(x, 256)
    x = conv_block(x, 512)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def main():
    parser = argparse.ArgumentParser(description="Train FER2013 emotion CNN")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    csv_path = ensure_dataset()
    print("[data] loading ...")
    X, y = load_fer2013(csv_path)

    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    history = model.fit(
        X,
        y,
        epochs=args.epochs,
        batch_size=args.batch,
        validation_split=0.1,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5),
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        ],
    )

    model.save("emotion_model.h5")
    print("[done] saved emotion_model.h5")

    if os.environ.get("FER_SKIP_PLOT") != "1":
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history.history["accuracy"], label="train")
        ax1.plot(history.history["val_accuracy"], label="val")
        ax1.set_title("Accuracy"); ax1.legend()
        ax2.plot(history.history["loss"], label="train")
        ax2.plot(history.history["val_loss"], label="val")
        ax2.set_title("Loss"); ax2.legend()
        fig.savefig("training_history.png", dpi=150, bbox_inches="tight")
        print("[done] saved training_history.png")


if __name__ == "__main__":
    main()