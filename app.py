"""Real-time Facial Emotion Recognition webcam app.

Loads the trained model produced by train.py and classifies the dominant
emotion on every detected face in the webcam feed.

Usage:
    python train.py        # once, to produce emotion_model.h5
    python app.py          # then run the live demo
"""
import os

import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emotion_model.h5")
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model '{path}' not found.\n"
            "Train it first with:  python train.py\n"
            "(see README.md for the FER2013 dataset)"
        )
    return tf.keras.models.load_model(path, compile=False)


def main():
    model = load_model(MODEL_PATH)
    print("[app] model loaded:", MODEL_PATH)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        raise RuntimeError("OpenCV haarcascade file is missing.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam (camera 0).")

    print("[app] webcam ready - press 'q' to quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
        )

        for (x, y, w, h) in faces:
            roi = gray[y : y + h, x : x + w]
            roi = cv2.resize(roi, (48, 48)).reshape(1, 48, 48, 1) / 255.0

            pred = model.predict(roi, verbose=0)[0]
            label = EMOTIONS[int(np.argmax(pred))]
            confidence = float(np.max(pred))

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} {confidence:.0%}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Facial Emotion Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()