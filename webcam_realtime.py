"""
Real-time Facial Emotion Recognition via OpenCV webcam feed.
--------------------------------------------------------------
Run with:
    python webcam_realtime.py

Press 'q' to quit the window.
Requires a trained model at models/emotion_ann.keras (see train.py).
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "models/emotion_ann.keras"
IMG_SIZE = 48

EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]


def main():
    model = load_model(MODEL_PATH)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check your camera permissions/index.")

    print("Starting webcam feed. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face = face.astype("float32") / 255.0
            face = np.expand_dims(face, axis=(0, -1))

            prediction = model.predict(face, verbose=0)
            emotion = EMOTION_LABELS[int(np.argmax(prediction))]
            confidence = float(np.max(prediction)) * 100

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                frame, f"{emotion} ({confidence:.1f}%)",
                (x, max(y - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2,
            )

        cv2.imshow("Facial Emotion Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
