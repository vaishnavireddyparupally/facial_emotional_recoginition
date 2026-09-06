"""
Facial Emotion Recognition - Streamlit App
--------------------------------------------
Run with:
    streamlit run app.py

Two modes:
  1. Upload an image  -> detect face(s) -> predict emotion
  2. Webcam snapshot   -> detect face   -> predict emotion

Requires a trained model at models/emotion_ann.keras (see train.py).
"""

import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
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

EMOTION_EMOJI = {
    "Angry": "😠",
    "Disgust": "🤢",
    "Fear": "😨",
    "Happy": "😄",
    "Sad": "😢",
    "Surprise": "😲",
    "Neutral": "😐",
}


@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return load_model(MODEL_PATH)


@st.cache_resource
def get_face_cascade():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )


def detect_faces(gray_image, cascade):
    return cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)


def preprocess_face(face_gray):
    face = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=(0, -1))  # shape (1, 48, 48, 1)
    return face


def predict_emotion(model, face_gray):
    face_input = preprocess_face(face_gray)
    probs = model.predict(face_input, verbose=0)[0]
    predicted_idx = int(np.argmax(probs))
    return EMOTION_LABELS[predicted_idx], probs


def draw_results(image_bgr, faces, model):
    """Annotate image with bounding boxes + predicted emotion. Returns
    the annotated image plus the probability array of the first face
    found (for the sidebar chart)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    first_probs = None

    for (x, y, w, h) in faces:
        face_gray = gray[y:y + h, x:x + w]
        label, probs = predict_emotion(model, face_gray)
        if first_probs is None:
            first_probs = probs

        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(
            image_bgr, f"{label} ({probs.max()*100:.1f}%)",
            (x, max(y - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2,
        )

    return image_bgr, first_probs


def show_probability_chart(probs):
    if probs is None:
        return
    df = pd.DataFrame({"Emotion": EMOTION_LABELS, "Confidence": probs})
    df = df.sort_values("Confidence", ascending=False).set_index("Emotion")
    st.bar_chart(df)


def main():
    st.set_page_config(page_title="Facial Emotion Recognition", page_icon="😊", layout="centered")
    st.title("😊 Facial Emotion Recognition")
    st.caption("ANN model trained on FER-2013 · Upload a photo or use your webcam")

    model = get_model()
    cascade = get_face_cascade()

    if model is None:
        st.error(
            f"No trained model found at `{MODEL_PATH}`. "
            f"Run `python train.py` first to train and save a model."
        )
        return

    mode = st.radio("Choose input mode:", ["Upload Image", "Webcam Snapshot"], horizontal=True)

    image_bgr = None

    if mode == "Upload Image":
        uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            pil_image = Image.open(uploaded_file).convert("RGB")
            image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    else:  # Webcam Snapshot
        camera_image = st.camera_input("Take a photo")
        if camera_image is not None:
            pil_image = Image.open(camera_image).convert("RGB")
            image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    if image_bgr is not None:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray, cascade)

        if len(faces) == 0:
            st.warning("No face detected. Try a clearer, front-facing photo with good lighting.")
            st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Input image")
            return

        annotated, probs = draw_results(image_bgr.copy(), faces, model)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detected emotion(s)")

        if probs is not None:
            top_label = EMOTION_LABELS[int(np.argmax(probs))]
            st.subheader(f"Detected Emotion: {EMOTION_EMOJI[top_label]} {top_label}")
            st.write(f"Confidence: {probs.max()*100:.1f}%")
            st.markdown("**Emotion probabilities**")
            show_probability_chart(probs)


if __name__ == "__main__":
    main()
