"""
Facial Emotion Recognition using an Artificial Neural Network (ANN)
--------------------------------------------------------------------
Trains an ANN on the FER-2013 dataset (Kaggle) to classify facial
images into 7 emotion categories: Angry, Disgust, Fear, Happy, Sad,
Surprise, Neutral.

Usage:
    python train.py --data dataset/fer2013.csv --epochs 30 --batch-size 64

Expects a CSV with columns: emotion, pixels, (Usage - optional)
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]

IMG_SIZE = 48


def parse_args():
    parser = argparse.ArgumentParser(description="Train Facial Emotion Recognition ANN")
    parser.add_argument("--data", type=str, default="dataset/fer2013.csv",
                         help="Path to the FER-2013 CSV file")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--model-name", type=str, default="emotion_ann.keras")
    parser.add_argument("--plots-dir", type=str, default="plots")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_data(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find dataset at '{csv_path}'. Download FER-2013 from "
            f"Kaggle and place it there, or pass --data <path>."
        )
    print(f"Loading dataset from {csv_path} ...")
    df = pd.read_csv(csv_path)
    print("Dataset shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Missing values:\n", df.isnull().sum())
    print("Duplicate rows:", df.duplicated().sum())
    print("Emotion distribution:\n", df["emotion"].value_counts())
    return df


def preprocess(df: pd.DataFrame):
    print("Converting pixel strings to arrays ...")
    X = np.array(
        df["pixels"].apply(lambda x: np.array(x.split(), dtype=np.float32)).tolist()
    )
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X = X / 255.0  # normalize to [0, 1]
    y = df["emotion"].values
    print("X shape:", X.shape, "| y shape:", y.shape)
    print("Unique classes:", np.unique(y))
    return X, y


def build_model():
    model = Sequential([
        Flatten(input_shape=(IMG_SIZE, IMG_SIZE, 1)),

        Dense(512, activation="relu"),
        Dropout(0.3),

        Dense(256, activation="relu"),
        Dropout(0.3),

        Dense(128, activation="relu"),
        Dropout(0.2),

        Dense(len(EMOTION_LABELS), activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "accuracy.png"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "loss.png"), bbox_inches="tight")
    plt.close()

    print(f"Saved training curves to '{plots_dir}/'")


def plot_confusion_matrix(y_test, y_pred, plots_dir):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS,
        cmap="Blues",
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"), bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to '{plots_dir}/confusion_matrix.png'")


def main():
    args = parse_args()

    df = load_data(args.data)
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_split,
        random_state=args.seed,
        stratify=y,
    )
    print("X_train:", X_train.shape, "| X_test:", X_test.shape)

    model = build_model()
    model.summary()

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, args.model_name)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True),
    ]

    history = model.fit(
        X_train, y_train,
        validation_split=args.val_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    plot_history(history, args.plots_dir)

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=EMOTION_LABELS))

    plot_confusion_matrix(y_test, y_pred, args.plots_dir)

    # Final save (in case ModelCheckpoint didn't trigger, e.g. very short training)
    model.save(checkpoint_path)
    print(f"\nModel saved to '{checkpoint_path}'")


if __name__ == "__main__":
    main()
