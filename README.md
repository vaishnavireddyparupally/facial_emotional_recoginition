# Facial Emotion Recognition Using ANN

A deep-learning system that classifies facial images into 7 emotions
(Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) using an
Artificial Neural Network trained on the FER-2013 dataset.

## Project Structure

```
Facial_Emotion_Recognition_ANN/
├── dataset/
│   └── fer2013.csv          # Download from Kaggle, place here
├── models/
│   └── emotion_ann.keras    # Created by train.py
├── plots/                   # Training curves + confusion matrix (created by train.py)
├── train.py                 # Trains and evaluates the ANN
├── app.py                   # Streamlit app (upload image or webcam snapshot)
├── webcam_realtime.py       # Live OpenCV webcam window (no Streamlit)
├── requirements.txt
└── README.md
```

## 1. Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

## 2. Get the Dataset

Download **FER-2013** from Kaggle:
https://www.kaggle.com/datasets/msambare/fer2013
(or any FER-2013 CSV variant with `emotion` and `pixels` columns).

Place the CSV at:

```
dataset/fer2013.csv
```

## 3. Train the Model

```bash
python train.py --data dataset/fer2013.csv --epochs 30 --batch-size 64
```

This will:
- Load and preprocess the dataset (resize, grayscale, normalize)
- Split into train/validation/test sets
- Build and train the ANN (Flatten → Dense → Dropout ×3 → Softmax)
- Save training/validation accuracy & loss curves to `plots/`
- Print a classification report and save a confusion matrix to `plots/`
- Save the trained model to `models/emotion_ann.keras`

Useful flags:

| Flag | Default | Description |
|---|---|---|
| `--data` | `dataset/fer2013.csv` | Path to dataset CSV |
| `--epochs` | 30 | Number of training epochs |
| `--batch-size` | 64 | Batch size |
| `--val-split` | 0.1 | Fraction of training data used for validation |
| `--test-split` | 0.2 | Fraction of data held out for testing |
| `--output-dir` | `models` | Where to save the trained model |
| `--model-name` | `emotion_ann.keras` | Saved model filename |

## 4. Run the App

**Streamlit (recommended — upload image or take a webcam snapshot in-browser):**

```bash
streamlit run app.py
```

**Live OpenCV webcam window:**

```bash
python webcam_realtime.py
```
Press `q` to close the window.

## Model Architecture

```
Input (48x48x1)
   -> Flatten
   -> Dense(512, relu) -> Dropout(0.3)
   -> Dense(256, relu) -> Dropout(0.3)
   -> Dense(128, relu) -> Dropout(0.2)
   -> Dense(7, softmax)
```

Optimizer: Adam (lr=0.001) · Loss: sparse_categorical_crossentropy

## Notes

- An ANN operating on flattened pixels is a valid baseline, but facial
  images have strong spatial structure. Consider building a CNN
  version afterward and comparing ANN vs CNN — a great talking point
  for interviews/presentations.
- Face detection uses OpenCV's Haar Cascade
  (`haarcascade_frontalface_default.xml`), bundled with `opencv-python`.
- Typical confusions to expect in the confusion matrix: Sad↔Neutral,
  Fear↔Surprise, Angry↔Disgust.
