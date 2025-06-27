import librosa
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

GENRES = 'blues classical country disco hiphop'.split()
ROOT_DIR = 'data/genres_original'


def get_mel_spectrogram(file_path, sr=22050, n_mels=128, fmax=8000):

    try:
        y, sr = librosa.load(file_path, duration=30)
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


def resize_to_128x128(mel_db):
    return cv2.resize(mel_db, (128, 128), interpolation=cv2.INTER_AREA)


def create_dataset():
    X = []
    y = []
    genres = []

    for genre in GENRES:
        genre_path = os.path.join(ROOT_DIR, genre)
        if not os.path.isdir(genre_path):
            continue
        for file in os.listdir(genre_path):
            if file.endswith('.wav'):
                file_path = os.path.join(genre_path, file)
                mel = get_mel_spectrogram(file_path)
                if mel is None:
                    print(f"Skipping bad file: {file_path}")
                    continue
                mel_resized = resize_to_128x128(mel)
                X.append(mel_resized)
                y.append(genre)
                genres.append(genre)

    X = np.array(X).reshape(-1, 128, 128, 1)
    y_encoded = LabelEncoder().fit_transform(y)
    y = tf.keras.utils.to_categorical(y_encoded)
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y_encoded, test_size=0.2, random_state=42)

    return X_train, X_val, y, y_train, y_val


def create_cnn(input_shape=(128, 128, 1), num_classes=5):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def train_model(X_train, X_val, y, y_train, y_val):
    model = create_cnn(num_classes=y.shape[1])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_val, y_val)
    )

    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {val_accuracy:.4f}")


X_train, X_val, y, y_train, y_val = create_dataset()
train_model(X_train, X_val, y, y_train, y_val)