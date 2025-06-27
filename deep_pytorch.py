import librosa
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torchview import draw_graph


GENRES = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
ROOT_DIR = 'data/genres_original'


# --- CNN Model ---
class CNNGenreClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc_stack(self.conv_stack(x))


# --- Preprocessing ---
def get_mel_spectrogram(file_path, n_mels=128, fmax=8000):
    try:
        y, sr = librosa.load(file_path)
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


def resize_to_128x128(mel_db):
    return cv2.resize(mel_db, (128, 128), interpolation=cv2.INTER_AREA)


# --- Custom Dataset ---
class GenreDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --- Dataset Creation ---
def create_dataset():
    X = []
    y = []

    for genre in GENRES:
        genre_path = os.path.join(ROOT_DIR, genre)
        for file in os.listdir(genre_path):
            if file.endswith('.wav'):
                file_path = os.path.join(genre_path, file)
                mel = get_mel_spectrogram(file_path)
                if mel is None:
                    print(f"Skipping bad file: {file_path}")
                    continue
                mel_resized = resize_to_128x128(mel)
                mel_norm = (mel_resized - np.mean(mel_resized)) / np.std(mel_resized)
                X.append(mel_norm)
                y.append(genre)

    X = np.array(X).reshape(-1, 1, 128, 128)
    y_encoded = LabelEncoder().fit_transform(y)

    return train_test_split(X, y_encoded, stratify=y_encoded, test_size=0.2, random_state=42)


# --- Training + Evaluation ---
def train_and_evaluate():
    # Prepare data
    X_train, X_val, y_train, y_val = create_dataset()
    train_loader = DataLoader(GenreDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(GenreDataset(X_val, y_val), batch_size=32, shuffle=False)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNGenreClassifier(num_classes=len(GENRES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

    # Validation
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(batch_y.numpy())

    accuracy = accuracy_score(all_targets, all_preds)
    print(f"\nValidation Accuracy: {accuracy:.4f}")


# Run training
train_and_evaluate()
# display_cnn()