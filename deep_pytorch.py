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

from collections import defaultdict, Counter

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
    def __init__(self, X, y, track_ids):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.track_ids = track_ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.track_ids[idx]


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


def create_dataset_with_track_ids():
    def build_split(track_ids_subset):
        X, y, track_ids = [], [], []
        for tid in track_ids_subset:
            genre = track_to_label[tid]
            genre_encoded = GENRES.index(genre)
            for chunk in track_to_chunks[tid]:
                X.append(chunk)
                y.append(genre_encoded)
                track_ids.append(tid)
        return np.array(X).reshape(-1, 1, 128, 128), np.array(y), track_ids

    chunk_data = []  # each entry: (mel, genre, track_id)

    for genre in GENRES:
        genre_path = os.path.join(ROOT_DIR, genre)
        for file in os.listdir(genre_path):
            if file.endswith('.wav'):
                file_path = os.path.join(genre_path, file)
                track_id = os.path.splitext(file)[0]

                sr = 22050
                try:
                    y_audio, _ = librosa.load(file_path, sr=sr, duration=30)
                except Exception as e:
                    print(f"Failed to load {file_path}: {e}")
                    continue
                samples_per_chunk = sr * 3

                for i in range(10):  # 10 chunks
                    start = i * samples_per_chunk
                    end = start + samples_per_chunk
                    chunk = y_audio[start:end]
                    if len(chunk) < samples_per_chunk:
                        continue

                    mel = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128, fmax=8000)
                    mel_db = librosa.power_to_db(mel, ref=np.max)
                    mel_resized = resize_to_128x128(mel_db)
                    mel_norm = (mel_resized - np.mean(mel_resized)) / np.std(mel_resized)

                    chunk_data.append((mel_norm, genre, track_id))

    # Group by track
    track_to_chunks = defaultdict(list)
    track_to_label = {}
    for mel, genre, track_id in chunk_data:
        track_to_chunks[track_id].append(mel)
        track_to_label[track_id] = genre

    # Split by unique track IDs
    track_ids = list(track_to_chunks.keys())
    genres = [track_to_label[tid] for tid in track_ids]
    train_tracks, test_tracks = train_test_split(track_ids, stratify=genres, test_size=0.2, random_state=42)

    X_train, y_train, train_ids = build_split(train_tracks)
    X_test, y_test, test_ids = build_split(test_tracks)

    return X_train, X_test, y_train, y_test, train_ids, test_ids


def majority_vote(preds, ids, true_label_map):
    grouped = defaultdict(list)
    for pred, track_id in zip(preds, ids):
        grouped[track_id].append(pred)

    final_preds = []
    final_trues = []

    for track_id, pred_list in grouped.items():
        vote = Counter(pred_list).most_common(1)[0][0]
        final_preds.append(vote)
        final_trues.append(true_label_map[track_id])

    return final_trues, final_preds


# --- Training + Evaluation ---
def train_and_evaluate():
    # Prepare data
    X_train, X_val, y_train, y_val, train_ids, test_ids = create_dataset_with_track_ids()
    train_loader = DataLoader(GenreDataset(X_train, y_train, train_ids), batch_size=32, shuffle=True)
    val_loader = DataLoader(GenreDataset(X_val, y_val, test_ids), batch_size=32, shuffle=False)

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
        for batch_x, batch_y, _ in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

        # Validation with track ID voting
        model.eval()
        chunk_preds = []
        chunk_ids = []
        chunk_trues = []

        with torch.no_grad():
            for batch_x, batch_y, batch_track in val_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                chunk_preds.extend(preds)
                chunk_ids.extend(batch_track)
                chunk_trues.extend(batch_y.tolist())

        # Build true label map for voting
        true_label_map = {}
        for track_id, true_label in zip(chunk_ids, chunk_trues):
            if track_id not in true_label_map:
                true_label_map[track_id] = true_label  # assumes all chunks from a track have same label

        # Track-level accuracy via majority vote
        true, pred = majority_vote(chunk_preds, chunk_ids, true_label_map)
        acc = accuracy_score(true, pred)
        print(f"\nTrack-Level Accuracy via Majority Voting: {acc:.4f}")


# Run training
train_and_evaluate()
# display_cnn()
