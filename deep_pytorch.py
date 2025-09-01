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
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt



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


class CNNRNNGenreClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CNNRNNGenreClassifier, self).__init__()

        # CNN feature extractor
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)
        )

        # RNN input: (batch, seq_len, input_size)
        # After 3 MaxPool2d(2) on 128x128 input -> (128, 16, 16)
        self.gru = nn.GRU(
            input_size=16 * 128,  # channels * freq_bins
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Fully connected output layer
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 2, num_classes)  # *2 for bidirectional
        )

    def forward(self, x):
        batch_size = x.size(0)

        # CNN output: [B, 128, 16, 16]
        x = self.conv_stack(x)

        # Reshape for GRU: [B, 16 (time steps), 128*16]
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, time, channels, features]
        x = x.view(batch_size, 16, -1)  # [B, seq_len, input_size]

        # GRU output: [B, seq_len, 2*hidden_size] → take last step
        _, hidden = self.gru(x)  # hidden: [2, B, 128] (bidirectional)
        hidden = torch.cat([hidden[0], hidden[1]], dim=1)  # [B, 256]

        out = self.fc(hidden)
        return out


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


def create_dataset_with_track_ids(chunk_duration=6):
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
    hop_duration = 1

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
                samples_per_chunk = sr * chunk_duration
                hop_length = int(hop_duration * sr)
                total_chunks = 1 + (len(y_audio) - samples_per_chunk) // hop_length  # allows overlap

                for i in range(total_chunks):
                    start = i * hop_length
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


def get_label_from_filename(filename):
    return filename.split(".")[0]


def preprocess_test_file(file_path, sr=22050):
    y, _ = librosa.load(file_path, sr=sr)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_resized = resize_to_128x128(mel_db)
    mel_norm = (mel_resized - np.mean(mel_resized)) / np.std(mel_resized)

    tensor_input = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, 128, 128)
    return tensor_input

def predict_track(file_path, model, device):
    y, _ = librosa.load(file_path, sr=22050, duration=30)
    chunk_preds = []
    sr = 22050
    chunk_duration = 6  # seconds
    hop_duration = 1  # seconds
    samples_per_chunk = int(sr * chunk_duration)
    hop_length = int(sr * hop_duration)

    total_chunks = 1 + (len(y) - samples_per_chunk) // hop_length
    for i in range(total_chunks):
        start = i * hop_length
        end = start + samples_per_chunk
        chunk = y[start:end]
        if len(chunk) < samples_per_chunk:
            continue

        mel = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_resized = resize_to_128x128(mel_db)
        mel_norm = (mel_resized - np.mean(mel_resized)) / np.std(mel_resized)
        x = torch.tensor(mel_norm).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            output = model(x)
            pred = torch.argmax(output, dim=1).item()
            chunk_preds.append(pred)

    # Majority vote
    if chunk_preds:
        final_pred = Counter(chunk_preds).most_common(1)[0][0]
    else:
        final_pred = None  # or handle differently if you expect no valid chunks
    return final_pred


# --- Training + Evaluation ---
def train_and_evaluate():
    # Prepare data
    X_train, X_val, y_train, y_val, train_ids, test_ids = create_dataset_with_track_ids()
    train_loader = DataLoader(GenreDataset(X_train, y_train, train_ids), batch_size=32, shuffle=True)
    val_loader = DataLoader(GenreDataset(X_val, y_val, test_ids), batch_size=32, shuffle=False)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = CNNGenreClassifier(num_classes=len(GENRES)).to(device)
    model = CNNRNNGenreClassifier(num_classes=len(GENRES)).to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # Training loop
    num_epochs = 20
    train_accuracies = []
    val_accuracies = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y, _ in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Training accuracy
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        train_acc = correct / total
        train_accuracies.append(train_acc)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}, "
              f"Training Accuracy: {train_acc:.4f}")

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

        val_acc = accuracy_score(true, pred)
        val_accuracies.append(val_acc)
        print(f"Track-Level Validation Accuracy (Majority Vote): {val_acc:.4f}")
        # if val_acc >= 0.8000:
        #     break
        scheduler.step()

    file_path = "model.pth"
    # torch.save(model, file_path)
    torch.save(model.state_dict(), file_path)

    # disp = ConfusionMatrixDisplay.from_predictions(true, pred, display_labels=GENRES)
    # disp.plot(xticks_rotation=45)
    # plt.title(f"CNN Confusion Matrix")
    # plt.show()

    # print(train_accuracies, val_accuracies)
    # print("-----------------------------------------")
    # print("Starting test")
    #
    # correct = 0
    # total = 0
    # y_true = []
    # y_pred = []
    #
    # for file in os.listdir("test_data"):
    #     if file.endswith(".wav"):
    #         genre = get_label_from_filename(file)
    #         genre_to_idx = {g: i for i, g in enumerate(GENRES)}
    #         true_label = genre_to_idx[genre]
    #         filepath = os.path.join("test_data", file)
    #         pred_label = predict_track(filepath, model, device)
    #
    #         if pred_label is not None:
    #             correct += (pred_label == true_label)
    #             total += 1
    #             y_true.append(true_label)
    #             y_pred.append(pred_label)
    #
    # print(f"Test Accuracy (Majority Vote): {correct / total:.2%}")
    # disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=GENRES)
    # disp.plot(xticks_rotation=45)
    # plt.title(f"Test Set Confusion Matrix")
    # plt.show()


# Run training
# train_and_evaluate()
# display_cnn()
