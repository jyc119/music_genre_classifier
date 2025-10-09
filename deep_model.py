import torch
import torch.nn as nn

import librosa
import numpy as np
import cv2
from collections import Counter, defaultdict
from typing import Optional, Tuple
from deep_pytorch import CNNRNNGenreClassifier

class DeepLearningGenreClassifier:
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the classifier by loading a trained PyTorch model.
        """
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        # self.model = torch.load(model_path, map_location=self.device) # Accuracy: 82.5, Loss: 0.9984
        self.model = CNNRNNGenreClassifier(num_classes=10)  # <- You must redefine the architecture here
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        # self.model.eval()  # Evaluation mode

    def _resize_mel(self, mel_db: np.ndarray) -> np.ndarray:
        """
        Resize Mel-spectrogram to 128x128.
        """
        return cv2.resize(mel_db, (128, 128), interpolation=cv2.INTER_AREA)

    def _preprocess_chunk(self, chunk: np.ndarray, sr: int = 22050) -> torch.Tensor:
        """
        Convert audio chunk to normalized spectrogram tensor.
        """
        mel = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_resized = self._resize_mel(mel_db)
        mel_norm = (mel_resized - np.mean(mel_resized)) / np.std(mel_resized)
        x = torch.tensor(mel_norm).unsqueeze(0).unsqueeze(0).float().to(self.device)
        return x

    def predict(self, file_path: str) -> Optional[Tuple[int, float]]:
        """
        Predict genre of the input audio file using chunk-based voting.
        """
        # total_chunks = 1 + (len(y) - samples_per_chunk) // hop_length
        # for i in range(total_chunks):
        #     start = i * hop_length
        #     end = start + samples_per_chunk
        #     chunk = y[start:end]
        #
        #     if len(chunk) < samples_per_chunk:
        #         continue
        #
        #     x = self._preprocess_chunk(chunk, sr=sr)
        #
        #     with torch.no_grad():
        #         output = self.model(x)
        #         pred = torch.argmax(output, dim=1).item()
        #         chunk_preds.append(pred)
        #
        # if chunk_preds:
        #     return Counter(chunk_preds).most_common(1)[0][0]
        # else:
        #     return None

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

            x = self._preprocess_chunk(chunk, sr=sr)

            with torch.no_grad():
                output = self.model(x)  # logits
                probs = torch.softmax(output, dim=1)  # convert to probabilities
                pred = torch.argmax(probs, dim=1).item()
                conf = probs[0, pred].item()
                chunk_preds.append((pred, conf))

        if not chunk_preds:
            return None, None

        # Group confidences by class
        class_confidences = defaultdict(list)
        for pred, conf in chunk_preds:
            class_confidences[pred].append(conf)

        # Majority vote (pick class with most votes)
        majority_class = max(class_confidences, key=lambda k: len(class_confidences[k]))
        confidence = sum(class_confidences[majority_class]) / len(class_confidences[majority_class])

        return majority_class, confidence