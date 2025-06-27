import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import csv
from collections import Counter
import os
import soundfile as sf
from pydub import AudioSegment

def get_audio_signal(file):
    y, sr = librosa.load(file)

    plt.figure(figsize=(10, 4))
    plt.plot(y)
    plt.title('Audio Signal')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()


def plot_spectrogram(file):
    y, sr = librosa.load(file)

    # Compute Short-Time Fourier Transform (STFT)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Plot spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram (STFT)')
    plt.tight_layout()
    plt.show()

def plot_mel_spectrogram(file_path):
    y, sr = librosa.load(file_path, duration=15)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000, cmap='magma')
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

def plot_spectral_centroid(file_path):
    y, sr = librosa.load(file_path, duration=15)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    frames = range(len(spectral_centroids))
    times = librosa.frames_to_time(frames, sr=sr)

    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, alpha=0.4)
    plt.plot(times, spectral_centroids, color='r', label='Spectral Centroid')
    plt.title('Spectral Centroid Over Time')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_mfcc(file_path):
    y, sr = librosa.load(file_path, duration=15)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', sr=sr, cmap='coolwarm')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()

def plot_audio_features(file_path):
    y, sr = librosa.load(file_path, duration=15)

    # Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    frames = range(len(centroid))
    centroid_times = librosa.frames_to_time(frames, sr=sr)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    # Plot
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))

    # Mel Spectrogram
    librosa.display.specshow(mel_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=axs[0], cmap='magma')
    axs[0].set_title('Mel Spectrogram')
    fig.colorbar(axs[0].images[0], ax=axs[0], format='%+2.0f dB')

    # Spectral Centroid
    librosa.display.waveshow(y, sr=sr, alpha=0.4, ax=axs[1])
    axs[1].plot(centroid_times, centroid, color='r', label='Spectral Centroid')
    axs[1].set_title('Spectral Centroid Over Time')
    axs[1].legend()

    # MFCC
    librosa.display.specshow(mfcc, x_axis='time', sr=sr, ax=axs[2], cmap='coolwarm')
    axs[2].set_title('MFCC')
    fig.colorbar(axs[2].images[0], ax=axs[2])

    plt.tight_layout()
    plt.show()

def write_csv(data):
    with open("output.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Key", "Values...", "Full Value List", "Counts", "Majority"])

        for key, values in data.items():
            vote = max(set(values), key=values.count)
            counts = dict(Counter(values))  # Use if you care about frequency
            writer.writerow([key] + values + [str(counts)] + [vote])


def write_csv_predictions(predictions, true_values, filename="preds.csv"):
    if len(predictions) != len(true_values):
        raise ValueError("Arrays must be of equal length")

    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["prediction", "true_value"])

        for pred, true in zip(predictions, true_values):
            writer.writerow([pred, true])


# file = "data/genres_original/blues/blues.00000.wav"
# # plot_mel_spectrogram(file)
# # plot_spectral_centroid(file)
# # plot_mfcc(file)
# plot_audio_features(file)